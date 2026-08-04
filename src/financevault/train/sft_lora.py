"""LoRA fine-tuning on an exported split.

**This is a pipeline test, not an experiment.** MVP1's question is whether the last seam
connects: trajectory store → exporter → tokenizer → training loop → saved adapter → loaded
back. It runs a 0.6B model for a handful of steps on a few dozen examples, and the resulting
adapter is worthless as a model. Any loss number it prints says nothing about H1.

The real run is MVP2.3: thousands of trajectories, an 8B base, and B4 against B5 at equal
trajectory budget. What matters here is that the plumbing works before that costs money.

Conversations are rendered through the tokenizer's own chat template so the training text
matches what the model expects, rather than a format invented here that happens to look
plausible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DATA_DIR = Path("data/sft")
ADAPTER_DIR = Path("adapters")


@dataclass(slots=True)
class TrainConfig:
    model: str = DEFAULT_MODEL
    split: str = "step_filtered"
    epochs: int = 1
    batch_size: int = 1
    grad_accum: int = 4
    lr: float = 2e-4
    max_len: int = 1024
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # Attention projections only. Enough to prove the path; MVP2.3 tunes the target set.
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    max_examples: int = 0  # 0 = all


def load_examples(split: str, data_dir: Path = DATA_DIR, limit: int = 0) -> list[dict]:
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `make rollout` first")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # Hard negatives are labelled `reject`; training on them as positives would teach the
    # model to produce exactly the errors the verifier exists to catch.
    rows = [r for r in rows if r.get("label") != "reject"]
    return rows[:limit] if limit else rows


def render(example: dict, tokenizer: Any) -> str:
    """A trajectory as training text, via the tokenizer's own chat template.

    Tool calls are folded into the assistant turn as text rather than passed as structured
    `tool_calls`: chat templates disagree about tool-call formatting across model families,
    and a mismatch would train the model on a format it never emits at inference. A single
    explicit rendering is at least consistent between training and serving.
    """
    messages: list[dict[str, str]] = []
    for message in example["messages"]:
        role = message["role"]
        content = message.get("content") or ""
        if role == "assistant" and message.get("tool_calls"):
            call = message["tool_calls"][0]
            content = (
                f"{content}\n<tool_call>"
                f"{json.dumps({'name': call['name'], 'arguments': call['arguments']})}"
                f"</tool_call>"
            ).strip()
        elif role == "tool":
            role, content = "user", f"<tool_result>{content}</tool_result>"
        messages.append({"role": role, "content": content})

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def train(config: TrainConfig) -> dict:
    """Run the smoke fine-tune and return what happened."""
    import torch  # noqa: PLC0415 - heavy imports stay out of the module path
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    examples = load_examples(config.split, limit=config.max_examples)
    if not examples:
        raise RuntimeError(f"no usable examples in {config.split}")

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = [render(e, tokenizer) for e in examples]
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_len,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(config.model, dtype=torch.float32).to(device)
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            task_type="CAUSAL_LM",
        ),
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.lr)
    model.train()

    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    losses: list[float] = []

    for _ in range(config.epochs):
        for start in range(0, len(ids), config.batch_size):
            chunk_ids = ids[start : start + config.batch_size]
            chunk_mask = mask[start : start + config.batch_size]
            # Labels mirror inputs; padding is masked out so it contributes no gradient.
            labels = chunk_ids.clone()
            labels[chunk_mask == 0] = -100

            loss = model(input_ids=chunk_ids, attention_mask=chunk_mask, labels=labels).loss
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            losses.append(float(loss.detach()))

    out_dir = ADAPTER_DIR / f"{config.split}-smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    return {
        "model": config.model,
        "split": config.split,
        "n_examples": len(examples),
        "device": device,
        "trainable_params": trainable,
        "total_params": total,
        # E3: the LoRA ratio. A real number even from a smoke run.
        "trainable_fraction": round(trainable / total, 6),
        "steps": len(losses),
        "first_loss": round(losses[0], 4) if losses else None,
        "last_loss": round(losses[-1], 4) if losses else None,
        "adapter_dir": str(out_dir),
    }


def verify_adapter(adapter_dir: Path, base_model: str) -> dict:
    """Load the saved adapter back onto the base model.

    The point of the whole exercise: an adapter that trains but cannot be loaded has not
    proven the path connects.
    """
    import torch  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, str(adapter_dir))
    adapter_params = sum(p.numel() for n, p in merged.named_parameters() if "lora" in n.lower())
    return {"loaded": True, "adapter_params": adapter_params}
