"""Write splits to JSONL in the chat format a fine-tuning run expects.

A trajectory becomes a conversation: the question as the user turn, each step as an assistant
turn carrying a tool call, each observation as the tool result. That is the same shape the
executor sends at inference, so what the model is trained on and what it sees in production
are the same thing.

Hard negatives export with `"label": "reject"` rather than being silently mixed into the
positives. How a trainer uses them — preference pairs, a contrastive term, or simply excluded
from the SFT target — is a training decision, and burying it in the export format would make
that decision by accident.
"""

from __future__ import annotations

import json
from pathlib import Path

from financevault.data.splits import Example

OUT_DIR = Path("data/sft")


def to_conversation(example: Example) -> dict:
    messages: list[dict] = [{"role": "user", "content": example.question}]

    for step in example.steps:
        if step.get("tool") is None:
            continue
        thought = (step.get("thought") or "").strip()
        messages.append(
            {
                "role": "assistant",
                "content": thought,
                "tool_calls": [{"name": step["tool"], "arguments": step.get("args") or {}}],
            }
        )
        observation = step.get("observation") or {}
        messages.append(
            {
                "role": "tool",
                "name": step["tool"],
                "content": json.dumps(
                    observation.get("data") if observation.get("ok") else observation.get("error"),
                    default=str,
                )[:4000],
            }
        )

    return {
        "question_id": example.question_id,
        "variant": example.variant,
        "split": example.split,
        "label": "reject" if example.split == "hard_negative" else "accept",
        "defect": example.label,
        "source_run_id": example.source_run_id,
        "messages": messages,
        "answer": example.answer,
    }


def write(splits: dict[str, list[Example]], out_dir: Path = OUT_DIR) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, examples in splits.items():
        path = out_dir / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(to_conversation(e), sort_keys=True) + "\n" for e in examples)
        )
        counts[name] = len(examples)

    # The two filters H1 compares, materialised as files a trainer can point at directly.
    # `outcome_filtered` is the baseline condition: everything that reached a right answer,
    # bad steps included. Building both here means the comparison cannot drift later.
    step_filtered = splits["accepted"] + splits["repaired"]
    (out_dir / "step_filtered.jsonl").write_text(
        "".join(json.dumps(to_conversation(e), sort_keys=True) + "\n" for e in step_filtered)
    )
    counts["step_filtered"] = len(step_filtered)
    return counts
