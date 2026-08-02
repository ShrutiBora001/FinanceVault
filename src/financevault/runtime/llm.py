"""The single path to a language model.

Nothing in this project calls the Anthropic SDK directly. Every request goes through `call`,
which consults the journal first, charges the ledger, and records the result. That is what
makes three separate guarantees hold at once:

- a replay costs $0.00 and can be *checked* to have cost $0.00,
- cost per answer is measured rather than estimated,
- and a run is reproducible from its journal without the network.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from financevault.config import settings
from financevault.runtime import journal
from financevault.runtime.budget import Ledger, price


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    tool_calls: list[dict]
    raw: dict
    tokens_in: int
    tokens_out: int
    cost_usd: float
    cached: bool
    latency_ms: int

    @property
    def stop_reason(self) -> str | None:
        return self.raw.get("stop_reason")


def _client() -> Any:
    from anthropic import Anthropic  # noqa: PLC0415 - deferred so replay needs no SDK config

    key = settings().anthropic_api_key
    if not key:
        raise RuntimeError(
            "FV_ANTHROPIC_API_KEY is unset; set it in .env, or run with FV_REPLAY=true to "
            "serve every call from the journal"
        )
    return Anthropic(api_key=key)


def _parse(response: dict) -> tuple[str, list[dict]]:
    """Split a response into its text and its tool-use blocks."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in response.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {"id": block.get("id"), "name": block.get("name"), "input": block.get("input", {})}
            )
    return "".join(text_parts), tool_calls


def call(
    model: str,
    messages: list[dict],
    *,
    ledger: Ledger,
    tools: list[dict] | None = None,
    system: str | None = None,
    thinking: bool = False,
    effort: str | None = None,
    max_tokens: int = 4096,
    label: str = "",
) -> Completion:
    """One model call, journalled and charged.

    **No sampling parameters.** Current models reject `temperature`, `top_p` and `top_k` with
    a 400; behaviour is steered by prompting and `effort` instead.

    **Thinking is opt-in here, and off by default.** Omitting the parameter would let the
    model think adaptively, and `max_tokens` caps thinking *plus* response text -- so a short
    call like the router's would spend its whole budget thinking and return nothing. Callers
    that want thinking ask for it and size `max_tokens` accordingly.
    """
    thinking_cfg = {"type": "adaptive"} if thinking else {"type": "disabled"}

    hash_ = journal.key(
        model,
        messages,
        tools=tools,
        system=system,
        thinking=thinking_cfg,
        effort=effort,
        max_tokens=max_tokens,
    )

    entry = journal.get(hash_)
    if entry is not None:
        text, tool_calls = _parse(entry.response)
        ledger.charge(
            model,
            entry.tokens_in,
            entry.tokens_out,
            latency_ms=0,
            cached=True,
            label=label,
        )
        return Completion(
            text=text,
            tool_calls=tool_calls,
            raw=entry.response,
            tokens_in=entry.tokens_in,
            tokens_out=entry.tokens_out,
            cost_usd=0.0,
            cached=True,
            latency_ms=0,
        )

    if settings().replay:
        raise journal.ReplayMiss(
            f"no journal entry for {hash_[:12]}… (model={model}, label={label!r}). "
            "Replay never falls back to a live call: that would make a 'free' replay cost "
            "money and report determinism on freshly generated output."
        )

    # Refuse to start a call the ledger cannot afford, rather than discovering it afterwards.
    ledger.check()

    started = time.monotonic()
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "thinking": thinking_cfg,
    }
    if system:
        request["system"] = system
    if tools:
        request["tools"] = tools
    if effort:
        # effort lives inside output_config, not at the top level.
        request["output_config"] = {"effort": effort}

    response = _client().messages.create(**request).model_dump(mode="json")
    latency_ms = int((time.monotonic() - started) * 1000)

    usage = response.get("usage", {}) or {}
    tokens_in = usage.get("input_tokens", 0) or 0
    tokens_out = usage.get("output_tokens", 0) or 0
    cost = price(model, tokens_in, tokens_out)

    journal.put(
        hash_,
        model,
        request,
        response,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
    )
    ledger.charge(model, tokens_in, tokens_out, latency_ms=latency_ms, cached=False, label=label)

    text, tool_calls = _parse(response)
    return Completion(
        text=text,
        tool_calls=tool_calls,
        raw=response,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
        cached=False,
        latency_ms=latency_ms,
    )
