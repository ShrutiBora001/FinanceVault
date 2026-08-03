"""Policies under evaluation.

A policy is anything that turns a question into a `Run`. Keeping them behind one signature
means the harness, the verifier and the metrics do not know which is which — a baseline
cannot accidentally be measured differently from the system it is a baseline for.

MVP1 ships two:

- **B1 — single-shot RAG.** Retrieve, then answer in one pass, no tools and no second look.
  This is the thing the project claims to beat, and it is deliberately a *fair* version of it:
  same corpus, same as-of horizon, same model. The only thing it lacks is the agent loop.
- **B2 — tools.** The full executor.

B0 (no tools) and B3 (frontier model with tools) arrive in MVP2.2, where the analyst model
becomes an experimental variable rather than a cost setting.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable

from financevault.config import settings
from financevault.env.asof import AsOf
from financevault.runtime import executor, llm
from financevault.runtime.budget import BudgetExceeded, Ledger
from financevault.runtime.executor import Run, Step
from financevault.tools import REGISTRY, ToolContext

Policy = Callable[..., Run]

B1_SYSTEM = """You answer questions about company financials from the excerpts provided.

Use only the excerpts. If they do not contain the answer, say so rather than guessing.

End your reply with a line in exactly this form, giving the figure in base units with no
commas, symbols or scale words:

VALUE: <number>"""

VALUE_RE = re.compile(r"VALUE:\s*(-?[\d.]+(?:[eE][-+]?\d+)?)")


def _parse_value(text: str) -> float | None:
    match = VALUE_RE.search(text or "")
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def b1_rag(
    question: str,
    *,
    as_of: AsOf,
    cik: str | None = None,
    ticker: str | None = None,
    k: int = 8,
    persist: bool = True,
) -> Run:
    """Single-shot RAG: one retrieval, one answer, no loop.

    Retrieval goes through the same as-of-filtered tool the agent uses, so B1 is not
    handicapped on data access. What it lacks is the ability to look again after seeing what
    came back — which is precisely the capability under test.
    """
    cfg = settings()
    ledger = Ledger(max_usd=cfg.max_usd, max_steps=2)
    ctx = ToolContext(as_of=as_of, ledger=ledger, cik=cik, ticker=ticker)

    started = time.monotonic()
    retrieved = REGISTRY.dispatch("retrieve_filings", {"query": question, "k": k}, ctx)
    steps = [
        Step(
            idx=0,
            thought="",
            tool="retrieve_filings",
            args={"query": question, "k": k},
            observation=retrieved.to_json(),
        )
    ]

    passages = ""
    if retrieved.ok:
        passages = "\n\n".join(
            f"[{row.get('form')} {row.get('period_end')}] {row.get('text', '')}"
            for row in (retrieved.data or [])
        )

    answer, value = None, None
    outcome = "ok"
    try:
        completion = llm.call(
            cfg.analyst_model,
            [
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\nAs of: {as_of}\n\nEXCERPTS:\n{passages[:12000]}"
                    ),
                }
            ],
            ledger=ledger,
            system=B1_SYSTEM,
            thinking=False,
            max_tokens=512,
            label="b1-answer",
        )
        answer = completion.text.strip()
        value = _parse_value(answer)
        steps.append(
            Step(
                idx=1,
                thought="",
                tool="finish",
                args={"answer": answer, "value": value, "citations": []},
                observation={"ok": True, "data": {"answer": answer, "value": value}},
                tokens_in=completion.tokens_in,
                tokens_out=completion.tokens_out,
                cost_usd=completion.cost_usd,
            )
        )
    except BudgetExceeded:
        outcome = "budget_exceeded"

    run = Run(
        id=str(uuid.uuid4()),
        question=question,
        as_of=as_of,
        policy="b1-rag",
        path="P1",
        outcome=outcome,
        answer=answer,
        citations=[],
        value=value,
        unit="USD",
        steps=steps,
        ledger=ledger,
    )
    run.ledger.started = started
    if persist:
        executor.save(run)
    return run


def b2_tools(
    question: str,
    *,
    as_of: AsOf,
    cik: str | None = None,
    ticker: str | None = None,
    persist: bool = True,
) -> Run:
    """The full agent loop."""
    return executor.execute(
        question,
        as_of=as_of,
        cik=cik,
        ticker=ticker,
        policy="b2-tools",
        persist=persist,
    )


POLICIES: dict[str, Policy] = {
    "b1-rag": b1_rag,
    "b2-tools": b2_tools,
}


def describe() -> str:
    return json.dumps({name: fn.__doc__.split("\n")[0] for name, fn in POLICIES.items()}, indent=2)
