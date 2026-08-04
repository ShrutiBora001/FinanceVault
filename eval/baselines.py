"""Policies under evaluation.

A policy is anything that turns a question into a `Run`. Keeping them behind one signature
means the harness, the verifier and the metrics do not know which is which — a baseline
cannot accidentally be measured differently from the system it is a baseline for.

Four policies:

- **B0 — no tools.** The model answers from parametric memory alone. This is the floor, and it
  is not a joke baseline: a model that has read a lot of filings can produce a right answer
  with no retrieval at all, and the gap between B0 and the rest is how much of the system's
  accuracy is actually *the system*. It also has no as-of horizon it can respect, so every
  answer it gets right is one it could only have got from training data.
- **B1 — single-shot RAG.** Retrieve, then answer in one pass. Deliberately a *fair* version
  of the thing the project claims to beat: same corpus, same horizon, same model. What it
  lacks is the ability to look again after seeing what came back.
- **B2 — tools.** The full executor on the cheap model.
- **B3 — frontier model with tools.** The same executor on a stronger model. The ceiling, and
  the only run that gives the efficiency frontier something to plot against.
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

B0_SYSTEM = """You answer questions about company financials from your own knowledge.

You have no tools and no documents. Answer as accurately as you can, and say so plainly if you
do not know rather than guessing at a figure.

End your reply with a line in exactly this form, giving the figure in base units with no
commas, symbols or scale words:

VALUE: <number>"""

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


def b0_no_tools(
    question: str,
    *,
    as_of: AsOf,
    cik: str | None = None,
    ticker: str | None = None,
    persist: bool = True,
) -> Run:
    """No tools, no retrieval: whatever the model already knows.

    Note what a correct answer here means. There is no corpus and no horizon, so B0 cannot
    respect an as-of date even in principle — anything it gets right came from training data,
    which for recent filings is both unreliable and unauditable. B0 scoring well on a question
    is a reason to distrust that question, not to trust B0.
    """
    cfg = settings()
    ledger = Ledger(max_usd=cfg.max_usd, max_steps=2)
    started = time.monotonic()

    answer, value, outcome = None, None, "ok"
    steps: list[Step] = []
    try:
        completion = llm.call(
            cfg.analyst_model,
            [{"role": "user", "content": f"Question: {question}\nCompany: {ticker or cik}"}],
            ledger=ledger,
            system=B0_SYSTEM,
            thinking=False,
            max_tokens=512,
            label="b0-answer",
        )
        answer = completion.text.strip()
        value = _parse_value(answer)
        steps.append(
            Step(
                idx=0,
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
        policy="b0-no-tools",
        path="P0",
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


def b3_frontier(
    question: str,
    *,
    as_of: AsOf,
    cik: str | None = None,
    ticker: str | None = None,
    persist: bool = True,
) -> Run:
    """The full agent loop on a stronger model.

    Identical to B2 in every respect but the model, so the difference between them is the
    model and nothing else. That is the only way the efficiency frontier means anything: if
    B3 also changed the prompt or the toolset, its position on the plot would be
    uninterpretable.
    """
    return executor.execute(
        question,
        as_of=as_of,
        cik=cik,
        ticker=ticker,
        policy="b3-frontier",
        model=settings().frontier_model,
        persist=persist,
    )


POLICIES: dict[str, Policy] = {
    "b0-no-tools": b0_no_tools,
    "b1-rag": b1_rag,
    "b2-tools": b2_tools,
    "b3-frontier": b3_frontier,
}


def describe() -> str:
    return json.dumps({name: fn.__doc__.split("\n")[0] for name, fn in POLICIES.items()}, indent=2)
