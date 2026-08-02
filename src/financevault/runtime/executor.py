"""The step executor: plan, act, observe, bounded.

This is what produces trajectories. Every step is persisted to `steps` with its tool call,
arguments and observation, because those rows are simultaneously three things: an audit
trail, the input to the verifier, and — after filtering — training data.

Two decisions worth naming:

**A run ends for a recorded reason.** `ok`, `max_steps`, `budget_exceeded`, `no_tool_call`.
The reason is stored, so a policy that habitually runs out of budget is visible in the
metrics rather than showing up as a vague accuracy drop.

**Tool failures are observations, not exceptions.** A malformed query or a missing tag is
returned to the model as an observation so it can recover on the next step. That is normal
agent behaviour and, importantly, it produces exactly the near-miss steps that hard-negative
mining needs later.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from financevault.config import settings
from financevault.env.asof import AsOf
from financevault.runtime import llm, router
from financevault.runtime.budget import BudgetExceeded, Ledger
from financevault.store import pg
from financevault.tools import REGISTRY, ToolContext

SYSTEM = """You answer questions about company financials using the tools provided.

Rules:
- Every figure in your answer must come from a tool result, never from memory. You are
  working with data as of a specific date and your own recollection may be stale or wrong.
- Prefer lookup_fact for reported figures. Use retrieve_filings for narrative or commentary.
- Use python for arithmetic. Do not compute in your head.
- Call finish when you have the answer, and cite every figure you state.
- If a tool returns an error, read it and try a different approach. Errors often name the
  correct tag or table.

You have a limited step and cost budget. Do not explore; go to the answer."""


@dataclass(slots=True)
class Step:
    idx: int
    thought: str
    tool: str | None
    args: dict
    observation: dict
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0


@dataclass(slots=True)
class Run:
    id: str
    question: str
    as_of: AsOf
    policy: str
    path: str
    outcome: str
    answer: str | None
    citations: list[dict]
    value: float | None
    unit: str | None
    steps: list[Step] = field(default_factory=list)
    ledger: Ledger | None = None

    def summary(self) -> dict:
        return {
            "run_id": self.id,
            "path": self.path,
            "outcome": self.outcome,
            "answer": self.answer,
            "value": self.value,
            "unit": self.unit,
            "n_steps": len(self.steps),
            **(self.ledger.summary() if self.ledger else {}),
        }


def _observation_text(result_json: dict) -> str:
    """Tool output as the model sees it. Errors are phrased to invite a retry."""
    if result_json["ok"]:
        return json.dumps(result_json["data"], default=str)[:6000]
    return f"ERROR: {result_json['error']}"


def execute(
    question: str,
    *,
    as_of: AsOf,
    cik: str | None = None,
    ticker: str | None = None,
    policy: str = "b2-tools",
    model: str | None = None,
    max_steps: int | None = None,
    max_usd: float | None = None,
    persist: bool = True,
) -> Run:
    """Run one episode and return its trajectory."""
    cfg = settings()
    model = model or cfg.analyst_model
    ledger = Ledger(
        max_usd=max_usd if max_usd is not None else cfg.max_usd,
        max_steps=max_steps if max_steps is not None else cfg.max_steps,
    )

    route = router.route(question, ledger=ledger, model=cfg.judge_model)
    # The router's per-path ceiling narrows the budget but never widens it past the caller's.
    ledger.max_steps = min(ledger.max_steps, route.max_steps)

    ctx = ToolContext(as_of=as_of, ledger=ledger, cik=cik, ticker=ticker)
    specs = REGISTRY.specs(route.tools)

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"As of: {as_of} — information published after this instant does not exist.\n"
                f"Company: {ticker or cik or 'unspecified'}"
            ),
        }
    ]

    run = Run(
        id=str(uuid.uuid4()),
        question=question,
        as_of=as_of,
        policy=policy,
        path=route.path,
        outcome="running",
        answer=None,
        citations=[],
        value=None,
        unit=None,
        ledger=ledger,
    )

    while True:
        try:
            ledger.step()
        except BudgetExceeded as exc:
            run.outcome = "max_steps" if exc.resource == "steps" else "budget_exceeded"
            break

        started = time.monotonic()
        try:
            # Thinking on, with headroom. max_tokens bounds thinking and response text
            # together, so a tight budget here would truncate mid-tool-call.
            completion = llm.call(
                model,
                messages,
                ledger=ledger,
                tools=specs,
                system=SYSTEM,
                thinking=True,
                effort=cfg.analyst_effort,
                max_tokens=4096,
                label=f"step-{len(run.steps)}",
            )
        except BudgetExceeded:
            run.outcome = "budget_exceeded"
            break

        latency_ms = int((time.monotonic() - started) * 1000)

        if not completion.tool_calls:
            # No tool call and no finish: the model answered in prose. Recorded as its own
            # outcome rather than salvaged, so the rate is visible in the metrics.
            run.steps.append(
                Step(
                    idx=len(run.steps),
                    thought=completion.text,
                    tool=None,
                    args={},
                    observation={"ok": False, "error": "no tool call"},
                    tokens_in=completion.tokens_in,
                    tokens_out=completion.tokens_out,
                    cost_usd=completion.cost_usd,
                    latency_ms=latency_ms,
                )
            )
            run.outcome = "no_tool_call"
            run.answer = completion.text or None
            break

        messages.append({"role": "assistant", "content": completion.raw["content"]})
        tool_results = []
        terminated = False

        for call in completion.tool_calls:
            result = REGISTRY.dispatch(call["name"], call["input"], ctx)
            result_json = result.to_json()

            run.steps.append(
                Step(
                    idx=len(run.steps),
                    thought=completion.text,
                    tool=call["name"],
                    args=call["input"],
                    observation=result_json,
                    tokens_in=completion.tokens_in,
                    tokens_out=completion.tokens_out,
                    cost_usd=completion.cost_usd,
                    latency_ms=latency_ms,
                )
            )

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": _observation_text(result_json),
                    "is_error": not result.ok,
                }
            )

            tool = REGISTRY.get(call["name"])
            if tool and tool.terminal and result.ok:
                run.answer = result.data["answer"]
                run.citations = result.data["citations"]
                run.value = result.data["value"]
                run.unit = result.data["unit"]
                run.outcome = "ok"
                terminated = True

        if terminated:
            break
        messages.append({"role": "user", "content": tool_results})

    if persist:
        save(run)
    return run


INSERT_RUN = """
INSERT INTO runs (id, question, as_of, policy, path, budget_usd, budget_steps, outcome,
                  answer, citations, n_steps, cost_usd, latency_ms)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

INSERT_STEP = """
INSERT INTO steps (run_id, idx, thought, tool, args, obs, tokens_in, tokens_out,
                   cost_usd, latency_ms)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def save(run: Run) -> None:
    ledger = run.ledger
    pg.execute(
        INSERT_RUN,
        (
            run.id,
            run.question,
            run.as_of.instant,
            run.policy,
            run.path,
            ledger.max_usd if ledger else None,
            ledger.max_steps if ledger else None,
            run.outcome,
            run.answer,
            json.dumps(run.citations),
            len(run.steps),
            ledger.usd if ledger else 0,
            int(ledger.elapsed * 1000) if ledger else None,
        ),
    )
    pg.execute_many(
        INSERT_STEP,
        [
            (
                run.id,
                s.idx,
                s.thought,
                s.tool,
                json.dumps(s.args, default=str),
                json.dumps(s.observation, default=str),
                s.tokens_in,
                s.tokens_out,
                s.cost_usd,
                s.latency_ms,
            )
            for s in run.steps
        ],
    )
