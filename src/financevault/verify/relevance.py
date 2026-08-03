"""s4 — retrieval relevance.

Did this step's evidence actually advance the question? A step can be perfectly well-formed
(`s1` = 1.0) and return real data, and still be a wasted turn — the right tool called with the
wrong tag, a search that returned adjacent boilerplate. This is the signal that separates
"did something" from "got somewhere", and it is the one that makes step-level filtering
different from outcome filtering.

Applies only to retrieval steps. Calculation and terminal steps score `n/a` rather than 0, so
a trajectory is not penalised for a check that never applied to it.
"""

from __future__ import annotations

import json

from financevault.runtime.budget import Ledger
from financevault.verify.judge import ask, truncate
from financevault.verify.signals import Signal, na

RETRIEVAL_TOOLS = {"retrieve_filings", "lookup_fact", "sql"}

SYSTEM = """You judge whether a retrieval step moved an agent closer to answering a question.

Score 1.0 when the result contains information the question needs.
Score around 0.5 when it is on-topic but not what the question asked for — the right company
but the wrong period, a related metric, boilerplate surrounding the real answer.
Score 0.0 when the result is irrelevant or empty.

Judge usefulness toward the question, not whether the tool call was well formed."""


def score(step: dict, question: str, *, ledger: Ledger) -> Signal:
    tool = step.get("tool")
    if tool not in RETRIEVAL_TOOLS:
        return na(f"{tool or 'no tool'} is not a retrieval step")

    obs = step.get("obs") or step.get("observation") or {}
    if not obs.get("ok"):
        # A failed retrieval retrieved nothing. Scored 0 rather than n/a: the step consumed
        # a turn and returned no evidence, which is exactly what this signal measures.
        return Signal(0.0, f"retrieval failed: {str(obs.get('error'))[:100]}")

    data = obs.get("data")
    rows = data if isinstance(data, list) else (data or {}).get("rows", data)
    if not rows:
        return Signal(0.0, "retrieval returned no rows")

    args = json.dumps(step.get("args") or {}, default=str)
    result = json.dumps(rows, default=str)

    return ask(
        SYSTEM,
        f"QUESTION: {question}\n\nTOOL: {tool}\nARGUMENTS: {truncate(args, 300)}\n\n"
        f"RESULT:\n{truncate(result)}",
        ledger=ledger,
        label="s4-relevance",
    )
