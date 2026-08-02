"""Task and complexity routing.

The router picks one of five execution paths and, with it, the toolset the agent may use.
Restricting tools per path is the point: it keeps cheap questions cheap, and it makes the
routing decision itself something the verifier can score, since choosing P0 for a question
that needed a filing lookup is a specific, identifiable failure.

The classifier is a single cheap call with a fixed rubric. It is deliberately not learned —
MVP1 measures how good the fixed rubric is before anyone tries to improve on it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from financevault.config import settings
from financevault.runtime import llm
from financevault.runtime.budget import Ledger

# Path -> tools available on it. `finish` is on every path; a run must always be able to stop.
PATHS: dict[str, list[str]] = {
    "P0": ["finish"],
    "P1": ["retrieve_filings", "finish"],
    "P2": ["lookup_fact", "finish"],
    "P3": ["sql", "python", "finish"],
    "P4": ["retrieve_filings", "lookup_fact", "sql", "python", "finish"],
}

RUBRIC = """You route financial questions to an execution path. Reply with the path id only.

P0 - answerable from general knowledge, no company data needed
     ("what does EBITDA stand for")
P1 - needs narrative text from a filing: risks, strategy, commentary, MD&A
     ("what risks did Apple flag around supply chain")
P2 - needs one reported figure from the financial statements
     ("what was Apple's net income in FY2025")
P3 - needs aggregation, comparison across periods, or a derived calculation
     ("how did Apple's gross margin change over the last three years")
P4 - needs several of the above combined, or the path is unclear
     ("did the risks Apple flagged show up in its margins")

When torn between two paths, choose the more capable one. A path that is too narrow
strands the question; a path that is too wide only costs tokens."""

VALID = re.compile(r"\bP[0-4]\b")


@dataclass(frozen=True, slots=True)
class Route:
    path: str
    tools: list[str]
    reason: str
    cached: bool

    @property
    def max_steps(self) -> int:
        """A step ceiling proportional to the path. P0 needs one call; P4 may need several."""
        return {"P0": 2, "P1": 4, "P2": 4, "P3": 6, "P4": 8}[self.path]


def route(question: str, *, ledger: Ledger, model: str | None = None) -> Route:
    """Classify a question. Falls back to P4 rather than failing the run.

    An unparseable classification is a router bug, not a reason to refuse the question, and
    P4 is the path that can answer anything. The fallback is recorded in `reason` so it shows
    up in the metrics rather than hiding as a normal P4.
    """
    completion = llm.call(
        model or settings().judge_model,
        [{"role": "user", "content": f"Question: {question}\n\nPath:"}],
        ledger=ledger,
        system=RUBRIC,
        max_tokens=8,
        label="router",
    )

    match = VALID.search(completion.text.upper())
    if match is None:
        return Route(
            path="P4",
            tools=PATHS["P4"],
            reason=f"unparseable classification {completion.text!r}; fell back to P4",
            cached=completion.cached,
        )

    path = match.group(0)
    return Route(
        path=path, tools=PATHS[path], reason=completion.text.strip(), cached=completion.cached
    )
