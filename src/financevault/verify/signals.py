"""Verifier signal types.

Each signal returns a score in [0, 1] and a reason. The reason is not decoration: when the
verifier is calibrated against human labels (B1) and seeded errors (B2), the reason is what
makes a disagreement diagnosable rather than just countable.
"""

from __future__ import annotations

from dataclasses import dataclass

SIGNALS = ("s1", "s2", "s3", "s4", "s5")

NAMES = {
    "s1": "tool_validity",
    "s2": "citation_support",
    "s3": "numeric_correctness",
    "s4": "retrieval_relevance",
    "s5": "answer_correctness",
}


@dataclass(frozen=True, slots=True)
class Signal:
    score: float
    reason: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"signal score must be in [0, 1], got {self.score}")

    @property
    def passed(self) -> bool:
        return self.score >= 0.5


def na(reason: str) -> Signal:
    """A signal that does not apply to this step.

    Scored 1.0 rather than 0.0 on purpose: a step is not penalised for a check that was never
    relevant to it. `retrieve_filings` should not lose points on numeric correctness.
    """
    return Signal(score=1.0, reason=f"n/a: {reason}")
