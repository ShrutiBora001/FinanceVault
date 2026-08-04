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
    # The score at or above which the step passes. Most signals use 0.5, where partial credit
    # is meaningful — a well-formed call that errored is a reasonable search move.
    #
    # `s3` sets it to 1.0, because "does every figure trace to evidence" is a conjunction, not
    # an average. Seeded-error calibration exposed why: an answer restating a figure both
    # correctly and with a 1000x error scored 0.6, so the corruption was *detected* and the
    # step still passed. Averaging a detected fabrication against correct claims lets
    # corrupted answers into training data.
    threshold: float = 0.5
    # Whether this signal applied to the step at all. An inapplicable signal scores 1.0 so a
    # step is not penalised for a check that never concerned it — but 1.0 then means two
    # different things, and conflating them is how a run that never produced an answer got
    # counted as answering correctly. Anything asking "was this *right*" must check
    # `applicable`, not just the score.
    applicable: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"signal score must be in [0, 1], got {self.score}")

    @property
    def passed(self) -> bool:
        return self.score >= self.threshold

    @property
    def verified(self) -> bool:
        """Passed *and* actually applied. Use this for correctness, never `passed`."""
        return self.applicable and self.passed


def na(reason: str) -> Signal:
    """A signal that does not apply to this step.

    Scored 1.0 rather than 0.0 on purpose: a step is not penalised for a check that was never
    relevant to it. `retrieve_filings` should not lose points on numeric correctness. The
    `applicable` flag is what keeps that from being read as evidence of correctness.
    """
    return Signal(score=1.0, reason=f"n/a: {reason}", applicable=False)
