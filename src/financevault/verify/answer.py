"""s5 — final-answer correctness.

Terminal step only, and programmatic wherever ground truth exists: a numeric answer is
compared against the expected value within tolerance, which no model needs to adjudicate.

**Deliberately kept out of any training reward.** `s5` is the outcome, and the hypothesis
under test is whether *step-level* signals beat outcome filtering. Letting `s5` into the
filtering reward would collapse the two conditions being compared and make the experiment
answer its own question. It is computed for evaluation and for building the outcome-filtered
baseline (B4), and nowhere else.
"""

from __future__ import annotations

from financevault.verify.signals import Signal, na

# Answers restate rounded figures, so exact equality would fail correct answers.
REL_TOL = 0.005


def score(
    value: float | None,
    expected: float | None,
    *,
    answer: str = "",
    expected_text: str | None = None,
) -> Signal:
    """Compare a terminal answer against ground truth. `n/a` when there is none to compare to."""
    if expected is None and expected_text is None:
        return na("no ground truth for this question")

    if expected is not None:
        if value is None:
            return Signal(
                0.0,
                "expected a numeric answer but the trajectory reported none; "
                "`finish` should set `value` for single-figure questions",
            )
        if expected == 0:
            ok = abs(value) < 1e-9
        else:
            ok = abs(value - expected) / abs(expected) <= REL_TOL
        return Signal(
            1.0 if ok else 0.0,
            f"{value:,.6g} vs expected {expected:,.6g}"
            + ("" if ok else f" (off by {abs(value - expected) / abs(expected or 1):.2%})"),
        )

    # Text ground truth: substring containment, case-insensitive. Deliberately crude -- string
    # comparison of prose is not meaningful, so MVP1's eval split is numeric-only and this
    # path exists for the handful of narrative questions added later.
    assert expected_text is not None
    hit = expected_text.strip().lower() in (answer or "").lower()
    return Signal(1.0 if hit else 0.0, f"expected text {'found' if hit else 'not found'}")
