"""The five-signal step verifier.

Verification is a separate pass from generation, by design: either can be re-run without
redoing the other. Re-verifying a trajectory after changing a rubric costs nothing beyond the
judged signals, and those are journalled too, so an unchanged rubric re-runs for free.

| Signal | What it asks | How |
|---|---|---|
| `s1` | Was the call well formed and available? | programmatic |
| `s2` | Do the citations resolve, and do they support the answer? | structural + judged |
| `s3` | Does every figure trace to retrieved evidence? | programmatic |
| `s4` | Did this retrieval advance the question? | judged |
| `s5` | Is the final answer right? | programmatic, evaluation only |

The two signals that decide whether a trajectory is *correct* are the programmatic ones. A
judge asked whether a plausible number is right says yes, which is the exact failure mode this
project exists to catch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from financevault.env.asof import AsOf
from financevault.runtime.budget import Ledger
from financevault.store import pg
from financevault.verify import answer as answer_sig
from financevault.verify import citation, numeric, relevance, tool_validity
from financevault.verify.signals import NAMES, SIGNALS, Signal, na

__all__ = [
    "NAMES",
    "SIGNALS",
    "Signal",
    "StepVerdict",
    "answered_correctly",
    "na",
    "verify_run",
]


@dataclass(slots=True)
class StepVerdict:
    idx: int
    tool: str | None
    signals: dict[str, Signal] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """A step passes only if every applicable signal does. `n/a` signals score 1.0."""
        return all(s.passed for s in self.signals.values())

    def scores(self) -> dict[str, float]:
        return {k: v.score for k, v in self.signals.items()}

    def reasons(self) -> dict[str, str]:
        return {k: v.reason for k, v in self.signals.items()}


def answered_correctly(verdicts: list[StepVerdict]) -> bool:
    """Did this trajectory produce an answer that was checked and found right?

    The single place that question is answered, because getting it wrong is cheap and
    invisible. A run that never calls `finish` has no terminal step, so `s5` is `n/a` and
    scores 1.0 — reading the last verdict's score directly counted such runs as correct and
    inflated a reported accuracy from 85% to 100%.

    Three conditions, all required: a `finish` step exists, `s5` actually applied (there was
    ground truth to compare against), and it passed.
    """
    finish = next((v for v in reversed(verdicts) if v.tool == "finish"), None)
    return bool(finish and finish.signals["s5"].verified)


UPDATE_STEP = """
UPDATE steps SET s1=%s, s2=%s, s3=%s, s4=%s, s5=%s, s_reasons=%s
WHERE run_id=%s AND idx=%s
"""


def _evidence_text(steps: list[dict]) -> str:
    """Retrieved observations, concatenated, for `s2` to check the answer against."""
    parts: list[str] = []
    for step in steps:
        obs = step.get("obs") or step.get("observation") or {}
        if step.get("tool") in (None, "finish") or not obs.get("ok"):
            continue
        parts.append(json.dumps(obs.get("data"), default=str))
    return "\n".join(parts)


def verify_run(
    run_id: str,
    *,
    cik: str | None = None,
    expected_value: float | None = None,
    expected_text: str | None = None,
    ledger: Ledger | None = None,
    persist: bool = True,
) -> list[StepVerdict]:
    """Score every step of a stored run, and write the signals back to `steps`."""
    run = pg.fetch_one(
        "SELECT id, question, as_of, answer, path FROM runs WHERE id = %s", (run_id,)
    )
    if run is None:
        raise KeyError(f"no run {run_id}")

    steps = pg.fetch_all(
        "SELECT idx, tool, args, obs FROM steps WHERE run_id = %s ORDER BY idx", (run_id,)
    )
    if not steps:
        return []

    # Judging is bounded separately from the run that produced the trajectory: verification
    # cost must not be attributable to, or limited by, the policy's own budget.
    ledger = ledger or Ledger(max_usd=0.05, max_steps=len(steps) * 4)

    horizon = AsOf(run["as_of"])
    evidence = numeric.collect_evidence(steps)
    evidence_text = _evidence_text(steps)
    last_idx = max(s["idx"] for s in steps)

    from financevault.runtime.router import PATHS  # noqa: PLC0415 - avoids an import cycle

    allowed = PATHS.get(run["path"] or "P4")

    verdicts: list[StepVerdict] = []
    for step in steps:
        is_terminal = step["idx"] == last_idx and step["tool"] == "finish"
        signals: dict[str, Signal] = {
            "s1": tool_validity.score_step(step, allowed),
            "s4": relevance.score(step, run["question"], ledger=ledger),
        }

        if is_terminal:
            args = step.get("args") or {}
            signals["s2"] = citation.score(
                run["answer"] or "",
                args.get("citations") or [],
                steps,
                ledger=ledger,
                evidence_text=evidence_text,
            )
            signals["s3"] = numeric.score(
                run["answer"] or "",
                evidence=evidence,
                cik=cik,
                as_of=horizon,
                stated_value=args.get("value"),
            )
            signals["s5"] = answer_sig.score(
                args.get("value"),
                expected_value,
                answer=run["answer"] or "",
                expected_text=expected_text,
            )
        else:
            signals["s2"] = na("not the terminal step")
            signals["s3"] = na("not the terminal step")
            signals["s5"] = na("not the terminal step")

        verdicts.append(StepVerdict(idx=step["idx"], tool=step["tool"], signals=signals))

    if persist:
        pg.execute_many(
            UPDATE_STEP,
            [
                (
                    v.signals["s1"].score,
                    v.signals["s2"].score,
                    v.signals["s3"].score,
                    v.signals["s4"].score,
                    v.signals["s5"].score,
                    json.dumps(v.reasons()),
                    run_id,
                    v.idx,
                )
                for v in verdicts
            ],
        )

    return verdicts
