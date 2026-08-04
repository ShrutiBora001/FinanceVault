"""Turn verified trajectories into the three training splits.

| split | what it contains | why it exists |
|---|---|---|
| `accepted` | every step verified | the positive signal |
| `repaired` | a bad step excised, the rest re-verified | recovers prefixes a pure filter discards |
| `hard_negative` | a near-miss: wrong unit, period or scale | teaches the boundary |

**Repair here is excision, not regeneration.** The plan calls for regenerating the suffix
after the failing step; that needs the executor to resume from a seeded message history,
which it cannot yet do. What this does instead is drop the failing step and keep the rest,
then re-verify — so a trajectory that reached a correct answer *despite* a wasted step yields
a clean shortened trajectory. It recovers the same data in the cases that matter most (a
detour that did not change the destination) and recovers nothing where the failing step
actually mattered. That is a real limitation, not a stylistic choice, and full regeneration is
MVP2.3 work.

Hard negatives are **constructed, not collected**. Organic near-misses are too rare at this
scale to build a split from, so they are derived from accepted trajectories by perturbing a
figure in a way the verifier is known to catch. That makes them synthetic and easy — a real
model's mistakes are subtler — which is worth remembering when reading any result that leans
on them.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from financevault.data.rollout import Trajectory
from financevault.verify import numeric


@dataclass(slots=True)
class Example:
    """One training example: a question, a trajectory, and how it was obtained."""

    question_id: str
    variant: str
    split: str
    question: str
    steps: list[dict]
    answer: str | None
    label: str  # for hard negatives, which defect was introduced
    source_run_id: str


def _step_records(trajectory: Trajectory) -> list[dict]:
    return [
        {
            "idx": s.idx,
            "thought": s.thought,
            "tool": s.tool,
            "args": s.args,
            "observation": s.observation,
        }
        for s in trajectory.run.steps
    ]


def accepted(trajectories: list[Trajectory]) -> list[Example]:
    return [
        Example(
            question_id=t.question_id,
            variant=t.variant,
            split="accepted",
            question=t.run.question,
            steps=_step_records(t),
            answer=t.run.answer,
            label="clean",
            source_run_id=t.run.id,
        )
        for t in trajectories
        if t.all_steps_pass and t.answer_correct
    ]


def repaired(trajectories: list[Trajectory]) -> list[Example]:
    """Excise failing steps from trajectories that still reached a correct answer.

    Only trajectories whose answer is correct are repairable: if the answer is wrong, the
    failing step is not a detour, it is the cause, and deleting it would fabricate a
    trajectory that never happened and did not work.
    """
    out: list[Example] = []
    for t in trajectories:
        if t.all_steps_pass or not t.answer_correct:
            continue

        failing = {v.idx for v in t.verdicts if not v.passed}
        kept = [s for s in _step_records(t) if s["idx"] not in failing]
        # A trajectory is only useful if something survives besides the final answer.
        if len(kept) < 2:
            continue

        out.append(
            Example(
                question_id=t.question_id,
                variant=t.variant,
                split="repaired",
                question=t.run.question,
                steps=[{**s, "idx": i} for i, s in enumerate(kept)],
                answer=t.run.answer,
                label=f"excised:{sorted(failing)}",
                source_run_id=t.run.id,
            )
        )
    return out


# Perturbations applied to a *correct* answer to manufacture a near miss. Each is a defect
# the verifier demonstrably catches, so the label is trustworthy by construction.
def _scale_error(answer: str, value: float | None) -> tuple[str, float | None]:
    return (f"The figure was ${(value or 0) / 1000:,.0f}.", (value / 1000) if value else None)


def _magnitude_error(answer: str, value: float | None) -> tuple[str, float | None]:
    return (f"The figure was ${(value or 0) * 1000:,.0f}.", (value * 1000) if value else None)


PERTURBATIONS = {
    "off_by_1000_low": _scale_error,
    "off_by_1000_high": _magnitude_error,
}


def hard_negatives(trajectories: list[Trajectory]) -> list[Example]:
    """Construct near-misses from accepted trajectories.

    The trajectory is left intact and only the final figure is corrupted, so the example is
    a plausible one: everything the agent did was reasonable, and the answer is still wrong.
    That is the boundary the split is meant to teach.
    """
    out: list[Example] = []
    for t in trajectories:
        if not (t.all_steps_pass and t.answer_correct):
            continue

        terminal = next((s for s in reversed(t.run.steps) if s.tool == "finish"), None)
        if terminal is None:
            continue
        value = (terminal.args or {}).get("value")
        if value is None:
            continue

        for label, perturb in PERTURBATIONS.items():
            bad_answer, bad_value = perturb(t.run.answer or "", float(value))
            steps = _step_records(t)
            steps[-1] = copy.deepcopy(steps[-1])
            steps[-1]["args"] = {**(steps[-1]["args"] or {}), "value": bad_value}
            steps[-1]["args"]["answer"] = bad_answer

            # Only keep it if the verifier actually rejects it. A "hard negative" the
            # verifier considers fine is mislabelled data, which is worse than none.
            evidence = numeric.collect_evidence(
                [{"tool": s["tool"], "obs": s["observation"]} for s in steps]
            )
            if numeric.score(bad_answer, evidence=evidence, stated_value=bad_value).passed:
                continue

            out.append(
                Example(
                    question_id=t.question_id,
                    variant=t.variant,
                    split="hard_negative",
                    question=t.run.question,
                    steps=steps,
                    answer=bad_answer,
                    label=label,
                    source_run_id=t.run.id,
                )
            )
    return out


def build(trajectories: list[Trajectory]) -> dict[str, list[Example]]:
    return {
        "accepted": accepted(trajectories),
        "repaired": repaired(trajectories),
        "hard_negative": hard_negatives(trajectories),
    }
