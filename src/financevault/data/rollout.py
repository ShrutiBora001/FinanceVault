"""Generate trajectories, verify them, and sort them into training splits.

**Diversity comes from the input, not from sampling.** Current models reject `temperature`,
and every call is journalled, so asking the same question twice returns the identical cached
trajectory. Rollout diversity therefore has to be engineered: each question is run under
several *variants* that change what the agent can see or do.

| variant | what changes | why it yields different trajectories |
|---|---|---|
| `full` | all tools (P4) | the agent chooses its own route |
| `facts_only` | XBRL lookup only (P2) | forces structured retrieval, no prose |
| `text_only` | passage retrieval only (P1) | forces prose grounding, no structured facts |
| `compute` | SQL and Python (P3) | forces derivation rather than lookup |
| `tight` | all tools, half the step budget | forces the agent to commit early |

This is a weaker diversity source than temperature sampling and the difference matters for
what the data can support: these trajectories vary in *approach*, not in the model's
moment-to-moment choices, so they under-represent the near-misses a stochastic policy would
produce. MVP2.3 generates from a local model through vLLM, where sampling is available again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from financevault import verify
from financevault.env.asof import AsOf
from financevault.runtime import executor
from financevault.runtime.budget import Ledger
from financevault.runtime.executor import Run

# Each variant pins the router's choice, so the toolset is the independent variable rather
# than something the router may or may not decide to vary.
VARIANTS: dict[str, dict] = {
    "full": {"path": "P4"},
    "facts_only": {"path": "P2"},
    "text_only": {"path": "P1"},
    "compute": {"path": "P3"},
    "tight": {"path": "P4", "max_steps": 3},
}


@dataclass(slots=True)
class Trajectory:
    run: Run
    question_id: str
    variant: str
    expected_value: float | None
    verdicts: list = field(default_factory=list)

    @property
    def step_scores(self) -> list[dict[str, float]]:
        return [v.scores() for v in self.verdicts]

    @property
    def all_steps_pass(self) -> bool:
        """Every step verifies. This is the accepted split's criterion."""
        return bool(self.verdicts) and all(v.passed for v in self.verdicts)

    @property
    def answer_correct(self) -> bool:
        """The outcome only. This is the *other* filter — the one H1 compares against.

        Delegates to `verify.answered_correctly` rather than reading s5 off the last verdict:
        a run that never called `finish` has no terminal step, so s5 is n/a and scores 1.0,
        and reading it directly counts a run that produced no answer as having answered.
        """
        from financevault import verify  # noqa: PLC0415 - avoids an import cycle

        return verify.answered_correctly(self.verdicts)

    @property
    def first_failing_step(self) -> int | None:
        for verdict in self.verdicts:
            if not verdict.passed:
                return verdict.idx
        return None

    def summary(self) -> dict:
        return {
            "run_id": self.run.id,
            "question_id": self.question_id,
            "variant": self.variant,
            "outcome": self.run.outcome,
            "n_steps": len(self.run.steps),
            "all_steps_pass": self.all_steps_pass,
            "answer_correct": self.answer_correct,
            "first_failing_step": self.first_failing_step,
            "list_usd": round(self.run.ledger.list_usd, 6) if self.run.ledger else 0.0,
        }


def generate(
    questions: list[dict],
    *,
    variants: list[str] | None = None,
    verify_ledger_usd: float = 0.10,
) -> list[Trajectory]:
    """Roll out every question under every variant, verifying each trajectory."""
    chosen = variants or list(VARIANTS)
    out: list[Trajectory] = []

    for question in questions:
        for name in chosen:
            spec = VARIANTS[name]
            run = executor.execute(
                question["question"],
                as_of=AsOf.parse(question["as_of"]),
                cik=question.get("cik"),
                ticker=question.get("ticker"),
                policy=f"rollout:{name}",
                max_steps=spec.get("max_steps"),
                force_path=spec["path"],
            )
            verdicts = verify.verify_run(
                run.id,
                cik=question.get("cik"),
                expected_value=question.get("expected_value"),
                ledger=Ledger(max_usd=verify_ledger_usd, max_steps=60),
            )
            out.append(
                Trajectory(
                    run=run,
                    question_id=question["id"],
                    variant=name,
                    expected_value=question.get("expected_value"),
                    verdicts=verdicts,
                )
            )
    return out


def accept_rate(trajectories: list[Trajectory]) -> dict:
    """C1, plus the comparison that motivates the whole project.

    `step_filtered` and `outcome_filtered` are the two conditions H1 compares. Reporting them
    side by side here — before any training happens — shows how much they actually differ on
    this data, which bounds how large an effect the experiment could possibly detect.
    """
    n = len(trajectories) or 1
    step_ok = [t for t in trajectories if t.all_steps_pass]
    outcome_ok = [t for t in trajectories if t.answer_correct]
    both = [t for t in trajectories if t.all_steps_pass and t.answer_correct]

    total_steps = sum(len(t.verdicts) for t in trajectories)
    passing_steps = sum(sum(1 for v in t.verdicts if v.passed) for t in trajectories)

    return {
        "n_trajectories": len(trajectories),
        "c1_accept_rate": round(len(step_ok) / n, 4),
        "outcome_accept_rate": round(len(outcome_ok) / n, 4),
        "both": round(len(both) / n, 4),
        # The divergence: correct answers reached through at least one bad step. These are
        # exactly the trajectories the two filters disagree about.
        "correct_but_flawed": round(
            sum(1 for t in trajectories if t.answer_correct and not t.all_steps_pass) / n, 4
        ),
        "step_pass_rate": round(passing_steps / total_steps, 4) if total_steps else 0.0,
        "by_variant": {
            v: {
                "n": sum(1 for t in trajectories if t.variant == v),
                "accept": round(
                    sum(1 for t in trajectories if t.variant == v and t.all_steps_pass)
                    / max(1, sum(1 for t in trajectories if t.variant == v)),
                    4,
                ),
                "correct": round(
                    sum(1 for t in trajectories if t.variant == v and t.answer_correct)
                    / max(1, sum(1 for t in trajectories if t.variant == v)),
                    4,
                ),
            }
            for v in sorted({t.variant for t in trajectories})
        },
    }
