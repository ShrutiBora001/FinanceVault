"""Run policies over the frozen split and report the metric table.

Every policy answers the same questions at the same as-of horizons and is scored by the same
verifier, so the comparison is between policies rather than between measurement setups.

Under `FV_REPLAY=true` the whole sweep resolves from the journal: the table reproduces at
$0.00, which is what makes iterating on the harness affordable and what F2 measures.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from financevault import verify
from financevault.env.asof import AsOf
from financevault.runtime.budget import Ledger
from financevault.store import pg

SPLIT = Path("eval/splits/mvp_150.jsonl")
RESULTS = Path("eval/results")


def load_split(path: Path = SPLIT, subset: str | None = None) -> list[dict]:
    """Load the frozen split, optionally restricted to train/dev/test.

    `subset` defaults to everything. Development should read `dev` and leave `test` alone:
    a test set you have iterated against is a validation set wearing the wrong label.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python scripts/make_split.py`")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if subset:
        rows = [r for r in rows if r.get("split") == subset]
    return rows


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """A 95% confidence interval for a proportion, by the Wilson score method.

    Wilson rather than the normal approximation because the normal one is badly wrong at the
    edges — with 31 questions and a policy scoring 100%, it gives the interval [1.0, 1.0],
    which asserts certainty from a sample that cannot support it. Wilson keeps a sensible
    width at 0 and 1, which is exactly where a benchmark's headline numbers sit.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    spread = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


@dataclass(slots=True)
class Outcome:
    question_id: str
    archetype: str
    run_id: str
    correct: bool
    value: float | None
    expected: float
    cost_usd: float  # actually spent (0 on a journal hit)
    list_usd: float  # would-be cost on a cold journal
    latency_ms: int
    n_steps: int
    outcome: str
    n_calls: int = 0
    cached_calls: int = 0
    step_scores: list[dict[str, float]] = field(default_factory=list)
    unsupported_claims: float = 0.0

    @property
    def steps_passed(self) -> int:
        return sum(
            1
            for s in self.step_scores
            if all(s.get(k, 1.0) >= 0.5 for k in ("s1", "s2", "s3", "s4"))
        )

    @property
    def timed_cold(self) -> bool:
        """Whether this run's wall time measures the policy rather than the journal.

        A run served entirely from cache returns in the time it takes to hash a key and read
        a row. That is a real number about the harness and a meaningless one about the
        policy, so it must not enter E2.
        """
        return self.n_calls > 0 and self.cached_calls == 0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return ordered[idx]


def run_policy(name: str, questions: list[dict], *, verify_steps: bool = True) -> list[Outcome]:
    from eval.baselines import POLICIES  # noqa: PLC0415 - keeps the module importable standalone

    policy = POLICIES[name]
    outcomes: list[Outcome] = []

    for q in questions:
        started = time.monotonic()
        run = policy(
            q["question"],
            as_of=AsOf.parse(q["as_of"]),
            cik=q.get("cik"),
            ticker=q.get("ticker"),
        )
        latency_ms = int((time.monotonic() - started) * 1000)

        verdicts = []
        if verify_steps:
            # Verification runs on its own ledger so its cost is never attributed to the
            # policy being measured.
            verdicts = verify.verify_run(
                run.id,
                cik=q.get("cik"),
                expected_value=q["expected_value"],
                ledger=Ledger(max_usd=0.05, max_steps=40),
            )

        terminal = verdicts[-1] if verdicts else None
        outcomes.append(
            Outcome(
                question_id=q["id"],
                archetype=q["archetype"],
                run_id=run.id,
                correct=verify.answered_correctly(verdicts),
                value=run.value,
                expected=q["expected_value"],
                cost_usd=run.ledger.usd if run.ledger else 0.0,
                list_usd=run.ledger.list_usd if run.ledger else 0.0,
                latency_ms=latency_ms,
                n_steps=len(run.steps),
                outcome=run.outcome,
                n_calls=len(run.ledger.entries) if run.ledger else 0,
                cached_calls=(
                    sum(1 for e in run.ledger.entries if e["cached"]) if run.ledger else 0
                ),
                step_scores=[v.scores() for v in verdicts],
                unsupported_claims=(1.0 - terminal.signals["s3"].score if terminal else 1.0),
            )
        )
    return outcomes


def summarise(name: str, outcomes: list[Outcome]) -> dict[str, Any]:
    n = len(outcomes) or 1
    correct = [o for o in outcomes if o.correct]
    total_steps = sum(len(o.step_scores) for o in outcomes)
    passed_steps = sum(o.steps_passed for o in outcomes)
    # E2 is measured only on runs that actually called the model. Mixing cached runs into the
    # distribution does not add noise, it changes what is being measured: a journal hit is a
    # hash and a row read, so a fully-cached sweep reports single-digit milliseconds for a
    # policy that takes seconds. The first sweep compared 4-73ms cache lookups for three
    # policies against 17,333ms of live API calls for the fourth and presented it as a latency
    # comparison between policies.
    timed = [o for o in outcomes if o.timed_cold]
    latencies = [float(o.latency_ms) for o in timed]
    spend = sum(o.cost_usd for o in outcomes)
    list_spend = sum(o.list_usd for o in outcomes)

    return {
        "policy": name,
        "n": len(outcomes),
        # D1: did it get the number right?
        "d1_accuracy": round(len(correct) / n, 4),
        "d1_ci95": [round(b, 4) for b in wilson_interval(len(correct), len(outcomes))],
        # D2: share of numeric claims not traceable to retrieved evidence.
        "d2_unsupported_claim_rate": round(sum(o.unsupported_claims for o in outcomes) / n, 4),
        # D3: the share of *steps* that verify, which is what localises a gain.
        "d3_step_pass_rate": round(passed_steps / total_steps, 4) if total_steps else 0.0,
        # E1: the joint metric. Infinite when nothing is correct, which is the honest value.
        # List cost, not actual: E1 describes the policy, so it must not collapse to
        # zero merely because a previous sweep already journalled these calls.
        "e1_cost_per_correct": round(list_spend / len(correct), 6) if correct else None,
        # None rather than 0 when nothing ran cold: a replayed sweep has no latency to report,
        # and saying so is the only honest option. `e2_n` travels with the figures so a reader
        # can see how many runs they rest on.
        "e2_p50_latency_ms": int(_percentile(latencies, 50)) if timed else None,
        "e2_p95_latency_ms": int(_percentile(latencies, 95)) if timed else None,
        "e2_n": len(timed),
        "total_list_usd": round(list_spend, 6),
        "total_spent_usd": round(spend, 6),
        "mean_steps": round(sum(o.n_steps for o in outcomes) / n, 2),
        "outcomes": {
            o: sum(1 for x in outcomes if x.outcome == o) for o in {x.outcome for x in outcomes}
        },
        "accuracy_by_archetype": {
            a: round(
                sum(1 for o in outcomes if o.archetype == a and o.correct)
                / max(1, sum(1 for o in outcomes if o.archetype == a)),
                4,
            )
            for a in sorted({o.archetype for o in outcomes})
        },
    }


def journal_state() -> dict[str, Any]:
    from financevault.runtime import journal  # noqa: PLC0415

    return {k: float(v) if hasattr(v, "quantize") else v for k, v in journal.stats().items()}


def main() -> int:
    import argparse

    from rich.console import Console
    from rich.table import Table

    from financevault.config import settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", default="b0-no-tools,b1-rag,b2-tools,b3-frontier")
    parser.add_argument("--subset", default="dev", help="train | dev | test | all")
    parser.add_argument("--limit", type=int, default=0, help="first N questions (0 = all)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    console = Console()
    subset = None if args.subset == "all" else args.subset
    questions = load_split(subset=subset)
    if args.limit:
        questions = questions[: args.limit]

    if pg.missing_tables():
        console.print("[red]schema not applied[/red] — run `make migrate`")
        return 1

    replay = settings().replay
    console.print(
        f"[bold]{len(questions)}[/bold] questions ({args.subset}) | "
        f"analyst=[cyan]{settings().analyst_model}[/cyan] | "
        f"judge=[cyan]{settings().judge_model}[/cyan] | "
        + ("[green]REPLAY[/green]" if replay else "[yellow]LIVE[/yellow]")
    )

    before = journal_state()
    summaries = []
    for name in [p.strip() for p in args.policies.split(",") if p.strip()]:
        console.print(f"\n[bold]{name}[/bold] …")
        outcomes = run_policy(name, questions)
        summary = summarise(name, outcomes)
        summaries.append(summary)
        console.print(
            f"  accuracy {summary['d1_accuracy']:.0%} | "
            f"step pass {summary['d3_step_pass_rate']:.0%} | "
            f"list ${summary['total_list_usd']:.4f}"
        )
    after = journal_state()

    table = Table(title="MVP1 baselines", header_style="bold")
    table.add_column("policy")
    table.add_column("D1 acc", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("D2 unsup", justify="right")
    table.add_column("D3 step", justify="right")
    table.add_column("E1 $/correct", justify="right")
    table.add_column("E2 p95 ms", justify="right")
    table.add_column("E2 n", justify="right")
    table.add_column("steps", justify="right")
    table.add_column("list $", justify="right")
    for s in summaries:
        table.add_row(
            s["policy"],
            f"{s['d1_accuracy']:.0%}",
            f"{s['d1_ci95'][0]:.0%}-{s['d1_ci95'][1]:.0%}",
            f"{s['d2_unsupported_claim_rate']:.0%}",
            f"{s['d3_step_pass_rate']:.0%}",
            "—" if s["e1_cost_per_correct"] is None else f"${s['e1_cost_per_correct']:.4f}",
            "—" if s["e2_p95_latency_ms"] is None else f"{s['e2_p95_latency_ms']:,}",
            f"{s['e2_n']}/{s['n']}",
            f"{s['mean_steps']}",
            f"${s['total_list_usd']:.4f}",
        )
    console.print()
    console.print(table)

    if any(s["e2_n"] < s["n"] for s in summaries):
        console.print(
            "\n[yellow]E2 note[/yellow] latency covers only runs that called the model. "
            "Cached runs measure the journal, not the policy, and are excluded — "
            "a policy showing [bold]0/n[/bold] replayed entirely and has no latency to report."
        )

    spent = round(float(after.get("usd", 0)) - float(before.get("usd", 0)), 6)
    console.print(
        f"\n[bold]F2[/bold] journal spend this sweep: [cyan]${spent:.4f}[/cyan]"
        + ("  [green](replay: $0.00 expected)[/green]" if replay else "")
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    default_name = f"mvp2.2_{args.subset}{'_replay' if replay else ''}.json"
    out = Path(args.out) if args.out else RESULTS / default_name
    out.write_text(
        json.dumps(
            {
                "replay": replay,
                "analyst_model": settings().analyst_model,
                "judge_model": settings().judge_model,
                "n_questions": len(questions),
                "subset": args.subset,
                "journal_spend_usd": spent,
                "policies": summaries,
            },
            indent=2,
        )
        + "\n"
    )
    console.print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
