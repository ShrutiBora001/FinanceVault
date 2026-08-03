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

SPLIT = Path("eval/splits/mvp_20.jsonl")
RESULTS = Path("eval/results")


def load_split(path: Path = SPLIT) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python scripts/make_split.py`")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
    step_scores: list[dict[str, float]] = field(default_factory=list)
    unsupported_claims: float = 0.0

    @property
    def steps_passed(self) -> int:
        return sum(
            1
            for s in self.step_scores
            if all(s.get(k, 1.0) >= 0.5 for k in ("s1", "s2", "s3", "s4"))
        )


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
                correct=bool(terminal and terminal.signals["s5"].score >= 0.5),
                value=run.value,
                expected=q["expected_value"],
                cost_usd=run.ledger.usd if run.ledger else 0.0,
                list_usd=run.ledger.list_usd if run.ledger else 0.0,
                latency_ms=latency_ms,
                n_steps=len(run.steps),
                outcome=run.outcome,
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
    latencies = [float(o.latency_ms) for o in outcomes]
    spend = sum(o.cost_usd for o in outcomes)
    list_spend = sum(o.list_usd for o in outcomes)

    return {
        "policy": name,
        "n": len(outcomes),
        # D1: did it get the number right?
        "d1_accuracy": round(len(correct) / n, 4),
        # D2: share of numeric claims not traceable to retrieved evidence.
        "d2_unsupported_claim_rate": round(sum(o.unsupported_claims for o in outcomes) / n, 4),
        # D3: the share of *steps* that verify, which is what localises a gain.
        "d3_step_pass_rate": round(passed_steps / total_steps, 4) if total_steps else 0.0,
        # E1: the joint metric. Infinite when nothing is correct, which is the honest value.
        # List cost, not actual: E1 describes the policy, so it must not collapse to
        # zero merely because a previous sweep already journalled these calls.
        "e1_cost_per_correct": round(list_spend / len(correct), 6) if correct else None,
        "e2_p50_latency_ms": int(_percentile(latencies, 50)),
        "e2_p95_latency_ms": int(_percentile(latencies, 95)),
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
    parser.add_argument("--policies", default="b1-rag,b2-tools")
    parser.add_argument("--limit", type=int, default=0, help="first N questions (0 = all)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    console = Console()
    questions = load_split()
    if args.limit:
        questions = questions[: args.limit]

    if pg.missing_tables():
        console.print("[red]schema not applied[/red] — run `make migrate`")
        return 1

    replay = settings().replay
    console.print(
        f"[bold]{len(questions)}[/bold] questions | "
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
    table.add_column("D2 unsup", justify="right")
    table.add_column("D3 step", justify="right")
    table.add_column("E1 $/correct", justify="right")
    table.add_column("E2 p95 ms", justify="right")
    table.add_column("steps", justify="right")
    table.add_column("list $", justify="right")
    for s in summaries:
        table.add_row(
            s["policy"],
            f"{s['d1_accuracy']:.0%}",
            f"{s['d2_unsupported_claim_rate']:.0%}",
            f"{s['d3_step_pass_rate']:.0%}",
            "—" if s["e1_cost_per_correct"] is None else f"${s['e1_cost_per_correct']:.4f}",
            f"{s['e2_p95_latency_ms']:,}",
            f"{s['mean_steps']}",
            f"${s['total_list_usd']:.4f}",
        )
    console.print()
    console.print(table)

    spent = round(float(after.get("usd", 0)) - float(before.get("usd", 0)), 6)
    console.print(
        f"\n[bold]F2[/bold] journal spend this sweep: [cyan]${spent:.4f}[/cyan]"
        + ("  [green](replay: $0.00 expected)[/green]" if replay else "")
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RESULTS / ("mvp1_replay.json" if replay else "mvp1.json")
    out.write_text(
        json.dumps(
            {
                "replay": replay,
                "analyst_model": settings().analyst_model,
                "judge_model": settings().judge_model,
                "n_questions": len(questions),
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
