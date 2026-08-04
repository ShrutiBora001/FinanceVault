"""Generate trajectories, verify them, build the splits, and export.

Reports C1 (accept rate) alongside the outcome-filter accept rate, because the gap between
them is what H1 proposes to exploit and it bounds how large an effect the experiment could
detect.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from eval.harness import load_split
from financevault.config import settings
from financevault.data import export, rollout, splits
from financevault.store import pg

console = Console()
OUT = Path("eval/results/c1_rollout.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=10, help="questions from the split")
    parser.add_argument("--variants", default=",".join(rollout.VARIANTS))
    args = parser.parse_args()

    if pg.missing_tables():
        console.print("[red]schema not applied[/red] — run `make migrate`")
        return 1

    questions = load_split()[: args.questions]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    console.print(
        f"[bold]{len(questions)}[/bold] questions x [bold]{len(variants)}[/bold] variants "
        f"= {len(questions) * len(variants)} trajectories | "
        f"analyst=[cyan]{settings().analyst_model}[/cyan]"
        + ("  [green]REPLAY[/green]" if settings().replay else "")
    )

    trajectories = rollout.generate(questions, variants=variants)
    stats = rollout.accept_rate(trajectories)
    built = splits.build(trajectories)
    counts = export.write(built)

    table = Table(title="C1 — trajectory generation", header_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("trajectories", str(stats["n_trajectories"]))
    table.add_row("C1 accept rate (all steps pass)", f"{stats['c1_accept_rate']:.0%}")
    table.add_row("outcome accept rate (answer correct)", f"{stats['outcome_accept_rate']:.0%}")
    table.add_row("correct but flawed", f"{stats['correct_but_flawed']:.0%}")
    table.add_row("step pass rate", f"{stats['step_pass_rate']:.0%}")
    console.print(table)

    by_variant = Table(title="by variant", header_style="bold")
    by_variant.add_column("variant")
    by_variant.add_column("n", justify="right")
    by_variant.add_column("accept", justify="right")
    by_variant.add_column("correct", justify="right")
    for name, row in stats["by_variant"].items():
        by_variant.add_row(name, str(row["n"]), f"{row['accept']:.0%}", f"{row['correct']:.0%}")
    console.print(by_variant)

    exported = Table(title="exported splits", header_style="bold")
    exported.add_column("split")
    exported.add_column("examples", justify="right")
    for name, count in counts.items():
        exported.add_row(name, str(count))
    console.print(exported)

    console.print(
        f"\n[bold]the divergence[/bold]: outcome filtering keeps "
        f"[cyan]{stats['outcome_accept_rate']:.0%}[/cyan] of trajectories, "
        f"step filtering keeps [cyan]{stats['c1_accept_rate']:.0%}[/cyan]. "
        f"The {stats['correct_but_flawed']:.0%} in between is what H1 is about."
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                **stats,
                "exported": counts,
                "trajectories": [t.summary() for t in trajectories],
            },
            indent=2,
        )
        + "\n"
    )
    console.print(f"wrote {OUT} and data/sft/*.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
