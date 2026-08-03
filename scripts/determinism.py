"""F1 and F2: does the replayed sweep reproduce the live one, and what did it cost?

F1 compares the two metric tables field by field. It is a claim about the *pipeline* — that
the journal key is stable and a replay resolves to the same rows — not about the model, which
cannot be pinned to a temperature and may answer the same question differently on two live
calls. Stating it the other way would overclaim.

F2 is the sweep's actual spend, live against replayed. That gap is what makes iterating on
the harness affordable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

LIVE = Path("eval/results/mvp1.json")
REPLAY = Path("eval/results/mvp1_replay.json")

COMPARED = (
    "d1_accuracy",
    "d2_unsupported_claim_rate",
    "d3_step_pass_rate",
    "e1_cost_per_correct",
    "total_list_usd",
    "mean_steps",
    "n",
)


def main() -> int:
    for path in (LIVE, REPLAY):
        if not path.exists():
            console.print(f"[red]{path} missing[/red] — run `make eval` then `make replay`")
            return 1

    live = json.loads(LIVE.read_text())
    replay = json.loads(REPLAY.read_text())

    if replay.get("replay") is not True:
        console.print("[red]the replay file was not produced under FV_REPLAY[/red]")
        return 1

    same = 0
    mismatches: list[str] = []
    for a, b in zip(live["policies"], replay["policies"], strict=True):
        for key in COMPARED:
            if a.get(key) == b.get(key):
                same += 1
            else:
                mismatches.append(f"{a['policy']}.{key}: live={a.get(key)} replay={b.get(key)}")

    total = same + len(mismatches)
    rate = same / total if total else 0.0

    table = Table(title="F1 / F2", header_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("F1 metric determinism", f"{rate:.2%}  ({same}/{total} fields)")
    table.add_row("F2 sweep cost, live", f"${live['journal_spend_usd']:.4f}")
    table.add_row("F2 sweep cost, replayed", f"${replay['journal_spend_usd']:.4f}")
    console.print(table)

    for line in mismatches:
        console.print(f"  [red]mismatch[/red] {line}")

    ok = not mismatches and replay["journal_spend_usd"] == 0.0
    if ok:
        console.print("[green]ok[/green] — replay reproduced the table at $0.00")
    else:
        console.print("[red]failed[/red] — replay diverged or was not free")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
