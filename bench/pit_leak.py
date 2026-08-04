"""A1 — lookahead leak rate.

How often does a retriever return a document that had not been published yet?

Three strategies are compared at identical horizons over identical queries:

| strategy | filter | what it represents |
|---|---|---|
| `none` | no date filter at all | the floor — what you get by not thinking about time |
| `period_end` | `period_end <= as_of` | **the real comparison** |
| `accepted_at` | `accepted_at <= as_of` | FinanceVault |

`period_end` is the one that matters. It is not a strawman: filtering on the fiscal period a
document *describes* is the obvious, careful-looking choice, and it is wrong. A 10-K covering
a year that ended in September is not public until it is accepted in late October, so every
horizon inside that reporting lag admits a document from the future. The gap between the
`period_end` and `accepted_at` columns is the entire argument for the design.

Horizons are swept across the corpus rather than taken from the eval split, whose horizons all
sit after the newest filing and would understate every strategy equally.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from rich.console import Console
from rich.table import Table

from financevault.store import pg

console = Console()
OUT = Path("eval/results/a1_leak.json")

CIK = "320193"
N_HORIZONS = 24

STRATEGIES = {
    "none": "TRUE",
    "period_end": "f.period_end <= %(as_of)s::date",
    "accepted_at": "f.accepted_at <= %(as_of)s",
}

# Leakage is judged against the one true criterion regardless of how a strategy filtered.
FILINGS_SQL = """
SELECT f.accession, f.form, f.period_end, f.accepted_at,
       (f.accepted_at > %(as_of)s) AS leaked
FROM filings f
WHERE f.cik = %(cik)s AND ({filter})
"""

FACTS_SQL = """
SELECT count(*) AS n,
       count(*) FILTER (WHERE f.accepted_at > %(as_of)s) AS leaked
FROM xbrl_facts f
WHERE f.cik = %(cik)s AND ({filter})
"""


@dataclass(slots=True)
class Tally:
    returned: int = 0
    leaked: int = 0
    # Per-horizon leak rates. The pooled rate averages horizons where nothing leaks together
    # with horizons deep inside a reporting lag, which understates how bad the bad ones are.
    per_horizon: list[float] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.leaked / self.returned if self.returned else 0.0

    @property
    def horizons_affected(self) -> int:
        return sum(1 for r in self.per_horizon if r > 0)

    @property
    def worst_horizon_rate(self) -> float:
        return max(self.per_horizon, default=0.0)

    def summary(self) -> dict:
        return {
            "returned": self.returned,
            "leaked": self.leaked,
            "pooled_rate": round(self.rate, 6),
            "horizons_affected": self.horizons_affected,
            "worst_horizon_rate": round(self.worst_horizon_rate, 6),
        }


def horizons(n: int) -> list[str]:
    """Instants spread across the corpus, so filters are exercised inside reporting lags."""
    span = pg.fetch_one(
        "SELECT min(accepted_at) lo, max(accepted_at) hi FROM filings WHERE cik = %s", (CIK,)
    )
    assert span and span["lo"] and span["hi"]
    # Start a little before the first acceptance and end a little after the last, so the
    # sweep covers the empty and saturated ends as well as the interesting middle.
    lo = span["lo"] - timedelta(days=30)
    total = (span["hi"] + timedelta(days=30) - lo).total_seconds()
    return [(lo + timedelta(seconds=total * i / (n - 1))).isoformat() for i in range(n)]


def measure() -> dict:
    points = horizons(N_HORIZONS)
    filings: dict[str, Tally] = {k: Tally() for k in STRATEGIES}
    facts: dict[str, Tally] = {k: Tally() for k in STRATEGIES}
    worst: dict[str, dict] = {}

    for as_of in points:
        params = {"cik": CIK, "as_of": as_of}
        for name, clause in STRATEGIES.items():
            rows = pg.fetch_all(FILINGS_SQL.format(filter=clause), params)
            leaked = [r for r in rows if r["leaked"]]
            filings[name].returned += len(rows)
            filings[name].leaked += len(leaked)
            filings[name].per_horizon.append(len(leaked) / len(rows) if rows else 0.0)

            if leaked and name == "period_end":
                # Keep the most egregious example: the document published furthest after the
                # horizon that a period_end filter still admitted.
                for row in leaked:
                    gap = (row["accepted_at"] - row["accepted_at"].fromisoformat(as_of)).days
                    if gap > worst.get("gap_days", -1):
                        worst = {
                            "gap_days": gap,
                            "as_of": as_of[:10],
                            "form": row["form"],
                            "period_end": str(row["period_end"]),
                            "accepted_at": row["accepted_at"].isoformat(),
                        }

            row = pg.fetch_one(FACTS_SQL.format(filter=clause), params)
            if row:
                n, lk = row["n"] or 0, row["leaked"] or 0
                facts[name].returned += n
                facts[name].leaked += lk
                facts[name].per_horizon.append(lk / n if n else 0.0)

    return {
        "cik": CIK,
        "n_horizons": len(points),
        "horizon_range": [points[0][:10], points[-1][:10]],
        "n_filings_in_corpus": pg.fetch_value(
            "SELECT count(*) FROM filings WHERE cik = %s", (CIK,)
        ),
        "filings": {k: v.summary() for k, v in filings.items()},
        "xbrl_facts": {k: v.summary() for k, v in facts.items()},
        "worst_period_end_leak": worst or None,
    }


def main() -> int:
    if pg.missing_tables() or not pg.fetch_value("SELECT count(*) FROM filings"):
        console.print("[red]no corpus[/red] — run `make ingest`")
        return 1

    result = measure()
    console.print(
        f"[bold]A1 lookahead leak rate[/bold] — {result['n_horizons']} horizons, "
        f"{result['horizon_range'][0]} .. {result['horizon_range'][1]}\n"
    )

    table = Table(header_style="bold")
    table.add_column("filter")
    table.add_column("filings leaked", justify="right")
    table.add_column("pooled", justify="right")
    table.add_column("horizons hit", justify="right")
    table.add_column("worst horizon", justify="right")
    table.add_column("facts pooled", justify="right")
    for name in STRATEGIES:
        f, x = result["filings"][name], result["xbrl_facts"][name]
        label = {
            "none": "no filter",
            "period_end": "period_end (naive)",
            "accepted_at": "accepted_at (FinanceVault)",
        }[name]
        table.add_row(
            label,
            f"{f['leaked']:,}/{f['returned']:,}",
            f"{f['pooled_rate']:.2%}",
            f"{f['horizons_affected']}/{result['n_horizons']}",
            f"{f['worst_horizon_rate']:.2%}",
            f"{x['pooled_rate']:.2%}",
        )
    console.print(table)

    if worst := result["worst_period_end_leak"]:
        console.print(
            f"\nworst case under a period_end filter: a {worst['form']} for the period ending "
            f"{worst['period_end']} was visible at as_of {worst['as_of']}, "
            f"[red]{worst['gap_days']} days before it was accepted[/red] "
            f"({worst['accepted_at'][:10]})."
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    console.print(f"\nwrote {OUT}")

    ours = (
        result["filings"]["accepted_at"]["pooled_rate"]
        + result["xbrl_facts"]["accepted_at"]["pooled_rate"]
    )
    if ours != 0.0:
        console.print("[red]FAIL[/red] — the as-of path leaked; this must be exactly 0")
        return 1
    console.print("[green]ok[/green] — the as-of path leaked nothing at any horizon")
    return 0


if __name__ == "__main__":
    sys.exit(main())
