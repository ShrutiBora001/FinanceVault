"""Generate the frozen evaluation split from XBRL facts.

Ground truth is derived from the same facts the environment serves, which makes labelling
free and correct by construction — but also means an answer can be right without the agent
having understood anything. That is acceptable for MVP1, whose question is whether the
pipeline works. It is *not* acceptable for the published benchmark, where at least some
values must be read off the filing document itself. Recorded as an open concern.

Two archetypes, both numeric:

- **lookup** — one reported figure for one period.
- **delta** — the change between two periods, which requires two retrievals and arithmetic,
  so it exercises the multi-step path rather than a single tool call.

The `as_of` on every question is set *after* the answering filing was accepted, so a correct
answer is always reachable. Questions whose answer postdates their horizon would measure the
as-of filter rather than the agent.

Output is written once and committed. Regenerating it changes the benchmark, so the file is
the artefact, not this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, timedelta
from pathlib import Path

from rich.console import Console

from financevault.store import pg

console = Console()

OUT = Path("eval/splits/mvp_20.jsonl")

# Tags with several years of clean coverage, phrased the way someone would actually ask.
LOOKUP_TAGS: dict[str, str] = {
    "NetIncomeLoss": "net income",
    "GrossProfit": "gross profit",
    "OperatingIncomeLoss": "operating income",
    "CostOfGoodsAndServicesSold": "cost of sales",
    "IncomeTaxExpenseBenefit": "income tax expense",
    "OperatingExpenses": "total operating expenses",
    "ResearchAndDevelopmentExpense": "research and development expense",
}
INSTANT_TAGS: dict[str, str] = {
    "Assets": "total assets",
    "StockholdersEquity": "total shareholders' equity",
    "CashAndCashEquivalentsAtCarryingValue": "cash and cash equivalents",
    "InventoryNet": "net inventory",
}

FACTS_SQL = """
SELECT DISTINCT ON (tag, period_end) tag, period_end, value, accepted_at
FROM xbrl_facts
WHERE cik = %(cik)s AND unit = 'USD' AND form = '10-K' AND tag = ANY(%(tags)s)
  AND period_end IS NOT NULL
  AND (period_start IS NOT NULL) = %(duration)s
ORDER BY tag, period_end DESC, accepted_at DESC
"""


def fiscal_year(period_end) -> int:
    """The fiscal year a period belongs to, derived from its end date.

    Deliberately *not* the `fy` column. In companyfacts, `fy` is the fiscal year of the
    **filing that reported the fact**, not of the fact's own period — Apple's FY2025 10-K
    carries FY2025, FY2024 and FY2023 figures all tagged `fy=2025`. Using it produced two
    different "fiscal year 2025" questions with different answers, and a delta question
    spanning "fiscal year 2025 to fiscal year 2025".

    The end date's calendar year is the right label for companies whose fiscal year ends in
    the second half of the calendar year (Apple, September) and for the common January and
    June year-ends, which is every company in the ingest list.
    """
    return period_end.year


def _horizon(accepted_at) -> str:
    """A day after acceptance: late enough that the answer exists, early enough to be tight."""
    return (accepted_at + timedelta(days=1)).astimezone(UTC).date().isoformat()


def build(cik: str, ticker: str, limit: int) -> list[dict]:
    questions: list[dict] = []

    for tags, duration in ((LOOKUP_TAGS, True), (INSTANT_TAGS, False)):
        rows = pg.fetch_all(FACTS_SQL, {"cik": cik, "tags": list(tags), "duration": duration})
        by_tag: dict[str, list[dict]] = {}
        for row in rows:
            by_tag.setdefault(row["tag"], []).append(row)

        for tag, label in tags.items():
            facts = sorted(by_tag.get(tag, []), key=lambda r: r["period_end"], reverse=True)[:3]
            for fact in facts:
                fy = fiscal_year(fact["period_end"])
                questions.append(
                    {
                        "id": f"{ticker.lower()}-{tag.lower()}-{fy}",
                        "archetype": "lookup",
                        "question": (
                            f"What was {ticker}'s {label} "
                            + (
                                f"in fiscal year {fy}?"
                                if duration
                                else f"at the end of fiscal year {fy}?"
                            )
                        ),
                        "ticker": ticker,
                        "cik": cik,
                        "as_of": _horizon(fact["accepted_at"]),
                        "expected_value": float(fact["value"]),
                        "expected_unit": "USD",
                        "source_tag": tag,
                        "period_end": str(fact["period_end"]),
                    }
                )

            # One delta per tag: two retrievals plus arithmetic, so the multi-step path is
            # exercised rather than only single-call lookups.
            if len(facts) >= 2:
                new, old = facts[0], facts[1]
                new_fy = fiscal_year(new["period_end"])
                questions.append(
                    {
                        "id": f"{ticker.lower()}-{tag.lower()}-delta-{new_fy}",
                        "archetype": "delta",
                        "question": (
                            f"By how much did {ticker}'s {label} change from fiscal year "
                            f"{fiscal_year(old['period_end'])} to fiscal year {new_fy}? "
                            "Give the difference in dollars."
                        ),
                        "ticker": ticker,
                        "cik": cik,
                        "as_of": _horizon(new["accepted_at"]),
                        "expected_value": float(new["value"]) - float(old["value"]),
                        "expected_unit": "USD",
                        "source_tag": tag,
                        "period_end": str(new["period_end"]),
                    }
                )

    # Interleave archetypes so a truncated split is not all one kind.
    lookups = [q for q in questions if q["archetype"] == "lookup"]
    deltas = [q for q in questions if q["archetype"] == "delta"]
    mixed: list[dict] = []
    while (lookups or deltas) and len(mixed) < limit:
        for bucket in (lookups, lookups, deltas):  # roughly 2:1, matching real usage
            if bucket and len(mixed) < limit:
                mixed.append(bucket.pop(0))
    return mixed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cik", default="320193")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--force", action="store_true", help="overwrite a frozen split")
    args = parser.parse_args()

    if OUT.exists() and not args.force:
        console.print(
            f"[yellow]{OUT} exists[/yellow] — the split is frozen. Pass --force to replace."
        )
        return 1

    questions = build(args.cik, args.ticker, args.limit)
    if len(questions) < args.limit:
        console.print(f"[red]only {len(questions)} questions available[/red], wanted {args.limit}")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(q, sort_keys=True) for q in questions) + "\n")

    counts: dict[str, int] = {}
    for q in questions:
        counts[q["archetype"]] = counts.get(q["archetype"], 0) + 1
    console.print(f"[green]wrote {len(questions)}[/green] questions to {OUT}")
    console.print(f"  archetypes: {counts}")
    console.print(
        f"  as_of range: {min(q['as_of'] for q in questions)} .. "
        f"{max(q['as_of'] for q in questions)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
