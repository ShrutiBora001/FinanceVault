"""Generate the frozen evaluation split from XBRL facts.

Ground truth is derived from the same facts the environment serves, which makes labelling free
and correct by construction — but also means an answer can be right without the agent having
understood anything. That limitation is unchanged from MVP1 and still needs values read off
the filing document itself before the benchmark is published.

**Splits are disjoint by question, and assigned by hash rather than by shuffle.** A hash of the
question id decides train/dev/test, so adding a company or a tag later does not reshuffle the
questions already assigned — an existing test question stays in test. A seeded shuffle would
silently reassign everything the moment the input list changed, which is how a "frozen" split
quietly stops being frozen.

Four archetypes:

- **lookup** — one reported figure for one period.
- **delta** — the change between two periods: two retrievals plus arithmetic.
- **ratio** — one figure over another for the same period, which forces two lookups the agent
  must not confuse.
- **cross_company** — the same metric for two companies, which is the only archetype that
  cannot be answered by locking onto a single filing.

Tag availability differs by sector and the generator respects it rather than assuming. A bank
has no `GrossProfit`; asking for it would produce questions with no answer, and an agent
scoring 0 on an unanswerable question tells you nothing.

Output is written once and committed. Regenerating it changes the benchmark, so the file is
the artefact, not this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, timedelta
from pathlib import Path

from rich.console import Console
from rich.table import Table

from financevault.store import pg

console = Console()

OUT = Path("eval/splits/mvp_150.jsonl")

# Fractions are applied to a hash of the question id, so assignment is stable under additions.
SPLIT_BOUNDS = {"train": 0.60, "dev": 0.20, "test": 0.20}

COMPANIES: dict[str, str] = {
    "AAPL": "320193",
    "MSFT": "789019",
    "NVDA": "1045810",
    "JPM": "19617",
    "BRK-B": "1067983",
    "WMT": "104169",
    "COST": "909832",
    "XOM": "34088",
    "JNJ": "200406",
    "CAT": "18230",
}

DURATION_TAGS: dict[str, str] = {
    "NetIncomeLoss": "net income",
    "Revenues": "total revenue",
    "GrossProfit": "gross profit",
    "OperatingIncomeLoss": "operating income",
    "CostOfGoodsAndServicesSold": "cost of sales",
    "IncomeTaxExpenseBenefit": "income tax expense",
    "ResearchAndDevelopmentExpense": "research and development expense",
    "OperatingExpenses": "total operating expenses",
}
INSTANT_TAGS: dict[str, str] = {
    "Assets": "total assets",
    "Liabilities": "total liabilities",
    "StockholdersEquity": "total shareholders' equity",
    "CashAndCashEquivalentsAtCarryingValue": "cash and cash equivalents",
    "InventoryNet": "net inventory",
}

# Ratios only make sense between figures of the same kind over the same period.
RATIOS = [
    ("NetIncomeLoss", "Revenues", "net profit margin"),
    ("GrossProfit", "Revenues", "gross margin"),
    ("OperatingIncomeLoss", "Revenues", "operating margin"),
]

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

    Deliberately *not* the `fy` column: in companyfacts that is the fiscal year of the filing
    that reported the fact, not of the fact's own period, so a single 10-K tags three years of
    figures with one value.

    The end date's calendar year is the right label for every company in the list — verified
    against their own conventions, including NVDA and WMT (January year ends, called fiscal
    2026) and MSFT (June). A company with a spring year end would need re-checking.
    """
    return period_end.year


def _position(question_id: str) -> float:
    """A stable number in [0, 1) from the question id. No RNG, no shuffle, no seed to lose."""
    return int(hashlib.sha256(question_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def assign_splits(questions: list[dict]) -> None:
    """Assign train/dev/test in place, stratified by archetype.

    Stratified rather than globally hashed. A global hash gave a test set containing no
    `cross_company` questions at all — the hardest archetype, absent from the split that
    matters most, and invisible unless you look. Hashing within each archetype guarantees
    every split sees every kind.

    Assignment is still a hash, not a shuffle: adding a company or a tag later must not
    reshuffle questions already assigned, or a "frozen" split quietly stops being frozen.
    Within an archetype a new question can shift a neighbour across a boundary, which is a
    real limitation — regenerate deliberately, and never mid-experiment.
    """
    by_archetype: dict[str, list[dict]] = {}
    for question in questions:
        by_archetype.setdefault(question["archetype"], []).append(question)

    for group in by_archetype.values():
        group.sort(key=lambda q: _position(q["id"]))
        n = len(group)
        train_end = round(n * SPLIT_BOUNDS["train"])
        dev_end = train_end + round(n * SPLIT_BOUNDS["dev"])
        for i, question in enumerate(group):
            question["split"] = "train" if i < train_end else "dev" if i < dev_end else "test"


def _horizon(accepted_at) -> str:
    """A day after acceptance: late enough that the answer exists, early enough to be tight."""
    return (accepted_at + timedelta(days=1)).astimezone(UTC).date().isoformat()


def _facts(cik: str, tags: dict[str, str], duration: bool) -> dict[str, list[dict]]:
    rows = pg.fetch_all(FACTS_SQL, {"cik": cik, "tags": list(tags), "duration": duration})
    by_tag: dict[str, list[dict]] = {}
    for row in rows:
        by_tag.setdefault(row["tag"], []).append(row)
    for facts in by_tag.values():
        facts.sort(key=lambda r: r["period_end"], reverse=True)
    return by_tag


def build(per_company: int) -> list[dict]:
    questions: list[dict] = []

    for ticker, cik in COMPANIES.items():
        duration = _facts(cik, DURATION_TAGS, True)
        instant = _facts(cik, INSTANT_TAGS, False)
        made: list[dict] = []

        for tags, by_tag, is_duration in (
            (DURATION_TAGS, duration, True),
            (INSTANT_TAGS, instant, False),
        ):
            for tag, label in tags.items():
                facts = by_tag.get(tag, [])[:3]
                for fact in facts:
                    fy = fiscal_year(fact["period_end"])
                    when = (
                        f"in fiscal year {fy}" if is_duration else f"at the end of fiscal year {fy}"
                    )
                    made.append(
                        {
                            "id": f"{ticker.lower()}-{tag.lower()}-{fy}",
                            "archetype": "lookup",
                            "question": f"What was {ticker}'s {label} {when}?",
                            "expected_value": float(fact["value"]),
                            "as_of": _horizon(fact["accepted_at"]),
                            "source_tags": [tag],
                            "period_end": str(fact["period_end"]),
                        }
                    )

                if len(facts) >= 2:
                    new, old = facts[0], facts[1]
                    new_fy, old_fy = fiscal_year(new["period_end"]), fiscal_year(old["period_end"])
                    made.append(
                        {
                            "id": f"{ticker.lower()}-{tag.lower()}-delta-{new_fy}",
                            "archetype": "delta",
                            "question": (
                                f"By how much did {ticker}'s {label} change from fiscal year "
                                f"{old_fy} to fiscal year {new_fy}? Give the difference in dollars."
                            ),
                            "expected_value": float(new["value"]) - float(old["value"]),
                            "as_of": _horizon(new["accepted_at"]),
                            "source_tags": [tag],
                            "period_end": str(new["period_end"]),
                        }
                    )

        # Ratios: both legs must exist for the same period, which excludes sectors that do
        # not report one of them rather than producing an unanswerable question.
        for numerator, denominator, label in RATIOS:
            num_facts = {f["period_end"]: f for f in duration.get(numerator, [])}
            den_facts = {f["period_end"]: f for f in duration.get(denominator, [])}
            shared = sorted(set(num_facts) & set(den_facts), reverse=True)[:2]
            for period in shared:
                num, den = num_facts[period], den_facts[period]
                if not float(den["value"]):
                    continue
                fy = fiscal_year(period)
                made.append(
                    {
                        "id": f"{ticker.lower()}-{label.replace(' ', '')}-{fy}",
                        "archetype": "ratio",
                        "question": (
                            f"What was {ticker}'s {label} in fiscal year {fy}? "
                            "Give the answer as a percentage to two decimal places."
                        ),
                        "expected_value": round(float(num["value"]) / float(den["value"]) * 100, 2),
                        "expected_unit": "percent",
                        "as_of": _horizon(max(num["accepted_at"], den["accepted_at"])),
                        "source_tags": [numerator, denominator],
                        "period_end": str(period),
                    }
                )

        # Interleave by archetype before truncating. Appending ratios last and then slicing
        # dropped every one of them -- the archetype existed in the code and not in the data.
        buckets: dict[str, list[dict]] = {}
        for question in made:
            buckets.setdefault(question["archetype"], []).append(question)
        interleaved: list[dict] = []
        while any(buckets.values()) and len(interleaved) < per_company:
            for kind in ("lookup", "lookup", "delta", "ratio"):
                bucket = buckets.get(kind)
                if bucket and len(interleaved) < per_company:
                    interleaved.append(bucket.pop(0))

        for question in interleaved:
            question.setdefault("expected_unit", "USD")
            questions.append({**question, "ticker": ticker, "cik": cik})

    return questions


def cross_company(questions: list[dict], limit: int) -> list[dict]:
    """Same metric, two companies — the only archetype a single filing cannot answer."""
    by_key: dict[tuple[str, str], list[dict]] = {}
    for q in questions:
        if q["archetype"] != "lookup":
            continue
        by_key.setdefault((q["source_tags"][0], q["period_end"][:4]), []).append(q)

    out: list[dict] = []
    for (tag, year), group in sorted(by_key.items()):
        if len(group) < 2 or len(out) >= limit:
            continue
        a, b = group[0], group[1]
        label = DURATION_TAGS.get(tag) or INSTANT_TAGS.get(tag) or tag
        qid = f"{a['ticker'].lower()}-vs-{b['ticker'].lower()}-{tag.lower()}-{year}"
        out.append(
            {
                "id": qid,
                "archetype": "cross_company",
                "question": (
                    f"Which had higher {label} in fiscal year {year}, {a['ticker']} or "
                    f"{b['ticker']}? Give the difference in dollars as a positive number."
                ),
                "ticker": a["ticker"],
                "cik": a["cik"],
                "second_ticker": b["ticker"],
                "second_cik": b["cik"],
                "as_of": max(a["as_of"], b["as_of"]),
                "expected_value": abs(a["expected_value"] - b["expected_value"]),
                "expected_unit": "USD",
                "source_tags": [tag],
                "period_end": a["period_end"],
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-company", type=int, default=14)
    parser.add_argument("--cross", type=int, default=12)
    parser.add_argument("--force", action="store_true", help="overwrite a frozen split")
    args = parser.parse_args()

    if OUT.exists() and not args.force:
        console.print(f"[yellow]{OUT} exists[/yellow] — frozen. Pass --force to replace.")
        return 1

    questions = build(args.per_company)
    questions += cross_company(questions, args.cross)
    assign_splits(questions)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(q, sort_keys=True) for q in questions) + "\n")

    by_split = Table(title=f"{len(questions)} questions", header_style="bold")
    by_split.add_column("split")
    by_split.add_column("n", justify="right")
    by_split.add_column("archetypes")
    for name in ("train", "dev", "test"):
        rows = [q for q in questions if q["split"] == name]
        kinds = sorted({q["archetype"] for q in rows})
        by_split.add_row(name, str(len(rows)), ", ".join(kinds))
    console.print(by_split)

    by_kind = Table(header_style="bold")
    by_kind.add_column("archetype")
    by_kind.add_column("n", justify="right")
    for kind in sorted({q["archetype"] for q in questions}):
        by_kind.add_row(kind, str(sum(1 for q in questions if q["archetype"] == kind)))
    console.print(by_kind)

    coverage = sorted({q["ticker"] for q in questions})
    console.print(f"companies: {len(coverage)} — {', '.join(coverage)}")
    console.print(f"[green]wrote[/green] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
