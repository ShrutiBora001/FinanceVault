"""Ingest one or more companies: filings, chunks, XBRL facts and prices.

Idempotent throughout, so re-running picks up new filings without duplicating anything.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from rich.console import Console
from rich.table import Table

from financevault.env import chunk, edgar, prices, xbrl
from financevault.store import pg

console = Console()

# MVP1 is one company. The wider slice mixes tagging habits: a bank, a retailer and a tech
# company tag the same concepts differently, which surfaces parser bugs early.
COMPANIES: dict[str, int] = {
    "AAPL": 320193,
    "JPM": 19617,
    "WMT": 104169,
    "MSFT": 789019,
    "XOM": 34088,
}


def ingest_company(ticker: str, cik: int, *, n_filings: int, price_years: int) -> dict[str, int]:
    stats = {"filings": 0, "chunks": 0, "facts": 0, "facts_undatable": 0, "prices": 0}

    console.print(f"[bold]{ticker}[/bold] (CIK {cik})")

    console.print("  filings…", end=" ")
    filings = edgar.list_filings(cik, limit=n_filings)
    for filing in filings:
        filing_id = edgar.upsert_filing(filing)
        stats["filings"] += 1
        text = edgar.fetch_document_text(filing)
        chunks = chunk.split(text)
        stats["chunks"] += chunk.store_chunks(filing_id, chunks)
        console.print(
            f"\n    {filing.form} {filing.period_end.date() if filing.period_end else '?'} "
            f"accepted {filing.accepted_at:%Y-%m-%d %H:%M} → {len(chunks)} chunks",
            end="",
        )
    console.print()

    console.print("  xbrl facts…", end=" ")
    facts, undatable = xbrl.fetch_facts(cik)
    stats["facts"] = xbrl.store_facts(facts)
    stats["facts_undatable"] = undatable
    console.print(f"{len(facts):,} parsed, {undatable} undatable")

    console.print("  prices…", end=" ")
    start = date.today() - timedelta(days=365 * price_years)
    bars = prices.fetch_bars(ticker, start=start)
    stats["prices"] = prices.store_bars(bars)
    console.print(f"{len(bars):,} bars")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default="AAPL", help="comma-separated, or 'all'")
    parser.add_argument("--filings", type=int, default=4, help="most recent 10-K/10-Q per company")
    parser.add_argument("--price-years", type=int, default=2)
    args = parser.parse_args()

    if args.tickers == "all":
        selected = COMPANIES
    else:
        wanted = [t.strip().upper() for t in args.tickers.split(",")]
        unknown = [t for t in wanted if t not in COMPANIES]
        if unknown:
            console.print(f"[red]unknown tickers:[/red] {', '.join(unknown)}")
            console.print(f"[dim]known: {', '.join(COMPANIES)}[/dim]")
            return 1
        selected = {t: COMPANIES[t] for t in wanted}

    if pg.missing_tables():
        console.print("[red]schema not applied[/red] — run `make migrate` first")
        return 1

    totals = {"filings": 0, "chunks": 0, "facts": 0, "facts_undatable": 0, "prices": 0}
    for ticker, cik in selected.items():
        stats = ingest_company(ticker, cik, n_filings=args.filings, price_years=args.price_years)
        for key, value in stats.items():
            totals[key] += value

    table = Table(title="ingest", header_style="bold")
    table.add_column("table")
    table.add_column("rows", justify="right")
    for name, count in pg.table_counts().items():
        table.add_row(name, f"{count:,}")
    console.print(table)

    if totals["facts_undatable"]:
        # Facts without a resolvable acceptance instant are dropped rather than guessed:
        # a guess in the wrong direction is a silent lookahead leak.
        console.print(
            f"[yellow]{totals['facts_undatable']} facts dropped[/yellow] — no acceptance "
            "instant resolvable from the submissions index"
        )

    empty = [
        t
        for t, c in pg.table_counts().items()
        if c == 0 and t in ("filings", "chunks", "xbrl_facts", "prices")
    ]
    if empty:
        console.print(f"[red]empty after ingest:[/red] {', '.join(empty)}")
        return 1

    console.print("[green]ok[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
