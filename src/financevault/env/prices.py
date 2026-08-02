"""Daily price bars, backing the SQL tool.

Sourced via yfinance. Stooq was the first choice — plain CSV, no key — but it now gates every
request behind a JavaScript bot-detection challenge, and working around bot detection is not
something this project does.

Prices carry no point-in-time column. A daily bar for date D is knowable at D's close, so the
as-of filter on this table is `date <= as_of`, applied by the SQL tool rather than stored per
row. Note `adj_close` is retroactively rewritten by every later split and dividend, so it is
*not* point-in-time safe; anything as-of-sensitive must use `close`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import yfinance as yf

from financevault.store import pg


@dataclass(frozen=True, slots=True)
class Bar:
    ticker: str
    date: date
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    adj_close: float | None
    volume: int | None


def _num(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if result != result else result  # drop NaN


def fetch_bars(ticker: str, start: date | None = None) -> list[Bar]:
    """Daily bars from `start` (inclusive) to today, oldest first."""
    frame = yf.Ticker(ticker).history(
        start=start.isoformat() if start else None,
        period=None if start else "2y",
        interval="1d",
        auto_adjust=False,
    )
    if frame.empty:
        return []

    bars: list[Bar] = []
    for index, row in frame.iterrows():
        volume = _num(row.get("Volume"))
        bars.append(
            Bar(
                ticker=ticker.upper(),
                date=index.date(),
                open=_num(row.get("Open")),
                high=_num(row.get("High")),
                low=_num(row.get("Low")),
                close=_num(row.get("Close")),
                adj_close=_num(row.get("Adj Close")),
                volume=int(volume) if volume is not None else None,
            )
        )
    return bars


INSERT_BAR = """
INSERT INTO prices (ticker, date, open, high, low, close, adj_close, volume)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (ticker, date) DO UPDATE SET
    open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
    close = EXCLUDED.close, adj_close = EXCLUDED.adj_close, volume = EXCLUDED.volume
"""


def store_bars(bars: list[Bar]) -> int:
    rows = [(b.ticker, b.date, b.open, b.high, b.low, b.close, b.adj_close, b.volume) for b in bars]
    return pg.execute_many(INSERT_BAR, rows)
