"""SEC EDGAR client and filing ingest.

SEC caps automated access at 10 requests/second and requires a descriptive User-Agent with
real contact details. Both are handled here: every request goes through `_get`, which
throttles and sets headers, so no other module needs to remember the rules.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Any

import httpx
from bs4 import BeautifulSoup

from financevault.config import settings
from financevault.store import pg

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"

# SEC's published ceiling is 10 req/s. Half that leaves headroom and still ingests a
# company in well under a minute.
_MIN_INTERVAL = 0.2
_last_request = 0.0
_throttle = Lock()


def _get(url: str, *, timeout: float = 30.0) -> httpx.Response:
    """Throttled, correctly-headed GET. All EDGAR traffic goes through here."""
    global _last_request
    with _throttle:
        elapsed = time.monotonic() - _last_request
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _last_request = time.monotonic()

    response = httpx.get(
        url, headers=settings().sec_headers, timeout=timeout, follow_redirects=True
    )
    response.raise_for_status()
    return response


@dataclass(frozen=True, slots=True)
class Filing:
    cik: str
    ticker: str | None
    accession: str
    form: str
    period_end: datetime | None
    filed_at: datetime | None
    accepted_at: datetime
    url: str


def _parse_instant(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def fetch_submissions(cik: int) -> dict[str, Any]:
    return _get(SUBMISSIONS_URL.format(cik=cik)).json()


def list_filings(
    cik: int, forms: tuple[str, ...] = ("10-K", "10-Q"), limit: int = 4
) -> list[Filing]:
    """Most recent filings of the given forms, newest first.

    `acceptanceDateTime` is carried through as `accepted_at`. It is the only field here that
    reflects when the document became public; `filingDate` is a date and `reportDate` is the
    fiscal period end, and neither is a safe point-in-time key.
    """
    data = fetch_submissions(cik)
    ticker = (data.get("tickers") or [None])[0]
    recent = data["filings"]["recent"]
    cik_str = str(int(data["cik"]))

    out: list[Filing] = []
    for i, form in enumerate(recent["form"]):
        if form not in forms:
            continue
        accepted_at = _parse_instant(recent["acceptanceDateTime"][i])
        if accepted_at is None:
            # Without an acceptance instant the row cannot be filtered safely, so it is
            # skipped rather than admitted with a guessed timestamp.
            continue
        accession = recent["accessionNumber"][i]
        out.append(
            Filing(
                cik=cik_str,
                ticker=ticker,
                accession=accession,
                form=form,
                period_end=_parse_instant(recent["reportDate"][i]),
                filed_at=_parse_instant(recent["filingDate"][i]),
                accepted_at=accepted_at,
                url=ARCHIVE_URL.format(
                    cik=cik_str,
                    accession_nodash=accession.replace("-", ""),
                    document=recent["primaryDocument"][i],
                ),
            )
        )
        if len(out) >= limit:
            break
    return out


def fetch_document_text(filing: Filing) -> str:
    """Primary document as plain text, with script/style and inline-XBRL noise stripped."""
    html = _get(filing.url).text
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


# ---------------------------------------------------------------- persistence


UPSERT_FILING = """
INSERT INTO filings (cik, ticker, accession, form, period_end, filed_at, accepted_at, url)
VALUES (%(cik)s, %(ticker)s, %(accession)s, %(form)s, %(period_end)s, %(filed_at)s,
        %(accepted_at)s, %(url)s)
ON CONFLICT (accession) DO UPDATE SET
    ticker      = EXCLUDED.ticker,
    form        = EXCLUDED.form,
    period_end  = EXCLUDED.period_end,
    filed_at    = EXCLUDED.filed_at,
    accepted_at = EXCLUDED.accepted_at,
    url         = EXCLUDED.url
RETURNING id
"""


def upsert_filing(filing: Filing) -> int:
    """Insert or refresh a filing, returning its id. Idempotent on accession number."""
    row = pg.fetch_one(
        UPSERT_FILING,
        {
            "cik": filing.cik,
            "ticker": filing.ticker,
            "accession": filing.accession,
            "form": filing.form,
            "period_end": filing.period_end.date() if filing.period_end else None,
            "filed_at": filing.filed_at,
            "accepted_at": filing.accepted_at,
            "url": filing.url,
        },
    )
    assert row is not None
    return row["id"]
