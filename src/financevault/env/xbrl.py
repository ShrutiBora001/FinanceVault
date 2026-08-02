"""XBRL companyfacts ingest.

These facts are the ground truth for the `s3` numeric-correctness signal, so two properties
matter more here than raw coverage:

**Every fact carries a real acceptance instant.** The companyfacts API reports `filed` (a
date) but not `acceptanceDateTime`, so acceptance is joined in from the submissions index on
accession number. Submissions are paginated — the `recent` block holds the last 1000 filings
and older ones live in `filings.files[]` — and all pages are merged before the join. A fact
whose accession cannot be dated is dropped rather than admitted with a guessed timestamp: a
guess in the wrong direction is a silent lookahead leak.

**Values are absolute, in the stated unit.** companyfacts normalizes magnitudes, so a revenue
of $215.639B arrives as 215639000000 with unit USD, not as 215639 with a millions scale. The
`scale` column therefore stays NULL from this source; it exists for the *agent's* reported
scale, which is what the verifier checks an answer's magnitude against.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from financevault.env.edgar import COMPANYFACTS_URL, SUBMISSIONS_URL, _get
from financevault.store import pg

SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"


@dataclass(frozen=True, slots=True)
class Fact:
    cik: str
    taxonomy: str
    tag: str
    unit: str
    value: float
    fy: int | None
    fp: str | None
    period_start: date | None
    period_end: date | None
    form: str | None
    accession: str
    accepted_at: datetime
    frame: str | None


def _parse_instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def acceptance_map(cik: int) -> dict[str, datetime]:
    """Accession number -> acceptance instant, across every page of the submissions index.

    The paginated older files are required: for Apple, 27 of 72 fact-bearing accessions fall
    outside the `recent` window, and skipping them would drop a decade of history.
    """
    data = _get(SUBMISSIONS_URL.format(cik=cik)).json()
    recent = data["filings"]["recent"]
    mapping = {
        accession: _parse_instant(accepted)
        for accession, accepted in zip(
            recent["accessionNumber"], recent["acceptanceDateTime"], strict=True
        )
        if accepted
    }

    for page in data["filings"].get("files", []):
        older = _get(SUBMISSIONS_PAGE_URL.format(name=page["name"])).json()
        mapping.update(
            {
                accession: _parse_instant(accepted)
                for accession, accepted in zip(
                    older["accessionNumber"], older["acceptanceDateTime"], strict=True
                )
                if accepted
            }
        )
    return mapping


def parse_companyfacts(
    payload: dict[str, Any], accepted: dict[str, datetime]
) -> tuple[list[Fact], int]:
    """Flatten companyfacts into rows. Returns the facts and the count dropped as undatable."""
    cik = str(int(payload["cik"]))
    facts: list[Fact] = []
    undatable = 0

    for taxonomy, tags in payload["facts"].items():
        for tag, body in tags.items():
            for unit, rows in body["units"].items():
                for row in rows:
                    accession = row.get("accn")
                    accepted_at = accepted.get(accession) if accession else None
                    if accepted_at is None:
                        undatable += 1
                        continue
                    facts.append(
                        Fact(
                            cik=cik,
                            taxonomy=taxonomy,
                            tag=tag,
                            unit=unit,
                            value=row["val"],
                            fy=row.get("fy"),
                            fp=row.get("fp"),
                            period_start=_parse_date(row.get("start")),
                            period_end=_parse_date(row.get("end")),
                            form=row.get("form"),
                            accession=accession,
                            accepted_at=accepted_at,
                            frame=row.get("frame"),
                        )
                    )
    return facts, undatable


def fetch_facts(cik: int) -> tuple[list[Fact], int]:
    accepted = acceptance_map(cik)
    payload = _get(COMPANYFACTS_URL.format(cik=cik), timeout=120.0).json()
    return parse_companyfacts(payload, accepted)


# ---------------------------------------------------------------- persistence

INSERT_FACT = """
INSERT INTO xbrl_facts
    (cik, taxonomy, tag, unit, value, scale, fy, fp, period_start, period_end,
     form, accession, accepted_at, frame)
VALUES (%s, %s, %s, %s, %s, NULL, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (accession, taxonomy, tag, unit, period_start, period_end) DO NOTHING
"""


def store_facts(facts: list[Fact], batch: int = 1000) -> int:
    """Insert facts, skipping duplicates. Idempotent, so re-ingest is safe."""
    rows = [
        (
            f.cik,
            f.taxonomy,
            f.tag,
            f.unit,
            f.value,
            f.fy,
            f.fp,
            f.period_start,
            f.period_end,
            f.form,
            f.accession,
            f.accepted_at,
            f.frame,
        )
        for f in facts
    ]
    written = 0
    for start in range(0, len(rows), batch):
        written += pg.execute_many(INSERT_FACT, rows[start : start + batch])
    return written


def lookup(
    cik: str,
    tag: str,
    as_of: object,
    *,
    period_end: date | None = None,
    unit: str = "USD",
) -> list[dict]:
    """Facts for a tag, restricted to what was knowable at `as_of`.

    `as_of` is an `AsOf`; the import is local to avoid a cycle with the tools layer.
    Ordered so the most recently accepted statement of a period wins, which is what makes
    restatements resolve correctly rather than arbitrarily.
    """
    from financevault.env.asof import AsOf  # noqa: PLC0415 - local to break an import cycle

    assert isinstance(as_of, AsOf), "lookup requires an explicit AsOf horizon"

    sql = f"""
        SELECT tag, unit, value, fy, fp, period_start, period_end, form, accession,
               accepted_at, frame
        FROM xbrl_facts
        WHERE cik = %(cik)s AND tag = %(tag)s AND unit = %(unit)s
          AND {as_of.sql("xbrl_facts")}
          {"AND period_end = %(period_end)s" if period_end else ""}
        ORDER BY period_end DESC, accepted_at DESC
        LIMIT 50
    """
    params: dict[str, Any] = {"cik": cik, "tag": tag, "unit": unit, **as_of.params()}
    if period_end:
        params["period_end"] = period_end
    return as_of.check(pg.fetch_all(sql, params))
