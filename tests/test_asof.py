"""The leakage guarantee.

The invariant under test: no query carrying an as-of horizon may ever return a row that
became public after it. These tests run against the live database, because the guarantee is
about SQL behaviour, and a mocked store would test the mock.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from financevault.env import xbrl
from financevault.env.asof import AsOf, LookaheadError
from financevault.store import pg

CIK_AAPL = "320193"


@pytest.fixture(scope="module")
def seeded() -> bool:
    if pg.missing_tables():
        pytest.skip("schema not applied; run `make migrate`")
    if (pg.fetch_value("SELECT count(*) FROM xbrl_facts") or 0) == 0:
        pytest.skip("no facts ingested; run `make ingest`")
    return True


# ---------------------------------------------------------------- construction


def test_naive_datetime_is_rejected() -> None:
    """A naive datetime silently assumes local time, shifting the horizon by hours."""
    with pytest.raises(ValueError, match="timezone-aware"):
        AsOf(datetime(2025, 1, 1))  # noqa: DTZ001 - deliberately naive


def test_bare_date_becomes_utc_midnight() -> None:
    assert AsOf.parse("2025-01-01").instant == datetime(2025, 1, 1, tzinfo=UTC)


def test_z_suffix_parses() -> None:
    assert AsOf.parse("2025-10-31T10:01:26Z").instant == datetime(
        2025, 10, 31, 10, 1, 26, tzinfo=UTC
    )


def test_unregistered_table_raises() -> None:
    """A new table must declare how it is filtered rather than defaulting to unfiltered."""
    with pytest.raises(KeyError, match="point-in-time column"):
        AsOf.now().sql("some_new_table")


# ---------------------------------------------------------------- the invariant


def test_no_fact_after_horizon_is_returned(seeded: bool) -> None:
    """The core guarantee, swept across the full history of the corpus."""
    span = pg.fetch_one("SELECT min(accepted_at) lo, max(accepted_at) hi FROM xbrl_facts")
    assert span and span["lo"] and span["hi"]

    # Ten horizons spread across the corpus, so the filter is exercised at points where
    # some facts are visible and others are not.
    total = (span["hi"] - span["lo"]).total_seconds()
    for i in range(11):
        horizon = AsOf(span["lo"] + timedelta(seconds=total * i / 10))
        rows = pg.fetch_all(
            f"SELECT accepted_at FROM xbrl_facts WHERE {horizon.sql('xbrl_facts')}",
            horizon.params(),
        )
        assert all(r["accepted_at"] <= horizon.instant for r in rows), f"leak at horizon {horizon}"


def test_horizon_before_corpus_returns_nothing(seeded: bool) -> None:
    horizon = AsOf.parse("1990-01-01")
    rows = pg.fetch_all(
        f"SELECT 1 FROM xbrl_facts WHERE {horizon.sql('xbrl_facts')} LIMIT 1", horizon.params()
    )
    assert rows == []


def test_lookup_excludes_a_fact_published_one_second_later(seeded: bool) -> None:
    """The boundary case: acceptance is an instant, and the filter is inclusive of it."""
    fact = pg.fetch_one(
        """
        SELECT tag, unit, accepted_at FROM xbrl_facts
        WHERE cik = %s AND unit = 'USD' ORDER BY accepted_at DESC LIMIT 1
        """,
        (CIK_AAPL,),
    )
    assert fact

    at = AsOf(fact["accepted_at"])
    just_before = AsOf(fact["accepted_at"] - timedelta(seconds=1))

    visible = xbrl.lookup(CIK_AAPL, fact["tag"], at, unit=fact["unit"])
    hidden = xbrl.lookup(CIK_AAPL, fact["tag"], just_before, unit=fact["unit"])

    assert any(r["accepted_at"] == fact["accepted_at"] for r in visible), (
        "a fact must be visible at exactly its acceptance instant"
    )
    assert all(r["accepted_at"] < fact["accepted_at"] for r in hidden), (
        "a fact must be invisible one second before acceptance"
    )


def test_period_end_is_not_a_safe_horizon(seeded: bool) -> None:
    """Why `accepted_at` exists: filings are accepted well after the period they describe.

    Filtering on `period_end` would admit documents weeks before they were public. This
    asserts the gap is real in the ingested data, so the whole design has a reason.
    """
    row = pg.fetch_one(
        """
        SELECT period_end, accepted_at, accepted_at::date - period_end AS lag_days
        FROM filings WHERE form = '10-K' ORDER BY accepted_at DESC LIMIT 1
        """
    )
    assert row and row["lag_days"] > 0, "expected acceptance to postdate the reporting period"


# ---------------------------------------------------------------- defence in depth


def test_check_raises_on_a_row_past_the_horizon() -> None:
    horizon = AsOf.parse("2025-01-01")
    leaked = [{"accepted_at": datetime(2025, 6, 1, tzinfo=UTC)}]
    with pytest.raises(LookaheadError, match="exceeds as_of"):
        horizon.check(leaked)


def test_check_passes_rows_through_unchanged() -> None:
    horizon = AsOf.parse("2025-01-01")
    rows = [{"accepted_at": datetime(2024, 6, 1, tzinfo=UTC), "value": 1}]
    assert horizon.check(rows) is rows
