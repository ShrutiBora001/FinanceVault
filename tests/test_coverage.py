"""Corpus coverage at a horizon.

The two halves of the environment have very different reach: XBRL facts go back to 2009,
filing documents only one year. At an older horizon the document corpus is legitimately empty,
and silence is indistinguishable from a bad query — so the agent retries into a table that
will never have rows. On the MVP2.2 dev split that was 9 of 30 questions, and those 9 carried
nearly every aborted run at roughly double the step count.

These tests pin that the environment says so rather than returning nothing.
"""

from __future__ import annotations

import pytest

from financevault.env import coverage
from financevault.env.asof import AsOf
from financevault.store import pg

# Before any filing was accepted (corpus starts 2025-08-04) but well after XBRL facts begin.
OLD = AsOf.parse("2012-01-01")
RECENT = AsOf.parse("2026-08-01")
CIK_AAPL = "320193"


@pytest.fixture(scope="module")
def seeded() -> bool:
    if pg.missing_tables():
        pytest.skip("schema not applied; run `make migrate`")
    if not pg.fetch_value("SELECT count(*) FROM filings"):
        pytest.skip("no corpus ingested; run `make ingest`")
    return True


def test_an_old_horizon_sees_no_documents(seeded: bool) -> None:
    assert coverage.at(OLD).documents_empty is True


def test_an_old_horizon_still_sees_facts(seeded: bool) -> None:
    """The asymmetry is the whole point: facts reach back, documents do not."""
    assert coverage.at(OLD).facts_visible > 0


def test_a_recent_horizon_sees_documents(seeded: bool) -> None:
    cover = coverage.at(RECENT)
    assert cover.documents_empty is False
    assert cover.documents_visible > 0


def test_advice_is_silent_when_documents_exist(seeded: bool) -> None:
    """A note on every run would be noise, and noise in a prompt costs tokens and attention."""
    assert coverage.at(RECENT).advice() == ""


def test_advice_names_the_window_and_redirects(seeded: bool) -> None:
    advice = coverage.at(OLD).advice()
    assert advice, "an empty corpus must be stated, not implied by silence"
    assert "retrieve_filings" in advice, "it must name the tool that will not work"
    assert "v_xbrl_facts" in advice or "lookup_fact" in advice, "and where to go instead"


def test_coverage_can_be_scoped_to_one_company(seeded: bool) -> None:
    scoped = coverage.at(RECENT, cik=CIK_AAPL)
    everyone = coverage.at(RECENT)
    assert 0 < scoped.documents_visible <= everyone.documents_visible


def test_retrieval_at_an_empty_horizon_explains_itself(seeded: bool) -> None:
    """Regression: this returned an empty list, and the agent retried until it ran out."""
    from financevault.runtime.budget import Ledger
    from financevault.tools import REGISTRY, ToolContext

    ctx = ToolContext(
        as_of=OLD,
        ledger=Ledger(max_usd=1.0, max_steps=40),
        cik=CIK_AAPL,
        ticker="AAPL",
    )
    result = REGISTRY.dispatch("retrieve_filings", {"query": "supply chain risk", "k": 5}, ctx)
    assert result.ok is False
    assert "corpus covers" in (result.error or "")
