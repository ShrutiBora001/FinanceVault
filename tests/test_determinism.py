"""Tool output must be byte-stable across identical calls.

This is the precondition for replay, and it is easy to lose without noticing. A trajectory is
a chain: a tool observation becomes part of the next prompt, which becomes the next journal
key. So an unordered `SELECT` deep in a tool does not produce a slightly different result — it
produces a *different trajectory*, and the replay that was supposed to cost nothing goes and
asks the model again.

The bug these tests pin cost a real sweep: `ORDER BY similarity(...) DESC` with no tie-break,
and a `DISTINCT ... LIMIT` with no ordering at all. Both returned rows in whatever order the
planner picked, which was stable enough to look fine and unstable enough to break replay.
"""

from __future__ import annotations

import pytest

from financevault.env import rerank
from financevault.env.asof import AsOf
from financevault.runtime.budget import Ledger
from financevault.store import pg
from financevault.tools import REGISTRY, ToolContext

CIK_AAPL = "320193"
REPEATS = 4

STABLE_SQL = "SELECT tag, value FROM v_xbrl_facts WHERE tag = 'Assets' ORDER BY value LIMIT 5"


@pytest.fixture(scope="module")
def seeded() -> bool:
    if pg.missing_tables():
        pytest.skip("schema not applied; run `make migrate`")
    if (pg.fetch_value("SELECT count(*) FROM chunks") or 0) == 0:
        pytest.skip("no corpus ingested; run `make ingest`")
    return True


@pytest.fixture
def ctx() -> ToolContext:
    return ToolContext(
        as_of=AsOf.parse("2026-08-01"),
        ledger=Ledger(max_usd=1.0, max_steps=40),
        cik=CIK_AAPL,
        ticker="AAPL",
    )


def _repeat(tool: str, args: dict, ctx: ToolContext, extract) -> set:
    return {extract(REGISTRY.dispatch(tool, args, ctx)) for _ in range(REPEATS)}


# ---------------------------------------------------------------- retrieval


def test_retrieval_returns_the_same_chunks_every_time(seeded: bool, ctx: ToolContext) -> None:
    results = _repeat(
        "retrieve_filings",
        {"query": "supply chain risk", "k": 5},
        ctx,
        lambda r: tuple(d["chunk_id"] for d in r.data),
    )
    assert len(results) == 1, "retrieval order drifted between identical calls"


def test_retrieval_ties_are_broken_deterministically(seeded: bool, ctx: ToolContext) -> None:
    """Reciprocal-rank fusion produces exact ties routinely, so the tie-break carries weight."""
    results = _repeat(
        "retrieve_filings",
        {"query": "the", "k": 8},
        ctx,
        lambda r: tuple(d["chunk_id"] for d in r.data) if r.ok else ("error",),
    )
    assert len(results) == 1


def test_section_filtered_retrieval_is_stable(seeded: bool, ctx: ToolContext) -> None:
    results = _repeat(
        "retrieve_filings",
        {"query": "competition", "k": 4, "section": "risk factors"},
        ctx,
        lambda r: tuple(d["chunk_id"] for d in r.data) if r.ok else ("error",),
    )
    assert len(results) == 1


# ---------------------------------------------------------------- error messages
#
# Error text reaches the model as an observation, so an unstable message is an unstable
# prompt. These are as load-bearing as the successful paths.


def test_a_tag_suggestion_error_is_byte_stable(seeded: bool, ctx: ToolContext) -> None:
    """Regression: `ORDER BY similarity(...) DESC` with no tie-break."""
    results = _repeat("lookup_fact", {"tag": "NetIncome"}, ctx, lambda r: r.error)
    assert len(results) == 1


def test_an_unknown_section_error_is_byte_stable(seeded: bool, ctx: ToolContext) -> None:
    """Regression: `SELECT DISTINCT ... LIMIT` with no ordering at all."""
    results = _repeat(
        "retrieve_filings",
        {"query": "anything", "k": 3, "section": "no such section"},
        ctx,
        lambda r: r.error,
    )
    assert len(results) == 1


# ---------------------------------------------------------------- lookups


def test_fact_lookup_is_stable(seeded: bool, ctx: ToolContext) -> None:
    results = _repeat(
        "lookup_fact",
        {"tag": "NetIncomeLoss"},
        ctx,
        lambda r: tuple((d["period_end"], d["value"]) for d in r.data),
    )
    assert len(results) == 1


def test_sql_results_are_stable(seeded: bool, ctx: ToolContext) -> None:
    results = _repeat(
        "sql",
        {"query": STABLE_SQL},
        ctx,
        lambda r: tuple(tuple(sorted(row.items())) for row in r.data["rows"]),
    )
    assert len(results) == 1


def test_the_sandbox_is_deterministic(ctx: ToolContext) -> None:
    results = _repeat(
        "python", {"code": "result = sum(range(100)) / 7"}, ctx, lambda r: r.data["result"]
    )
    assert len(results) == 1


# ---------------------------------------------------------------- reranking


def test_reranking_is_stable_across_calls() -> None:
    rows = [{"chunk_id": i, "text": f"Net income figure number {i}."} for i in range(6)]
    orders = {
        tuple(d["chunk_id"] for d in rerank.rerank("net income", rows, top_k=4, drop_weak=False))
        for _ in range(3)
    }
    assert len(orders) == 1


def test_reranking_ties_break_on_index() -> None:
    """Identical passages score identically; without a tie-break the order is arbitrary."""
    rows = [{"chunk_id": i, "text": "Exactly the same sentence."} for i in range(5)]
    out = rerank.rerank("anything", rows, top_k=5, drop_weak=False)
    assert [d["chunk_id"] for d in out] == [0, 1, 2, 3, 4]
