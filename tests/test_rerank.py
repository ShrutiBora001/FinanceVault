"""Cross-encoder reranking.

The behaviours worth pinning are the ones that fail quietly. A reranker that returns rows in
the order it received them looks identical to a working one until you inspect the scores, and
one that drops everything on a hard query looks like a retrieval failure.

The model is loaded once and cached, so these are slower than the rest of the suite but not
per-test slow.
"""

from __future__ import annotations

import pytest

from financevault.env import rerank

QUERY = "What was Apple's net income in fiscal year 2025?"

RELEVANT = "Net income was $112,010 million for fiscal 2025, up from $93,736 million."
TOPICAL = "Total net sales increased during 2025, driven by higher Services revenue."
IRRELEVANT = "The Company's headquarters are located in Cupertino, California."
ABSURD = "Photosynthesis converts light energy into chemical energy in plant cells."


def _rows(*texts: str) -> list[dict]:
    return [{"chunk_id": i, "text": t} for i, t in enumerate(texts)]


# ---------------------------------------------------------------- ordering


def test_the_relevant_passage_outranks_the_irrelevant_one() -> None:
    ranked = rerank.score(QUERY, [IRRELEVANT, RELEVANT])
    assert ranked[0].index == 1


def test_ordering_is_by_descending_score() -> None:
    ranked = rerank.score(QUERY, [IRRELEVANT, RELEVANT, TOPICAL])
    assert [s.score for s in ranked] == sorted((s.score for s in ranked), reverse=True)


def test_reranking_actually_reorders() -> None:
    """The failure that looks like success: returning input order with scores attached."""
    rows = _rows(ABSURD, IRRELEVANT, RELEVANT)
    out = rerank.rerank(QUERY, rows, top_k=3, drop_weak=False)
    assert out[0]["chunk_id"] == 2, "the relevant row should be promoted from last to first"


def test_a_topical_passage_ranks_between_relevant_and_absurd() -> None:
    ranked = {s.index: s.score for s in rerank.score(QUERY, [RELEVANT, TOPICAL, ABSURD])}
    assert ranked[0] > ranked[1] > ranked[2]


# ---------------------------------------------------------------- trimming


def test_top_k_is_respected() -> None:
    out = rerank.rerank(QUERY, _rows(RELEVANT, TOPICAL, IRRELEVANT, ABSURD), top_k=2)
    assert len(out) <= 2


def test_weak_candidates_are_dropped_rather_than_padding_the_result() -> None:
    """Fewer good passages beat more with noise: every extra one costs input tokens."""
    out = rerank.rerank(QUERY, _rows(RELEVANT, ABSURD), top_k=2, drop_weak=True)
    assert len(out) == 1
    assert out[0]["chunk_id"] == 0


def test_drop_weak_can_be_disabled() -> None:
    out = rerank.rerank(QUERY, _rows(RELEVANT, ABSURD), top_k=2, drop_weak=False)
    assert len(out) == 2


def test_never_returns_empty_when_given_candidates() -> None:
    """An empty result tells the agent less than a weak one, so the best always survives."""
    out = rerank.rerank("completely unrelated query about marine biology", _rows(ABSURD), top_k=3)
    assert len(out) == 1


# ---------------------------------------------------------------- shape and edges


def test_scores_are_attached_to_returned_rows() -> None:
    out = rerank.rerank(QUERY, _rows(RELEVANT, TOPICAL), top_k=2, drop_weak=False)
    assert all("rerank_score" in row for row in out)


def test_original_row_fields_survive() -> None:
    rows = [{"chunk_id": 7, "text": RELEVANT, "section": "financial statements"}]
    out = rerank.rerank(QUERY, rows, top_k=1)
    assert out[0]["section"] == "financial statements"
    assert out[0]["chunk_id"] == 7


def test_input_rows_are_not_mutated() -> None:
    rows = _rows(RELEVANT, TOPICAL)
    rerank.rerank(QUERY, rows, top_k=2, drop_weak=False)
    assert all("rerank_score" not in row for row in rows)


@pytest.mark.parametrize("rows", [[], None])
def test_empty_input_returns_empty(rows: list | None) -> None:
    assert rerank.rerank(QUERY, rows or [], top_k=3) == []


def test_scoring_an_empty_passage_list_returns_empty() -> None:
    assert rerank.score(QUERY, []) == []


def test_a_row_with_no_text_does_not_crash() -> None:
    out = rerank.rerank(QUERY, [{"chunk_id": 1}], top_k=1)
    assert len(out) == 1
