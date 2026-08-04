"""The frozen evaluation split.

A defect here does not look like a bug. It looks like a *result*: every policy scored 0% on
ratio questions, which reads as "the agents cannot compute margins" and was in fact a
generator pairing a full-year numerator with a single-quarter denominator. Bad ground truth
produces confident, wrong conclusions about capability, and nothing downstream can detect it.

These tests run against the committed split file, so they check the artefact the benchmark
actually uses rather than what the generator would produce today.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

SPLIT = Path("eval/splits/mvp_150.jsonl")
ARCHETYPES = {"lookup", "delta", "ratio", "cross_company"}


@pytest.fixture(scope="module")
def questions() -> list[dict]:
    if not SPLIT.exists():
        pytest.skip(f"{SPLIT} missing; run `make split`")
    return [json.loads(line) for line in SPLIT.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------- ground truth


def test_ratio_answers_are_plausible_margins(questions: list[dict]) -> None:
    """Regression: FY numerator over Q4 denominator gave Apple a 94.64% net margin.

    A margin outside roughly -50% to 90% is not impossible in the real world, but on this
    corpus of large profitable filers it means the two legs did not cover the same period.
    """
    ratios = [q for q in questions if q["archetype"] == "ratio"]
    assert ratios, "the split has no ratio questions at all"
    for q in ratios:
        assert -50 < q["expected_value"] < 90, f"{q['id']}: {q['expected_value']}%"


def test_ratio_questions_name_two_source_tags(questions: list[dict]) -> None:
    for q in questions:
        if q["archetype"] == "ratio":
            assert len(q["source_tags"]) == 2, q["id"]


def test_every_question_has_ground_truth(questions: list[dict]) -> None:
    for q in questions:
        assert q.get("expected_value") is not None, q["id"]
        assert q.get("expected_unit"), q["id"]


def test_deltas_are_not_trivially_zero(questions: list[dict]) -> None:
    """A zero delta usually means the same fact was differenced against itself."""
    deltas = [q for q in questions if q["archetype"] == "delta"]
    assert deltas
    assert sum(1 for q in deltas if q["expected_value"] == 0) == 0


def test_cross_company_differences_are_positive(questions: list[dict]) -> None:
    """The question asks for the difference as a positive number."""
    for q in questions:
        if q["archetype"] == "cross_company":
            assert q["expected_value"] >= 0, q["id"]
            assert q["ticker"] != q["second_ticker"], q["id"]


# ---------------------------------------------------------------- horizons


def test_every_answer_is_knowable_at_its_horizon(questions: list[dict]) -> None:
    """A question whose answer postdates its as-of measures the filter, not the agent."""
    from financevault.env.asof import AsOf
    from financevault.store import pg

    if pg.missing_tables() or not pg.fetch_value("SELECT count(*) FROM xbrl_facts"):
        pytest.skip("no corpus ingested")

    for q in questions:
        if q["archetype"] != "lookup":
            continue
        horizon = AsOf.parse(q["as_of"])
        visible = pg.fetch_value(
            f"""
            SELECT count(*) FROM xbrl_facts
            WHERE cik = %(cik)s AND tag = %(tag)s AND period_end = %(period_end)s
              AND {horizon.sql("xbrl_facts")}
            """,
            {
                "cik": q["cik"],
                "tag": q["source_tags"][0],
                "period_end": q["period_end"],
                **horizon.params(),
            },
        )
        assert visible, f"{q['id']}: the answer is not visible at its own as_of"


# ---------------------------------------------------------------- split hygiene


def test_ids_are_unique(questions: list[dict]) -> None:
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))


def test_every_question_is_assigned_exactly_one_split(questions: list[dict]) -> None:
    for q in questions:
        assert q["split"] in {"train", "dev", "test"}, q["id"]


def test_splits_are_disjoint(questions: list[dict]) -> None:
    by_split: dict[str, set[str]] = {}
    for q in questions:
        by_split.setdefault(q["split"], set()).add(q["id"])
    assert not (by_split["train"] & by_split["dev"])
    assert not (by_split["train"] & by_split["test"])
    assert not (by_split["dev"] & by_split["test"])


def test_every_split_contains_every_archetype(questions: list[dict]) -> None:
    """Regression: a global hash left `cross_company` out of test entirely.

    An archetype missing from test cannot be measured there, and the omission is silent.
    """
    for split in ("train", "dev", "test"):
        present = {q["archetype"] for q in questions if q["split"] == split}
        assert present == ARCHETYPES, f"{split} is missing {ARCHETYPES - present}"


def test_split_assignment_is_stable_under_reordering() -> None:
    """Assignment must come from the id, not from position in the list."""
    from scripts.make_split import assign_splits

    rows = [{"id": f"q{i}", "archetype": "lookup"} for i in range(50)]
    assign_splits(rows)
    first = {r["id"]: r["split"] for r in rows}

    shuffled = list(reversed(rows))
    for row in shuffled:
        row.pop("split")
    assign_splits(shuffled)
    assert {r["id"]: r["split"] for r in shuffled} == first


def test_all_ten_companies_appear(questions: list[dict]) -> None:
    assert len({q["ticker"] for q in questions}) == 10


def test_no_archetype_dominates_a_split(questions: list[dict]) -> None:
    """A split that is 90% one archetype measures that archetype, not the system."""
    for split in ("train", "dev", "test"):
        counts = Counter(q["archetype"] for q in questions if q["split"] == split)
        total = sum(counts.values())
        assert max(counts.values()) / total < 0.75, f"{split}: {counts}"
