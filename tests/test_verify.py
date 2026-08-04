"""The two programmatic verifier signals.

`s1` and `s3` are the signals a judge model never touches, so their behaviour has to be
pinned by tests. If they are wrong, step filtering is wrong, and every claim after it rests
on nothing.
"""

from __future__ import annotations

import pytest

from financevault.env.asof import AsOf
from financevault.store import pg
from financevault.verify import numeric, tool_validity
from financevault.verify.numeric import Evidence

CIK_AAPL = "320193"
HORIZON = AsOf.parse("2026-08-01")

# FY2025, from the 10-K accepted 2025-10-31.
NET_INCOME = 112_010_000_000.0
REVENUE = 416_161_000_000.0


@pytest.fixture(scope="module")
def seeded() -> bool:
    if pg.missing_tables():
        pytest.skip("schema not applied; run `make migrate`")
    if (pg.fetch_value("SELECT count(*) FROM xbrl_facts") or 0) == 0:
        pytest.skip("no facts ingested; run `make ingest`")
    return True


@pytest.fixture
def evidence() -> Evidence:
    """What a trajectory would have retrieved via lookup_fact."""
    ev = Evidence()
    ev.add(
        NET_INCOME, source="lookup_fact", tag="NetIncomeLoss", period_end="2025-09-27", unit="USD"
    )
    ev.add(REVENUE, source="lookup_fact", tag="Revenues", period_end="2025-09-27", unit="USD")
    return ev


# ---------------------------------------------------------------- s1


def test_s1_rewards_a_clean_call() -> None:
    assert tool_validity.score("lookup_fact", {"tag": "NetIncomeLoss"}, {"ok": True}).score == 1.0


def test_s1_rejects_a_hallucinated_tool() -> None:
    s = tool_validity.score("fetch_stock_price", {}, {"ok": True})
    assert s.score == 0.0
    assert "hallucinated" in s.reason


def test_s1_penalises_a_tool_outside_the_path() -> None:
    s = tool_validity.score(
        "sql", {"query": "SELECT 1"}, {"ok": True}, allowed=["lookup_fact", "finish"]
    )
    assert s.score == 0.25


def test_s1_penalises_malformed_arguments() -> None:
    s = tool_validity.score("lookup_fact", {"wrong_field": 1}, {"ok": True})
    assert s.score == 0.25
    assert "validation" in s.reason


def test_s1_gives_partial_credit_for_a_valid_call_that_errored() -> None:
    """Asking a well-formed question and getting 'no such tag' is a reasonable search step."""
    s = tool_validity.score(
        "lookup_fact", {"tag": "MadeUpTag"}, {"ok": False, "error": "no fact for tag"}
    )
    assert s.score == 0.5


def test_s1_penalises_prose_instead_of_action() -> None:
    assert tool_validity.score(None, {}, {}).score == 0.0


# ---------------------------------------------------------------- claim extraction


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Revenue was $416,161 million", 416_161_000_000.0),
        ("Revenue was $416.16 billion", 416_160_000_000.0),
        ("Net income of 112010000000", 112_010_000_000.0),
        ("It fell to $1.5 trillion", 1_500_000_000_000.0),
        ("A loss of $(2,500) thousand", -2_500_000.0),
        ("Cash of $30bn", 30_000_000_000.0),
    ],
)
def test_scale_words_resolve_into_the_value(text: str, expected: float) -> None:
    claims = numeric.extract_claims(text)
    assert claims and claims[0].value == pytest.approx(expected)


def test_percentages_are_flagged_not_treated_as_amounts() -> None:
    assert numeric.extract_claims("Margin improved to 46.2%")[0].is_percent is True


@pytest.mark.parametrize(
    "text",
    [
        "fiscal year 2025",
        "period ended September 27, 2025",
        "the quarter ending 2025-09-27",
        "for FY 2024",
    ],
)
def test_dates_are_not_financial_claims(text: str) -> None:
    """Answers state periods alongside figures; counting dates rejects correct answers."""
    assert all(not c.is_financial for c in numeric.extract_claims(text)), text


@pytest.mark.parametrize(
    "text",
    ["$112.010 billion", "$416,161 million", "112010000000", "revenue of $30bn"],
)
def test_monetary_amounts_are_financial_claims(text: str) -> None:
    claims = numeric.extract_claims(text)
    assert claims and any(c.is_financial for c in claims), text


def test_a_real_answer_scores_full_marks_despite_its_dates(evidence: Evidence) -> None:
    """Regression: this exact phrasing scored 0.25 before dates were excluded."""
    answer = (
        "Apple's net income for fiscal year 2025 (period ended September 27, 2025) "
        f"was ${NET_INCOME:,.0f}."
    )
    s = numeric.score(answer, evidence=evidence)
    assert s.score == 1.0, s.reason


def test_multiple_claims_are_all_extracted() -> None:
    assert len(numeric.extract_claims("Revenue $416.16 billion, up from $391.04 billion")) == 2


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Paren outside the dollar: a restatement, still positive.
        ("($112.01 billion)", 112_010_000_000.0),
        ("Revenue ($416.16 billion) grew", 416_160_000_000.0),
        # Dollar outside the paren, or no dollar at all: an accounting negative.
        ("$(2,500) thousand", -2_500_000.0),
        ("a loss of (2,500)", -2_500.0),
    ],
)
def test_parenthetical_restatements_are_not_accounting_negatives(
    text: str, expected: float
) -> None:
    """Regression: `($112.01 billion)` parsed as *negative* $112B on a correct answer."""
    claims = [c for c in numeric.extract_claims(text) if c.is_financial]
    assert claims and claims[0].value == pytest.approx(expected), text


def test_a_restated_figure_scores_full_marks(evidence: Evidence) -> None:
    """The exact phrasing that scored 0.67: the same figure written twice, two ways."""
    answer = f"Apple's net income was ${NET_INCOME:,.0f} (${NET_INCOME / 1e9:.2f} billion)."
    s = numeric.score(answer, evidence=evidence)
    assert s.score == 1.0, s.reason


# ---------------------------------------------------------------- evidence collection


def test_evidence_is_collected_from_tool_observations() -> None:
    steps = [
        {
            "tool": "lookup_fact",
            "obs": {
                "ok": True,
                "data": [{"tag": "NetIncomeLoss", "value": NET_INCOME, "period_end": "2025-09-27"}],
            },
        },
        {"tool": "finish", "obs": {"ok": True, "data": {"answer": "..."}}},
    ]
    ev = numeric.collect_evidence(steps)
    assert len(ev) == 1
    assert ev.values[0]["tag"] == "NetIncomeLoss"


def test_failed_tool_calls_contribute_no_evidence() -> None:
    steps = [{"tool": "lookup_fact", "obs": {"ok": False, "error": "no such tag"}}]
    assert len(numeric.collect_evidence(steps)) == 0


# ---------------------------------------------------------------- s3


def test_s3_verifies_a_figure_that_came_from_evidence(evidence: Evidence) -> None:
    s = numeric.score(f"Net income was ${NET_INCOME:,.0f}.", evidence=evidence)
    assert s.score == 1.0, s.reason


def test_s3_accepts_a_rounded_restatement(evidence: Evidence) -> None:
    """Answers restate rounded figures; exact equality would fail correct answers."""
    s = numeric.score("Net income was about $112.01 billion.", evidence=evidence)
    assert s.score == 1.0, s.reason


def test_s3_catches_an_off_by_one_thousand_error(evidence: Evidence) -> None:
    """The headline silent error: right digits, wrong magnitude."""
    s = numeric.score(f"Net income was ${NET_INCOME / 1000:,.0f}.", evidence=evidence)
    assert s.score == 0.0
    assert "off by" in s.reason


def test_s3_catches_a_fabricated_figure(seeded: bool, evidence: Evidence) -> None:
    """A value matching nothing retrieved and nothing on record."""
    s = numeric.score(
        "Net income was $12,345,678,901,234.", evidence=evidence, cik=CIK_AAPL, as_of=HORIZON
    )
    assert s.score == 0.0
    assert "matches nothing" in s.reason


def test_a_random_number_collides_with_the_fact_universe(seeded: bool, evidence: Evidence) -> None:
    """Why claims are matched against evidence rather than against all facts.

    $77,777,777,777 was invented at random and still lands within tolerance of a real AAPL
    fact. Under a fact-universe check it would have scored as verified. Under evidence
    matching it scores 0.0, and is merely labelled `uncited` rather than `fabricated`.
    """
    s = numeric.score(
        "Net income was $77,777,777,777.", evidence=evidence, cik=CIK_AAPL, as_of=HORIZON
    )
    assert s.score == 0.0
    assert "never retrieved" in s.reason


def test_s3_flags_a_correct_figure_the_agent_never_retrieved(seeded: bool) -> None:
    """Guessing right from memory is not knowing, and is not reproducible."""
    empty_but_plausible = Evidence()
    empty_but_plausible.add(1.0, source="lookup_fact", tag="Unrelated")
    s = numeric.score(
        f"Net income was ${NET_INCOME:,.0f}.",
        evidence=empty_but_plausible,
        cik=CIK_AAPL,
        as_of=HORIZON,
    )
    assert s.score == 0.0
    assert "never retrieved" in s.reason


def test_s3_scores_zero_when_the_trajectory_retrieved_nothing() -> None:
    s = numeric.score(f"Net income was ${NET_INCOME:,.0f}.", evidence=Evidence())
    assert s.score == 0.0
    assert "no evidence" in s.reason


def test_s3_is_not_applicable_without_claims(evidence: Evidence) -> None:
    s = numeric.score("Apple discussed supply chain risks.", evidence=evidence)
    assert s.score == 1.0
    assert s.reason.startswith("n/a")


def test_s3_checks_the_structured_value_too(evidence: Evidence) -> None:
    s = numeric.score("See below.", evidence=evidence, stated_value=NET_INCOME)
    assert s.score == 1.0, s.reason


def test_s3_partial_credit_when_some_claims_verify(evidence: Evidence) -> None:
    s = numeric.score(
        f"Revenue was ${REVENUE:,.0f} and net income was $99,999,999,999.",
        evidence=evidence,
        cik=CIK_AAPL,
        as_of=HORIZON,
    )
    assert s.score == pytest.approx(0.5)


def test_s3_catches_the_wrong_period(evidence: Evidence) -> None:
    s = numeric.score(
        f"Q1 net income was ${NET_INCOME:,.0f}.", evidence=evidence, period_end="2025-12-27"
    )
    assert s.score == 0.0
    assert "period" in s.reason


# ---------------------------------------------------------------- correctness accounting


def test_na_is_not_evidence_of_correctness() -> None:
    """`n/a` scores 1.0 so a step is not penalised, which is not the same as passing."""
    from financevault.verify.signals import Signal, na

    assert na("does not apply").passed is True
    assert na("does not apply").verified is False
    assert Signal(1.0, "real pass").verified is True


def test_a_run_that_never_finished_did_not_answer_correctly() -> None:
    """Regression: this counted as correct and inflated a reported accuracy 85% -> 100%.

    A run ending in prose has no `finish` step, so s5 is n/a and scores 1.0. Reading the last
    verdict's score directly treated "no answer" as "right answer".
    """
    from financevault.verify import StepVerdict, answered_correctly, na
    from financevault.verify.signals import Signal

    unfinished = [
        StepVerdict(idx=0, tool="lookup_fact", signals={"s5": na("not the terminal step")}),
        StepVerdict(idx=1, tool=None, signals={"s5": na("not the terminal step")}),
    ]
    assert answered_correctly(unfinished) is False

    finished = [
        StepVerdict(idx=0, tool="lookup_fact", signals={"s5": na("not the terminal step")}),
        StepVerdict(idx=1, tool="finish", signals={"s5": Signal(1.0, "matches expected")}),
    ]
    assert answered_correctly(finished) is True


def test_a_finished_run_with_no_ground_truth_is_not_correct() -> None:
    """Unknown is not the same as right."""
    from financevault.verify import StepVerdict, answered_correctly, na

    verdicts = [StepVerdict(idx=0, tool="finish", signals={"s5": na("no ground truth")})]
    assert answered_correctly(verdicts) is False
