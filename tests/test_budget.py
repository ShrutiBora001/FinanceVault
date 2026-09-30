"""The budget ledger must stop a run, not warn it."""

from __future__ import annotations

import pytest

from financevault.runtime.budget import PRICES, BudgetExceeded, Ledger, price


def test_price_matches_published_rates() -> None:
    # 1M in + 1M out on Haiku 4.5 at $1/$5 per million.
    assert price("claude-haiku-4-5-20251001", 1_000_000, 1_000_000) == pytest.approx(6.00)
    assert price("claude-sonnet-5", 1_000_000, 0) == pytest.approx(3.00)


def test_local_models_are_free() -> None:
    assert price("qwen3-8b", 5_000_000, 5_000_000) == 0.0


def test_unknown_remote_model_raises_rather_than_costing_zero() -> None:
    """A silent zero would understate cost per answer, the project's headline metric."""
    with pytest.raises(KeyError, match="no price"):
        price("some-new-model", 1000, 1000)


def test_charge_accumulates() -> None:
    ledger = Ledger(max_usd=1.0, max_steps=10)
    ledger.charge("claude-haiku-4-5-20251001", 100_000, 10_000)
    assert ledger.usd == pytest.approx(0.15)
    assert ledger.tokens_in == 100_000
    assert ledger.tokens_out == 10_000


def test_charge_aborts_on_dollar_ceiling() -> None:
    ledger = Ledger(max_usd=0.05, max_steps=10)
    with pytest.raises(BudgetExceeded) as exc:
        ledger.charge("claude-sonnet-5", 1_000_000, 0)  # $3.00, far past the ceiling
    assert exc.value.resource == "usd"
    assert exc.value.limit == 0.05


def test_step_aborts_on_step_ceiling() -> None:
    ledger = Ledger(max_usd=10.0, max_steps=3)
    for _ in range(3):
        ledger.step()
    with pytest.raises(BudgetExceeded) as exc:
        ledger.step()
    assert exc.value.resource == "steps"


def test_would_exceed_predicts_without_charging() -> None:
    ledger = Ledger(max_usd=0.10, max_steps=10)
    ledger.charge("claude-haiku-4-5-20251001", 50_000, 5_000)  # $0.075
    assert ledger.would_exceed(0.05) is True
    assert ledger.would_exceed(0.01) is False
    assert ledger.usd == pytest.approx(0.075), "prediction must not mutate the ledger"


def test_cached_calls_cost_nothing_but_still_count_tokens() -> None:
    """The replay guarantee, at ledger level: journal hits are free and say so."""
    ledger = Ledger(max_usd=100.0, max_steps=10)
    ledger.charge("claude-sonnet-5", 1_000_000, 1_000_000, cached=True)
    assert ledger.usd == 0.0
    assert ledger.tokens_in == 1_000_000
    assert ledger.summary()["cache_hit_rate"] == 1.0


def test_a_cached_call_still_counts_against_the_dollar_ceiling() -> None:
    """Regression: enforcing on actual spend made the ceiling vanish on replay.

    A journal hit costs nothing, so a warm run used to sail past a limit the cold run had
    aborted on, and the two produced different trajectories for the same question. The budget
    is a property of the policy, so it is enforced on list cost.
    """
    ledger = Ledger(max_usd=0.001, max_steps=10)
    with pytest.raises(BudgetExceeded):
        ledger.charge("claude-sonnet-5", 1_000_000, 1_000_000, cached=True)
    assert ledger.usd == 0.0, "it must still be recorded as free"
    assert ledger.list_usd > 0.001


def test_a_run_aborts_at_the_same_point_cold_or_warm() -> None:
    """The property that matters: cache state must not change the trajectory."""

    def spend(cached: bool) -> int:
        ledger = Ledger(max_usd=0.05, max_steps=100)
        calls = 0
        while True:
            try:
                ledger.charge("claude-sonnet-5", 5_000, 500, cached=cached)
            except BudgetExceeded:
                return calls
            calls += 1

    assert spend(cached=True) == spend(cached=False)


def test_would_exceed_uses_list_cost_so_it_agrees_with_the_ceiling() -> None:
    ledger = Ledger(max_usd=0.10, max_steps=10)
    ledger.charge("claude-haiku-4-5-20251001", 50_000, 5_000, cached=True)
    assert ledger.usd == 0.0
    assert ledger.would_exceed(0.05) is True, "a free call still consumed the budget"


def test_summary_reports_hit_rate() -> None:
    ledger = Ledger(max_usd=1.0, max_steps=10)
    ledger.charge("claude-haiku-4-5-20251001", 1000, 100, cached=True)
    ledger.charge("claude-haiku-4-5-20251001", 1000, 100, cached=False)
    summary = ledger.summary()
    assert summary["calls"] == 2
    assert summary["cached_calls"] == 1
    assert summary["cache_hit_rate"] == 0.5


def test_every_priced_model_has_two_rates() -> None:
    for model, rates in PRICES.items():
        assert len(rates) == 2, model
        assert all(r > 0 for r in rates), model
