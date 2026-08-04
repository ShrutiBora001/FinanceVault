"""Metric aggregation in the eval harness.

Two metrics here are computed from the same runs but mean opposite things, and the harness has
already got one of them wrong once. E1 must use *list* cost, or replaying a journalled sweep
reports every policy as free. E2 must use *only uncached* runs, or the same replay reports
millisecond latencies for policies that take seconds.

The first dev sweep published a latency table comparing three fully-cached policies at 4-73ms
against one live policy at 17,333ms. Nothing errored; the column was simply measuring the
journal for three rows and the network for the fourth.
"""

from __future__ import annotations

from eval.harness import Outcome, summarise


def _outcome(
    *,
    correct: bool = True,
    latency_ms: int = 100,
    n_calls: int = 2,
    cached_calls: int = 0,
    cost_usd: float = 0.01,
    list_usd: float = 0.01,
    archetype: str = "lookup",
) -> Outcome:
    return Outcome(
        question_id="q",
        archetype=archetype,
        run_id="r",
        correct=correct,
        value=1.0,
        expected=1.0,
        cost_usd=cost_usd,
        list_usd=list_usd,
        latency_ms=latency_ms,
        n_steps=2,
        outcome="ok",
        n_calls=n_calls,
        cached_calls=cached_calls,
    )


# ---------------------------------------------------------------- E2


def test_a_fully_replayed_policy_reports_no_latency() -> None:
    """Regression: cache lookups were published as policy latency.

    None is the only honest value. A zero would be read as "instant".
    """
    cached = [_outcome(latency_ms=3, n_calls=2, cached_calls=2) for _ in range(10)]
    s = summarise("b2-tools", cached)
    assert s["e2_p50_latency_ms"] is None
    assert s["e2_p95_latency_ms"] is None
    assert s["e2_n"] == 0


def test_cached_runs_do_not_drag_the_percentiles_down() -> None:
    """A mixed sweep must report the live runs, not an average of two different things."""
    live = [_outcome(latency_ms=9000, n_calls=2, cached_calls=0) for _ in range(5)]
    cached = [_outcome(latency_ms=4, n_calls=2, cached_calls=2) for _ in range(45)]
    s = summarise("b3-frontier", live + cached)
    assert s["e2_p50_latency_ms"] == 9000
    assert s["e2_n"] == 5


def test_a_partially_cached_run_is_excluded() -> None:
    """One cache hit anywhere in the trajectory makes the wall time unrepresentative."""
    s = summarise("p", [_outcome(latency_ms=50, n_calls=4, cached_calls=1)])
    assert s["e2_n"] == 0


def test_a_run_that_called_nothing_is_excluded() -> None:
    """A run that failed before its first model call has no latency worth reporting."""
    s = summarise("p", [_outcome(latency_ms=2, n_calls=0, cached_calls=0)])
    assert s["e2_n"] == 0


def test_e2_sample_size_travels_with_the_figures() -> None:
    """The reader must be able to see how many runs the percentile rests on."""
    s = summarise("p", [_outcome(n_calls=1, cached_calls=0) for _ in range(3)])
    assert s["e2_n"] == 3
    assert s["n"] == 3


def test_a_fully_live_sweep_uses_every_run() -> None:
    s = summarise("p", [_outcome(latency_ms=i * 100) for i in range(1, 11)])
    assert s["e2_n"] == 10
    assert s["e2_p95_latency_ms"] == 1000


# ---------------------------------------------------------------- E1


def test_e1_uses_list_cost_so_replay_does_not_make_a_policy_look_free() -> None:
    """E1 describes the policy. Actual cost is zero on a replay; list cost is not."""
    replayed = [_outcome(cost_usd=0.0, list_usd=0.02, cached_calls=2) for _ in range(4)]
    s = summarise("p", replayed)
    assert s["e1_cost_per_correct"] == 0.02
    assert s["total_spent_usd"] == 0.0


def test_e1_is_none_when_nothing_is_correct() -> None:
    """Infinite cost per correct answer. None is the honest rendering, not zero."""
    s = summarise("p", [_outcome(correct=False) for _ in range(5)])
    assert s["e1_cost_per_correct"] is None


def test_e1_divides_by_correct_answers_not_by_run_count() -> None:
    outcomes = [_outcome(correct=True, list_usd=0.10), _outcome(correct=False, list_usd=0.10)]
    s = summarise("p", outcomes)
    assert s["e1_cost_per_correct"] == 0.20


# ---------------------------------------------------------------- accuracy


def test_accuracy_by_archetype_covers_every_archetype_present() -> None:
    outcomes = [
        _outcome(archetype="ratio", correct=False),
        _outcome(archetype="ratio", correct=True),
        _outcome(archetype="lookup", correct=True),
    ]
    s = summarise("p", outcomes)
    assert s["accuracy_by_archetype"] == {"lookup": 1.0, "ratio": 0.5}


def test_the_confidence_interval_widens_as_the_sample_shrinks() -> None:
    """A 50% on 4 questions and a 50% on 100 are not the same claim."""
    small = summarise("p", [_outcome(correct=i % 2 == 0) for i in range(4)])
    large = summarise("p", [_outcome(correct=i % 2 == 0) for i in range(100)])
    small_width = small["d1_ci95"][1] - small["d1_ci95"][0]
    large_width = large["d1_ci95"][1] - large["d1_ci95"][0]
    assert small_width > large_width
