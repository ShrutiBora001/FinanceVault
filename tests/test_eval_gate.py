"""The regression gate's classification. Pure: no database, no model."""

from scripts.eval_gate import by_archetype, compare

ROW = {
    "outcome": "ok",
    "n_steps": 2,
    "path": "P2",
    "list_usd": 0.01,
    "verified": True,
    "correct": True,
}


def _baseline(**overrides):
    return {
        "questions": [
            {"id": "q1", "archetype": "lookup", "policies": {"b2-tools": {**ROW, **overrides}}}
        ]
    }


def _current(row):
    return {"q1": {"b2-tools": row}}


def test_identical_replay_passes():
    assert compare(_baseline(), _current(dict(ROW))) == {
        "regressions": [],
        "drift": [],
        "gains": [],
    }


def test_journal_miss_is_a_regression():
    assert len(compare(_baseline(), _current(None))["regressions"]) == 1


def test_correct_to_wrong_is_a_regression():
    assert len(compare(_baseline(), _current({**ROW, "correct": False}))["regressions"]) == 1


def test_outcome_leaving_ok_is_a_regression_even_when_baseline_was_wrong():
    found = compare(
        _baseline(correct=False), _current({**ROW, "correct": False, "outcome": "max_steps"})
    )
    assert len(found["regressions"]) == 1


def test_wrong_to_correct_is_a_gain_not_a_block():
    found = compare(_baseline(correct=False), _current(dict(ROW)))
    assert found["gains"] and not found["regressions"] and not found["drift"]


def test_any_other_field_change_is_drift():
    found = compare(_baseline(), _current({**ROW, "n_steps": 3}))
    assert found["drift"] == ["q1 [b2-tools]: n_steps 2 -> 3"]


def test_archetype_counts_a_miss_as_not_correct():
    assert by_archetype(_baseline(), _current(None)) == {"lookup": (1, 0, 1)}
