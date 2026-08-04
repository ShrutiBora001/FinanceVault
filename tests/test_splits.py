"""Training-split construction and export.

These produce the data H1 is tested on, so a defect here is not a bug in a feature — it
silently changes what the experiment measures. The two conditions being compared
(`step_filtered` against outcome filtering) are built by this code, and if the split boundary
drifts, the comparison drifts with it and nothing else in the suite notices.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from financevault.data import export, splits
from financevault.data.rollout import Trajectory
from financevault.env.asof import AsOf
from financevault.runtime.executor import Run, Step
from financevault.verify import StepVerdict
from financevault.verify.signals import Signal, na

NET_INCOME = 112_010_000_000.0


def _run(steps: list[Step], answer: str, value: float | None) -> Run:
    return Run(
        id="00000000-0000-0000-0000-000000000000",
        question="What was net income?",
        as_of=AsOf.parse("2026-01-01"),
        policy="test",
        path="P2",
        outcome="ok",
        answer=answer,
        citations=[],
        value=value,
        unit="USD",
        steps=steps,
    )


def _lookup_step(idx: int, ok: bool = True) -> Step:
    return Step(
        idx=idx,
        thought="looking it up",
        tool="lookup_fact",
        args={"tag": "NetIncomeLoss"},
        observation=(
            {"ok": True, "data": [{"tag": "NetIncomeLoss", "value": NET_INCOME}]}
            if ok
            else {"ok": False, "error": "no such tag"}
        ),
    )


def _finish_step(idx: int, value: float | None = NET_INCOME) -> Step:
    return Step(
        idx=idx,
        thought="",
        tool="finish",
        args={"answer": f"Net income was ${value:,.0f}." if value else "n/a", "value": value},
        observation={"ok": True, "data": {"answer": "...", "value": value}},
    )


def _verdict(idx: int, tool: str | None, passed: bool, s5: float | None = None) -> StepVerdict:
    signals = {
        "s1": Signal(1.0 if passed else 0.0, "t"),
        "s2": na("n/a"),
        "s3": na("n/a"),
        "s4": Signal(1.0 if passed else 0.0, "t"),
        "s5": Signal(s5, "t") if s5 is not None else na("not terminal"),
    }
    return StepVerdict(idx=idx, tool=tool, signals=signals)


def _trajectory(
    *, steps: list[Step], verdicts: list[StepVerdict], variant: str = "full"
) -> Trajectory:
    answer = f"Net income was ${NET_INCOME:,.0f}."
    value = next((s.args.get("value") for s in steps if s.tool == "finish"), None)
    return Trajectory(
        run=_run(steps, answer, value),
        question_id="q1",
        variant=variant,
        expected_value=NET_INCOME,
        verdicts=verdicts,
    )


@pytest.fixture
def clean() -> Trajectory:
    """Every step passes and the answer is right."""
    return _trajectory(
        steps=[_lookup_step(0), _finish_step(1)],
        verdicts=[_verdict(0, "lookup_fact", True), _verdict(1, "finish", True, s5=1.0)],
    )


@pytest.fixture
def detour() -> Trajectory:
    """A wasted first step, then a correct answer. The H1 disagreement case."""
    return _trajectory(
        steps=[_lookup_step(0, ok=False), _lookup_step(1), _finish_step(2)],
        verdicts=[
            _verdict(0, "lookup_fact", False),
            _verdict(1, "lookup_fact", True),
            _verdict(2, "finish", True, s5=1.0),
        ],
    )


@pytest.fixture
def wrong() -> Trajectory:
    """A bad step and a wrong answer. Not repairable."""
    return _trajectory(
        steps=[_lookup_step(0, ok=False), _finish_step(1, value=999.0)],
        verdicts=[_verdict(0, "lookup_fact", False), _verdict(1, "finish", True, s5=0.0)],
    )


# ---------------------------------------------------------------- accepted


def test_accepted_holds_only_fully_clean_trajectories(
    clean: Trajectory, detour: Trajectory, wrong: Trajectory
) -> None:
    out = splits.accepted([clean, detour, wrong])
    assert len(out) == 1
    assert out[0].split == "accepted"


def test_a_correct_answer_with_a_bad_step_is_not_accepted(detour: Trajectory) -> None:
    """The exact case the two filters disagree about."""
    assert detour.answer_correct is True
    assert detour.all_steps_pass is False
    assert splits.accepted([detour]) == []


# ---------------------------------------------------------------- repaired


def test_repair_excises_the_failing_step(detour: Trajectory) -> None:
    out = splits.repaired([detour])
    assert len(out) == 1
    tools = [s["tool"] for s in out[0].steps]
    assert tools == ["lookup_fact", "finish"], "the failed lookup should be gone"


def test_repair_reindexes_the_surviving_steps(detour: Trajectory) -> None:
    out = splits.repaired([detour])
    assert [s["idx"] for s in out[0].steps] == [0, 1]


def test_repair_records_what_was_removed(detour: Trajectory) -> None:
    assert "excised" in splits.repaired([detour])[0].label


def test_a_wrong_answer_is_never_repaired(wrong: Trajectory) -> None:
    """If the answer is wrong the bad step is the cause, not a detour.

    Excising it would fabricate a trajectory that never happened and did not work.
    """
    assert splits.repaired([wrong]) == []


def test_a_clean_trajectory_is_not_repaired(clean: Trajectory) -> None:
    assert splits.repaired([clean]) == []


# ---------------------------------------------------------------- hard negatives


def test_hard_negatives_are_built_from_clean_trajectories(clean: Trajectory) -> None:
    out = splits.hard_negatives([clean])
    assert out and all(e.split == "hard_negative" for e in out)


def test_every_hard_negative_is_one_the_verifier_actually_rejects(clean: Trajectory) -> None:
    """A negative the verifier considers fine is mislabelled data — worse than none."""
    from financevault.verify import numeric

    for example in splits.hard_negatives([clean]):
        evidence = numeric.collect_evidence(
            [{"tool": s["tool"], "obs": s["observation"]} for s in example.steps]
        )
        stated = example.steps[-1]["args"].get("value")
        assert not numeric.score(
            example.answer or "", evidence=evidence, stated_value=stated
        ).passed


def test_hard_negatives_keep_the_trajectory_and_corrupt_only_the_answer(
    clean: Trajectory,
) -> None:
    """The agent's work was reasonable; only the conclusion is wrong. That is the boundary."""
    out = splits.hard_negatives([clean])
    assert out[0].steps[0]["tool"] == "lookup_fact"
    assert out[0].steps[0]["observation"]["ok"] is True


def test_a_flawed_trajectory_yields_no_hard_negative(detour: Trajectory) -> None:
    assert splits.hard_negatives([detour]) == []


# ---------------------------------------------------------------- export


def test_export_writes_every_split(tmp_path: Path, clean: Trajectory, detour: Trajectory) -> None:
    counts = export.write(splits.build([clean, detour]), out_dir=tmp_path)
    for name in ("accepted", "repaired", "hard_negative", "step_filtered"):
        assert (tmp_path / f"{name}.jsonl").exists()
        assert name in counts


def test_step_filtered_is_accepted_plus_repaired(
    tmp_path: Path, clean: Trajectory, detour: Trajectory
) -> None:
    """The condition H1 compares. Built here so the comparison cannot drift later."""
    built = splits.build([clean, detour])
    counts = export.write(built, out_dir=tmp_path)
    assert counts["step_filtered"] == counts["accepted"] + counts["repaired"]


def test_hard_negatives_are_labelled_reject(tmp_path: Path, clean: Trajectory) -> None:
    """Training on these as positives teaches exactly the errors the verifier catches."""
    export.write(splits.build([clean]), out_dir=tmp_path)
    rows = [
        json.loads(line)
        for line in (tmp_path / "hard_negative.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows and all(r["label"] == "reject" for r in rows)


def test_positive_splits_are_labelled_accept(tmp_path: Path, clean: Trajectory) -> None:
    export.write(splits.build([clean]), out_dir=tmp_path)
    rows = [
        json.loads(line)
        for line in (tmp_path / "accepted.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows and all(r["label"] == "accept" for r in rows)


def test_a_conversation_alternates_assistant_and_tool(clean: Trajectory) -> None:
    """The exported shape must match what the executor sends at inference."""
    example = splits.accepted([clean])[0]
    roles = [m["role"] for m in export.to_conversation(example)["messages"]]
    assert roles[0] == "user"
    assert roles[1::2] == ["assistant"] * (len(roles) // 2)
    assert roles[2::2] == ["tool"] * ((len(roles) - 1) // 2)


def test_tool_calls_survive_the_conversion(clean: Trajectory) -> None:
    example = splits.accepted([clean])[0]
    calls = [
        m["tool_calls"][0]["name"]
        for m in export.to_conversation(example)["messages"]
        if m.get("tool_calls")
    ]
    assert calls == ["lookup_fact", "finish"]


def test_export_is_valid_jsonl(tmp_path: Path, clean: Trajectory, detour: Trajectory) -> None:
    export.write(splits.build([clean, detour]), out_dir=tmp_path)
    for path in tmp_path.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                json.loads(line)
