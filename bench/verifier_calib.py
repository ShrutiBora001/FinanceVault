"""B2 — seeded-error detection.

Take trajectories the verifier already passed, inject a known defect, and measure whether the
verifier catches it. The ground truth is constructed, so this is stronger evidence than
agreement with a human labeller: there is no ambiguity about whether the injected step is
wrong.

Recall is reported **per error class, never pooled**. A single averaged number would hide the
one limitation already known to exist — `s3`'s scale check is deliberately tolerant for
figures quoted in prose, because filing tables state amounts "in millions" and the extractor
cannot see the column header. Pooling would let strong classes mask that.

The control matters as much as the seeds: unmutated trajectories are scored too, and a
verifier that flags those is useless however good its recall. Recall without a false-positive
rate is not a measurement.

Runs against stored trajectories and calls the signals directly, so the programmatic classes
cost nothing. Only the classes that reach a judge spend anything.
"""

from __future__ import annotations

import copy
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from financevault.store import pg
from financevault.verify import citation, numeric, tool_validity

console = Console()
OUT = Path("eval/results/b2_seeded.json")

GATE = 0.70


@dataclass(slots=True)
class Case:
    """One trajectory, plus the mutation applied to it."""

    run_id: str
    question: str
    answer: str
    steps: list[dict]
    citations: list[dict]
    value: float | None


@dataclass(slots=True)
class ClassResult:
    seeded: int = 0
    caught: int = 0
    signal: str = ""
    misses: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.caught / self.seeded if self.seeded else 0.0


# ---------------------------------------------------------------- mutations
#
# Each returns a mutated Case, or None when the trajectory cannot carry that defect.


def _terminal(case: Case) -> dict | None:
    for step in reversed(case.steps):
        if step.get("tool") == "finish":
            return step
    return None


def off_by_1000(case: Case) -> Case | None:
    """The headline silent error: right digits, wrong magnitude."""
    if case.value is None:
        return None
    bad = copy.deepcopy(case)
    bad.value = case.value / 1000
    bad.answer = f"The figure was ${bad.value:,.0f}."
    return bad


def wrong_scale_word(case: Case) -> Case | None:
    """Same digits, the scale word swapped — how the error appears in prose."""
    if "billion" not in (case.answer or "").lower():
        return None
    bad = copy.deepcopy(case)
    bad.answer = case.answer.replace("billion", "million").replace("Billion", "Million")
    bad.value = case.value / 1000 if case.value is not None else None
    return bad


def wrong_period(case: Case) -> Case | None:
    """A real figure for this company, but for a different period."""
    other = pg.fetch_one(
        """
        SELECT value FROM xbrl_facts
        WHERE cik = '320193' AND unit = 'USD' AND value <> %s AND value > 1e9
        ORDER BY random() LIMIT 1
        """,
        (case.value,),
    )
    if other is None or case.value is None:
        return None
    bad = copy.deepcopy(case)
    bad.value = float(other["value"])
    bad.answer = f"The figure was ${bad.value:,.0f}."
    return bad


def fabricated_value(case: Case) -> Case | None:
    """A number matching nothing retrieved and nothing on record."""
    bad = copy.deepcopy(case)
    bad.value = 12_345_678_901_234.0
    bad.answer = f"The figure was ${bad.value:,.0f}."
    return bad


def fabricated_citation(case: Case) -> Case | None:
    """Cites a source the trajectory never retrieved."""
    bad = copy.deepcopy(case)
    bad.citations = [{"kind": "fact", "ref": "TotallyInventedTag", "accession": None}]
    return bad


def hallucinated_tool(case: Case) -> Case | None:
    step = _terminal(case)
    if step is None:
        return None
    bad = copy.deepcopy(case)
    _terminal(bad)["tool"] = "fetch_stock_price"  # type: ignore[index]
    return bad


def malformed_args(case: Case) -> Case | None:
    bad = copy.deepcopy(case)
    for step in bad.steps:
        if step.get("tool") == "lookup_fact":
            step["args"] = {"not_a_field": 1}
            return bad
    return None


MUTATIONS: dict[str, tuple[Callable[[Case], Case | None], str]] = {
    "off_by_1000": (off_by_1000, "s3"),
    "wrong_scale_word": (wrong_scale_word, "s3"),
    "wrong_period": (wrong_period, "s3"),
    "fabricated_value": (fabricated_value, "s3"),
    "fabricated_citation": (fabricated_citation, "s2"),
    "hallucinated_tool": (hallucinated_tool, "s1"),
    "malformed_args": (malformed_args, "s1"),
}


# ---------------------------------------------------------------- scoring


def score_case(case: Case) -> dict[str, bool]:
    """The programmatic signals for a (possibly mutated) trajectory."""
    evidence = numeric.collect_evidence(case.steps)

    s1 = min(
        (tool_validity.score_step(s).score for s in case.steps if s.get("tool")),
        default=1.0,
    )
    s2 = citation.score(
        case.answer, case.citations, case.steps, ledger=None, evidence_text=""
    ).score
    s3 = numeric.score(case.answer, evidence=evidence, cik="320193", stated_value=case.value)
    # `passed`, not a bare score comparison: signals carry their own threshold, and s3's is
    # 1.0. Re-implementing the comparison here would let the calibration disagree with the
    # verifier it is calibrating.
    return {"s1": s1 >= 0.5, "s2": s2 >= 0.5, "s3": s3.passed}


def load_cases(limit: int) -> list[Case]:
    runs = pg.fetch_all(
        """
        SELECT id, question, answer FROM runs
        WHERE policy = 'b2-tools' AND outcome = 'ok' AND answer IS NOT NULL
        ORDER BY created_at DESC LIMIT %s
        """,
        (limit,),
    )
    cases: list[Case] = []
    for run in runs:
        steps = pg.fetch_all(
            "SELECT idx, tool, args, obs FROM steps WHERE run_id = %s ORDER BY idx", (run["id"],)
        )
        terminal = next((s for s in reversed(steps) if s["tool"] == "finish"), None)
        if terminal is None:
            continue
        args = terminal.get("args") or {}
        cases.append(
            Case(
                run_id=str(run["id"]),
                question=run["question"],
                answer=run["answer"],
                steps=steps,
                citations=args.get("citations") or [],
                value=args.get("value"),
            )
        )
    return cases


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    cases = load_cases(args.limit)
    if not cases:
        console.print("[red]no clean b2-tools trajectories[/red] — run `make eval` first")
        return 1

    # Control: the unmutated trajectories. A verifier that flags these is useless.
    control_flagged, control_detail = 0, []
    for case in cases:
        scores = score_case(case)
        failing = [k for k, ok in scores.items() if not ok]
        if failing:
            control_flagged += 1
            control_detail.append({"run_id": case.run_id, "failing": failing})

    results: dict[str, ClassResult] = {}
    for name, (mutate, signal) in MUTATIONS.items():
        result = ClassResult(signal=signal)
        for case in cases:
            bad = mutate(case)
            if bad is None:
                continue
            result.seeded += 1
            if not score_case(bad).get(signal, True):
                result.caught += 1
            else:
                result.misses.append(case.run_id)
        results[name] = result

    table = Table(
        title=f"B2 seeded-error detection ({len(cases)} trajectories)", header_style="bold"
    )
    table.add_column("error class")
    table.add_column("signal")
    table.add_column("seeded", justify="right")
    table.add_column("caught", justify="right")
    table.add_column("recall", justify="right")
    table.add_column("gate", justify="center")
    for name, r in results.items():
        ok = r.seeded == 0 or r.recall >= GATE
        table.add_row(
            name,
            r.signal,
            str(r.seeded),
            str(r.caught),
            "—" if not r.seeded else f"{r.recall:.0%}",
            "—" if not r.seeded else ("[green]pass[/green]" if ok else "[red]FAIL[/red]"),
        )
    console.print(table)

    fpr = control_flagged / len(cases)
    console.print(
        f"\ncontrol false-positive rate: [bold]{fpr:.0%}[/bold] "
        f"({control_flagged}/{len(cases)} clean trajectories flagged)"
    )

    tested = {k: v for k, v in results.items() if v.seeded}
    failed = [k for k, v in tested.items() if v.recall < GATE]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "n_trajectories": len(cases),
                "gate": GATE,
                "control_false_positive_rate": round(fpr, 4),
                "control_detail": control_detail,
                "classes": {
                    k: {
                        "signal": v.signal,
                        "seeded": v.seeded,
                        "caught": v.caught,
                        "recall": round(v.recall, 4),
                    }
                    for k, v in results.items()
                },
                "failed_classes": failed,
            },
            indent=2,
        )
        + "\n"
    )
    console.print(f"wrote {OUT}")

    if failed:
        console.print(
            f"\n[red]GATE FAILED[/red] — {', '.join(failed)} below {GATE:.0%}. "
            "Fix these before spending on trajectory generation."
        )
        return 1
    if fpr > 0.1:
        console.print(f"\n[red]GATE FAILED[/red] — {fpr:.0%} of clean trajectories flagged.")
        return 1
    console.print(f"\n[green]GATE PASSED[/green] — every tested class at or above {GATE:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
