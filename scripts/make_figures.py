"""Regenerate every figure in the report from the results file and the runs table.

Figures are derived, never hand-drawn: `make figures` after a sweep reproduces all of them, so
a plot cannot drift from the numbers it claims to show.

Trajectory data comes from the *latest* run per (policy, question). Several sweeps of the same
split exist in `runs` — the ceiling investigation alone produced three — and averaging across
them would silently mix configurations, which is the same class of error as mixing cached and
live runs into a latency percentile.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from financevault.store import pg  # noqa: E402

RESULTS = Path("eval/results/mvp2.2_dev.json")
OUT = Path("report/figures")

# One colour per policy, used everywhere. A reader should be able to track b3 across figures
# without consulting a legend each time.
COLOURS = {
    "b0-no-tools": "#b0b7c3",
    "b1-rag": "#7a9cc6",
    "b2-tools": "#2f6f9f",
    "b3-frontier": "#1b3a57",
}
LABELS = {
    "b0-no-tools": "B0\nno tools",
    "b1-rag": "B1\nRAG",
    "b2-tools": "B2\nHaiku+tools",
    "b3-frontier": "B3\nSonnet+tools",
}
ARCHETYPES = ["lookup", "delta", "ratio", "cross_company"]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    }
)


def summaries() -> list[dict]:
    data = json.loads(RESULTS.read_text())
    rows = data["policies"] if "policies" in data else data
    return list(rows.values()) if isinstance(rows, dict) else rows


SPLIT = Path("eval/splits/mvp_150.jsonl")


def dev_questions() -> list[str]:
    """The dev subset only. `runs` also holds MVP1's 20-question split and every earlier
    sweep; including those would plot a different benchmark alongside this one."""
    rows = (json.loads(line) for line in SPLIT.read_text().splitlines() if line.strip())
    return [r["question"] for r in rows if r["split"] == "dev"]


def latest_runs() -> list[dict]:
    """The most recent run per (policy, question). See the module docstring."""
    return pg.fetch_all(
        """
        SELECT DISTINCT ON (policy, question)
               policy, question, path, outcome, n_steps, id
        FROM runs
        WHERE question = ANY(%(questions)s)
        ORDER BY policy, question, created_at DESC
        """,
        {"questions": dev_questions()},
    )


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"  {OUT / name}")


# ---------------------------------------------------------------- figures


def fig_accuracy(rows: list[dict]) -> None:
    """D1 with Wilson intervals. The intervals are the point: n=30 is a wide instrument."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    xs = range(len(rows))
    acc = [r["d1_accuracy"] * 100 for r in rows]
    lo = [a - r["d1_ci95"][0] * 100 for a, r in zip(acc, rows, strict=True)]
    hi = [r["d1_ci95"][1] * 100 - a for a, r in zip(acc, rows, strict=True)]

    ax.bar(xs, acc, color=[COLOURS[r["policy"]] for r in rows], width=0.62)
    ax.errorbar(xs, acc, yerr=[lo, hi], fmt="none", ecolor="#333", capsize=4, lw=1.1)
    for x, a in zip(xs, acc, strict=True):
        ax.text(x, a + 4.5, f"{a:.0f}%", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(list(xs))
    ax.set_xticklabels([LABELS[r["policy"]] for r in rows])
    ax.set_ylabel("D1 numeric accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Answer accuracy on the dev split (n=30)\nbars are 95% Wilson intervals")
    _save(fig, "f1_accuracy.png")


def fig_archetype(rows: list[dict]) -> None:
    """Where the difficulty actually lives. A single accuracy number hides this entirely."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    width = 0.2
    for i, r in enumerate(rows):
        vals = [r["accuracy_by_archetype"].get(a, 0) * 100 for a in ARCHETYPES]
        ax.bar(
            [x + i * width for x in range(len(ARCHETYPES))],
            vals,
            width=width,
            color=COLOURS[r["policy"]],
            label=r["policy"],
        )
    ax.set_xticks([x + 1.5 * width for x in range(len(ARCHETYPES))])
    ax.set_xticklabels(["lookup\n(16)", "delta\n(7)", "ratio\n(5)", "cross-company\n(2)"])
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 108)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.set_title("Accuracy by question archetype\nsubset sizes in parentheses")
    _save(fig, "f2_archetype.png")


def fig_frontier(rows: list[dict]) -> None:
    """E1 against D1. The question is which policies are on the frontier, not which is best."""
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for r in rows:
        if r["e1_cost_per_correct"] is None:
            continue
        ax.scatter(
            r["e1_cost_per_correct"],
            r["d1_accuracy"] * 100,
            s=150,
            color=COLOURS[r["policy"]],
            zorder=3,
            edgecolor="white",
            linewidth=1.4,
        )
        ax.annotate(
            r["policy"].split("-")[0].upper(),
            (r["e1_cost_per_correct"], r["d1_accuracy"] * 100),
            textcoords="offset points",
            xytext=(9, 5),
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlabel("E1 cost per correct answer (USD, list price)")
    ax.set_ylabel("D1 accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Accuracy against cost\nup and to the left is better")
    _save(fig, "f3_frontier.png")


# Measured across the ceiling investigation; see report §7.3. Kept as literals because the
# "before" configuration no longer exists in the code and cannot be re-derived from the DB.
CEILING_ABLATION = {
    "b2-tools": {"before": 66.7, "after": 73.3, "aborts_before": 10, "aborts_after": 7},
    "b3-frontier": {"before": 70.0, "after": 86.7, "aborts_before": 7, "aborts_after": 2},
}


def fig_ceilings() -> None:
    """The confound. Both ceilings bound hardest on the model that used them best."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.3))
    policies = list(CEILING_ABLATION)
    xs = range(len(policies))

    for ax, keys, ylab, title, fmt in (
        (ax1, ("before", "after"), "D1 accuracy (%)", "Accuracy", "{:.0f}%"),
        (ax2, ("aborts_before", "aborts_after"), "runs aborted (of 30)", "Aborted runs", "{:.0f}"),
    ):
        for i, key in enumerate(keys):
            vals = [CEILING_ABLATION[p][key] for p in policies]
            bars = ax.bar(
                [x + i * 0.36 for x in xs],
                vals,
                width=0.36,
                color=["#c9d3dd", "#1b3a57"][i],
                label=["ceilings binding", "ceilings raised"][i],
            )
            ax.bar_label(bars, fmt=fmt, fontsize=8, padding=2)
        ax.set_xticks([x + 0.18 for x in xs])
        ax.set_xticklabels([LABELS[p].replace("\n", " ") for p in policies], fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title(title)
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 13)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Effect of the execution ceilings on measured capability", y=1.02)
    _save(fig, "f4_ceilings.png")


def fig_signals(rows: list[dict]) -> None:
    """Mean per-signal score. D3 is a conjunction, so its weakest term dominates."""
    data = pg.fetch_all(
        """
        WITH last AS (
            SELECT DISTINCT ON (policy, question) id, policy
            FROM runs
            WHERE policy IN ('b2-tools','b3-frontier') AND question = ANY(%(questions)s)
            ORDER BY policy, question, created_at DESC
        )
        SELECT l.policy, avg(s.s1) s1, avg(s.s2) s2, avg(s.s3) s3, avg(s.s4) s4
        FROM steps s JOIN last l ON l.id = s.run_id GROUP BY 1 ORDER BY 1
    """,
        {"questions": dev_questions()},
    )
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    names = [
        "s1 tool\nvalidity",
        "s2 citation\nsupport",
        "s3 numeric\ncorrectness",
        "s4 retrieval\nrelevance",
    ]
    for i, r in enumerate(data):
        vals = [float(r[k]) for k in ("s1", "s2", "s3", "s4")]
        bars = ax.bar(
            [x + i * 0.36 for x in range(4)],
            vals,
            width=0.36,
            color=COLOURS[r["policy"]],
            label=r["policy"],
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.axhline(0.5, color="#b23", ls="--", lw=1, zorder=1)
    ax.text(1.55, 0.53, "pass threshold", color="#b23", fontsize=7, ha="center")
    ax.set_xticks([x + 0.18 for x in range(4)])
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("mean signal score")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Verifier signals, averaged over steps\ns4 is the binding constraint on D3")
    _save(fig, "f5_signals.png")


def fig_routing(runs: list[dict]) -> None:
    """Every abort was on P3, and P4 was never used at all."""
    rows = [r for r in runs if r["policy"] in ("b2-tools", "b3-frontier")]
    paths = ["P0", "P1", "P2", "P3", "P4"]
    total = Counter(r["path"] for r in rows)
    aborted = Counter(r["path"] for r in rows if r["outcome"] != "ok")

    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ok = [total[p] - aborted[p] for p in paths]
    bad = [aborted[p] for p in paths]
    ax.bar(paths, ok, color="#2f6f9f", label="completed")
    ax.bar(paths, bad, bottom=ok, color="#c0392b", label="did not complete")
    for i, p in enumerate(paths):
        if total[p]:
            ax.text(i, total[p] + 1.2, str(total[p]), ha="center", fontsize=8)
    ax.set_ylabel("runs (B2 and B3 combined)")
    ax.set_xlabel("execution path chosen by the router")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(
        "Router path usage and where runs died\nevery ceiling abort is on P3; P4 is never selected"
    )
    _save(fig, "f6_routing.png")


def fig_tools(runs: list[dict]) -> None:
    """What the agents actually do, as opposed to what the toolset offers."""
    ids = {r["id"]: r["policy"] for r in runs if r["policy"] in ("b2-tools", "b3-frontier")}
    counts: dict[str, Counter] = {p: Counter() for p in ("b2-tools", "b3-frontier")}
    for row in pg.fetch_all("SELECT run_id, tool FROM steps"):
        policy = ids.get(row["run_id"])
        if policy:
            counts[policy][row["tool"] or "no tool call"] += 1

    tools = ["sql", "lookup_fact", "python", "retrieve_filings", "finish", "no tool call"]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for i, (policy, c) in enumerate(counts.items()):
        bars = ax.bar(
            [x + i * 0.36 for x in range(len(tools))],
            [c[t] for t in tools],
            width=0.36,
            color=COLOURS[policy],
            label=policy,
        )
        ax.bar_label(bars, fontsize=7, padding=2)
    ax.set_xticks([x + 0.18 for x in range(len(tools))])
    ax.set_xticklabels(tools, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("tool calls")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Tool-call distribution\nboth agents reach for sql over lookup_fact")
    _save(fig, "f7_tools.png")


def fig_steps(runs: list[dict]) -> None:
    """Trajectory length. The tail is where the ceilings used to cut."""
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    for policy in ("b2-tools", "b3-frontier"):
        lens = [r["n_steps"] for r in runs if r["policy"] == policy]
        ax.hist(
            lens,
            bins=range(1, 13),
            alpha=0.65,
            color=COLOURS[policy],
            label=f"{policy} (mean {sum(lens) / len(lens):.1f})",
        )
    ax.axvline(6, color="#c0392b", ls="--", lw=1)
    ax.text(6.15, ax.get_ylim()[1] * 0.9, "old P3 ceiling", color="#c0392b", fontsize=7)
    ax.set_xlabel("steps per run")
    ax.set_ylabel("runs")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Trajectory length")
    _save(fig, "f8_steps.png")


def fig_pipeline() -> None:
    """The two paths, drawn once. Online serves an answer; offline turns runs into data."""
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.axis("off")
    ax.grid(False)

    online = [
        ("question\n+ as_of\n+ budget", "#e8eef4"),
        ("router\nP0-P4", "#cfdce8"),
        ("step executor\nplan / act\nobserve", "#2f6f9f"),
        ("tools\n5 contracts", "#cfdce8"),
        ("answer\n+ citations", "#e8eef4"),
    ]
    offline = [
        ("journal\ncontent\naddressed", "#e8eef4"),
        ("step verifier\ns1-s5", "#1b3a57"),
        ("dataset builder\naccept / repair\nhard-neg", "#cfdce8"),
        ("LoRA SFT", "#e8eef4"),
        ("eval harness\nB0-B5", "#e8eef4"),
    ]
    for row, (items, label) in enumerate(((online, "online"), (offline, "offline"))):
        y = 0.62 - row * 0.42
        for i, (text, colour) in enumerate(items):
            x = 0.04 + i * 0.192
            dark = colour in ("#2f6f9f", "#1b3a57")
            ax.add_patch(
                plt.Rectangle((x, y), 0.163, 0.24, facecolor=colour, edgecolor="#41506080", lw=0.8)
            )
            ax.text(
                x + 0.082,
                y + 0.12,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if dark else "#20303f",
                fontweight="bold" if dark else "normal",
            )
            if i < len(items) - 1:
                ax.annotate(
                    "",
                    xy=(x + 0.19, y + 0.12),
                    xytext=(x + 0.167, y + 0.12),
                    arrowprops={"arrowstyle": "->", "color": "#415060", "lw": 1},
                )
        ax.text(
            0.02,
            y + 0.12,
            label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=8,
            color="#415060",
            fontweight="bold",
        )

    ax.annotate(
        "",
        xy=(0.12, 0.44),
        xytext=(0.45, 0.6),
        arrowprops={
            "arrowstyle": "->",
            "color": "#c0392b",
            "lw": 1.2,
            "connectionstyle": "arc3,rad=0.25",
        },
    )
    ax.text(0.30, 0.47, "every call journalled", fontsize=7, color="#c0392b")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 0.95)
    ax.set_title("FinanceVault: the online and offline paths", fontsize=10)
    _save(fig, "f0_pipeline.png")


def main() -> int:
    rows = sorted(summaries(), key=lambda r: r["policy"])
    runs = latest_runs()
    print(f"figures from {len(rows)} policies, {len(runs)} runs")
    fig_pipeline()
    fig_accuracy(rows)
    fig_archetype(rows)
    fig_frontier(rows)
    fig_ceilings()
    fig_signals(rows)
    fig_routing(runs)
    fig_tools(runs)
    fig_steps(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
