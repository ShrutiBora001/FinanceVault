"""Regression gate: replay the baseline questions and block on any change.

The baseline is the demo catalog (`make catalog`): every question/policy pair known to replay,
with its outcome, step count, routing path, list cost and verifier result. Under replay every
model call resolves from the journal, so the same code must reproduce the same row exactly.
That makes the rule binary (Q25: "dev accuracy, replayed: any change is a bug"):

- regression, blocks: was correct and is not, outcome left `ok`, or the pair no longer replays
  (a journal miss means the prompt, router or tool specs changed the request).
- drift, blocks: any other field changed. Replay is supposed to be byte-deterministic.
- gain, does not block: was wrong, now correct. Refresh the baseline with `make catalog`.

Costs $0.00 and calls no model.

    python -m scripts.eval_gate                 # gate against api/catalog.json
    python -m scripts.eval_gate --limit 5       # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

COMPARED = ("outcome", "n_steps", "path", "list_usd", "verified", "correct")


def compare(baseline: dict, current: dict) -> dict[str, list[str]]:
    """Classify every (question, policy) difference. `current[qid][policy]` is None on a miss."""
    found: dict[str, list[str]] = {"regressions": [], "drift": [], "gains": []}
    for q in baseline["questions"]:
        for policy, was in q["policies"].items():
            now = current.get(q["id"], {}).get(policy)
            pair = f"{q['id']} [{policy}]"
            if now is None:
                found["regressions"].append(f"{pair}: no longer replays (journal miss)")
                continue
            if was["correct"] and not now["correct"]:
                found["regressions"].append(f"{pair}: correct -> {now['correct']}")
            elif was["outcome"] == "ok" and now["outcome"] != "ok":
                found["regressions"].append(f"{pair}: outcome ok -> {now['outcome']}")
            elif not was["correct"] and now["correct"]:
                found["gains"].append(f"{pair}: {was['correct']} -> correct")
            else:
                diffs = [
                    f"{k} {was.get(k)} -> {now.get(k)}"
                    for k in COMPARED
                    if was.get(k) != now.get(k)
                ]
                if diffs:
                    found["drift"].append(f"{pair}: " + ", ".join(diffs))
    return found


def by_archetype(baseline: dict, current: dict) -> dict[str, tuple[int, int, int]]:
    """archetype -> (baseline correct, current correct, pairs). Per class, never pooled (Q25)."""
    acc: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for q in baseline["questions"]:
        for policy, was in q["policies"].items():
            now = current.get(q["id"], {}).get(policy) or {}
            row = acc[q.get("archetype") or "?"]
            row[0] += bool(was["correct"])
            row[1] += bool(now.get("correct"))
            row[2] += 1
    return {a: tuple(v) for a, v in sorted(acc.items())}


def report(found: dict[str, list[str]], archetypes: dict[str, tuple[int, int, int]]) -> str:
    lines = [
        "## Eval regression gate (replay, $0.00)",
        "",
        "| archetype | baseline | now | pairs |",
        "|---|---|---|---|",
    ]
    lines += [f"| {a} | {b} | {c} | {n} |" for a, (b, c, n) in archetypes.items()]
    for kind in ("regressions", "drift", "gains"):
        lines += ["", f"**{kind}: {len(found[kind])}**"] + [f"- {x}" for x in found[kind]]
    return "\n".join(lines) + "\n"


def main() -> int:
    from financevault.api.service import CATALOG_PATH, POLICIES  # noqa: PLC0415
    from financevault.runtime import llm  # noqa: PLC0415
    from scripts.demo_catalog import check  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=CATALOG_PATH)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text())
    if args.limit:
        baseline["questions"] = baseline["questions"][: args.limit]

    current: dict[str, dict] = {}
    with llm.replay_mode(True):
        for q in baseline["questions"]:
            current[q["id"]] = {p: check(q, p) for p in q["policies"] if p in POLICIES}

    found = compare(baseline, current)
    text = report(found, by_archetype(baseline, current))
    print(text)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as fh:
            fh.write(text)
    return 1 if found["regressions"] or found["drift"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
