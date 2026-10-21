"""Build the demo's question catalog: the eval questions that replay from the journal.

A replayed question costs nothing, but only if every model call it makes is already in the
journal -- the same question, horizon, company and model the sweep used. That cannot be
checked by matching rows, because the prompt the executor builds has changed since some runs
were recorded (the coverage note, for one). So each candidate is actually replayed, and only
the ones that resolve completely are listed.

Each check stores its run under the policy `catalog-check` so that verification can read it,
then deletes it. Nothing else in the database is touched.

    python -m scripts.demo_catalog            # every split question, both policies
    python -m scripts.demo_catalog --limit 20 # quicker, for a smoke test
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console

from financevault import verify
from financevault.api.service import CATALOG_PATH, POLICIES, policy_model
from financevault.env.asof import AsOf
from financevault.runtime import executor, llm
from financevault.runtime.budget import Ledger
from financevault.runtime.journal import ReplayMiss
from financevault.store import pg

console = Console()
SPLIT = Path("eval/splits/mvp_150.jsonl")


def check(q: dict, policy: str) -> dict | None:
    """Replay one question under one policy. None if any model call misses the journal."""
    try:
        run = executor.execute(
            q["question"],
            as_of=AsOf.parse(q["as_of"]),
            cik=q.get("cik"),
            ticker=q.get("ticker"),
            policy="catalog-check",
            model=policy_model(policy),
        )
    except ReplayMiss:
        return None
    try:
        verified = True
        try:
            verdicts = verify.verify_run(
                run.id,
                cik=q.get("cik"),
                expected_value=q["expected_value"],
                ledger=Ledger(max_usd=0.05, max_steps=40),
                persist=False,
            )
        except ReplayMiss:
            verdicts, verified = [], False
        return {
            "outcome": run.outcome,
            "n_steps": len(run.steps),
            "path": run.path,
            "list_usd": round(run.ledger.list_usd, 6) if run.ledger else 0.0,
            "verified": verified,
            "correct": verify.answered_correctly(verdicts) if verified else None,
        }
    finally:
        pg.execute("DELETE FROM runs WHERE id = %s", (run.id,))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="check only the first N questions")
    args = parser.parse_args()

    questions = [json.loads(line) for line in SPLIT.read_text().splitlines() if line.strip()]
    if args.limit:
        questions = questions[: args.limit]

    listed: list[dict] = []
    misses = 0
    with llm.replay_mode(True):
        for i, q in enumerate(questions, 1):
            results = {}
            for policy in POLICIES:
                result = check(q, policy)
                if result is None:
                    misses += 1
                else:
                    results[policy] = result
            mark = "[green]ok[/green]" if results else "[dim]miss[/dim]"
            console.print(f"{i:>3}/{len(questions)} {mark} {q['question'][:80]}")
            if results:
                listed.append(
                    {
                        "id": q["id"],
                        "question": q["question"],
                        "as_of": q["as_of"],
                        "ticker": q.get("ticker"),
                        "cik": q.get("cik"),
                        "archetype": q.get("archetype"),
                        "split": q.get("split"),
                        "expected_value": q["expected_value"],
                        "expected_unit": q.get("expected_unit"),
                        "policies": results,
                    }
                )

    CATALOG_PATH.write_text(
        json.dumps(
            {
                "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "source": str(SPLIT),
                "checked": len(questions),
                "questions": listed,
            },
            indent=1,
        )
        + "\n"
    )
    console.print(
        f"\n{len(listed)} of {len(questions)} questions replay under at least one policy "
        f"({misses} policy-question pairs missed the journal) -> {CATALOG_PATH}"
    )
    return 0 if listed else 1


if __name__ == "__main__":
    raise SystemExit(main())
