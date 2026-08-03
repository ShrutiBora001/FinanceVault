"""s2 — citation support.

Does the answer's evidence actually support what it claims? Two layers, cheapest first:

**Structural, programmatic.** Are citations present at all, and do they resolve to something
the trajectory actually retrieved — a tag it looked up, a chunk it read? A citation naming a
source that was never fetched is a fabrication, and catching it costs nothing. Most failures
are caught here.

**Semantic, judged.** Only if the structure holds: does the cited span actually say what the
answer says it says? This is the part that genuinely needs a model, and it is the only reason
`s2` costs anything.

Splitting them this way matters for calibration: a structural failure and a semantic failure
are different problems, and a single blended score would hide which one is happening.
"""

from __future__ import annotations

from financevault.runtime.budget import Ledger
from financevault.verify.judge import ask, truncate
from financevault.verify.signals import Signal, na

SYSTEM = """You check whether cited evidence supports a claim in a financial answer.

Score 1.0 when every figure and assertion in the answer is directly stated in the evidence.
Score around 0.5 when the evidence is related but does not actually state the claim — for
example it covers the right company but a different period, or supports part of the answer.
Score 0.0 when the evidence does not support the answer at all.

Judge only whether the evidence supports the answer. Do not judge whether the answer is
independently true, well written, or complete."""


def retrieved_refs(steps: list[dict]) -> set[str]:
    """Every source identifier the trajectory actually saw: XBRL tags and chunk ids."""
    refs: set[str] = set()
    for step in steps:
        obs = step.get("obs") or step.get("observation") or {}
        if step.get("tool") in (None, "finish") or not obs.get("ok"):
            continue
        data = obs.get("data")
        rows = data if isinstance(data, list) else (data or {}).get("rows", [])
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            if tag := row.get("tag"):
                refs.add(str(tag))
            if chunk_id := row.get("chunk_id"):
                refs.add(str(chunk_id))
            if accession := row.get("accession"):
                refs.add(str(accession))
    return refs


def score(
    answer: str,
    citations: list[dict],
    steps: list[dict],
    *,
    ledger: Ledger,
    evidence_text: str = "",
) -> Signal:
    """Score whether an answer's citations are present, real, and supporting."""
    if not answer:
        return na("no answer to check")

    if not citations:
        return Signal(0.0, "answer cites no sources")

    # Structural check first -- it is free, and it catches the outright fabrications.
    seen = retrieved_refs(steps)
    if seen:
        unresolved = [c.get("ref") for c in citations if c.get("ref") and str(c["ref"]) not in seen]
        if unresolved:
            return Signal(
                0.0,
                f"cites {len(unresolved)} source(s) the trajectory never retrieved: "
                f"{', '.join(str(r) for r in unresolved[:3])}",
            )

    if not evidence_text.strip():
        # Citations resolve, but there is no text to check them against -- structurally sound
        # and semantically unverified. Scored as partial rather than passed.
        return Signal(0.5, "citations resolve to retrieved sources, but no text to verify against")

    return ask(
        SYSTEM,
        f"ANSWER:\n{truncate(answer, 1200)}\n\nEVIDENCE:\n{truncate(evidence_text)}",
        ledger=ledger,
        label="s2-citation",
    )
