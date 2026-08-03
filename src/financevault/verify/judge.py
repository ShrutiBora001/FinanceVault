"""The shared LLM-judge helper for `s2` and `s4`.

Only two of the five signals are judged by a model, and both are the soft ones — whether a
citation supports a claim, whether a retrieval advanced the subgoal. The signals that decide
whether a trajectory is *correct* (`s1` tool validity, `s3` numeric correctness) stay
programmatic, because a judge asked "is this plausible" answers yes.

Judging runs on every step of every trajectory, so it is the dominant verification cost.
Three things keep it cheap: the cheapest model, a hard cap on output tokens, and evidence
truncated before it reaches the prompt. Every call is journalled like any other, so
re-verifying an already-judged trajectory is free.
"""

from __future__ import annotations

import re

from financevault.config import settings
from financevault.runtime import llm
from financevault.runtime.budget import Ledger
from financevault.verify.signals import Signal

# The judge answers in a fixed two-line shape. A parse failure is scored 0.5 with the raw
# reply as the reason, so a malformed judgement is visible in calibration rather than being
# silently counted as a pass or a fail.
PROTOCOL = """Reply in exactly this form, nothing else:

SCORE: <a number from 0.0 to 1.0>
REASON: <one short sentence>"""

SCORE_RE = re.compile(r"SCORE:\s*([01](?:\.\d+)?|\.\d+)", re.IGNORECASE)
REASON_RE = re.compile(r"REASON:\s*(.+)", re.IGNORECASE)

MAX_EVIDENCE_CHARS = 2000


def truncate(text: str, limit: int = MAX_EVIDENCE_CHARS) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + f"… [+{len(text) - limit} chars]"


def ask(system: str, prompt: str, *, ledger: Ledger, label: str) -> Signal:
    """One judged signal. Never raises: a judge failure degrades the signal, not the run."""
    try:
        completion = llm.call(
            settings().judge_model,
            [{"role": "user", "content": prompt}],
            ledger=ledger,
            system=f"{system}\n\n{PROTOCOL}",
            thinking=False,
            max_tokens=120,
            label=label,
        )
    except Exception as exc:  # noqa: BLE001 - a judge outage must not abort verification
        return Signal(0.5, f"judge unavailable ({type(exc).__name__}); scored neutral")

    text = completion.text.strip()
    score_match = SCORE_RE.search(text)
    reason_match = REASON_RE.search(text)

    if score_match is None:
        return Signal(0.5, f"unparseable judgement: {text[:120]!r}")

    score = min(1.0, max(0.0, float(score_match.group(1))))
    reason = reason_match.group(1).strip() if reason_match else "no reason given"
    return Signal(score, reason)
