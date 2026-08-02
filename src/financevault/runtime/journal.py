"""The content-addressed journal.

Every model call is keyed on a hash of everything that determines its output: model, full
message list, tool specifications, and sampling parameters. An identical request always
resolves to the same row, so a past run replays offline, byte-for-byte, at zero token cost.

Two properties are load-bearing and easy to lose by accident:

**The key must cover everything that changes the output, and nothing that does not.** Include
temperature and the tool schemas; exclude wall-clock time, request ids and retry counts.
A key that is too narrow returns wrong cached answers. A key that is too wide never hits.

**Serialization must be canonical.** Dict ordering, float formatting and unicode escaping all
have to be pinned, or the same logical request hashes two different ways on two machines and
the F1 determinism metric quietly collapses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from financevault.store import pg

# Bumped when the hashing scheme changes, so old entries cannot be mistaken for current ones.
SCHEME = "v1"


class ReplayMiss(RuntimeError):
    """A request was not in the journal while replay was required.

    Deliberately fatal. Falling back to a live call would make a "free" replay quietly cost
    money and, worse, would make a determinism measurement report success on fresh output.
    """


# Determinism note. Sampling parameters were removed from current models, so a request cannot
# be pinned to temperature 0 and two live calls with identical inputs may differ. F1 therefore
# measures what it can actually guarantee: that the *key* is stable, so a replayed run resolves
# to the same journal rows and reproduces byte-for-byte. It is not a claim about model
# reproducibility, and the report must not present it as one.


def canonical(payload: Any) -> str:
    """Stable JSON: sorted keys, no incidental whitespace, escaped non-ASCII."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def key(
    model: str,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    system: str | None = None,
    thinking: dict | None = None,
    effort: str | None = None,
    max_tokens: int = 1024,
    extra: dict | None = None,
) -> str:
    """The content address of a request.

    Note there is no `temperature`: current models reject sampling parameters outright, so
    the key covers `thinking` and `effort` instead -- the knobs that actually steer output now.

    `extra` carries anything else that steers generation. It is a named parameter rather than
    free kwargs so that adding a knob without threading it into the key is a visible omission
    rather than an invisible one.
    """
    request = {
        "scheme": SCHEME,
        "model": model,
        "system": system,
        "messages": messages,
        "tools": tools or [],
        "thinking": thinking,
        "effort": effort,
        "max_tokens": max_tokens,
        "extra": extra or {},
    }
    return hashlib.sha256(canonical(request).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class Entry:
    hash: str
    model: str
    request: dict
    response: dict
    tokens_in: int
    tokens_out: int
    cost_usd: float


def get(hash_: str) -> Entry | None:
    row = pg.fetch_one(
        "SELECT hash, model, request, response, tokens_in, tokens_out, cost_usd "
        "FROM journal WHERE hash = %s",
        (hash_,),
    )
    if row is None:
        return None
    return Entry(
        hash=row["hash"],
        model=row["model"],
        request=row["request"],
        response=row["response"],
        tokens_in=row["tokens_in"] or 0,
        tokens_out=row["tokens_out"] or 0,
        cost_usd=float(row["cost_usd"] or 0.0),
    )


PUT = """
INSERT INTO journal (hash, model, request, response, tokens_in, tokens_out, cost_usd)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (hash) DO NOTHING
"""


def put(
    hash_: str,
    model: str,
    request: dict,
    response: dict,
    *,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
) -> None:
    """Record a call. First write wins.

    `DO NOTHING` rather than `DO UPDATE`: if the same request produced a different response,
    that is a determinism failure worth detecting later, not something to overwrite away.
    """
    pg.execute(
        PUT,
        (
            hash_,
            model,
            json.dumps(request, sort_keys=True),
            json.dumps(response, sort_keys=True),
            tokens_in,
            tokens_out,
            cost_usd,
        ),
    )


def stats() -> dict:
    row = pg.fetch_one(
        "SELECT count(*) AS entries, coalesce(sum(cost_usd), 0) AS usd, "
        "coalesce(sum(tokens_in), 0) AS tokens_in, coalesce(sum(tokens_out), 0) AS tokens_out "
        "FROM journal"
    )
    return dict(row) if row else {}
