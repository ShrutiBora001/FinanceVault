"""Cross-encoder reranking over fused retrieval candidates.

Bi-encoder retrieval embeds the query and the passage separately, so it never sees them
together and cannot tell "net income was $112,010 million" from "the Company designs and
markets smartphones" beyond topical similarity. A cross-encoder scores the *pair*, which is
much sharper — on a spot check the relevant passage scored +7.0 against −11.3 for two
irrelevant ones.

The cost is that it cannot be precomputed: every candidate is a forward pass at query time.
So it runs as a second stage over a shortlist the cheap retriever already narrowed, never over
the corpus.

Scores are unbounded logits, not probabilities. They are comparable *within* one query's
candidate list and meaningless across queries, so they are used to order and to threshold
relative to the best candidate — never compared to a fixed constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_LENGTH = 512

# A candidate is dropped when it scores this far below the best one. Relative, because the
# scale shifts per query: an absolute floor would keep everything on an easy query and drop
# everything on a hard one.
RELATIVE_FLOOR = 8.0


@dataclass(frozen=True, slots=True)
class Scored:
    index: int
    score: float


@lru_cache(maxsize=1)
def model() -> CrossEncoder:
    from sentence_transformers import CrossEncoder  # noqa: PLC0415 - heavy optional import

    return CrossEncoder(MODEL, max_length=MAX_LENGTH)


def score(query: str, passages: list[str]) -> list[Scored]:
    """Score each passage against the query, best first."""
    if not passages:
        return []
    raw = model().predict([(query, p) for p in passages])
    ranked = [Scored(index=i, score=float(s)) for i, s in enumerate(raw)]
    ranked.sort(key=lambda s: s.score, reverse=True)
    return ranked


def rerank(
    query: str,
    rows: list[dict],
    *,
    top_k: int,
    text_key: str = "text",
    drop_weak: bool = True,
) -> list[dict]:
    """Reorder candidate rows by cross-encoder relevance and trim to `top_k`.

    `drop_weak` removes candidates far below the best one rather than padding the result to
    `top_k` with noise. Returning fewer, better passages is the right trade here: an agent
    that reads three relevant chunks beats one that reads three relevant and five misleading,
    and every extra passage costs input tokens on the next call.
    """
    if not rows:
        return []

    ranked = score(query, [str(r.get(text_key, "")) for r in rows])
    if not ranked:
        return rows[:top_k]

    best = ranked[0].score
    out: list[dict] = []
    for item in ranked[:top_k]:
        if drop_weak and (best - item.score) > RELATIVE_FLOOR:
            break
        row = dict(rows[item.index])
        row["rerank_score"] = round(item.score, 4)
        out.append(row)

    # Never return nothing: if every candidate looked weak, the top one is still the answer to
    # "which of these is most relevant", and an empty result tells the agent less than a bad one.
    return out or [{**rows[ranked[0].index], "rerank_score": round(best, 4)}]
