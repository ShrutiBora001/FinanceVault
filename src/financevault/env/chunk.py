"""Chunking and embedding of filing text.

MVP1 uses word-window chunking with overlap. Section-aware chunking over 10-K/10-Q item
structure is deliberately deferred: it is a retrieval-quality improvement, and quality is
measured before it is tuned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from financevault.config import settings
from financevault.store import pg

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# 10-K/10-Q item headings, used to label chunks with the section they fall in.
ITEM_RE = re.compile(r"^\s*(item\s+\d+[a-z]?)\s*[.:\-—]", re.IGNORECASE | re.MULTILINE)

WORDS_PER_CHUNK = 350
OVERLAP_WORDS = 50


@dataclass(frozen=True, slots=True)
class Chunk:
    idx: int
    section: str | None
    text: str
    n_tokens: int


@lru_cache(maxsize=1)
def embedder() -> SentenceTransformer:
    """Loaded once per process; the model is ~130MB and CPU inference is fast enough."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415 - heavy optional import

    return SentenceTransformer(settings().embed_model)


def _section_at(text: str, position: int) -> str | None:
    """The most recent item heading at or before `position`."""
    section = None
    for match in ITEM_RE.finditer(text):
        if match.start() > position:
            break
        section = match.group(1).strip().lower()
    return section


def split(text: str) -> list[Chunk]:
    """Overlapping word windows, each labelled with the item section it starts in."""
    words = text.split()
    if not words:
        return []

    # Offset of each word in the original text, so a chunk can be mapped back to a section.
    offsets: list[int] = []
    cursor = 0
    for word in words:
        cursor = text.find(word, cursor)
        offsets.append(cursor)
        cursor += len(word)

    chunks: list[Chunk] = []
    step = WORDS_PER_CHUNK - OVERLAP_WORDS
    for idx, start in enumerate(range(0, len(words), step)):
        window = words[start : start + WORDS_PER_CHUNK]
        if len(window) < 20 and chunks:
            break  # trailing scrap, already covered by the previous window's overlap
        chunks.append(
            Chunk(
                idx=idx,
                section=_section_at(text, offsets[start]),
                text=" ".join(window),
                n_tokens=len(window),
            )
        )
    return chunks


INSERT_CHUNK = """
INSERT INTO chunks (filing_id, section, idx, text, n_tokens, embedding)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (filing_id, idx) DO UPDATE SET
    section = EXCLUDED.section,
    text = EXCLUDED.text,
    n_tokens = EXCLUDED.n_tokens,
    embedding = EXCLUDED.embedding
"""


def store_chunks(filing_id: int, chunks: list[Chunk]) -> int:
    """Embed and persist. Idempotent on (filing_id, idx)."""
    if not chunks:
        return 0
    vectors = embedder().encode(
        [c.text for c in chunks],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    rows = [
        (filing_id, c.section, c.idx, c.text, c.n_tokens, vector)
        for c, vector in zip(chunks, vectors, strict=True)
    ]
    return pg.execute_many(INSERT_CHUNK, rows)


def embed_query(text: str) -> list[float]:
    """Embed a single query with the same normalization used at index time."""
    return embedder().encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
