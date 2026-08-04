"""Chunking and embedding of filing text.

**Chunks never cross an Item boundary.** A 10-K is a fixed set of numbered Items, and they are
about genuinely different things — Item 1A is risk factors, Item 7 is management's discussion,
Item 8 is the financial statements. A window that straddles the end of one and the start of
another produces a passage about two unrelated topics, which retrieves badly and reads worse
when cited. So the document is split on Item headings first and windowed within each section.

The section label is then trustworthy rather than approximate, which matters beyond retrieval:
a citation can say *which part of the filing* a claim came from, and MVP2.1's retrieval can
filter to a section when the question implies one.

Windowing inside a section is still fixed-width with overlap. Paragraph- or sentence-aware
splitting is a further refinement and is not obviously worth it while the section boundary is
doing the heavy lifting.
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

# Item headings in 10-K and 10-Q filings.
#
# Anchored to line start because "item 7" appears in running prose ("as discussed in Item 7")
# far more often than it starts a section, and a false boundary there would fragment the very
# section being referred to.
#
# The trailing group is `.*`, deliberately unbounded. An earlier `.{0,80}$` silently dropped
# Item 7 from every Apple 10-K, because the real heading reads "Item 7. Management's Discussion
# and Analysis of Financial Condition and Results of Operations" -- past 80 characters, so the
# line-end anchor never matched and MD&A was absorbed into Item 6. MD&A is one of the most
# asked-about sections, and the failure was invisible: chunks still existed, just mislabelled.
ITEM_RE = re.compile(
    r"^\s*(item\s+(\d{1,2}[a-z]?))\s*[.:–—-]?\s*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)

# Item numbering is form-specific and the two forms disagree on almost everything. Item 2 is
# "Properties" in a 10-K and "Management's Discussion and Analysis" in a 10-Q; Item 1 is
# "Business" against "Financial Statements". Using the 10-K map for both mislabelled every
# 10-Q -- and the symptom was subtle, because "properties" simply looked implausibly large
# rather than obviously wrong.
ITEM_NAMES_10K = {
    "1": "business",
    "1a": "risk factors",
    "1b": "unresolved staff comments",
    "1c": "cybersecurity",
    "2": "properties",
    "3": "legal proceedings",
    "4": "mine safety disclosures",
    "5": "market for registrant's equity",
    "7": "management's discussion and analysis",
    "7a": "market risk",
    "8": "financial statements",
    "9a": "controls and procedures",
}

ITEM_NAMES_10Q = {
    "1": "financial statements",
    "1a": "risk factors",
    "2": "management's discussion and analysis",
    "3": "market risk",
    "4": "controls and procedures",
    "5": "other information",
    "6": "exhibits",
}


def item_names(form: str | None) -> dict[str, str]:
    return ITEM_NAMES_10Q if (form or "").upper().startswith("10-Q") else ITEM_NAMES_10K


WORDS_PER_CHUNK = 350
OVERLAP_WORDS = 50
MIN_SECTION_WORDS = 30


@dataclass(frozen=True, slots=True)
class Chunk:
    idx: int
    section: str | None
    text: str
    n_tokens: int


@dataclass(frozen=True, slots=True)
class Section:
    key: str | None
    label: str | None
    text: str


@lru_cache(maxsize=1)
def embedder() -> SentenceTransformer:
    """Loaded once per process; the model is ~130MB and CPU inference is fast enough."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415 - heavy optional import

    return SentenceTransformer(settings().embed_model)


def split_sections(text: str, form: str | None = None) -> list[Section]:
    """Cut the document at Item headings.

    Text before the first heading — cover page, table of contents — is kept as an unlabelled
    section rather than discarded: it carries the filing date and entity details, and dropping
    it would silently lose them.
    """
    matches = list(ITEM_RE.finditer(text))
    if not matches:
        return [Section(key=None, label=None, text=text)]

    sections: list[Section] = []
    preamble = text[: matches[0].start()].strip()
    if len(preamble.split()) >= MIN_SECTION_WORDS:
        sections.append(Section(key=None, label="front matter", text=preamble))

    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.start() : end].strip()
        if len(body.split()) < MIN_SECTION_WORDS:
            # A heading with nothing under it is a table-of-contents entry, not a section.
            continue
        key = match.group(2).lower()
        names = item_names(form)
        sections.append(Section(key=key, label=names.get(key, f"item {key}"), text=body))
    return sections or [Section(key=None, label=None, text=text)]


def _window(words: list[str]) -> list[list[str]]:
    step = WORDS_PER_CHUNK - OVERLAP_WORDS
    out: list[list[str]] = []
    for start in range(0, len(words), step):
        window = words[start : start + WORDS_PER_CHUNK]
        if len(window) < 20 and out:
            break  # trailing scrap, already covered by the previous window's overlap
        out.append(window)
    return out


def split(text: str, form: str | None = None) -> list[Chunk]:
    """Section-aware chunks: windowed within a section, never across one.

    `form` selects the item-number vocabulary. Omitting it falls back to the 10-K map, which
    is wrong for a 10-Q -- pass it.
    """
    chunks: list[Chunk] = []
    for section in split_sections(text, form):
        words = section.text.split()
        if not words:
            continue
        for window in _window(words):
            chunks.append(
                Chunk(
                    idx=len(chunks),
                    section=section.label,
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
