"""Retrieval over filing text, and XBRL fact lookup.

Both are as-of filtered in SQL, and both re-check results through `AsOf.check` before
returning. The horizon comes from the run context, never from tool arguments.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from financevault.store import pg
from financevault.tools.registry import REGISTRY, Tool, ToolContext, ToolResult

# ---------------------------------------------------------------- retrieve_filings


class RetrieveInput(BaseModel):
    query: str = Field(description="Natural-language description of the passage to find.")
    k: int = Field(default=5, ge=1, le=20, description="Number of passages to return.")
    form: str | None = Field(
        default=None, description="Restrict to a filing form, e.g. '10-K' or '10-Q'."
    )
    section: str | None = Field(
        default=None,
        description=(
            "Restrict to a filing section. Useful values: 'risk factors', "
            "'management's discussion and analysis', 'financial statements', 'business', "
            "'legal proceedings'. Omit unless the question clearly implies one."
        ),
    )


# How many candidates the cheap retriever hands to the reranker. Wider than `k` because the
# reranker's job is to reorder, and it can only promote a passage that reached the shortlist.
RERANK_CANDIDATES = 30


# Hybrid retrieval: vector similarity and full-text rank, fused by reciprocal rank.
# RRF is used rather than a weighted score sum because the two scores are on
# incomparable scales, and tuning a weight before measuring quality would be guesswork.
RETRIEVE_SQL = """
WITH vec AS (
    SELECT c.id, row_number() OVER (ORDER BY c.embedding <=> %(embedding)s) AS rank
    FROM chunks c JOIN filings f ON f.id = c.filing_id
    WHERE {asof} {form_filter} {section_filter}
    ORDER BY c.embedding <=> %(embedding)s
    LIMIT 50
),
fts AS (
    SELECT c.id,
           row_number() OVER (
               ORDER BY ts_rank(c.tsv, plainto_tsquery('english', %(query)s)) DESC
           ) AS rank
    FROM chunks c JOIN filings f ON f.id = c.filing_id
    WHERE c.tsv @@ plainto_tsquery('english', %(query)s) AND {asof} {form_filter} {section_filter}
    LIMIT 50
),
fused AS (
    SELECT id, sum(score) AS score FROM (
        SELECT id, 1.0 / (60 + rank) AS score FROM vec
        UNION ALL
        SELECT id, 1.0 / (60 + rank) AS score FROM fts
    ) s GROUP BY id
)
SELECT c.id, c.section, c.idx, c.text, f.accession, f.form, f.period_end,
       f.accepted_at, f.url, fused.score
FROM fused
JOIN chunks c ON c.id = fused.id
JOIN filings f ON f.id = c.filing_id
-- c.id is the tie-break: reciprocal-rank fusion produces exact ties routinely, and an
-- unordered tie makes the retrieved set differ between identical runs.
ORDER BY fused.score DESC, c.id
LIMIT %(candidates)s
"""


def retrieve_handler(args: RetrieveInput, ctx: ToolContext) -> ToolResult:
    from financevault.env import coverage, rerank  # noqa: PLC0415 - defers the torch import
    from financevault.env.chunk import embed_query  # noqa: PLC0415

    sql = RETRIEVE_SQL.format(
        asof=ctx.as_of.sql("filings", alias="f"),
        form_filter="AND f.form = %(form)s" if args.form else "",
        section_filter="AND c.section = %(section)s" if args.section else "",
    )

    params: dict[str, Any] = {
        "embedding": str(embed_query(args.query).tolist()),
        "query": args.query,
        "candidates": RERANK_CANDIDATES,
        **ctx.as_of.params(),
    }
    if args.form:
        params["form"] = args.form
    if args.section:
        params["section"] = args.section

    rows = ctx.as_of.check(pg.fetch_all(sql, params))
    if not rows:
        # An empty result has two very different causes and the agent cannot tell them apart
        # from silence. If no documents exist at this horizon at all, no rephrasing will help
        # and saying so redirects the run to XBRL instead of spending its budget here.
        cover = coverage.at(ctx.as_of, cik=ctx.cik)
        if cover.documents_empty:
            return ToolResult(ok=False, error=cover.advice())
        if args.section:
            return ToolResult(
                ok=False,
                error=(
                    f"no passages in section {args.section!r}. Available sections: "
                    f"{', '.join(available_sections(ctx))}"
                ),
            )

    candidates = [
        {
            "chunk_id": r["id"],
            "text": r["text"],
            "section": r["section"],
            "form": r["form"],
            "period_end": str(r["period_end"]) if r["period_end"] else None,
            "accession": r["accession"],
            "accepted_at": r["accepted_at"].isoformat(),
            "retrieval_score": round(float(r["score"]), 6),
        }
        for r in rows
    ]
    # Two stages: cheap hybrid retrieval narrows the corpus, the cross-encoder orders what
    # survives. The reranker can only promote what reached the shortlist, so the shortlist is
    # deliberately wider than k.
    return ToolResult(ok=True, data=rerank.rerank(args.query, candidates, top_k=args.k))


def available_sections(ctx: ToolContext, limit: int = 8) -> list[str]:
    """Sections that actually exist for this company, so a bad filter is recoverable."""
    rows = pg.fetch_all(
        """
        SELECT DISTINCT c.section FROM chunks c JOIN filings f ON f.id = c.filing_id
        WHERE f.cik = %(cik)s AND c.section IS NOT NULL
        ORDER BY c.section
        LIMIT %(limit)s
        """,
        {"cik": ctx.cik, "limit": limit},
    )
    return [r["section"] for r in rows]


REGISTRY.register(
    Tool(
        name="retrieve_filings",
        description=(
            "Search the text of SEC filings for passages relevant to a question. Returns "
            "passages with their source filing and a chunk_id to cite. Use this for "
            "narrative or qualitative questions; use lookup_fact for reported numbers."
        ),
        input_model=RetrieveInput,
        handler=retrieve_handler,
    )
)


# ---------------------------------------------------------------- lookup_fact


class LookupInput(BaseModel):
    tag: str = Field(
        description=(
            "US-GAAP tag, e.g. 'Revenues', 'NetIncomeLoss', 'Assets', "
            "'RevenueFromContractWithCustomerExcludingAssessedTax'."
        )
    )
    period_end: date | None = Field(
        default=None, description="Fiscal period end (YYYY-MM-DD). Omit for the latest."
    )
    unit: str = Field(default="USD", description="Unit, e.g. 'USD', 'shares', 'USD/shares'.")
    fp: str | None = Field(default=None, description="Fiscal period: 'FY', 'Q1', 'Q2', 'Q3'.")


LOOKUP_SQL = """
SELECT tag, unit, value, fy, fp, period_start, period_end, form, accession, accepted_at
FROM xbrl_facts
WHERE cik = %(cik)s AND tag = %(tag)s AND unit = %(unit)s
  AND {asof}
  {period_filter}
  {fp_filter}
ORDER BY period_end DESC, accepted_at DESC, accession
LIMIT 20
"""


def lookup_handler(args: LookupInput, ctx: ToolContext) -> ToolResult:
    if not ctx.cik:
        return ToolResult(ok=False, error="no company in context")

    sql = LOOKUP_SQL.format(
        asof=ctx.as_of.sql("xbrl_facts"),
        period_filter="AND period_end = %(period_end)s" if args.period_end else "",
        fp_filter="AND fp = %(fp)s" if args.fp else "",
    )
    params = {
        "cik": ctx.cik,
        "tag": args.tag,
        "unit": args.unit,
        **ctx.as_of.params(),
    }
    if args.period_end:
        params["period_end"] = args.period_end
    if args.fp:
        params["fp"] = args.fp

    rows = ctx.as_of.check(pg.fetch_all(sql, params))
    if not rows:
        return ToolResult(
            ok=False,
            error=(
                f"no fact for tag={args.tag!r} unit={args.unit!r} knowable at {ctx.as_of}. "
                f"Try a different tag; similar available tags: "
                f"{', '.join(_similar_tags(ctx, args.tag))}"
            ),
        )

    # Restatements: the same period reported twice. Ordering puts the most recently accepted
    # first, so the newest statement knowable at the horizon wins.
    return ToolResult(
        ok=True,
        data=[
            {
                "tag": r["tag"],
                "value": float(r["value"]),
                "unit": r["unit"],
                "fy": r["fy"],
                "fp": r["fp"],
                "period_start": str(r["period_start"]) if r["period_start"] else None,
                "period_end": str(r["period_end"]) if r["period_end"] else None,
                "form": r["form"],
                "accession": r["accession"],
                "accepted_at": r["accepted_at"].isoformat(),
            }
            for r in rows
        ],
    )


def _similar_tags(ctx: ToolContext, tag: str, limit: int = 5) -> list[str]:
    """Nearest available tags by trigram similarity, to make a miss recoverable."""
    # DISTINCT and ORDER BY similarity() cannot coexist in one SELECT -- the sort expression
    # is not in the select list -- so the dedupe happens in a subquery.
    rows = pg.fetch_all(
        """
        SELECT t.tag FROM (
            SELECT DISTINCT tag FROM xbrl_facts
            WHERE cik = %(cik)s AND tag %% %(tag)s
        ) t
        -- tag is the tie-break. Without it, equally-similar tags come back in whatever
        -- order the planner chose, the error message differs between runs, and the next
        -- prompt differs -- which changes the journal key and breaks replay determinism.
        ORDER BY similarity(t.tag, %(tag)s) DESC, t.tag
        LIMIT %(limit)s
        """,
        {"cik": ctx.cik, "tag": tag, "limit": limit},
    )
    return [r["tag"] for r in rows]


REGISTRY.register(
    Tool(
        name="lookup_fact",
        description=(
            "Look up a reported financial figure by its US-GAAP tag. Returns the exact "
            "value as filed, with its unit, fiscal period and source accession. Prefer this "
            "over retrieve_filings for any number that appears in the financial statements."
        ),
        input_model=LookupInput,
        handler=lookup_handler,
    )
)
