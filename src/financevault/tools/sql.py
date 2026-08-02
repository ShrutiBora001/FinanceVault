"""The SQL tool: read-only aggregate queries over as-of-filtered views.

Deviation from the plan, recorded deliberately: the plan specified DuckDB. This runs against
Postgres instead. DuckDB would need either the postgres_scanner extension (a runtime download
that makes the tool network-dependent) or a full table copy per call. Postgres already holds
the data and can enforce the point-in-time horizon in the database, which no in-process engine
can do for agent-written SQL.

**The horizon is enforced in the database, not in Python.** The tool exposes only the `v_*`
views, which filter on `current_setting('fv.as_of')`. That call raises when the variable is
unset, so a query that somehow escapes the wrapper fails closed rather than returning the
whole corpus. This matters more here than anywhere else in the system: the agent writes the
SQL, so a Python-side filter would be advisory at best.

Remaining layers are for error quality, not safety: a statement allowlist and a view
allowlist so the agent gets an actionable message instead of a Postgres error it cannot fix.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal

import psycopg
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from financevault.config import settings
from financevault.tools.registry import REGISTRY, Tool, ToolContext, ToolResult

# Views only. The base tables are unreachable, as are `runs`, `steps` and `journal` -- an
# agent that could read `runs` could read another run's answer.
ALLOWED_TABLES = {"v_xbrl_facts", "v_prices", "v_filings", "v_chunks"}

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|vacuum|"
    r"reindex|refresh|call|do|set|begin|commit|rollback|current_setting|set_config)\b",
    re.IGNORECASE,
)
TABLE_REF = re.compile(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", re.IGNORECASE)
CTE_NAME = re.compile(r"(?:with|,)\s+([a-z_][a-z0-9_]*)\s+as\s*\(", re.IGNORECASE)

MAX_ROWS = 200
TIMEOUT_MS = 5000

SCHEMA_HINT = (
    "v_xbrl_facts(cik, tag, unit, value, fy, fp, period_start, period_end, form, accession), "
    "v_prices(ticker, date, open, high, low, close, volume), "
    "v_filings(cik, ticker, form, period_end, accession, url), "
    "v_chunks(filing_id, section, idx, text, cik, form, period_end)"
)


class SqlInput(BaseModel):
    query: str = Field(
        description=(
            f"A single read-only SELECT over these views: {SCHEMA_HINT}. "
            "The views are already restricted to information available at the question's "
            "as-of date; do not add a date filter for that purpose."
        )
    )


def reject(query: str) -> str | None:
    """Static checks. Returns a rejection reason, or None if the query may be attempted."""
    stripped = query.strip().rstrip(";")
    if not stripped:
        return "empty query"
    if ";" in stripped:
        return "only a single statement is allowed"
    if not re.match(r"^\s*(with|select)\b", stripped, re.IGNORECASE):
        return "query must be a SELECT"
    if FORBIDDEN.search(stripped):
        return "only read-only SELECT queries are allowed"

    referenced = {t.lower() for t in TABLE_REF.findall(stripped)}
    cte_names = {n.lower() for n in CTE_NAME.findall(stripped)}
    unknown = referenced - ALLOWED_TABLES - cte_names
    if unknown:
        return (
            f"unknown table(s): {', '.join(sorted(unknown))}. "
            f"Available views: {', '.join(sorted(ALLOWED_TABLES))}"
        )
    return None


def _jsonable(value: object) -> object:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    return value


def sql_handler(args: SqlInput, ctx: ToolContext) -> ToolResult:
    reason = reject(args.query)
    if reason:
        return ToolResult(ok=False, error=reason)

    query = args.query.strip().rstrip(";")

    try:
        with psycopg.connect(settings().pg_dsn) as conn:
            conn.read_only = True
            with conn.cursor(row_factory=dict_row) as cur:
                # The horizon, passed as a parameter so it cannot be injected through it.
                cur.execute("SELECT set_config('fv.as_of', %s, true)", (str(ctx.as_of),))
                cur.execute(f"SET LOCAL statement_timeout = {TIMEOUT_MS}")
                cur.execute(f"SELECT * FROM ({query}) AS agent_query LIMIT {MAX_ROWS}")  # noqa: S608
                rows = cur.fetchall()
    except psycopg.errors.QueryCanceled:
        return ToolResult(ok=False, error=f"query exceeded {TIMEOUT_MS}ms")
    except psycopg.Error as exc:
        return ToolResult(ok=False, error=f"{type(exc).__name__}: {str(exc).strip()}")

    return ToolResult(
        ok=True,
        data={
            "rows": [{k: _jsonable(v) for k, v in row.items()} for row in rows],
            "row_count": len(rows),
            "truncated": len(rows) == MAX_ROWS,
        },
    )


REGISTRY.register(
    Tool(
        name="sql",
        description=(
            "Run a read-only SELECT for aggregates, comparisons and time series. Results "
            "are capped at 200 rows. Use lookup_fact for a single reported figure; use this "
            "when you need to group, join or aggregate across periods or companies."
        ),
        input_model=SqlInput,
        handler=sql_handler,
    )
)
