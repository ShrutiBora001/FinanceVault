"""The tool layer, with emphasis on the two components that execute agent-authored input.

The SQL tool runs agent-written queries and the Python tool runs agent-written code. Both had
no tests, which is the wrong place to have none. The point-in-time guarantee in particular is
enforced by SQL views rather than by Python, so a regression there would be invisible to every
other test in the suite.
"""

from __future__ import annotations

import pytest

from financevault.env.asof import AsOf
from financevault.runtime.budget import Ledger
from financevault.store import pg
from financevault.tools import REGISTRY, ToolContext
from financevault.tools.python_sandbox import static_check
from financevault.tools.sql import reject

CIK_AAPL = "320193"


@pytest.fixture(scope="module")
def seeded() -> bool:
    if pg.missing_tables():
        pytest.skip("schema not applied; run `make migrate`")
    if (pg.fetch_value("SELECT count(*) FROM xbrl_facts") or 0) == 0:
        pytest.skip("no facts ingested; run `make ingest`")
    return True


@pytest.fixture
def ctx() -> ToolContext:
    return ToolContext(
        as_of=AsOf.parse("2026-08-01"),
        ledger=Ledger(max_usd=1.0, max_steps=20),
        cik=CIK_AAPL,
        ticker="AAPL",
    )


# ---------------------------------------------------------------- registry contracts


def test_every_registered_tool_produces_a_valid_spec() -> None:
    for name in REGISTRY.names():
        spec = REGISTRY[name].spec()
        assert spec["name"] == name
        assert spec["description"].strip()
        assert spec["input_schema"]["type"] == "object"


def test_exactly_one_tool_is_terminal() -> None:
    """More than one way to end a run would make trajectory parsing ambiguous."""
    terminal = [n for n in REGISTRY.names() if REGISTRY[n].terminal]
    assert terminal == ["finish"]


def test_validation_rejects_unknown_tools() -> None:
    parsed, error = REGISTRY.validate("no_such_tool", {})
    assert parsed is None and "unknown tool" in error


def test_validation_is_separate_from_execution() -> None:
    """`s1` scores calls without running them, so validate must not have side effects."""
    parsed, error = REGISTRY.validate("lookup_fact", {"tag": "NetIncomeLoss"})
    assert error is None and parsed is not None


def test_a_handler_exception_becomes_a_result_not_a_crash(ctx: ToolContext) -> None:
    """Tool failures are observations the agent can recover from."""
    result = REGISTRY.dispatch(
        "sql", {"query": "SELECT * FROM v_xbrl_facts WHERE bad_col = 1"}, ctx
    )
    assert result.ok is False and result.error


# ---------------------------------------------------------------- sql: static guard


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM xbrl_facts",
        "UPDATE v_xbrl_facts SET value = 0",
        "INSERT INTO prices VALUES (1)",
        "DROP TABLE filings",
        "SELECT 1; DROP TABLE filings",
        "GRANT ALL ON filings TO PUBLIC",
        "SELECT set_config('fv.as_of', '2099-01-01', true)",
        "SELECT current_setting('fv.as_of')",
    ],
)
def test_write_and_horizon_tampering_queries_are_rejected(query: str) -> None:
    assert reject(query) is not None, query


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM xbrl_facts",
        "SELECT * FROM runs",
        "SELECT * FROM steps",
        "SELECT * FROM journal",
    ],
)
def test_base_and_trajectory_tables_are_unreachable(query: str) -> None:
    """An agent that could read `runs` could read another run's answer."""
    assert reject(query) is not None, query


def test_the_as_of_views_are_allowed() -> None:
    assert reject("SELECT tag, value FROM v_xbrl_facts LIMIT 5") is None


def test_a_cte_over_an_allowed_view_is_allowed() -> None:
    query = "WITH x AS (SELECT value FROM v_xbrl_facts) SELECT count(*) FROM x"
    assert reject(query) is None


def test_empty_query_is_rejected() -> None:
    assert reject("   ") is not None


# ---------------------------------------------------------------- sql: the horizon


def test_sql_cannot_see_past_the_horizon(seeded: bool) -> None:
    """The guarantee that Python cannot enforce, because the agent writes the query."""
    early = ToolContext(
        as_of=AsOf.parse("2020-01-01"),
        ledger=Ledger(max_usd=1.0, max_steps=5),
        cik=CIK_AAPL,
    )
    result = REGISTRY.dispatch(
        "sql", {"query": "SELECT max(accepted_at) AS newest FROM v_xbrl_facts"}, early
    )
    assert result.ok
    newest = result.data["rows"][0]["newest"]
    assert newest is None or newest <= "2020-01-01"


def test_a_later_horizon_sees_more(seeded: bool, ctx: ToolContext) -> None:
    late = REGISTRY.dispatch("sql", {"query": "SELECT count(*) AS n FROM v_xbrl_facts"}, ctx)
    early_ctx = ToolContext(
        as_of=AsOf.parse("2020-01-01"), ledger=Ledger(max_usd=1.0, max_steps=5), cik=CIK_AAPL
    )
    early = REGISTRY.dispatch("sql", {"query": "SELECT count(*) AS n FROM v_xbrl_facts"}, early_ctx)
    assert late.data["rows"][0]["n"] > early.data["rows"][0]["n"]


def test_results_are_capped(seeded: bool, ctx: ToolContext) -> None:
    result = REGISTRY.dispatch("sql", {"query": "SELECT value FROM v_xbrl_facts"}, ctx)
    assert result.ok and result.data["row_count"] <= 200


# ---------------------------------------------------------------- python sandbox


@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "from subprocess import run",
        "__import__('os').system('ls')",
        "eval('1+1')",
        "exec('x=1')",
        "open('/etc/passwd')",
        "().__class__.__bases__",
        "globals()",
        "class Foo: pass",
        "lambda: 1",
    ],
)
def test_the_sandbox_rejects_anything_that_is_not_arithmetic(code: str) -> None:
    assert static_check(code) is not None, code


@pytest.mark.parametrize(
    "code",
    [
        "result = 1 + 1",
        "result = round((416161 - 391035) / 391035 * 100, 2)",
        "vals = [1, 2, 3]\nresult = sum(vals) / len(vals)",
        "result = max(3, 7) ** 2",
    ],
)
def test_arithmetic_is_allowed(code: str) -> None:
    assert static_check(code) is None, code


def test_syntax_errors_are_reported_not_raised() -> None:
    assert "SyntaxError" in (static_check("result = (1 +") or "")


def test_the_sandbox_computes(ctx: ToolContext) -> None:
    result = REGISTRY.dispatch("python", {"code": "result = round(112010 / 1000, 2)"}, ctx)
    assert result.ok and result.data["result"] == pytest.approx(112.01)


def test_code_without_a_result_is_an_error(ctx: ToolContext) -> None:
    result = REGISTRY.dispatch("python", {"code": "x = 5"}, ctx)
    assert result.ok is False and "result" in result.error


def test_a_runtime_error_is_returned_as_an_observation(ctx: ToolContext) -> None:
    result = REGISTRY.dispatch("python", {"code": "result = 1 / 0"}, ctx)
    assert result.ok is False and "ZeroDivisionError" in result.error


def test_filesystem_access_is_blocked_at_runtime_too(ctx: ToolContext) -> None:
    """Belt and braces: the static check catches `open`, but builtins are stripped as well."""
    result = REGISTRY.dispatch("python", {"code": "result = open('/etc/hosts').read()"}, ctx)
    assert result.ok is False


# ---------------------------------------------------------------- finish


def test_finish_carries_structured_citations(ctx: ToolContext) -> None:
    """Citations must be structured, or `s2` and D2 cannot be computed at all."""
    result = REGISTRY.dispatch(
        "finish",
        {
            "answer": "Net income was $112,010,000,000.",
            "value": 112010000000,
            "unit": "USD",
            "citations": [{"kind": "fact", "ref": "NetIncomeLoss"}],
        },
        ctx,
    )
    assert result.ok
    assert result.data["citations"][0]["ref"] == "NetIncomeLoss"
    assert result.data["value"] == 112010000000


def test_finish_tolerates_a_missing_value(ctx: ToolContext) -> None:
    """Narrative answers have no single figure; that is not an error."""
    result = REGISTRY.dispatch("finish", {"answer": "They cited supply chain risk."}, ctx)
    assert result.ok and result.data["value"] is None


# ---------------------------------------------------------------- lookup_fact


def test_lookup_returns_facts_within_the_horizon(seeded: bool, ctx: ToolContext) -> None:
    result = REGISTRY.dispatch("lookup_fact", {"tag": "NetIncomeLoss", "fp": "FY"}, ctx)
    assert result.ok and result.data
    assert all(row["accepted_at"] <= str(ctx.as_of) for row in result.data)


def test_a_wrong_tag_returns_a_recoverable_error(seeded: bool, ctx: ToolContext) -> None:
    """The error names the right tag, so the agent can fix it on the next step."""
    result = REGISTRY.dispatch("lookup_fact", {"tag": "NetIncome"}, ctx)
    assert result.ok is False
    assert "NetIncomeLoss" in result.error
