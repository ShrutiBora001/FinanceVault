"""Postgres access: a lazily-opened connection pool plus small query helpers.

Everything in the system goes through this module rather than opening its own
connection, so pool sizing and pgvector registration happen in exactly one place.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from financevault.config import settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_pool: ConnectionPool | None = None


def _configure(conn: psycopg.Connection) -> None:
    """Run on every pooled connection: pgvector types must be registered per-connection.

    Tolerant of a missing extension so the pool can still be opened against a database
    that has not been migrated yet. `migrate()` bootstraps on a direct connection instead,
    and every connection handed out afterwards registers cleanly.
    """
    try:
        register_vector(conn)
    except psycopg.ProgrammingError:
        conn.rollback()


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            settings().pg_dsn,
            min_size=1,
            max_size=8,
            configure=_configure,
            open=True,
        )
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    with pool().connection() as conn:
        yield conn


@contextmanager
def cursor(row_factory: Any = dict_row) -> Iterator[psycopg.Cursor]:
    with connection() as conn, conn.cursor(row_factory=row_factory) as cur:
        yield cur


def fetch_all(sql: str, params: Sequence[Any] | dict[str, Any] | None = None) -> list[dict]:
    with cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def fetch_one(sql: str, params: Sequence[Any] | dict[str, Any] | None = None) -> dict | None:
    with cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def fetch_value(sql: str, params: Sequence[Any] | dict[str, Any] | None = None) -> Any:
    with cursor(row_factory=None) as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        return None if row is None else row[0]


def execute(sql: str, params: Sequence[Any] | dict[str, Any] | None = None) -> int:
    with cursor() as cur:
        cur.execute(sql, params)
        return cur.rowcount


def execute_many(sql: str, rows: Sequence[Sequence[Any]]) -> int:
    if not rows:
        return 0
    with cursor() as cur:
        cur.executemany(sql, rows)
        return cur.rowcount


# ---------------------------------------------------------------- migration

TABLES = ("filings", "chunks", "xbrl_facts", "prices", "runs", "steps", "journal")


def migrate() -> None:
    """Apply schema.sql. Idempotent, so it is safe to run on every boot.

    Uses a direct connection rather than the pool: schema.sql is what creates the `vector`
    extension, and the pool's configure hook wants that extension to already exist.
    """
    sql = SCHEMA_PATH.read_text()
    with psycopg.connect(settings().pg_dsn, autocommit=True) as conn:
        conn.execute(sql)


def table_counts() -> dict[str, int]:
    """Row count per table, used by `make migrate` and the ingest report."""
    counts: dict[str, int] = {}
    for table in TABLES:
        counts[table] = fetch_value(f"SELECT count(*) FROM {table}") or 0
    return counts


def missing_tables() -> list[str]:
    rows = fetch_all(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename = ANY(%s)",
        (list(TABLES),),
    )
    present = {r["tablename"] for r in rows}
    return [t for t in TABLES if t not in present]
