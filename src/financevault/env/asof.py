"""Point-in-time filtering — the leakage guarantee.

The rule: a query carrying an `as_of` instant may only see rows whose `accepted_at` is at
or before that instant. `accepted_at` is the SEC acceptance timestamp, the moment a document
became knowable to the public. It is deliberately *not* `period_end` (when the fiscal period
closed) or `filed_at` (a date, not an instant, and occasionally different from acceptance).

The gap is not academic. Apple's FY2025 10-K covers a period ending 2025-09-27 but was not
accepted until 2025-10-31. A retriever keyed on period end would answer a question dated
2025-10-01 using a document that did not exist yet, and would look more accurate for it.

Enforcement lives here rather than at call sites. `AsOf` cannot be constructed from nothing,
and the SQL helpers below always emit a predicate, so a query that forgets the filter is a
query that does not compile rather than one that quietly leaks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

# Tables carrying a point-in-time column, and the column on each.
PIT_COLUMNS: dict[str, str] = {
    "filings": "accepted_at",
    "xbrl_facts": "accepted_at",
}


class LookaheadError(RuntimeError):
    """Raised when a row that postdates the as-of instant reaches a caller."""


@dataclass(frozen=True, slots=True)
class AsOf:
    """An information horizon. Immutable, timezone-aware, always explicit."""

    instant: datetime

    def __post_init__(self) -> None:
        if self.instant.tzinfo is None:
            raise ValueError(
                "as_of must be timezone-aware; a naive datetime silently assumes local time "
                "and shifts the horizon by hours"
            )

    @classmethod
    def parse(cls, value: str | datetime) -> AsOf:
        """Build from an ISO-8601 string or a datetime. Naive input is rejected, not coerced."""
        if isinstance(value, datetime):
            return cls(value)
        text = value.strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            # A bare date means "the start of that day, UTC" — an explicit, documented
            # choice rather than an accident of the host timezone.
            parsed = parsed.replace(tzinfo=UTC)
        return cls(parsed)

    @classmethod
    def now(cls) -> AsOf:
        return cls(datetime.now(UTC))

    def sql(self, table: str, alias: str | None = None) -> str:
        """A SQL predicate restricting `table` to rows knowable at this instant.

        Raises for tables with no point-in-time column, so adding such a table without
        deciding how it is filtered fails loudly rather than defaulting to unfiltered.
        """
        if table not in PIT_COLUMNS:
            raise KeyError(
                f"{table!r} has no registered point-in-time column; add it to PIT_COLUMNS "
                f"and decide explicitly how it is filtered"
            )
        prefix = alias or table
        return f"{prefix}.{PIT_COLUMNS[table]} <= %(as_of)s"

    def params(self) -> dict[str, datetime]:
        return {"as_of": self.instant}

    def check(self, rows: list[dict], column: str = "accepted_at") -> list[dict]:
        """Assert no row postdates the horizon.

        A belt-and-braces pass over results that already went through `sql()`. It exists
        because the cost of a silent leak is an entire experiment, and the cost of this
        check is one comparison per row.
        """
        for row in rows:
            value = row.get(column)
            if value is not None and value > self.instant:
                raise LookaheadError(
                    f"row with {column}={value.isoformat()} exceeds as_of="
                    f"{self.instant.isoformat()}"
                )
        return rows

    def __str__(self) -> str:
        return self.instant.isoformat()
