"""s3 — numeric correctness.

Every number an answer states is checked against the evidence the trajectory actually
retrieved. This is the signal the project exists for: a figure that is right except for a
factor of a thousand, or right except that it is last quarter's, reads as completely fluent
and is completely wrong.

**Claims are matched against retrieved evidence, not against the fact universe.** This is the
central design decision, and getting it wrong makes the verifier worthless. A company has
tens of thousands of XBRL facts spanning many magnitudes, so almost any plausible number
matches *something* within tolerance -- a first implementation here "verified" a fabricated
$77,777,777,777. Restricting the match to facts the agent actually observed turns the check
from "does this number exist somewhere" into "did this number come from where you say it
did", which is the question that matters.

**Programmatic, never LLM-judged.** A judge asked "is 416.2 billion the right revenue" agrees
with anything plausible, and plausible is the failure mode. This is arithmetic against stored
values and cannot be talked into agreeing.

Failure classes are kept distinct because they mean different things and need different fixes:

- **scale** -- right digits, wrong magnitude. Usually units-vs-millions confusion, and the
  most common silent error in financial writing.
- **period** -- matches retrieved evidence, but for a different fiscal period.
- **uncited** -- matches a real fact the agent never retrieved. It guessed correctly from
  memory, which is not the same as knowing, and is not reproducible.
- **fabricated** -- matches nothing on record at all.

Only `exact` counts as verified. The rest are the numerator of D2, the unsupported-claim rate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from financevault.env.asof import AsOf
from financevault.store import pg
from financevault.verify.signals import Signal, na

# Relative tolerance. Filings round and answers restate rounded figures, so exact equality
# would fail correct answers. Tight, because the match set is small.
REL_TOL = 0.005

SCALE_WORDS = {
    "thousand": 1e3,
    "thousands": 1e3,
    "k": 1e3,
    "million": 1e6,
    "millions": 1e6,
    "m": 1e6,
    "mm": 1e6,
    "billion": 1e9,
    "billions": 1e9,
    "b": 1e9,
    "bn": 1e9,
    "trillion": 1e12,
    "trillions": 1e12,
    "t": 1e12,
}
SCALE_ALT = "|".join(sorted(SCALE_WORDS, key=len, reverse=True))

NUMBER_RE = re.compile(
    r"(?<![\w.])"
    r"(?P<sign>-|\()?\s*\$?\s*"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*\)?"  # accounting negatives close the paren before any scale word
    r"\s*(?P<scale>" + SCALE_ALT + r")?\b"
    r"\s*(?P<pct>%|percent)?",
    re.IGNORECASE,
)

# Powers of ten a scale confusion typically lands on.
SCALE_FACTORS = (1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)


MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
    "|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
# A number is a date part if a month name sits just before it, or if it is inside an ISO date.
DATE_CONTEXT_RE = re.compile(rf"(?:{MONTHS})\.?\s*$", re.IGNORECASE)
ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True, slots=True)
class Claim:
    text: str
    value: float
    is_percent: bool
    is_financial: bool = True


@dataclass(slots=True)
class Evidence:
    """Numeric values the trajectory actually observed, with where each came from."""

    values: list[dict] = field(default_factory=list)

    def add(
        self,
        value: float,
        *,
        source: str,
        tag: str | None = None,
        period_end: str | None = None,
        unit: str | None = None,
    ) -> None:
        self.values.append(
            {
                "value": float(value),
                "source": source,
                "tag": tag,
                "period_end": period_end,
                "unit": unit,
            }
        )

    def __len__(self) -> int:
        return len(self.values)


def _walk_numbers(node: Any, evidence: Evidence, source: str) -> None:
    """Collect every numeric leaf from a tool observation."""
    if isinstance(node, dict):
        value = node.get("value")
        if isinstance(value, int | float) and not isinstance(value, bool):
            evidence.add(
                value,
                source=source,
                tag=node.get("tag"),
                period_end=node.get("period_end"),
                unit=node.get("unit"),
            )
            return
        for item in node.values():
            _walk_numbers(item, evidence, source)
    elif isinstance(node, list):
        for item in node:
            _walk_numbers(item, evidence, source)
    elif isinstance(node, int | float) and not isinstance(node, bool):
        evidence.add(node, source=source)


def collect_evidence(steps: list[dict]) -> Evidence:
    """Every number the agent saw, from every successful tool observation before `finish`."""
    evidence = Evidence()
    for step in steps:
        tool = step.get("tool")
        obs = step.get("obs") or step.get("observation") or {}
        if tool in (None, "finish") or not obs.get("ok"):
            continue
        _walk_numbers(obs.get("data"), evidence, source=tool)
    return evidence


def _close(a: float, b: float, rel_tol: float = REL_TOL) -> bool:
    if b == 0:
        return abs(a) < 1e-9
    return abs(a - b) / abs(b) <= rel_tol


def classify(
    value: float,
    evidence: Evidence,
    *,
    cik: str | None = None,
    as_of: AsOf | None = None,
    period_end: str | None = None,
) -> tuple[str, dict | None]:
    """Classify one claim against retrieved evidence, then against the record as a fallback."""
    scale_hit: dict | None = None
    period_hit: dict | None = None

    for row in evidence.values:
        if _close(value, row["value"]):
            if period_end and row.get("period_end") and row["period_end"] != period_end:
                period_hit = period_hit or row
                continue
            return "exact", row
        for factor in SCALE_FACTORS:
            if _close(value * factor, row["value"]):
                scale_hit = scale_hit or {**row, "implied_factor": factor}
                break

    if scale_hit:
        return "scale", scale_hit
    if period_hit:
        return "period", period_hit

    # Not in evidence. Distinguishing a correct-but-unretrieved figure from an invented one
    # is diagnostic only -- neither is verified, but they indicate different problems.
    if cik and as_of and _exists_on_record(value, cik, as_of):
        return "uncited", None
    return "fabricated", None


def _exists_on_record(value: float, cik: str, as_of: AsOf) -> bool:
    """Whether any fact knowable at the horizon carries this value, within tolerance."""
    lo, hi = sorted((value * (1 - REL_TOL), value * (1 + REL_TOL)))
    row = pg.fetch_one(
        f"""
        SELECT 1 FROM xbrl_facts
        WHERE cik = %(cik)s AND {as_of.sql("xbrl_facts")}
          AND value BETWEEN %(lo)s AND %(hi)s
        LIMIT 1
        """,
        {"cik": cik, "lo": lo, "hi": hi, **as_of.params()},
    )
    return row is not None


def _is_financial(text: str, match: re.Match, raw: str, has_scale: bool) -> bool:
    """Whether a matched number is a monetary amount rather than a date or ordinal.

    Necessary because answers state periods alongside figures -- "net income for fiscal year
    2025 (period ended September 27, 2025) was $112.010 billion" contains one financial claim
    and three date parts. Counting the dates dropped a fully correct answer to 0.25, so a
    verifier without this rejects correct trajectories on sight.

    A number is financial if it is marked as money -- a currency symbol, a scale word, comma
    grouping -- or is simply too large to be anything else.
    """
    if ISO_DATE_RE.search(text[max(0, match.start() - 2) : match.end() + 8]):
        return False
    if DATE_CONTEXT_RE.search(text[max(0, match.start() - 12) : match.start()]):
        return False

    marked_as_money = "$" in match.group(0) or has_scale or "," in raw
    if marked_as_money:
        return True

    # Unmarked: a bare four-digit number in calendar range is a year, not an amount.
    if re.fullmatch(r"\d{4}", raw) and 1900 <= float(raw) <= 2100:
        return False
    # Otherwise only large bare numbers are plausible amounts; "27" is not.
    return len(raw.replace(".", "").lstrip("0")) >= 6


def extract_claims(text: str) -> list[Claim]:
    """Pull numeric claims from an answer, resolving scale words into the value."""
    text = text or ""
    claims: list[Claim] = []
    for match in NUMBER_RE.finditer(text):
        raw = match.group("num")
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        scale_word = (match.group("scale") or "").lower()
        if scale_word:
            value *= SCALE_WORDS[scale_word]
        if match.group("sign"):
            value = -value
        claims.append(
            Claim(
                text=match.group(0).strip(),
                value=value,
                is_percent=bool(match.group("pct")),
                is_financial=_is_financial(text, match, raw, bool(scale_word)),
            )
        )
    return claims


def _describe(claim: Claim, kind: str, hit: dict | None) -> str:
    if kind == "scale" and hit:
        return (
            f"{claim.text!r} is off by {hit['implied_factor']:g}x "
            f"(evidence had {hit.get('tag') or 'value'}={hit['value']:,.0f})"
        )
    if kind == "period" and hit:
        return (
            f"{claim.text!r} matches evidence for period {hit.get('period_end')}, not the one asked"
        )
    if kind == "uncited":
        return f"{claim.text!r} is on record but was never retrieved by this trajectory"
    return f"{claim.text!r} matches nothing retrieved or on record"


def score(
    answer: str,
    *,
    evidence: Evidence,
    cik: str | None = None,
    as_of: AsOf | None = None,
    period_end: str | None = None,
    stated_value: float | None = None,
) -> Signal:
    """Fraction of an answer's numeric claims that trace to retrieved evidence."""
    claims = extract_claims(answer)
    # Two exclusions. Percentages are derived rather than reported, so they are not in
    # xbrl_facts -- verifying them means re-deriving from operands, which belongs to a later
    # derived-value check. Dates and ordinals are not claims about money at all.
    checkable = [c for c in claims if not c.is_percent and c.is_financial]
    if stated_value is not None:
        checkable.append(Claim(f"value={stated_value:g}", stated_value, is_percent=False))

    if not checkable:
        return na("no checkable numeric claims in the answer")
    if not evidence.values:
        return Signal(
            0.0,
            f"{len(checkable)} numeric claim(s) but the trajectory retrieved no evidence; "
            "every figure is unsupported",
        )

    verdicts = [
        (c, *classify(c.value, evidence, cik=cik, as_of=as_of, period_end=period_end))
        for c in checkable
    ]
    verified = sum(1 for _, kind, _ in verdicts if kind == "exact")
    problems = [_describe(c, k, h) for c, k, h in verdicts if k != "exact"]

    reason = f"{verified}/{len(verdicts)} claims traced to retrieved evidence"
    if problems:
        reason += "; " + "; ".join(problems[:3])
    return Signal(verified / len(verdicts), reason)
