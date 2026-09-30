"""What the corpus actually contains at a given horizon.

The two halves of the environment have very different reach. `xbrl_facts` comes from SEC
`companyfacts`, which carries a company's full reporting history — facts here go back to 2009.
Filing *documents* are ingested individually, and only the last year of them exists. So a
question with a 2010 horizon has a fully populated fact table and an **empty document corpus**.

That asymmetry is correct point-in-time behaviour, not a bug: in 2010 nobody could see a 2025
10-K. But an agent cannot tell "no documents exist at this horizon" from "your query was
wrong", so it retries, rephrases, and burns its step budget on an empty table. On the MVP2.2
dev split, 9 of 30 questions had no visible documents at all, and those 9 accounted for nearly
every aborted run and roughly double the step count of the rest.

The fix is for the environment to say so. These helpers let the tools and the executor state
the coverage window plainly, so the agent redirects to XBRL instead of exploring.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from financevault.env.asof import AsOf
from financevault.store import pg


@dataclass(frozen=True, slots=True)
class Coverage:
    """What is visible at one horizon, and what exists in the corpus overall."""

    documents_visible: int
    facts_visible: int
    documents_from: date | None
    documents_to: date | None
    facts_from: date | None

    @property
    def documents_empty(self) -> bool:
        return self.documents_visible == 0

    def advice(self) -> str:
        """One line an agent can act on. Empty when documents are available as normal."""
        if not self.documents_empty:
            return ""
        window = (
            f"{self.documents_from} to {self.documents_to}"
            if self.documents_from
            else "not ingested"
        )
        note = (
            f"No filing documents are visible at this as-of date; the document corpus covers "
            f"{window}. Do not use retrieve_filings or query v_filings/v_chunks — they are "
            f"empty here and will stay empty."
        )
        if self.facts_visible:
            note += (
                f" XBRL facts are available from {self.facts_from} "
                f"({self.facts_visible:,} visible): use lookup_fact or query v_xbrl_facts."
            )
        return note


def at(as_of: AsOf, *, cik: str | None = None) -> Coverage:
    """Corpus coverage at a horizon, optionally scoped to one company."""
    params: dict = {**as_of.params()}
    clause = ""
    if cik:
        clause = " AND cik = %(cik)s"
        params["cik"] = cik

    docs = pg.fetch_all(
        f"SELECT count(*) n, min(accepted_at)::date lo, max(accepted_at)::date hi "
        f"FROM filings WHERE {as_of.sql('filings')}{clause}",
        params,
    )[0]
    facts = pg.fetch_all(
        f"SELECT count(*) n, min(accepted_at)::date lo "
        f"FROM xbrl_facts WHERE {as_of.sql('xbrl_facts')}{clause}",
        params,
    )[0]
    corpus = pg.fetch_all(
        "SELECT min(accepted_at)::date lo, max(accepted_at)::date hi FROM filings"
    )[0]

    return Coverage(
        documents_visible=docs["n"] or 0,
        facts_visible=facts["n"] or 0,
        documents_from=corpus["lo"],
        documents_to=corpus["hi"],
        facts_from=facts["lo"],
    )
