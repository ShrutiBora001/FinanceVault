"""Section-aware chunking.

Both bugs pinned here were invisible in the output: chunks still existed and looked fine,
they were simply attached to the wrong section. Retrieval degrades quietly when that happens,
and a citation that names the wrong part of a filing is worse than one that names none.
"""

from __future__ import annotations

from financevault.env import chunk

TENK = """
UNITED STATES SECURITIES AND EXCHANGE COMMISSION
Annual report pursuant to section 13 of the Securities Exchange Act of 1934 for the fiscal
year ended September 27, 2025. Commission file number 001-36743. Apple Inc. is filing this.

Item 1. Business
The Company designs, manufactures and markets smartphones, personal computers and tablets,
and sells a variety of related services across its several reportable operating segments.
Its products are sold through retail stores, an online storefront, direct sales and a network
of third-party cellular carriers, wholesalers, retailers and value-added resellers worldwide.

Item 1A. Risk Factors
The Company's business is subject to numerous risks including global economic conditions,
supply chain interruptions, and intense competition across every market in which it operates.
Adverse macroeconomic conditions have in the past materially affected demand, and could again
do so in ways the Company is unable to predict or effectively mitigate in any given period.

Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
Total net sales increased during the period, driven by higher Services revenue and partially
offset by lower sales of certain hardware categories across specific geographic segments.

Item 8. Financial Statements and Supplementary Data
The consolidated financial statements and accompanying notes appear in the pages that follow
this item, together with the report of the independent registered public accounting firm.
"""

TENQ = """
Quarterly report pursuant to section 13 for the quarterly period ended December 27, 2025.
The registrant is a large accelerated filer as defined in Rule 12b-2 of the Exchange Act.

Item 1. Financial Statements
Condensed consolidated statements of operations for the three months ended, unaudited, are
presented together with the accompanying notes on the following pages of this quarterly report.

Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations
Net sales for the quarter rose year over year, reflecting growth in Services offset by
declines in certain product categories across several of the Company's geographic segments.

Item 4. Controls and Procedures
Management evaluated the effectiveness of the Company's disclosure controls and procedures
and concluded that they were effective at the reasonable assurance level as of that date.
"""


def _labels(text: str, form: str) -> list[str]:
    return [s.label for s in chunk.split_sections(text, form) if s.key]


# ---------------------------------------------------------------- the long-heading bug


def test_a_long_item_heading_is_still_detected() -> None:
    """Regression: `.{0,80}$` dropped Item 7 from every Apple 10-K.

    The real heading is "Item 7. Management's Discussion and Analysis of Financial Condition
    and Results of Operations" — past 80 characters, so a line-end anchor never matched and
    MD&A was silently absorbed into the preceding item.
    """
    labels = _labels(TENK, "10-K")
    assert "management's discussion and analysis" in labels


def test_every_10k_item_in_the_fixture_is_found() -> None:
    labels = _labels(TENK, "10-K")
    assert labels == [
        "business",
        "risk factors",
        "management's discussion and analysis",
        "financial statements",
    ]


# ---------------------------------------------------------------- the form-numbering bug


def test_item_2_means_different_things_in_the_two_forms() -> None:
    """Regression: the 10-K map was applied to 10-Qs, labelling their MD&A as "properties"."""
    assert chunk.item_names("10-K")["2"] == "properties"
    assert chunk.item_names("10-Q")["2"] == "management's discussion and analysis"
    assert chunk.item_names("10-Q")["1"] == "financial statements"
    assert chunk.item_names("10-K")["1"] == "business"


def test_a_10q_is_labelled_with_10q_item_names() -> None:
    labels = _labels(TENQ, "10-Q")
    assert labels == [
        "financial statements",
        "management's discussion and analysis",
        "controls and procedures",
    ]
    assert "properties" not in labels


def test_an_unknown_form_falls_back_to_the_10k_map() -> None:
    assert chunk.item_names(None)["2"] == "properties"
    assert chunk.item_names("8-K")["2"] == "properties"


# ---------------------------------------------------------------- boundaries


def test_chunks_never_cross_a_section_boundary() -> None:
    """The whole point: one chunk, one section."""
    chunks = chunk.split(TENK, "10-K")
    assert chunks
    for c in chunks:
        # Every chunk's text must be contained in exactly one section's text.
        containing = [
            s for s in chunk.split_sections(TENK, "10-K") if c.text[:60] in " ".join(s.text.split())
        ]
        assert len(containing) == 1, f"chunk {c.idx} spans sections: {c.text[:60]!r}"


def test_front_matter_is_kept_not_discarded() -> None:
    """The cover page carries the filing date and entity details."""
    labels = [s.label for s in chunk.split_sections(TENK, "10-K")]
    assert "front matter" in labels


def test_a_document_with_no_items_still_chunks() -> None:
    chunks = chunk.split("Just some prose with no item headings at all. " * 40, "10-K")
    assert chunks and all(c.section is None for c in chunks)
