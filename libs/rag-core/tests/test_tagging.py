"""Unit tests for FR-B7 auto-metadata & taxonomy tagging — pure logic."""

from __future__ import annotations

from rag_core.tagging import apply_taxonomy, auto_tags, classify_doc_type, document_date, latest_year


def test_a_year_in_the_body_does_not_date_the_document() -> None:
    """CHANGED DELIBERATELY. This used to return "the most recent 4-digit year anywhere in the
    text", which pinned a real defect: measured on a live corpus it dated a 2026 handbook as 2029
    (a future year quoted inside a table) and a travel policy as 2000. Once anything RANKS or
    resolves conflicts on that field, a confidently wrong date is worse than no date, because
    recency then prefers the wrong document."""
    assert latest_year("Signed in 2019, amended in 2026 and 2023.") is None
    assert latest_year("no years here") is None


def test_the_date_comes_from_the_document_own_release_block() -> None:
    """Acme documents carry "Document Release History" — Version | Prepared | Review | Release.
    The last row is the current version, so the latest RELEASE date in that block dates it."""
    text = (
        "Document Release History\n"
        "Version | Prepared Date | Review Date | Release Date\n"
        "1.0 | 1st December 2022 | 15th December 2022 | 1st January 2023\n"
        "2.0 | 1st December 2023 | 15th December 2023 | 1st January 2024\n"
    )
    assert document_date(text) == "2024-01-01"
    assert latest_year(text) == "2024"


def test_classify_doc_type_from_cues() -> None:
    assert classify_doc_type("This Agreement is hereby agree between the parties.") == "agreement"
    assert classify_doc_type("Total due: 500. Invoice number 12.") == "invoice"
    assert classify_doc_type("Eligibility: this policy applies.") == "policy"
    assert classify_doc_type("just some prose") == "document"


def test_apply_taxonomy_matches_keywords() -> None:
    tax = {"HR": ["leave", "salary"], "Security": ["encryption", "breach"]}
    assert apply_taxonomy("Annual leave and salary details.", tax) == ["HR"]
    assert set(apply_taxonomy("leave policy and encryption at rest", tax)) == {"HR", "Security"}
    assert apply_taxonomy("nothing relevant", tax) == []


def test_auto_tags_builds_metadata() -> None:
    text = "Acme Night Shift Allowance Policy. Effective Date: 1st January 2026. Eligibility applies to all staff."
    tags = auto_tags(text, {"HR": ["allowance", "shift"]})
    assert tags["doc_type"] == "policy"
    assert tags["doc_date"] == "2026-01-01"
    assert tags["year"] == "2026"
    assert tags["category"] == "HR"
    assert "Acme" in tags["entities"]


def test_an_undated_document_carries_no_date_rather_than_a_guessed_one() -> None:
    """Undated is the common case and a real answer. A caller must read a missing date as
    UNKNOWN, never as old — defaulting it would silently deprioritise every document that simply
    does not state its date."""
    tags = auto_tags("A simple memo from 2024.", None)
    assert "doc_date" not in tags
    assert "year" not in tags
    assert "category" not in tags


def test_every_key_auto_tags_can_write_is_declared() -> None:
    """DERIVED_KEYS drives what a re-index CLEARS before re-deriving. A key that auto_tags writes
    but this set omits would survive a re-index forever, which is exactly the stale-`year` bug:
    documents with no release block kept a wrong year through a re-index that reported success."""
    from rag_core.tagging import DERIVED_KEYS

    rich = (
        "Acme Night Shift Allowance Policy. Effective Date: 1st January 2026. "
        "Eligibility applies to all Acme staff and contractors."
    )
    produced = set(auto_tags(rich, {"HR": ["allowance"], "Ops": ["staff"]}))
    assert produced <= DERIVED_KEYS, f"auto_tags writes keys not declared: {produced - DERIVED_KEYS}"
