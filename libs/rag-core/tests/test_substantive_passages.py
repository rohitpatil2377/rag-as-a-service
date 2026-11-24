"""Tier 3 — substantive-passage awareness for the whole-document coverage measure (pure logic).

The whole-document read reported a fully-covered SOP file as 60% incomplete because it counted
BOILERPLATE — page markers, letterhead footers, SOP master-listing rows, revision-history
scaffolding, tiny label lines and duplicate headers — as uncovered obligations. These tests pin the
boilerplate/substance boundary (conservative: ambiguous → substantive) and the coverage tally that
excludes boilerplate from expected/covered.
"""

from __future__ import annotations

import pytest

from rag_core.completeness import passage_is_substantive, substantive_coverage

_BOILERPLATE = [
    "Page 1 of 1",
    "Page 12",
    "1 of 1",
    "- 3 -",
    "   ",  # whitespace-only
    "6.2.3",  # bare section number
    "Annexure A",  # short label / heading
    "Revision History",  # change-history scaffolding header
    "Document Control",
    "PBSOP001 | SOP for Preparation of Documents | Rev 3 | 2024-01-01",  # SOP master-listing row
    "QA-SOP-12 | Handling of Deviations | Rev No. 2 | 2023",
    "Introduction .......................... 3",  # table-of-contents leader
    "Tel: 011-2345678   Fax: 011-2345679",  # letterhead / footer
    "Email: info@example.com   www.example.com",
]

_SUBSTANTIVE = [
    "The vendor must encrypt all data at rest using AES-256 for the full term of the engagement.",
    "Operators must record every batch in the change history log before release.",
    "The Location Head authorizes the SOP, signing in blue ink on the front page.",
    "This procedure applies to the preparation, review and issue of all standard operating procedures.",
    # A requirements table row WITHOUT a doc-code + revision index shape stays substantive (real data).
    "AES-256 | required | all client data at rest must be encrypted",
    # Ambiguous / short-ish real prose — conservative rule keeps it substantive.
    "This section applies to all staff and contractors.",
]


@pytest.mark.parametrize("text", _BOILERPLATE)
def test_boilerplate_is_not_substantive(text: str) -> None:
    assert passage_is_substantive(text) is False


@pytest.mark.parametrize("text", _SUBSTANTIVE)
def test_real_content_is_substantive(text: str) -> None:
    assert passage_is_substantive(text) is True


def test_conservative_when_unsure_keeps_content() -> None:
    """A passage that matches no clear boilerplate shape is SUBSTANTIVE — real content is never hidden."""
    assert passage_is_substantive("Retain audit logs for seven years.") is True
    assert passage_is_substantive("Notify the client within twenty-four hours.") is True


def test_coverage_excludes_boilerplate_from_expected_and_covered() -> None:
    """A mix of substantive + boilerplate chunks: only the substantive ones count, and a fully-cited
    substantive set reports covered == expected even amid heavy boilerplate."""
    passages = [
        ("c0", "The vendor must encrypt all data at rest using AES-256."),  # substantive, cited
        ("c1", "Page 1 of 1"),  # boilerplate
        ("c2", "The vendor must retain audit logs for seven years."),  # substantive, cited
        ("c3", "PBSOP001 | Master SOP List | Rev 3 | 2024"),  # boilerplate (master listing)
        ("c4", "Tel: 011-2345678  Fax: 011-2345679"),  # boilerplate (footer)
    ]
    covered, expected, uncovered = substantive_coverage(passages, {"c0", "c2"})
    assert (covered, expected) == (2, 2)
    assert uncovered == []


def test_coverage_reports_uncovered_substantive_in_document_order() -> None:
    passages = [
        ("c0", "The operator must gown before entering the clean room."),  # cited
        ("c1", "Page 2 of 9"),  # boilerplate
        ("c2", "Deviations must be recorded and investigated within 24 hours."),  # uncovered
        ("c3", "Samples must be retained for the shelf-life plus one year."),  # uncovered
    ]
    covered, expected, uncovered = substantive_coverage(passages, {"c0"})
    assert (covered, expected) == (1, 3)
    assert uncovered == ["c2", "c3"]


def test_obligation_register_row_is_kept_despite_index_shape() -> None:
    """NEVER-MISS: a requirements/obligations-register row shares the doc-code + revision shape of a
    master-listing index row, but it STATES an obligation — it must be kept, not dropped."""
    assert passage_is_substantive("REQ-101 | Data must be encrypted at rest | Rev 2") is True
    assert passage_is_substantive("STD-014 | Backups shall be retained for seven years | Rev 5") is True


def test_pure_master_list_index_row_is_dropped() -> None:
    """A master-list INDEX row (code + revision + short title, NO obligation verb) is boilerplate."""
    assert passage_is_substantive("PBSOP006 | Radioactive Material Handling and Disposal | Rev 1") is False


def test_document_of_obligation_rows_is_not_a_false_complete() -> None:
    """A document made ENTIRELY of obligation-verb table rows must yield expected > 0 (never an early
    complete=true, covered=0 from every row being mistaken for an index)."""
    passages = [
        ("r0", "REQ-1 | The vendor must encrypt all data at rest | Rev 1"),
        ("r1", "REQ-2 | The vendor shall notify the client within 24 hours | Rev 1"),
        ("r2", "REQ-3 | Audit logs are to be retained for seven years | Rev 2"),
    ]
    covered, expected, uncovered = substantive_coverage(passages, {"r0"})
    assert expected == 3
    assert covered == 1
    assert uncovered == ["r1", "r2"]


def test_short_obligation_with_cue_survives_the_floor() -> None:
    """A short passage that carries an obligation cue stays substantive even below the word floor."""
    assert passage_is_substantive("PPE must be worn.") is True
    assert passage_is_substantive("Gloves required.") is True


def test_duplicate_of_neighbour_counts_once() -> None:
    """A header/footer reprinted on consecutive pages is one obligation at most, not many."""
    footer = "This document is confidential and proprietary to Acme Pharma Limited."
    passages = [
        ("c0", footer),
        ("c1", footer),  # verbatim duplicate of its neighbour → not counted again
        ("c2", "The batch record must be reviewed by QA before disposition."),
    ]
    covered, expected, _uncovered = substantive_coverage(passages, set())
    assert expected == 2  # the footer once + the QA obligation
    assert covered == 0
