"""Near-duplicate chunks (P16).

Ingest already refuses a byte-identical re-upload. This is the case it cannot see: the same
TEXT under two different doc_ids. Measured on the real corpus, `acme__hr_policies` has 15 such
pairs at similarity 1.0 — the Code of Conduct is reproduced verbatim inside the HR Policy
Handbook.

That is the mechanism behind the QA answer that said "the same requirement is reiterated in..."
and cited [1], [3] and [5]: one rule printed three times, read as three independent
confirmations.
"""

from __future__ import annotations

from rag_core.dedup import band_key, near_duplicates, similarity

_CLAUSE = (
    "All employees shall conduct themselves with integrity and shall not disclose confidential "
    "information relating to the Company, its customers or its partners to any third party."
)
_REPRINT_WITH_EDITS = (
    "All employees shall conduct themselves with integrity and shall not disclose confidential "
    "information relating to the Company, its customers or its partners to any outside party."
)
_UNRELATED = (
    "The night shift allowance is paid to operations engineers rostered at the Nashik facility "
    "and is credited with the following month's salary."
)


def test_the_same_text_under_two_doc_ids_is_found() -> None:
    """The real case: Code_Of_Conduct.pdf::11 and HR_POLICY_HANDBOOK.pdf::32 are one clause."""
    found = near_duplicates([("handbook::32", _CLAUSE)], [("conduct::11", _CLAUSE)], threshold=0.8)
    assert found["handbook::32"][0] == "conduct::11"
    assert found["handbook::32"][1] == 1.0


def test_a_reprint_with_small_edits_is_still_a_duplicate() -> None:
    """An exact hash misses these, and they are exactly the ones that read as corroboration:
    the same rule reprinted with a word changed."""
    assert similarity(_CLAUSE, _REPRINT_WITH_EDITS) > 0.8
    assert near_duplicates([("b", _REPRINT_WITH_EDITS)], [("a", _CLAUSE)], threshold=0.8)


def test_unrelated_policy_text_is_not_flagged() -> None:
    """Two chunks from the same handbook share vocabulary and register. Flagging those would
    make the signal useless — and worse, would suppress genuine second sources."""
    assert similarity(_CLAUSE, _UNRELATED) < 0.3
    assert near_duplicates([("b", _UNRELATED)], [("a", _CLAUSE)], threshold=0.8) == {}


def test_short_chunks_are_not_compared() -> None:
    """A heading or a page number shares shingles with everything by accident."""
    assert band_key("6.1 Process") == ""
    assert near_duplicates([("b", "6.1 Process")], [("a", "6.1 Process")], threshold=0.8) == {}


def test_a_chunk_is_never_its_own_duplicate() -> None:
    """So a document can be checked against itself — repeating itself IS a finding."""
    assert near_duplicates([("a", _CLAUSE)], [("a", _CLAUSE)], threshold=0.8) == {}


def test_the_strongest_match_wins() -> None:
    """A clause reprinted in three documents should point at what it most resembles, not at
    whichever happened to be scanned first."""
    found = near_duplicates(
        [("new", _CLAUSE)],
        [("edited", _REPRINT_WITH_EDITS), ("exact", _CLAUSE)],
        threshold=0.8,
    )
    assert found["new"][0] == "exact"


def test_a_document_that_repeats_itself_is_found() -> None:
    """Boilerplate reprinted on many pages of one document — the same harm, one doc_id."""
    chunks = [("d::1", _CLAUSE), ("d::7", _CLAUSE), ("d::9", _UNRELATED)]
    found = near_duplicates(chunks, chunks, threshold=0.8)
    assert set(found) == {"d::1", "d::7"}
    assert "d::9" not in found
