"""A question that NAMES the things it asks about is about things INSIDE the documents.

Reported live on a pharmacy SOP corpus: "explain PBSOP001, PBSOP002, ... PBSOP010" was answered
with a per-FILE summary of the knowledge base -- "Summary of 1 document in this knowledge base"
followed by a list of the three uploaded files. The ten PBSOPs are sections inside ONE of those
files. The reader asked about ten procedures and was handed an inventory of the folder.

The signal is already computed elsewhere for retrieval coverage: a question carrying several
identifier-shaped tokens is enumerating items. Identifiers are what a document CONTAINS; a
knowledge base's documents are named by filenames, which this never matches.
"""

from __future__ import annotations

import pytest

from rag_core.intent import classify_intent_heuristic

_TEN = "PBSOP001, PBSOP002, PBSOP003, PBSOP004, PBSOP005, PBSOP006, PBSOP007, PBSOP008, PBSOP009, and PBSOP010"


@pytest.mark.parametrize(
    "query",
    [
        f"explain {_TEN}",
        f"summarize {_TEN}",
        f"describe in detail {_TEN}",
        f"give me a detailed summary analysis for all the 10 SOPs from {_TEN}",
        "walk me through XQA-001, XQA-002 and XQA-003",
    ],
)
def test_naming_the_items_is_never_a_summary_of_the_documents(query: str) -> None:
    assert classify_intent_heuristic(query) == "rag"


@pytest.mark.parametrize(
    "query",
    [
        "summarize each document in detail",
        "explain every document we have",
        "give me an overview of all the documents",
        "what does each file say",
    ],
)
def test_a_real_per_document_request_still_routes_to_doc_summary(query: str) -> None:
    """The guard must not swallow the intent it is narrowing -- these ask about the FILES."""
    assert classify_intent_heuristic(query) == "doc_summary"


def test_a_catalog_question_is_unaffected() -> None:
    assert classify_intent_heuristic("what documents do we have?") == "catalog"


def test_two_identifiers_are_not_an_enumeration() -> None:
    """Two is a comparison the ordinary path handles; the guard starts at three, matching the
    retrieval-coverage rule so the two cannot disagree about what counts as enumerated."""
    assert classify_intent_heuristic("compare PBSOP001 and PBSOP002") == "compare"


def test_a_year_is_not_an_identifier() -> None:
    """Otherwise "summarize all documents from 2024, 2025 and 2026" would stop being a
    doc_summary because of three four-digit numbers."""
    assert classify_intent_heuristic("summarize each document from 2024, 2025 and 2026") == "doc_summary"
