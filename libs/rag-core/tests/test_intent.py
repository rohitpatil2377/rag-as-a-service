"""Unit tests for semantic intent classification (pure logic).

The heuristic is the deterministic FALLBACK; the LLM path is the real semantic one. These
cover the mapping the fallback must get right, the robustness of ``parse_intent`` (junk and
summarize-without-history both degrade to the safe ``rag`` default), and that the prompt
offers exactly the allowed labels for the caller's history state.
"""

from __future__ import annotations

import pytest

from rag_core import (
    build_intent_prompt,
    classify_intent_heuristic,
    extract_filter_criteria,
    match_compare_targets,
    parse_intent,
)


@pytest.mark.parametrize(
    "query",
    [
        "list the documents in this KB",
        "what files do we have?",
        "show me the artifacts",
        "what's in the knowledge base inventory",
    ],
)
def test_heuristic_catalog(query: str) -> None:
    assert classify_intent_heuristic(query) == "catalog"


@pytest.mark.parametrize(
    "query",
    [
        "how many documents are uploaded",
        "how many documents are there?",
        "number of files in the knowledge base",
        "how many records do we have",
        "count of documents",
    ],
)
def test_heuristic_count(query: str) -> None:
    assert classify_intent_heuristic(query) == "count"


@pytest.mark.parametrize(
    "query",
    [
        "which documents are from 2026",
        "what changed recently",
        "latest documents",
        "policies of type agreement",
        "show me the files from 2024",
    ],
)
def test_heuristic_filter(query: str) -> None:
    assert classify_intent_heuristic(query) == "filter"


@pytest.mark.parametrize(
    "query",
    [
        "difference between the Code of Conduct and the HR Handbook",
        "compare policy A and policy B",
        "policy A vs policy B",
        "contrast the two agreements",
    ],
)
def test_heuristic_compare(query: str) -> None:
    assert classify_intent_heuristic(query) == "compare"


@pytest.mark.parametrize(
    "query",
    [
        "summarize the chat",
        "recap this conversation",
        "give me a tl;dr",
        "summarise the thread so far",
    ],
)
def test_heuristic_summarize(query: str) -> None:
    assert classify_intent_heuristic(query) == "summarize"


@pytest.mark.parametrize(
    "query",
    [
        "what is the leave policy?",
        "who approved the budget increase",
        "explain the uptime SLA",
        "summarize the leave policy",  # a summarize VERB with no conversation anchor stays rag
        "who signed the agreement",  # a bare doc_type word (no doc word/criterion) stays rag
        "what does the report conclude",  # ditto — conservative dispatch, not a filter
    ],
)
def test_heuristic_rag(query: str) -> None:
    assert classify_intent_heuristic(query) == "rag"


def test_heuristic_word_boundary_no_false_positive() -> None:
    # "profile" must not match the "file" catalog word.
    assert classify_intent_heuristic("update my profile settings") == "rag"


def test_parse_intent_clean_labels() -> None:
    assert parse_intent("catalog", has_history=False) == "catalog"
    assert parse_intent("  RAG\n", has_history=False) == "rag"
    assert parse_intent("summarize", has_history=True) == "summarize"
    # near-synonym spellings normalise
    assert parse_intent("catalogue", has_history=False) == "catalog"
    assert parse_intent("summary", has_history=True) == "summarize"


def test_parse_intent_new_labels() -> None:
    assert parse_intent("count", has_history=False) == "count"
    assert parse_intent("filter", has_history=False) == "filter"
    assert parse_intent("compare", has_history=False) == "compare"
    assert parse_intent("comparison", has_history=False) == "compare"
    # the new labels do not need conversation history
    assert parse_intent("count", has_history=True) == "count"
    # two labels at once is ambiguous → safe default
    assert parse_intent("count or compare", has_history=False) == "rag"


def test_parse_intent_junk_defaults_to_rag() -> None:
    assert parse_intent("", has_history=True) == "rag"
    assert parse_intent("I think this is about the weather", has_history=True) == "rag"
    assert parse_intent("catalog or summarize?", has_history=True) == "rag"  # ambiguous → safe default


def test_parse_intent_summarize_without_history_is_rag() -> None:
    assert parse_intent("summarize", has_history=False) == "rag"
    assert parse_intent("summary", has_history=False) == "rag"


def test_build_intent_prompt_lists_allowed_labels() -> None:
    with_history = build_intent_prompt("recap", has_history=True)
    assert "catalog, summarize, count, filter, compare, doc_summary, rag" in with_history

    no_history = build_intent_prompt("list files", has_history=False)
    # summarize is not offered as a valid label without a conversation to summarize; the new
    # count/filter/compare/doc_summary labels are offered on both paths.
    assert "catalog, count, filter, compare, doc_summary, rag" in no_history


def test_build_intent_prompt_separates_catalog_from_doc_summary() -> None:
    # The prompt must teach the LLM that catalog is NAMES/inventory only and doc_summary is
    # per-document CONTENTS, with an example of each.
    prompt = build_intent_prompt("summarize each doc", has_history=False)
    assert "NAMES" in prompt  # catalog rewritten to names/inventory only
    assert "doc_summary" in prompt
    assert "summarize each doc in detail" in prompt  # doc_summary example
    assert "what documents do we have?" in prompt  # catalog example


@pytest.mark.parametrize(
    "query",
    [
        "summarize each doc which we have in our KB in detail",
        "explain each and every doc in detail",
        "summarise all the documents",
        "describe every document we have",
        "walk me through each of the docs",
        "give an overview of all documents",
    ],
)
def test_heuristic_doc_summary(query: str) -> None:
    # A CONTENT verb about EACH/ALL of the documents is a per-doc summary — NOT a bare catalog list.
    assert classify_intent_heuristic(query) == "doc_summary"


def test_heuristic_doc_summary_regression_not_catalog() -> None:
    # The exact regression: these must be doc_summary, and must NOT be catalog.
    for q in ("summarize each doc which we have in our KB in detail", "explain each and every doc in detail"):
        got = classify_intent_heuristic(q)
        assert got != "catalog", f"regressed to catalog: {q}"  # the regression this fix closes
        assert got == "doc_summary", got


def test_heuristic_catalog_still_wins_for_inventory() -> None:
    # Inventory requests (no content verb) stay catalog.
    for q in ("what documents do we have", "list the docs"):
        assert classify_intent_heuristic(q) == "catalog"


def test_heuristic_single_topic_content_verb_stays_rag() -> None:
    # A content verb about a SINGLE topic (no each/all/plural-docs) is plain rag, never catalog.
    assert classify_intent_heuristic("what is the leave policy") == "rag"
    assert classify_intent_heuristic("summarize the leave policy") == "rag"
    assert classify_intent_heuristic("explain the uptime SLA") == "rag"
    assert classify_intent_heuristic("describe the document") == "rag"  # singular, no each/all


def test_parse_intent_doc_summary() -> None:
    assert parse_intent("doc_summary", has_history=False) == "doc_summary"
    assert parse_intent("  doc_summary\n", has_history=True) == "doc_summary"
    assert parse_intent("doc summary", has_history=False) == "doc_summary"
    # doc_summary must not be double-counted as summarize (its "summary" token) → ambiguous rag
    assert parse_intent("doc-summary", has_history=False) == "doc_summary"
    # a plain summarize reply is unaffected
    assert parse_intent("summarize", has_history=True) == "summarize"


def test_extract_filter_criteria() -> None:
    assert extract_filter_criteria("which documents are from 2026") == {"year": "2026"}
    assert extract_filter_criteria("policies of type agreement") == {"doc_type": "agreement"}
    assert extract_filter_criteria("what changed in the last 7 days") == {"recency_days": "7"}
    assert extract_filter_criteria("the latest documents") == {"recency_days": "30"}
    assert extract_filter_criteria("reports from 2024") == {"doc_type": "report", "year": "2024"}
    # nothing parseable → empty, so the handler falls back to catalog (list all) rather than abstaining
    assert extract_filter_criteria("which documents do we have") == {}


def test_match_compare_targets_identifies_exactly_two() -> None:
    cands = [("coc", "Code of Conduct"), ("hrh", "HR Handbook"), ("tp", "Travel Policy")]
    two = match_compare_targets("difference between the Code of Conduct and the HR Handbook", cands)
    assert set(two) == {"coc", "hrh"}


def test_match_compare_targets_falls_back_when_not_two() -> None:
    cands = [("coc", "Code of Conduct"), ("hrh", "HR Handbook"), ("tp", "Travel Policy")]
    # only one document is named → caller must fall back to rag (len != 2)
    assert match_compare_targets("compare the Travel Policy", cands) == ["tp"]
    # nothing named → no targets
    assert match_compare_targets("compare everything please", cands) == []


# --- doc_summary must mean "all the DOCUMENTS", not "all" and "document" in the same sentence ---
# Reported live on the SOP corpus: "now give similar detailed summary analysis for all the 10 SOPs
# mentioned in the document from PBSOP001, PBSOP002, ..." was routed to doc_summary, which tried
# to summarise the three FILES in the knowledge base and abstained with "I couldn't ground a
# summary for any document". The user was asking about ten SOPs INSIDE one document. "all"
# belonged to the SOPs and "the document" was singular.


def test_all_plus_a_singular_document_reference_is_not_a_doc_summary() -> None:
    q = (
        "now give similar detailed summary analysis for all the 10 SOPs mentioned in the document "
        "from PBSOP001, PBSOP002, PBSOP003, PBSOP004, PBSOP005"
    )
    assert classify_intent_heuristic(q) == "rag"


def test_content_items_inside_one_document_stay_rag() -> None:
    assert classify_intent_heuristic("summarize all the clauses in the document") == "rag"
    assert classify_intent_heuristic("explain every step described in the file") == "rag"


def test_a_real_per_document_summary_still_routes_to_doc_summary() -> None:
    """The intent must keep working for what it was built for."""
    for q in (
        "summarize each document",
        "summarize all documents",
        "give me a summary of every document in this KB",
        "explain all the uploaded files",
        "what does each of the documents say",
    ):
        assert classify_intent_heuristic(q) == "doc_summary", q
