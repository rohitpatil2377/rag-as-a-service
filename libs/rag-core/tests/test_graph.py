"""Unit tests for GraphRAG extraction (FR-C6) — pure logic."""

from __future__ import annotations

from rag_core import HeuristicGraphExtractor, cooccurrence_edges, extract_entities, normalize_entity


def test_extracts_proper_nouns_not_filler() -> None:
    text = "The Acme Torch Bearers Policy rewards employees. This section explains eligibility."
    ents = extract_entities(text)
    assert "Acme Torch Bearers Policy" in ents
    # Sentence-opener filler is not an entity.
    assert not any(e.lower() in {"the", "this", "section"} for e in ents)


def test_dedupes_case_insensitively() -> None:
    ents = extract_entities("Acme leads. Later, ACME again and Acme once more.")
    assert sum(1 for e in ents if e.lower() == "acme") == 1


def test_cooccurrence_edges_are_sorted_pairs() -> None:
    edges = cooccurrence_edges(["Zebra Corp", "Alpha Ltd"])
    assert edges == [("Alpha Ltd", "Zebra Corp")]


def test_normalize_collapses_and_strips() -> None:
    assert normalize_entity("  Acme   AIOps .") == "Acme AIOps"


async def test_heuristic_extractor_builds_triple() -> None:
    triple = await HeuristicGraphExtractor().extract("Anita Desai works at Acme in Nashik.")
    assert {"Anita Desai", "Acme", "Nashik"} <= set(triple.entities)
    # Entities in one chunk are pairwise related.
    assert len(triple.edges) == len(triple.entities) * (len(triple.entities) - 1) // 2


# --- image-document co-occurrence noise filter (additive, image-only) --------
from rag_core.graph import filter_image_text_entities  # noqa: E402


def test_filter_image_text_entities_drops_single_word_scaffolding() -> None:
    survivors = filter_image_text_entities(["Architecture", "SQL", "Image", "Load Balancer"])
    assert survivors == ["Load Balancer"]


def test_filter_image_text_entities_drops_bare_generic_categories() -> None:
    survivors = filter_image_text_entities(["API Servers", "Databases", "Primary Database", "API Server 1"])
    assert survivors == ["Primary Database", "API Server 1"]


def test_filter_image_text_entities_keeps_specific_component_names() -> None:
    entities = ["Load Balancer", "API Gateway", "Primary Database", "Message Queue Broker"]
    assert filter_image_text_entities(entities) == entities


# --- Slice 2: communities + global/thematic search ---------------------------
from rag_core.graph import (  # noqa: E402
    GraphCommunity,
    detect_communities,
    is_global_query,
    rank_communities,
    summarize_community_heuristic,
)


def test_detect_communities_separates_disconnected_clusters() -> None:
    # Two triangles with no edge between them -> two communities.
    edges = [("a", "b"), ("b", "c"), ("a", "c"), ("x", "y"), ("y", "z"), ("x", "z")]
    labels = detect_communities(["a", "b", "c", "x", "y", "z"], edges)
    assert labels["a"] == labels["b"] == labels["c"]
    assert labels["x"] == labels["y"] == labels["z"]
    assert labels["a"] != labels["x"]


def test_detect_communities_is_deterministic() -> None:
    edges = [("a", "b"), ("b", "c"), ("d", "e")]
    nodes = ["a", "b", "c", "d", "e"]
    assert detect_communities(nodes, edges) == detect_communities(nodes, edges)


def test_detect_communities_ids_dense_and_largest_first() -> None:
    # One 3-clique + one isolated pair -> community 0 is the larger cluster.
    edges = [("a", "b"), ("b", "c"), ("a", "c"), ("x", "y")]
    labels = detect_communities(["a", "b", "c", "x", "y"], edges)
    assert labels["a"] == 0  # largest community re-numbered to 0
    assert set(labels.values()) == {0, 1}


def test_is_global_query_detects_thematic_questions() -> None:
    assert is_global_query("What are the main themes across the KB?")
    assert is_global_query("Give me an overview")
    assert is_global_query("Summarise the handbook")


def test_is_global_query_rejects_specific_questions() -> None:
    assert not is_global_query("What is the night shift allowance for a level 3 engineer?")
    assert not is_global_query("Who leads Project Falcon?")


def test_rank_communities_prefers_query_overlap_then_size() -> None:
    travel = GraphCommunity(0, "Travel", size=3, top_entities=["Business Travel", "Reimbursement"], summary="travel")
    leave = GraphCommunity(1, "Leave", size=9, top_entities=["Casual Leave", "Sick Leave"], summary="leave")
    ranked = rank_communities("what about business travel reimbursement", [leave, travel])
    assert ranked[0].community_id == 0  # smaller but term-relevant wins
    # A generic query with no distinctive terms falls back to size.
    ranked_generic = rank_communities("overview", [travel, leave])
    assert ranked_generic[0].community_id == 1


def test_summarize_community_heuristic_names_entities() -> None:
    s = summarize_community_heuristic(["Acme", "Central Government"], 5)
    assert "Acme" in s and "5" in s
