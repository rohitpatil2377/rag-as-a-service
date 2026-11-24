"""The Phase-8 retrieval strategies: the recipes, and the aggregate section selection.

The selection is the part the plan states an acceptance criterion against, and it is pure — the
scores arrive as a mapping — so every claim here is assertable with no reranker and no catalog.
What is NOT here is whether a real reranker scores a real section above the threshold; that is
measured end to end against the AEGCL corrigendum in
``services/control-plane/tests/test_strategy_pipeline.py`` and in the acceptance run recorded in
``docs/rag_v2/``.
"""

from __future__ import annotations

import pytest

from rag_core.strategies import (
    STRATEGY_VERSION,
    SectionCard,
    contiguous_sections,
    levels_for_intent,
    max_scores,
    parent_sections,
    plan_strategy,
    select_per_target,
    select_sections,
)
from rag_schemas.common import ChunkLevel

_DOC = "corrigendum-v.pdf"


def _card(n: int, **kw: object) -> SectionCard:
    base: dict[str, object] = {
        "section_id": f"s{n}",
        "doc_id": _DOC,
        "breadcrumb": f"Section {n}",
        "ordinal": n,
    }
    base.update(kw)
    return SectionCard(**base)  # type: ignore[arg-type]


# --- the recipes ---------------------------------------------------------------------------------


def test_each_intent_has_a_strategy() -> None:
    """A missing row would silently fall back to lookup — the right fallback for an UNKNOWN
    intent, and the wrong one for an intent the router emits."""
    from rag_core.router import ROUTER_INTENTS

    for intent in ROUTER_INTENTS:
        assert plan_strategy(intent).intent == intent


def test_an_unknown_intent_falls_back_to_lookup() -> None:
    assert plan_strategy("enumerate").intent == "lookup"


def test_only_the_passage_strategies_ask_for_a_grain() -> None:
    """Aggregate and summarize read every section out of the catalog instead of searching, so a
    level filter would narrow a retrieval they do not run."""
    assert levels_for_intent("lookup") == (ChunkLevel.CHUNK, ChunkLevel.ROW, ChunkLevel.PROPOSITION)
    assert levels_for_intent("procedural") == levels_for_intent("lookup")
    assert levels_for_intent("aggregate") == ()
    assert levels_for_intent("summarize") == ()


def test_the_aggregate_class_is_exactly_the_section_pass() -> None:
    assert [
        i
        for i in ("lookup", "aggregate", "compare", "summarize", "procedural", "relational")
        if plan_strategy(i).section_pass
    ] == [
        "aggregate",
        "compare",
        "summarize",
    ]


# --- step 2: scoring -----------------------------------------------------------------------------


def test_a_section_keeps_its_best_score_across_the_sub_queries() -> None:
    """The measured reason for MAX rather than mean: a section scores ~0 against the vague main
    query and high against the one sub-query that is about it, and a mean would bury that."""
    best = max_scores([{"s1": 0.00001, "s2": 0.00001}, {"s1": 0.65}, {"s2": 0.02, "s1": 0.1}])
    assert best == {"s1": 0.65, "s2": 0.02}


def test_scoring_no_queries_scores_nothing() -> None:
    assert max_scores([]) == {}


# --- steps 3-6: selection ------------------------------------------------------------------------


def test_a_section_above_the_threshold_is_selected_on_its_score() -> None:
    sel = select_sections([_card(1), _card(2)], {"s1": 0.5, "s2": 0.01}, threshold=0.15)
    assert sel.section_ids == ("s1",)
    assert sel.selected[0].reasons == ("score",)
    assert sel.considered == 2


def test_a_facet_match_is_selected_whatever_it_scored() -> None:
    """Step 4 is a UNION, not a tie-break: the router read the document's own card and said this
    facet is what the question is about."""
    cards = [_card(1, facet_ids=("eligibility",)), _card(2)]
    sel = select_sections(cards, {"s1": 0.0, "s2": 0.0}, facets=((_DOC, "eligibility"),))
    assert sel.section_ids == ("s1",)
    assert sel.selected[0].reasons == ("facet",)


def test_a_facet_of_another_document_does_not_match() -> None:
    """The facet pair is (doc_id, facet_id) because two documents can both have `eligibility` and
    they are not the same facet."""
    cards = [_card(1, facet_ids=("eligibility",))]
    sel = select_sections(cards, {}, facets=(("other.pdf", "eligibility"),))
    assert sel.section_ids == ()


def test_the_amendment_view_pulls_in_every_amending_section() -> None:
    cards = [_card(1, amends=True), _card(2), _card(3, amends=True)]
    sel = select_sections(cards, {}, amendment_view=True)
    assert set(sel.section_ids) == {"s1", "s3"}


def test_the_amendment_view_is_off_by_default() -> None:
    sel = select_sections([_card(1, amends=True)], {})
    assert sel.section_ids == ()


def test_one_section_can_be_selected_for_several_reasons() -> None:
    cards = [_card(1, facet_ids=("f",), amends=True)]
    sel = select_sections(cards, {"s1": 0.9}, facets=((_DOC, "f"),), amendment_view=True)
    assert sel.selected[0].reasons == ("score", "facet", "amendment")


def test_a_section_nothing_scored_is_treated_as_zero_not_missing() -> None:
    """A reranker returns what it was given, but a caller merging several passes may simply not
    have a number for a section; absent must not crash and must not select."""
    sel = select_sections([_card(1)], {}, threshold=0.0)
    assert sel.section_ids == ("s1",), "threshold 0.0 selects everything, including the unscored"


# --- step 6: the cap -----------------------------------------------------------------------------


def test_the_cap_keeps_the_highest_scoring_and_records_the_rest() -> None:
    cards = [_card(i) for i in range(1, 5)]
    scores = {"s1": 0.2, "s2": 0.9, "s3": 0.5, "s4": 0.3}
    sel = select_sections(cards, scores, max_sections=2)
    assert sel.section_ids == ("s2", "s3")
    assert sel.dropped == ("s4", "s1"), "dropped in the order they lost, strongest first"


def test_the_cap_does_not_undo_an_explicit_union() -> None:
    """The deliberate reading of step 6. A literal "keep highest-scoring" would let the cap drop
    the amendment table — the single most important thing in the plan's own acceptance document —
    in favour of a better-scoring ordinary section, making steps 4 and 5 conditional on a number
    they never see."""
    cards = [_card(1, amends=True), _card(2), _card(3)]
    scores = {"s1": 0.01, "s2": 0.9, "s3": 0.8}
    sel = select_sections(cards, scores, amendment_view=True, max_sections=2)
    assert sel.section_ids == ("s1", "s2"), "the amending section survives a cap it loses on score"
    assert sel.dropped == ("s3",)


def test_a_tie_breaks_in_document_order() -> None:
    cards = [_card(3), _card(1), _card(2)]
    sel = select_sections(cards, {"s1": 0.5, "s2": 0.5, "s3": 0.5})
    assert sel.section_ids == ("s1", "s2", "s3")


def test_nothing_selected_is_a_real_result_not_an_error() -> None:
    """All-zero scores, no facets, no amendment view: there is nothing for either floor to be a
    fraction of, and an empty selection is a result rather than a failure."""
    sel = select_sections([_card(1), _card(2)], {"s1": 0.0, "s2": 0.0})
    assert sel.section_ids == () and sel.considered == 2
    assert "selected" in sel.as_json()


# --- step 3: the relative floor ------------------------------------------------------------------


def test_the_relative_floor_selects_what_the_absolute_one_misses() -> None:
    """The measured case. On the real corrigendum every section scored below 0.15 — below 0.001,
    in fact — while still separating by three orders of magnitude, so only the ratio carries
    the signal."""
    cards = [_card(1), _card(2), _card(3)]
    scores = {"s1": 0.00046, "s2": 0.00030, "s3": 0.0000012}

    absolute_only = select_sections(cards, scores, relative=0.0)
    with_relative = select_sections(cards, scores, relative=0.1)

    assert absolute_only.section_ids == (), "0.15 against a 4.6e-4 best selects nothing"
    assert with_relative.section_ids == ("s1", "s2"), "and the 390x-weaker section stays out"


def test_the_relative_floor_never_raises_the_bar_above_the_absolute_one() -> None:
    """A well-separated passage-scale result must not become HARDER to select because one
    section scored unusually high — the two floors are an OR, not a replacement."""
    cards = [_card(1), _card(2)]
    sel = select_sections(cards, {"s1": 0.95, "s2": 0.2}, threshold=0.15, relative=0.5)
    assert sel.section_ids == ("s1", "s2"), "0.2 clears the absolute floor even at 0.5 x 0.95"


def test_the_applied_floor_is_what_gets_recorded() -> None:
    """A stored selection has to say which number produced it; on a real document the floor
    applied and the floor requested differ by orders of magnitude."""
    sel = select_sections([_card(1)], {"s1": 0.004}, threshold=0.15, relative=0.1)
    assert sel.threshold == pytest.approx(0.0004)


def test_the_relative_floor_is_off_at_zero() -> None:
    sel = select_sections([_card(1)], {"s1": 0.14}, threshold=0.15, relative=0.0)
    assert sel.section_ids == ()


def test_the_selection_serialises_with_its_version() -> None:
    import json

    sel = select_sections([_card(1)], {"s1": 0.5})
    stored = json.loads(sel.as_json())
    assert stored["strategy_version"] == STRATEGY_VERSION
    assert stored["selected"] == [{"section_id": "s1", "doc_id": _DOC, "score": 0.5, "reasons": ["score"]}]


# --- the rerank candidate ------------------------------------------------------------------------


def test_the_rerank_text_leads_with_the_breadcrumb() -> None:
    card = _card(1, breadcrumb="Attachment-1 > DRS > Switches", heading="Switches", summary="UL 62368, CSA 22.2.")
    assert card.rerank_text().startswith("Attachment-1 > DRS > Switches")
    assert "UL 62368" in card.rerank_text()


def test_the_heading_is_not_repeated_when_the_breadcrumb_already_ends_in_it() -> None:
    card = _card(1, breadcrumb="DRS > Switches", heading="Switches")
    assert card.rerank_text().count("Switches") == 1


def test_an_unenriched_section_still_has_candidate_text() -> None:
    """The ordinary partial-enrichment case: a heading alone carries real signal for a facet
    sub-query, and a section with no candidate text at all could never be selected."""
    assert _card(1, breadcrumb="", heading="Video Wall Technical Requirements").rerank_text()


def test_the_candidate_is_truncated_and_the_query_never_is() -> None:
    card = _card(1, summary="x" * 5000)
    assert len(card.rerank_text()) <= 1200


# --- expansion -----------------------------------------------------------------------------------


def test_parents_dedupe_but_keep_the_ranking() -> None:
    assert parent_sections(["s3", "s1", "s3", "s2"]) == ("s3", "s1", "s2")


def test_a_chunk_with_no_section_is_skipped_not_grouped_under_null() -> None:
    assert parent_sections([None, "s1", None]) == ("s1",)


def test_procedural_fills_the_gaps_between_matched_steps() -> None:
    cards = [_card(i) for i in range(1, 6)]
    assert contiguous_sections(["s1", "s3", "s5"], cards) == ("s1", "s2", "s3", "s4", "s5")


def test_procedural_does_not_fill_a_gap_wider_than_the_bound() -> None:
    """Two distant matches are two places the answer was found, not one procedure with a hole."""
    cards = [_card(i) for i in range(1, 21)]
    assert contiguous_sections(["s1", "s20"], cards) == ("s1", "s20")


def test_procedural_returns_document_order_not_ranked_order() -> None:
    cards = [_card(i) for i in range(1, 4)]
    assert contiguous_sections(["s3", "s1"], cards) == ("s1", "s2", "s3")


def test_procedural_expands_each_document_separately() -> None:
    """A gap is a gap WITHIN a document; two documents' sections are not one sequence."""
    cards = [_card(1), _card(2), _card(3), SectionCard(section_id="o1", doc_id="other.pdf", ordinal=1)]
    assert contiguous_sections(["s1", "s3", "o1"], cards) == ("s1", "s2", "s3", "o1")


def test_a_gap_is_counted_in_sections_present_not_in_ordinals() -> None:
    """``cards`` is the document's COMPLETE section list (that is what ``document_sections``
    returns), so "the sections between these two" is their distance in that list. Ordinals can be
    sparse — a section the extractor dropped leaves a hole in the numbering, not in the document
    — and counting the gap in ordinals would refuse to fill a run that has no gap at all."""
    cards = [_card(1), _card(50), _card(99)]
    assert contiguous_sections(["s1", "s99"], cards) == ("s1", "s50", "s99")


# --- compare: the budget is split per target -----------------------------------------------------


def _other(n: int, **kw: object) -> SectionCard:
    base: dict[str, object] = {"section_id": f"o{n}", "doc_id": "other.pdf", "breadcrumb": f"Other {n}", "ordinal": n}
    base.update(kw)
    return SectionCard(**base)  # type: ignore[arg-type]


def test_compare_gives_each_document_a_share_of_the_cap() -> None:
    """A shared cap is not a comparison: the larger document takes every slot and the answer
    compares a document with itself while reporting that it compared two."""
    cards = [_card(i) for i in range(1, 7)] + [_other(1), _other(2)]
    scores = {f"s{i}": 0.9 - i * 0.01 for i in range(1, 7)} | {"o1": 0.3, "o2": 0.2}

    shared = select_sections(cards, scores, max_sections=4)
    split = select_per_target(cards, scores, max_sections=4)

    assert {s.doc_id for s in shared.selected} == {_DOC}, "the big document took the whole budget"
    assert {s.doc_id for s in split.selected} == {_DOC, "other.pdf"}, "both sides are represented"
    assert len(split.selected) == 4


def test_the_floor_is_shared_across_targets_not_recomputed_per_document() -> None:
    """A relative floor computed per document would mean "the best section IN THIS DOCUMENT", so
    a document with nothing relevant would contribute its best irrelevant section as a match.
    The floor is a property of the question."""
    cards = [_card(1), _other(1)]
    scores = {"s1": 0.5, "o1": 0.000001}  # the other document has nothing

    split = select_per_target(cards, scores, relative=0.1, max_sections=10)

    assert [s.section_id for s in split.selected] == ["s1"]


def test_one_document_in_scope_is_the_ordinary_single_pass() -> None:
    cards = [_card(i) for i in range(1, 4)]
    scores = {"s1": 0.9, "s2": 0.8, "s3": 0.7}
    assert (
        select_per_target(cards, scores, max_sections=2).section_ids
        == select_sections(cards, scores, max_sections=2).section_ids
    )


def test_a_capped_target_records_what_it_dropped() -> None:
    cards = [_card(1), _card(2), _other(1), _other(2)]
    scores = {"s1": 0.9, "s2": 0.8, "o1": 0.7, "o2": 0.6}

    split = select_per_target(cards, scores, max_sections=2)

    assert len(split.selected) == 2 and len(split.dropped) == 2
    assert split.considered == 4


# --- what the facet union is WORTH, not just that it fires -----------------------------------


def test_removing_the_facet_union_costs_coverage_a_future_simplification_must_recover() -> None:
    """Step 4 is LOAD-BEARING, measured, and this test exists so that is not forgotten.

    `test_a_facet_match_is_selected_whatever_it_scored` proves the mechanism fires. It does not
    say what the mechanism is WORTH, and a simplification that dropped step 4 would pass it by
    deleting the facet argument and the assertion together.

    MEASURED ON THE REAL CORPUS (Experiment A, 2026-11-02, `zzgoal-cf3b136d`, 53 sections, the
    aggregate question, production budget, and a selection byte-identical in set AND order to
    what production selected in two preserved cycles): the score floor alone covered **25 of the
    37** sections in the derived evidence set, and the facet union added **7 more, every one of
    them in that set** -- 67.6% -> 86.5%, +18.9 points, with nothing wasted. The cross-encoder
    did not rank those 7 low; it scored them BELOW THE FLOOR, so no threshold change recovers
    them. Only the router's facets do.

    This fixture is that measurement in miniature, with the same arithmetic: 53 sections, 37
    evidence-bearing, 25 reachable by score, 7 reachable only by facet. If a change to
    `select_sections` makes the facet union redundant, this test fails and the replacement has to
    show where the 7 came back from."""
    evidence = {f"s{n}" for n in range(1, 38)}  # 37 of 53 carry the answer's evidence
    by_score = {f"s{n}" for n in range(1, 26)}  # 25 of those clear the floor
    by_facet = {f"s{n}" for n in range(26, 33)}  # 7 more are only reachable as a facet match
    cards = [_card(n, facet_ids=("eligibility",) if f"s{n}" in by_facet else ()) for n in range(1, 54)]
    # Below the ABSOLUTE floor and below the RELATIVE one: 0.001 against a best of 0.9 is 0.1%,
    # under the 10% relative floor, so neither floor reaches these and the union is the only path.
    scores = {f"s{n}": (0.9 if f"s{n}" in by_score else 0.001) for n in range(1, 54)}

    score_only = select_sections(cards, scores, facets=(), threshold=0.15, relative=0.1)
    with_facets = select_sections(cards, scores, facets=((_DOC, "eligibility"),), threshold=0.15, relative=0.1)

    covered_before = evidence & set(score_only.section_ids)
    covered_after = evidence & set(with_facets.section_ids)
    added = set(with_facets.section_ids) - set(score_only.section_ids)

    assert len(covered_before) == 25, "the score floor alone reaches 25 of the 37"
    assert len(covered_after) == 32, "the union carries it to 32 — Experiment A's 86.5%"
    assert added == by_facet, "and exactly the 7 the facets named"
    assert added <= evidence, "every section the union added is evidence-bearing — no waste"
    assert all("facet" in s.reasons for s in with_facets.selected if s.section_id in by_facet), (
        "and they are admitted AS facet matches, so the reason survives into the audit"
    )
