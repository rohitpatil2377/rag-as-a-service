"""The query router: intent, scope, facets, sub-queries (RAG v2 Phase 7).

WHAT THIS FILE CAN AND CANNOT CLAIM. The plan's Phase-7 acceptance is "the router labels a
40-query mixed test set (owner + Appendix A) with >= 90% intent accuracy". Neither half of that
set exists here: the owner's queries have not been supplied and Appendix A is the AEGCL
corrigendum (discovery §7, §9.1). So :func:`test_intent_accuracy_on_a_composed_forty_query_set`
measures the same thing at the same size against a set composed HERE, and that is weaker evidence
in one specific way worth naming: the queries are phrased the way I expect queries to be phrased,
so the measurement bounds the classifier's logic and not its coverage of how this corpus's users
actually write. The acceptance stays UNVERIFIED (§8.24).

AND THE SET AND THE CLASSIFIER CO-EVOLVED, which bounds the number further. The first run scored
38/40; both misses were real bugs the set exposed — "amend" matched as a SUBSTRING turned "tl;dr
of the amendment" (a summarize question whose topic is an amendment) into a relational one, and
"refers to" missed "which clauses does this corrigendum refer to" because the verb was inflected
differently — and fixing them took it to 40/40. The fixes are principled (word boundaries, verb
inflections) rather than query-specific, but a score measured on the set that shaped the code is
not an independent measurement, and 40/40 should be read as "no known ordering or matching defect"
rather than as an accuracy estimate.

What the set IS good for is the thing a cue-table classifier gets wrong: ORDER. Half these cases
exist because two cues collide — "summarise all the obligations" carries a summarize cue and an
aggregate cue, "what are the steps" carries a procedural cue and an aggregate cue, "the
relationship between A and B" carries a relational cue and a compare cue. The plan's definitions
decide each one, and a regression in the ordering is what this set catches.
"""

from __future__ import annotations

import json

from rag_core.router import (
    ROUTER_INTENTS,
    ROUTER_VERSION,
    DocCard,
    RouterDecision,
    RouterScope,
    build_router_prompt,
    classify_router_intent,
    heuristic_decision,
    match_facets,
    parse_router_decision,
    plan_for_intent,
)

_CARD = DocCard(
    doc_id="corrigendum-v.pdf",
    title="Corrigendum-V",
    doc_type="corrigendum",
    purpose="Amends the bidding document.",
    facet_ids=("financial_qualification", "technical_specification", "eligibility", "warranty"),
)
_OTHER = DocCard(doc_id="tender.pdf", title="Tender", doc_type="tender", purpose="x", facet_ids=("scope_of_work",))
_SCOPE = RouterScope(cards=(_CARD, _OTHER), explicit=True)
_SAMPLE = RouterScope(cards=(_CARD, _OTHER), explicit=False)

# --- the 40-query intent set --------------------------------------------------------------------
#
# Composed here; see the module docstring for what that is and is not worth. Mixed across the
# plan's six intents, with the collisions the ordering has to resolve deliberately over-sampled.
_LABELLED: tuple[tuple[str, str], ...] = (
    # lookup — a specific fact
    ("who approves an SOP?", "lookup"),
    ("what is the earnest money deposit?", "lookup"),
    ("when does the bid close?", "lookup"),
    ("is a site visit mandatory?", "lookup"),
    ("what is the minimum average annual turnover?", "lookup"),
    ("which designation authorises a deviation?", "lookup"),
    ("where should the bid be submitted?", "lookup"),
    ("what is the warranty period?", "lookup"),
    # aggregate — all / list / details / requirements of X
    ("list all the compliance requirements", "aggregate"),
    ("what are the eligibility criteria?", "aggregate"),
    ("give me the details of the technical specification", "aggregate"),
    ("every document the bidder must submit", "aggregate"),
    ("what are the obligations of the vendor?", "aggregate"),
    ("a complete list of the penalties", "aggregate"),
    ("enumerate the deliverables", "aggregate"),
    ("all the security requirements in this corrigendum", "aggregate"),
    # the collision: a summarize verb with an aggregate scope word. The plan's definitions make
    # this aggregate — it wants exhaustive coverage, which is what aggregate means.
    ("summarise all the obligations", "aggregate"),
    ("give me the full list of annexures", "aggregate"),
    # compare — two or more entities set against each other
    ("compare the warranty terms of the two bidders", "compare"),
    ("what is the difference between Corrigendum-II and Corrigendum-V?", "compare"),
    ("how does the revised turnover differ from the original?", "compare"),
    ("bidder A versus bidder B on technical score", "compare"),
    ("which of the two options is better for us?", "compare"),
    ("compare the eligibility criteria across both tenders", "compare"),
    # summarize — an overview
    ("summarise this document", "summarize"),
    ("give me an overview of the corrigendum", "summarize"),
    ("what is this document about?", "summarize"),
    ("briefly, what does the tender cover?", "summarize"),
    ("a high-level view of the scope", "summarize"),
    ("tl;dr of the amendment", "summarize"),
    # procedural — steps, in order
    ("what are the steps to revise an SOP?", "procedural"),
    ("how do I submit a bid?", "procedural"),
    ("what is the procedure for requesting an extension?", "procedural"),
    ("walk me through the evaluation process", "procedural"),
    ("in what order are the annexures signed?", "procedural"),
    ("how to raise a clarification request", "procedural"),
    # relational — links between documents or entities
    ("which documents amend this tender?", "relational"),
    ("what supersedes Corrigendum-II?", "relational"),
    ("what is the relationship between the tender and its annexures?", "relational"),
    ("which clauses does this corrigendum refer to?", "relational"),
)

_ACCEPTANCE_FLOOR = 0.90


def test_intent_accuracy_on_a_composed_forty_query_set() -> None:
    """The plan's acceptance SHAPE, measured on a set composed here rather than the owner's.

    Reported per intent as well as overall: an aggregate that hides one intent catching nothing
    would clear the bar while leaving a whole label unroutable."""
    assert len(_LABELLED) == 40, "the plan's acceptance is stated at this size"
    wrong: list[tuple[str, str, str]] = []
    by_intent: dict[str, list[bool]] = {}
    for query, want in _LABELLED:
        got = classify_router_intent(query)
        by_intent.setdefault(want, []).append(got == want)
        if got != want:
            wrong.append((query, want, got))

    accuracy = (len(_LABELLED) - len(wrong)) / len(_LABELLED)
    assert accuracy >= _ACCEPTANCE_FLOOR, f"intent accuracy {accuracy:.0%} < {_ACCEPTANCE_FLOOR:.0%}: {wrong}"
    for intent, hits in by_intent.items():
        assert any(hits), f"every {intent} query was mislabelled"


def test_every_label_in_the_set_is_one_the_router_emits() -> None:
    assert {want for _, want in _LABELLED} <= ROUTER_INTENTS


def test_an_unmatched_query_is_a_lookup() -> None:
    """The default never mis-routes: it is the behaviour of the pipeline with the router off."""
    assert classify_router_intent("the controller shall comply with IEC 62443-4-2") == "lookup"
    assert classify_router_intent("") == "lookup"


# --- the ordering the cue tables depend on ---------------------------------------------------------


def test_relational_is_checked_before_compare() -> None:
    """ "The relationship between A and B" is a link question, not a comparison of them."""
    assert classify_router_intent("what is the relationship between the tender and the corrigendum?") == "relational"


def test_procedural_is_checked_before_aggregate() -> None:
    """ "What are the steps" carries an aggregate cue too; the plan defines procedural as steps,
    and steps want an order rather than a list."""
    assert classify_router_intent("what are the steps to appeal a rejection?") == "procedural"


def test_aggregate_is_checked_before_summarize() -> None:
    """An exhaustive request phrased with a summarize verb is still exhaustive."""
    assert classify_router_intent("summarise all the requirements") == "aggregate"
    assert classify_router_intent("summarise the document") == "summarize"


# --- facets ------------------------------------------------------------------------------------------


def test_a_query_using_a_facets_own_words_matches_it() -> None:
    """The deterministic half of the plan's "map query words onto each document's own facets": a
    facet id is snake_case drawn from the document's content, so its tokens ARE that document's
    vocabulary for the area."""
    assert match_facets("what is the financial qualification threshold?", _SCOPE) == (
        ("corrigendum-v.pdf", "financial_qualification"),
    )


def test_a_two_token_facet_matches_on_either_of_its_words() -> None:
    """Deliberately loose, and the looseness is the point: a real query almost never restates a
    full snake_case id, so a strict-majority rule would mean the matcher never fires. The cost of
    a loose match is one extra facet-specific sub-query; the cost of a strict one is losing the
    only deterministic facet signal the router has."""
    assert match_facets("what are the financial arrangements?", _SCOPE) == (
        ("corrigendum-v.pdf", "financial_qualification"),
    )


def test_a_facet_whose_words_the_query_never_uses_does_not_match() -> None:
    assert match_facets("when does the bid close?", _SCOPE) == ()


def test_a_three_token_facet_needs_two_of_its_words() -> None:
    scope = RouterScope(cards=(DocCard(doc_id="d", title="t", facet_ids=("supply_chain_security",)),))
    assert match_facets("what about supply logistics?", scope) == ()
    assert match_facets("what are the supply chain rules?", scope) == (("d", "supply_chain_security"),)


def test_a_short_facet_token_is_ignored() -> None:
    """They are the words every document shares."""
    scope = RouterScope(cards=(DocCard(doc_id="d", title="t", facet_ids=("the_fee",)),))
    assert match_facets("what is the cost?", scope) == ()


def test_facets_are_matched_per_document() -> None:
    matched = match_facets("what is the scope of work and the eligibility?", _SCOPE)
    assert set(matched) == {("tender.pdf", "scope_of_work"), ("corrigendum-v.pdf", "eligibility")}


# --- the heuristic decision ----------------------------------------------------------------------------


def test_an_explicit_scope_becomes_the_decisions_doc_ids() -> None:
    decision = heuristic_decision("what is the warranty period?", _SCOPE, query_lang="en")
    assert decision.doc_ids == ("corrigendum-v.pdf", "tender.pdf")


def test_a_sampled_scope_does_not_become_doc_ids() -> None:
    """A sample of the KB is not a routing decision about those documents, and echoing it back
    would scope retrieval to an arbitrary sample — worse than no router at all."""
    assert heuristic_decision("what is the warranty period?", _SAMPLE, query_lang="en").doc_ids == ()


def test_sub_queries_are_produced_only_for_aggregate_and_compare() -> None:
    aggregate = heuristic_decision(
        "list all the financial qualification and technical specification requirements", _SCOPE
    )
    lookup = heuristic_decision("what is the financial qualification and technical specification?", _SCOPE)

    assert len(aggregate.sub_queries) == 2
    assert all("financial qualification" in s or "technical specification" in s for s in aggregate.sub_queries)
    assert lookup.sub_queries == (), "decomposing a single-fact lookup costs retrieval for no gain"


def test_a_sub_query_is_a_question_about_the_facet_not_the_query_plus_a_word() -> None:
    """Measured on the real AEGCL corrigendum (Phase 8): appending the facet's words to a vague
    aggregate query made the string LESS discriminating than the bare query, because every vague
    word stays and one is added. A question about the facet alone is what the cross-encoder can
    actually separate on."""
    aggregate = heuristic_decision("list all the financial qualification and technical specification terms", _SCOPE)

    assert set(aggregate.sub_queries) == {
        "What are the financial qualification requirements?",
        "What are the technical specification requirements?",
    }
    assert all("list all" not in s for s in aggregate.sub_queries), "the vague original is not carried along"


def test_one_matched_facet_yields_no_sub_queries() -> None:
    """One sub-query is the query again."""
    assert heuristic_decision("list all the warranty terms", _SCOPE).sub_queries == ()


def test_the_amendment_view_is_flagged_by_a_currency_question() -> None:
    assert heuristic_decision("what is the current turnover requirement?", _SCOPE).needs_amendment_view is True
    assert heuristic_decision("what is the turnover requirement?", _SCOPE).needs_amendment_view is False


def test_an_empty_scope_still_decides_an_intent() -> None:
    """Ordinary, not an error: the card is written only on the v2 ingest path, so every KB has no
    cards until a document is ingested with the flag on."""
    decision = heuristic_decision("list all the requirements", RouterScope())
    assert decision.intent == "aggregate" and decision.facets == () and decision.doc_ids == ()


# --- the prompt ------------------------------------------------------------------------------------------


def test_the_prompt_shows_each_card_and_its_facets() -> None:
    prompt = build_router_prompt("what are the financial requirements?", _SCOPE)
    assert "corrigendum-v.pdf" in prompt and "corrigendum" in prompt
    assert "financial_qualification" in prompt
    assert "Amends the bidding document." in prompt


def test_the_prompt_says_so_when_no_card_is_in_scope() -> None:
    """Rather than rendering an empty section, which a model reads as a missing input."""
    assert "no document in scope has a profile card" in build_router_prompt("q", RouterScope())


# --- parsing a reply ----------------------------------------------------------------------------------------


def _reply(**overrides: object) -> str:
    body: dict[str, object] = {
        "intent": "aggregate",
        "doc_ids": ["corrigendum-v.pdf"],
        "facets": [{"doc_id": "corrigendum-v.pdf", "facet_id": "financial_qualification"}],
        "sub_queries": ["what is the turnover floor?", "what certificates are required?"],
        "query_lang": "en",
        "answer_lang": "en",
        "needs_amendment_view": True,
        "rationale": "asks for all requirements",
        "confidence": 0.8,
    }
    body.update(overrides)
    return json.dumps(body)


def test_a_usable_reply_becomes_a_decision() -> None:
    parse = parse_router_decision(_reply(), scope=_SCOPE, query_lang="en")
    assert not parse.fell_back
    decision = parse.decision
    assert decision.intent == "aggregate"
    assert decision.doc_ids == ("corrigendum-v.pdf",)
    assert decision.facets == (("corrigendum-v.pdf", "financial_qualification"),)
    assert len(decision.sub_queries) == 2 and decision.needs_amendment_view is True


def test_a_doc_id_outside_the_scope_is_dropped_and_reported() -> None:
    """`doc_ids` scopes retrieval, and scoping it to a document this question cannot see
    retrieves nothing — a confident, cited, empty answer."""
    parse = parse_router_decision(_reply(doc_ids=["corrigendum-v.pdf", "ghost.pdf"]), scope=_SCOPE, query_lang="en")
    assert parse.decision.doc_ids == ("corrigendum-v.pdf",)
    assert parse.unknown_docs == ("ghost.pdf",)


def test_a_facet_the_card_does_not_define_is_dropped_and_reported() -> None:
    parse = parse_router_decision(
        _reply(facets=[{"doc_id": "corrigendum-v.pdf", "facet_id": "invented_area"}]), scope=_SCOPE, query_lang="en"
    )
    assert parse.decision.facets == ()
    assert parse.unknown_facets == ("corrigendum-v.pdf:invented_area",)


def test_a_facet_label_is_normalised_before_it_is_matched() -> None:
    """Shared `snake_id` with Phase 3, so the label the router returns and the id the taxonomy
    stored agree."""
    parse = parse_router_decision(
        _reply(facets=[{"doc_id": "corrigendum-v.pdf", "facet_id": "Financial Qualification"}]),
        scope=_SCOPE,
        query_lang="en",
    )
    assert parse.decision.facets == (("corrigendum-v.pdf", "financial_qualification"),)


def test_sub_queries_are_dropped_for_an_intent_the_plan_does_not_give_them_to() -> None:
    parse = parse_router_decision(_reply(intent="lookup"), scope=_SCOPE, query_lang="en")
    assert parse.decision.sub_queries == ()
    assert parse.dropped_sub_queries == 2


def test_a_single_sub_query_is_dropped() -> None:
    parse = parse_router_decision(_reply(sub_queries=["only one"]), scope=_SCOPE, query_lang="en")
    assert parse.decision.sub_queries == ()


def test_the_callers_language_detection_wins_over_the_models() -> None:
    """The same correction Phase 3 made after a real-model run returned the prose word "english"
    where the platform's columns hold ISO codes."""
    parse = parse_router_decision(_reply(query_lang="english"), scope=_SCOPE, query_lang="en")
    assert parse.decision.query_lang == "en"


def test_an_answer_language_is_taken_only_when_it_is_a_code() -> None:
    """It is the one language field a user can legitimately ask to differ from the query's."""
    assert parse_router_decision(_reply(answer_lang="hi"), scope=_SCOPE, query_lang="en").decision.answer_lang == "hi"
    assert (
        parse_router_decision(_reply(answer_lang="Hindi"), scope=_SCOPE, query_lang="en").decision.answer_lang == "en"
    )


def test_an_unknown_intent_falls_back_rather_than_being_coerced() -> None:
    """A guessed intent selects a different answer mechanism; the plan's own fallback is lookup."""
    parse = parse_router_decision(_reply(intent="enumerate"), scope=_SCOPE, query_lang="en")
    assert parse.fell_back and parse.decision.intent == "lookup"


def test_an_unusable_reply_falls_back_to_the_supplied_decision() -> None:
    """A model failure degrades to the DETERMINISTIC backend, not to nothing — the pipeline has
    to route the question somewhere."""
    safe = heuristic_decision("list all the requirements", _SCOPE, query_lang="en")
    for reply in ("not json", "{broken", json.dumps([1, 2])):
        parse = parse_router_decision(reply, scope=_SCOPE, query_lang="en", fallback=safe)
        assert parse.fell_back and parse.decision.intent == "aggregate" and parse.error


def test_a_fenced_reply_is_read() -> None:
    assert not parse_router_decision(f"```json\n{_reply()}\n```", scope=_SCOPE, query_lang="en").fell_back


# --- mapping onto the existing taxonomy (discovery §8.4) -------------------------------------------------------


def test_each_intent_maps_onto_an_existing_strategy() -> None:
    """A v2 decision selects an EXISTING answer mechanism, so the router can ship before Phases
    8-9 without the pipeline needing a second routing vocabulary."""
    for intent in ROUTER_INTENTS:
        plan = plan_for_intent(intent)
        assert plan.strategy in {"passages", "whole_document", "compare", "whole_kb", "catalog", "count", "filter"}


def test_a_whole_document_strategy_needs_exactly_one_target() -> None:
    """Two candidates mean the router did not narrow it to one, and the passage path is then the
    honest mechanism."""
    one = plan_for_intent("summarize", RouterDecision(intent="summarize", doc_ids=("a.pdf",)))
    two = plan_for_intent("summarize", RouterDecision(intent="summarize", doc_ids=("a.pdf", "b.pdf")))

    assert one.strategy == "whole_document" and one.target_doc == "a.pdf"
    assert two.strategy == "passages" and two.target_doc is None


# --- what is stored ---------------------------------------------------------------------------------------------


def test_the_decision_json_is_the_router_log_shape() -> None:
    """Phase 12 gave `query_log.router_json` its writer and left it NULL with a TODO(Phase-7);
    this is what fills it."""
    decision = parse_router_decision(_reply(), scope=_SCOPE, query_lang="en").decision
    stored = json.loads(decision.as_json())

    assert stored["router_version"] == ROUTER_VERSION
    assert stored["intent"] == "aggregate"
    assert stored["facets"] == [{"doc_id": "corrigendum-v.pdf", "facet_id": "financial_qualification"}]
    assert stored["needs_amendment_view"] is True
