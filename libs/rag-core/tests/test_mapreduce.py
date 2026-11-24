"""The Phase-9 map/reduce contract: the item parse, the quote rule, coverage, and citation
resolution.

Everything here is pure — replies in, validated items out — so the rule the whole phase rests on
is assertable with no gateway. What is NOT here is whether a real model extracts the right items
from a real section; that is the phase's acceptance and is measured against the AEGCL
corrigendum (see ``docs/rag_v2/phase9_aegcl.md``).
"""

from __future__ import annotations

import json

from rag_core.mapreduce import (
    MAP_VERSION,
    UNFACETED,
    Coverage,
    MapParse,
    MappedItem,
    ReduceInput,
    assemble_answer,
    build_map_prompt,
    build_reduce_prompt,
    group_by_facet,
    items_block,
    map_cache_key,
    parse_map_items,
    query_signature,
    resolve_item_chunk,
)

_SECTION = "sec-maat"
_DOC = "corrigendum-v.pdf"
_TEXT = (
    "The Bidder's income sheet should show Minimum Average Annual Turnover (MAAT) of Rs. 35 Crores "
    "for the best three years in the last 5 years. Is amended as follows- The Bidder's income sheet "
    "should show Minimum Average Annual Turnover (MAAT) of Rs. 19.44 Crores for the best three years."
)
_FACETS = ("financial_qualification", "eligibility")


def _reply(**overrides: object) -> str:
    body: dict[str, object] = {
        "facet_id": "financial_qualification",
        "statement": "The minimum average annual turnover is reduced to Rs. 19.44 Crores.",
        "values": [{"name": "MAAT", "value": "19.44", "unit": "Rs. Crores"}],
        "clause_ref": "Sl. 5",
        "is_amendment": True,
        "old_value": "Rs. 35 Crores",
        "new_value": "Rs. 19.44 Crores",
        "quote": "Minimum Average Annual Turnover (MAAT) of Rs. 19.44 Crores",
    }
    body.update(overrides)
    return json.dumps([body])


def _parse(reply: str, start_index: int = 1) -> MapParse:
    return parse_map_items(
        reply,
        section_id=_SECTION,
        doc_id=_DOC,
        section_text=_TEXT,
        facet_ids=_FACETS,
        start_index=start_index,
    )


# --- the parse ------------------------------------------------------------------------------------


def test_a_usable_reply_becomes_an_item() -> None:
    parse = _parse(_reply())

    assert parse.usable and len(parse.items) == 1
    item = parse.items[0]
    assert item.item_id == "m1" and item.section_id == _SECTION and item.doc_id == _DOC
    assert item.facet_id == "financial_qualification"
    assert item.clause_ref == "Sl. 5"
    assert item.is_amendment and item.old_value == "Rs. 35 Crores" and item.new_value == "Rs. 19.44 Crores"
    assert item.values[0].name == "MAAT" and item.values[0].value == "19.44"


def test_an_empty_list_is_a_valid_reply_and_not_a_failure() -> None:
    """The expected answer for most sections of a long document. Treating it as a failure would
    make the coverage statement count a correct extraction as a loss."""
    parse = _parse("[]")
    assert parse.usable and parse.items == () and not parse.error


def test_a_reply_that_is_not_a_list_is_unusable() -> None:
    assert not _parse('{"facet_id": "x"}').usable
    assert not _parse("no json here").usable


def test_a_fenced_reply_still_parses() -> None:
    assert _parse(f"```json\n{_reply()}\n```").usable


# --- the quote rule -------------------------------------------------------------------------------


def test_an_item_whose_quote_is_not_in_the_section_is_dropped() -> None:
    """THE gate. The reduce step claims to use only these items, so an item the section does not
    say is a sentence the answer will assert and no page of the document supports."""
    parse = _parse(_reply(quote="of Rs. 99 Crores for the best three years"))

    assert parse.items == ()
    assert parse.dropped_unquoted == 1 and parse.seen == 1
    assert parse.usable, "a dropped item is not a failed parse"


def test_a_quote_matches_across_whitespace_and_case() -> None:
    """``quote_key`` normalisation — the same rule Phase 3 validates a relation with and Phase 4
    an enrichment with. A model asked for a verbatim span routinely returns one re-wrapped."""
    parse = _parse(_reply(quote="minimum average annual turnover (maat)\n   of rs. 19.44 crores"))
    assert len(parse.items) == 1


def test_an_item_with_no_quote_is_dropped() -> None:
    assert _parse(_reply(quote="")).dropped_unquoted == 1


def test_an_item_with_no_statement_is_skipped_without_counting_as_unquoted() -> None:
    """Nothing to say is not the same failure as saying it unsupported, and conflating them
    would make the unquoted rate — the signal that the model is fabricating spans — unreadable."""
    parse = _parse(_reply(statement="   "))
    assert parse.items == () and parse.dropped_unquoted == 0


# --- facets ---------------------------------------------------------------------------------------


def test_an_unknown_facet_is_cleared_and_the_item_survives() -> None:
    """Deliberately asymmetric with Phase 3 and Phase 4, which drop the whole entry. There a bad
    facet would be WRITTEN to the catalog and joined against later; here it only decides which
    heading a bullet sits under, and the statement and quote are still evidence."""
    parse = _parse(_reply(facet_id="invented_label"))

    assert len(parse.items) == 1 and parse.items[0].facet_id == ""
    assert parse.dropped_unknown_facet == 1 and parse.unknown_facets == ("invented_label",)


def test_a_missing_facet_is_not_an_unknown_facet() -> None:
    parse = _parse(_reply(facet_id=None))
    assert len(parse.items) == 1 and parse.items[0].facet_id == ""
    assert parse.dropped_unknown_facet == 0


# --- ids and the cache key --------------------------------------------------------------------------


def test_item_ids_are_short_dense_and_continue_from_the_offset() -> None:
    """The model has to copy them into the answer verbatim, and a 64-character hash is a thing
    models truncate."""
    two = json.dumps(
        [
            {"statement": "a", "quote": "Rs. 35 Crores"},
            {"statement": "b", "quote": "Rs. 19.44 Crores"},
        ]
    )
    assert [i.item_id for i in _parse(two).items] == ["m1", "m2"]
    assert [i.item_id for i in _parse(two, start_index=7).items] == ["m7", "m8"]


def test_the_cache_key_changes_with_every_input_that_changes_the_extraction() -> None:
    base = map_cache_key("s1", "sig", "model-a")
    assert base != map_cache_key("s2", "sig", "model-a"), "a different section"
    assert base != map_cache_key("s1", "other", "model-a"), "a different question"
    assert base != map_cache_key("s1", "sig", "model-b"), "a different model"
    assert base == map_cache_key("s1", "sig", "model-a"), "and is stable"


def test_the_query_signature_covers_the_facets_not_just_the_query() -> None:
    """The facets change what the map call asks for, so a cache that ignored them would serve
    items grouped under labels that no longer exist."""
    assert query_signature("q", ("a",)) != query_signature("q", ("a", "b"))
    assert query_signature("q", ("b", "a")) == query_signature("q", ("a", "b")), "order is not meaning"
    assert query_signature(" Q ", ("a",)) == query_signature("q", ("a",)), "nor is case or padding"


def test_the_map_version_is_in_the_key() -> None:
    """A build whose extraction contract changed must not read the previous build's items back
    and present them as its own."""
    import rag_core.mapreduce as mr

    before = map_cache_key("s1", "sig", "m")
    original = mr.MAP_VERSION
    try:
        mr.MAP_VERSION = original + 1
        assert map_cache_key("s1", "sig", "m") != before
    finally:
        mr.MAP_VERSION = original
    assert original == MAP_VERSION


# --- the round trip through the cache ----------------------------------------------------------------


def test_an_item_survives_the_cache_round_trip() -> None:
    item = _parse(_reply()).items[0]
    restored = MappedItem.from_json(item.as_json())

    assert restored.statement == item.statement and restored.quote == item.quote
    assert restored.values[0].name == "MAAT" and restored.values[0].unit == "Rs. Crores"
    assert restored.is_amendment and restored.old_value == "Rs. 35 Crores"
    assert restored.clause_ref == "Sl. 5"


# --- coverage ------------------------------------------------------------------------------------


def test_the_coverage_line_says_when_a_section_was_only_partly_read() -> None:
    """A section longer than the map's window bound still CONTRIBUTED, which is true and on its
    own overstates the answer. Three of the real corrigendum's 38 selected sections were in
    that state and what went unread were real requirements."""
    assert "too long to read in full" not in Coverage(12, 38, 72).line()
    partial = Coverage(used=12, selected=38, total=72, partial=3).line()
    assert "3 of the selected sections were too long to read in full" in partial


def test_the_coverage_line_says_the_answer_is_incomplete_when_a_section_could_not_be_read() -> None:
    """A failed section and a section with nothing relevant in it subtract from `used`
    identically, so the arithmetic alone cannot tell a reader which happened. During the
    scheduler benchmark a degraded endpoint lost 11 sections and the answer read exactly like an
    answer over a thinner document."""
    assert "INCOMPLETE" not in Coverage(used=12, selected=38, total=72).line()
    failed = Coverage(used=12, selected=38, total=72, failed=6).line()
    assert "INCOMPLETE" in failed
    assert "6 of the selected sections could not be read at all" in failed


def test_an_incomplete_answer_still_reports_what_it_did_read() -> None:
    """The incompleteness notice is an ADDITION, not a replacement: the counts a reader uses to
    judge the answer have to survive it, and so does the partial-read qualifier."""
    line = Coverage(used=12, selected=38, total=72, partial=3, failed=6).line()
    assert "12 of 38 selected sections contributed" in line
    assert "too long to read in full" in line
    assert "INCOMPLETE" in line


def test_the_coverage_line_states_all_three_numbers() -> None:
    """An aggregate answer over part of a document must not read like one over all of it."""
    line = Coverage(used=12, selected=38, total=72).line()
    assert "12 of 38 selected sections contributed" in line
    assert "72 sections in scope" in line


# --- the prompts ------------------------------------------------------------------------------------


def test_the_map_prompt_carries_the_section_the_facets_and_the_question() -> None:
    prompt = build_map_prompt(
        profile="a corrigendum",
        breadcrumb="4.2 MAAT",
        section_text=_TEXT,
        query="what changed?",
        facet_ids=_FACETS,
    )
    assert "4.2 MAAT" in prompt and "19.44" in prompt and "what changed?" in prompt
    assert "financial_qualification" in prompt


def test_the_reduce_prompt_carries_one_facets_items() -> None:
    items = list(_parse(_reply()).items)
    data = ReduceInput(
        query="what changed?",
        profiles="- corrigendum-v.pdf",
        items=items,
        coverage=Coverage(1, 2, 3),
        answer_lang="en",
    )
    prompt = build_reduce_prompt(data, "financial_qualification", items, {_SECTION: "4.2 MAAT"})

    assert "[m1]" in prompt and "4.2 MAAT" in prompt
    assert "financial qualification" in prompt
    assert "Coverage:" not in prompt, "the assembler prints the coverage line, not the model"


def test_items_group_by_facet_in_first_appearance_order() -> None:
    raw = json.dumps(
        [
            {"facet_id": "eligibility", "statement": "a", "quote": "Rs. 35 Crores"},
            {"facet_id": "financial_qualification", "statement": "b", "quote": "Rs. 19.44 Crores"},
            {"facet_id": "eligibility", "statement": "c", "quote": "best three years"},
            {"statement": "d", "quote": "income sheet"},
        ]
    )
    groups = group_by_facet(_parse(raw).items)

    assert [facet for facet, _ in groups] == ["eligibility", "financial_qualification", UNFACETED]
    assert [len(got) for _f, got in groups] == [2, 1, 1]


def test_the_assembled_answer_frames_the_sections_with_facts_the_model_never_writes() -> None:
    """Rule 1 (what the document is) and rule 7 (the coverage statement) are facts the caller
    already holds. When the single-call reduce ran out of budget, the coverage line was the
    first thing lost; printed by the assembler, it cannot be."""
    data = ReduceInput(
        query="q",
        profiles="- corrigendum-v.pdf: amends the tender",
        items=[],
        coverage=Coverage(12, 38, 72),
        answer_lang="en",
    )
    answer = assemble_answer(data, [("financial_qualification", "- MAAT is Rs. 19.44 Crores [m1]")])

    assert answer.startswith("- corrigendum-v.pdf: amends the tender")
    assert "### financial qualification" in answer
    assert answer.rstrip().endswith("12 of 38 selected sections contributed; 72 sections in scope.")


def test_a_facet_whose_call_returned_nothing_contributes_no_section() -> None:
    data = ReduceInput(query="q", profiles="- doc", items=[], coverage=Coverage(1, 1, 1))
    answer = assemble_answer(data, [("a", "- kept [m1]"), ("b", "   ")])

    assert "### a" in answer and "### b" not in answer


def test_an_answer_with_no_section_at_all_is_empty_not_a_frame_around_nothing() -> None:
    data = ReduceInput(query="q", profiles="- doc", items=[], coverage=Coverage(0, 1, 1))
    assert assemble_answer(data, [("a", ""), ("b", "  ")]) == ""


def test_an_items_line_shows_the_amendment_and_the_quote() -> None:
    block = items_block(list(_parse(_reply()).items))
    assert "AMENDMENT: Rs. 35 Crores -> Rs. 19.44 Crores" in block
    assert "clause=Sl. 5" in block
    assert "MAAT=19.44 Rs. Crores" in block


# --- citation resolution ------------------------------------------------------------------------------


def test_a_quote_resolves_to_the_chunk_that_contains_it() -> None:
    item = _parse(_reply()).items[0]
    chunks = [("c1", "The Bidder's income sheet should show"), ("c2", _TEXT)]
    assert resolve_item_chunk(item, chunks) == "c2"


def test_the_first_containing_chunk_wins() -> None:
    """A quote spanning a boundary is attributed to where it STARTS, which is where a reader
    clicking the citation should land."""
    item = _parse(_reply()).items[0]
    assert resolve_item_chunk(item, [("c1", _TEXT), ("c2", _TEXT)]) == "c1"


def test_a_quote_in_no_chunk_resolves_to_nothing() -> None:
    """Real, not a bug to handle elsewhere: the quote was validated against the SECTION, which is
    the concatenation of its chunks, so one straddling the join is in the section and in neither
    chunk. The caller drops the citation, not the item."""
    item = _parse(_reply()).items[0]
    assert resolve_item_chunk(item, [("c1", "something else entirely")]) is None


def test_resolution_is_whitespace_and_case_insensitive_like_the_quote_check() -> None:
    item = _parse(_reply()).items[0]
    assert resolve_item_chunk(item, [("c1", _TEXT.upper().replace(" ", "\n"))]) == "c1"


def test_items_past_the_per_section_cap_are_counted_not_just_dropped() -> None:
    """The bound fires in practice, and used to fire invisibly.

    Measured on a 3,000-character window of a real specification table: the model returned
    40, 40, 40 items across three runs. A figure that round and that repeatable is a ceiling
    being hit, not a model deciding it is done — and every item past it went nowhere with no
    count, no log and no mark on the answer. Dropping them is fine; dropping them silently is
    what this asserts against."""
    from rag_core.mapreduce import _MAX_ITEMS_PER_WINDOW, parse_map_items

    text = " ".join(f"Requirement {i} shall apply." for i in range(60))
    over = 7
    reply = json.dumps(
        [
            {"statement": f"Requirement {i}", "quote": f"Requirement {i} shall apply.", "facet_id": "f"}
            for i in range(_MAX_ITEMS_PER_WINDOW + over)
        ]
    )
    parsed = parse_map_items(reply, section_id="s", doc_id="d", section_text=text, facet_ids=["f"])

    assert len(parsed.items) == _MAX_ITEMS_PER_WINDOW
    assert parsed.dropped_over_cap == over
    assert parsed.seen == _MAX_ITEMS_PER_WINDOW + over


def test_a_reply_inside_the_cap_reports_nothing_dropped() -> None:
    from rag_core.mapreduce import parse_map_items

    text = "The bidder shall supply an SBOM."
    reply = json.dumps([{"statement": "SBOM required", "quote": "shall supply an SBOM", "facet_id": "f"}])
    parsed = parse_map_items(reply, section_id="s", doc_id="d", section_text=text, facet_ids=["f"])
    assert parsed.dropped_over_cap == 0
