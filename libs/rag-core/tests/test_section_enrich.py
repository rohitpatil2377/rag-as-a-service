"""Section enrichment: the prompt, the parser and the merge (RAG v2 Phase 4).

WHAT THESE TESTS CAN AND CANNOT SHOW. The plan's Phase-4 acceptance is written against the AEGCL
corrigendum — "the MAAT section yields amends {old: 'Rs. 35 Crores', new: 'Rs. 19.44 Crores'}",
"the router section's propositions include both HBOM and SBOM" — and that document is not in this
repository (discovery §7). Those are claims about what a MODEL EXTRACTS from a real document, so
a fixture reply asserting them would only show that a reply I wrote parses into the fields I wrote
it to parse into. It would be self-fulfilling, and no amount of it is evidence for the plan's
acceptance.

What IS testable without the document, and is what follows, is the CONTRACT this module owes
whatever the model says: a quote that is not in the text loses its item; a facet outside the
document's taxonomy is dropped rather than created; a reply with no summary yields no enrichment
at all; the windows of a long section merge without losing or duplicating anything; and the
dropped items are reported rather than silently discarded. Those hold for any document.
"""

from __future__ import annotations

import json

from rag_core.layout import parse_layout
from rag_core.section_enrich import (
    ENRICH_VERSION,
    MANIFEST_VERSION,
    EnrichmentManifest,
    EnrichParse,
    EnrichReport,
    KeyValue,
    Proposition,
    SectionAmends,
    SectionEnrichment,
    assess_processing,
    build_section_enrich_prompt,
    merge_enrichments,
    merge_windows,
    parse_section_enrichment,
    section_prompt_text,
    section_windows,
)

_SECTION_ID = "sec-1"
_TEXT = (
    "4.2 Financial Qualification. The bidder shall have an average annual turnover of "
    "Rs. 19.44 Crores. This supersedes the figure of Rs. 35 Crores stated in Corrigendum-II. "
    "Certificates shall be furnished with the bid."
)
_FACETS = ("financial_qualification", "eligibility")


def _reply(**overrides: object) -> str:
    body: dict[str, object] = {
        "summary": "Sets the bidder's minimum average annual turnover and supersedes the earlier figure.",
        "facet_ids": ["financial_qualification"],
        "key_values": [
            {
                "name": "average annual turnover",
                "value": "19.44",
                "unit": "Rs. Crores",
                "quote": "average annual turnover of Rs. 19.44 Crores",
            }
        ],
        "amends": {
            "target_ref": "Corrigendum-II",
            "old": "Rs. 35 Crores",
            "new": "Rs. 19.44 Crores",
            "quote": "supersedes the figure of Rs. 35 Crores stated in Corrigendum-II",
        },
        "propositions": [
            {
                "text": "The bidder must have an average annual turnover of Rs. 19.44 Crores.",
                "quote": "The bidder shall have an average annual turnover",
            }
        ],
    }
    body.update(overrides)
    return json.dumps(body)


def _parse(reply: str, *, source_text: str = _TEXT) -> EnrichParse:
    return parse_section_enrichment(reply, section_id=_SECTION_ID, source_text=source_text, facet_ids=_FACETS)


# --- the verdict ---------------------------------------------------------------------------------


def test_a_usable_reply_becomes_an_enrichment() -> None:
    parse = _parse(_reply())
    e = parse.enrichment
    assert e is not None
    assert e.section_id == _SECTION_ID and e.enrich_version == ENRICH_VERSION
    assert e.facet_ids == ("financial_qualification",)
    assert [kv.value for kv in e.key_values] == ["19.44"]
    assert e.amends is not None and e.amends.new == "Rs. 19.44 Crores"
    assert len(e.propositions) == 1
    assert parse.quotes_kept == parse.quotes_seen == 3


def test_a_reply_with_no_summary_yields_no_enrichment() -> None:
    """All-or-nothing on the summary, the same rule Phase 3 applies to doc_type/purpose: a row
    with a NULL summary and populated facets reads as a section that was enriched and has nothing
    to say about itself, which is not what happened."""
    parse = _parse(_reply(summary=""))
    assert parse.enrichment is None and parse.error == "missing summary"


def test_a_reply_that_is_not_json_is_reported_not_raised() -> None:
    for reply in ("I cannot help with that.", "{not json}", json.dumps([1, 2])):
        parse = _parse(reply)
        assert parse.enrichment is None and parse.error


def test_a_fenced_reply_is_read() -> None:
    """Shared with Phase 3 through `json_reply.extract_json_object`, so a model that fences its
    JSON is not a lost section."""
    assert _parse(f"```json\n{_reply()}\n```").enrichment is not None


# --- the plan's validation rule: every quote must be in the section text --------------------------


def test_a_key_value_whose_quote_is_not_in_the_text_is_dropped_and_reported() -> None:
    parse = _parse(
        _reply(
            key_values=[
                {"name": "turnover", "value": "19.44", "unit": None, "quote": "average annual turnover of Rs. 19.44"},
                {"name": "emd", "value": "5", "unit": "lakh", "quote": "an EMD of Rs. 5 lakh is payable"},
            ]
        )
    )
    e = parse.enrichment
    assert e is not None and [kv.name for kv in e.key_values] == ["turnover"]
    assert parse.unquoted_key_values and "EMD" in parse.unquoted_key_values[0]


def test_a_proposition_whose_quote_is_not_in_the_text_is_dropped_and_reported() -> None:
    parse = _parse(
        _reply(propositions=[{"text": "The bidder must submit a bank guarantee.", "quote": "a bank guarantee"}])
    )
    e = parse.enrichment
    assert e is not None and e.propositions == ()
    assert parse.unquoted_propositions == ("a bank guarantee",)


def test_an_amends_whose_quote_is_not_in_the_text_is_dropped_and_flagged() -> None:
    """The single most consequential loss this phase has: an amendment dropped silently is
    indistinguishable from a section that amends nothing, and the difference is whether a later
    phase prefers one clause over another."""
    parse = _parse(
        _reply(amends={"target_ref": "Corrigendum-II", "old": None, "new": "Rs. 1", "quote": "hereby cancelled"})
    )
    e = parse.enrichment
    assert e is not None and e.amends is None
    assert parse.amends_unquoted is True


def test_quote_matching_normalises_whitespace_and_case() -> None:
    """The plan's rule is "a substring after whitespace normalisation", and `graph.quote_key` is
    the one definition of it in this codebase — shared so a quote cannot pass validation here and
    fail to anchor when Phase 6 resolves it to a chunk."""
    parse = _parse(
        _reply(
            propositions=[
                {"text": "Certificates are required.", "quote": "certificates   shall\n be   FURNISHED with the bid"}
            ]
        )
    )
    e = parse.enrichment
    assert e is not None and len(e.propositions) == 1


def test_an_amends_with_no_target_or_no_new_value_is_neither_kept_nor_reported() -> None:
    """It says nothing about what changed, so there is no amendment to lose."""
    for amends in ({"target_ref": "", "new": "x", "quote": "Certificates"}, {"target_ref": "C-II", "new": ""}):
        parse = _parse(_reply(amends=amends))
        e = parse.enrichment
        assert e is not None and e.amends is None
        assert parse.amends_unquoted is False


# --- the facet taxonomy is the document's, not the model's ----------------------------------------


def test_a_facet_outside_the_document_taxonomy_is_dropped_and_reported() -> None:
    """`facets` is keyed (doc_id, facet_id) and Phase 3 defines the vocabulary. A section
    inventing one would put a label in `sections.facet_ids` that joins to no facet row."""
    parse = _parse(_reply(facet_ids=["financial_qualification", "penalty_clauses"]))
    e = parse.enrichment
    assert e is not None and e.facet_ids == ("financial_qualification",)
    assert parse.unknown_facets == ("penalty_clauses",)


def test_a_facet_label_is_normalised_before_it_is_matched() -> None:
    """ "Financial Qualification" and "financial-qualification" are the same facet; `snake_id` is
    shared with Phase 3 so the label a section carries and the id the taxonomy stored agree."""
    parse = _parse(_reply(facet_ids=["Financial Qualification", "financial-qualification"]))
    e = parse.enrichment
    assert e is not None and e.facet_ids == ("financial_qualification",), "...and deduplicated"


def test_with_no_taxonomy_every_label_is_dropped() -> None:
    """A document whose profile call failed still gets summaries and propositions; what it loses
    is the vocabulary, and inventing one here would be worse than having none."""
    parse = parse_section_enrichment(_reply(), section_id=_SECTION_ID, source_text=_TEXT, facet_ids=())
    e = parse.enrichment
    assert e is not None and e.facet_ids == ()
    assert e.summary and e.propositions, "the rest of the enrichment survives"


# --- what is stored ------------------------------------------------------------------------------


def test_amends_json_uses_the_keys_the_graph_seeding_reads() -> None:
    """`PostgresCatalog.document_relations` reads `amends->>'target_ref'` and `amends->>'quote'`
    out of this column — a query written in Phase 6 against a column nothing wrote."""
    e = _parse(_reply()).enrichment
    assert e is not None
    stored = json.loads(e.amends_json() or "{}")
    assert stored["target_ref"] == "Corrigendum-II" and stored["quote"]
    assert set(stored) == {"target_ref", "old", "new", "quote"}


def test_key_values_are_stored_as_a_list_so_a_repeated_name_survives() -> None:
    """A section routinely states the same name twice — a value before and after revision — and
    a {name: value} object would silently keep one of them."""
    e = SectionEnrichment(
        section_id=_SECTION_ID,
        summary="s",
        key_values=(
            KeyValue("turnover", "35", None, "q"),
            KeyValue("turnover", "19.44", None, "q"),
        ),
    )
    assert [kv["value"] for kv in json.loads(e.key_values_json())] == ["35", "19.44"]


def test_a_section_with_no_amends_stores_null() -> None:
    assert SectionEnrichment(section_id=_SECTION_ID, summary="s").amends_json() is None


# --- the prompt ----------------------------------------------------------------------------------


def test_the_prompt_carries_the_taxonomy_as_an_explicit_list() -> None:
    """A deviation from the plan's prompt, and a deliberate one: "from the provided
    facet_taxonomy only" is a hard validation rule, and burying the permitted values inside a JSON
    blob that also holds relations, dates and a purpose makes the model hunt for the one list it
    is not allowed to depart from."""
    prompt = build_section_enrich_prompt(
        profile_json=json.dumps({"doc_type": "corrigendum"}),
        facet_ids=_FACETS,
        breadcrumb="4. Qualification > 4.2 Financial",
        section_text=_TEXT,
    )
    for facet in _FACETS:
        assert f"- {facet}" in prompt
    assert "4.2 Financial" in prompt and "19.44" in prompt and "corrigendum" in prompt


def test_the_prompt_says_so_when_there_is_no_profile_or_taxonomy() -> None:
    """Rather than rendering an empty section, which reads to a model as a missing input."""
    prompt = build_section_enrich_prompt(profile_json="", facet_ids=(), breadcrumb="", section_text=_TEXT)
    assert "no profile card" in prompt and "empty facet_ids" in prompt and "no heading path" in prompt


# --- the prompt's view of a section ---------------------------------------------------------------

_MARKUP = (
    "<h2>4.2 Financial</h2><p>Prose about turnover.</p>"
    "<table><tr><th>Item</th><th>Value</th></tr><tr><td>Turnover</td><td>19.44</td></tr></table>"
)


def _overlay(markup: str) -> dict[str, object]:
    return {
        "schema_version": 2,
        "extractor": "chandra",
        "pages": [{"page": 1, "markup_format": "html", "markup": markup}],
    }


def test_section_prompt_text_sends_a_table_once_not_twice() -> None:
    """Phase 2 emits BOTH a `table` block and one `table_row` block per row, so
    `LayoutSection.text` carries every table twice. Sending both doubles a wide table's share of
    the prompt and costs the surrounding prose the attention it needs."""
    tree = parse_layout(_overlay(_MARKUP), "d1")
    section = next(s for s in tree.sections if s.blocks)
    body = section_prompt_text(section)

    assert body.count("19.44") == 1
    assert "Prose about turnover." in body
    assert body != section.text, "...and `LayoutSection.text` is the shape being avoided"
    assert section.text.count("19.44") > 1


# --- windowing and merging -------------------------------------------------------------------------


def test_a_short_section_is_one_window() -> None:
    assert section_windows(_TEXT, 10_000, 4) == [_TEXT]
    assert section_windows("   ", 10_000, 4) == []


def test_windows_split_on_the_block_separator_where_there_is_one() -> None:
    """A window beginning mid-sentence costs the model the subject of the sentence, and a quote
    spanning the cut validates against neither window."""
    blocks = [f"Block {i} says something of moderate length." for i in range(6)]
    windows = section_windows("\n\n".join(blocks), 100, 10)

    assert len(windows) > 1
    assert all(not w.startswith(" ") for w in windows)
    for block in blocks:
        assert any(block in w for w in windows), "no block was cut in half"


def test_a_single_oversized_block_is_cut_and_nothing_is_lost() -> None:
    text = "x" * 250
    windows = section_windows(text, 100, 10)
    assert "".join(windows) == text


def test_windows_are_capped_and_the_tail_is_simply_not_read() -> None:
    """An appendix that is one 400-page "section" is a real extraction outcome; reading it in
    forty calls would spend a document's whole budget on one node. The caller logs the drop."""
    windows = section_windows("\n\n".join(f"Block {i}." for i in range(100)), 40, 3)
    assert len(windows) == 3


def test_merging_windows_unions_the_facets_and_keeps_every_distinct_item() -> None:
    kv, prop = KeyValue, Proposition
    a = SectionEnrichment(
        section_id=_SECTION_ID,
        summary="First half.",
        facet_ids=("eligibility",),
        key_values=(kv("turnover", "19.44", None, "q"),),
        propositions=(prop("A.", "q"),),
    )
    b = SectionEnrichment(
        section_id=_SECTION_ID,
        summary="Second half.",
        facet_ids=("eligibility", "financial_qualification"),
        key_values=(kv("turnover", "19.44", None, "q"), kv("emd", "5", "lakh", "q")),
        propositions=(prop("A.", "q"), prop("B.", "q")),
    )
    merged = merge_enrichments([a, b])

    assert merged is not None
    assert merged.summary == "First half. Second half."
    assert merged.facet_ids == ("eligibility", "financial_qualification")
    assert [k.name for k in merged.key_values] == ["turnover", "emd"], "the repeat collapsed"
    assert [p.text for p in merged.propositions] == ["A.", "B."]


def test_merging_keeps_the_first_amends_because_the_column_holds_one() -> None:
    amends = SectionAmends
    parts = [
        SectionEnrichment(section_id=_SECTION_ID, summary="a"),
        SectionEnrichment(section_id=_SECTION_ID, summary="b", amends=amends("C-II", None, "19.44", "q")),
        SectionEnrichment(section_id=_SECTION_ID, summary="c", amends=amends("C-III", None, "35", "q")),
    ]
    merged = merge_enrichments(parts)
    assert merged is not None and merged.amends is not None and merged.amends.target_ref == "C-II"


def test_merging_nothing_is_nothing() -> None:
    assert merge_enrichments([]) is None


# --- the atomicity rule ---------------------------------------------------------------------------
#
# `merge_enrichments` merges whatever it is handed, which is the right contract for a merge and
# the wrong one for deciding whether a section is ENRICHED. These tests fix that line.


def _part(summary: str, facet: str) -> SectionEnrichment:
    return SectionEnrichment(section_id=_SECTION_ID, summary=summary, facet_ids=(facet,))


def test_every_window_present_merges_exactly_as_before() -> None:
    merged, partial = merge_windows([_part("a", "eligibility"), _part("b", "financial_qualification")], windows=2)

    assert partial is False
    assert merged is not None and merged.summary == "a b"
    assert merged.facet_ids == ("eligibility", "financial_qualification")


def test_a_missing_window_refuses_the_whole_section() -> None:
    """The defect this exists for: two of three windows is a VALID enrichment, and storing it puts
    a third of the section's facets, key values and summary beyond Phase-8 selection with nothing
    recording the loss. A clean failure is disclosed by the manifest; this would not be."""
    merged, partial = merge_windows([_part("a", "eligibility"), _part("b", "eligibility")], windows=3)

    assert merged is None
    assert partial is True, "some of it parsed, which is the state worth a warning"


def test_no_window_parsing_is_an_ordinary_failure_not_a_partial() -> None:
    """`partial` must mean "the model answered for part of it". A section where nothing parsed is
    already classified by its reason code, and calling it partial would make a plain truncation
    look like a windowing problem."""
    assert merge_windows([], windows=2) == (None, False)


def test_a_section_with_no_windows_at_all_is_neither() -> None:
    """`section_windows` returns nothing for blank text. There is no result to publish and nothing
    was half-read."""
    assert merge_windows([], windows=0) == (None, False)
    assert merge_windows([_part("a", "eligibility")], windows=0) == (None, False)


def test_a_single_window_section_passes_straight_through() -> None:
    merged, partial = merge_windows([_part("only", "eligibility")], windows=1)
    assert partial is False and merged is not None and merged.summary == "only"


# --- the report ------------------------------------------------------------------------------------


def test_the_quote_rate_is_none_rather_than_one_when_there_were_no_quotes() -> None:
    """A rate of 1.0 over zero quotes reads as a perfect score and is not one; the plan's
    threshold is a claim about evidence that exists."""
    assert EnrichReport().quote_rate is None
    assert EnrichReport(quotes_seen=4, quotes_kept=3).quote_rate == 0.75


# --- why there is no enrichment, as a code rather than a sentence -----------------------------


def test_a_truncated_reply_is_not_the_same_failure_as_prose() -> None:
    """THE DISTINCTION THIS CODE EXISTS FOR, and it was paid for once already.

    On 2026-11-02 three substantive sections of `corrigendum-5.pdf` lost their Phase-4
    enrichment. The parser said `no JSON object in the reply`, and that string was committed as
    "the model returned prose instead of a JSON object" -- a conclusion drawn from the message
    without reading a reply. Every one of those replies was a valid JSON object cut off
    mid-structure at the output cap, 965-1,227 tokens against a 1,200 ceiling.

    The two have opposite fixes. Prose wants a prompt change or another attempt. Truncation wants
    output budget or a smaller input, and another attempt only re-rolls the reply's length -- a
    five-attempt run swapped WHICH sections failed rather than reducing how many."""
    truncated = parse_section_enrichment(
        '{"summary": "This section outlines the technical specifications for the',
        section_id="s1",
        source_text=_TEXT,
    )
    prose = parse_section_enrichment(
        "This section is about firewalls and their requirements.",
        section_id="s1",
        source_text=_TEXT,
    )

    assert truncated.reason_code == "TRUNCATED"
    assert prose.reason_code == "UNUSABLE_OUTPUT"
    assert truncated.enrichment is None and prose.enrichment is None
    assert "truncated" in truncated.error, "and the human string says so too"


def test_a_fenced_truncated_reply_is_still_truncated() -> None:
    """The first attempt's reply opens with a ```json fence in every observed case. The fence is
    stripped before the scan, so it must not turn a truncation into an absence."""
    parse = parse_section_enrichment(
        '```json\n{"summary": "This section details the technical requirement',
        section_id="s1",
        source_text=_TEXT,
    )
    assert parse.reason_code == "TRUNCATED"


def test_malformed_and_wrong_shape_are_distinguished_from_each_other() -> None:
    """INVALID_JSON is a syntax failure; INVALID_SCHEMA is a valid object of the wrong shape.
    Both are the model's fault and neither is a truncation, so neither is fixed by more room."""
    malformed = parse_section_enrichment('{"summary": "x",,}', section_id="s1", source_text=_TEXT)
    no_summary = parse_section_enrichment('{"facet_ids": ["eligibility"]}', section_id="s1", source_text=_TEXT)

    assert malformed.reason_code == "INVALID_JSON"
    assert no_summary.reason_code == "INVALID_SCHEMA"


def test_a_bare_array_is_read_as_its_first_object_which_the_array_scanner_promises_it_is_not() -> None:
    """A BOUNDARY THAT IS NOT WHAT THE NEIGHBOURING DOCSTRING CLAIMS, recorded rather than
    asserted loosely.

    `extract_json_array` says the two scanners "must not be interchangeable: a caller expecting
    an object and handed an array -- or the reverse -- should get None and retry". For this
    direction that is not what happens: `scan_json_object` finds the first `{`, so a reply that
    is a bare ARRAY of one well-formed object parses as that object and the enrichment succeeds.

    Pinned as the current behaviour, not endorsed. It is harmless where the first element is the
    object the schema wanted and wrong where the array was meant to carry several, and nothing
    downstream of here could tell the difference."""
    parse = parse_section_enrichment(
        '[{"summary": "What this clause requires.", "facet_ids": [], "key_values": [],'
        ' "amends": null, "propositions": []}]',
        section_id="s1",
        source_text=_TEXT,
    )
    assert parse.enrichment is not None, "the array wrapper did not stop it"
    assert parse.reason_code == ""


def test_a_usable_reply_carries_no_reason_code() -> None:
    parse = parse_section_enrichment(
        '{"summary": "What this clause requires.", "facet_ids": [], "key_values": [],'
        ' "amends": null, "propositions": []}',
        section_id="s1",
        source_text=_TEXT,
    )
    assert parse.enrichment is not None and parse.reason_code == "" and parse.error == ""


# --- the enrichment manifest ------------------------------------------------------------------


def test_a_missing_manifest_is_none_and_never_a_default_one() -> None:
    """THE SAFEGUARD THAT MATTERS MOST HERE. An empty manifest reads as a document with zero
    sections and nothing failed, which is indistinguishable from a clean ingest — so returning
    one for a document that has no stored manifest would hand every legacy document a
    completeness guarantee it never earned."""
    assert EnrichmentManifest.from_dict(None) is None
    assert EnrichmentManifest.from_dict({}) is None
    assert EnrichmentManifest.from_dict("not a mapping") is None  # type: ignore[arg-type]


def test_a_manifest_from_a_version_this_code_does_not_know_is_refused() -> None:
    """Refusing is the point: a future version may mean something different by the same field
    names, and misreading it is worse than reporting the document as unknown."""
    data = EnrichmentManifest(content_hash="abc").as_dict()
    assert EnrichmentManifest.from_dict(data) is not None
    assert EnrichmentManifest.from_dict({**data, "manifest_version": MANIFEST_VERSION + 1}) is None


def test_a_manifest_that_cannot_say_which_bytes_it_describes_matches_nothing() -> None:
    """A stale success must never outlive a re-ingest that failed, so the hash is what binds a
    manifest to a document — and an EMPTY hash is not a wildcard, it is an unanswerable question."""
    assert EnrichmentManifest(content_hash="3dc59d95").matches("3dc59d95")
    assert not EnrichmentManifest(content_hash="3dc59d95").matches("other")
    assert not EnrichmentManifest(content_hash="").matches("")
    assert not EnrichmentManifest(content_hash="").matches("3dc59d95")


def test_policy_skips_do_not_make_a_document_incomplete_but_a_cap_does() -> None:
    """Measured on `corrigendum-5.pdf`: 15 of 18 unenriched sections are 13-157 characters of
    furniture and table fragments the minimum-length rule correctly declined, and 3 were offered
    and lost. Folding those together would report a 34% processing failure for a document with a
    5.7% one.

    A CAP IS DIFFERENT and is not a policy skip: those sections passed the length rule and were
    never offered to a model at all, so nothing read them."""
    skipped = EnrichmentManifest(
        content_hash="h",
        sections_total=53,
        sections_offered=38,
        sections_enriched=38,
        sections_skipped_policy=15,
    )
    failed = EnrichmentManifest(
        content_hash="h",
        sections_total=53,
        sections_offered=38,
        sections_enriched=35,
        sections_skipped_policy=15,
        sections_failed=3,
        failure_classes={"TRUNCATED": 3},
    )
    capped = EnrichmentManifest(content_hash="h", sections_total=900, sections_offered=400, capped=True)

    assert skipped.complete, "declined by policy is not a loss"
    assert not failed.complete
    assert not capped.complete, "enrichable and never offered is a loss nothing reports otherwise"


def test_a_partial_window_refusal_is_persisted_apart_from_the_other_failures() -> None:
    """The manifest is read a day after the ingest log is gone, and "this section is readable and
    one window ran out of room" has a different fix from "this section could not be read". The
    count is <= `sections_failed` because a partial IS a failure, not an addition to them."""
    manifest = EnrichmentManifest(
        content_hash="h",
        sections_offered=24,
        sections_enriched=21,
        sections_failed=3,
        sections_partial=1,
        failure_classes={"TRUNCATED": 2, "GATEWAY_FAILURE": 1},
    )
    back = EnrichmentManifest.from_dict(manifest.as_dict())

    assert back is not None and back.sections_partial == 1
    assert back.sections_partial <= back.sections_failed


def test_a_manifest_written_before_partials_were_refused_reads_zero_not_unknown() -> None:
    """Such a run could not have refused a partial -- it MERGED it, and counted the section as
    enriched. 0 is the honest reading, and it is not a claim that no partial occurred."""
    data = EnrichmentManifest(content_hash="h", sections_offered=4, sections_enriched=4).as_dict()
    del data["sections_partial"]
    back = EnrichmentManifest.from_dict(data)

    assert back is not None and back.sections_partial == 0


def test_the_failure_classes_survive_a_round_trip_unflattened() -> None:
    """Five parser codes, not two. They do not share a fix, and collapsing them at the
    persistence layer would restore the ambiguity the codes were added to remove."""
    m = EnrichmentManifest(
        content_hash="h",
        sections_failed=4,
        failure_classes={"TRUNCATED": 2, "UNUSABLE_OUTPUT": 1, "GATEWAY_FAILURE": 1},
        failed_section_ids=("a", "b", "c", "d"),
    )
    back = EnrichmentManifest.from_dict(m.as_dict())
    assert back == m
    assert back is not None and back.failure_classes == {"TRUNCATED": 2, "UNUSABLE_OUTPUT": 1, "GATEWAY_FAILURE": 1}


# --- processing completeness, over the assessed document scope ---------------------------------


def _ok(doc_hash: str = "h", offered: int = 24) -> tuple[EnrichmentManifest, str]:
    return EnrichmentManifest(content_hash=doc_hash, sections_offered=offered, sections_enriched=offered), doc_hash


def _partial(doc_hash: str = "h", *, failed: int = 3) -> tuple[EnrichmentManifest, str]:
    return (
        EnrichmentManifest(
            content_hash=doc_hash,
            sections_offered=38,
            sections_enriched=38 - failed,
            sections_skipped_policy=15,
            sections_failed=failed,
            failure_classes={"TRUNCATED": failed},
            failed_section_ids=tuple(f"s{i}" for i in range(failed)),
        ),
        doc_hash,
    )


def test_two_scoped_documents_one_partial_reports_partial() -> None:
    got = assess_processing(scoped_doc_ids=["a", "b"], manifests={"a": _partial("ha"), "b": _ok("hb")})
    assert got.status == "PARTIAL" and got.documents_assessed == 2 and got.documents_partial == 1
    assert got.scope_basis == "resolved_doc_ids" and got.documents_unknown == 0


def test_a_partial_document_that_contributed_nothing_is_still_assessed() -> None:
    """THE ACCEPTANCE TEST THAT MATTERS MOST. A scoped document whose every section failed
    enrichment contributes no selected evidence and no citation — so a selection-based or
    citation-based scope would drop it from its own assessment, and the answer would report clean
    processing precisely because the failure was total. Its id enters through `doc_ids`."""
    got = assess_processing(
        scoped_doc_ids=["silent", "cited"],
        cited_doc_ids=["cited"],
        manifests={"silent": _partial("h1"), "cited": _ok("h2")},
    )
    assert got.status == "PARTIAL" and got.scope_basis == "mixed"


def test_every_selected_section_contributing_does_not_make_processing_complete() -> None:
    """The contribution rate and this dimension answer different questions. 32 of 32 selected
    sections contributing is true and says nothing about three sections never offered to the
    selector."""
    got = assess_processing(scoped_doc_ids=["a"], cited_doc_ids=["a"], manifests={"a": _partial("h")})
    assert got.status == "PARTIAL"


def test_an_unscoped_answer_citing_a_partial_document_reports_partial() -> None:
    got = assess_processing(cited_doc_ids=["a"], manifests={"a": _partial("h")})
    assert got.status == "PARTIAL" and got.scope_basis == "cited_documents"


def test_an_unscoped_answer_citing_only_complete_documents_is_complete_for_what_it_assessed() -> None:
    """COMPLETE here means "every assessed source document has a complete manifest". It does NOT
    mean nothing in the knowledge base was missed before retrieval, and `scope_basis` is what
    lets a reader tell the difference."""
    got = assess_processing(cited_doc_ids=["a", "b"], manifests={"a": _ok("ha"), "b": _ok("hb")})
    assert got.status == "COMPLETE" and got.scope_basis == "cited_documents"
    assert got.disclosure() == "", "nothing to disclose, and no claim of KB-wide completeness"


def test_a_document_with_no_manifest_is_unknown_not_complete() -> None:
    got = assess_processing(scoped_doc_ids=["a", "b"], manifests={"a": _ok("ha")})
    assert got.status == "UNKNOWN" and got.documents_unknown == 1


def test_a_known_failure_outranks_an_unknown_manifest_but_the_unknown_is_kept() -> None:
    got = assess_processing(scoped_doc_ids=["a", "b"], manifests={"a": _partial("ha")})
    assert got.status == "PARTIAL", "there is a definite problem to disclose"
    assert got.documents_partial == 1, "A's known failure is counted"
    assert got.documents_unknown == 1, "and B's silence does not erase it"
    assert got.documents_assessed == 2
    # THE TWO FAILURE BOUNDARIES ARE DIFFERENT. An assessment that cannot be trusted at all
    # returns None from `_attach_processing` and reads as UNKNOWN. An assessment that CAN be
    # trusted about one document and not another reports what it knows -- an unreadable manifest
    # on B must never erase a measured failure on A.
    assert got.sections_failed == 3 and got.failure_classes == {"TRUNCATED": 3}


def test_a_stale_manifest_is_unknown_because_it_describes_other_bytes() -> None:
    """A manifest whose hash does not match the document's is a previous ingest's result. A stale
    success must never outlive a re-ingest that failed."""
    manifest, _hash = _ok("old")
    got = assess_processing(scoped_doc_ids=["a"], manifests={"a": (manifest, "new")})
    assert got.status == "UNKNOWN" and got.documents_unknown == 1


def test_no_documents_at_all_is_unknown_rather_than_complete() -> None:
    got = assess_processing()
    assert got.status == "UNKNOWN" and got.scope_basis == "none" and got.documents_assessed == 0


def test_policy_skips_alone_leave_a_document_complete() -> None:
    manifest = EnrichmentManifest(
        content_hash="h",
        sections_total=53,
        sections_offered=38,
        sections_enriched=38,
        sections_skipped_policy=15,
    )
    got = assess_processing(scoped_doc_ids=["a"], manifests={"a": (manifest, "h")})
    assert got.status == "COMPLETE", "15 declined fragments are not a processing failure"


def test_the_disclosure_names_no_section_and_no_document() -> None:
    """The platform knows which section RECORDS failed. It has no extracted evidence about what
    those sections CONTAIN, so a heading in the answer body would be a claim about missing content
    drawn from the absence of that content."""
    got = assess_processing(scoped_doc_ids=["corrigendum-5.pdf"], manifests={"corrigendum-5.pdf": _partial("h")})
    line = got.disclosure()
    assert "3 of 38" in line and "may omit" in line
    assert "corrigendum" not in line and "s0" not in line
    assert got.failed_section_ids == ("s0", "s1", "s2"), "the ids exist, for diagnostics only"


def test_the_quote_rate_is_not_rounded_into_passing_the_floor() -> None:
    """A FLOOR IS DECIDED ON THE EXACT RATE, not on the number anyone reads.

    Measured on a real ingest (2026-11-03): 451 of 475 quotes validated, which is 0.949474 and
    BELOW the 0.95 acceptance floor — and one more kept quote, 452/475 = 0.951579, would pass. At
    three decimal places the first reads 0.949 and a run at 0.9496 would read 0.950, i.e. as
    passing a floor it fails. This pins the property the comparison depends on: the rate is the
    exact quotient, so any rounding is the caller's presentation choice and never the gate."""
    below = EnrichReport(quotes_seen=475, quotes_kept=451)
    assert below.quote_rate is not None
    assert below.quote_rate < 0.95, "0.949474 must not pass"
    assert round(below.quote_rate, 3) == 0.949

    # The hazard itself: a rate that FAILS and would read as passing at three places.
    borderline = EnrichReport(quotes_seen=10_000, quotes_kept=9_496)
    assert borderline.quote_rate is not None
    assert borderline.quote_rate < 0.95 and round(borderline.quote_rate, 3) == 0.95

    above = EnrichReport(quotes_seen=475, quotes_kept=452)
    assert above.quote_rate is not None and above.quote_rate >= 0.95
