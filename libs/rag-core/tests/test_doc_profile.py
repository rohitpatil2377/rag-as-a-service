"""The document profile card: prompt, validation and the contextualize prefix (RAG v2 Phase 3).

WHAT THESE TESTS DO NOT SHOW, STATED UP FRONT. The plan's Phase-3 acceptance is written entirely
against the AEGCL SOC/NOC Corrigendum-V — "doc_type≈corrigendum, a relation 'amends' pointing to
the original bidding document / Corrigendum-II, and facets covering eligibility, financial
qualification, …" — and that PDF is not in this repository (discovery §7, §9.1). None of it is
asserted here and none of it can be.

The difference from Phase 2's and Phase 5's stand-in fixtures is worth being precise about,
because those were defensible and a stand-in here would not be. Phase 2's acceptance is a
STRUCTURAL INVARIANT ("no chunk crosses a section boundary", "the page header no longer appears
inside block text") and any document of the right shape exercises it honestly. Phase 3's
acceptance is an EXTRACTION RESULT — a judgement a model makes about one specific document. A
fixture written here would be a reply this file's author composed, so asserting that the parser
keeps it would prove only that the author can write a passing fixture. It would say nothing about
whether a model reads the real corrigendum as a corrigendum.

So what is tested here is everything that is TRUE REGARDLESS OF THE MODEL: that a reply missing
the card's two required fields yields no card at all; that a relation whose quote is not in the
text the model was shown is dropped and REPORTED; that the quote check is the same normalisation
the graph seeding uses, so a relation cannot pass here and fail to anchor there; that facet ids
are normalised and collisions resolved; that a date is only accepted in one unambiguous form; and
that the card prefix is bounded. Those are properties of this module. The acceptance criteria are
properties of a document nobody here has.
"""

from __future__ import annotations

import json
from datetime import date

from rag_core.doc_profile import (
    FACET_BAND,
    PROFILE_VERSION,
    DocProfile,
    ProfileFacet,
    build_doc_profile_prompt,
    parse_doc_profile,
    profile_card,
)
from rag_core.graph import RELATION_TYPES, quote_key

# The text a model would have been shown. Short on purpose: the evidence window is what an
# evidence quote is checked against, and keeping it small makes "this quote is not in it" exact.
_SOURCE = (
    "Corrigendum-V to NIT AEGCL/CGM/SLDC/T-161. This corrigendum amends Corrigendum-II "
    "dated 31.08.2026.\nThe minimum average annual turnover is revised to Rs. 19.44 Crores."
)


def _reply(**overrides: object) -> str:
    body: dict[str, object] = {
        "doc_type": "corrigendum",
        "purpose": "Amends an earlier corrigendum to the bidding document.",
        "issuer": "AEGCL",
        "issue_date": "2026-09-24",
        "audience": "prospective bidders",
        "lang": "en",
        "relations": [
            {
                "type": "amends",
                "target_ref": "Corrigendum-II",
                "evidence_quote": "This corrigendum amends Corrigendum-II",
            }
        ],
        "facet_taxonomy": [{"facet_id": "financial_qualification", "name": "Financial", "description": "turnover"}],
    }
    body.update(overrides)
    return json.dumps(body)


def _parsed(**overrides: object) -> DocProfile:
    parse = parse_doc_profile(_reply(**overrides), source_text=_SOURCE, title="Corrigendum-V")
    assert parse.profile is not None, parse.error
    return parse.profile


# --- the card is all-or-nothing -----------------------------------------------------------------


def test_a_reply_without_a_doc_type_yields_no_profile_at_all() -> None:
    parse = parse_doc_profile(_reply(doc_type=""), source_text=_SOURCE, title="t")
    assert parse.profile is None
    assert "doc_type" in parse.error


def test_a_reply_without_a_purpose_yields_no_profile_at_all() -> None:
    parse = parse_doc_profile(_reply(purpose="   "), source_text=_SOURCE, title="t")
    assert parse.profile is None
    assert "purpose" in parse.error


def test_both_missing_are_named_together_so_one_retry_can_fix_both() -> None:
    parse = parse_doc_profile(_reply(doc_type="", purpose=""), source_text=_SOURCE, title="t")
    assert parse.profile is None
    assert "doc_type" in parse.error and "purpose" in parse.error


def test_prose_around_the_json_and_a_code_fence_are_tolerated() -> None:
    noisy = f"Sure, here is the profile:\n```json\n{_reply()}\n```\nLet me know if you need more."
    assert parse_doc_profile(noisy, source_text=_SOURCE, title="t").profile is not None


def test_a_reply_that_is_not_json_is_an_error_not_an_exception() -> None:
    parse = parse_doc_profile("I cannot analyse this document.", source_text=_SOURCE, title="t")
    assert parse.profile is None and parse.error


def test_a_json_array_is_rejected_rather_than_coerced() -> None:
    assert parse_doc_profile('[{"doc_type": "x"}]', source_text=_SOURCE, title="t").profile is None


# --- relations: the quote is the whole contract -------------------------------------------------


def test_a_relation_whose_quote_is_in_the_text_is_kept() -> None:
    profile = _parsed()
    assert [(r.rel_type, r.target_ref) for r in profile.relations] == [("amends", "Corrigendum-II")]


def test_a_relation_whose_quote_is_not_in_the_text_is_dropped_and_reported() -> None:
    invented = {"type": "supersedes", "target_ref": "Corrigendum-IV", "evidence_quote": "supersedes Corrigendum-IV"}
    parse = parse_doc_profile(_reply(relations=[invented]), source_text=_SOURCE, title="t")
    assert parse.profile is not None
    assert parse.profile.relations == ()
    # REPORTED, not silently discarded — the dropped amendment is the consequential failure here.
    assert parse.unquoted == ("supersedes Corrigendum-IV",)


def test_a_relation_type_outside_the_four_is_dropped_and_reported_not_guessed() -> None:
    odd = {"type": "replaces", "target_ref": "X", "evidence_quote": "This corrigendum amends Corrigendum-II"}
    parse = parse_doc_profile(_reply(relations=[odd]), source_text=_SOURCE, title="t")
    assert parse.profile is not None and parse.profile.relations == ()
    assert parse.unknown_types == ("replaces",)
    assert "replaces" not in RELATION_TYPES


def test_a_quote_differing_only_in_whitespace_and_case_still_validates() -> None:
    # The plan's own Phase-4 rule is "a substring after whitespace normalisation"; case-folding is
    # on top of it because a model routinely re-capitalises what it was asked to copy verbatim.
    loose = {
        "type": "amends",
        "target_ref": "Corrigendum-II",
        "evidence_quote": "this   corrigendum\n amends CORRIGENDUM-II",
    }
    parse = parse_doc_profile(_reply(relations=[loose]), source_text=_SOURCE, title="t")
    assert parse.profile is not None and len(parse.profile.relations) == 1


def test_the_quote_check_uses_the_same_normalisation_the_graph_seeding_resolves_with() -> None:
    """A relation that validates here MUST be anchorable there: two normalisations would leave an
    amendment edge in `doc_relations` and nowhere in the graph, with nothing reporting the gap."""
    profile = _parsed()
    quote = profile.relations[0].evidence_quote
    assert quote_key(quote) in quote_key(_SOURCE)


def test_a_relation_missing_its_quote_or_its_target_is_dropped() -> None:
    parse = parse_doc_profile(
        _reply(
            relations=[
                {"type": "amends", "target_ref": "X", "evidence_quote": ""},
                {"type": "amends", "target_ref": "", "evidence_quote": "This corrigendum amends Corrigendum-II"},
            ]
        ),
        source_text=_SOURCE,
        title="t",
    )
    assert parse.profile is not None and parse.profile.relations == ()


# --- facets: the id is a primary key ------------------------------------------------------------


def test_a_facet_id_is_normalised_to_snake_case() -> None:
    parse = parse_doc_profile(
        _reply(facet_taxonomy=[{"facet_id": "Supply-Chain Security", "name": "Supply chain", "description": "d"}]),
        source_text=_SOURCE,
        title="t",
    )
    assert parse.profile is not None
    assert [f.facet_id for f in parse.profile.facets] == ["supply_chain_security"]


def test_two_facets_normalising_onto_one_id_keep_the_first_and_report_the_collision() -> None:
    parse = parse_doc_profile(
        _reply(
            facet_taxonomy=[
                {"facet_id": "eligibility", "name": "Eligibility", "description": "who may bid"},
                {"facet_id": "Eligibility", "name": "Eligibility (2)", "description": "something else"},
            ]
        ),
        source_text=_SOURCE,
        title="t",
    )
    assert parse.profile is not None
    assert [(f.facet_id, f.description) for f in parse.profile.facets] == [("eligibility", "who may bid")]
    assert parse.dropped_facets == ("eligibility",)


def test_a_facet_with_no_id_falls_back_to_its_name() -> None:
    parse = parse_doc_profile(
        _reply(facet_taxonomy=[{"name": "OEM Credentials", "description": "d"}]), source_text=_SOURCE, title="t"
    )
    assert parse.profile is not None
    assert [f.facet_id for f in parse.profile.facets] == ["oem_credentials"]


def test_a_devanagari_facet_name_survives_normalisation() -> None:
    """An ASCII-only normaliser would empty it and drop the whole taxonomy of a Hindi document;
    `\\w` would keep the letters and strip the combining marks that carry its vowels, producing a
    mangled id that looks usable. This corpus is Indian and the chunker already calibrates for the
    script, so neither is acceptable."""
    parse = parse_doc_profile(
        _reply(facet_taxonomy=[{"facet_id": "वित्तीय योग्यता", "name": "Financial", "description": "d"}]),
        source_text=_SOURCE,
        title="t",
    )
    assert parse.profile is not None
    assert [f.facet_id for f in parse.profile.facets] == ["वित्तीय_योग्यता"]
    assert parse.dropped_facets == ()


def test_a_facet_name_of_pure_punctuation_is_dropped_and_reported() -> None:
    parse = parse_doc_profile(
        _reply(facet_taxonomy=[{"facet_id": "---", "name": "", "description": "d"}]),
        source_text=_SOURCE,
        title="t",
    )
    assert parse.profile is not None and parse.profile.facets == ()
    assert parse.dropped_facets == ("(unnamed)",)


def test_a_standards_code_in_a_facet_name_keeps_its_digits() -> None:
    parse = parse_doc_profile(
        _reply(facet_taxonomy=[{"facet_id": "61000-4-2 compliance", "name": "ESD", "description": "d"}]),
        source_text=_SOURCE,
        title="t",
    )
    assert parse.profile is not None
    assert [f.facet_id for f in parse.profile.facets] == ["61000_4_2_compliance"]


def test_a_facet_count_outside_the_prompts_band_is_reported_but_the_card_is_kept() -> None:
    """The band is prompt guidance about usefulness, not a correctness invariant — rejecting the
    card over it would throw away the validated relations with it."""
    parse = parse_doc_profile(_reply(), source_text=_SOURCE, title="t")
    assert parse.profile is not None and len(parse.profile.facets) < FACET_BAND[0]
    assert parse.facet_count_off_band is True


def test_a_facet_count_inside_the_band_is_not_flagged() -> None:
    taxonomy = [{"facet_id": f"facet_{i}", "name": f"F{i}", "description": "d"} for i in range(FACET_BAND[0])]
    parse = parse_doc_profile(_reply(facet_taxonomy=taxonomy), source_text=_SOURCE, title="t")
    assert parse.profile is not None and parse.facet_count_off_band is False


def test_a_hundred_facets_cannot_become_a_hundred_rows() -> None:
    taxonomy = [{"facet_id": f"f{i}", "name": f"F{i}", "description": "d"} for i in range(100)]
    parse = parse_doc_profile(_reply(facet_taxonomy=taxonomy), source_text=_SOURCE, title="t")
    assert parse.profile is not None and len(parse.profile.facets) <= 30


# --- scalar fields ------------------------------------------------------------------------------


def test_the_doc_type_is_lowercased_because_it_is_a_filter_value() -> None:
    assert _parsed(doc_type="Corrigendum").doc_type == "corrigendum"


def test_an_iso_date_is_parsed_and_anything_else_is_left_null() -> None:
    assert _parsed().issue_date == date(2026, 9, 24)
    # "24-09-2026" is ambiguous between two real calendars and this document class is Indian.
    assert _parsed(issue_date="24-09-2026").issue_date is None
    assert _parsed(issue_date="September 24, 2026").issue_date is None
    assert _parsed(issue_date="2026-02-30").issue_date is None  # syntactically fine, impossible


def test_the_title_comes_from_the_caller_and_is_never_read_off_the_reply() -> None:
    parse = parse_doc_profile(_reply(title="A title the model invented"), source_text=_SOURCE, title="Corrigendum-V")
    assert parse.profile is not None and parse.profile.title == "Corrigendum-V"


def test_absent_optional_fields_are_none_rather_than_empty_strings() -> None:
    profile = _parsed(issuer="", audience=None)
    assert profile.issuer is None and profile.audience is None


def test_the_callers_language_detection_overrides_the_models() -> None:
    """Measured against this deployment's gemma4-31b on a real document, the reply's `lang` came
    back "english". `documents.lang` must hold one vocabulary — the ISO-639-1 codes
    `detect_language` produces everywhere else — or a filter joining them matches nothing."""
    parse = parse_doc_profile(_reply(lang="english"), source_text=_SOURCE, title="t", lang="en")
    assert parse.profile is not None and parse.profile.lang == "en"


def test_the_models_language_is_used_only_when_the_caller_detected_none() -> None:
    parse = parse_doc_profile(_reply(lang="English"), source_text=_SOURCE, title="t", lang=None)
    assert parse.profile is not None and parse.profile.lang == "english"


def test_the_stored_version_is_the_contract_version() -> None:
    assert _parsed().profile_version == PROFILE_VERSION


def test_profile_json_round_trips_the_whole_card() -> None:
    profile = _parsed()
    blob = json.loads(profile.as_json())
    assert blob["doc_type"] == "corrigendum"
    assert blob["issue_date"] == "2026-09-24"
    assert blob["relations"][0]["target_ref"] == "Corrigendum-II"
    assert blob["facet_taxonomy"][0]["facet_id"] == "financial_qualification"
    assert blob["profile_version"] == PROFILE_VERSION


# --- the contextualize card ---------------------------------------------------------------------


def test_the_card_is_the_plans_prefix() -> None:
    card = profile_card(_parsed())
    assert card == "[corrigendum: Corrigendum-V — Amends an earlier corrigendum to the bidding document.] "


def test_no_profile_means_no_card_so_phase_five_chunking_is_unchanged() -> None:
    assert profile_card(None) == ""


def test_a_runaway_purpose_cannot_eat_every_chunks_embedding_budget() -> None:
    card = profile_card(
        DocProfile(doc_type="report", purpose="x" * 5000, title="T", facets=(ProfileFacet("f", "F", "d"),))
    )
    assert len(card) <= 200
    assert card.startswith("[report: T — ") and card.endswith("] ")


def test_a_card_whose_head_alone_overruns_is_still_bounded() -> None:
    card = profile_card(DocProfile(doc_type="report", purpose="short", title="T" * 500))
    assert len(card) <= 200


# --- the prompt ---------------------------------------------------------------------------------


def test_the_prompt_carries_all_four_of_the_plans_inputs() -> None:
    prompt = build_doc_profile_prompt(
        title="Corrigendum-V", metadata="- mime: application/pdf", breadcrumbs="- A > B", head_text=_SOURCE
    )
    assert "Corrigendum-V" in prompt
    assert "- mime: application/pdf" in prompt
    assert "- A > B" in prompt
    assert _SOURCE in prompt


def test_the_prompt_keeps_the_schema_braces_the_plan_wrote() -> None:
    """`str.format` would raise on this template, which is why the loader uses `string.Template`."""
    prompt = build_doc_profile_prompt(title="t", metadata="m", breadcrumbs="b", head_text="h")
    assert '"facet_taxonomy"' in prompt and '"evidence_quote"' in prompt
    assert "$title" not in prompt and "$head_text" not in prompt


def test_a_document_with_no_headings_says_so_rather_than_showing_an_empty_section() -> None:
    prompt = build_doc_profile_prompt(title="t", metadata="m", breadcrumbs="", head_text="h")
    assert "no detected section headings" in prompt


def test_a_dollar_sign_in_the_document_text_is_not_read_as_a_placeholder() -> None:
    prompt = build_doc_profile_prompt(title="t", metadata="m", breadcrumbs="b", head_text="Cost: $head_text and $x")
    assert "Cost: $head_text and $x" in prompt
