"""The facet vocabulary: pinned to a revision, resolved exactly, never inherited across bytes.

These are the rules that keep Phase 8's selection from being re-rolled by a re-ingest. A facet id
is the facet union's join key AND (de-underscored) the cross-encoder probe text, so "the ids
changed" and "selection changed" are the same sentence.
"""

from __future__ import annotations

from rag_core.facet_vocab import (
    VOCAB_VERSION,
    CanonicalFacet,
    FacetVocabulary,
    document_revision,
    reconcile,
)

_HASH = "3dc59d958833024d23437d2b8ec51114c9dbda35434986950edf64f2c5d89e5a"
_REV = f"{_HASH}/p1"


def _vocab(*ids: str, aliases: dict[str, tuple[str, ...]] | None = None, revision: str = _REV) -> FacetVocabulary:
    alias = aliases or {}
    return FacetVocabulary(
        revision=revision,
        facets=tuple(CanonicalFacet(facet_id=i, name=i.replace("_", " "), aliases=alias.get(i, ())) for i in ids),
    )


# --- the revision ---------------------------------------------------------------------------------


def test_the_revision_is_the_bytes_and_the_schema() -> None:
    assert document_revision(_HASH, 1) == _REV
    assert document_revision(_HASH, 2) != _REV, "a changed profile schema is a changed revision"


def test_a_document_with_no_content_hash_has_no_revision() -> None:
    """And therefore no vocabulary can be pinned to it. A vocabulary that cannot say which bytes
    it describes is not evidence about the document in front of you."""
    assert document_revision("", 1) == ""
    assert _vocab("a", revision="").matches("") is False


def test_a_vocabulary_matches_only_its_own_revision() -> None:
    v = _vocab("eligibility")
    assert v.matches(_REV)
    assert not v.matches(f"{_HASH}/p2")
    assert not v.matches("other/p1")


# --- resolution: exact, or an approved alias, or nothing -------------------------------------------


def test_a_canonical_id_resolves_to_itself() -> None:
    assert (
        _vocab("hardware_and_infrastructure_specifications").resolve("hardware_and_infrastructure_specifications")
        == "hardware_and_infrastructure_specifications"
    )


def test_spelling_of_the_same_token_sequence_resolves() -> None:
    """Case and the -/space/underscore spelling of the SAME tokens are one id written differently,
    which is a transcription difference and not a vocabulary difference."""
    v = _vocab("financial_eligibility")
    for spelling in (
        "Financial_Eligibility",
        "financial eligibility",
        "FINANCIAL-ELIGIBILITY",
        " financial_eligibility ",
    ):
        assert v.resolve(spelling) == "financial_eligibility", spelling


def test_a_near_synonym_does_not_resolve_until_it_is_approved() -> None:
    """THE CENTRAL RULE. These two mean the same thing to a reader and nothing to each other as
    join keys, and guessing the join from similarity would trade a nondeterministic model label
    for a nondeterministic threshold."""
    v = _vocab("performance_and_scalability")
    assert v.resolve("performance_and_scalability_metrics") is None

    approved = _vocab(
        "performance_and_scalability", aliases={"performance_and_scalability": ("performance_and_scalability_metrics",)}
    )
    assert approved.resolve("performance_and_scalability_metrics") == "performance_and_scalability"


def test_an_unknown_id_resolves_to_nothing_not_to_the_closest_thing() -> None:
    v = _vocab("security_detection_and_analytics")
    assert v.resolve("threat_detection_and_analytics") is None
    assert v.resolve("") is None


# --- reuse, adoption, and the report ---------------------------------------------------------------


def test_the_same_revision_reuses_the_stored_vocabulary() -> None:
    """The whole point: identical bytes ingested twice get the SAME ids, so the facet union joins
    the same sections and the score stage is probed with the same strings."""
    stored = _vocab("financial_eligibility", "hardware_and_infrastructure_specifications")
    proposed = [CanonicalFacet("financial_capacity"), CanonicalFacet("hardware_and_infrastructure_specifications")]

    decision = reconcile(stored, proposed, revision=_REV)

    assert decision.reused is True
    assert decision.vocabulary.ids == ("financial_eligibility", "hardware_and_infrastructure_specifications")
    assert decision.mapped == {
        "hardware_and_infrastructure_specifications": "hardware_and_infrastructure_specifications"
    }
    assert decision.unmapped == ("financial_capacity",), "a fresh invention is a candidate, not a facet"
    assert decision.unproposed == ("financial_eligibility",)


def test_changed_bytes_adopt_the_new_taxonomy_rather_than_inheriting() -> None:
    """An amended document gets the taxonomy of the amended text. Carrying facets forward would
    describe text that may no longer be there -- and the decision records that a vocabulary for
    other bytes existed, so the change is visible."""
    stored = _vocab("financial_eligibility", revision="OLDHASH/p1")
    proposed = [CanonicalFacet("financial_eligibility"), CanonicalFacet("new_requirement_class")]

    decision = reconcile(stored, proposed, revision=_REV)

    assert decision.reused is False
    assert decision.stored_revision == "OLDHASH/p1"
    assert decision.vocabulary.revision == _REV
    assert decision.vocabulary.ids == ("financial_eligibility", "new_requirement_class")


def test_a_first_ingest_adopts_what_phase_3_proposed() -> None:
    decision = reconcile(None, [CanonicalFacet("eligibility")], revision=_REV)

    assert decision.reused is False and decision.stored_revision == ""
    assert decision.vocabulary.ids == ("eligibility",)
    assert decision.vocabulary.matches(_REV)


def test_a_failed_profile_call_leaves_an_empty_vocabulary_and_not_a_stale_one() -> None:
    """A document whose profile call failed has no facet vocabulary, which is what the pipeline
    already assumes. Reusing a vocabulary here would be the one case where carrying forward is
    worse than having none: Phase 4 would label sections against a taxonomy this ingest never
    established."""
    decision = reconcile(None, [], revision=_REV)
    assert decision.vocabulary.facets == () and decision.reused is False


def test_an_alias_hit_is_reported_as_mapped_so_the_rename_is_visible() -> None:
    stored = _vocab(
        "data_retention_and_storage", aliases={"data_retention_and_storage": ("data_retention_and_storage_management",)}
    )

    decision = reconcile(stored, [CanonicalFacet("data_retention_and_storage_management")], revision=_REV)

    assert decision.reused is True
    assert decision.mapped == {"data_retention_and_storage_management": "data_retention_and_storage"}
    assert decision.unmapped == () and decision.unproposed == ()


# --- persistence ---------------------------------------------------------------------------------


def test_a_vocabulary_survives_a_round_trip_with_its_aliases() -> None:
    v = _vocab("eligibility", aliases={"eligibility": ("bidder_eligibility", "eligibility_requirements")})
    back = FacetVocabulary.from_dict(v.as_dict())

    assert back is not None
    assert back.revision == _REV and back.ids == ("eligibility",)
    assert back.facets[0].aliases == ("bidder_eligibility", "eligibility_requirements")
    assert back.resolve("bidder_eligibility") == "eligibility"


def test_a_vocabulary_from_a_version_this_code_does_not_know_is_refused() -> None:
    data = _vocab("eligibility").as_dict()
    data["vocab_version"] = VOCAB_VERSION + 1
    assert FacetVocabulary.from_dict(data) is None


def test_an_absent_or_empty_stored_vocabulary_reads_as_none_not_as_no_facets() -> None:
    """None means "nothing to reuse". An empty-but-present vocabulary would mean "this document
    has no facets", which would silence Phase 8's facet union for it."""
    assert FacetVocabulary.from_dict(None) is None
    assert FacetVocabulary.from_dict({}) is None
    assert FacetVocabulary.from_dict({"vocab_version": VOCAB_VERSION, "revision": _REV, "facets": []}) is None
    assert FacetVocabulary.from_dict([1, 2]) is None  # type: ignore[arg-type]


def test_a_facet_with_no_id_is_dropped_rather_than_stored_as_an_empty_join_key() -> None:
    data = {
        "vocab_version": VOCAB_VERSION,
        "revision": _REV,
        "facets": [{"facet_id": "", "name": "nameless"}, {"facet_id": "eligibility"}],
    }
    back = FacetVocabulary.from_dict(data)
    assert back is not None and back.ids == ("eligibility",)
