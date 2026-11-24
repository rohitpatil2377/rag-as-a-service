"""Per-doc_type chunk sizing and the fingerprint component that goes with it (RAG v2 Phase 3).

Plan §3 ends with "Routing may use ``doc_type`` to pick chunking parameters (config map:
``doc_type -> chunk profile``)". The mechanism is here and SHIPS EMPTY: nothing in this repository
has measured what a corrigendum or a contract should be chunked at — ``layout_chunk_max_chars``'s
own 1600 was calibrated against a token budget, not a document class — so the default map carries
no invented numbers and every document chunks exactly as it did in Phase 5.

What that leaves worth testing is the wiring either side of the empty default: that an operator's
entry is applied to the matching doc_type and to no other, that a malformed entry cannot take
ingestion down, and that changing either the map or the profile contract marks the corpus stale
(a card the index does not contain is not a card any answer can use).
"""

from __future__ import annotations

from typing import Any

import pytest
from svc_indexing.service import _layout_max_chars, pipeline_fingerprint

from rag_common import get_settings
from rag_core.doc_profile import DocProfile


def _settings(**overrides: Any) -> Any:
    return get_settings().model_copy(update=overrides)


def _profile(doc_type: str) -> DocProfile:
    return DocProfile(doc_type=doc_type, purpose="p", title="t")


# --- the map ships empty --------------------------------------------------------------------------


def test_the_map_is_empty_by_default_so_every_document_keeps_the_platform_size() -> None:
    settings = _settings()
    assert settings.doc_type_chunk_chars == {}
    assert _layout_max_chars(settings, _profile("corrigendum")) == settings.layout_chunk_max_chars


def test_a_document_with_no_profile_keeps_the_platform_size() -> None:
    settings = _settings(layout_chunk_max_chars_by_doc_type="corrigendum=1200")
    assert _layout_max_chars(settings, None) == settings.layout_chunk_max_chars


# --- an operator's entry ----------------------------------------------------------------------------


def test_a_configured_doc_type_gets_its_own_size() -> None:
    settings = _settings(layout_chunk_max_chars_by_doc_type="corrigendum=1200,contract=2400")
    assert _layout_max_chars(settings, _profile("corrigendum")) == 1200
    assert _layout_max_chars(settings, _profile("contract")) == 2400


def test_an_unlisted_doc_type_falls_back_rather_than_guessing() -> None:
    settings = _settings(layout_chunk_max_chars_by_doc_type="corrigendum=1200")
    assert _layout_max_chars(settings, _profile("policy")) == settings.layout_chunk_max_chars


def test_the_key_is_matched_case_insensitively_because_doc_type_is_lowercased() -> None:
    settings = _settings(layout_chunk_max_chars_by_doc_type="Corrigendum=1200")
    assert _layout_max_chars(settings, _profile("corrigendum")) == 1200


@pytest.mark.parametrize(
    "value",
    [
        "corrigendum",  # no "="
        "corrigendum=",  # no size
        "corrigendum=small",  # not a number
        "=1200",  # no doc_type
        "corrigendum=10",  # below the size field's own floor
        "corrigendum=100000",  # above its ceiling
        ",,,",
    ],
)
def test_a_malformed_entry_is_skipped_rather_than_taking_ingestion_down(value: str) -> None:
    """This is read on the ingest path for every document; a typo in an operator knob must not be
    able to stop a deployment ingesting."""
    settings = _settings(layout_chunk_max_chars_by_doc_type=value)
    assert settings.doc_type_chunk_chars == {}
    assert _layout_max_chars(settings, _profile("corrigendum")) == settings.layout_chunk_max_chars


def test_a_good_entry_beside_a_malformed_one_still_applies() -> None:
    settings = _settings(layout_chunk_max_chars_by_doc_type="corrigendum=oops,contract=2400")
    assert settings.doc_type_chunk_chars == {"contract": 2400}


def test_the_map_cannot_set_a_size_the_field_itself_would_reject() -> None:
    settings = _settings(layout_chunk_max_chars_by_doc_type="a=199,b=200,c=8192,d=8193")
    assert settings.doc_type_chunk_chars == {"b": 200, "c": 8192}


# --- staleness ----------------------------------------------------------------------------------------


def test_with_the_v2_flag_off_phase_three_does_not_touch_the_fingerprint() -> None:
    """Turning v2 on marks a corpus stale; turning it off must mark nothing — the same
    conditional-append rule the multivector and Phase-5 components follow."""
    base = _settings(v2_ingest=False)
    changed = base.model_copy(update={"layout_chunk_max_chars_by_doc_type": "corrigendum=1200"})
    assert pipeline_fingerprint(changed) == pipeline_fingerprint(base)


def test_with_the_v2_flag_on_changing_the_map_marks_the_corpus_stale() -> None:
    """The map changes where chunks are SPLIT, so a corpus indexed under a different value of it
    genuinely is stale."""
    base = _settings(v2_ingest=True)
    changed = base.model_copy(update={"layout_chunk_max_chars_by_doc_type": "corrigendum=1200"})
    assert pipeline_fingerprint(changed) != pipeline_fingerprint(base)


def test_turning_the_v2_flag_on_marks_the_corpus_stale() -> None:
    assert pipeline_fingerprint(_settings(v2_ingest=True)) != pipeline_fingerprint(_settings(v2_ingest=False))
