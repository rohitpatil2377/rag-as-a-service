"""Section enrichment through the indexing service: staleness and the chunker hand-off (Phase 4).

Phase 4 adds two LEVELS to the index — ``section`` (the summary) and ``proposition`` — which
means a corpus indexed without enrichment holds strictly fewer units than this pipeline would now
produce. That is staleness, and the fingerprint is the only thing that says so; ``ENRICH_VERSION``
is what makes a changed prompt or parser reach the vectors rather than sitting in the catalog.

The CONDITIONAL-APPEND RULE is the part worth pinning. With ``v2_ingest`` off the fingerprint
must be byte-identical to what it was before this phase existed, or deploying Phase 4 would mark
every document in every deployment stale for a feature none of them has turned on.
"""

from __future__ import annotations

from typing import Any

import pytest
from svc_indexing.service import pipeline_fingerprint

from rag_common import get_settings
from rag_core.section_enrich import ENRICH_VERSION


def _settings(**overrides: Any) -> Any:
    return get_settings().model_copy(update=overrides)


def test_the_flag_off_fingerprint_does_not_see_the_enrichment_settings() -> None:
    """The whole point of the conditional append: a deployment with v2 ingest off must not have
    its corpus marked stale by a phase it is not running."""
    off = _settings(v2_ingest=False)
    assert pipeline_fingerprint(off) == pipeline_fingerprint(_settings(v2_ingest=False, section_enrich_enabled=False))


def test_turning_section_enrichment_off_marks_the_corpus_stale() -> None:
    """Two fewer levels in the index is a different index, and a reindex is what restores them."""
    on = _settings(v2_ingest=True, section_enrich_enabled=True)
    off = _settings(v2_ingest=True, section_enrich_enabled=False)
    assert pipeline_fingerprint(on) != pipeline_fingerprint(off)


def test_the_sizing_knobs_do_not_mark_a_corpus_stale() -> None:
    """They change WHICH sections get a summary, not the shape of the pipeline. A corpus is not
    stale because an operator widened a floor, and marking it so would invite a full reindex of
    every tenant for a tuning change."""
    base = _settings(v2_ingest=True)
    for override in (
        {"section_enrich_min_chars": 500},
        {"section_enrich_max_sections": 50},
        {"section_enrich_window_chars": 6000},
        {"section_enrich_max_windows": 2},
        {"section_enrich_concurrency": 1},
        {"section_enrich_max_tokens": 900},
    ):
        assert pipeline_fingerprint(base.model_copy(update=override)) == pipeline_fingerprint(base)


def test_the_enrichment_contract_version_is_in_the_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bumping ``ENRICH_VERSION`` after a prompt change must mark every enriched document for
    re-indexing — it is the only way the new summaries and propositions reach the vectors."""
    before = pipeline_fingerprint(_settings(v2_ingest=True))
    monkeypatch.setattr("svc_indexing.service.ENRICH_VERSION", ENRICH_VERSION + 1)
    assert pipeline_fingerprint(_settings(v2_ingest=True)) != before
