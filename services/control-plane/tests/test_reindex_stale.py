"""Stale-index detection (FR-C8): the fingerprint's discrimination rules, and what acts on it.

The value of the whole feature is in WHICH settings change the fingerprint. A setting that
should force a rebuild but doesn't leaves a corpus silently answering from the old pipeline; a
query-time setting that does leaves every KB permanently "stale" for a change no re-index can
fix, which trains everyone to ignore the banner.
"""

from __future__ import annotations

from typing import Any

import pytest
from svc_control_plane import reindex
from svc_indexing.service import META_PIPELINE_VERSION, pipeline_fingerprint

from rag_common import get_settings


def _settings() -> Any:
    return get_settings().model_copy()


def test_the_same_settings_always_fingerprint_the_same() -> None:
    assert pipeline_fingerprint(_settings()) == pipeline_fingerprint(_settings())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunk_size", 999),
        ("chunk_overlap", 7),
        ("chunk_max_chars", 12345),
        ("chunk_strategy", "fixed"),
        ("embed_model", "some-other-embedder"),
        ("embed_dim", 4096),
        ("contextual_retrieval_enabled", True),
        ("contextual_retrieval_backend", "llm"),
        ("graphrag_enabled", True),
        ("graph_extract_backend", "llm"),
        ("ocr_backend", "chandra"),
        ("image_extract_backend", "vision"),
        ("vision_llm_backend", "chandra"),
        ("multimodal_enabled", True),
    ],
)
def test_a_setting_that_changes_what_the_index_contains_changes_the_fingerprint(field: str, value: object) -> None:
    base = _settings()
    changed = base.model_copy(update={field: value})
    assert getattr(changed, field) != getattr(base, field), "test would pass vacuously"
    assert pipeline_fingerprint(changed) != pipeline_fingerprint(base)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rerank_top_k", 42),
        ("heavy_top_k", 33),
        ("abstain_threshold", 0.99),
        ("rerank_backend", "tei"),
        ("llm_model", "some-other-llm"),
        ("faithfulness_enabled", False),
        ("semantic_cache_enabled", True),
    ],
)
def test_a_query_time_setting_does_not_mark_the_corpus_stale(field: str, value: object) -> None:
    """Changing how an answer is RETRIEVED or GENERATED needs no re-index, so it must not be
    reported as one. Otherwise the stale count never reaches zero and stops meaning anything."""
    base = _settings()
    changed = base.model_copy(update={field: value})
    assert getattr(changed, field) != getattr(base, field), "test would pass vacuously"
    assert pipeline_fingerprint(changed) == pipeline_fingerprint(base)


class _Doc:
    def __init__(self, doc_id: str, version: str | None) -> None:
        self.doc_id = doc_id
        self.metadata: dict[str, str] = {} if version is None else {META_PIPELINE_VERSION: version}


class _Catalog:
    def __init__(self, docs: list[_Doc]) -> None:
        self._docs = docs

    async def list_documents(self, kb_id: str) -> list[_Doc]:
        return self._docs


@pytest.mark.asyncio
async def test_stale_report_counts_other_versions_and_unstamped_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    current = pipeline_fingerprint(get_settings())
    docs = [_Doc("current", current), _Doc("old", "deadbeefcafe"), _Doc("never-stamped", None)]
    monkeypatch.setattr(reindex, "get_catalog", lambda: _Catalog(docs))

    report = await reindex.stale_report("kb")
    assert report.total == 3
    assert report.stale == 2
    # A document ingested before the stamp existed reads as stale: nothing recorded how it was
    # built, so nothing can claim it matches what is running now.
    assert sorted(report.stale_doc_ids) == ["never-stamped", "old"]
    assert report.current_version == current


@pytest.mark.asyncio
async def test_reindex_kb_rebuilds_only_stale_documents_and_survives_one_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = pipeline_fingerprint(get_settings())
    docs = [_Doc("fine", current), _Doc("bad", "old"), _Doc("good", "old")]
    monkeypatch.setattr(reindex, "get_catalog", lambda: _Catalog(docs))

    seen: list[str] = []

    async def _reindex_document(kb_id: str, doc_id: str, *, actor: str) -> None:
        seen.append(doc_id)
        if doc_id == "bad":
            raise RuntimeError("embedder timed out")

    from svc_control_plane import pipeline

    monkeypatch.setattr(pipeline, "reindex_document", _reindex_document)

    result = await reindex.reindex_kb("kb", actor="tester", stale_only=True, limit=50)
    # The current document is never touched -- a re-index is an extraction + an embedding pass
    # against the shared endpoints, so rebuilding what is already current is pure contention.
    assert seen == ["bad", "good"]
    assert result.skipped_current == 1
    assert result.reindexed == 1
    # One document failing must not abandon the rest of the corpus, and must be reported rather
    # than swallowed -- it keeps serving its previous index (write-then-swap), which is only
    # safe if somebody is told it did not move.
    assert [f["doc_id"] for f in result.failed] == ["bad"]
    assert "embedder timed out" in result.failed[0]["error"]


@pytest.mark.asyncio
async def test_a_limit_bounds_one_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = [_Doc(f"d{i}", "old") for i in range(10)]
    monkeypatch.setattr(reindex, "get_catalog", lambda: _Catalog(docs))
    seen: list[str] = []

    async def _reindex_document(kb_id: str, doc_id: str, *, actor: str) -> None:
        seen.append(doc_id)

    from svc_control_plane import pipeline

    monkeypatch.setattr(pipeline, "reindex_document", _reindex_document)
    result = await reindex.reindex_kb("kb", actor="t", stale_only=True, limit=3)
    assert len(seen) == 3
    assert result.requested == 3


def test_auto_reindex_is_off_until_an_operator_names_knowledge_bases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_AUTO_REINDEX_KBS", "")
    get_settings.cache_clear()
    assert reindex.auto_reindex_kbs() == []
    monkeypatch.setenv("RAG_AUTO_REINDEX_KBS", " acme__hr , acme__legal ,")
    get_settings.cache_clear()
    assert reindex.auto_reindex_kbs() == ["acme__hr", "acme__legal"]
    get_settings.cache_clear()
