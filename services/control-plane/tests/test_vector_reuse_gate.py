"""When already-stored vectors may be kept (`pipeline._may_reuse_vectors`, RAG v2 Phase 6).

The re-embed skip is only safe if the stored vectors came from the pipeline that is running now —
a different embed model or dimension, different chunk sizing, or contextual retrieval turned on or
off all mean the stored vector is not what this deployment would produce. `pipeline_fingerprint`
already encodes exactly that set, so the gate is a comparison against it, and it must fail CLOSED
everywhere else: the cost of a wrong "no" is one re-embed, the cost of a wrong "yes" is a vector
that silently disagrees with its text.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag_schemas import DocumentEnvelope
from rag_schemas.common import SourceKind


def _doc(**metadata: str) -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id="acme",
        doc_id="d1",
        source_id="s1",
        source_kind=SourceKind.UPLOAD,
        mime="text/plain",
        ingested_at=datetime(2026, 10, 21, tzinfo=UTC),
        metadata=metadata,
    )


def test_a_document_stamped_by_this_pipeline_may_reuse_its_vectors(inmemory: None) -> None:
    from svc_control_plane.pipeline import _may_reuse_vectors
    from svc_indexing.service import META_PIPELINE_VERSION, pipeline_fingerprint

    from rag_common import get_settings

    current = pipeline_fingerprint(get_settings())
    assert _may_reuse_vectors(_doc(**{META_PIPELINE_VERSION: current})) is True


def test_a_first_ingest_reuses_nothing(inmemory: None) -> None:
    """There is no previous version, so there is nothing to compare and nothing to keep."""
    from svc_control_plane.pipeline import _may_reuse_vectors

    assert _may_reuse_vectors(None) is False


def test_a_document_with_no_stamp_reuses_nothing(inmemory: None) -> None:
    """Ingested before the stamp existed. Unknown provenance is not permission."""
    from svc_control_plane.pipeline import _may_reuse_vectors

    assert _may_reuse_vectors(_doc()) is False


def test_a_document_from_a_different_pipeline_reuses_nothing(inmemory: None) -> None:
    from svc_control_plane.pipeline import _may_reuse_vectors
    from svc_indexing.service import META_PIPELINE_VERSION

    assert _may_reuse_vectors(_doc(**{META_PIPELINE_VERSION: "0000deadbeef"})) is False


@pytest.mark.parametrize(
    ("env", "value"),
    [
        ("RAG_EMBED_MODEL", "some-other-embedder"),
        ("RAG_EMBED_DIM", "384"),
        ("RAG_CHUNK_MAX_CHARS", "999"),
        ("RAG_CONTEXTUAL_RETRIEVAL_ENABLED", "true"),
    ],
)
def test_changing_anything_that_shapes_a_vector_withdraws_permission(
    inmemory: None, monkeypatch: pytest.MonkeyPatch, env: str, value: str
) -> None:
    """The gate is only as good as the fingerprint behind it. Each of these changes what a stored
    vector WOULD be, so a document stamped before the change must stop being reusable — asserted
    per setting rather than trusting that `pipeline_fingerprint` covers them."""
    from svc_control_plane.pipeline import _may_reuse_vectors
    from svc_indexing.service import META_PIPELINE_VERSION, pipeline_fingerprint

    from rag_common import get_settings

    stamped = _doc(**{META_PIPELINE_VERSION: pipeline_fingerprint(get_settings())})
    assert _may_reuse_vectors(stamped) is True

    monkeypatch.setenv(env, value)
    get_settings.cache_clear()
    try:
        assert _may_reuse_vectors(stamped) is False, f"{env} does not reach the fingerprint"
    finally:
        get_settings.cache_clear()
