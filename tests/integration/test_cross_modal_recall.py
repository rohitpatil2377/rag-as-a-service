"""Cross-modal recall, end-to-end through the real pipeline (in-memory twins).

Proves the VisionEmbedder seam is WIRED, not just present: ingesting an image indexes its CLIP
vector into a separate vision collection, the hybrid retriever's third leg finds the image by
that vector, and deleting the document purges it. All default-OFF — these behaviours appear only
with ``RAG_MULTIMODAL_ENABLED`` + ``RAG_VISION_RECALL_ENABLED`` both on.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime

import pytest

from rag_schemas import DocumentEnvelope, RetrievalQuery
from rag_schemas.common import SourceKind

# A tiny but non-trivial PNG-ish byte blob — the twins are content-derived, not real decoders.
_IMG = b"\x89PNG\r\n\x1a\n" + b"cross-modal diagram bytes: a bar chart of Q3 revenue" * 3
_TENANT = "vismm"


@pytest.fixture
def cross_modal(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn on multimodal + cross-modal recall and rebuild the singletons that read the flags."""
    from rag_clients import factory
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_MULTIMODAL_ENABLED", "true")
    monkeypatch.setenv("RAG_VISION_RECALL_ENABLED", "true")
    get_settings.cache_clear()
    factory.get_retriever.cache_clear()
    factory.get_vision_vector_store.cache_clear()


async def _ingest_image(doc_id: str) -> None:
    from svc_control_plane import pipeline

    await pipeline.ingest_document(
        DocumentEnvelope(
            tenant_id=_TENANT,
            doc_id=doc_id,
            source_id="s",
            source_kind=SourceKind.UPLOAD,
            mime="image/png",
            content_b64=base64.b64encode(_IMG).decode(),
            ingested_at=datetime(2026, 8, 5, tzinfo=UTC),
        ),
        actor="loader",
    )


async def _vision_hits(query_text: str, top_k: int = 10) -> list[str]:
    """Chunk ids the CLIP vision index returns for a text query (via the same twin ingest used)."""
    from rag_clients.factory import get_vision_embedder, get_vision_vector_store

    [qv] = await get_vision_embedder().embed_text([query_text])
    hits = await get_vision_vector_store().search(_TENANT, qv, top_k)
    return [cid for cid, _ in hits]


async def test_image_is_indexed_into_the_cross_modal_store(cross_modal: None) -> None:
    await _ingest_image("diagram")
    # The twin embeds an image from the sha1 of its bytes and text from its tokens into ONE space,
    # so a query equal to that digest lands on the image vector — a deterministic cross-modal hit.
    digest = hashlib.sha1(_IMG).hexdigest()
    hits = await _vision_hits(digest)
    assert any(cid.startswith("diagram::") for cid in hits), "the image's CLIP vector was not indexed"


async def test_retriever_third_leg_surfaces_the_image(cross_modal: None) -> None:
    from rag_clients.factory import get_retriever

    await _ingest_image("diagram")
    digest = hashlib.sha1(_IMG).hexdigest()
    # A query that shares NO caption words — only the visual vector can retrieve it.
    spans, _ = await get_retriever().retrieve(RetrievalQuery(tenant_id=_TENANT, text=digest, top_k=5))
    assert any(s.doc_id == "diagram" for s in spans), "cross-modal leg did not surface the image"


async def test_delete_purges_the_image_vector(cross_modal: None) -> None:
    from svc_control_plane import pipeline

    await _ingest_image("diagram")
    digest = hashlib.sha1(_IMG).hexdigest()
    assert await _vision_hits(digest), "precondition: image indexed"
    await pipeline.delete_document(_TENANT, "diagram", actor="loader")
    assert not await _vision_hits(digest), "the image vector must be purged on document delete"
