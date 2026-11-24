"""FR-C8 stale-index detection end to end: a real ingest through the real pipeline is stamped
with the running pipeline, and a document built by different settings is REPORTED as stale and
rebuilt by the corpus-wide re-index.

The unit tests pin the fingerprint's rules; this pins that the stamp actually reaches the
catalog on the normal ingest path -- a fingerprint nothing writes down detects nothing.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime

import pytest

pytestmark = pytest.mark.e2e

from svc_control_plane import reindex  # noqa: E402
from svc_indexing.service import META_PIPELINE_VERSION, pipeline_fingerprint  # noqa: E402

from rag_clients.factory import get_catalog  # noqa: E402
from rag_common import get_settings  # noqa: E402
from rag_schemas import DocumentEnvelope  # noqa: E402
from rag_schemas.common import SourceKind  # noqa: E402

_TENANT = "stalekb"


async def _ingest(doc_id: str, text: str) -> object:
    from svc_control_plane import pipeline

    return await pipeline.ingest_document(
        DocumentEnvelope(
            tenant_id=_TENANT,
            doc_id=doc_id,
            source_id="s",
            source_kind=SourceKind.UPLOAD,
            mime="text/plain",
            content_b64=base64.b64encode(text.encode()).decode(),
            ingested_at=datetime(2026, 9, 17, tzinfo=UTC),
        ),
        actor="uploader",
    )


async def test_a_fresh_ingest_is_stamped_and_reads_as_current(inmemory: None) -> None:
    await _ingest("fresh", "The retention period for audit logs is seven years.")

    docs = {d.doc_id: d for d in await get_catalog().list_documents(_TENANT)}
    assert docs["fresh"].metadata[META_PIPELINE_VERSION] == pipeline_fingerprint(get_settings())

    report = await reindex.stale_report(_TENANT)
    assert report.stale == 0, "a document just ingested by this pipeline cannot be stale"
    assert report.total >= 1


async def test_a_document_built_by_another_pipeline_is_reported_and_rebuilt(inmemory: None) -> None:
    await _ingest("drifted", "The escalation matrix has four tiers.")

    # Simulate the real cause: the document was indexed, then the deployment's chunker/embedder
    # changed. Rewriting the stamp is the same state the catalog would hold after such a change.
    doc = next(d for d in await get_catalog().list_documents(_TENANT) if d.doc_id == "drifted")
    await get_catalog().register_document(
        doc.model_copy(update={"metadata": {**doc.metadata, META_PIPELINE_VERSION: "0ldp1pel1ne"}})
    )

    report = await reindex.stale_report(_TENANT)
    assert "drifted" in report.stale_doc_ids

    result = await reindex.reindex_kb(_TENANT, actor="admin", stale_only=True, limit=10)
    assert result.reindexed >= 1
    assert result.failed == []

    # The point of the whole feature: after the rebuild the corpus reads as current, so the
    # banner clears on its own rather than needing a separate "mark as done".
    after = await reindex.stale_report(_TENANT)
    assert "drifted" not in after.stale_doc_ids
