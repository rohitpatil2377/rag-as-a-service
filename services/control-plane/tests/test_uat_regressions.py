"""Regression tests for the UAT bug-hunt fixes (F1-F5).

Each test pins the behaviour a fix introduced so it can't silently re-break:
- F1  a trashed KB is inaccessible at the access chokepoint (404), not merely hidden from the list
- F2  creating a KB whose name collides with a trashed one is rejected (409), not silently revived
- F3  purging a KB drops the vector collection + BM25 index (not just per-document chunks)
- F4  a missing/invalid content_b64 on upload is a clean 422, not a bare 500
- F5  an extractor failure (corrupt/encrypted/unsupported file) is a clean 422, not a bare 500

(F6 is a console/React change — graph refresh after re-index — verified by tsc + manual QA, not
pytest.)
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime

import pytest
from svc_control_plane import api, pipeline, wiring

from rag_common import tenant_scope
from rag_common.auth import Principal
from rag_common.errors import ConflictError, NotFoundError, ValidationError
from rag_schemas import DocumentEnvelope
from rag_schemas.common import SourceKind


def _principal(tenant: str = "acme", *roles: str) -> Principal:
    return Principal(subject=f"u@{tenant}", tenant_id=tenant, roles=frozenset(roles or ("admin",)), attrs={})


# --- F1: trashed KB -> 404 at the chokepoint -----------------------------------------


async def test_f1_assert_kb_access_404s_a_trashed_kb(inmemory: None) -> None:
    store = wiring.kb_store()
    await store.add("acme", "acme__hr", "HR")
    await store.trash("acme", "acme__hr")
    with pytest.raises(NotFoundError):
        await api._assert_kb_access(_principal(), "acme__hr")


async def test_f1_assert_kb_access_allows_an_active_kb(inmemory: None) -> None:
    store = wiring.kb_store()
    await store.add("acme", "acme__hr", "HR")
    # No raise: an active, in-scope, Public KB stays readable for a same-tenant caller.
    await api._assert_kb_access(_principal(), "acme__hr")


# --- F2: create over a trashed name -> 409 -------------------------------------------


async def test_f2_create_kb_409s_when_name_is_in_trash(inmemory: None) -> None:
    store = wiring.kb_store()
    await store.add("acme", "acme__hr", "HR")
    await store.trash("acme", "acme__hr")  # slug of "HR" == acme__hr
    with pytest.raises(ConflictError):
        await api.create_kb(api.CreateKBRequest(display_name="HR"), _principal("acme", "owner"))


async def test_f2_create_kb_succeeds_for_a_fresh_name(inmemory: None) -> None:
    info = await api.create_kb(api.CreateKBRequest(display_name="Finance"), _principal("acme", "owner"))
    assert info.kb_id == "acme__finance"
    active = {r.kb_id for r in await wiring.kb_store().list("acme")}
    assert "acme__finance" in active  # visible in the active list, not stuck in Trash


# --- F3: purge drops the store namespaces --------------------------------------------


async def test_f3_purge_drops_vector_and_sparse_namespaces(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_clients import factory

    calls: list[tuple[str, str]] = []
    vs = factory.get_vector_store()
    sx = factory.get_sparse_index()
    vs_orig, sx_orig = vs.delete_tenant, sx.delete_tenant

    async def spy_vs(tenant_id: str) -> None:
        calls.append(("vector", tenant_id))
        await vs_orig(tenant_id)

    async def spy_sx(tenant_id: str) -> None:
        calls.append(("sparse", tenant_id))
        await sx_orig(tenant_id)

    monkeypatch.setattr(vs, "delete_tenant", spy_vs)
    monkeypatch.setattr(sx, "delete_tenant", spy_sx)

    await wiring.kb_store().add("acme", "acme__hr", "HR")
    await pipeline.delete_knowledge_base("acme", "acme__hr", actor="u")

    assert ("vector", "acme__hr") in calls  # Qdrant collection dropped, not just per-doc chunks
    assert ("sparse", "acme__hr") in calls  # OpenSearch index dropped too


# --- F4: bad / missing content_b64 -> 422 (ValidationError) --------------------------


async def test_f4_upload_rejects_invalid_base64() -> None:
    from svc_ingestion.connectors import UploadConnector

    doc = DocumentEnvelope(
        tenant_id="t",
        doc_id="d",
        source_id="s",
        source_kind=SourceKind.UPLOAD,
        mime="text/plain",
        content_b64="abcde",  # 5 chars -> binascii.Error
        ingested_at=datetime(2026, 7, 2, tzinfo=UTC),
    )
    with pytest.raises(ValidationError):
        await UploadConnector().fetch(doc)


async def test_f4_upload_rejects_missing_content() -> None:
    from svc_ingestion.connectors import UploadConnector

    doc = DocumentEnvelope(
        tenant_id="t",
        doc_id="d",
        source_id="s",
        source_kind=SourceKind.UPLOAD,
        mime="text/plain",
        ingested_at=datetime(2026, 7, 2, tzinfo=UTC),
    )
    with pytest.raises(ValidationError):
        await UploadConnector().fetch(doc)


async def test_f4_upload_accepts_valid_base64() -> None:
    from svc_ingestion.connectors import UploadConnector

    doc = DocumentEnvelope(
        tenant_id="t",
        doc_id="d",
        source_id="s",
        source_kind=SourceKind.UPLOAD,
        mime="text/plain",
        content_b64=base64.b64encode(b"hello").decode(),
        ingested_at=datetime(2026, 7, 2, tzinfo=UTC),
    )
    assert await UploadConnector().fetch(doc) == b"hello"


# --- F5: extractor failure -> 422 (ValidationError), rollback intact -----------------


async def test_f5_extractor_failure_maps_to_422(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    class _BoomIntel:
        async def extract(self, doc: object, raw: bytes) -> object:
            raise RuntimeError("corrupt / encrypted PDF")

    async def _intel(_tenant_id: str) -> _BoomIntel:
        return _BoomIntel()

    monkeypatch.setattr(wiring, "doc_intel_service_for", _intel)

    doc = DocumentEnvelope(
        tenant_id="acme__hr",
        doc_id="d1",
        source_id="s",
        source_kind=SourceKind.UPLOAD,
        mime="application/pdf",
        content_b64=base64.b64encode(b"%PDF-broken").decode(),
        ingested_at=datetime(2026, 7, 2, tzinfo=UTC),
    )
    with tenant_scope("acme__hr"), pytest.raises(ValidationError):
        await pipeline.ingest_document(doc, actor="u")

    # Rollback intact: a failed NEW-doc ingest left nothing behind.
    from rag_clients.factory import get_catalog

    docs = await get_catalog().list_documents("acme__hr")
    assert all(d.doc_id != "d1" for d in docs)
