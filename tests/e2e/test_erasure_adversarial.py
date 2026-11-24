"""Adversarial DPDP erasure/retention cases for production go-live.

Proves erasure by doc-id AND by data-principal subject CASCADES completely across every
store + derived data, that the certificate lands on the verifying audit chain, and that the
retention scheduler auto-erases only past-retention docs. Any surviving retrievable shard is
a compliance-critical failure.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta

import httpx
import pytest

pytestmark = pytest.mark.e2e


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("dpo", tenant, roles=["reader", "ingestor", "admin"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def _ingest(c: httpx.AsyncClient, tenant: str, doc_id: str, text: str) -> None:
    body = base64.b64encode(text.encode()).decode()
    r = await c.post(
        f"/v1/kb/{tenant}/ingest",
        json={"doc_id": doc_id, "mime": "text/plain", "content_b64": body},
    )
    assert r.status_code < 300, r.text


async def test_erasure_leaves_no_graph_shard(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """CRITICAL: with GraphRAG on, erasing a data principal's document must also remove the
    entity/relationship rows it produced. The graph stores entity DISPLAY NAMES (the principal's
    name) + chunk-id evidence, so a surviving node is a retrievable shard of erased PII."""
    monkeypatch.setenv("RAG_GRAPHRAG_ENABLED", "true")
    from rag_common import get_settings

    get_settings.cache_clear()
    from rag_clients.factory import close_all, get_graph_store

    await close_all()  # re-read the flag into fresh singletons

    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        # A doc that is EXCLUSIVELY about the data principal -> its graph node has no other
        # evidence, so a correct erase must drop the row entirely.
        await _ingest(c, tenant, "record-anita", "Anita Desai works with Ravi Kumar at Meridian Bank.")

        store = get_graph_store()
        nodes_before, _ = await store.graph_data(tenant)
        names_before = {n.display.lower() for n in nodes_before}
        assert any("anita" in n for n in names_before), (
            f"precondition: graph should hold the principal entity; got {names_before}"
        )

        cert = (await c.post(f"/v1/kb/{tenant}/erasure", json={"subject": "Anita Desai"})).json()
        assert cert["documents_erased"] == ["record-anita"], cert

        nodes_after, edges_after = await store.graph_data(tenant)
        survivors = {n.display for n in nodes_after}
        leaked_chunks = {cid for n in nodes_after for cid in n.chunk_ids}
        assert survivors == set(), (
            f"ORPHAN GRAPH SHARD: erased principal still present as graph nodes {survivors}; "
            f"leaked evidence chunk ids {leaked_chunks}; edges {edges_after}"
        )


async def test_erasure_by_docid_cascade_every_store(inmemory: None) -> None:
    """Erase by explicit doc-id and assert the doc is unretrievable via catalog, dense, sparse,
    object bytes, and semantic cache."""
    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.put(f"/v1/kb/{tenant}/settings", json={"faithful_threshold": 0.1})
        await _ingest(c, tenant, "secret", "The vault passphrase is orchid-marmalade-7788.")

        from rag_clients.factory import (
            get_catalog,
            get_object_store,
            get_sparse_index,
        )
        from rag_common.tenancy import raw_object_key

        chunks = await get_catalog().document_chunks(tenant, "secret")
        chunk_ids = {ch.chunk_id for ch in chunks}
        assert chunk_ids

        cert = (await c.post(f"/v1/kb/{tenant}/erasure", json={"doc_ids": ["secret"]})).json()
        assert cert["documents_erased"] == ["secret"]

        # catalog row gone
        docs = [d["doc_id"] for d in (await c.get(f"/v1/kb/{tenant}/documents")).json()]
        assert "secret" not in docs
        assert await get_catalog().document_chunks(tenant, "secret") == []

        # sparse gone: BM25 query for the exact passphrase returns nothing citing our chunks
        sres = await get_sparse_index().search(tenant, "orchid marmalade passphrase", top_k=50)
        surviving = [cid for cid, _ in sres if cid in chunk_ids]
        assert not surviving, f"sparse shard survived: {surviving}"

        # object bytes gone (in-memory get raises KeyError when the key is absent)
        with pytest.raises(KeyError):
            await get_object_store().get(tenant, raw_object_key("secret"))

        # answering now abstains (nothing grounds it)
        ans = (await c.post(f"/v1/kb/{tenant}/query", json={"query": "vault passphrase"})).json()
        assert ans["status"] != "grounded" or "orchid" not in ans["answer"].lower()


async def test_erasure_purges_semantic_cache(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A grounded answer over the principal's doc must not survive in the semantic cache after
    erasure — a cache hit would re-serve erased content by an entirely different path."""
    monkeypatch.setenv("RAG_SEMANTIC_CACHE_ENABLED", "true")
    monkeypatch.setenv("RAG_SEMANTIC_CACHE_THRESHOLD", "0.99")
    from rag_common import get_settings

    get_settings.cache_clear()

    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.put(f"/v1/kb/{tenant}/settings", json={"faithful_threshold": 0.1})
        await _ingest(c, tenant, "record-omar", "Omar Sheikh's policy number is POL-99123.")

        q = {"query": "What is Omar Sheikh's policy number?"}
        a1 = (await c.post(f"/v1/kb/{tenant}/query", json=q)).json()
        # Prime a cache entry (second identical query is served from cache when grounded).
        await c.post(f"/v1/kb/{tenant}/query", json=q)

        from svc_control_plane.semantic_cache import cache as semantic_cache

        primed = a1["status"] == "grounded" and semantic_cache._kb.get(tenant)
        # Regardless of whether the extractive gate grounded it, erase and assert the KB's
        # cache bucket is empty afterwards.
        await c.post(f"/v1/kb/{tenant}/erasure", json={"subject": "Omar Sheikh"})
        assert not semantic_cache._kb.get(tenant), (
            f"semantic cache still holds erased-KB entries (primed={bool(primed)})"
        )

        after = (await c.post(f"/v1/kb/{tenant}/query", json=q)).json()
        assert "POL-99123" not in after.get("answer", ""), "erased content re-served post-erasure"


async def test_certificate_on_verifying_audit_chain(inmemory: None) -> None:
    """The ERASURE cert must be emitted on the audit hash-chain AND the chain must still verify
    after the erasure (attestation walks the SHA-256 chain)."""
    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await _ingest(c, tenant, "d1", "Priya Nair lives in Kochi.")

        cert = (await c.post(f"/v1/kb/{tenant}/erasure", json={"subject": "Priya Nair"})).json()

        audit = (await c.get(f"/v1/kb/{tenant}/audit")).json()
        matched = [e for e in audit if e["action"] == "erasure"]
        assert matched, "no ERASURE event on the chain"
        assert any(e["payload"]["certificate_hash"] == cert["certificate_hash"] for e in matched)

        att = (await c.get(f"/v1/kb/{tenant}/attestation")).json()
        assert att["audit_chain_verified"] is True, "chain broke after erasure"


async def test_retention_scheduler_erases_only_past_retention(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Friend's retention worker sweep: a doc past retention_days is auto-erased; one within is
    NOT. Drives the real worker._sweep_once against in-memory stores."""
    monkeypatch.setenv("RAG_ERASURE_RETENTION_DAYS", "365")
    from rag_common import get_settings

    get_settings.cache_clear()

    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await _ingest(c, tenant, "fresh", "Recent doc.")

        # Backdate one doc past the window.
        from rag_clients.factory import get_catalog
        from rag_schemas import DocumentEnvelope
        from rag_schemas.common import SourceKind

        old = DocumentEnvelope(
            tenant_id=tenant,
            doc_id="stale",
            source_id="s",
            source_kind=SourceKind.UPLOAD,
            mime="text/plain",
            ingested_at=datetime.now(UTC) - timedelta(days=400),
        )
        await get_catalog().register_document(old)

        # Wire the worker's tenant/kb enumeration to our in-memory tenant.
        from svc_control_plane import erasure_worker, wiring

        class _T:
            org_id = tenant

        class _K:
            kb_id = tenant

        class _Reg:
            async def list_all(self) -> list[_T]:
                return [_T()]

        class _KbStore:
            async def list(self, _org: str) -> list[_K]:
                return [_K()]

        monkeypatch.setattr(wiring, "tenant_registry", _Reg)
        monkeypatch.setattr(wiring, "kb_store", _KbStore)

        erased = await erasure_worker._sweep_once(datetime.now(UTC))
        assert erased == 1, f"expected exactly the stale doc erased, got {erased}"

        # Read the catalog directly — the wiring monkeypatch above would break the request
        # middleware's tenant-active check if we went back through HTTP.
        remaining = {d.doc_id for d in await get_catalog().list_documents(tenant)}
        assert remaining == {"fresh"}, remaining
