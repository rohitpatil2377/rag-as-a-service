"""E2E: by-reference ingestion from object storage (FR-A2, ADR-0012).

Proves a document already sitting in the tenant's object-store namespace can be
ingested by reference (no inline bytes) and travels the SAME parse→chunk→index→retrieve
path as an upload — plus the request-contract guardrails that keep inline and
by-reference payloads mutually exclusive and tenant-scoped.
"""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e

_BRIEF = (
    "Acme RAG as a Service is a sovereign multi-tenant RAG platform. "
    "The platform's uptime target is 99.5 percent baseline and 99.9 percent at peak."
)


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["admin", "reader", "ingestor"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def test_ingest_by_reference_from_object_store(inmemory: None) -> None:
    """Seed an object into the tenant namespace, then ingest it by key (no content_b64)."""
    from rag_clients.factory import get_object_store

    tenant = "acme"
    object_key = "inbox/brief.txt"

    # A document lands in the tenant's own object-store namespace out-of-band.
    await get_object_store().put(tenant, object_key, _BRIEF.encode(), "text/plain")

    client = await _client(tenant)
    async with client:
        assert (await client.post("/v1/knowledge-bases", json={"display_name": "KB"})).status_code == 201

        r = await client.post(
            f"/v1/kb/{tenant}/ingest",
            json={
                "doc_id": "brief",
                "mime": "text/plain",
                "source_kind": "object_store",
                "object_key": object_key,
                "strategy": "clause",
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["chunk_count"] >= 1

        # Same downstream path → grounded, cited answer.
        r = await client.post(f"/v1/kb/{tenant}/query", json={"query": "What is the uptime target?", "top_k": 3})
        ans = r.json()
        assert ans["status"] == "grounded"
        assert ans["citations"]
        assert "99.5" in ans["answer"]


async def test_missing_object_reference_is_not_found(inmemory: None) -> None:
    """A by-reference ingest for a key that isn't in the namespace fails closed (404)."""
    tenant = "acme"
    client = await _client(tenant)
    async with client:
        await client.post("/v1/knowledge-bases", json={"display_name": "KB"})
        r = await client.post(
            f"/v1/kb/{tenant}/ingest",
            json={"doc_id": "ghost", "source_kind": "object_store", "object_key": "nope/missing.txt"},
        )
        assert r.status_code == 404
        assert r.json()["code"] == "not_found"


async def test_ingest_payload_contract_is_enforced(inmemory: None) -> None:
    """Inline and by-reference payloads are mutually exclusive, keyed by source_kind (422)."""
    tenant = "acme"
    client = await _client(tenant)
    inline = base64.b64encode(_BRIEF.encode()).decode()
    async with client:
        await client.post("/v1/knowledge-bases", json={"display_name": "KB"})

        # upload without content_b64
        r = await client.post(f"/v1/kb/{tenant}/ingest", json={"doc_id": "a", "source_kind": "upload"})
        assert r.status_code == 422

        # object_store without object_key
        r = await client.post(f"/v1/kb/{tenant}/ingest", json={"doc_id": "b", "source_kind": "object_store"})
        assert r.status_code == 422

        # both inline bytes AND an object_key
        r = await client.post(
            f"/v1/kb/{tenant}/ingest",
            json={"doc_id": "c", "source_kind": "object_store", "object_key": "k", "content_b64": inline},
        )
        assert r.status_code == 422


async def test_delete_document_purges_everything(inmemory: None) -> None:
    """Deleting a document removes it from the catalog and it stops appearing in answers."""
    tenant = "acme"
    client = await _client(tenant)
    async with client:
        await client.post("/v1/knowledge-bases", json={"display_name": "KB"})
        doc = base64.b64encode(b"The secret code is orange-42.").decode()
        r = await client.post(
            f"/v1/kb/{tenant}/ingest", json={"doc_id": "temp", "mime": "text/plain", "content_b64": doc}
        )
        assert r.status_code == 200

        # It's present before delete, and the KB can retrieve it.
        assert [d["doc_id"] for d in (await client.get(f"/v1/kb/{tenant}/documents")).json()] == ["temp"]
        before = (await client.post(f"/v1/kb/{tenant}/retrieve", json={"query": "secret code"})).json()
        assert before["spans"]

        # Delete it → gone from the catalog...
        assert (await client.delete(f"/v1/kb/{tenant}/documents/temp")).status_code == 204
        assert (await client.get(f"/v1/kb/{tenant}/documents")).json() == []

        # ...and from the indexes: nothing is retrievable, so the query abstains.
        after = (await client.post(f"/v1/kb/{tenant}/query", json={"query": "What is the secret code?"})).json()
        assert after["status"] == "abstained"

        # Deleting again is a no-op (idempotent), not an error.
        assert (await client.delete(f"/v1/kb/{tenant}/documents/temp")).status_code == 204


async def test_reindex_replays_from_raw_bytes(inmemory: None) -> None:
    """FR-C8: re-index rebuilds the chunks from stored raw bytes (no re-upload) and the
    document stays answerable; chunk count may change with the strategy."""
    tenant = "acme"
    client = await _client(tenant)
    async with client:
        await client.post("/v1/knowledge-bases", json={"display_name": "KB"})
        text = "The uptime target is 99.5 percent. The latency target is 3 seconds."
        r = await client.post(
            f"/v1/kb/{tenant}/ingest",
            json={
                "doc_id": "sla",
                "mime": "text/plain",
                "content_b64": base64.b64encode(text.encode()).decode(),
                "strategy": "fixed",
            },
        )
        assert r.status_code == 200

        # Re-index with a different strategy — replayed from the stored raw bytes.
        r = await client.post(f"/v1/kb/{tenant}/documents/sla/reindex", json={"strategy": "clause"})
        assert r.status_code == 200, r.text
        assert r.json()["chunk_count"] >= 1

        # Still exactly one document, still retrievable.
        assert [d["doc_id"] for d in (await client.get(f"/v1/kb/{tenant}/documents")).json()] == ["sla"]
        spans = (await client.post(f"/v1/kb/{tenant}/retrieve", json={"query": "uptime target"})).json()["spans"]
        assert spans and spans[0]["doc_id"] == "sla"

        # Re-indexing an unknown doc is a clean 404.
        assert (await client.post(f"/v1/kb/{tenant}/documents/ghost/reindex", json={})).status_code == 404
