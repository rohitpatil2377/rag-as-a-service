"""E2E walking-skeleton slice through the real API (in-process ASGI, in-memory profile).

Proves the architecture end-to-end: auth + tenancy guard -> ingest (parse/mask/chunk/
embed/index) -> hybrid retrieve + rerank -> grounded-or-abstain + citations. This is
the POC proof point (PRD Agent-2-style grounded RAG).
"""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e

_BRIEF = (
    "Acme RAG as a Service is a sovereign multi-tenant RAG platform. "
    "The platform's uptime target is 99.5 percent baseline and 99.9 percent at peak. "
    "The P95 query latency target is 3 seconds or less. "
    "Contact officer@acme.example or phone 9876543210 for access."
)


async def _client(tenant: str) -> tuple[httpx.AsyncClient, dict[str, str]]:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["admin", "reader", "ingestor"])
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})
    return client, {}


async def test_ingest_query_grounded_and_abstain(inmemory: None, lexical_gate: None) -> None:
    tenant = "acme"
    client, _ = await _client(tenant)
    async with client:
        assert (await client.post("/v1/knowledge-bases", json={"display_name": "KB"})).status_code == 201

        r = await client.post(
            f"/v1/kb/{tenant}/ingest",
            json={
                "doc_id": "brief",
                "mime": "text/plain",
                "content_b64": base64.b64encode(_BRIEF.encode()).decode(),
                "strategy": "clause",
            },
        )
        assert r.status_code == 200
        ing = r.json()
        assert ing["chunk_count"] >= 1
        assert ing["pii_masked"] >= 2  # email + phone masked (FR-G2)

        # Grounded query with citations.
        r = await client.post(f"/v1/kb/{tenant}/query", json={"query": "What is the uptime target?", "top_k": 3})
        assert r.status_code == 200
        ans = r.json()
        assert ans["status"] == "grounded"
        assert ans["citations"], "grounded answer must carry citations"
        assert "99.5" in ans["answer"]
        # PII stays masked in the returned/grounded text.
        assert "officer@acme.example" not in ans["answer"]

        # Unanswerable query -> abstain (grounded-or-silent).
        r = await client.post(f"/v1/kb/{tenant}/query", json={"query": "Who won the 1998 world cup?"})
        ans = r.json()
        assert ans["status"] == "abstained"
        assert ans["citations"] == []
        assert ans["abstain_reason"] == "retrieval_confidence_below_threshold"


async def test_cross_tenant_query_is_denied(inmemory: None) -> None:
    """A principal for tenant B cannot query tenant A's KB (API-layer isolation)."""
    client, _ = await _client("tenant-b")
    async with client:
        r = await client.post("/v1/kb/tenant-a/query", json={"query": "anything"})
        assert r.status_code == 403
        assert r.json()["code"] == "tenant_isolation_violation"


async def test_the_audit_records_the_mode_that_actually_ran(inmemory: None, lexical_gate: None) -> None:
    """The QUERY event is on the SHA-256 hash chain — the artefact a compliance reviewer trusts —
    and it recorded `query.mode`, the mode the caller ASKED for, while HybridRetriever ignores
    that field and always runs dense + BM25 + RRF. An immutable log that faithfully records
    something that did not happen is worse than one that records nothing (P26)."""
    from datetime import UTC, datetime

    from svc_control_plane import pipeline

    from rag_clients.factory import get_audit_sink
    from rag_schemas import DocumentEnvelope, RetrievalQuery
    from rag_schemas.common import AuditAction, RetrievalMode, SourceKind

    tenant = "modeaudit"
    await pipeline.ingest_document(
        DocumentEnvelope(
            tenant_id=tenant,
            doc_id="brief",
            source_id="s",
            source_kind=SourceKind.UPLOAD,
            mime="text/plain",
            content_b64=base64.b64encode(_BRIEF.encode()).decode(),
            ingested_at=datetime(2026, 7, 31, tzinfo=UTC),
        ),
        actor="loader",
    )
    await pipeline.answer_query(
        RetrievalQuery(tenant_id=tenant, text="What is the uptime target?", top_k=3, mode=RetrievalMode.DENSE),
        actor="u",
    )
    events = [e for e in await get_audit_sink().list_events(tenant, None, 50) if e.action is AuditAction.QUERY]
    assert events, "no QUERY event recorded"
    payload = events[0].payload
    assert payload["mode"] == "hybrid", "the log still claims a mode the retriever never ran"
    assert payload["mode_requested"] == "dense", "the caller's request must stay visible, not be erased"
