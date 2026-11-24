"""E2E: maker-checker (dual control) on a destructive action (banking pack).

With dual control enabled, a purge does not execute immediately — it returns 202 and stages a
pending approval. The maker cannot approve their own request (segregation of duties); a DIFFERENT
admin must approve, which then runs the purge.
"""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e


def _app_client(subject: str, tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    # Cleared for classified uploads: the maker-checker gate is a SECOND control on top of the
    # RBAC ingestion-level rule (an uploader must itself be cleared for the level it ingests).
    token = issue_dev_jwt(subject, tenant, roles=["owner", "admin", "ingestor"], attrs={"clearance": "restricted"})
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://t",
        headers={"Authorization": f"Bearer {token}"},
    )


async def test_purge_requires_second_admin_approval(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from svc_control_plane import wiring
    from svc_control_plane.dual_control_actions import register_executors

    from rag_common import get_settings

    monkeypatch.setenv("RAG_DUAL_CONTROL_ENABLED", "true")
    get_settings.cache_clear()
    register_executors()
    wiring.approval_store.cache_clear()  # fresh in-memory approval store for this test

    tenant = "bank"
    maker = _app_client("maker", tenant)
    async with maker:
        await maker.post("/v1/knowledge-bases", json={"display_name": "HR"})
        kb_id = "bank__hr"
        b64 = base64.b64encode(b"Leave policy: 12 days casual leave.").decode()
        assert (
            await maker.post(f"/v1/kb/{kb_id}/ingest", json={"doc_id": "d1", "mime": "text/plain", "content_b64": b64})
        ).status_code == 200

        # Purge is GATED → 202 + a pending request, NOT executed.
        r = await maker.post(f"/v1/knowledge-bases/{kb_id}/purge", json={"confirm_name": "HR"})
        assert r.status_code == 202
        body = r.json()
        assert body["code"] == "approval_required"
        req_id = body["details"]["request_id"]

        # The KB still exists (nothing was purged).
        assert (await maker.get(f"/v1/kb/{kb_id}/documents")).status_code == 200

        # Pending appears in the approvals list.
        pend = (await maker.get("/v1/admin/approvals", params={"status": "pending"})).json()
        assert any(a["id"] == req_id and a["action"] == "purge" for a in pend)

        # Separation of duties: the MAKER cannot approve their own request.
        assert (await maker.post(f"/v1/admin/approvals/{req_id}/approve")).status_code == 403

    # A DIFFERENT admin approves → the purge runs.
    checker = _app_client("checker", tenant)
    async with checker:
        done = await checker.post(f"/v1/admin/approvals/{req_id}/approve")
        assert done.status_code == 200
        assert done.json()["status"] == "executed"
        # The KB is now gone (documents endpoint 404s or lists nothing for a purged KB).
        docs = await checker.get(f"/v1/kb/{tenant}__hr/documents")
        assert docs.status_code == 404 or docs.json() == []

    get_settings.cache_clear()
    wiring.approval_store.cache_clear()


async def test_classified_ingest_is_gated_public_is_not(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from svc_control_plane import wiring
    from svc_control_plane.dual_control_actions import register_executors

    from rag_common import get_settings

    monkeypatch.setenv("RAG_DUAL_CONTROL_ENABLED", "true")
    get_settings.cache_clear()
    register_executors()
    wiring.approval_store.cache_clear()

    tenant = "bank2"
    maker = _app_client("maker", tenant)
    async with maker:
        await maker.post("/v1/knowledge-bases", json={"display_name": "Deals"})
        kb_id = "bank2__deals"
        b64 = base64.b64encode(b"Project Kohinoor consideration is 8600 crore.").decode()

        # A PUBLIC upload ingests immediately (not gated).
        pub = await maker.post(
            f"/v1/kb/{kb_id}/ingest", json={"doc_id": "pub", "mime": "text/plain", "content_b64": b64}
        )
        assert pub.status_code == 200

        # A CONFIDENTIAL upload is gated → 202 + pending, NOT ingested yet.
        conf = await maker.post(
            f"/v1/kb/{kb_id}/ingest",
            json={"doc_id": "secret", "mime": "text/plain", "content_b64": b64, "classification": "confidential"},
        )
        assert conf.status_code == 202
        req_id = conf.json()["details"]["request_id"]
        docs = (await maker.get(f"/v1/kb/{kb_id}/documents")).json()
        assert {d["doc_id"] for d in docs} == {"pub"}  # secret not ingested yet

    # A second admin approves → the classified doc is ingested.
    checker = _app_client("checker", tenant)
    async with checker:
        done = await checker.post(f"/v1/admin/approvals/{req_id}/approve")
        assert done.status_code == 200 and done.json()["status"] == "executed"
        docs = (await checker.get(f"/v1/kb/{kb_id}/documents")).json()
        assert {d["doc_id"] for d in docs} == {"pub", "secret"}

    get_settings.cache_clear()
    wiring.approval_store.cache_clear()


async def test_publish_kb_requires_approval_and_blocks_queries(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from svc_control_plane import wiring
    from svc_control_plane.dual_control_actions import register_executors

    from rag_common import get_settings

    monkeypatch.setenv("RAG_DUAL_CONTROL_ENABLED", "true")
    get_settings.cache_clear()
    register_executors()
    wiring.approval_store.cache_clear()

    tenant = "bank3"
    maker = _app_client("maker", tenant)
    async with maker:
        # A new KB is created as a DRAFT (unpublished) under the publish gate.
        created = (await maker.post("/v1/knowledge-bases", json={"display_name": "Corp"})).json()
        assert created["published"] is False
        kb_id = "bank3__corp"
        b64 = base64.b64encode(b"The corporate travel cap is 50000 rupees.").decode()
        assert (
            await maker.post(f"/v1/kb/{kb_id}/ingest", json={"doc_id": "t", "mime": "text/plain", "content_b64": b64})
        ).status_code == 200

        # End-user querying a draft KB is blocked (403).
        assert (await maker.post(f"/v1/kb/{kb_id}/query", json={"query": "travel cap?"})).status_code == 403

        # Requesting publish is gated → 202 pending; the KB stays a draft.
        pub = await maker.post(f"/v1/kb/{kb_id}/publish")
        assert pub.status_code == 202
        req_id = pub.json()["details"]["request_id"]
        assert (await maker.post(f"/v1/kb/{kb_id}/query", json={"query": "travel cap?"})).status_code == 403

    # A second admin approves publishing → queries are allowed.
    checker = _app_client("checker", tenant)
    async with checker:
        done = await checker.post(f"/v1/admin/approvals/{req_id}/approve")
        assert done.status_code == 200 and done.json()["status"] == "executed"
        ans = await checker.post(f"/v1/kb/{kb_id}/query", json={"query": "What is the travel cap?"})
        assert ans.status_code == 200  # published now → queries answer

    get_settings.cache_clear()
    wiring.approval_store.cache_clear()
