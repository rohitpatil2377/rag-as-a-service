"""E2E: by-reference ingestion from an EXTERNAL tenant bucket (FR-A2b, ADR-0013).

The critical property under test is sovereignty: the platform pulls from an external
S3 endpoint ONLY when that endpoint is on the egress allowlist, credentials arrive by
reference (never in the request), and a non-allowlisted endpoint is refused before any
secret is resolved or byte fetched — the guard is never loosened to make a pull work.
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
_ENDPOINT = "obj.sovereign.local"
_BUCKET = "tenant-exports"
_KEY = "briefs/uptime.txt"
_CRED_REF = "acmesrc"


def _external_source(secure: bool = True) -> dict[str, object]:
    return {"endpoint": _ENDPOINT, "bucket": _BUCKET, "secure": secure, "credentials_ref": _CRED_REF}


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["reader", "ingestor"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


def _allow(monkeypatch: pytest.MonkeyPatch, *hosts: str) -> None:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", ",".join(("localhost", *hosts)))
    get_settings.cache_clear()


def _seed_external() -> None:
    from rag_clients.factory import get_external_object_fetcher

    fetcher = get_external_object_fetcher()
    fetcher.seed(_ENDPOINT, _BUCKET, _KEY, _BRIEF.encode())  # type: ignore[attr-defined]


async def test_ingest_from_allowlisted_external_bucket(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Allowlisted endpoint + resolvable creds → pulled, indexed, and answerable."""
    _allow(monkeypatch, _ENDPOINT)
    monkeypatch.setenv("RAG_ACMESRC_ACCESS_KEY", "AKIA-dev")
    monkeypatch.setenv("RAG_ACMESRC_SECRET_KEY", "shh-dev")
    _seed_external()

    tenant = "acme"
    client = await _client(tenant)
    async with client:
        await client.post("/v1/knowledge-bases", json={"display_name": "KB"})
        r = await client.post(
            f"/v1/kb/{tenant}/ingest",
            json={
                "doc_id": "ext-brief",
                "mime": "text/plain",
                "source_kind": "object_store",
                "object_key": _KEY,
                "external_source": _external_source(),
                "strategy": "clause",
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["chunk_count"] >= 1

        r = await client.post(f"/v1/kb/{tenant}/query", json={"query": "What is the uptime target?", "top_k": 3})
        ans = r.json()
        assert ans["status"] == "grounded"
        assert "99.5" in ans["answer"]


async def test_non_allowlisted_endpoint_is_refused(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The sovereignty guard blocks a non-allowlisted endpoint (403), fetching nothing."""
    # Do NOT allowlist _ENDPOINT; creds present so failure is unambiguously the guard.
    monkeypatch.setenv("RAG_ACMESRC_ACCESS_KEY", "AKIA-dev")
    monkeypatch.setenv("RAG_ACMESRC_SECRET_KEY", "shh-dev")
    _seed_external()

    tenant = "acme"
    client = await _client(tenant)
    async with client:
        await client.post("/v1/knowledge-bases", json={"display_name": "KB"})
        r = await client.post(
            f"/v1/kb/{tenant}/ingest",
            json={
                "doc_id": "ext-brief",
                "source_kind": "object_store",
                "object_key": _KEY,
                "external_source": _external_source(),
            },
        )
        assert r.status_code == 403
        assert r.json()["code"] == "sovereignty_violation"

        # Nothing was ingested → the KB has no grounding for the query.
        r = await client.post(f"/v1/kb/{tenant}/query", json={"query": "What is the uptime target?"})
        assert r.json()["status"] == "abstained"


async def test_missing_credential_secret_fails(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Allowlisted endpoint but the referenced secret isn't loaded → server config error."""
    _allow(monkeypatch, _ENDPOINT)
    _seed_external()  # no RAG_ACMESRC_*_KEY set

    tenant = "acme"
    client = await _client(tenant)
    async with client:
        await client.post("/v1/knowledge-bases", json={"display_name": "KB"})
        r = await client.post(
            f"/v1/kb/{tenant}/ingest",
            json={
                "doc_id": "ext-brief",
                "source_kind": "object_store",
                "object_key": _KEY,
                "external_source": _external_source(),
            },
        )
        assert r.status_code == 500
        assert r.json()["code"] == "config_error"


async def test_external_source_requires_object_store_kind(inmemory: None) -> None:
    """external_source with an upload is a contract error (422), never silently ignored."""
    tenant = "acme"
    client = await _client(tenant)
    inline = base64.b64encode(_BRIEF.encode()).decode()
    async with client:
        await client.post("/v1/knowledge-bases", json={"display_name": "KB"})
        r = await client.post(
            f"/v1/kb/{tenant}/ingest",
            json={"doc_id": "x", "source_kind": "upload", "content_b64": inline, "external_source": _external_source()},
        )
        assert r.status_code == 422
