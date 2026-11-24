"""E2E: FR-B7 auto-metadata & taxonomy tagging + D8 filtering on the derived tags."""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_AUTO_METADATA_ENABLED", "true")
    # GraphRAG + semantic cache on too: regression-guard that graph-augmented chunks respect
    # the metadata filter, and that a filtered query isn't served an un-filtered cached answer.
    monkeypatch.setenv("RAG_GRAPHRAG_ENABLED", "true")
    monkeypatch.setenv("RAG_SEMANTIC_CACHE_ENABLED", "true")
    from rag_common.config import get_settings

    get_settings.cache_clear()


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    token = issue_dev_jwt("admin", tenant, roles=["reader", "ingestor", "admin"])
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()),
        base_url="http://t",
        headers={"Authorization": f"Bearer {token}"},
    )


async def _ingest(c, kb, doc_id, text):
    b = base64.b64encode(text.encode()).decode()
    await c.post(
        f"/v1/kb/{kb}/ingest", json={"doc_id": doc_id, "mime": "text/plain", "content_b64": b, "strategy": "clause"}
    )


async def test_auto_tags_and_filter(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    kb = "tagco"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.put(f"/v1/kb/{kb}/settings", json={"faithful_threshold": 0.0, "abstain_threshold": 0.0})
        await c.put(
            f"/v1/kb/{kb}/taxonomy", json={"taxonomy": {"HR": ["leave", "allowance"], "Legal": ["agreement", "hereby"]}}
        )

        # The policy STATES its effective date, so it is dated. The agreement only mentions a
        # year in passing, so it is not — a year in body prose is not a document's date, and
        # treating it as one dated a real 2026 handbook as 2029 and a travel policy as 2000.
        await _ingest(
            c,
            kb,
            "leave",
            "The Night Shift Allowance Policy. Effective Date: 1st January 2026. Covers eligibility for all staff.",
        )
        await _ingest(c, kb, "nda", "This Agreement is hereby agree between the parties in 2025.")

        docs = {d["doc_id"]: d["metadata"] for d in (await c.get(f"/v1/kb/{kb}/documents")).json()}
        assert docs["leave"]["doc_type"] == "policy" and docs["leave"]["year"] == "2026"
        assert docs["leave"]["doc_date"] == "2026-01-01"
        assert "year" not in docs["nda"], "a year mentioned in prose must not date the document"
        assert docs["leave"]["category"] == "HR"
        assert docs["nda"]["doc_type"] == "agreement" and docs["nda"]["category"] == "Legal"

        # D8 filter on an auto-tag: only the HR-categorised doc is retrievable — and the
        # graph augment must NOT leak the excluded 'nda' back in.
        r = await c.post(f"/v1/kb/{kb}/query", json={"query": "What is covered?", "filters": {"category": "HR"}})
        cited = {cit["doc_id"] for cit in r.json().get("citations", [])}
        assert "nda" not in cited
        r2 = await c.post(
            f"/v1/kb/{kb}/query", json={"query": "What is covered?", "filters": {"doc_type": "agreement"}}
        )
        assert "leave" not in {cit["doc_id"] for cit in r2.json().get("citations", [])}


async def test_tagging_off_by_default(inmemory: None) -> None:
    kb = "tagoff"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await _ingest(c, kb, "d", "Some policy text from 2026.")
        meta = (await c.get(f"/v1/kb/{kb}/documents")).json()[0]["metadata"]
        assert "doc_type" not in meta  # tagging is opt-in


async def test_a_reindex_refreshes_derived_tags(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """B7 tags and the detected language are derived at INGEST only, so a reindex refreshed the
    extraction provenance and left everything else as it was.

    That makes a re-index half a re-index. Adopting a new taxonomy is one of the two reasons to
    run one at all (the other is a new chunking strategy), and the tags are D8 retrieval
    filters — so a KB could re-index, see the job succeed, and still filter on the old
    vocabulary with nothing anywhere saying why."""
    monkeypatch.setenv("RAG_AUTO_METADATA_ENABLED", "true")
    from rag_common.config import get_settings

    get_settings.cache_clear()

    kb = "retagco"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.put(f"/v1/kb/{kb}/taxonomy", json={"taxonomy": {"leave": ["annual leave", "entitlement"]}})
        body = base64.b64encode(b"Annual leave entitlement is 24 days for all confirmed staff.").decode()
        await c.post(f"/v1/kb/{kb}/ingest", json={"doc_id": "a", "mime": "text/plain", "content_b64": body})

        docs = (await c.get(f"/v1/kb/{kb}/documents")).json()
        assert docs[0]["metadata"].get("category") == "leave"

        # The tenant re-classifies their corpus and re-indexes to adopt it.
        await c.put(f"/v1/kb/{kb}/taxonomy", json={"taxonomy": {"benefits": ["annual leave", "entitlement"]}})
        r = await c.post(f"/v1/kb/{kb}/documents/a/reindex", json={})
        assert r.status_code == 200, r.text

        docs = (await c.get(f"/v1/kb/{kb}/documents")).json()
        assert docs[0]["metadata"].get("category") == "benefits", "reindex kept the old taxonomy's tag"
    get_settings.cache_clear()
