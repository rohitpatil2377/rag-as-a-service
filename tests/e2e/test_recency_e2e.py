"""E2E: FR-D9 recency-aware retrieval — time-window filter over real ingest."""

from __future__ import annotations

import base64
from datetime import UTC, datetime

import httpx
import pytest

pytestmark = pytest.mark.e2e


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    token = issue_dev_jwt("o", tenant, roles=["admin", "reader", "ingestor"])
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()),
        base_url="http://t",
        headers={"Authorization": f"Bearer {token}"},
    )


async def test_time_window_excludes_old_documents(inmemory: None) -> None:
    from rag_clients.factory import get_catalog

    kb = "recencyco"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.put(f"/v1/kb/{kb}/settings", json={"faithful_threshold": 0.0, "abstain_threshold": 0.0})
        for doc_id, txt in [
            ("old", "The parking policy in 2019 allowed free parking."),
            ("new", "The parking policy in 2026 requires a permit."),
        ]:
            b = base64.b64encode(txt.encode()).decode()
            await c.post(
                f"/v1/kb/{kb}/ingest",
                json={"doc_id": doc_id, "mime": "text/plain", "content_b64": b, "strategy": "clause"},
            )

        # Backdate the 'old' document's ingest time directly in the catalog twin.
        cat = get_catalog()
        old_doc = next(d for d in await cat.list_documents(kb) if d.doc_id == "old")
        await cat.register_document(old_doc.model_copy(update={"ingested_at": datetime(2019, 1, 1, tzinfo=UTC)}))

        # Unbounded ask can see both docs.
        allq = await c.post(f"/v1/kb/{kb}/query", json={"query": "What is the parking policy?", "top_k": 5})
        all_docs = {cit["doc_id"] for cit in allq.json().get("citations", [])}

        # Last-365-days window must exclude the 2019 doc.
        recent = await c.post(
            f"/v1/kb/{kb}/query", json={"query": "What is the parking policy?", "top_k": 5, "recency_days": 365}
        )
        recent_docs = {cit["doc_id"] for cit in recent.json().get("citations", [])}
        assert "old" not in recent_docs
        assert "old" in all_docs or "new" in all_docs  # sanity: retrieval worked at all


async def test_a_freshness_boost_reorders_without_understating_confidence(inmemory: None) -> None:
    """The freshness boost is a SORT key, not a rescoring — `apply_recency` deliberately keeps
    each span's own relevance score. So after the re-sort the first span is the freshness
    winner, not the relevance winner, and reading confidence off position 0 under-reports it:
    a genuinely strong match on an older document gets its score replaced by whatever the
    newest document happened to score. The gate then abstains for a reason that has nothing
    to do with relevance."""
    from rag_clients.factory import get_catalog

    kb = "freshconf"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.put(f"/v1/kb/{kb}/settings", json={"faithful_threshold": 0.0, "abstain_threshold": 0.0})
        for doc_id, txt in [
            ("old", "The parking permit rules are published annually."),
            ("new", "Visitor parking is unaffected by this notice."),
        ]:
            b = base64.b64encode(txt.encode()).decode()
            await c.post(
                f"/v1/kb/{kb}/ingest",
                json={"doc_id": doc_id, "mime": "text/plain", "content_b64": b, "strategy": "clause"},
            )

        # 'old' is the far better match but is years old, so a freshness boost sinks it.
        cat = get_catalog()
        old_doc = next(d for d in await cat.list_documents(kb) if d.doc_id == "old")
        await cat.register_document(old_doc.model_copy(update={"ingested_at": datetime(2019, 1, 1, tzinfo=UTC)}))

        # What retrieval judged, before any freshness re-sort: "old" is the far stronger match.
        spans = (await c.post(f"/v1/kb/{kb}/retrieve", json={"query": "parking permit rules", "top_k": 5})).json()[
            "spans"
        ]
        by_doc = {s["doc_id"]: s["score"] for s in spans}
        assert by_doc["old"] > by_doc["new"], "the fixture must give 'old' the better score, or this proves nothing"

        r = (
            await c.post(f"/v1/kb/{kb}/query", json={"query": "parking permit rules", "top_k": 5, "freshness": 1.0})
        ).json()
        cites = r["citations"]
        # The boost sank the better match: the answer is written from the FRESHEST span. (Sources are
        # cited-only since ground() started dropping spans the answer never used, so the weaker-but-
        # newer span is the only one listed — which is exactly the trap this test exists for.)
        assert cites[0]["doc_id"] == "new", "the boost must have actually reordered, or this proves nothing"
        # ...and the reported confidence is still the best score RETRIEVAL found, not the score of
        # whatever the re-sort left at position 0. Reading it off position 0 reported 0.33 for a
        # corpus that held a 1.0 match, and the gate then abstained for a reason unrelated to
        # relevance. Confidence answers "was anything relevant enough to ground an answer", so it
        # is deliberately NOT the cited span's score — see HybridRetriever + ADR-0015.
        assert r["confidence"] == by_doc["old"]
        assert r["confidence"] > cites[0]["score"]


async def test_recency_bypasses_semantic_cache(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: with the semantic cache ON, a recency-scoped query must NOT be served the
    earlier un-scoped cached answer (the cache key ignores the time window)."""
    monkeypatch.setenv("RAG_SEMANTIC_CACHE_ENABLED", "true")
    from rag_common.config import get_settings

    get_settings.cache_clear()

    kb = "recencycache"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.put(f"/v1/kb/{kb}/settings", json={"faithful_threshold": 0.0, "abstain_threshold": 0.0})
        b = base64.b64encode(b"The parking policy requires a permit.").decode()
        await c.post(
            f"/v1/kb/{kb}/ingest", json={"doc_id": "p", "mime": "text/plain", "content_b64": b, "strategy": "clause"}
        )

        # 1) Un-scoped query populates the cache with an answer citing doc 'p'.
        r1 = (await c.post(f"/v1/kb/{kb}/query", json={"query": "What is the parking policy?"})).json()
        assert r1["status"] == "grounded" and "p" in {cit["doc_id"] for cit in r1["citations"]}

        # 2) Same text, but a window that excludes the (just-ingested) doc. If the cache were
        #    consulted it would wrongly re-serve doc 'p'; the fix filters it out.
        r2 = (
            await c.post(
                f"/v1/kb/{kb}/query", json={"query": "What is the parking policy?", "before": "2000-01-01T00:00:00Z"}
            )
        ).json()
        assert "p" not in {cit["doc_id"] for cit in r2.get("citations", [])}
    get_settings.cache_clear()
