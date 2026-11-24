"""E2E: GraphRAG local search (FR-C6/D4) — multi-hop recall + graph stats.

The graph links entities across DIFFERENT documents. A question about entity A pulls in
the chunks of A's graph neighbour B (which live in another doc) — recall that pure
passage-similarity misses. Flag-gated; deterministic heuristic extraction (no model).
"""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_GRAPHRAG_ENABLED", "true")
    from rag_common.config import get_settings

    get_settings.cache_clear()


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["admin", "reader", "ingestor"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def _ingest(c: httpx.AsyncClient, tenant: str, doc_id: str, text: str) -> None:
    body = base64.b64encode(text.encode()).decode()
    await c.post(
        f"/v1/kb/{tenant}/ingest",
        json={"doc_id": doc_id, "mime": "text/plain", "content_b64": body, "strategy": "clause"},
    )


async def test_graph_stats_and_multihop_recall(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch)
    tenant = "cag"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.put(f"/v1/kb/{tenant}/settings", json={"faithful_threshold": 0.1})
        # Doc A links "Project Falcon" ↔ "Rohit Sharma"; Doc B links "Rohit Sharma" ↔ "Finance Ministry".
        await _ingest(c, tenant, "a", "Project Falcon is led by Rohit Sharma.")
        await _ingest(c, tenant, "b", "Rohit Sharma reports to the Finance Ministry.")

        # The graph was built across both docs.
        graph = (await c.get(f"/v1/kb/{tenant}/graph")).json()
        assert graph["enabled"] is True
        assert graph["nodes"] >= 3 and graph["edges"] >= 2
        names = {e["name"] for e in graph["top_entities"]}
        assert {"Project Falcon", "Rohit Sharma", "Finance Ministry"} <= names

        # A question about Project Falcon pulls in the Finance Ministry chunk (doc B) via the
        # shared "Rohit Sharma" node — a 2-doc hop plain retrieval would miss on a real corpus.
        # (No control assertion on plain retrieval here: this KB holds two documents and the
        # in-memory embedder is a hash, so top_k returns both whatever the question is. What the
        # hop adds is only separable from ordinary recall on a corpus larger than the test's.)
        # top_k=1 is what makes this a HOP test rather than a recall test: passage search alone
        # reaches doc A only, so doc B can arrive by one route — the shared "Rohit Sharma" node.
        # (Without the cap this two-document KB returns both documents to any query, and the
        # assertion passes whether or not the graph did anything, which is how it used to pass.)
        ans = (
            await c.post(f"/v1/kb/{tenant}/query", json={"query": "What is Project Falcon connected to?", "top_k": 1})
        ).json()
        # The graph step reports what it contributed. Asserted on the trace rather than on the
        # citations because sources are CITED-ONLY (see grounding._cited_only): the evidence the
        # graph adds is only cited if the model happens to quote it, and the test twin quotes
        # exactly one span — so a citation check here measures the twin, not the hop.
        graph_steps = [s for s in ans["agent_trace"] if s["step"] == "graph"]
        assert graph_steps, f"the graph contributed nothing: {ans['agent_trace']}"
        assert "linked passages" in graph_steps[0]["detail"]


async def test_graph_chunks_are_scored_not_assigned_a_constant(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Graph-linked chunks used to be stamped with a fabricated score of 0.5, above the default
    0.35 abstain threshold. `extract_entities` is a proper-noun regex and `linked_chunks` a pure
    structural lookup — neither carries ANY relevance signal — so a question naming one indexed
    entity lifted max_score to 0.5 no matter what the graph dragged in, and the KB could not
    abstain. The fabricated number also out-ranked genuine spans below 0.5 inside
    `merge_retrieved`, crowding the prompt.

    Driven at `_graph_augment` directly: plain retrieval on a tiny corpus returns everything,
    so end-to-end the graph would have nothing left to contribute."""
    _enable(monkeypatch)
    from svc_control_plane.pipeline import _graph_augment

    from rag_schemas import RetrievalQuery, RetrievalResult, RetrievedSpan

    tenant = "gategraph"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await _ingest(c, tenant, "a", "Project Falcon is led by Rohit Sharma.")
        await _ingest(c, tenant, "b", "Rohit Sharma reports to the Finance Ministry.")

        # A question naming the entity but sharing no vocabulary with the neighbour chunk.
        text = "Project Falcon aquarium snorkelling warranty"
        query = RetrievalQuery(tenant_id=tenant, text=text, top_k=5)
        # The seed's chunk_id is READ from the catalog, not spelled out: ids are content-derived
        # (P44), and a hand-written one silently stops matching the real chunk, so the augment
        # would re-add document "a" as if it were new and lift max_score off its own seed.
        doc_a = (await c.get(f"/v1/kb/{tenant}/documents/a/content")).json()["chunks"][0]
        seed = RetrievedSpan(chunk_id=doc_a["chunk_id"], doc_id="a", text=doc_a["text"], score=0.25, scorer="lexical")
        result, added = await _graph_augment(query, RetrievalResult(query=query, spans=[seed], max_score=0.25))

        assert added > 0, "the graph must actually have contributed, or this proves nothing"
        graph_spans = [s for s in result.spans if s.doc_id == "b"]
        assert graph_spans, "the neighbour chunk should have been pulled in"
        # It shares no terms with the question, so an honest scorer rates it 0.0 — and whatever
        # it rates it, something must have DONE the rating.
        assert all(s.scorer != "unscored" for s in graph_spans)
        assert graph_spans[0].score == 0.0
        # ...so the graph cannot drag the confidence above what real retrieval found.
        assert result.max_score == 0.25


async def test_graph_disabled_by_default(inmemory: None) -> None:
    tenant = "cag"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        assert (await c.get(f"/v1/kb/{tenant}/graph")).json()["enabled"] is False


async def test_communities_and_global_thematic_search(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Slice 2: build communities, then a broad question answers from community evidence
    across the KB (breadth a narrow top-k misses), with real citations."""
    _enable(monkeypatch)
    tenant = "cag2"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.put(f"/v1/kb/{tenant}/settings", json={"faithful_threshold": 0.0, "abstain_threshold": 0.0})
        # Two clearly separate themes (no shared entities) -> two communities.
        await _ingest(c, tenant, "travel", "Business Travel Policy covers Airfare and Hotel Reimbursement for Acme.")
        await _ingest(c, tenant, "night", "Night Shift Allowance is paid to Operations Engineers on the Nashik roster.")

        # Communities are empty until built (on-demand, never per-ingest).
        assert (await c.get(f"/v1/kb/{tenant}/graph/communities")).json()["communities"] == []

        built = (await c.post(f"/v1/kb/{tenant}/graph/communities")).json()
        assert built["communities"] >= 2

        coms = (await c.get(f"/v1/kb/{tenant}/graph/communities")).json()
        assert coms["enabled"] is True
        labels = " ".join(x["label"] for x in coms["communities"])
        assert "Business Travel Policy" in labels or "Night Shift Allowance" in labels

        # A broad question is served from community evidence and stays grounded with real citations.
        ans = (await c.post(f"/v1/kb/{tenant}/query", json={"query": "What are the main themes covered?"})).json()
        assert ans["status"] == "grounded"
        assert len(ans["citations"]) >= 1


async def test_a_rebuilt_document_never_leaves_communities_citing_dead_chunks(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Community rows hold EVIDENCE CHUNK IDS. Nodes and edges are rebuilt per document by
    _index_graph, but communities are built on demand and were not rebuilt at all, so once a
    document was re-chunked they pointed at chunks that no longer existed. Since P44 made ids
    content-derived, changed text changes ids for certain. Global/thematic search then answers
    from evidence that cannot be resolved (P48)."""
    _enable(monkeypatch)
    tenant = "comreindex"
    c = await _client(tenant)
    async with c:
        from rag_clients.factory import get_graph_store

        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await _ingest(c, tenant, "a", "Project Falcon is led by Rohit Sharma in Mumbai.")
        await _ingest(c, tenant, "b", "Rohit Sharma reports to the Finance Ministry in Delhi.")
        assert (await c.post(f"/v1/kb/{tenant}/graph/communities")).json()["communities"] >= 1

        before = {cid for com in await get_graph_store().list_communities(tenant) for cid in com.chunk_ids}
        assert before, "fixture produced no evidence chunk ids — the assertion would be vacuous"

        # Replace the document's content: content-derived ids guarantee the old ones are gone.
        await _ingest(c, tenant, "a", "Project Falcon was cancelled and its budget reassigned.")
        live = {
            ch["chunk_id"]
            for d in ("a", "b")
            for ch in (await c.get(f"/v1/kb/{tenant}/documents/{d}/content")).json()["chunks"]
        }
        assert before - live, "the rewrite did not move any id — the assertion would be vacuous"

        cited = {cid for com in await get_graph_store().list_communities(tenant) for cid in com.chunk_ids}
        assert cited <= live, f"communities still cite chunks that no longer exist: {cited - live}"
