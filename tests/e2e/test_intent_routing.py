"""E2E: semantic intent routing (catalog / summarize / count / filter / compare) — deterministic,
in-memory.

Flag-gated (RAG_INTENT_ROUTING_ENABLED), heuristic backend (no model needed). Proves:
* a "list the documents" question returns the KB inventory as a GROUNDED answer with >=1
  real citation (one per visible document), and records ``intent=catalog`` on the QUERY audit;
* an empty / no-access KB abstains (grounded-or-abstain intact, no citations);
* a "recap this conversation" turn WITH prior history is a GROUNDED summary carrying the prior
  turns' citations; with NO history it abstains;
* "how many documents" reports the KB size (count) and abstains on an empty KB;
* "which documents recently/from 2026" filters the inventory — matching → GROUNDED, no match →
  ABSTAIN, no parseable criterion → falls back to the catalog list-all;
* "difference between A and B" contrasts two confidently-identified docs, and falls back to normal
  RAG when it cannot identify exactly two (never fabricates a comparison);
* a normal factual question is still plain RAG (regression);
* with the flag OFF a "list artifacts" question does NOT get the catalog handler (byte-identical
  routing — no ``intent`` on the audit, no catalog citation).
"""

from __future__ import annotations

import base64

import httpx
import pytest

from rag_schemas.common import AuditAction

pytestmark = pytest.mark.e2e

_DOC = "The uptime target for the platform is 99.5 percent measured monthly. The latency target is 3 seconds at P95."


def _enable_intent(monkeypatch: pytest.MonkeyPatch, *, on: bool) -> None:
    from rag_common.config import get_settings

    if on:
        monkeypatch.setenv("RAG_INTENT_ROUTING_ENABLED", "true")
        monkeypatch.setenv("RAG_INTENT_BACKEND", "heuristic")
    else:
        monkeypatch.delenv("RAG_INTENT_ROUTING_ENABLED", raising=False)
    get_settings.cache_clear()


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["reader", "ingestor"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def _seed(c: httpx.AsyncClient, tenant: str, *, doc_id: str = "sla") -> None:
    await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
    await c.put(f"/v1/kb/{tenant}/settings", json={"faithful_threshold": 0.1})
    doc = base64.b64encode(_DOC.encode()).decode()
    await c.post(f"/v1/kb/{tenant}/ingest", json={"doc_id": doc_id, "mime": "text/plain", "content_b64": doc})


async def test_catalog_intent_lists_documents(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_intent(monkeypatch, on=True)
    from rag_clients.factory import get_audit_sink

    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await _seed(c, tenant)
        r = await c.post(f"/v1/kb/{tenant}/query", json={"query": "list the documents in this knowledge base"})
        ans = r.json()

    assert ans["status"] == "grounded"
    assert ans["citations"], "catalog answer must carry >=1 citation"
    assert ans["structured"]["intent"] == "catalog"
    assert ans["structured"]["count"] == 1
    assert any(cit["doc_id"] == "sla" for cit in ans["citations"])

    events = await get_audit_sink().list_events(tenant, None, 100)
    q = next(e for e in events if e.action is AuditAction.QUERY)
    assert q.payload["intent"] == "catalog"


async def test_catalog_intent_abstains_on_empty_kb(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_intent(monkeypatch, on=True)
    tenant = "emptyorg"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        ans = (await c.post(f"/v1/kb/{tenant}/query", json={"query": "what files do we have?"})).json()

    assert ans["status"] == "abstained"
    assert ans["citations"] == []
    assert ans["abstain_reason"] == "catalog_empty"


async def test_summarize_turn_grounded_with_prior_citations(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_intent(monkeypatch, on=True)
    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await _seed(c, tenant)
        conv = (await c.post(f"/v1/kb/{tenant}/conversations", json={"query": "What is the uptime target?"})).json()
        cid = conv["conversation_id"]
        first = (
            await c.post(f"/v1/kb/{tenant}/conversations/{cid}/ask", json={"query": "What is the uptime target?"})
        ).json()
        assert first["status"] == "grounded"
        assert first["citations"]

        summ = (
            await c.post(f"/v1/kb/{tenant}/conversations/{cid}/ask", json={"query": "recap this conversation"})
        ).json()

    assert summ["status"] == "grounded"
    assert summ["citations"], "a summary must carry the prior turns' citations"
    assert summ["structured"]["intent"] == "summarize"
    assert summ["structured"]["turns"] == 1


async def test_summarize_turn_abstains_without_history(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_intent(monkeypatch, on=True)
    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await _seed(c, tenant)
        conv = (await c.post(f"/v1/kb/{tenant}/conversations", json={"query": "recap the chat"})).json()
        cid = conv["conversation_id"]
        summ = (
            await c.post(f"/v1/kb/{tenant}/conversations/{cid}/ask", json={"query": "recap this conversation"})
        ).json()

    assert summ["status"] == "abstained"
    assert summ["citations"] == []
    assert summ["abstain_reason"] == "nothing_to_summarize"


async def test_normal_question_still_rag(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_intent(monkeypatch, on=True)
    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await _seed(c, tenant)
        ans = (await c.post(f"/v1/kb/{tenant}/query", json={"query": "What is the uptime target?"})).json()

    assert ans["status"] == "grounded"
    assert "99.5" in ans["answer"]
    # NOT a catalog/summarize meta-answer.
    assert (ans["structured"] or {}).get("intent") != "catalog"
    assert not any(str(cit["chunk_id"]).endswith("#catalog") for cit in ans["citations"])


async def test_count_intent_reports_kb_size(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_intent(monkeypatch, on=True)
    from rag_clients.factory import get_audit_sink

    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await _seed(c, tenant, doc_id="sla")
        doc = base64.b64encode(_DOC.encode()).decode()
        await c.post(f"/v1/kb/{tenant}/ingest", json={"doc_id": "sla2", "mime": "text/plain", "content_b64": doc})
        ans = (await c.post(f"/v1/kb/{tenant}/query", json={"query": "how many documents are there?"})).json()

    assert ans["status"] == "grounded"
    assert ans["structured"]["intent"] == "count"
    assert ans["structured"]["count"] == 2
    assert "**2**" in ans["answer"]
    assert len(ans["citations"]) == 2  # one real citation per counted document

    events = await get_audit_sink().list_events(tenant, None, 100)
    q = next(e for e in events if e.action is AuditAction.QUERY)
    assert q.payload["intent"] == "count"


async def test_count_intent_abstains_on_empty_kb(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_intent(monkeypatch, on=True)
    tenant = "emptyorg"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        ans = (await c.post(f"/v1/kb/{tenant}/query", json={"query": "how many documents do we have?"})).json()

    assert ans["status"] == "abstained"
    assert ans["citations"] == []
    assert ans["abstain_reason"] == "catalog_empty"


async def test_filter_intent_recent_matches(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_intent(monkeypatch, on=True)
    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await _seed(c, tenant, doc_id="sla")  # ingested now → within any recency window
        ans = (await c.post(f"/v1/kb/{tenant}/query", json={"query": "which documents were added recently?"})).json()

    assert ans["status"] == "grounded"
    assert ans["structured"]["intent"] == "filter"
    assert ans["structured"]["count"] == 1
    assert any(cit["doc_id"] == "sla" for cit in ans["citations"])


async def test_filter_intent_no_match_abstains(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_intent(monkeypatch, on=True)
    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await _seed(c, tenant, doc_id="sla")  # plain text → no year metadata
        ans = (await c.post(f"/v1/kb/{tenant}/query", json={"query": "which documents are from 2026?"})).json()

    # A criterion that matches nothing ABSTAINS — never GROUNDED-with-0-citations.
    assert ans["status"] == "abstained"
    assert ans["citations"] == []
    assert ans["abstain_reason"] == "filter_no_match"


async def test_filter_intent_no_criteria_lists_all(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_intent(monkeypatch, on=True)
    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await _seed(c, tenant, doc_id="sla")
        # "which documents do we have" routes to filter, but no criterion parses → list all (catalog).
        ans = (await c.post(f"/v1/kb/{tenant}/query", json={"query": "which documents do we have?"})).json()

    assert ans["status"] == "grounded"
    assert ans["structured"]["intent"] == "catalog"  # fell back to the catalog inventory
    assert any(cit["doc_id"] == "sla" for cit in ans["citations"])


async def test_compare_intent_falls_back_to_rag_when_not_two(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_intent(monkeypatch, on=True)
    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await _seed(c, tenant, doc_id="sla")
        # Neither name matches a document label → cannot confidently identify two docs → normal RAG.
        ans = (
            await c.post(f"/v1/kb/{tenant}/query", json={"query": "compare the pricing sheet and the roadmap"})
        ).json()

    # NOT a compare meta-answer: it fell through to ordinary retrieval (no compare intent recorded).
    assert (ans["structured"] or {}).get("intent") != "compare"


async def test_compare_intent_contrasts_two_named_docs(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_intent(monkeypatch, on=True)
    from rag_clients.factory import get_audit_sink

    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.put(f"/v1/kb/{tenant}/settings", json={"faithful_threshold": 0.1})
        alpha = base64.b64encode(b"The alpha plan sets the uptime target at 99.5 percent measured monthly.").decode()
        beta = base64.b64encode(b"The beta plan sets the uptime target at 99.9 percent measured monthly.").decode()
        await c.post(f"/v1/kb/{tenant}/ingest", json={"doc_id": "alpha", "mime": "text/plain", "content_b64": alpha})
        await c.post(f"/v1/kb/{tenant}/ingest", json={"doc_id": "beta", "mime": "text/plain", "content_b64": beta})
        q = "difference between the alpha plan and the beta plan"
        ans = (await c.post(f"/v1/kb/{tenant}/query", json={"query": q})).json()

    # Two documents were confidently identified → the compare path ran (recorded on the QUERY audit).
    events = await get_audit_sink().list_events(tenant, None, 100)
    compare_q = next(
        (e for e in events if e.action is AuditAction.QUERY and e.payload.get("intent") == "compare"), None
    )
    assert compare_q is not None
    assert set(compare_q.payload["doc_ids"]) == {"alpha", "beta"}
    # Grounded-or-abstain holds: a grounded contrast carries citations; otherwise it abstains cleanly.
    if ans["status"] == "grounded":
        assert ans["citations"]
    else:
        assert ans["status"] == "abstained"
        assert ans["citations"] == []


async def test_doc_summary_intent_summarizes_each_doc(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression fix: "summarize each doc in detail" must route to doc_summary (NOT catalog)
    and return a per-document grounded summary citing EACH document's real chunks — never the bare
    name list."""
    _enable_intent(monkeypatch, on=True)
    from rag_clients.factory import get_audit_sink

    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.put(f"/v1/kb/{tenant}/settings", json={"faithful_threshold": 0.1})
        alpha = base64.b64encode(b"The alpha plan sets the uptime target at 99.5 percent measured monthly.").decode()
        beta = base64.b64encode(b"The beta plan sets the latency target at 3 seconds at the 95th percentile.").decode()
        await c.post(f"/v1/kb/{tenant}/ingest", json={"doc_id": "alpha", "mime": "text/plain", "content_b64": alpha})
        await c.post(f"/v1/kb/{tenant}/ingest", json={"doc_id": "beta", "mime": "text/plain", "content_b64": beta})
        ans = (
            await c.post(f"/v1/kb/{tenant}/query", json={"query": "summarize each doc which we have in detail"})
        ).json()

    # The intent was doc_summary (recorded on the QUERY audit) — not catalog.
    events = await get_audit_sink().list_events(tenant, None, 100)
    q = next(e for e in events if e.action is AuditAction.QUERY and e.payload.get("intent") == "doc_summary")
    assert q.payload["intent"] == "doc_summary"
    assert q.payload["total"] == 2

    assert ans["structured"]["intent"] == "doc_summary"
    # Grounded-or-abstain holds: a grounded per-doc summary cites real chunks from EACH doc.
    if ans["status"] == "grounded":
        assert ans["citations"], "a doc_summary must carry >=1 real citation"
        cited_docs = {cit["doc_id"] for cit in ans["citations"]}
        assert cited_docs == {"alpha", "beta"}, cited_docs  # a section (and citations) per document
        assert "**" in ans["answer"]  # per-doc sections are labelled
        assert not any(str(cit["chunk_id"]).endswith("#catalog") for cit in ans["citations"])  # not the name list
    else:
        assert ans["status"] == "abstained"
        assert ans["citations"] == []


async def test_doc_summary_intent_abstains_on_empty_kb(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_intent(monkeypatch, on=True)
    tenant = "emptyorg"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        ans = (await c.post(f"/v1/kb/{tenant}/query", json={"query": "summarize every document in detail"})).json()

    assert ans["status"] == "abstained"
    assert ans["citations"] == []
    assert ans["abstain_reason"] == "catalog_empty"


async def test_flag_off_path_unchanged(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_intent(monkeypatch, on=False)
    from rag_clients.factory import get_audit_sink

    tenant = "acme"
    c = await _client(tenant)
    async with c:
        await _seed(c, tenant)
        r = await c.post(f"/v1/kb/{tenant}/query", json={"query": "list the artifacts in this knowledge base"})
        ans = r.json()

    # With intent routing OFF this is a plain RAG query — never the catalog handler.
    assert (ans["structured"] or {}).get("intent") is None
    assert not any(str(cit["chunk_id"]).endswith("#catalog") for cit in ans["citations"])

    events = await get_audit_sink().list_events(tenant, None, 100)
    q = next(e for e in events if e.action is AuditAction.QUERY)
    assert "intent" not in q.payload
