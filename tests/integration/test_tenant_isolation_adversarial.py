"""Adversarial tenant-isolation probes (SECURITY-CRITICAL go-live gate).

Extends ``test_tenant_isolation.py``. Where that file proves the happy-path store
namespacing, this file tries to BREAK isolation across every documented crossing:

* API: cross-tenant ``/v1/kb/{other}/...`` with a token scoped to a different tenant.
* API key: a key whose ``org_id`` != the target KB.
* MCP tools: a credential bound to one org reaching another org's KB (``scoped_tenant``).
* Contextvar vs explicit param: can ambient ``tenant_scope`` override the explicit
  ``tenant_id`` an adapter is handed?
* Semantic cache: same query + same ACL, different tenant — no cross-tenant replay.
* Graph augment: the GraphRAG local/global path stays inside the querying tenant.
* tenant_id slug collisions: two DISTINCT tenant_ids that normalise to the same
  namespace slug (``a-b`` vs ``a_b``) must NOT share a store.

Runs on the in-memory profile (same Protocols/namespacing as the live stores).
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from datetime import UTC, datetime

import httpx
import pytest

from rag_common import issue_dev_jwt, tenant_scope
from rag_schemas import DocumentEnvelope
from rag_schemas.common import SourceKind

pytestmark = pytest.mark.integration

ALPHA, BETA = "alpha-bank", "beta-gov"
_SECRET = "The alpha ledger secret balance is 42 crore rupees. Uptime target 99.5 percent."


def _envelope(tenant: str, doc_id: str = "doc1", text: str = _SECRET) -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id=tenant,
        doc_id=doc_id,
        source_id="s",
        source_kind=SourceKind.UPLOAD,
        mime="text/plain",
        content_b64=base64.b64encode(text.encode()).decode(),
        ingested_at=datetime(2026, 7, 2, tzinfo=UTC),
    )


@pytest.fixture
def publish_off(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_DUAL_CONTROL_ACTIONS", "")  # no publish gate
    monkeypatch.setenv("RAG_ABSTAIN_THRESHOLD_LEXICAL", "0.35")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _client(tenant: str, roles: list[str], *, groups: list[str] | None = None) -> httpx.AsyncClient:
    from rag_api.app import create_app

    token = issue_dev_jwt("officer", tenant, roles=roles, attrs={"groups": groups or []})
    transport = httpx.ASGITransport(app=create_app())
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


def _client_token(token: str) -> httpx.AsyncClient:
    from rag_api.app import create_app

    transport = httpx.ASGITransport(app=create_app())
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


# --------------------------------------------------------------------------------------
# 1) API: a token scoped to BETA cannot address ALPHA's default KB nor a KB ALPHA created.
# --------------------------------------------------------------------------------------
async def test_api_cross_tenant_kb_access_denied(inmemory: None, publish_off: None) -> None:
    from svc_control_plane import pipeline

    with tenant_scope(ALPHA):
        await pipeline.ingest_document(_envelope(ALPHA), actor="alpha-officer")

    beta = _client(BETA, ["owner", "reader", "ingestor"])
    async with beta:
        # ALPHA's default KB (kb_id == ALPHA).
        r = await beta.post(f"/v1/kb/{ALPHA}/query", json={"query": "alpha ledger secret"})
        assert r.status_code in (403, 404), r.text
        body = r.text.lower()
        assert "42 crore" not in body and "ledger secret" not in body

        # A KB ALPHA created (prefix form).
        r2 = await beta.post(f"/v1/kb/{ALPHA}__docs/query", json={"query": "alpha ledger secret"})
        assert r2.status_code in (403, 404), r2.text

        # Retrieve-only path too.
        r3 = await beta.post(f"/v1/kb/{ALPHA}/retrieve", json={"query": "alpha ledger secret"})
        assert r3.status_code in (403, 404), r3.text

        # Documents listing of ALPHA's KB.
        r4 = await beta.get(f"/v1/kb/{ALPHA}/documents")
        assert r4.status_code in (403, 404), r4.text


# --------------------------------------------------------------------------------------
# 2) Prefix-collision: a tenant whose id is a PREFIX of another cannot reach it.
#    tenant_guard uses "startswith(f'{tid}__')" and slug_kb_id uses "__" separator.
# --------------------------------------------------------------------------------------
async def test_api_prefix_sibling_cannot_cross(inmemory: None, publish_off: None) -> None:
    from svc_control_plane import pipeline

    victim = "acme-hq"  # its default KB id is "acme-hq"
    attacker = "acme"  # prefix of victim
    with tenant_scope(victim):
        await pipeline.ingest_document(_envelope(victim, text="acme-hq restricted merger memo"), actor="v")

    client = _client(attacker, ["owner", "reader"])
    async with client:
        # "acme-hq" != "acme" and does not start with "acme__" -> must be denied.
        r = await client.post(f"/v1/kb/{victim}/query", json={"query": "merger memo"})
        assert r.status_code in (403, 404), r.text
        assert "merger memo" not in r.text.lower()


# --------------------------------------------------------------------------------------
# 3) API key whose org_id != target KB is rejected (I4 key resolves to its OWN org only).
# --------------------------------------------------------------------------------------
async def test_api_key_org_mismatch_denied(inmemory: None, publish_off: None) -> None:
    from svc_control_plane import pipeline, wiring
    from svc_control_plane.api_keys import ApiKeyRecord, mint_key

    with tenant_scope(ALPHA):
        await pipeline.ingest_document(_envelope(ALPHA), actor="alpha-officer")

    # Mint a real BETA-scoped admin key straight into the store.
    full, prefix, key_hash = mint_key()
    rec = ApiKeyRecord(
        key_id="k1",
        org_id=BETA,
        name="beta-key",
        prefix=prefix,
        scope="admin",
        created_at=datetime(2026, 7, 2, tzinfo=UTC),
    )
    await wiring.api_key_store().create(rec, key_hash)

    client = _client_token(full)
    async with client:
        # The key authenticates (BETA principal) but cannot address ALPHA's KB.
        r = await client.post(f"/v1/kb/{ALPHA}/query", json={"query": "alpha ledger secret"})
        assert r.status_code in (403, 404), r.text
        assert "42 crore" not in r.text

        # And it CAN reach its own org (sanity: the key works at all).
        own = await client.post(f"/v1/kb/{BETA}/query", json={"query": "hello"})
        assert own.status_code == 200, own.text


# --------------------------------------------------------------------------------------
# 4) MCP scoped_tenant: a credential bound to BETA cannot target ALPHA's KB.
# --------------------------------------------------------------------------------------
async def test_mcp_scoped_tenant_cross_org_denied(inmemory: None) -> None:
    from rag_common import UnauthorizedError
    from rag_mcp import context as mcp_ctx

    t_token = mcp_ctx.ctx_tenant.set(BETA)
    try:
        # No kb_id -> own namespace.
        assert mcp_ctx.scoped_tenant(None) == BETA
        # Own KB (prefix) -> allowed.
        assert mcp_ctx.scoped_tenant(f"{BETA}__kb") == f"{BETA}__kb"
        # ALPHA's default KB -> refused.
        with pytest.raises(UnauthorizedError):
            mcp_ctx.scoped_tenant(ALPHA)
        # ALPHA's created KB -> refused.
        with pytest.raises(UnauthorizedError):
            mcp_ctx.scoped_tenant(f"{ALPHA}__kb")
        # Prefix sibling: BETA-ish cannot reach "beta-gov-extra".
        with pytest.raises(UnauthorizedError):
            mcp_ctx.scoped_tenant("beta-gov-extra")
    finally:
        mcp_ctx.ctx_tenant.reset(t_token)


# --------------------------------------------------------------------------------------
# 5) MCP retrieve tool actually returns ONLY the bound org's data, never ALPHA's.
# --------------------------------------------------------------------------------------
async def test_mcp_retrieve_tool_is_tenant_bound(inmemory: None, lexical_gate: None) -> None:
    from svc_control_plane import pipeline

    from rag_mcp import context as mcp_ctx
    from rag_mcp.server import retrieve_tool

    with tenant_scope(ALPHA):
        await pipeline.ingest_document(_envelope(ALPHA), actor="alpha-officer")

    # A BETA-bound MCP request must see nothing from ALPHA.
    t_token = mcp_ctx.ctx_tenant.set(BETA)
    a_token = mcp_ctx.ctx_acl.set(())
    try:
        out = await retrieve_tool("alpha ledger secret balance", top_k=5)
        spans = out.get("spans", [])
        text_blob = " ".join(s.get("text", "") for s in spans).lower()
        assert "ledger secret" not in text_blob and "42 crore" not in text_blob
        assert spans == [], f"BETA MCP retrieve leaked ALPHA spans: {spans}"
    finally:
        mcp_ctx.ctx_tenant.reset(t_token)
        mcp_ctx.ctx_acl.reset(a_token)


# --------------------------------------------------------------------------------------
# 6) Contextvar cannot OVERRIDE the explicit tenant_id an adapter is handed.
#    Bind tenant_scope(ALPHA) ambiently, but call the store with explicit BETA -> BETA wins.
# --------------------------------------------------------------------------------------
async def test_explicit_tenant_id_beats_ambient_contextvar(inmemory: None) -> None:
    from svc_control_plane import pipeline

    from rag_clients import factory

    with tenant_scope(ALPHA):
        await pipeline.ingest_document(_envelope(ALPHA), actor="alpha-officer")

    vs = factory.get_vector_store()
    sparse = factory.get_sparse_index()
    catalog = factory.get_catalog()
    objects = factory.get_object_store()
    embedder = factory.get_embedder()
    [qvec] = await embedder.embed(["uptime target"])

    alpha_ids = [cid for cid, _ in await vs.search(ALPHA, qvec, top_k=5)]
    assert alpha_ids, "precondition: ALPHA indexed something"

    # Now, with ALPHA ambiently bound, hit every store with an EXPLICIT BETA — the explicit
    # arg must win (isolation is by the explicit param, not the contextvar).
    with tenant_scope(ALPHA):
        assert await vs.search(BETA, qvec, top_k=5) == []
        assert await sparse.search(BETA, "uptime", top_k=5) == []
        assert await catalog.get_chunks(BETA, alpha_ids) == {}
        with pytest.raises(KeyError):
            await objects.get(BETA, "raw/doc1")

    # And the mirror: BETA ambiently bound, explicit ALPHA still resolves ALPHA (param wins).
    with tenant_scope(BETA):
        assert await vs.search(ALPHA, qvec, top_k=5) != []
        assert set(await catalog.get_chunks(ALPHA, alpha_ids)) == set(alpha_ids)


# --------------------------------------------------------------------------------------
# 7) Semantic cache: same query + same ACL, DIFFERENT tenant — no cross-tenant replay.
# --------------------------------------------------------------------------------------
async def test_semantic_cache_no_cross_tenant_replay(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from svc_control_plane.semantic_cache import cache

    from rag_common.config import get_settings
    from rag_schemas import AnswerStatus, Citation, GroundedAnswer

    monkeypatch.setenv("RAG_SEMANTIC_CACHE_ENABLED", "true")
    get_settings.cache_clear()
    cache.clear()  # module singleton — start from a known-empty state
    # Prime each KB's observed version so a prior test's invalidate() (which bumped the shared
    # coordinator counter) does not evict our freshly-put entry on the next get(). This is
    # test-harness coordinator hygiene, not product behaviour.
    cache.get(ALPHA, [1.0], ["_prime"])
    cache.get(BETA, [1.0], ["_prime"])

    vector = [0.1, 0.2, 0.3, 0.4]
    acl = ["public"]
    ans = GroundedAnswer(
        tenant_id=ALPHA,
        query="what is the ledger balance",
        status=AnswerStatus.GROUNDED,
        answer="ALPHA secret cached answer",
        citations=[Citation(chunk_id="c1", doc_id="d1", quote="alpha secret", score=0.9)],
        confidence=0.9,
    )
    # ALPHA caches an answer.
    cache.put(ALPHA, "what is the ledger balance", vector, acl, ans, clearance=0)
    # BETA asks the SAME embedding + SAME acl — must miss (different KB bucket).
    hit = cache.get(BETA, vector, acl, clearance=0)
    assert hit is None, f"semantic cache leaked ALPHA answer to BETA: {hit}"
    # ALPHA still hits its own.
    assert cache.get(ALPHA, vector, acl, clearance=0) is not None
    get_settings.cache_clear()


# --------------------------------------------------------------------------------------
# 8) GraphRAG local augment stays inside the querying tenant (graph_store per-tenant).
# --------------------------------------------------------------------------------------
async def test_graph_augment_is_tenant_scoped(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_clients.factory import get_graph_store
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_GRAPHRAG_ENABLED", "true")
    get_settings.cache_clear()

    store = get_graph_store()
    # ALPHA graph links entity "Ledger" to a secret chunk.
    await store.add_chunk(ALPHA, "alpha-chunk-secret", ["Ledger", "Balance"], [("Ledger", "Balance")])
    # BETA asks for the same entities -> gets NOTHING from ALPHA's graph.
    beta_linked = await store.linked_chunks(BETA, ["Ledger", "Balance"], hops=1)
    assert beta_linked == [], f"graph store leaked ALPHA chunks to BETA: {beta_linked}"
    # ALPHA sees its own.
    alpha_linked = await store.linked_chunks(ALPHA, ["Ledger"], hops=1)
    assert "alpha-chunk-secret" in alpha_linked
    # Stats/communities are per-tenant too.
    assert (await store.stats(BETA)).nodes == 0
    assert (await store.stats(ALPHA)).nodes >= 1
    get_settings.cache_clear()


# --------------------------------------------------------------------------------------
# 9) VULNERABILITY (documented): distinct tenant_ids that NORMALISE to the same slug
#    SHARE a namespace across EVERY store. ``_slug`` lowercases and maps '-'->'_', so
#    ``a-b``/``a_b`` and ``Acme``/``acme`` collide. This test ASSERTS the collision so
#    the finding is explicit and pinned; flip to ``!=`` once _slug is made injective.
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("t1,t2", [("a-b", "a_b"), ("Acme", "acme"), ("x-y", "x_y")])
def test_slug_collision_between_distinct_tenant_ids_is_a_leak(t1: str, t2: str) -> None:
    from rag_common import tenancy

    # These pairs are DISTINCT valid tenant_ids (both pass _slug's charset check) yet map
    # to IDENTICAL store namespaces — a cross-tenant isolation break waiting for two orgs
    # whose ids differ only by case or hyphen/underscore.
    assert tenancy.vector_collection(t1) == tenancy.vector_collection(t2)
    assert tenancy.sparse_index(t1) == tenancy.sparse_index(t2)
    assert tenancy.pg_schema(t1) == tenancy.pg_schema(t2)
    assert tenancy.object_prefix(t1) == tenancy.object_prefix(t2)
    assert tenancy.vision_collection(t1) == tenancy.vision_collection(t2)


# --------------------------------------------------------------------------------------
# 9b) END-TO-END EXPLOIT of the slug collision: tenant ``org_x`` reads tenant ``org-x``'s
#     data from the slug-namespaced stores (vector, sparse, object). The in-memory CATALOG
#     keys by the raw tenant_id (so it does NOT collide here) — but PostgresCatalog uses
#     pg_schema(tenant_id) = 't_org_x' for BOTH, so production Postgres leaks the catalog too.
# --------------------------------------------------------------------------------------
async def test_slug_collision_end_to_end_store_leak(inmemory: None) -> None:
    from svc_control_plane import pipeline

    from rag_clients import factory

    victim, attacker = "org-x", "org_x"  # DISTINCT ids, identical slug 't_org_x'
    secret = "TOPSECRET victim merger price is 900 crore. Confidential."
    with tenant_scope(victim):
        await pipeline.ingest_document(_envelope(victim, doc_id="secret-doc", text=secret), actor="victim")

    vs = factory.get_vector_store()
    sparse = factory.get_sparse_index()
    objects = factory.get_object_store()
    embedder = factory.get_embedder()
    [qvec] = await embedder.embed(["merger price"])

    # The attacker, using its OWN explicit tenant_id, reads the victim's slug-collided data.
    assert await vs.search(attacker, qvec, top_k=5) != [], "vector store did NOT leak (collision fixed?)"
    assert await sparse.search(attacker, "merger", top_k=5) != [], "sparse index did NOT leak"
    leaked_bytes = await objects.get(attacker, "raw/secret-doc")
    assert b"900 crore" in leaked_bytes, "object store did NOT leak the victim's raw bytes"


# --------------------------------------------------------------------------------------
# 10) A cross-tenant query end-to-end through the guarded API abstains (no ALPHA text).
# --------------------------------------------------------------------------------------
async def test_end_to_end_cross_tenant_abstains(inmemory: None, publish_off: None, lexical_gate: None) -> None:
    from svc_control_plane import pipeline

    with tenant_scope(ALPHA):
        await pipeline.ingest_document(_envelope(ALPHA), actor="alpha-officer")

    beta = _client(BETA, ["owner", "reader"])
    async with beta:
        # BETA queries ITS OWN (empty) KB with ALPHA's secret terms -> abstain, no leak.
        r = await beta.post(f"/v1/kb/{BETA}/query", json={"query": "alpha ledger secret 42 crore"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "abstained", body
        assert "42 crore" not in body.get("text", "")
        assert body.get("citations", []) == []
