"""E2E: domain-adaptive profile (Slice 6) — detect-and-suggest + typed domain-graph edges.

Detection senses a KB's domain and SUGGESTS knobs but moves none; apply confirms only the
query-time knobs and refuses reindex-class knobs without an explicit opt-in; typed domain-relation
edges land at ingest additively. Flag-off is byte-identical. Deterministic (heuristic backends).
"""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e

_CHEM_DOCS = {
    "syn": (
        "The synthesis of Aspirin proceeds when H2SO4 reacts with the alcohol. "
        "The recommended dose is 5 mg/kg administered at 25 degrees Celsius."
    ),
    "pk": (
        "The compound CC(=O)Oc1ccccc1C(=O)O is metabolised in the liver. "
        "A concentration of 10 mmol was measured and the formula H2O2 was detected."
    ),
}


def _enable_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_DOMAIN_PROFILE_ENABLED", "true")
    from rag_common.config import get_settings

    get_settings.cache_clear()


def _enable_profile_and_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_DOMAIN_PROFILE_ENABLED", "true")
    monkeypatch.setenv("RAG_GRAPHRAG_ENABLED", "true")
    from rag_common.config import get_settings

    get_settings.cache_clear()


async def _client(tenant: str, roles: list[str] | None = None) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=roles or ["admin", "reader", "ingestor"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def _ingest(c: httpx.AsyncClient, tenant: str, doc_id: str, text: str) -> None:
    body = base64.b64encode(text.encode()).decode()
    await c.post(
        f"/v1/kb/{tenant}/ingest",
        json={"doc_id": doc_id, "mime": "text/plain", "content_b64": body, "strategy": "clause"},
    )


async def test_detect_writes_profile_and_suggestion_but_moves_no_knob(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_profile(monkeypatch)
    tenant = "chempharma"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        for did, text in _CHEM_DOCS.items():
            await _ingest(c, tenant, did, text)

        # Before detection: no profile.
        pre = (await c.get(f"/v1/kb/{tenant}/profile")).json()
        assert pre["profile"] == {}
        assert pre["current_knobs"] == {}

        detected = (await c.post(f"/v1/kb/{tenant}/profile/detect")).json()
        assert detected["profile"]["domain"] == "chemistry"
        # A suggestion is produced, split into the two cost classes.
        assert "exact_boost" in detected["suggestion"]["knobs"]
        assert detected["suggestion"]["reindex_knobs"]  # non-empty (SUGGEST-ONLY)
        assert detected["suggestion"]["rationale"]

        # The crux of detect-and-suggest: NO knob was moved by detection.
        assert detected["current_knobs"] == {}
        assert (await c.get(f"/v1/kb/{tenant}/profile")).json()["current_knobs"] == {}


async def test_apply_moves_only_query_time_and_refuses_reindex_without_optin(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_profile(monkeypatch)
    tenant = "chemapply"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        for did, text in _CHEM_DOCS.items():
            await _ingest(c, tenant, did, text)
        await c.post(f"/v1/kb/{tenant}/profile/detect")

        # Apply the query-time knobs AND ask for a reindex knob WITHOUT opting in.
        r = await c.post(
            f"/v1/kb/{tenant}/profile/apply",
            json={
                "query_knobs": {"exact_boost": 3.0, "notation_grounding_enabled": 1.0},
                "reindex_knobs": {"domain_relations_enabled": 1.0},
                "allow_reindex": False,
            },
        )
        assert r.status_code == 200, r.text
        result = r.json()
        # Query-time knobs applied immediately.
        assert result["applied"] == {"exact_boost": 3.0, "notation_grounding_enabled": 1.0}
        # Reindex-class knob REFUSED (not written) + the operator is warned.
        assert result["refused_reindex"] == {"domain_relations_enabled": 1.0}
        assert result["reindex_warning"]
        # The knobs actually landed in tenant_config via the SAME override seam set_kb_settings uses
        # (current_knobs == pipeline.kb_threshold_overrides).
        assert result["current_knobs"]["exact_boost"] == 3.0
        assert result["current_knobs"]["notation_grounding_enabled"] == 1.0
        # The refused reindex knob did NOT land.
        assert "domain_relations_enabled" not in result["current_knobs"]


async def test_apply_reindex_knob_only_with_explicit_optin(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_profile(monkeypatch)
    tenant = "chemreindex"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        for did, text in _CHEM_DOCS.items():
            await _ingest(c, tenant, did, text)
        await c.post(f"/v1/kb/{tenant}/profile/detect")

        r = await c.post(
            f"/v1/kb/{tenant}/profile/apply",
            json={"reindex_knobs": {"domain_relations_enabled": 1.0}, "allow_reindex": True},
        )
        assert r.status_code == 200, r.text
        result = r.json()
        assert result["applied"] == {"domain_relations_enabled": 1.0}
        assert result["refused_reindex"] == {}
        assert result["current_knobs"]["domain_relations_enabled"] == 1.0


async def test_reader_cannot_detect_or_apply(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_profile(monkeypatch)
    tenant = "chemreader"
    admin = await _client(tenant)
    async with admin:
        await admin.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await _ingest(admin, tenant, "syn", _CHEM_DOCS["syn"])

    reader = await _client(tenant, roles=["reader"])
    async with reader:
        assert (await reader.post(f"/v1/kb/{tenant}/profile/detect")).status_code == 403
        assert (await reader.get(f"/v1/kb/{tenant}/profile")).status_code == 403
        assert (
            await reader.post(f"/v1/kb/{tenant}/profile/apply", json={"query_knobs": {"exact_boost": 3.0}})
        ).status_code == 403


async def test_flag_off_disables_endpoints_and_writes_no_profile(inmemory: None) -> None:
    """Master flag OFF: the detect/apply endpoints refuse (422) and ingest writes no profile keys —
    the KB behaves exactly as today (byte-identical additive metadata is never created)."""
    tenant = "chemoff"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await _ingest(c, tenant, "syn", _CHEM_DOCS["syn"])

        assert (await c.post(f"/v1/kb/{tenant}/profile/detect")).status_code == 422
        # GET is a read (not gated by the master flag) but returns nothing was ever written.
        view = (await c.get(f"/v1/kb/{tenant}/profile")).json()
        assert view["enabled"] is False
        assert view["profile"] == {}
        assert view["suggestion"] == {}

        # A normal query still works — the flag-off retrieval path is unchanged.
        ans = (await c.post(f"/v1/kb/{tenant}/query", json={"query": "What is the dose?"})).json()
        assert ans["status"] in ("grounded", "flagged_for_review", "abstained")


async def test_typed_domain_edges_land_at_ingest(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sub-slice B: with a chemistry profile + the per-KB gate on, ingesting a document ALSO writes
    directed, labeled domain-relation edges via the existing add_diagram_edges path — additive, on
    top of the untyped co-occurrence edges."""
    _enable_profile_and_graph(monkeypatch)
    from rag_clients.factory import get_graph_store

    tenant = "chemtyped"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        for did, text in _CHEM_DOCS.items():
            await _ingest(c, tenant, did, text)
        # Detect chemistry, then explicitly opt in to the reindex-class relations gate.
        assert (await c.post(f"/v1/kb/{tenant}/profile/detect")).json()["profile"]["domain"] == "chemistry"
        await c.post(
            f"/v1/kb/{tenant}/profile/apply",
            json={"reindex_knobs": {"domain_relations_enabled": 1.0}, "allow_reindex": True},
        )

        # A NEW document ingested now that the profile + gate are set: its relations become typed edges.
        await _ingest(c, tenant, "rel", "Aspirin inhibits COX-2. Warfarin binds to Albumin.")

        store = get_graph_store()
        relations = store._edge_relations.get(tenant, {})  # type: ignore[attr-defined]  # in-memory twin detail
        labels = set(relations.values())
        assert "inhibits" in labels, f"typed relation edge not written: {relations}"
        # A directed edge with the label exists (source/target order preserved, not sorted).
        assert ("aspirin", "cox-2") in relations

        # Co-occurrence edges are STILL written alongside (additive) — the graph has both kinds.
        _nodes, edges = await store.graph_data(tenant)
        assert edges, "co-occurrence edges must still be written"

        # Grounded-or-abstain is intact: a query still answers within the invariant.
        ans = (await c.post(f"/v1/kb/{tenant}/query", json={"query": "What does Aspirin inhibit?"})).json()
        assert ans["status"] in ("grounded", "flagged_for_review", "abstained")


async def test_flag_off_writes_no_typed_edges(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the profile master flag OFF (graphrag on), ingest writes ONLY co-occurrence edges — no
    typed domain relations — even if a stray domain_relations_enabled knob is present."""
    monkeypatch.setenv("RAG_GRAPHRAG_ENABLED", "true")
    from rag_common.config import get_settings

    get_settings.cache_clear()
    from rag_clients.factory import get_graph_store

    tenant = "chemnotyped"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        # Even with the gate key set, the master flag being off means no typed extraction runs.
        await c.put(f"/v1/kb/{tenant}/settings", json={"domain_relations_enabled": 1.0})
        await _ingest(c, tenant, "rel", "Aspirin inhibits COX-2. Warfarin binds to Albumin.")

        store = get_graph_store()
        relations = store._edge_relations.get(tenant, {})  # type: ignore[attr-defined]
        assert relations == {}, f"typed edges were written with the master flag off: {relations}"


async def test_apply_validates_all_keys_before_writing(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad reindex key rejects the WHOLE request (422) with NO partial apply — the valid
    query-time knob in the same request is not persisted."""
    _enable_profile(monkeypatch)
    tenant = "chemvalidate"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        for did, text in _CHEM_DOCS.items():
            await _ingest(c, tenant, did, text)
        await c.post(f"/v1/kb/{tenant}/profile/detect")

        r = await c.post(
            f"/v1/kb/{tenant}/profile/apply",
            json={"query_knobs": {"exact_boost": 3.0}, "reindex_knobs": {"not_a_real_knob": 1.0}},
        )
        assert r.status_code == 422, r.text
        # The valid query-time knob was NOT persisted (validate-all-then-write, no partial apply).
        knobs = (await c.get(f"/v1/kb/{tenant}/profile")).json()["current_knobs"]
        assert "exact_boost" not in knobs, f"a rejected request left a partial apply: {knobs}"


async def test_apply_invalidates_semantic_cache(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Applying a query-time knob invalidates the KB's semantic cache so the change is live
    immediately (a stale cached answer would otherwise reflect the pre-apply knobs)."""
    _enable_profile(monkeypatch)
    from svc_control_plane.semantic_cache import cache

    invalidated: list[str] = []
    real_invalidate = cache.invalidate
    monkeypatch.setattr(cache, "invalidate", lambda kb: (invalidated.append(kb), real_invalidate(kb))[1])

    tenant = "chemcache"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        for did, text in _CHEM_DOCS.items():
            await _ingest(c, tenant, did, text)
        await c.post(f"/v1/kb/{tenant}/profile/detect")
        invalidated.clear()  # ignore any ingest-time invalidations

        await c.post(f"/v1/kb/{tenant}/profile/apply", json={"query_knobs": {"exact_boost": 3.0}})
        assert tenant in invalidated, "apply did not invalidate the KB's semantic cache"


async def test_relations_gate_on_but_graphrag_off_warns_and_writes_nothing(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The AND-graphrag coupling: with domain profiling on but GraphRAG OFF, a KB that enabled the
    relations gate gets a one-line WARN and ZERO typed edges (never a silent no-op)."""
    _enable_profile(monkeypatch)  # domain profile master ON, graphrag stays OFF
    from svc_control_plane import pipeline

    warnings: list[str] = []
    monkeypatch.setattr(pipeline._log, "warning", lambda event, **kw: warnings.append(event))

    from rag_clients.factory import get_graph_store

    tenant = "chemgraphoff"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        for did, text in _CHEM_DOCS.items():
            await _ingest(c, tenant, did, text)
        await c.post(f"/v1/kb/{tenant}/profile/detect")
        await c.post(
            f"/v1/kb/{tenant}/profile/apply",
            json={"reindex_knobs": {"domain_relations_enabled": 1.0}, "allow_reindex": True},
        )
        warnings.clear()
        # A fresh ingest triggers _index_graph, which (graphrag off) warns and writes no typed edges.
        await _ingest(c, tenant, "rel", "Aspirin inhibits COX-2.")

        assert "domain_relations_enabled_but_graphrag_off" in warnings
        store = get_graph_store()
        assert store._edge_relations.get(tenant, {}) == {}  # type: ignore[attr-defined]
