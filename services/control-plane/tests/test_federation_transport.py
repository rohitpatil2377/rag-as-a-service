"""Unit tests for the federation transport seam (ADR-0017/0020): Local + Mesh + registry,
grant-topic enforcement (L1), and partial-failure reachability honesty (T1)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from svc_control_plane import federation
from svc_control_plane.federation import (
    FederationRegistry,
    LocalTransport,
    MeshTransport,
    NullTransport,
    _span_from_wire,
)

from rag_schemas import Grant, RetrievalQuery, RetrievalResult, RetrievedSpan


def _grant(**over: Any) -> Grant:
    now = datetime(2026, 8, 3, tzinfo=UTC)
    base: dict[str, Any] = {
        "grant_id": "grant_test01",
        "issuer_dept_id": "hr",
        "subject_dept_id": "finance",
        "topics": ["leave-policy"],
        "kb_ids": ["hr__policies"],
        "max_results_per_query": 5,
        "not_before": now - timedelta(days=1),
        "expires_at": now + timedelta(days=180),
        "signed_token": "",
        "created_at": now,
    }
    base.update(over)
    return Grant(**base)


# ---- _span_from_wire ---------------------------------------------------------


def test_span_from_wire_clamps_and_names_scorer() -> None:
    span = _span_from_wire({"chunk_id": "c1", "doc_id": "d1", "text": "hi", "score": 1.7}, "hr")
    assert span.score == 1.0 and span.scorer == "cross_encoder"  # clamped into [0,1]
    assert span.chunk_id == "hr::c1" and span.doc_id == "hr/d1"  # namespaced by peer


def test_span_from_wire_zero_score_stays_unscored() -> None:
    span = _span_from_wire({"chunk_id": "c", "doc_id": "d", "text": "t", "score": 0}, "hr")
    assert span.score == 0.0 and span.scorer == "unscored"  # validator would reject score>0+unscored
    neg = _span_from_wire({"chunk_id": "c", "doc_id": "d", "text": "t", "score": -3}, "hr")
    assert neg.score == 0.0 and neg.scorer == "unscored"


def test_span_from_wire_carries_category() -> None:
    span = _span_from_wire(
        {"chunk_id": "c", "doc_id": "d", "text": "t", "score": 0.5, "category": "leave-policy"}, "hr"
    )
    assert span.category == "leave-policy"


# ---- MeshTransport -----------------------------------------------------------


def _mesh_with(handler: Any) -> MeshTransport:
    return MeshTransport("http://node:7000", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def _fetch(transport: Any, **over: Any) -> Any:
    kw: dict[str, Any] = {
        "caller_dept": "finance",
        "peer_dept_id": "hr",
        "grants": [_grant()],
        "query": "q",
        "top_k": 5,
    }
    kw.update(over)
    return await transport.fetch(**kw)


async def test_mesh_returns_spans_when_grant_valid() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/federate"
        return httpx.Response(
            200,
            json={
                "grant_valid": True,
                "peer_dept_id": "hr",
                "spans": [{"chunk_id": "c1", "doc_id": "d1", "text": "x", "score": 0.9}],
            },
        )

    pf = await _fetch(_mesh_with(handler))
    assert len(pf.sources) == 1 and pf.sources[0][0].chunk_id == "hr::c1"
    assert pf.used_grants[0]["transport"] == "mesh" and pf.used_grants[0]["grant_id"] == "grant_test01"
    assert pf.reachable is True and pf.reason == "ok"


async def test_mesh_fails_closed_on_invalid_grant() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"grant_valid": False, "error": "revoked", "spans": []})

    pf = await _fetch(_mesh_with(handler))
    assert pf.sources == [[]]  # a rejected grant yields no spans
    # A rejected grant is REACHED (the node answered "no"), not an availability failure.
    assert pf.reachable is True and pf.reason == "grant_rejected"


async def test_mesh_unreachable_peer_reports_not_reachable() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    pf = await _fetch(_mesh_with(handler))
    assert pf.sources == [[]]  # unreachable peer contributes nothing, does not crash the answer
    # T1: a down peer is REPORTED unreachable, never silently treated as complete.
    assert pf.reachable is False and pf.reason == "unreachable"
    assert pf.used_grants[0]["dept"] == "hr"


# ---- LocalTransport ----------------------------------------------------------


async def test_local_transport_reproduces_inprocess_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeRetrieval:
        async def retrieve(self, q: RetrievalQuery) -> RetrievalResult:
            captured["acl"] = q.acl
            captured["tenant"] = q.tenant_id
            span = RetrievedSpan(chunk_id="raw1", doc_id="docA", text="body", score=0.5, scorer="cross_encoder")
            return RetrievalResult(query=q, spans=[span], max_score=0.5)

    async def _fake_service_for(kb: str) -> _FakeRetrieval:
        return _FakeRetrieval()

    monkeypatch.setattr("svc_control_plane.federation.wiring.retrieval_service_for", _fake_service_for)

    pf = await _fetch(LocalTransport(), grants=[_grant(kb_ids=["hr__policies"])])
    # Least privilege: foreign retrieval always public-ACL.
    assert captured["acl"] == [] and captured["tenant"] == "hr__policies"
    # Spans relabelled for citation; used-grant record keeps the historical shape (no transport key).
    assert pf.sources[0][0].chunk_id == "hr__policies::raw1" and pf.sources[0][0].doc_id == "hr/docA"
    assert pf.used_grants == [
        {"dept": "hr", "grant_id": "grant_test01", "kb_ids": ["hr__policies"], "topics": ["leave-policy"]}
    ]
    # Co-located: always reachable.
    assert pf.reachable is True


# ---- FederationRegistry ------------------------------------------------------


class _FakeSettings:
    def __init__(
        self,
        transport: str,
        node_url: str = "",
        peers: str = "",
        topic_enforcement: bool = False,
        peer_timeout: float = 5.0,
    ) -> None:
        self.federation_transport = transport
        self.federation_node_url = node_url
        self.federation_peers = peers
        self.federation_topic_enforcement = topic_enforcement
        self.federation_peer_timeout_seconds = peer_timeout


def test_registry_default_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(federation, "get_settings", lambda: _FakeSettings("local"))
    assert isinstance(FederationRegistry.from_settings().transport_for("hr"), LocalTransport)


def test_registry_mesh_uses_node_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(federation, "get_settings", lambda: _FakeSettings("mesh", node_url="http://node:7000"))
    t = FederationRegistry.from_settings().transport_for("hr")
    assert isinstance(t, MeshTransport)


def test_registry_per_peer_override(monkeypatch: pytest.MonkeyPatch) -> None:
    peers = '{"hr": {"transport": "mesh", "node_url": "http://hr-node:7000"}}'
    monkeypatch.setattr(federation, "get_settings", lambda: _FakeSettings("local", peers=peers))
    reg = FederationRegistry.from_settings()
    assert isinstance(reg.transport_for("hr"), MeshTransport)  # overridden peer
    assert isinstance(reg.transport_for("legal"), LocalTransport)  # default


def test_registry_mesh_without_node_url_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    # An explicitly-mesh peer with no reachable node contributes NO spans (fail closed),
    # rather than silently degrading to an in-process lookup (security review L2).
    monkeypatch.setattr(federation, "get_settings", lambda: _FakeSettings("mesh", node_url=""))
    assert isinstance(FederationRegistry.from_settings().transport_for("hr"), NullTransport)


async def test_null_transport_reports_unreachable() -> None:
    pf = await _fetch(NullTransport())
    # T1: an unroutable mesh peer is reported unreachable (evidence missing), not silently empty.
    assert pf.sources == [] and pf.used_grants == []
    assert pf.reachable is False and pf.reason == "no_route"


def test_registry_coerces_non_string_node_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # A JSON number for node_url must not crash MeshTransport.rstrip (security review L3).
    peers = '{"hr": {"transport": "mesh", "node_url": 8080}}'
    monkeypatch.setattr(federation, "get_settings", lambda: _FakeSettings("local", peers=peers))
    t = FederationRegistry.from_settings().transport_for("hr")
    assert isinstance(t, MeshTransport)


# ---- L1: grant-topic enforcement (ADR-0017) ---------------------------------


class _CategoryRetrieval:
    async def retrieve(self, q: RetrievalQuery) -> RetrievalResult:
        spans = [
            RetrievedSpan(
                chunk_id="a", doc_id="d", text="t", score=0.5, scorer="cross_encoder", category="leave-policy"
            ),
            RetrievedSpan(chunk_id="b", doc_id="d", text="t", score=0.4, scorer="cross_encoder", category="salaries"),
            RetrievedSpan(chunk_id="c", doc_id="d", text="t", score=0.3, scorer="cross_encoder"),  # no category
        ]
        return RetrievalResult(query=q, spans=spans, max_score=0.5)


async def _svc_category(kb: str) -> _CategoryRetrieval:
    return _CategoryRetrieval()


async def test_topic_enforcement_on_keeps_only_granted_category(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("svc_control_plane.federation.wiring.retrieval_service_for", _svc_category)
    monkeypatch.setattr(federation, "get_settings", lambda: _FakeSettings("local", topic_enforcement=True))
    pf = await _fetch(LocalTransport(), grants=[_grant(topics=["leave-policy"])])
    kept = [s.chunk_id for src in pf.sources for s in src]
    # Fail closed: only the leave-policy span survives; wrong category AND missing category are dropped.
    assert kept == ["hr__policies::a"]


async def test_topic_enforcement_off_keeps_all(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("svc_control_plane.federation.wiring.retrieval_service_for", _svc_category)
    monkeypatch.setattr(federation, "get_settings", lambda: _FakeSettings("local", topic_enforcement=False))
    pf = await _fetch(LocalTransport(), grants=[_grant(topics=["leave-policy"])])
    kept = [s.chunk_id for src in pf.sources for s in src]
    assert len(kept) == 3  # default behaviour unchanged


async def test_mesh_topic_enforcement_filters_by_category(monkeypatch: pytest.MonkeyPatch) -> None:
    # A mesh node carries category on each span; with enforcement ON the initiator drops
    # spans outside the grant's topics, exactly like the local path (ADR-0017 L1, end-to-end).
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "grant_valid": True,
                "peer_dept_id": "hr",
                "spans": [
                    {"chunk_id": "a", "doc_id": "d", "text": "t", "score": 0.9, "category": "leave-policy"},
                    {"chunk_id": "b", "doc_id": "d", "text": "t", "score": 0.8, "category": "salaries"},
                ],
            },
        )

    monkeypatch.setattr(
        federation,
        "get_settings",
        lambda: _FakeSettings("mesh", node_url="http://node:7000", topic_enforcement=True),
    )
    pf = await _fetch(_mesh_with(handler), grants=[_grant(topics=["leave-policy"])])
    kept = [s.chunk_id for src in pf.sources for s in src]
    assert kept == ["hr::a"]  # only the granted-topic category survives, peer-namespaced


# ---- T1: partial-failure completeness note ----------------------------------


def test_federation_note_complete_vs_unreachable() -> None:
    from svc_control_plane.pipeline import _federation_note

    complete = _federation_note("finance", ["hr"], [])
    assert "finance" in complete and "hr" in complete and "unreachable" not in complete.lower()

    partial = _federation_note("finance", ["hr"], ["legal"])
    # Names what answered AND what is missing — never implies completeness.
    assert "finance" in partial and "hr" in partial
    assert "legal" in partial and "unreachable" in partial.lower()


# ---- T1 item 5: denial anomaly alert (SIEM) ---------------------------------


def test_federation_denial_alert_threshold() -> None:
    from svc_control_plane.pipeline import federation_denial_alert

    assert federation_denial_alert(["a", "b", "c"], 3) is True  # burst trips the alert
    assert federation_denial_alert(["a", "b"], 3) is False  # below threshold
    assert federation_denial_alert(["a", "b", "c", "d"], 0) is False  # 0 disables
    assert federation_denial_alert([], 3) is False  # no denials, no alert


# ---- T1 item 4: traceparent propagation (distributed tracing) ---------------


async def test_mesh_injects_traceparent(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen["body"] = _json.loads(request.content)
        return httpx.Response(200, json={"grant_valid": True, "peer_dept_id": "hr", "spans": []})

    # An active trace resolves to this traceparent; MeshTransport must forward it on /federate.
    monkeypatch.setattr(federation, "current_traceparent", lambda: "00-abc-def-01")
    await _fetch(_mesh_with(handler))
    assert seen["body"]["traceparent"] == "00-abc-def-01"


async def test_mesh_omits_traceparent_when_no_span(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen["body"] = _json.loads(request.content)
        return httpx.Response(200, json={"grant_valid": True, "peer_dept_id": "hr", "spans": []})

    monkeypatch.setattr(federation, "current_traceparent", lambda: None)
    await _fetch(_mesh_with(handler))
    assert "traceparent" not in seen["body"]  # nothing to propagate, nothing sent
