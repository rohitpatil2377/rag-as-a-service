"""Federated scatter-gather is CONCURRENT and degrades gracefully under a bad peer.

answer_federated fans out to every granted peer at once (asyncio.gather), so a federated
answer's latency is the SLOWEST peer, not the SUM — and one slow or unreachable peer cannot
serialize-block the peers behind it, nor abort the whole answer. These are the resilience
properties that a plain sequential ``for peer in ...: await fetch`` loop did NOT provide.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from svc_control_plane import pipeline, wiring
from svc_control_plane.federation import PeerFetch
from svc_control_plane.grants import InMemoryGrantStore

from rag_schemas import Grant

_PEER_DELAY = 0.3  # each peer's transport sleeps this long


def _attrs(span: Any) -> Mapping[str, Any]:
    """Narrow an OTel span's ``attributes`` (typed ``Mapping | None``) for indexing in assertions."""
    assert span.attributes is not None
    return cast("Mapping[str, Any]", span.attributes)


def _grant(issuer: str, subject: str, gid: str) -> Grant:
    now = datetime(2026, 8, 3, tzinfo=UTC)
    return Grant(
        grant_id=gid,
        issuer_dept_id=issuer,
        subject_dept_id=subject,
        topics=["payroll"],
        kb_ids=[f"{issuer}__kb"],
        max_results_per_query=5,
        not_before=now - timedelta(days=1),
        expires_at=now + timedelta(days=180),
        signed_token="",
        created_at=now,
    )


class _SleepTransport:
    """A transport that takes _PEER_DELAY to answer — stands in for a real remote peer."""

    def __init__(self, *, reachable: bool = True, reason: str = "ok") -> None:
        self._reachable = reachable
        self._reason = reason

    async def fetch(self, *, caller_dept: str, peer_dept_id: str, grants: Any, query: str, top_k: int) -> PeerFetch:
        import asyncio

        await asyncio.sleep(_PEER_DELAY)
        # Contributes no spans (keeps the answer on the cheap abstain path); the point is the
        # timing + that reachability is honoured, not the retrieved content.
        return PeerFetch([], [], reachable=self._reachable, reason=self._reason)


class _SleepRegistry:
    """Resolves every peer to a sleeping transport; peers in ``downs`` are unreachable."""

    def __init__(self, downs: set[str]) -> None:
        self._downs = downs

    def transport_for(self, peer_dept_id: str) -> _SleepTransport:
        if peer_dept_id in self._downs:
            return _SleepTransport(reachable=False, reason="unreachable")
        return _SleepTransport()


async def test_fanout_is_concurrent_and_survives_a_down_peer(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    caller = "finance"
    peers = ["hr", "legal", "ops"]  # 3 granted peers; "ops" is DOWN

    store = InMemoryGrantStore()
    for p in peers:
        await store.create(_grant(issuer=p, subject=caller, gid=f"g-{p}"))

    monkeypatch.setattr(wiring, "grant_store", lambda: store)
    monkeypatch.setattr(wiring, "federation_registry", lambda: _SleepRegistry(downs={"ops"}))

    t0 = time.perf_counter()
    res = await pipeline.answer_federated(caller, [], peers, "quarterly payroll figures", top_k=5, acl=[], actor="u")
    elapsed = time.perf_counter() - t0

    # Lower bound: the peer sleeps actually ran (the fan-out executed past the guardrail).
    assert elapsed >= _PEER_DELAY * 0.8, f"fan-out did not run (elapsed={elapsed:.3f}s)"
    # Upper bound: 3 peers concurrently ~= one delay, NOT three. Sequential would be >=0.9s.
    assert elapsed < _PEER_DELAY * 2, f"fan-out was serialized, not concurrent (elapsed={elapsed:.3f}s)"

    # Graceful degradation: the DOWN peer neither crashed nor aborted the answer.
    assert res is not None


async def test_federation_fanout_emits_nested_trace_spans(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The distributed-tracing claim is REAL: the fan-out emits one scatter-gather span with a
    child span per department reached, carrying decision/reachability attributes — the trace a
    jury can open in Tempo/Jaeger and watch cross the department boundary."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Route pipeline's tracer to our provider (get_tracer is bound in the pipeline module).
    monkeypatch.setattr(pipeline, "get_tracer", lambda _name: provider.get_tracer("test"))

    caller = "finance"
    peers = ["hr", "legal", "ops"]  # "ops" is DOWN
    store = InMemoryGrantStore()
    for p in peers:
        await store.create(_grant(issuer=p, subject=caller, gid=f"g-{p}"))
    monkeypatch.setattr(wiring, "grant_store", lambda: store)
    monkeypatch.setattr(wiring, "federation_registry", lambda: _SleepRegistry(downs={"ops"}))

    await pipeline.answer_federated(caller, [], peers, "payroll", top_k=5, acl=[], actor="u")

    spans = exporter.get_finished_spans()
    parents = [s for s in spans if s.name == "federation.scatter_gather"]
    children = [s for s in spans if s.name == "federation.peer"]

    assert len(parents) == 1, f"expected one scatter-gather span, got {[s.name for s in spans]}"
    parent = parents[0]
    assert _attrs(parent)["federation.peers_requested"] == 3
    assert _attrs(parent)["federation.reached"] == 2  # hr, legal
    assert _attrs(parent)["federation.unreachable"] == 1  # ops

    # One child span per peer, each NESTED under the scatter-gather parent (real trace tree).
    assert len(children) == 3
    assert all(c.parent is not None and c.parent.span_id == parent.context.span_id for c in children)
    depts = {_attrs(c)["federation.peer_dept"] for c in children}
    assert depts == {"hr", "legal", "ops"}
    down = next(c for c in children if _attrs(c)["federation.peer_dept"] == "ops")
    assert _attrs(down)["federation.reachable"] is False
