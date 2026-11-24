"""The auto-reindexer is leader-gated so N workers don't all reindex the same KB at once.

The auto-reindex loop runs in every uvicorn worker. Without a gate, each worker would reindex
the same knowledge base simultaneously: concurrent graph-on writes deadlock in Postgres and
every event loop goes CPU-bound, starving live traffic. Exactly one worker must run a pass.
Single-process deployments (in-memory coordinator) must be unaffected.
"""

from __future__ import annotations

import asyncio

import pytest
from svc_control_plane import pipeline, reindex
from svc_indexing.service import META_PIPELINE_VERSION, pipeline_fingerprint

from rag_common import get_settings
from rag_common.coordination import InMemoryCoordinator


class _Doc:
    def __init__(self, doc_id: str, version: str | None) -> None:
        self.doc_id = doc_id
        self.metadata: dict[str, str] = {} if version is None else {META_PIPELINE_VERSION: version}


class _Catalog:
    def __init__(self, docs: list[_Doc]) -> None:
        self._docs = docs

    async def list_documents(self, kb_id: str) -> list[_Doc]:
        return self._docs


class _Coordinator:
    """A coordinator whose leadership verdict is fixed (stands in for a shared Redis lease)."""

    def __init__(self, *, leader: bool) -> None:
        self._leader = leader

    def acquire_leader(self, name: str, ttl_s: int, token: str) -> bool:
        return self._leader


def test_single_process_is_always_the_leader(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no Redis (the default), the in-memory coordinator always grants the lease, so the
    gate is a no-op and the single-worker deployment behaves exactly as before."""
    monkeypatch.setattr(reindex, "get_coordinator", InMemoryCoordinator)
    assert reindex._reindex_leader() is True


@pytest.mark.asyncio
async def test_a_non_leader_worker_never_reindexes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The replica that does not hold the lease must not touch the corpus on a tick."""
    seen: list[str] = []

    async def _reindex_document(kb_id: str, doc_id: str, *, actor: str) -> None:
        seen.append(doc_id)

    monkeypatch.setattr(reindex, "auto_reindex_kbs", lambda: ["kb"])
    monkeypatch.setattr(reindex, "get_catalog", lambda: _Catalog([_Doc("old", "deadbeef")]))
    monkeypatch.setattr(pipeline, "reindex_document", _reindex_document)
    monkeypatch.setattr(reindex, "get_coordinator", lambda: _Coordinator(leader=False))

    await reindex._tick(asyncio.Event())
    assert seen == [], "a non-leader worker reindexed documents"


@pytest.mark.asyncio
async def test_the_leader_worker_reindexes_stale_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    """The replica that holds the lease runs the pass and rebuilds only the stale documents."""
    current = pipeline_fingerprint(get_settings())
    docs = [_Doc("fresh", current), _Doc("old", "deadbeef")]
    seen: list[str] = []

    async def _reindex_document(kb_id: str, doc_id: str, *, actor: str) -> None:
        seen.append(doc_id)

    monkeypatch.setattr(reindex, "auto_reindex_kbs", lambda: ["kb"])
    monkeypatch.setattr(reindex, "get_catalog", lambda: _Catalog(docs))
    monkeypatch.setattr(pipeline, "reindex_document", _reindex_document)
    monkeypatch.setattr(reindex, "get_coordinator", lambda: _Coordinator(leader=True))

    await reindex._tick(asyncio.Event())
    assert seen == ["old"], "the leader did not reindex exactly the stale document"


@pytest.mark.asyncio
async def test_reindex_kb_yields_at_the_next_document_when_the_heartbeat_drops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long pass must stop promptly once leadership is lost — the heartbeat is checked before
    each document, so losing the lease mid-pass stops it at the next boundary, never mid-write."""
    docs = [_Doc(f"d{i}", "old") for i in range(5)]
    monkeypatch.setattr(reindex, "get_catalog", lambda: _Catalog(docs))
    seen: list[str] = []

    async def _reindex_document(kb_id: str, doc_id: str, *, actor: str) -> None:
        seen.append(doc_id)

    monkeypatch.setattr(pipeline, "reindex_document", _reindex_document)

    beats = {"n": 0}

    def _heartbeat() -> bool:
        beats["n"] += 1
        return beats["n"] <= 2  # leader for the first two documents, then lost

    result = await reindex.reindex_kb("kb", actor="t", stale_only=True, limit=50, heartbeat=_heartbeat)
    assert seen == ["d0", "d1"], "the pass did not yield at the next document after losing the lease"
    assert result.reindexed == 2
