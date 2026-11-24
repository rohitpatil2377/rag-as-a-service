"""Retention worker loop — sweeps KBs and erases overdue documents (impure half)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from svc_control_plane import erasure_worker
from svc_control_plane.compliance import ErasureCertificate

from rag_common import get_settings


class _Tenant:
    def __init__(self, org_id: str) -> None:
        self.org_id = org_id


class _Kb:
    def __init__(self, kb_id: str) -> None:
        self.kb_id = kb_id


class _Doc:
    def __init__(self, doc_id: str, ingested_at: datetime) -> None:
        self.doc_id = doc_id
        self.ingested_at = ingested_at


class _Reg:
    async def list_all(self) -> list[_Tenant]:
        return [_Tenant("bank")]


class _KbStore:
    async def list(self, _org: str) -> list[_Kb]:
        return [_Kb("bank__deals")]


class _Catalog:
    def __init__(self, docs: list[_Doc]) -> None:
        self._docs = docs

    async def list_documents(self, _kb: str) -> list[_Doc]:
        return self._docs


async def test_sweep_erases_only_overdue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ERASURE_RETENTION_DAYS", "365")
    get_settings.cache_clear()
    now = datetime.now(UTC)
    docs = [_Doc("old", now - timedelta(days=400)), _Doc("fresh", now - timedelta(days=10))]

    from svc_control_plane import wiring

    from rag_clients import factory

    monkeypatch.setattr(wiring, "tenant_registry", _Reg)
    monkeypatch.setattr(wiring, "kb_store", _KbStore)
    monkeypatch.setattr(factory, "get_catalog", lambda: _Catalog(docs))

    calls: list[tuple[str, list[str]]] = []

    async def fake_erase(kb_id: str, doc_ids: list[str]) -> ErasureCertificate:
        calls.append((kb_id, doc_ids))
        return ErasureCertificate(
            kb_id=kb_id,
            requested_by="retention-scheduler",
            reason="retention_expiry",
            subject=None,
            documents_erased=doc_ids,
            chunks_purged=0,
            conversations_purged=0,
            certificate_hash="h" * 64,
            erased_at=now,
        )

    monkeypatch.setattr(erasure_worker, "_erase", fake_erase)
    erased = await erasure_worker._sweep_once(now)
    assert erased == 1
    assert calls == [("bank__deals", ["old"])]  # only the over-retention doc


async def test_worker_loop_honors_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    swept: list[datetime] = []

    async def fake_sweep(now: datetime) -> int:
        swept.append(now)
        return 0

    monkeypatch.setattr(erasure_worker, "_sweep_once", fake_sweep)
    stop = asyncio.Event()
    task = asyncio.create_task(erasure_worker.worker_loop(stop, interval_seconds=0.01))
    await asyncio.sleep(0.03)
    stop.set()
    await asyncio.wait_for(task, timeout=1)
    assert len(swept) >= 1  # swept at least once, then stopped cleanly
