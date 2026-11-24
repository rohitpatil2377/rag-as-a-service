"""Bulk-ingest worker is LEADER-GATED — multi-replica never ingests an item twice.

Only the replica holding the ``bulk_ingest`` lease drains the queue; a non-leader idles. On
acquiring leadership the worker recovers a crashed prior leader's RUNNING jobs before draining.
Single-process (InMemoryCoordinator) is always leader, so the default path is unchanged.
"""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
from typing import Any

import pytest
from svc_control_plane import ingest_jobs, pipeline
from svc_control_plane.ingest_jobs import InMemoryIngestJobStore


class _NoopObjs:
    async def put(self, *a: object, **k: object) -> str:
        return "obj://x"

    async def delete(self, *a: object, **k: object) -> None:
        return None


def _noop_objs() -> _NoopObjs:
    return _NoopObjs()


async def _staged_started(store: InMemoryIngestJobStore, n: int = 2) -> str:
    job = await ingest_jobs.create_job(store, "kb", actor="u")
    for i in range(n):
        await ingest_jobs.add_document(
            store,
            job.job_id,
            doc_id=f"doc{i}",
            content_b64=base64.b64encode(b"body").decode(),
            mime="text/plain",
            strategy="clause",
        )
    await ingest_jobs.start_job(store, job.job_id)  # -> queued
    return job.job_id


async def test_a_non_leader_replica_does_not_drain(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingest_jobs, "_is_leader", lambda: False)
    store = InMemoryIngestJobStore()
    job_id = await _staged_started(store)

    stop = asyncio.Event()
    task = asyncio.create_task(ingest_jobs.worker_loop(stop, store))
    await asyncio.sleep(0.2)  # give the loop time to (not) act
    rec = await store.get(job_id)
    assert rec is not None and rec.status == "queued", "a non-leader must not drain the queue"
    stop.set()
    await asyncio.wait_for(task, timeout=5)


async def test_the_leader_drains_the_queue(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    processed: list[str] = []

    async def _fake_ingest(doc: Any, *, actor: str, strategy: Any, **_: Any) -> Any:
        processed.append(doc.doc_id)
        return SimpleNamespace(outcome="created")

    monkeypatch.setattr(pipeline, "ingest_document", _fake_ingest)
    monkeypatch.setattr(ingest_jobs, "get_object_store", _noop_objs)
    monkeypatch.setattr(ingest_jobs, "_is_leader", lambda: True)

    store = InMemoryIngestJobStore()
    job_id = await _staged_started(store, n=3)

    stop = asyncio.Event()
    task = asyncio.create_task(ingest_jobs.worker_loop(stop, store))
    for _ in range(200):
        rec = await store.get(job_id)
        if rec is not None and rec.status in ("completed", "partial", "failed"):
            break
        await asyncio.sleep(0.02)
    stop.set()
    await asyncio.wait_for(task, timeout=5)

    final = await store.get(job_id)
    assert final is not None and final.status == "completed" and final.done == 3
    assert sorted(processed) == ["doc0", "doc1", "doc2"]
