"""Async bulk-ingestion job runtime: create → add → run → progress, with resume + retry."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any

import pytest
from svc_control_plane import ingest_jobs, pipeline
from svc_control_plane.ingest_jobs import IngestJob, InMemoryIngestJobStore, run_job

from rag_common.config import get_settings


class _NoopObjs:
    async def put(self, *a: object, **k: object) -> str:
        return "obj://x"

    async def delete(self, *a: object, **k: object) -> None:
        return None


def _noop_objs() -> _NoopObjs:
    return _NoopObjs()


@pytest.fixture(autouse=True)
def _clean(inmemory: None) -> None:
    get_settings.cache_clear()


def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


async def _staged_job(store: InMemoryIngestJobStore, kb: str, n: int) -> IngestJob:
    job = await ingest_jobs.create_job(store, kb, actor="u")
    for i in range(n):
        await ingest_jobs.add_document(
            store, job.job_id, doc_id=f"doc{i}", content_b64=_b64(f"body {i}"), mime="text/plain", strategy="clause"
        )
    staged = await store.get(job.job_id)
    assert staged is not None
    return staged


async def test_create_add_start_flow() -> None:
    store = InMemoryIngestJobStore()
    job = await _staged_job(store, "kb", 3)
    assert job.status == "building" and job.total == 3
    started = await ingest_jobs.start_job(store, job.job_id)
    assert started.status == "queued"


async def test_worker_processes_all_items(monkeypatch: pytest.MonkeyPatch) -> None:
    processed: list[str] = []

    async def _fake_ingest(doc: Any, *, actor: str, strategy: Any, **_: Any) -> Any:
        processed.append(doc.doc_id)
        return SimpleNamespace(outcome="created")

    monkeypatch.setattr(pipeline, "ingest_document", _fake_ingest)
    monkeypatch.setattr(ingest_jobs, "get_object_store", _noop_objs)

    store = InMemoryIngestJobStore()
    job = await _staged_job(store, "kb", 4)
    await ingest_jobs.start_job(store, job.job_id)
    claimed = await store.claim_next_queued()
    assert claimed is not None
    await run_job(store, claimed)

    final = await store.get(job.job_id)
    assert final is not None and final.status == "completed" and final.done == 4 and final.failed == 0
    assert sorted(processed) == ["doc0", "doc1", "doc2", "doc3"]


async def test_failed_items_are_recorded_and_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _flaky(doc: Any, *, actor: str, strategy: Any, **_: Any) -> Any:
        if doc.doc_id == "doc1":
            raise RuntimeError("boom on doc1")
        return SimpleNamespace(outcome="created")

    monkeypatch.setattr(pipeline, "ingest_document", _flaky)
    monkeypatch.setattr(ingest_jobs, "get_object_store", _noop_objs)

    store = InMemoryIngestJobStore()
    job = await _staged_job(store, "kb", 3)
    await ingest_jobs.start_job(store, job.job_id)
    claimed = await store.claim_next_queued()
    assert claimed is not None
    await run_job(store, claimed)

    view = await ingest_jobs.job_view(store, job.job_id)
    assert view.job.done == 2 and view.job.failed == 1
    assert view.failures and view.failures[0]["doc_id"] == "doc1"

    async def _ok(doc: Any, *, actor: str, strategy: Any, **_: Any) -> Any:
        return SimpleNamespace(outcome="created")

    monkeypatch.setattr(pipeline, "ingest_document", _ok)
    assert await store.retry_failed(job.job_id) == 1
    claimed2 = await store.claim_next_queued()
    assert claimed2 is not None
    await run_job(store, claimed2)
    final = await store.get(job.job_id)
    assert final is not None and final.failed == 0


async def test_mixed_batch_ends_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch with some successes AND some failures ends 'partial' — NOT 'completed' — so a
    caller reading job.status alone is not misled into thinking the whole batch ingested."""

    async def _flaky(doc: Any, *, actor: str, strategy: Any, **_: Any) -> Any:
        if doc.doc_id == "doc1":
            raise RuntimeError("boom on doc1")
        return SimpleNamespace(outcome="created")

    monkeypatch.setattr(pipeline, "ingest_document", _flaky)
    monkeypatch.setattr(ingest_jobs, "get_object_store", _noop_objs)

    store = InMemoryIngestJobStore()
    job = await _staged_job(store, "kb", 3)
    await ingest_jobs.start_job(store, job.job_id)
    claimed = await store.claim_next_queued()
    assert claimed is not None
    await run_job(store, claimed)

    final = await store.get(job.job_id)
    assert final is not None
    assert final.status == "partial" and final.done == 2 and final.failed == 1


async def test_all_items_fail_ends_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every item failing ends 'failed' (nothing ingested), distinct from 'partial'."""

    async def _boom(doc: Any, *, actor: str, strategy: Any, **_: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "ingest_document", _boom)
    monkeypatch.setattr(ingest_jobs, "get_object_store", _noop_objs)

    store = InMemoryIngestJobStore()
    job = await _staged_job(store, "kb", 2)
    await ingest_jobs.start_job(store, job.job_id)
    claimed = await store.claim_next_queued()
    assert claimed is not None
    await run_job(store, claimed)

    final = await store.get(job.job_id)
    assert final is not None and final.status == "failed" and final.done == 0 and final.failed == 2


async def test_requeue_running_recovers_after_restart() -> None:
    store = InMemoryIngestJobStore()
    job = await _staged_job(store, "kb", 1)
    await ingest_jobs.start_job(store, job.job_id)
    await store.claim_next_queued()  # -> running
    running = await store.get(job.job_id)
    assert running is not None and running.status == "running"
    assert await store.requeue_running() == 1  # simulate restart recovery
    requeued = await store.get(job.job_id)
    assert requeued is not None and requeued.status == "queued"


async def test_add_document_stream_stages_raw_bytes_without_base64(inmemory: None) -> None:
    """Option B: the streaming add-item path puts raw binary straight to the job's staging key
    (no base64) and the existing by-reference worker picks it up — total bumps, bytes match."""
    import io

    from rag_clients.factory import get_object_store

    store = InMemoryIngestJobStore()
    job = await ingest_jobs.create_job(store, "acme__hr", actor="u")
    data = b"%PDF raw binary bytes, never base64-encoded"
    await ingest_jobs.add_document_stream(
        store,
        job.job_id,
        doc_id="big.pdf",
        stream=io.BytesIO(data),
        length=len(data),
        mime="application/pdf",
        strategy="clause",
    )
    staged = await store.get(job.job_id)
    assert staged is not None and staged.total == 1
    key = ingest_jobs._staging_key(job.job_id, "big.pdf")
    assert await get_object_store().get("acme__hr", key) == data  # staged verbatim, by reference
