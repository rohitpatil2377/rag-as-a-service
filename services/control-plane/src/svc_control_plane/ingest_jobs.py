"""Async bulk-ingestion pipeline — ingest 1000+ documents without blocking the request.

The single-document ingest endpoint runs the whole parse→mask→chunk→embed→index pipeline
inline, which is fine for a few files but impossible for thousands (each call blocks, and the
CPU/GPU embedder is the bottleneck). This module adds a persisted job queue drained by an
in-process worker (same pattern as the eval/sync scheduler):

* the caller creates a JOB, uploads each file to it (the bytes are STAGED to the object store,
  so the job DB stays lean and the batch survives a restart), then STARTS the job;
* an in-process worker claims queued jobs and ingests their pending items with bounded
  concurrency, updating a live progress counter (done / failed / total);
* jobs and per-item state persist, so a restart RESUMES where it left off (done items skip,
  pending items re-run); failed items carry their reason and can be retried.

Multi-replica note: a single worker drains the queue; horizontal scale would gate the loop on
the coordination leader lock (as the scheduler does) — a documented follow-up.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import uuid
from datetime import UTC, datetime
from typing import IO, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from rag_clients.factory import get_object_store
from rag_common import get_coordinator, get_logger, get_settings
from rag_common.errors import NotFoundError, ValidationError
from rag_schemas import DocumentEnvelope
from rag_schemas.common import ChunkStrategy, SourceKind

_log = get_logger(__name__)
_IDENTITY_SCHEMA = "rag_identity"

# Identifies THIS replica when contending for the bulk-ingest leader lease (uuid per process).
_WORKER_TOKEN = uuid.uuid4().hex


def _is_leader() -> bool:
    """True when this replica may DRAIN the ingest queue. Single-process (no RAG_REDIS_URL) is
    always leader; when scaled, a Redis lease ensures EXACTLY ONE replica drains — so a bulk item
    is never ingested twice across pods. Mirrors the scheduler's leader gate (continuous_eval)."""
    settings = get_settings()
    return get_coordinator().acquire_leader("bulk_ingest", settings.leader_lock_ttl_seconds, _WORKER_TOKEN)


JobStatus = Literal["building", "queued", "running", "completed", "partial", "failed", "cancelled"]
ItemStatus = Literal["pending", "done", "failed"]


class IngestJob(BaseModel):
    job_id: str
    kb_id: str
    actor: str
    status: JobStatus
    total: int = 0
    done: int = 0
    failed: int = 0
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    # Captured at job creation so the async runner (which may resume after a restart, without the
    # request principal) can enforce the PII classification floor: the creator's clearance caps how
    # secret a doc it may auto-raise, and the resolved policy (off|raise|block) for this batch.
    clearance: int = 0
    pii_policy: str = "off"
    # The manager-chosen data classification (0=Public..3=Restricted) applied to every doc in this
    # batch. Captured here (like clearance/pii_policy) so the async worker stamps it on each doc —
    # already CAPPED at the creator's clearance by the endpoint, mirroring the direct /ingest path.
    classification: int = 0


class IngestItem(BaseModel):
    job_id: str
    seq: int
    doc_id: str
    object_key: str  # staging key in the platform object store
    mime: str
    strategy: str
    status: ItemStatus = "pending"
    error: str | None = None


class JobView(BaseModel):
    """Job + a small sample of failure reasons for the console."""

    job: IngestJob
    failures: list[dict[str, str]] = Field(default_factory=list)


def _staging_key(job_id: str, doc_id: str) -> str:
    return f"staging/{job_id}/{doc_id}"


# --- store (global tables; in-memory twin for CI) ----------------------------
@runtime_checkable
class IngestJobStore(Protocol):
    async def ensure(self) -> None: ...
    async def create(self, job: IngestJob) -> None: ...
    async def get(self, job_id: str) -> IngestJob | None: ...
    async def list_for_kb(self, kb_id: str, limit: int) -> list[IngestJob]: ...
    async def set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None: ...
    async def bump(self, job_id: str, *, done: int = 0, failed: int = 0, total: int = 0) -> None: ...
    async def add_item(self, item: IngestItem) -> None: ...
    async def pending_items(self, job_id: str) -> list[IngestItem]: ...
    async def set_item(self, job_id: str, seq: int, status: ItemStatus, error: str | None) -> None: ...
    async def failures(self, job_id: str, limit: int) -> list[dict[str, str]]: ...
    async def claim_next_queued(self) -> IngestJob | None: ...
    async def requeue_running(self) -> int: ...
    async def retry_failed(self, job_id: str) -> int: ...


class InMemoryIngestJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, IngestJob] = {}
        self._items: dict[str, list[IngestItem]] = {}

    async def ensure(self) -> None:
        return None

    async def create(self, job: IngestJob) -> None:
        self._jobs[job.job_id] = job
        self._items[job.job_id] = []

    async def get(self, job_id: str) -> IngestJob | None:
        return self._jobs.get(job_id)

    async def list_for_kb(self, kb_id: str, limit: int) -> list[IngestJob]:
        jobs = [j for j in self._jobs.values() if j.kb_id == kb_id]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)[:limit]

    async def set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        j = self._jobs.get(job_id)
        if j:
            self._jobs[job_id] = j.model_copy(
                update={"status": status, "error": error, "updated_at": datetime.now(UTC)}
            )

    async def bump(self, job_id: str, *, done: int = 0, failed: int = 0, total: int = 0) -> None:
        j = self._jobs.get(job_id)
        if j:
            self._jobs[job_id] = j.model_copy(
                update={
                    "done": j.done + done,
                    "failed": j.failed + failed,
                    "total": j.total + total,
                    "updated_at": datetime.now(UTC),
                }
            )

    async def add_item(self, item: IngestItem) -> None:
        self._items.setdefault(item.job_id, []).append(item)

    async def pending_items(self, job_id: str) -> list[IngestItem]:
        return [i for i in self._items.get(job_id, []) if i.status == "pending"]

    async def set_item(self, job_id: str, seq: int, status: ItemStatus, error: str | None) -> None:
        for i, it in enumerate(self._items.get(job_id, [])):
            if it.seq == seq:
                self._items[job_id][i] = it.model_copy(update={"status": status, "error": error})

    async def failures(self, job_id: str, limit: int) -> list[dict[str, str]]:
        failed = [i for i in self._items.get(job_id, []) if i.status == "failed"]
        return [{"doc_id": i.doc_id, "error": i.error or ""} for i in failed][:limit]

    async def claim_next_queued(self) -> IngestJob | None:
        for j in sorted(self._jobs.values(), key=lambda x: x.created_at):
            if j.status == "queued":
                await self.set_status(j.job_id, "running")
                return self._jobs[j.job_id]
        return None

    async def requeue_running(self) -> int:
        n = 0
        for j in list(self._jobs.values()):
            if j.status == "running":
                await self.set_status(j.job_id, "queued")
                n += 1
        return n

    async def retry_failed(self, job_id: str) -> int:
        items = self._items.get(job_id, [])
        n = 0
        for i, it in enumerate(items):
            if it.status == "failed":
                items[i] = it.model_copy(update={"status": "pending", "error": None})
                n += 1
        if n:
            j = self._jobs[job_id]
            self._jobs[job_id] = j.model_copy(update={"failed": max(0, j.failed - n), "status": "queued"})
        return n


_JOB_COLS = (
    "job_id, kb_id, actor, status, total, done, failed, created_at, updated_at, error, "
    "clearance, pii_policy, classification"
)
_DDL = (
    f'CREATE SCHEMA IF NOT EXISTS "{_IDENTITY_SCHEMA}"',
    f'CREATE TABLE IF NOT EXISTS "{_IDENTITY_SCHEMA}".ingest_jobs ('
    "job_id TEXT PRIMARY KEY, kb_id TEXT NOT NULL, actor TEXT NOT NULL, status TEXT NOT NULL, "
    "total INT NOT NULL DEFAULT 0, done INT NOT NULL DEFAULT 0, failed INT NOT NULL DEFAULT 0, "
    "created_at TIMESTAMPTZ NOT NULL, updated_at TIMESTAMPTZ NOT NULL, error TEXT, "
    "clearance INT NOT NULL DEFAULT 0, pii_policy TEXT NOT NULL DEFAULT 'off', "
    "classification INT NOT NULL DEFAULT 0)",
    # Migration-safe: add the PII-floor + classification columns to deployments whose table predates
    # them (the CREATE ... IF NOT EXISTS above no-ops on an existing table, mirroring
    # content_hash/parent_grant_id).
    f'ALTER TABLE "{_IDENTITY_SCHEMA}".ingest_jobs ADD COLUMN IF NOT EXISTS clearance INT NOT NULL DEFAULT 0',
    f"ALTER TABLE \"{_IDENTITY_SCHEMA}\".ingest_jobs ADD COLUMN IF NOT EXISTS pii_policy TEXT NOT NULL DEFAULT 'off'",
    f'ALTER TABLE "{_IDENTITY_SCHEMA}".ingest_jobs ADD COLUMN IF NOT EXISTS classification INT NOT NULL DEFAULT 0',
    f'CREATE TABLE IF NOT EXISTS "{_IDENTITY_SCHEMA}".ingest_job_items ('
    "job_id TEXT NOT NULL, seq INT NOT NULL, doc_id TEXT NOT NULL, object_key TEXT NOT NULL, "
    "mime TEXT NOT NULL, strategy TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', error TEXT, "
    "PRIMARY KEY (job_id, seq))",
)


class PostgresIngestJobStore:
    def __init__(self, engine: AsyncEngine | None = None) -> None:
        self._engine = engine or create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)

    async def ensure(self) -> None:
        async with self._engine.begin() as c:
            for stmt in _DDL:
                await c.execute(text(stmt))

    async def create(self, job: IngestJob) -> None:
        await self.ensure()
        sql = (
            f'INSERT INTO "{_IDENTITY_SCHEMA}".ingest_jobs ({_JOB_COLS}) VALUES '
            "(:job_id, :kb_id, :actor, :status, :total, :done, :failed, :created_at, :updated_at, :error, "
            ":clearance, :pii_policy, :classification)"
        )
        async with self._engine.begin() as c:
            await c.execute(text(sql), job.model_dump())

    async def get(self, job_id: str) -> IngestJob | None:
        await self.ensure()
        async with self._engine.connect() as c:
            row = (
                (
                    await c.execute(
                        text(f'SELECT {_JOB_COLS} FROM "{_IDENTITY_SCHEMA}".ingest_jobs WHERE job_id = :j'),
                        {"j": job_id},
                    )
                )
                .mappings()
                .first()
            )
        return IngestJob(**dict(row)) if row else None

    async def list_for_kb(self, kb_id: str, limit: int) -> list[IngestJob]:
        await self.ensure()
        sql = (
            f'SELECT {_JOB_COLS} FROM "{_IDENTITY_SCHEMA}".ingest_jobs '
            "WHERE kb_id = :k ORDER BY created_at DESC LIMIT :l"
        )
        async with self._engine.connect() as c:
            rows = (await c.execute(text(sql), {"k": kb_id, "l": limit})).mappings().all()
        return [IngestJob(**dict(r)) for r in rows]

    async def set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        await self.ensure()
        async with self._engine.begin() as c:
            await c.execute(
                text(f'UPDATE "{_IDENTITY_SCHEMA}".ingest_jobs SET status=:s, error=:e, updated_at=:u WHERE job_id=:j'),
                {"s": status, "e": error, "u": datetime.now(UTC), "j": job_id},
            )

    async def bump(self, job_id: str, *, done: int = 0, failed: int = 0, total: int = 0) -> None:
        await self.ensure()
        async with self._engine.begin() as c:
            await c.execute(
                text(
                    f'UPDATE "{_IDENTITY_SCHEMA}".ingest_jobs SET done=done+:d, failed=failed+:f, '
                    "total=total+:t, updated_at=:u WHERE job_id=:j"
                ),
                {"d": done, "f": failed, "t": total, "u": datetime.now(UTC), "j": job_id},
            )

    async def add_item(self, item: IngestItem) -> None:
        await self.ensure()
        sql = (
            f'INSERT INTO "{_IDENTITY_SCHEMA}".ingest_job_items '
            "(job_id, seq, doc_id, object_key, mime, strategy, status, error) VALUES "
            "(:job_id, :seq, :doc_id, :object_key, :mime, :strategy, :status, :error)"
        )
        async with self._engine.begin() as c:
            await c.execute(text(sql), item.model_dump())

    async def pending_items(self, job_id: str) -> list[IngestItem]:
        await self.ensure()
        sql = (
            f"SELECT job_id, seq, doc_id, object_key, mime, strategy, status, error "
            f"FROM \"{_IDENTITY_SCHEMA}\".ingest_job_items WHERE job_id=:j AND status='pending' ORDER BY seq"
        )
        async with self._engine.connect() as c:
            rows = (await c.execute(text(sql), {"j": job_id})).mappings().all()
        return [IngestItem(**dict(r)) for r in rows]

    async def set_item(self, job_id: str, seq: int, status: ItemStatus, error: str | None) -> None:
        await self.ensure()
        async with self._engine.begin() as c:
            await c.execute(
                text(
                    f'UPDATE "{_IDENTITY_SCHEMA}".ingest_job_items SET status=:s, error=:e WHERE job_id=:j AND seq=:q'
                ),
                {"s": status, "e": error, "j": job_id, "q": seq},
            )

    async def failures(self, job_id: str, limit: int) -> list[dict[str, str]]:
        await self.ensure()
        sql = (
            f'SELECT doc_id, error FROM "{_IDENTITY_SCHEMA}".ingest_job_items '
            f"WHERE job_id=:j AND status='failed' ORDER BY seq LIMIT :l"
        )
        async with self._engine.connect() as c:
            rows = (await c.execute(text(sql), {"j": job_id, "l": limit})).mappings().all()
        return [{"doc_id": r["doc_id"], "error": r["error"] or ""} for r in rows]

    async def claim_next_queued(self) -> IngestJob | None:
        await self.ensure()
        # Atomic claim: flip the oldest queued job to running in one statement.
        sql = (
            f"UPDATE \"{_IDENTITY_SCHEMA}\".ingest_jobs SET status='running', updated_at=:u WHERE job_id = ("
            f"SELECT job_id FROM \"{_IDENTITY_SCHEMA}\".ingest_jobs WHERE status='queued' "
            "ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1) "
            f"RETURNING {_JOB_COLS}"
        )
        async with self._engine.begin() as c:
            row = (await c.execute(text(sql), {"u": datetime.now(UTC)})).mappings().first()
        return IngestJob(**dict(row)) if row else None

    async def requeue_running(self) -> int:
        await self.ensure()
        async with self._engine.begin() as c:
            res = await c.execute(
                text(f"UPDATE \"{_IDENTITY_SCHEMA}\".ingest_jobs SET status='queued' WHERE status='running'")
            )
        return res.rowcount or 0

    async def retry_failed(self, job_id: str) -> int:
        await self.ensure()
        async with self._engine.begin() as c:
            res = await c.execute(
                text(
                    f"UPDATE \"{_IDENTITY_SCHEMA}\".ingest_job_items SET status='pending', error=NULL "
                    "WHERE job_id=:j AND status='failed'"
                ),
                {"j": job_id},
            )
            n = res.rowcount or 0
            if n:
                await c.execute(
                    text(
                        f'UPDATE "{_IDENTITY_SCHEMA}".ingest_jobs SET failed=GREATEST(0, failed-:n), '
                        "status='queued', updated_at=:u WHERE job_id=:j"
                    ),
                    {"n": n, "u": datetime.now(UTC), "j": job_id},
                )
        return n


# --- operations --------------------------------------------------------------
async def create_job(
    store: IngestJobStore,
    kb_id: str,
    actor: str,
    *,
    clearance: int = 0,
    pii_policy: str = "off",
    classification: int = 0,
) -> IngestJob:
    now = datetime.now(UTC)
    # Fail-closed cap: never store a batch at a classification above the creator's clearance — the
    # same invariant the direct /ingest path enforces via can_ingest_at (you can't publish content
    # more secret than you're cleared to read). The endpoint validates the range; this is defence.
    capped = min(max(classification, 0), clearance)
    job = IngestJob(
        job_id=str(uuid.uuid4()),
        kb_id=kb_id,
        actor=actor,
        status="building",
        created_at=now,
        updated_at=now,
        clearance=clearance,
        pii_policy=pii_policy,
        classification=capped,
    )
    await store.create(job)
    return job


async def add_document(
    store: IngestJobStore, job_id: str, *, doc_id: str, content_b64: str, mime: str, strategy: str
) -> None:
    """Stage a file's bytes to the object store and append a pending item to the job."""
    job = await store.get(job_id)
    if job is None:
        raise NotFoundError("ingest job not found", details={"job_id": job_id})
    if job.status != "building":
        raise ValidationError("documents can only be added while the job is 'building'")
    try:
        raw = base64.b64decode(content_b64, validate=True)
    except Exception as exc:
        raise ValidationError("invalid base64 content", details={"doc_id": doc_id}) from exc
    key = _staging_key(job_id, doc_id)
    await get_object_store().put(job.kb_id, key, raw, mime)
    seq = job.total  # monotonic per job
    await store.add_item(
        IngestItem(job_id=job_id, seq=seq, doc_id=doc_id, object_key=key, mime=mime, strategy=strategy)
    )
    await store.bump(job_id, total=1)


async def add_document_stream(
    store: IngestJobStore, job_id: str, *, doc_id: str, stream: IO[bytes], length: int, mime: str, strategy: str
) -> None:
    """Stage a file into a building job by STREAMING raw binary straight to the object store — no
    base64, never buffering the whole file in memory. Same staged-then-processed-by-reference flow
    as :func:`add_document`; only the transport differs, so the worker and progress are unchanged.
    For large uploads (>~1 GB) where base64 would bloat the request ~1.37x and pin memory."""
    job = await store.get(job_id)
    if job is None:
        raise NotFoundError("ingest job not found", details={"job_id": job_id})
    if job.status != "building":
        raise ValidationError("documents can only be added while the job is 'building'")
    key = _staging_key(job_id, doc_id)
    await get_object_store().put_stream(job.kb_id, key, stream, length, mime)
    seq = job.total  # monotonic per job
    await store.add_item(
        IngestItem(job_id=job_id, seq=seq, doc_id=doc_id, object_key=key, mime=mime, strategy=strategy)
    )
    await store.bump(job_id, total=1)


async def start_job(store: IngestJobStore, job_id: str) -> IngestJob:
    job = await store.get(job_id)
    if job is None:
        raise NotFoundError("ingest job not found", details={"job_id": job_id})
    if job.total == 0:
        raise ValidationError("nothing to ingest — add documents before starting the job")
    await store.set_status(job_id, "queued")
    updated = await store.get(job_id)
    assert updated is not None
    return updated


async def job_view(store: IngestJobStore, job_id: str) -> JobView:
    job = await store.get(job_id)
    if job is None:
        raise NotFoundError("ingest job not found", details={"job_id": job_id})
    return JobView(job=job, failures=await store.failures(job_id, 10))


async def _process_item(
    kb_id: str, item: IngestItem, actor: str, *, clearance: int = 0, pii_policy: str = "off", classification: int = 0
) -> None:
    """Ingest one staged document through the normal pipeline, then drop the staging copy."""
    from svc_control_plane import pipeline  # noqa: PLC0415 — lazy import avoids a wiring↔pipeline cycle

    doc = DocumentEnvelope(
        tenant_id=kb_id,
        doc_id=item.doc_id,
        source_id="bulk-ingest",
        source_kind=SourceKind.OBJECT_STORE,
        mime=item.mime,
        uri=item.object_key,
        # The manager-chosen batch classification (already capped at the creator's clearance when the
        # job was created) — stamped on the doc exactly like the direct /ingest path sets it.
        classification=classification,
        ingested_at=datetime.now(UTC),
    )
    outcome = await pipeline.ingest_document(
        doc, actor=actor, strategy=ChunkStrategy(item.strategy), clearance=clearance, pii_policy=pii_policy
    )
    with contextlib.suppress(Exception):  # best-effort cleanup of the staging copy
        await get_object_store().delete(kb_id, item.object_key)
    # A PII-floor BLOCK isn't an exception (nothing was indexed by design) — surface it as a per-item
    # failure so the batch UI shows which docs bounced and why, instead of silently counting them done.
    if outcome.outcome == "blocked_pii":
        raise ValueError(outcome.reason or "blocked: detected PII requires a higher classification")


async def run_job(store: IngestJobStore, job: IngestJob) -> None:
    """Process all pending items of a job with bounded concurrency; update live progress."""
    sem = asyncio.Semaphore(get_settings().bulk_ingest_concurrency)
    items = await store.pending_items(job.job_id)

    async def _one(it: IngestItem) -> None:
        async with sem:
            if (await store.get(job.job_id) or job).status == "cancelled":
                return
            try:
                await _process_item(
                    job.kb_id,
                    it,
                    job.actor,
                    clearance=job.clearance,
                    pii_policy=job.pii_policy,
                    classification=job.classification,
                )
                await store.set_item(job.job_id, it.seq, "done", None)
                await store.bump(job.job_id, done=1)
            except Exception as exc:  # one bad document never aborts the batch
                reason = str(exc)[:200] or type(exc).__name__
                await store.set_item(job.job_id, it.seq, "failed", reason)
                await store.bump(job.job_id, failed=1)
                _log.warning("bulk_ingest_item_failed", job_id=job.job_id, doc_id=it.doc_id, error=reason)

    await asyncio.gather(*(_one(it) for it in items))
    final = await store.get(job.job_id)
    if final and final.status != "cancelled":
        # Three-way terminal status so a caller reading job.status alone sees partial failure:
        #   failed    — every item failed (nothing ingested)
        #   partial   — some ingested, some failed (the batch is NOT clean; retry the failures)
        #   completed — every item ingested, zero failures
        if final.failed and not final.done:
            terminal: JobStatus = "failed"
        elif final.failed:
            terminal = "partial"
        else:
            terminal = "completed"
        await store.set_status(job.job_id, terminal)
    _log.info(
        "bulk_ingest_job_done", job_id=job.job_id, done=final.done if final else 0, failed=final.failed if final else 0
    )


async def worker_loop(stop: asyncio.Event, store: IngestJobStore) -> None:
    """In-process worker: drain the queue, one job at a time, until stopped.

    LEADER-GATED (multi-replica): only the replica holding the ``bulk_ingest`` lease drains, so
    horizontal scale never ingests an item twice. On acquiring leadership the worker first
    requeues jobs the PREVIOUS leader left RUNNING (its crash orphaned them), then drains; a
    non-leader replica idles, ready to take over. Single-process is always leader — unchanged."""
    poll = get_settings().bulk_ingest_poll_seconds
    _log.info("bulk_ingest_worker_started", concurrency=get_settings().bulk_ingest_concurrency)
    recovered_as_leader = False
    while not stop.is_set():
        if not _is_leader():
            # Not (yet) leader: idle. Reset the recovery flag so we recover again if we later win.
            recovered_as_leader = False
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=poll)
            continue
        if not recovered_as_leader:
            # First tick of a leadership term: requeue a crashed prior leader's RUNNING jobs so
            # they resume here. Done ONCE per term, so it never disturbs our own in-flight job.
            with contextlib.suppress(Exception):
                n = await store.requeue_running()
                if n:
                    _log.info("bulk_ingest_recovered", jobs=n)
            recovered_as_leader = True
        job: IngestJob | None = None
        with contextlib.suppress(Exception):
            job = await store.claim_next_queued()
        if job is None:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=poll)
            continue
        with contextlib.suppress(Exception):
            await run_job(store, job)
    _log.info("bulk_ingest_worker_stopped")
