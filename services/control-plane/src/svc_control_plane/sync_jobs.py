"""Scheduled & delta connector sync (FR-A8).

The delta itself lives in the ingest pipeline (content-hash → skip unchanged). This module
adds the two triggers around it:
* **on-demand / webhook** — :func:`run_sync` re-pulls a KB's external folder; unchanged
  objects are skipped by the pipeline delta, so a re-sync only re-indexes what changed;
* **scheduled** — a KB persists its sync source + an interval; the platform scheduler
  (:func:`sync_tick`, driven from the same in-process loop as continuous eval) re-runs due
  KBs. Enrolment lives in one GLOBAL table (KBs can't be enumerated cross-tenant), mirroring
  the eval-schedule registry.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from rag_clients.external_creds import external_credentials
from rag_clients.factory import external_fetcher_for, get_catalog
from rag_common import assert_sovereign, get_logger, get_secrets_provider, get_settings
from rag_schemas import DocumentEnvelope, ExternalObjectSource
from rag_schemas.common import ChunkStrategy, SourceKind
from svc_control_plane import pipeline
from svc_control_plane.quota import check_document_cap

_log = get_logger(__name__)
_IDENTITY_SCHEMA = "rag_identity"
_SOURCE_KEY = "sync_source"

_EXT_MIME = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".html": "text/html",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


# --- persisted per-KB sync source (in tenant_config) -------------------------
class SyncSource(BaseModel):
    external_source: ExternalObjectSource
    prefix: str = ""
    mime: str = "application/octet-stream"
    strategy: str = "clause"
    limit: int = Field(default=200, ge=1, le=1000)


async def get_sync_source(kb_id: str) -> SyncSource | None:
    raw = (await get_catalog().get_tenant_config(kb_id)).get(_SOURCE_KEY)
    if not raw:
        return None
    try:
        return SyncSource.model_validate_json(raw)
    except ValueError:
        return None


async def set_sync_source(kb_id: str, source: SyncSource) -> None:
    await get_catalog().set_tenant_config(kb_id, _SOURCE_KEY, source.model_dump_json())


# --- the reusable runner (delta-aware; used by webhook + scheduler) ----------
class SyncResult(BaseModel):
    found: int = 0
    ingested: int = 0
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    failed: int = 0
    # The objects this run handled, in listing order. Also what the cursor advances over.
    keys: list[str] = Field(default_factory=list)
    # True when the listing was still full at the cap, i.e. more objects remain. Saying nothing
    # is how "found=200" on a 2000-object bucket read as a completed sync.
    more: bool = False


_CURSOR_KEY = "sync_cursor"


async def get_cursor(kb_id: str) -> str:
    """Where the last run stopped, per (source, prefix). Empty string = start from the beginning."""
    return str((await get_catalog().get_tenant_config(kb_id)).get(_CURSOR_KEY) or "")


async def set_cursor(kb_id: str, key: str) -> None:
    await get_catalog().set_tenant_config(kb_id, _CURSOR_KEY, key)


async def run_sync(kb_id: str, source: SyncSource, *, actor: str) -> SyncResult:
    """Re-pull an external folder and ingest each object; the pipeline delta skips unchanged
    content, so re-syncs are cheap. Sovereignty-gated like any A2b pull."""
    src = source.external_source
    assert_sovereign(f"{'https' if src.secure else 'http'}://{src.endpoint}")
    # Honour the source's PROVIDER, exactly as the single-object pull does. This path used to
    # resolve S3 credentials and call the S3 fetcher regardless, so an Azure folder sync failed
    # with "missing secret: {ref}_access_key" -- a name Azure does not have and the user could
    # not have supplied. Azure's account name comes from the endpoint and its key from
    # {ref}_account_key or {ref}_sas_token (see rag_clients.external_creds).
    access_key, secret_key = external_credentials(src, get_secrets_provider())
    # CONTINUE where the last run stopped. Without this every run listed from the beginning and
    # stopped at the cap, so a bucket larger than the cap could never be finished -- it ingested
    # the same first N objects forever and reported them as a successful sync.
    cursor = await get_cursor(kb_id)
    keys = await external_fetcher_for(getattr(src, "provider", "s3")).list_keys(
        endpoint=src.endpoint,
        bucket=src.bucket,
        prefix=source.prefix,
        access_key=access_key,
        secret_key=secret_key,
        secure=src.secure,
        limit=source.limit,
        start_after=cursor,
    )
    result = SyncResult(found=len(keys))
    for key in keys:
        try:
            await check_document_cap(kb_id)
            ext = "." + key.rsplit(".", 1)[-1].lower() if "." in key else ""
            doc = DocumentEnvelope(
                tenant_id=kb_id,
                doc_id=key,
                source_id="scheduled-sync",
                source_kind=SourceKind.OBJECT_STORE,
                mime=_EXT_MIME.get(ext, source.mime),
                uri=key,
                external_source=src,
                ingested_at=datetime.now(UTC),
            )
            outcome = await pipeline.ingest_document(doc, actor=actor, strategy=ChunkStrategy(source.strategy))
            result.ingested += 1
            setattr(result, outcome.outcome, getattr(result, outcome.outcome) + 1)
        except Exception as exc:  # per-file failure is reported, never fatal
            result.failed += 1
            _log.warning("sync_item_failed", kb_id=kb_id, key=key, error=str(exc)[:160])
    result.keys = list(keys)
    # A full page means the listing was still going at the cap, so more remains. Advance the
    # cursor to the last key handled; a short page means the end, so reset it -- otherwise a
    # finished sync would never notice an object added to the bucket later, and the delta check
    # already makes re-listing a drained prefix cheap.
    result.more = len(keys) >= source.limit
    await set_cursor(kb_id, keys[-1] if result.more and keys else "")
    _log.info(
        "sync_done",
        kb_id=kb_id,
        found=result.found,
        unchanged=result.unchanged,
        updated=result.updated,
        more=result.more,
    )
    return result


# --- global schedule registry (mirror of the eval-schedule store) ------------
class SyncSchedule(BaseModel):
    kb_id: str
    enabled: bool = False
    interval_hours: int = Field(default=24, ge=1, le=720)
    # Calendar-precise scheduling (banking pack): when ``cron`` is set it takes precedence over
    # interval_hours and is evaluated in ``timezone`` (e.g. "0 2 * * 1" Monday 02:00, Asia/Kolkata).
    cron: str = ""
    timezone: str = "UTC"
    last_run_at: datetime | None = None


@runtime_checkable
class SyncScheduleStore(Protocol):
    async def ensure(self) -> None: ...
    async def get(self, kb_id: str) -> SyncSchedule | None: ...
    async def put(self, sched: SyncSchedule) -> None: ...
    async def list_enabled(self) -> list[SyncSchedule]: ...
    async def mark_run(self, kb_id: str, when: datetime) -> None: ...


class InMemorySyncScheduleStore:
    def __init__(self) -> None:
        self._rows: dict[str, SyncSchedule] = {}

    async def ensure(self) -> None:
        return None

    async def get(self, kb_id: str) -> SyncSchedule | None:
        return self._rows.get(kb_id)

    async def put(self, sched: SyncSchedule) -> None:
        self._rows[sched.kb_id] = sched

    async def list_enabled(self) -> list[SyncSchedule]:
        return [s for s in self._rows.values() if s.enabled]

    async def mark_run(self, kb_id: str, when: datetime) -> None:
        row = self._rows.get(kb_id)
        if row:
            self._rows[kb_id] = row.model_copy(update={"last_run_at": when})


_DDL = (
    f'CREATE SCHEMA IF NOT EXISTS "{_IDENTITY_SCHEMA}"',
    f'''CREATE TABLE IF NOT EXISTS "{_IDENTITY_SCHEMA}".sync_schedules (
        kb_id TEXT PRIMARY KEY,
        enabled BOOLEAN NOT NULL DEFAULT FALSE,
        interval_hours INTEGER NOT NULL DEFAULT 24,
        last_run_at TIMESTAMPTZ
    )''',
    # Calendar-precise scheduling — migration-safe additive columns (banking pack).
    f"""ALTER TABLE "{_IDENTITY_SCHEMA}".sync_schedules ADD COLUMN IF NOT EXISTS cron TEXT NOT NULL DEFAULT ''""",
    f"ALTER TABLE \"{_IDENTITY_SCHEMA}\".sync_schedules ADD COLUMN IF NOT EXISTS timezone TEXT NOT NULL DEFAULT 'UTC'",
)
_COLS = "kb_id, enabled, interval_hours, cron, timezone, last_run_at"


class PostgresSyncScheduleStore:
    def __init__(self, engine: AsyncEngine | None = None) -> None:
        self._engine = engine or create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)

    async def ensure(self) -> None:
        async with self._engine.begin() as c:
            for stmt in _DDL:
                await c.execute(text(stmt))

    async def get(self, kb_id: str) -> SyncSchedule | None:
        await self.ensure()
        sql = f'SELECT {_COLS} FROM "{_IDENTITY_SCHEMA}".sync_schedules WHERE kb_id = :k'
        async with self._engine.connect() as c:
            row = (await c.execute(text(sql), {"k": kb_id})).mappings().first()
        return SyncSchedule(**dict(row)) if row else None

    async def put(self, sched: SyncSchedule) -> None:
        await self.ensure()
        async with self._engine.begin() as c:
            await c.execute(
                text(
                    f'INSERT INTO "{_IDENTITY_SCHEMA}".sync_schedules ({_COLS}) '
                    "VALUES (:kb_id, :enabled, :interval_hours, :cron, :timezone, :last_run_at) "
                    "ON CONFLICT (kb_id) DO UPDATE SET enabled = EXCLUDED.enabled, "
                    "interval_hours = EXCLUDED.interval_hours, cron = EXCLUDED.cron, timezone = EXCLUDED.timezone"
                ),
                sched.model_dump(),
            )

    async def list_enabled(self) -> list[SyncSchedule]:
        await self.ensure()
        async with self._engine.connect() as c:
            rows = (
                (await c.execute(text(f'SELECT {_COLS} FROM "{_IDENTITY_SCHEMA}".sync_schedules WHERE enabled = TRUE')))
                .mappings()
                .all()
            )
        return [SyncSchedule(**dict(r)) for r in rows]

    async def mark_run(self, kb_id: str, when: datetime) -> None:
        await self.ensure()
        async with self._engine.begin() as c:
            await c.execute(
                text(f'UPDATE "{_IDENTITY_SCHEMA}".sync_schedules SET last_run_at = :at WHERE kb_id = :k'),
                {"at": when, "k": kb_id},
            )


@lru_cache(maxsize=1)
def sync_schedule_store() -> SyncScheduleStore:
    """Global scheduled-sync enrolment registry (in-memory under CI; Postgres otherwise)."""
    return InMemorySyncScheduleStore() if get_settings().inmemory else PostgresSyncScheduleStore()


def is_due(sched: SyncSchedule, now: datetime) -> bool:
    if sched.cron:  # calendar-precise mode takes precedence over the hour interval
        from svc_control_plane.cron import cron_is_due  # noqa: PLC0415

        return cron_is_due(sched.cron, sched.timezone, sched.last_run_at, now)
    if sched.last_run_at is None:
        return True
    return now - sched.last_run_at >= timedelta(hours=sched.interval_hours)


async def sync_tick(now: datetime | None = None) -> int:
    """Run every enrolled KB whose sync interval is due. Returns how many ran."""
    now = now or datetime.now(UTC)
    store = sync_schedule_store()
    ran = 0
    for sched in await store.list_enabled():
        if not is_due(sched, now):
            continue
        source = await get_sync_source(sched.kb_id)
        if source is None:
            continue
        try:
            await run_sync(sched.kb_id, source, actor="scheduler")
            await store.mark_run(sched.kb_id, now)
            ran += 1
        except Exception as exc:  # one KB's failure never aborts the tick
            _log.warning("sync_tick_failed", kb_id=sched.kb_id, error=str(exc)[:200])
    return ran
