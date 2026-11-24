"""Continuous-evaluation schedule registry (FR-G9) — which KBs are watched, and how.

The scheduler cannot enumerate KBs across tenants (``KnowledgeBaseStore.list`` is
per-tenant), so enrolled KBs live in one small GLOBAL table in the ``rag_identity``
schema (same precedent as API keys). A row is a KB's opt-in: its run interval, its
per-KB metric floors, and when it last ran. The in-process scheduler reads
``list_enabled()`` each tick and runs the ones whose interval is due.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from rag_common import get_logger, get_settings

_log = get_logger(__name__)

_IDENTITY_SCHEMA = "rag_identity"


class EvalSchedule(BaseModel):
    """A KB's continuous-evaluation enrolment (FR-G9)."""

    kb_id: str
    enabled: bool = False
    interval_hours: int = Field(default=24, ge=1, le=720)
    min_grounded_rate: float = Field(default=0.6, ge=0.0, le=1.0)
    min_faithfulness: float = Field(default=0.5, ge=0.0, le=1.0)
    # Peers to also evaluate ACROSS THE MESH (per-grant quality attribution, FR-G9). Empty =
    # own-KB eval only; when set, each scheduled run also ablates each peer to catch one that
    # degrades the joint answer. See svc_control_plane.federated_eval.
    federate_from: list[str] = Field(default_factory=list)
    last_run_at: datetime | None = None


@runtime_checkable
class EvalScheduleStore(Protocol):
    async def ensure(self) -> None: ...
    async def get(self, kb_id: str) -> EvalSchedule | None: ...
    async def put(self, sched: EvalSchedule) -> None: ...
    async def list_enabled(self) -> list[EvalSchedule]: ...
    async def mark_run(self, kb_id: str, when: datetime) -> None: ...
    async def delete(self, kb_id: str) -> None: ...


class InMemoryEvalScheduleStore:
    def __init__(self) -> None:
        self._rows: dict[str, EvalSchedule] = {}

    async def ensure(self) -> None:
        return None

    async def get(self, kb_id: str) -> EvalSchedule | None:
        return self._rows.get(kb_id)

    async def put(self, sched: EvalSchedule) -> None:
        self._rows[sched.kb_id] = sched

    async def list_enabled(self) -> list[EvalSchedule]:
        return [s for s in self._rows.values() if s.enabled]

    async def mark_run(self, kb_id: str, when: datetime) -> None:
        row = self._rows.get(kb_id)
        if row:
            self._rows[kb_id] = row.model_copy(update={"last_run_at": when})

    async def delete(self, kb_id: str) -> None:
        self._rows.pop(kb_id, None)


_DDL = (
    f'CREATE SCHEMA IF NOT EXISTS "{_IDENTITY_SCHEMA}"',
    f'''CREATE TABLE IF NOT EXISTS "{_IDENTITY_SCHEMA}".eval_schedules (
        kb_id TEXT PRIMARY KEY,
        enabled BOOLEAN NOT NULL DEFAULT FALSE,
        interval_hours INTEGER NOT NULL DEFAULT 24,
        min_grounded_rate DOUBLE PRECISION NOT NULL DEFAULT 0.6,
        min_faithfulness DOUBLE PRECISION NOT NULL DEFAULT 0.5,
        federate_from TEXT[] NOT NULL DEFAULT '{{}}',
        last_run_at TIMESTAMPTZ
    )''',
    # ADD COLUMN IF NOT EXISTS: mirrors the same fix in grants.py — this table predates
    # federate_from on already-provisioned deployments, so CREATE TABLE IF NOT EXISTS above
    # silently no-ops and never adds it.
    f'ALTER TABLE "{_IDENTITY_SCHEMA}".eval_schedules ADD COLUMN IF NOT EXISTS federate_from '
    f"TEXT[] NOT NULL DEFAULT '{{}}'",
)

_COLS = "kb_id, enabled, interval_hours, min_grounded_rate, min_faithfulness, federate_from, last_run_at"


class PostgresEvalScheduleStore:
    def __init__(self, engine: AsyncEngine | None = None) -> None:
        self._engine = engine or create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)

    async def ensure(self) -> None:
        async with self._engine.begin() as c:
            for stmt in _DDL:
                await c.execute(text(stmt))

    async def get(self, kb_id: str) -> EvalSchedule | None:
        await self.ensure()
        sql = f'SELECT {_COLS} FROM "{_IDENTITY_SCHEMA}".eval_schedules WHERE kb_id = :k'
        async with self._engine.connect() as c:
            row = (await c.execute(text(sql), {"k": kb_id})).mappings().first()
        return EvalSchedule(**dict(row)) if row else None

    async def put(self, sched: EvalSchedule) -> None:
        await self.ensure()
        async with self._engine.begin() as c:
            await c.execute(
                text(
                    f'INSERT INTO "{_IDENTITY_SCHEMA}".eval_schedules ({_COLS}) '
                    "VALUES (:kb_id, :enabled, :interval_hours, :min_grounded_rate, :min_faithfulness, "
                    ":federate_from, :last_run_at) "
                    "ON CONFLICT (kb_id) DO UPDATE SET enabled = EXCLUDED.enabled, "
                    "interval_hours = EXCLUDED.interval_hours, min_grounded_rate = EXCLUDED.min_grounded_rate, "
                    "min_faithfulness = EXCLUDED.min_faithfulness, federate_from = EXCLUDED.federate_from"
                ),
                sched.model_dump(),
            )

    async def list_enabled(self) -> list[EvalSchedule]:
        await self.ensure()
        async with self._engine.connect() as c:
            rows = (
                (await c.execute(text(f'SELECT {_COLS} FROM "{_IDENTITY_SCHEMA}".eval_schedules WHERE enabled = TRUE')))
                .mappings()
                .all()
            )
        return [EvalSchedule(**dict(r)) for r in rows]

    async def mark_run(self, kb_id: str, when: datetime) -> None:
        await self.ensure()
        async with self._engine.begin() as c:
            await c.execute(
                text(f'UPDATE "{_IDENTITY_SCHEMA}".eval_schedules SET last_run_at = :at WHERE kb_id = :k'),
                {"at": when, "k": kb_id},
            )

    async def delete(self, kb_id: str) -> None:
        await self.ensure()
        async with self._engine.begin() as c:
            await c.execute(text(f'DELETE FROM "{_IDENTITY_SCHEMA}".eval_schedules WHERE kb_id = :k'), {"k": kb_id})
