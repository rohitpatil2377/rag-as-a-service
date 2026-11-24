"""Postgres audit sink (implements :class:`AuditSink`) — FR-G4.

Append-only, per-tenant hash-chained audit log. Each event's ``prev_hash`` links to
the previous event's ``event_hash``, so any tampering (insert/edit/delete) breaks the
chain and is detectable. Phase-1 mirrors this to WORM/Object-Lock and HSM-signs the
chain head (TDD §9); the record/interface is stable.

Isolation: one ``audit_log`` table per tenant schema (``t_<slug>``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from rag_common import get_logger, get_settings, org_of, pg_schema
from rag_schemas import AuditEvent
from rag_schemas.common import AuditAction

_log = get_logger(__name__)


def _ddl(schema: str) -> list[str]:
    return [
        f'CREATE SCHEMA IF NOT EXISTS "{schema}"',
        f'''CREATE TABLE IF NOT EXISTS "{schema}".audit_log (
            seq BIGSERIAL PRIMARY KEY,
            event_id TEXT NOT NULL,
            trace_id TEXT NOT NULL,
            action TEXT NOT NULL,
            actor TEXT NOT NULL,
            at TIMESTAMPTZ NOT NULL,
            payload JSONB NOT NULL,
            prev_hash TEXT,
            event_hash TEXT NOT NULL
        )''',
    ]


class PostgresAuditSink:
    """Append-only, tamper-evident audit log (hash-chained)."""

    def __init__(self, engine: AsyncEngine | None = None) -> None:
        self._engine = engine or create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)

    async def ensure_schema(self, tenant_id: str) -> None:
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as conn:
            for stmt in _ddl(schema):
                await conn.execute(text(stmt))

    async def append(self, event: object) -> str:
        """Chain + persist a sealed :class:`AuditEvent`; return its ``event_hash``."""
        if not isinstance(event, AuditEvent):
            raise TypeError("AuditSink.append expects an AuditEvent")
        schema = pg_schema(event.tenant_id)
        await self.ensure_schema(event.tenant_id)
        async with self._engine.begin() as conn:
            prev = (
                await conn.execute(text(f'SELECT event_hash FROM "{schema}".audit_log ORDER BY seq DESC LIMIT 1'))
            ).scalar_one_or_none()
            sealed = event.model_copy(update={"prev_hash": prev}).sealed()
            await conn.execute(
                text(
                    f'''INSERT INTO "{schema}".audit_log
                        (event_id, trace_id, action, actor, at, payload, prev_hash, event_hash)
                        VALUES (:event_id, :trace_id, :action, :actor, :at,
                                CAST(:payload AS JSONB), :prev_hash, :event_hash)'''
                ),
                {
                    "event_id": sealed.event_id,
                    "trace_id": sealed.trace_id,
                    "action": sealed.action.value,
                    "actor": sealed.actor,
                    "at": sealed.at,
                    "payload": _json(sealed.payload),
                    "prev_hash": sealed.prev_hash,
                    "event_hash": sealed.event_hash,
                },
            )
        assert sealed.event_hash is not None
        return sealed.event_hash

    async def list_events(self, tenant_id: str, trace_id: str | None, limit: int) -> list[AuditEvent]:
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        clause = "WHERE trace_id = :trace_id " if trace_id else ""
        params: dict[str, object] = {"limit": limit}
        if trace_id:
            params["trace_id"] = trace_id
        async with self._engine.connect() as conn:
            rows = (
                (
                    await conn.execute(
                        text(f'SELECT * FROM "{schema}".audit_log {clause}ORDER BY seq DESC LIMIT :limit'), params
                    )
                )
                .mappings()
                .all()
            )
        return [_row_to_event(dict(r), tenant_id) for r in rows]

    async def count_actions(self, tenant_id: str, since: datetime | None) -> dict[str, int]:
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        clause = "WHERE at >= :since " if since else ""
        params: dict[str, object] = {}
        if since:
            params["since"] = since
        async with self._engine.connect() as conn:
            rows = (
                (
                    await conn.execute(
                        text(f'SELECT action, COUNT(*) AS n FROM "{schema}".audit_log {clause}GROUP BY action'), params
                    )
                )
                .mappings()
                .all()
            )
        return {str(r["action"]): int(r["n"]) for r in rows}

    async def count_actions_kb(self, kb_id: str, since: datetime | None) -> dict[str, int]:
        """KB-scoped per-action counts unioned with org-schema events attributed to this KB."""
        merged = await self.count_actions(kb_id, since)  # KB's own schema (created if absent)
        org_tenant = org_of(kb_id)
        if pg_schema(org_tenant) == pg_schema(kb_id):  # kb_id IS the org → no separate org schema
            return merged
        for action, n in (await self._org_attributed_counts(org_tenant, kb_id, since)).items():
            merged[action] = merged.get(action, 0) + n
        return merged

    async def _org_attributed_counts(self, org_tenant_id: str, kb_id: str, since: datetime | None) -> dict[str, int]:
        """Per-action counts of org-schema events that name ``kb_id`` in their payload. READ-ONLY
        and fail-safe: the org schema/table may not exist (never provisioned) → return nothing."""
        schema = pg_schema(org_tenant_id)
        clause = "AND at >= :since " if since else ""
        params: dict[str, object] = {"kb_id": kb_id}
        if since:
            params["since"] = since
        query = (
            f'SELECT action, COUNT(*) AS n FROM "{schema}".audit_log '
            f"WHERE payload->>'kb_id' = :kb_id {clause}GROUP BY action"
        )
        try:
            async with self._engine.connect() as conn:
                rows = (await conn.execute(text(query), params)).mappings().all()
        except Exception as exc:  # org schema/table absent is expected; degrade to KB-only
            _log.debug("org_audit_union_unavailable", schema=schema, error=str(exc)[:160])
            return {}
        return {str(r["action"]): int(r["n"]) for r in rows}

    async def list_events_kb(
        self, kb_id: str, since: datetime | None, actions: list[str] | None, limit: int
    ) -> list[AuditEvent]:
        """KB-scoped events, KB schema UNION org-schema events attributed to this KB, deduped."""
        await self.ensure_schema(kb_id)  # a KB with no audit events yet still has a readable table
        kb_events = await self._scoped_events(pg_schema(kb_id), kb_id, since, actions, limit, kb_filter=False)
        org_tenant = org_of(kb_id)
        if pg_schema(org_tenant) == pg_schema(kb_id):  # kb_id IS the org → single schema, no union
            return kb_events
        org_events = await self._scoped_events(
            pg_schema(org_tenant), kb_id, since, actions, limit, kb_filter=True, fail_safe=True
        )
        seen: set[str] = set()
        merged: list[AuditEvent] = []
        for event in sorted([*kb_events, *org_events], key=lambda e: e.at, reverse=True):
            if event.event_id in seen:
                continue
            seen.add(event.event_id)
            merged.append(event)
        return merged[:limit]

    async def _scoped_events(
        self,
        schema: str,
        kb_id: str,
        since: datetime | None,
        actions: list[str] | None,
        limit: int,
        *,
        kb_filter: bool,
        fail_safe: bool = False,
    ) -> list[AuditEvent]:
        """Read events from one schema (newest-first). ``kb_filter`` restricts to rows whose payload
        names ``kb_id`` (used on the org schema). ``fail_safe`` swallows a missing schema/table."""
        conds: list[str] = []
        params: dict[str, object] = {"limit": limit}
        if since:
            conds.append("at >= :since")
            params["since"] = since
        if actions:
            conds.append("action = ANY(:actions)")
            params["actions"] = list(actions)
        if kb_filter:
            conds.append("payload->>'kb_id' = :kb_id")
            params["kb_id"] = kb_id
        where = ("WHERE " + " AND ".join(conds) + " ") if conds else ""
        query = f'SELECT * FROM "{schema}".audit_log {where}ORDER BY seq DESC LIMIT :limit'
        try:
            async with self._engine.connect() as conn:
                rows = (await conn.execute(text(query), params)).mappings().all()
        except Exception as exc:
            if fail_safe:  # org schema/table absent is expected on the union's org side
                _log.debug("org_audit_union_unavailable", schema=schema, error=str(exc)[:160])
                return []
            raise
        # tenant_id is implicit in the schema; the event is attributed to the KB being read.
        return [_row_to_event(dict(r), kb_id) for r in rows]

    async def token_totals(self, tenant_id: str, since: datetime | None) -> dict[str, int]:
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        clause = "AND at >= :since " if since else ""
        params: dict[str, object] = {}
        if since:
            params["since"] = since
        query = (
            "SELECT COALESCE(SUM((payload->>'prompt_tokens')::bigint), 0) AS p, "
            "COALESCE(SUM((payload->>'completion_tokens')::bigint), 0) AS c, "
            "COALESCE(SUM((payload->>'total_tokens')::bigint), 0) AS t "
            f"FROM \"{schema}\".audit_log WHERE action = 'generate' {clause}"
        )
        async with self._engine.connect() as conn:
            row = (await conn.execute(text(query), params)).mappings().first()
        if row is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return {"prompt_tokens": int(row["p"]), "completion_tokens": int(row["c"]), "total_tokens": int(row["t"])}

    async def quality_stats(self, tenant_id: str, since: datetime | None) -> dict[str, object]:
        await self.ensure_schema(tenant_id)
        schema = pg_schema(tenant_id)
        clause = "AND at >= :since " if since else ""
        params: dict[str, object] = {}
        if since:
            params["since"] = since
        faith_q = (
            "SELECT COUNT(*) AS n, AVG((payload->>'score')::float) AS avg, "
            "COALESCE(SUM(CASE WHEN (payload->>'score')::float < 0.4 THEN 1 ELSE 0 END), 0) AS low, "
            "COALESCE(SUM(CASE WHEN (payload->>'score')::float >= 0.7 THEN 1 ELSE 0 END), 0) AS high "
            f"FROM \"{schema}\".audit_log WHERE action = 'faithfulness' {clause}"
        )
        group_q = (
            "SELECT payload->>:field AS k, COUNT(*) AS n "
            f'FROM "{schema}".audit_log WHERE action = :action {clause}GROUP BY payload->>:field'
        )
        async with self._engine.connect() as conn:
            f = (await conn.execute(text(faith_q), params)).mappings().first()
            faith_params = {**params, "field": "outcome", "action": "faithfulness"}
            outcomes = (await conn.execute(text(group_q), faith_params)).all()
            reasons = (await conn.execute(text(group_q), {**params, "field": "reason", "action": "abstain"})).all()
            stages = (await conn.execute(text(group_q), {**params, "field": "stage", "action": "guardrail"})).all()
            feedback = (await conn.execute(text(group_q), {**params, "field": "rating", "action": "feedback"})).all()
        n = int(f["n"]) if f else 0
        low = int(f["low"]) if f else 0
        high = int(f["high"]) if f else 0
        return {
            "faithfulness_n": n,
            "faithfulness_avg": round(float(f["avg"]), 4) if f and f["avg"] is not None else None,
            "faithfulness_buckets": {"low": low, "mid": n - low - high, "high": high},
            "outcomes": {str(r.k): int(r.n) for r in outcomes if r.k},
            "abstain_reasons": {str(r.k): int(r.n) for r in reasons if r.k},
            "guardrail_stages": {str(r.k): int(r.n) for r in stages if r.k},
            "feedback": {str(r.k): int(r.n) for r in feedback if r.k},
        }

    async def aclose(self) -> None:
        await self._engine.dispose()


def _row_to_event(m: Mapping[str, Any], tenant_id: str) -> AuditEvent:
    payload = m["payload"]
    return AuditEvent(
        event_id=str(m["event_id"]),
        tenant_id=tenant_id,  # tenant is implicit in the schema, re-attached on read
        trace_id=str(m["trace_id"]),
        action=AuditAction(m["action"]),
        actor=str(m["actor"]),
        at=m["at"],
        payload=payload if isinstance(payload, dict) else json.loads(payload),
        prev_hash=m["prev_hash"],
        event_hash=m["event_hash"],
    )


def _json(value: object) -> str:
    return json.dumps(value, default=str)
