"""Audit service (TDD §4.7 · FR-G4).

Thin helper that stamps identity/time onto an :class:`AuditEvent` and appends it to
the tamper-evident :class:`AuditSink`. Every request and agent action routes through
here — including the **abstain** path (declining to answer is an audit-relevant event,
not a silent return; see :meth:`AuditService.record`).
"""

from __future__ import annotations

import builtins  # `list[...]` annotations below must not resolve to the AuditService.list method
import uuid
from datetime import UTC, datetime
from typing import Any

from rag_common import get_logger
from rag_common.siem import SiemForwarder
from rag_core.interfaces import AuditSink
from rag_schemas import AuditEvent
from rag_schemas.common import AuditAction

_log = get_logger(__name__)


class AuditService:
    def __init__(self, sink: AuditSink, siem: SiemForwarder | None = None) -> None:
        self._sink = sink
        self._siem = siem  # optional SIEM forwarder (banking pack); None = disabled

    async def record(
        self,
        *,
        tenant_id: str,
        trace_id: str,
        action: AuditAction,
        actor: str,
        payload: dict[str, Any] | None = None,
        kb_id: str | None = None,
    ) -> str:
        """Build, seal (via the sink's hash chain), and persist an audit event.

        ``kb_id`` self-attributes a query-time event to its knowledge base by stamping
        ``payload['kb_id']`` — so a KB-scoped union read (:meth:`list_events_kb` /
        :meth:`count_actions_kb`) can still recover the event even if it lands in the org
        schema. Pass it ONLY for events that belong to exactly one KB (the single-KB query/turn
        paths); a genuine cross-KB ask has no single owning KB and must not be stamped."""
        body = dict(payload or {})
        if kb_id is not None:
            body["kb_id"] = kb_id
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            trace_id=trace_id,
            action=action,
            actor=actor,
            at=datetime.now(UTC),
            payload=body,
        )
        event_hash = await self._sink.append(event)
        _log.info("audit_appended", trace_id=trace_id, action=action, event_hash=event_hash[:12])
        if self._siem is not None:  # fire-and-forget to the bank's SIEM (never blocks the request)
            self._siem.submit(event.model_dump(mode="json"), event_hash)
        return event_hash

    async def list(self, tenant_id: str, trace_id: str | None = None, limit: int = 100) -> list[AuditEvent]:
        """Read the audit trail for the console/compliance viewer (FR-G4)."""
        return await self._sink.list_events(tenant_id, trace_id, limit)

    async def count_actions(self, tenant_id: str, since: datetime | None) -> dict[str, int]:
        """Per-action event counts for a tenant since ``since`` — backs usage metering (FR-H4)."""
        return await self._sink.count_actions(tenant_id, since)

    async def count_actions_kb(self, kb_id: str, since: datetime | None) -> dict[str, int]:
        """KB-scoped per-action counts: the KB schema UNION any org-schema events that
        self-attribute to this KB (``payload.kb_id``). See :meth:`AuditSink.count_actions_kb`.
        Used by the quality dashboards so a multi-KB org's KB-scoped view is not blind to
        query-time activity; a no-op union when ``kb_id`` is the org itself."""
        return await self._sink.count_actions_kb(kb_id, since)

    async def list_events_kb(
        self, kb_id: str, since: datetime | None, actions: builtins.list[str] | None = None, limit: int = 1000
    ) -> builtins.list[AuditEvent]:
        """KB-scoped event read over the same KB-or-org union (deduped, newest-first) — backs the
        abstain/feedback analytics surface."""
        return await self._sink.list_events_kb(kb_id, since, actions, limit)

    async def token_totals(self, tenant_id: str, since: datetime | None) -> dict[str, int]:
        """Token sums over GENERATE events — token metering / showback (FR-H4)."""
        return await self._sink.token_totals(tenant_id, since)

    async def quality_stats(self, tenant_id: str, since: datetime | None) -> dict[str, object]:
        """Answer-quality aggregates for the dashboards (FR-G5)."""
        return await self._sink.quality_stats(tenant_id, since)
