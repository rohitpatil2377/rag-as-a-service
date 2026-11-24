"""KB-scoped union audit read (InMemoryAuditSink twin) — the read side of the hybrid audit fix.

A KB-scoped read must see the KB's own events UNION any org-schema events that self-attribute to
it (``payload.kb_id``), deduped by ``event_id`` — and be a strict no-op when the KB *is* the org
(single-KB deployment, ``kb_id == org``). These are the semantics the Postgres sink mirrors in SQL.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rag_clients.memory import InMemoryAuditSink
from rag_schemas import AuditEvent
from rag_schemas.common import AuditAction


def _event(tenant_id: str, *, event_id: str, action: AuditAction, kb_id: str | None = None) -> AuditEvent:
    payload: dict[str, object] = {"query": "q"}
    if kb_id is not None:
        payload["kb_id"] = kb_id
    return AuditEvent(
        event_id=event_id,
        tenant_id=tenant_id,
        trace_id="t",
        action=action,
        actor="tester",
        at=datetime.now(UTC),
        payload=payload,
    )


async def test_kb_scoped_read_unions_org_schema_by_payload_kb_id() -> None:
    sink = InMemoryAuditSink()
    kb, org = "acme__sales", "acme"
    # Two events land in the KB's own schema (the correct, going-forward path)...
    await sink.append(_event(kb, event_id="a", action=AuditAction.QUERY))
    await sink.append(_event(kb, event_id="b", action=AuditAction.ABSTAIN))
    # ...one legacy/cross-path event landed in the ORG schema but names this KB in its payload...
    await sink.append(_event(org, event_id="c", action=AuditAction.QUERY, kb_id=kb))
    # ...and an org-schema event for a DIFFERENT KB must never be attributed here.
    await sink.append(_event(org, event_id="d", action=AuditAction.QUERY, kb_id="acme__hr"))

    counts = await sink.count_actions_kb(kb, None)
    assert counts == {"query": 2, "abstain": 1}  # a + c (query), b (abstain); d excluded

    events = await sink.list_events_kb(kb, None, None, 100)
    assert {e.event_id for e in events} == {"a", "b", "c"}


async def test_kb_equals_org_is_a_noop_union_no_double_count() -> None:
    """A single-KB org (kb_id == org): the KB schema IS the org schema, so the union must NOT
    double-count — count_actions_kb equals plain count_actions."""
    sink = InMemoryAuditSink()
    solo = "solo"  # no "__" → org_of(solo) == solo
    for i in range(3):
        await sink.append(_event(solo, event_id=f"e{i}", action=AuditAction.QUERY, kb_id=solo))

    assert await sink.count_actions_kb(solo, None) == await sink.count_actions(solo, None) == {"query": 3}
    events = await sink.list_events_kb(solo, None, None, 100)
    assert len(events) == 3  # not 6


async def test_union_dedupes_by_event_id() -> None:
    """The same event_id present in both schemas (belt-and-suspenders) collapses to one."""
    sink = InMemoryAuditSink()
    kb, org = "acme__sales", "acme"
    await sink.append(_event(kb, event_id="dup", action=AuditAction.QUERY))
    await sink.append(_event(org, event_id="dup", action=AuditAction.QUERY, kb_id=kb))

    assert await sink.count_actions_kb(kb, None) == {"query": 1}
    assert len(await sink.list_events_kb(kb, None, None, 100)) == 1


async def test_list_events_kb_filters_by_action() -> None:
    sink = InMemoryAuditSink()
    kb = "acme__sales"
    await sink.append(_event(kb, event_id="q", action=AuditAction.QUERY))
    await sink.append(_event(kb, event_id="f", action=AuditAction.FAITHFULNESS))
    events = await sink.list_events_kb(kb, None, [AuditAction.FAITHFULNESS.value], 100)
    assert [e.event_id for e in events] == ["f"]


async def test_org_with_no_attributable_events_returns_kb_only() -> None:
    """The in-memory analogue of the Postgres fail-safe: no org-schema events → just the KB's."""
    sink = InMemoryAuditSink()
    kb = "acme__sales"
    await sink.append(_event(kb, event_id="only", action=AuditAction.QUERY))
    assert await sink.count_actions_kb(kb, None) == {"query": 1}
