"""E2E: audit hash-chain integrity verification (FR-H8, FR-G4).

The audit chain is a TREE, not a line: ``AuditSink.append`` reads the current tip and then
inserts, so two events written concurrently legitimately seal the SAME parent and the chain
forks. A real deployment measured 5 such forks in 250 events.

The verifier used to walk the chain strictly linearly, so the first fork ended the walk and
``audit_chain_verified`` came back FALSE on a perfectly intact trail — a regulator-facing
attestation accusing the platform of a broken audit log whenever it was busy enough to write
two events at once. These tests pin both halves of the contract: concurrency must not look
like tampering, and tampering must still be caught.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag_schemas import AuditAction, AuditEvent

pytestmark = pytest.mark.e2e


async def _record(tenant: str, action: AuditAction, actor: str = "tester") -> str:
    from svc_control_plane import wiring

    return await wiring.audit_service().record(
        tenant_id=tenant, trace_id="t-chain", action=action, actor=actor, payload={}
    )


def _log(tenant: str) -> list[AuditEvent]:
    """The in-memory sink's raw chain, so a test can damage it the way a DBA could."""
    from rag_clients.factory import get_audit_sink

    return get_audit_sink()._log[tenant]  # type: ignore[attr-defined,no-any-return]


def _fork(tenant: str, at_index: int) -> None:
    """Append a sibling that seals the SAME parent as an existing event.

    This is what a concurrent append produces: ``append`` reads the current tip and then
    inserts, so two writers racing between those two steps both seal the tip they each read.
    The in-memory sink appends without an await in between, so the race cannot be provoked by
    `asyncio.gather` there — the fork is built explicitly instead, which is deterministic and
    reproduces the exact shape observed in a live Postgres trail (5 forks in 250 events).
    """
    chain = _log(tenant)
    sibling = AuditEvent(
        event_id=f"sibling-{at_index}",
        tenant_id=tenant,
        trace_id="t-chain",
        action=AuditAction.FAITHFULNESS,
        actor="scheduler",
        at=datetime.now(UTC),
        payload={"concurrent": True},
        prev_hash=chain[at_index].prev_hash,
    ).sealed()
    chain.insert(at_index + 1, sibling)


async def test_concurrent_writes_do_not_look_like_tampering(inmemory: None) -> None:
    """The actual bug: a forked chain is intact, and must verify as intact."""
    from svc_control_plane import compliance

    tenant = "forky"
    for _ in range(6):
        await _record(tenant, AuditAction.QUERY)
    # Three forks, including one at the very tip — the scheduler racing live traffic.
    _fork(tenant, 1)
    _fork(tenant, 3)
    _fork(tenant, len(_log(tenant)) - 1)

    chain = await compliance._verify_audit_chain(tenant)

    assert chain.forks == 3, "the fixture must actually fork, or this proves nothing"
    assert chain.events == 9
    assert chain.tampered == 0
    assert chain.missing_parents == 0
    assert chain.orphan_genesis == 0
    assert chain.duplicate_hashes == 0
    assert chain.ok is True, f"a concurrently-written chain must verify, got {chain}"


async def test_attestation_reports_a_busy_trail_as_verified(inmemory: None) -> None:
    """The regression as a regulator sees it: the attestation field itself."""
    from svc_control_plane import compliance

    tenant = "busy-org"
    for _ in range(8):
        await _record(tenant, AuditAction.QUERY)
    _fork(tenant, 2)
    _fork(tenant, 5)

    att = await compliance.attestation(tenant)

    assert att.audit_chain_verified is True
    assert att.audit_events == 10
    assert not any("INTEGRITY FAILED" in n for n in att.notes)


async def test_rewritten_event_is_still_caught(inmemory: None) -> None:
    """Content verification is the half worth keeping — an edited payload must fail."""
    from svc_control_plane import compliance

    tenant = "tampered"
    for _ in range(4):
        await _record(tenant, AuditAction.QUERY)

    # Edit a stored event's payload WITHOUT re-sealing it, which is what someone with database
    # access would do. The hash columns are left untouched.
    stored = _log(tenant)
    stored[1] = stored[1].model_copy(update={"payload": {"query": "something else"}})

    chain = await compliance._verify_audit_chain(tenant)
    assert chain.tampered == 1
    assert chain.ok is False

    att = await compliance.attestation(tenant)
    assert att.audit_chain_verified is False
    assert any("INTEGRITY FAILED" in n for n in att.notes)


async def test_removed_event_is_caught(inmemory: None) -> None:
    """Deleting an interior event orphans its children — the topology half of the check."""
    from svc_control_plane import compliance

    tenant = "redacted"
    for _ in range(5):
        await _record(tenant, AuditAction.QUERY)

    del _log(tenant)[2]  # an interior event: the one after it now points at nothing

    chain = await compliance._verify_audit_chain(tenant)
    assert chain.missing_parents == 1
    assert chain.ok is False


async def test_a_second_genesis_is_caught(inmemory: None) -> None:
    """A chain that restarts mid-log means everything before the restart was discarded."""
    from svc_control_plane import compliance

    tenant = "restarted"
    for _ in range(4):
        await _record(tenant, AuditAction.QUERY)

    orphan = AuditEvent(
        event_id="orphan",
        tenant_id=tenant,
        trace_id="t-chain",
        action=AuditAction.QUERY,
        actor="tester",
        at=datetime.now(UTC),
        payload={},
        prev_hash=None,
    )
    _log(tenant).append(orphan.sealed())

    chain = await compliance._verify_audit_chain(tenant)
    assert chain.orphan_genesis == 1
    assert chain.ok is False
