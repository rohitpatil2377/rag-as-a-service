"""Action guard: safe/critical classification + fail-closed dual-control gating of critical steps."""

from __future__ import annotations

import pytest
from svc_control_plane import agent_guard, dual_control
from svc_control_plane.agent_types import PlanStep


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        ("ingest_document", "critical"),
        ("delete_kb", "critical"),
        ("write_file", "critical"),
        ("update_record", "critical"),
        ("create_kb", "critical"),
        ("execute_shell", "critical"),
        ("send_email", "critical"),
        ("approve_request", "critical"),
        ("provision_tenant", "critical"),
        ("erase_records", "critical"),
        ("revoke_grant", "critical"),
        ("grant_access", "critical"),
        ("INGEST_BULK", "critical"),  # case-insensitive
        ("retrieve_context", "safe"),
        ("query_kb", "safe"),
        ("list_documents", "safe"),
        ("search", "safe"),
    ],
)
def test_tool_criticality(tool: str, expected: str) -> None:
    assert agent_guard.tool_criticality(tool) == expected


async def test_critical_step_is_gated_and_recorded() -> None:
    store = dual_control.InMemoryApprovalStore()
    step = PlanStep(tool="delete_kb", args={"kb_id": "acme__hr"}, rationale="cleanup")

    approval_id = await agent_guard.gate_step(step, actor="agent@acme", tenant_id="acme", store=store)

    assert approval_id is not None
    rows = await store.list("acme")
    assert len(rows) == 1
    assert rows[0].id == approval_id
    assert rows[0].status == "pending"
    assert rows[0].requested_by == "agent@acme"
    assert rows[0].params == {"tool": "delete_kb", "args": {"kb_id": "acme__hr"}}


async def test_safe_step_is_not_gated_and_not_recorded() -> None:
    store = dual_control.InMemoryApprovalStore()
    step = PlanStep(tool="retrieve_context", args={"query": "hello"}, rationale="answer the question")

    approval_id = await agent_guard.gate_step(step, actor="agent@acme", tenant_id="acme", store=store)

    assert approval_id is None
    rows = await store.list("acme")
    assert rows == []
