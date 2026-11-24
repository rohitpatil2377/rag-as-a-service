"""Agent orchestrator — plan -> critic -> approval-gate -> execute -> synthesise."""

from __future__ import annotations

from typing import Any

import pytest
from svc_control_plane import agent_critic, agent_planner, dual_control
from svc_control_plane.agent_orchestrator import ToolSpec, run_agent
from svc_control_plane.agent_types import Plan, PlanStep, Review


class _Gateway:
    async def complete(self, prompt: str, *, max_tokens: int | None = None) -> str:
        return "final answer"


async def _echo_tool(_tenant: str, _actor: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"echo": args}


def _tools() -> dict[str, ToolSpec]:
    return {
        "retrieve": ToolSpec("read passages", _echo_tool),
        "ingest_document": ToolSpec("write a doc", _echo_tool),  # name trips the critical guard
    }


def _patch_plan(monkeypatch: pytest.MonkeyPatch, steps: list[PlanStep]) -> None:
    async def _plan(goal: str, catalog: list[Any], *, gateway: object, max_tokens: int = 1000) -> Plan:
        return Plan(goal=goal, steps=steps)

    monkeypatch.setattr(agent_planner, "plan", _plan)


def _patch_review(monkeypatch: pytest.MonkeyPatch, approved: bool) -> None:
    async def _review(plan: Plan, *, gateway: object, max_tokens: int = 600) -> Review:
        return Review(approved=approved, concerns=[] if approved else ["unsafe"])

    monkeypatch.setattr(agent_critic, "review_plan", _review)


async def test_safe_plan_executes_and_synthesises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_plan(monkeypatch, [PlanStep(tool="retrieve", args={"query": "x"})])
    _patch_review(monkeypatch, approved=True)
    store = dual_control.InMemoryApprovalStore()
    r = await run_agent("do x", tenant_id="k", actor="u", gateway=_Gateway(), store=store, tools=_tools())
    assert r.halted_reason == ""
    assert r.observations[0].ok and r.observations[0].result == {"echo": {"query": "x"}}
    assert r.answer == "final answer"


async def test_critical_step_pauses_for_human_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_plan(monkeypatch, [PlanStep(tool="ingest_document", args={"doc_id": "d"})])
    _patch_review(monkeypatch, approved=True)
    store = dual_control.InMemoryApprovalStore()
    r = await run_agent("ingest", tenant_id="k", actor="u", gateway=_Gateway(), store=store, tools=_tools())
    assert r.pending_approval_id is not None
    assert r.halted_reason == "awaiting_approval"
    assert r.observations[-1].approval_id == r.pending_approval_id  # step paused, did NOT run
    req = await store.get(r.pending_approval_id)
    assert req is not None and req.status == "pending"


async def test_critic_block_halts_before_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_plan(monkeypatch, [PlanStep(tool="retrieve", args={})])
    _patch_review(monkeypatch, approved=False)
    store = dual_control.InMemoryApprovalStore()
    r = await run_agent("x", tenant_id="k", actor="u", gateway=_Gateway(), store=store, tools=_tools())
    assert r.halted_reason == "critic_blocked"
    assert r.observations == []


async def test_unknown_tool_is_an_observation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_plan(monkeypatch, [PlanStep(tool="does_not_exist", args={})])
    _patch_review(monkeypatch, approved=True)
    store = dual_control.InMemoryApprovalStore()
    r = await run_agent("x", tenant_id="k", actor="u", gateway=_Gateway(), store=store, tools=_tools())
    assert r.observations[0].ok is False
    assert "unknown tool" in r.observations[0].error


async def test_empty_plan_halts(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_plan(monkeypatch, [])
    _patch_review(monkeypatch, approved=True)
    store = dual_control.InMemoryApprovalStore()
    r = await run_agent("x", tenant_id="k", actor="u", gateway=_Gateway(), store=store, tools=_tools())
    assert r.halted_reason == "no_plan"
