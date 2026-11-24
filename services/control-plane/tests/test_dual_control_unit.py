"""Maker-checker engine: separation of duties, execute-on-approve, reject (banking pack)."""

from __future__ import annotations

import pytest
from svc_control_plane import dual_control

from rag_common.errors import ForbiddenError


async def _submit(store: dual_control.InMemoryApprovalStore) -> dual_control.ApprovalRequest:
    return await dual_control.submit(
        store,
        action="purge",
        tenant_id="bank",
        kb_id="bank__hr",
        params={"kb_id": "bank__hr"},
        summary="purge HR",
        requested_by="maker@bank",
    )


async def test_maker_cannot_approve_own_request() -> None:
    store = dual_control.InMemoryApprovalStore()
    req = await _submit(store)
    with pytest.raises(ForbiddenError):
        await dual_control.decide(store, req_id=req.id, approve=True, decided_by="maker@bank")
    # Still pending — nothing executed.
    assert (await store.get(req.id)).status == "pending"  # type: ignore[union-attr]


async def test_second_admin_approval_runs_the_executor() -> None:
    store = dual_control.InMemoryApprovalStore()
    ran: list[str] = []

    async def _fake(req: dual_control.ApprovalRequest) -> str:
        ran.append(req.params["kb_id"])
        return "done"

    dual_control.register("purge", _fake)
    try:
        req = await _submit(store)
        decided = await dual_control.decide(store, req_id=req.id, approve=True, decided_by="checker@bank")
        assert decided.status == "executed"
        assert decided.decided_by == "checker@bank"
        assert ran == ["bank__hr"]  # the action actually ran
    finally:
        dual_control._EXECUTORS.pop("purge", None)


async def test_execution_failure_is_recorded_not_raised() -> None:
    store = dual_control.InMemoryApprovalStore()

    async def _boom(_: dual_control.ApprovalRequest) -> str:
        raise RuntimeError("store offline")

    dual_control.register("purge", _boom)
    try:
        req = await _submit(store)
        decided = await dual_control.decide(store, req_id=req.id, approve=True, decided_by="checker@bank")
        assert decided.status == "failed"
        assert "store offline" in (decided.error or "")
    finally:
        dual_control._EXECUTORS.pop("purge", None)


async def test_reject_does_not_execute() -> None:
    store = dual_control.InMemoryApprovalStore()
    ran: list[str] = []

    async def _exec(_: dual_control.ApprovalRequest) -> str:
        return await _noop(ran)

    dual_control.register("purge", _exec)
    try:
        req = await _submit(store)
        decided = await dual_control.decide(store, req_id=req.id, approve=False, decided_by="maker@bank")
        assert decided.status == "rejected"
        assert ran == []
    finally:
        dual_control._EXECUTORS.pop("purge", None)


async def _noop(ran: list[str]) -> str:
    ran.append("x")
    return "x"


def test_is_gated_respects_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_common import get_settings

    monkeypatch.setenv("RAG_DUAL_CONTROL_ENABLED", "false")
    get_settings.cache_clear()
    assert dual_control.is_gated("purge") is False

    monkeypatch.setenv("RAG_DUAL_CONTROL_ENABLED", "true")
    monkeypatch.setenv("RAG_DUAL_CONTROL_ACTIONS", "purge,erasure")
    get_settings.cache_clear()
    assert dual_control.is_gated("purge") is True
    assert dual_control.is_gated("security_change") is False  # not in the chosen set
    get_settings.cache_clear()
