"""Federation MCP tool pack — delegation is bounded + signed + role-gated, receipt verify
is fail-closed. Mirrors the REST semantics the tools wrap (svc_control_plane.grants /
provenance) so an agent gets identical guarantees."""

from __future__ import annotations

from contextvars import Token
from datetime import UTC, datetime, timedelta

import pytest

from rag_schemas import Grant


def _reset_settings() -> None:
    from rag_common.config import get_settings

    get_settings.cache_clear()


async def _seed_parent(topics: list[str], kb_ids: list[str], *, issuer: str, subject: str) -> Grant:
    """Create a signed, active parent grant (issuer -> subject) in the wiring grant store."""
    from svc_control_plane import wiring
    from svc_control_plane.grants import sign_grant

    now = datetime.now(UTC).replace(microsecond=0)
    g = Grant(
        grant_id="grant_parent01",
        issuer_dept_id=issuer,
        subject_dept_id=subject,
        topics=topics,
        kb_ids=kb_ids,
        max_results_per_query=5,
        not_before=now - timedelta(days=1),
        expires_at=now + timedelta(days=90),
        signed_token="",
        created_at=now,
    )
    g = g.model_copy(update={"signed_token": sign_grant(g)})
    await wiring.grant_store().create(g)
    return g


def _as_dept(dept: str, roles: tuple[str, ...]) -> tuple[Token[str | None], Token[tuple[str, ...]]]:
    """Bind the per-request MCP context to an HTTP principal (dept + roles). Returns the
    reset tokens the caller must reset() in a finally."""
    from rag_mcp import server

    return server._ctx_tenant.set(dept), server._ctx_roles.set(roles)


async def test_delegate_grant_narrows_and_signs(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_FEDERATION_ENABLED", "true")
    _reset_settings()
    from rag_mcp import server
    from rag_mcp.tools_federation import delegate_grant_tool

    await _seed_parent(["payroll", "leave"], ["hr__kb"], issuer="hr", subject="finance")
    t, r = _as_dept("finance", ("owner",))  # finance is the parent's subject, may re-delegate
    try:
        child = await delegate_grant_tool(
            parent_grant_id="grant_parent01",
            subject_dept_id="ops",
            topics=["payroll"],  # a strict subset of the parent's topics
            kb_ids=["hr__kb"],
            max_results_per_query=3,
            ttl_days=30,
        )
    finally:
        server._ctx_tenant.reset(t)
        server._ctx_roles.reset(r)

    assert child["parent_grant_id"] == "grant_parent01"
    assert child["issuer_dept_id"] == "finance" and child["subject_dept_id"] == "ops"
    assert set(child["topics"]) <= {"payroll", "leave"}  # narrowed, never widened
    assert child["signed_token"], "the delegated grant must be signed"

    # It is actually persisted (so the mesh can present it).
    from svc_control_plane import wiring

    stored = await wiring.grant_store().get(child["grant_id"])
    assert stored is not None and stored.parent_grant_id == "grant_parent01"


async def test_delegate_grant_rejects_widening(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_FEDERATION_ENABLED", "true")
    _reset_settings()
    from rag_common import ValidationError
    from rag_mcp import server
    from rag_mcp.tools_federation import delegate_grant_tool

    await _seed_parent(["payroll"], ["hr__kb"], issuer="hr", subject="finance")
    t, r = _as_dept("finance", ("admin",))
    try:
        with pytest.raises(ValidationError):
            await delegate_grant_tool(
                parent_grant_id="grant_parent01",
                subject_dept_id="ops",
                topics=["secret-hr"],  # NOT a subset -> attenuate must reject
                kb_ids=["hr__kb"],
            )
    finally:
        server._ctx_tenant.reset(t)
        server._ctx_roles.reset(r)


async def test_delegate_grant_requires_owner_or_admin(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_FEDERATION_ENABLED", "true")
    _reset_settings()
    from rag_common import ForbiddenError
    from rag_mcp import server
    from rag_mcp.tools_federation import delegate_grant_tool

    await _seed_parent(["payroll"], ["hr__kb"], issuer="hr", subject="finance")
    t, r = _as_dept("finance", ("reader",))  # a plain reader may not delegate
    try:
        with pytest.raises(ForbiddenError):
            await delegate_grant_tool(
                parent_grant_id="grant_parent01", subject_dept_id="ops", topics=["payroll"], kb_ids=["hr__kb"]
            )
    finally:
        server._ctx_tenant.reset(t)
        server._ctx_roles.reset(r)


async def test_delegate_grant_only_subject_can_redelegate(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_FEDERATION_ENABLED", "true")
    _reset_settings()
    from rag_common import ForbiddenError
    from rag_mcp import server
    from rag_mcp.tools_federation import delegate_grant_tool

    await _seed_parent(["payroll"], ["hr__kb"], issuer="hr", subject="finance")
    t, r = _as_dept("legal", ("owner",))  # legal is NOT the parent's subject
    try:
        with pytest.raises(ForbiddenError):
            await delegate_grant_tool(
                parent_grant_id="grant_parent01", subject_dept_id="ops", topics=["payroll"], kb_ids=["hr__kb"]
            )
    finally:
        server._ctx_tenant.reset(t)
        server._ctx_roles.reset(r)


async def test_verify_receipt_is_fail_closed_on_malformed(inmemory: None) -> None:
    from rag_mcp.tools_federation import verify_receipt_tool

    out = await verify_receipt_tool({})  # not a valid receipt shape
    assert out["valid"] is False and out["reason"] == "malformed_receipt"
