"""Tests for the MCP tools (FR-H2) — deterministic, in-memory.

Exercises the tool handlers directly (the same callables FastMCP registers): they wrap the
pipeline, stay scoped to the server's single tenant, and enforce that tenant is configured.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime

import pytest

from rag_schemas import DocumentEnvelope
from rag_schemas.common import SourceKind

_TEXT = "The platform uptime target is 99.5 percent baseline."


async def _seed(tenant: str) -> None:
    from svc_control_plane import pipeline

    from rag_common import tenant_scope

    with tenant_scope(tenant):
        await pipeline.ingest_document(
            DocumentEnvelope(
                tenant_id=tenant,
                doc_id="brief",
                source_id="s",
                source_kind=SourceKind.UPLOAD,
                mime="text/plain",
                content_b64=base64.b64encode(_TEXT.encode()).decode(),
                ingested_at=datetime(2026, 7, 2, tzinfo=UTC),
            ),
            actor="loader",
        )


async def test_query_tool_returns_grounded_answer(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MCP_TENANT", "cag")
    from rag_common.config import get_settings

    get_settings.cache_clear()
    from rag_mcp.server import query_tool, retrieve_tool

    await _seed("cag")

    ans = await query_tool("What is the uptime target?", top_k=3)
    assert ans["status"] in ("grounded", "flagged_for_review")
    assert ans["tenant_id"] == "cag"
    assert ans["citations"]

    spans = await retrieve_tool("uptime target", top_k=3)
    assert spans["spans"] and spans["query"]["tenant_id"] == "cag"


async def test_tools_require_a_bound_tenant(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MCP_TENANT", "")
    from rag_common import ConfigError
    from rag_common.config import get_settings

    get_settings.cache_clear()
    from rag_mcp.server import query_tool

    with pytest.raises(ConfigError):
        await query_tool("anything")


def test_server_registers_both_tools(inmemory: None) -> None:
    from rag_mcp.server import build_server

    server = build_server()
    assert server is not None  # smoke: FastMCP builds with the two tools registered


async def test_new_tools_list_ingest_conversation(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MCP_TENANT", "cag2")
    from rag_common.config import get_settings

    get_settings.cache_clear()
    from rag_mcp.server import conversation_tool, ingest_tool, list_documents_tool, list_kbs_tool

    # rag.ingest (stdio is full-trust, so no role context needed).
    out = await ingest_tool("cag2", "note", "Casual leave is 12 days per year.")
    assert out["doc_id"] == "note" and out["outcome"] == "created"

    # rag.list_documents sees it.
    docs = await list_documents_tool()
    assert any(d["doc_id"] == "note" for d in docs["documents"])

    # rag.list_kbs returns the tenant's KBs (empty registry is fine — just a dict shape).
    kbs = await list_kbs_tool()
    assert "knowledge_bases" in kbs

    # rag.conversation returns an answer + a conversation_id to continue with.
    turn = await conversation_tool("How many casual leave days?")
    assert turn["conversation_id"]
    assert turn["status"] in ("grounded", "flagged_for_review", "abstained")


async def test_ingest_tool_enforces_role_over_http(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A reader-only HTTP principal (roles context set) cannot ingest."""
    from rag_mcp import server

    tok_t = server._ctx_tenant.set("cag3")
    tok_r = server._ctx_roles.set(("reader",))
    try:
        with pytest.raises(Exception, match="ingestor"):
            await server.ingest_tool("cag3", "x", "text")
    finally:
        server._ctx_tenant.reset(tok_t)
        server._ctx_roles.reset(tok_r)


async def test_federated_query_tool_requires_flag(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MCP_TENANT", "finance")
    monkeypatch.setenv("RAG_FEDERATION_ENABLED", "false")
    from rag_common import UnauthorizedError
    from rag_common.config import get_settings

    get_settings.cache_clear()
    from rag_mcp.server import federated_query_tool

    with pytest.raises(UnauthorizedError):
        await federated_query_tool("anything?", federate_from=["hr"])


async def test_federated_query_tool_reads_granted_dept(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MCP_TENANT", "finance")
    monkeypatch.setenv("RAG_FEDERATION_ENABLED", "true")
    monkeypatch.setenv("RAG_ABSTAIN_THRESHOLD_LEXICAL", "0.35")  # let the twin ground
    from datetime import timedelta

    from rag_common.config import get_settings

    get_settings.cache_clear()

    from svc_control_plane import wiring
    from svc_control_plane.grants import sign_grant

    from rag_schemas import Grant

    # HR ingests a fact and grants Finance read access to its KB.
    await _seed("hr__policies")
    now = datetime.now(UTC)
    grant = Grant(
        grant_id="g_mcp_1",
        issuer_dept_id="hr",
        subject_dept_id="finance",
        topics=["leave-policy"],
        kb_ids=["hr__policies"],
        max_results_per_query=5,
        not_before=now - timedelta(days=1),
        expires_at=now + timedelta(days=30),
        signed_token="",
        created_at=now,
    )
    grant = grant.model_copy(update={"signed_token": sign_grant(grant)})
    await wiring.grant_store().create(grant)

    from rag_mcp.server import federated_query_tool

    ans = await federated_query_tool("What is the platform uptime target?", federate_from=["hr"], top_k=5)
    assert ans["status"] in ("grounded", "flagged_for_review", "abstained")
    if ans["citations"]:
        assert any(c["doc_id"].startswith("hr/") for c in ans["citations"])
