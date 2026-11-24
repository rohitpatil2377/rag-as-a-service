"""Agent-orchestration MCP tool — gated by the orchestrator flag, and when enabled returns a
well-formed plan-execute result (the real LLM planning happens on the live gateway; the
in-memory extractive gateway simply yields an empty plan, which the tool still surfaces)."""

from __future__ import annotations

import pytest


def _reset_settings() -> None:
    from rag_common.config import get_settings

    get_settings.cache_clear()


async def test_agent_run_disabled_by_default(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MCP_TENANT", "cag")
    monkeypatch.setenv("RAG_AGENT_ORCHESTRATOR_ENABLED", "false")
    _reset_settings()
    from rag_common import UnauthorizedError
    from rag_mcp.tools_agent import agent_run_tool

    with pytest.raises(UnauthorizedError):
        await agent_run_tool(goal="audit this pension case")


async def test_agent_run_enabled_returns_result(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MCP_TENANT", "cag")
    monkeypatch.setenv("RAG_AGENT_ORCHESTRATOR_ENABLED", "true")
    _reset_settings()
    from rag_mcp.tools_agent import agent_run_tool

    out = await agent_run_tool(goal="list the knowledge bases then answer: what is the uptime target?")
    # A well-formed AgentRunResult surfaced as a dict: the goal echoes back, a plan object is
    # present, and a halted_reason string records the outcome (no_plan under the extractive
    # dev gateway — real planning happens on the live LLM).
    assert out["goal"].startswith("list the knowledge bases")
    assert "plan" in out and "halted_reason" in out


def test_agent_run_tool_is_registered(inmemory: None) -> None:
    from rag_mcp.server import default_registry

    assert "rag.agent_run" in default_registry().names()
