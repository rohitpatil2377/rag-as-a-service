"""MCP (Model Context Protocol) seam — Phase-1 stub (FR-H2/F4, TDD §4.8/§7).

Phase-0 ships REST (`/v1`) only. The API handlers in :mod:`svc_control_plane.api` are
written as thin wrappers over the orchestration in :mod:`svc_control_plane.pipeline`,
so an MCP server can expose the SAME ``retrieve`` / ``query`` capabilities as MCP tools
with NO refactor — it will call the pipeline functions directly, under OPA policy.

This module intentionally exposes only the tool *manifest* (names + descriptions) so
the contract is visible now. Wiring a real MCP server (stdio/HTTP transport, auth,
policy) is Phase-1.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class McpTool:
    name: str
    description: str


# The tools an RAG as a Service MCP server will expose to agents (FR-F4). Kept in sync
# with the REST surface so the two front-ends share one orchestration core.
MCP_TOOL_MANIFEST: tuple[McpTool, ...] = (
    McpTool("rag.retrieve", "Tenant-scoped hybrid retrieval → ranked cited spans."),
    McpTool("rag.query", "Grounded-or-abstain answer with citations for a question."),
)


def mcp_manifest() -> list[dict[str, str]]:
    """Return the MCP tool manifest (served read-only in Phase-0 for discoverability)."""
    return [{"name": t.name, "description": t.description} for t in MCP_TOOL_MANIFEST]
