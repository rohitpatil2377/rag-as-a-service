"""rag_mcp — sovereign MCP server exposing RAG as a Service as agent tools (FR-H2)."""

from rag_mcp.server import build_server, query_tool, retrieve_tool, tool_catalog

__all__ = ["build_server", "query_tool", "retrieve_tool", "tool_catalog"]
