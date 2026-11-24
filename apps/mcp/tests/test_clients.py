"""Unit tests for the MCP client-config generator (plugin bundle + discovery endpoint)."""

from __future__ import annotations

import json

from rag_mcp.clients import TOKEN_PLACEHOLDER, client_configs, mcp_url


def test_mcp_url_normalises_trailing_slash() -> None:
    assert mcp_url("https://host") == "https://host/mcp/"
    assert mcp_url("https://host/") == "https://host/mcp/"


def test_client_configs_shape_and_placeholder() -> None:
    cfg = client_configs("https://rag.example.gov.in/")
    assert cfg["endpoint"] == "https://rag.example.gov.in/mcp/"
    # HTTP configs carry the bearer header with the placeholder token.
    server = cfg["claude_code_json"]["mcpServers"]["ragaas"]
    assert server["type"] == "http"
    assert server["url"] == "https://rag.example.gov.in/mcp/"
    assert server["headers"]["Authorization"] == f"Bearer {TOKEN_PLACEHOLDER}"
    # Cursor shares the same shape.
    assert cfg["cursor_json"] == cfg["claude_code_json"]
    # CLI one-liner references the real endpoint.
    assert "claude mcp add --transport http ragaas https://rag.example.gov.in/mcp/" in cfg["claude_code_cli"]
    # stdio config uses the rag-mcp command bound to a tenant.
    stdio = cfg["stdio_json"]["mcpServers"]["ragaas"]
    assert stdio["command"] == "rag-mcp"
    assert "RAG_MCP_TENANT" in stdio["env"]


def test_client_configs_injects_real_token_and_custom_name() -> None:
    cfg = client_configs("http://localhost:8000", token="ek_live_123", server_name="my-rag")
    server = cfg["claude_code_json"]["mcpServers"]["my-rag"]
    assert server["headers"]["Authorization"] == "Bearer ek_live_123"
    # Everything is JSON-serialisable (it's returned by an API and pasted into files).
    json.dumps(cfg)
