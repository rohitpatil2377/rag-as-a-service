"""Ready-to-paste MCP client wiring for THIS deployment.

One place that knows how an agent host connects to the platform's ``/mcp`` endpoint, so
the discovery endpoint, the console Agents page, and the plugin bundle all emit identical,
correct config. The server is JWT-gated, so every snippet carries an Authorization header;
the token is a placeholder unless a caller passes a real one.
"""

from __future__ import annotations

from typing import Any

TOKEN_PLACEHOLDER = "<PLATFORM_JWT_OR_API_KEY>"  # noqa: S105 — a documentation placeholder, not a credential
DEFAULT_SERVER_NAME = "ragaas"


def mcp_url(base_url: str) -> str:
    """The full streamable-HTTP endpoint URL (trailing slash — the mount 307s to it)."""
    return base_url.rstrip("/") + "/mcp/"


def client_configs(
    base_url: str,
    *,
    token: str = TOKEN_PLACEHOLDER,
    server_name: str = DEFAULT_SERVER_NAME,
) -> dict[str, Any]:
    """Every supported client's connection config for ``base_url``.

    ``claude_code_json`` doubles as the Cursor ``.cursor/mcp.json`` (identical shape).
    ``stdio_json`` is for a co-located Claude Desktop / generic stdio host running the
    ``rag-mcp`` command bound to one tenant."""
    url = mcp_url(base_url)
    http_json = {
        "mcpServers": {
            server_name: {
                "type": "http",
                "url": url,
                "headers": {"Authorization": f"Bearer {token}"},
            }
        }
    }
    return {
        "endpoint": url,
        "transport": "streamable-http",
        "auth": "Authorization: Bearer <platform JWT or service-account API key ek_…>",
        "claude_code_cli": (
            f'claude mcp add --transport http {server_name} {url} --header "Authorization: Bearer {token}"'
        ),
        "claude_code_json": http_json,
        "cursor_json": http_json,
        "stdio_json": {
            "mcpServers": {
                server_name: {
                    "command": "rag-mcp",
                    "env": {"RAG_MCP_TENANT": "<TENANT>", "RAG_MCP_TOKEN": token},
                }
            }
        },
    }
