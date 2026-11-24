"""E2E: FR-H2 agent connectivity info (MCP)."""

from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.e2e


async def test_mcp_info(inmemory: None) -> None:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    token = issue_dev_jwt("o", "mcpco", roles=["reader"])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()),
        base_url="http://t",
        headers={"Authorization": f"Bearer {token}"},
    ) as c:
        info = (await c.get("/v1/agent/mcp")).json()
        assert info["enabled"] is True and info["endpoint_path"] == "/mcp"
        assert info["transport"] == "streamable-http"
        names = {t["name"] for t in info["tools"]}
        assert {"rag.retrieve", "rag.query"} <= names
