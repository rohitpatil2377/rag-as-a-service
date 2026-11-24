"""E2E: multi-tenant MCP over streamable HTTP (FR-H2 complete).

The MCP endpoint is mounted inside the monolith at /mcp: every request needs a platform
JWT, and each tool call is scoped to THAT principal's org — the tenant is never
client-supplied. Stateless: no cross-request session state to hijack.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest

pytestmark = pytest.mark.e2e

_HEADERS_BASE = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


def _rpc(method: str, rpc_id: int, params: dict | None = None) -> dict:
    body: dict = {"jsonrpc": "2.0", "id": rpc_id, "method": method}
    if params is not None:
        body["params"] = params
    return body


async def _client() -> httpx.AsyncClient:
    from rag_api.app import create_app

    app = create_app()
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
    client._rag_app = app  # type: ignore[attr-defined] - keep the app for its lifespan
    return client


def _auth(tenant: str, groups: list[str] | None = None) -> dict[str, str]:
    from rag_common import issue_dev_jwt

    token = issue_dev_jwt("agent", tenant, roles=["admin", "reader", "ingestor"], attrs={"groups": groups or []})
    return {**_HEADERS_BASE, "Authorization": f"Bearer {token}"}


async def test_mcp_http_is_authenticated_and_tenant_scoped(inmemory: None) -> None:
    c = await _client()
    app = c._rag_app  # type: ignore[attr-defined]
    async with c, app.router.lifespan_context(app):
        # Unauthenticated → 401 before the MCP app is ever reached.
        r = await c.post("/mcp/", json=_rpc("tools/list", 1), headers=_HEADERS_BASE)
        assert r.status_code == 401

        # Authenticated: both tools are advertised.
        r = await c.post("/mcp/", json=_rpc("tools/list", 2), headers=_auth("acme"))
        assert r.status_code == 200
        tools = {t["name"] for t in r.json()["result"]["tools"]}
        assert {"rag.retrieve", "rag.query", "rag.list_kbs", "rag.ingest", "rag.conversation"} <= tools

        # Seed a doc into acme via the normal API, then query it through MCP.
        api_headers = _auth("acme")
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"}, headers=api_headers)
        doc = base64.b64encode(b"The escalation window is 4 hours.").decode()
        ing = await c.post(
            "/v1/kb/acme/ingest", headers=api_headers, json={"doc_id": "sla", "mime": "text/plain", "content_b64": doc}
        )
        assert ing.status_code == 200

        call = _rpc("tools/call", 3, {"name": "rag.query", "arguments": {"query": "What is the escalation window?"}})
        r = await c.post("/mcp/", json=call, headers=_auth("acme"))
        assert r.status_code == 200
        payload = json.loads(r.json()["result"]["content"][0]["text"])
        assert payload["tenant_id"] == "acme"
        assert "4 hours" in payload["answer"]

        # A DIFFERENT org calling the same tool sees nothing of acme's data.
        r = await c.post("/mcp/", json=call, headers=_auth("globex"))
        other = json.loads(r.json()["result"]["content"][0]["text"])
        assert other["tenant_id"] == "globex"
        assert other["status"] == "abstained"
