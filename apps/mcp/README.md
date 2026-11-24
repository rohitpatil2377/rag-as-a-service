# `rag_mcp` — MCP server

Exposes the platform to agent clients over the [Model Context Protocol](https://modelcontextprotocol.io),
using the official `mcp` SDK. See [ADR-0011](../../docs/adr/0011-mcp-server.md).

The tools wrap the **same pipeline the REST API uses**, so an agent inherits every safety control —
permission-aware retrieval, the grounded-or-abstain gate, guardrails, the audit chain — rather than
reaching around them.

## Two transports

| Transport | Tenancy | Use |
|---|---|---|
| **stdio** | Bound to one tenant at launch via `RAG_MCP_TENANT` | A local agent (Claude Desktop, an IDE) talking to one knowledge base. |
| **streamable HTTP** | Multi-tenant; each request is scoped by its platform JWT or API key | Mounted inside the `api` process at `/mcp`. There is no separate container — that mount *is* the deployment. |

Discover the live surface at `GET /v1/agent/mcp`: it returns the endpoint path, transport, auth
mode and the tool catalogue. The console's Agents page renders the same thing with a copy-paste
client config.

## Roles

`context.py` enforces per-tool roles. Note the subtlety it had to fix: an HTTP-authenticated
principal with *zero* roles also produces an empty role tuple, so the check gates on **whether the
request was HTTP-authenticated at all**, not on the tuple being non-empty. Gating on emptiness let
a zero-role token through.
