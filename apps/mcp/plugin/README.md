# RAG as a Service — MCP client plugin

Connect any MCP-capable agent host (Claude Code, Claude Desktop, Cursor, or a custom
client) to this sovereign RAG platform. The MCP server is **already running inside the
platform**, mounted at `/mcp` and gated by a platform JWT — this bundle is just the
client-side wiring plus a token helper.

Tools exposed (all inherit the platform's guardrails, tenancy, grounding, faithfulness
checks, and audit trail):

| Tool | What it does |
|---|---|
| `rag.list_kbs` | List the org's knowledge bases |
| `rag.list_documents` | List documents in a KB |
| `rag.retrieve` | Hybrid retrieval → ranked cited spans |
| `rag.query` | Grounded-or-abstain answer with citations |
| `rag.conversation` | Multi-turn ask (keeps context via `conversation_id`) |
| `rag.ingest` | Ingest a plain-text doc (needs `ingestor` role) |
| `rag.federated_query` | Cross-department answer over grant-authorised peers |
| `rag.attestation` | Sovereignty & data-protection attestation for a KB |
| `rag.usage` | Usage metering + indicative showback |
| `rag.list_grants` | Federation grants issued/received |

> The live tool list for a deployment is always available at `GET /v1/agent/mcp` and
> `GET /v1/mcp/manifest` (the latter also returns ready-to-paste client configs).

---

## 1. Get a token

The MCP endpoint needs `Authorization: Bearer <token>`.

- **Dev / POC:** mint one with the helper (needs `RAG_AUTH_MODE=dev_jwt`, non-prod):
  ```bash
  scripts/mcp-connect.sh -t <tenant> -u https://<host>
  ```
  It prints the exact `claude mcp add` command and a `.mcp.json` snippet with the token.
- **Production:** issue a service-account API key (`ek_…`) from the console and use it as
  the bearer token. Never mint dev tokens in prod.

## 2. Connect

### Claude Code (CLI)
```bash
claude mcp add --transport http ragaas https://<host>/mcp/ \
  --header "Authorization: Bearer <TOKEN>"
```

### Claude Code / Cursor (project file)
Copy `claude-code.mcp.json` to your project as `.mcp.json` (Claude Code) or
`.cursor/mcp.json` (Cursor) and fill in host + token:
```json
{
  "mcpServers": {
    "ragaas": {
      "type": "http",
      "url": "https://<host>/mcp/",
      "headers": { "Authorization": "Bearer <TOKEN>" }
    }
  }
}
```

### Claude Desktop (remote connector)
Add a custom connector pointing at `https://<host>/mcp/` with the same Authorization
header. Or, for a **co-located** install, use the DXT extension (`manifest.json`): it runs
the `rag-mcp` command in stdio mode bound to one tenant — see the note below.

---

## stdio vs HTTP

- **HTTP (recommended)** — connect to the platform's `/mcp` endpoint over the network.
  Multi-tenant, stateless, scoped to your token's org per request. This is the sovereign
  deployment shape.
- **stdio** (`rag-mcp`, DXT `manifest.json`) — the server drives the pipeline
  **in-process**, so it must run where the platform's stores/models are reachable and is
  bound to a single tenant via `RAG_MCP_TENANT` / `RAG_MCP_TOKEN`. Use it for a local,
  single-tenant, co-located agent — not as a thin remote client.

## Security notes

- Tokens are secrets. Don't commit `.mcp.json` with a real token; prefer an env var or the
  helper script at connect time.
- The tenant is **never** client-supplied — it comes from your credential (HTTP) or the
  launch binding (stdio). A credential can only reach KBs its org owns.
- Least privilege: a plain `reader` token sees public-ACL content only. Grant `groups` and
  roles deliberately.
