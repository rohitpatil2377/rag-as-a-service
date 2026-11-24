"""RAG as a Service MCP server (FR-H2) — sovereign, agent-callable tools over the pipeline.

Exposes two tools that wrap the EXISTING pipeline, so agents automatically get guardrails,
the adaptive router, query transformation, grounded-or-abstain, faithfulness verification,
and the tamper-evident audit trail — the safe path, for free:

* ``rag.retrieve`` — tenant-scoped hybrid retrieval → ranked cited spans (no generation).
* ``rag.query``    — grounded-or-abstain answer + citations (full pipeline).

Two transports, one tenancy rule — the tenant is NEVER client-supplied:

* **stdio** (local agent hosts): the server is bound to ONE tenant at launch
  (``RAG_MCP_TENANT``, optional ``RAG_MCP_TOKEN`` validated against it).
* **streamable HTTP** (networked, multi-tenant): mounted inside the monolith at ``/mcp``;
  every request must carry a platform JWT and is scoped to THAT principal's org for the
  duration of the request (contextvar). Stateless — no cross-request session state.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount
from svc_control_plane import pipeline, wiring

from rag_clients.factory import get_catalog
from rag_common import UnauthorizedError, get_logger, get_principal, get_settings, tenant_scope
from rag_mcp.context import ctx_acl, ctx_roles, ctx_tenant
from rag_mcp.context import current_tenant as _tenant
from rag_mcp.context import require_role as _require
from rag_mcp.context import scoped_tenant as _scoped_tenant
from rag_mcp.registry import ToolRegistry, plugin_paths
from rag_schemas import DocumentEnvelope, RetrievalQuery
from rag_schemas.common import SourceKind

_log = get_logger(__name__)

# Per-request identity lives in ``rag_mcp.context`` so tool-plugin modules can share it
# without importing this server. Aliased to the historical private names below so the tool
# functions and the auth middleware read unchanged.
_ctx_tenant = ctx_tenant
_ctx_acl = ctx_acl
_ctx_roles = ctx_roles


async def retrieve_tool(query: str, top_k: int = 5, kb_id: str | None = None) -> dict[str, Any]:
    """Hybrid retrieval → ranked cited spans (no generation). ``kb_id`` targets one of the
    org's knowledge bases (default: the org's own namespace)."""
    tenant = _scoped_tenant(kb_id)
    rq = RetrievalQuery(tenant_id=tenant, text=query, top_k=top_k, acl=list(_ctx_acl.get()))
    with tenant_scope(tenant):
        result = await (await wiring.retrieval_service_for(tenant)).retrieve(rq)
    return result.model_dump()


async def query_tool(query: str, top_k: int = 5, kb_id: str | None = None) -> dict[str, Any]:
    """Grounded-or-abstain answer with citations (full pipeline: guardrails→…→faithfulness)."""
    tenant = _scoped_tenant(kb_id)
    rq = RetrievalQuery(tenant_id=tenant, text=query, top_k=top_k, acl=list(_ctx_acl.get()))
    with tenant_scope(tenant):
        answer = await pipeline.answer_query(rq, actor="mcp-agent")
    return answer.model_dump()


async def list_kbs_tool() -> dict[str, Any]:
    """List the knowledge bases the caller's organisation owns (to target other tools)."""
    org = _tenant()
    kbs = await wiring.kb_store().list(org)
    return {"knowledge_bases": [{"kb_id": k.kb_id, "name": k.name, "is_default": k.is_default} for k in kbs]}


async def list_documents_tool(kb_id: str | None = None) -> dict[str, Any]:
    """List the documents in a knowledge base (id, type, tags, ingest time)."""
    tenant = _scoped_tenant(kb_id)
    docs = await get_catalog().list_documents(tenant)
    return {
        "documents": [
            {"doc_id": d.doc_id, "mime": d.mime, "ingested_at": d.ingested_at.isoformat(), "metadata": d.metadata}
            for d in docs
        ]
    }


async def ingest_tool(kb_id: str, doc_id: str, text: str) -> dict[str, Any]:
    """Ingest a plain-text document into a knowledge base. Requires the ``ingestor`` role."""
    _require("ingestor")
    tenant = _scoped_tenant(kb_id)
    doc = DocumentEnvelope(
        tenant_id=tenant,
        doc_id=doc_id,
        source_id="mcp-agent",
        source_kind=SourceKind.UPLOAD,
        mime="text/plain",
        content_b64=base64.b64encode(text.encode()).decode(),
        ingested_at=datetime.now(UTC),
    )
    with tenant_scope(tenant):
        outcome = await pipeline.ingest_document(doc, actor="mcp-agent")  # platform default
    return {"doc_id": outcome.doc_id, "chunk_count": outcome.chunk_count, "outcome": outcome.outcome}


async def conversation_tool(query: str, conversation_id: str | None = None, kb_id: str | None = None) -> dict[str, Any]:
    """Multi-turn ask — pass the returned ``conversation_id`` back on the next call to keep
    context. Follow-ups are condensed against history before retrieval; grounded per turn."""
    tenant = _scoped_tenant(kb_id)
    with tenant_scope(tenant):
        if not conversation_id:
            conv = await pipeline.start_conversation(tenant, query)
            conversation_id = conv.conversation_id
        answer = await pipeline.answer_turn(tenant, conversation_id, query, actor="mcp-agent", acl=list(_ctx_acl.get()))
    return {"conversation_id": conversation_id, **answer.model_dump()}


async def federated_query_tool(query: str, federate_from: list[str], top_k: int = 5) -> dict[str, Any]:
    """Federated RAG cross-department query. Answers over the caller's OWN knowledge bases
    plus any department named in ``federate_from`` that has published an active grant to the
    caller — grant-gated, least-privilege (public content only), and audited. A department
    with no grant is skipped silently. Requires RAG_FEDERATION_ENABLED."""
    if not get_settings().federation_enabled:
        raise UnauthorizedError("federation is not enabled")
    org = _tenant()
    own_kbs = [k.kb_id for k in await wiring.kb_store().list(org)]
    with tenant_scope(org):
        answer = await pipeline.answer_federated(
            org, own_kbs, federate_from, query, top_k=top_k, acl=list(_ctx_acl.get()), actor="mcp-agent"
        )
    return answer.model_dump()


# The core pipeline tools: (name, description, fn). Governance tools and operator plugins
# are contributed on top via the registry (see ``default_registry``).
_CORE_TOOLS = (
    ("rag.list_kbs", "List the knowledge bases the caller's organisation owns.", list_kbs_tool),
    (
        "rag.list_documents",
        "List the documents — also called files, artifacts, records, uploads, or sources — in a "
        "knowledge base (id, type, tags). Use for any 'what's in this KB / list everything we "
        "have' request.",
        list_documents_tool,
    ),
    ("rag.retrieve", "Hybrid retrieval → ranked cited spans (no generation).", retrieve_tool),
    ("rag.query", "Grounded-or-abstain answer with citations for a question.", query_tool),
    ("rag.conversation", "Multi-turn ask that keeps context via a conversation_id.", conversation_tool),
    ("rag.ingest", "Ingest a plain-text document (requires the ingestor role).", ingest_tool),
    (
        "rag.federated_query",
        "Federated RAG cross-department answer over own KBs + grant-authorised peer departments.",
        federated_query_tool,
    ),
)


def register_core_tools(registry: ToolRegistry) -> None:
    """Contribute the built-in pipeline tools to ``registry``."""
    for name, desc, fn in _CORE_TOOLS:
        registry.register(name, desc, fn)


def default_registry() -> ToolRegistry:
    """The full tool surface for this deployment, in a deterministic order:
    core pipeline tools → the built-in governance pack → any operator plugins named in
    ``RAG_MCP_PLUGINS``. With no plugins configured (the default) this is exactly the
    built-in set — nothing dynamic is imported."""
    registry = ToolRegistry()
    register_core_tools(registry)
    from rag_mcp.tools_governance import register as register_governance  # noqa: PLC0415

    register_governance(registry)
    from rag_mcp.tools_federation import register as register_federation  # noqa: PLC0415

    register_federation(registry)
    from rag_mcp.tools_agent import register as register_agent  # noqa: PLC0415

    register_agent(registry)
    from rag_mcp.tools_eval import register as register_eval  # noqa: PLC0415

    register_eval(registry)
    registry.load_plugins(plugin_paths(get_settings().mcp_plugins))
    return registry


def tool_catalog() -> list[dict[str, str]]:
    """Name + description of every registered MCP tool (for discovery UIs)."""
    return [{"name": s.name, "description": s.description} for s in default_registry().specs()]


def build_server(*, stateless_http: bool = False, registry: ToolRegistry | None = None) -> FastMCP:
    """Construct the FastMCP server with every registered RAG as a Service tool.

    Pass an explicit ``registry`` to control the tool surface (tests); otherwise the
    :func:`default_registry` (core + governance + configured plugins) is used."""
    server = FastMCP(
        "ragaas",
        stateless_http=stateless_http,
        json_response=stateless_http,
        streamable_http_path="/",  # the monolith mounts us at /mcp; avoid /mcp/mcp
        # Host-header (DNS-rebinding) checks assume a localhost-bound, unauthenticated
        # server. Ours is mounted inside the monolith where EVERY request already needs
        # a platform JWT, and it is reached via nginx under whatever host the deployment
        # uses — so the JWT is the gate, not the Host header.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    for spec in (registry or default_registry()).specs():
        server.tool(name=spec.name, description=spec.description)(spec.fn)
    return server


class _McpAuthMiddleware:
    """ASGI middleware: require a platform JWT on every MCP request and bind the caller's
    org (+ ACL groups) for the duration of the request. 401s never reach the MCP app."""

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        request = Request(scope)
        try:
            # get_principal (not authenticate): accept BOTH a platform JWT and a scoped
            # service-account API key (ek_…). The Agents page hands agents an API key as their
            # MCP credential and the client-config helper advertises it, but the JWT-only
            # authenticate() rejected it here — so the documented token 401'd at the transport.
            principal = await get_principal(request.headers.get("authorization"))
        except UnauthorizedError as exc:
            response = JSONResponse({"code": exc.code, "message": exc.message}, status_code=401)
            await response(scope, receive, send)
            return
        groups = principal.attrs.get("groups", [])
        acl = tuple(str(g) for g in groups) if isinstance(groups, list) else ()
        t_token = _ctx_tenant.set(principal.tenant_id)
        a_token = _ctx_acl.set(acl)
        r_token = _ctx_roles.set(tuple(principal.roles))
        try:
            await self._app(scope, receive, send)
        finally:
            _ctx_tenant.reset(t_token)
            _ctx_acl.reset(a_token)
            _ctx_roles.reset(r_token)


def build_http_server() -> FastMCP:
    """The multi-tenant HTTP-mode server (stateless: every request stands alone)."""
    return build_server(stateless_http=True)


def http_app(server: FastMCP) -> Starlette:
    """The streamable-HTTP ASGI app for ``server``, wrapped in platform-JWT auth.

    Mounted by the monolith at ``/mcp``; the caller must run
    ``server.session_manager.run()`` in the parent lifespan."""
    inner = server.streamable_http_app()
    return Starlette(routes=[Mount("/", app=_McpAuthMiddleware(inner))])


def main() -> None:
    _log.info("mcp_server_start", tenant=get_settings().mcp_tenant, transport="stdio")
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
