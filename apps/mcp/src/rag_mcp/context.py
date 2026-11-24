"""Per-request identity context shared by every MCP tool (core + plugins).

Extracted from ``server.py`` so tool-plugin modules can resolve the caller's
tenant/ACL/roles WITHOUT importing the server (which would create an import cycle:
``server`` builds the registry, the registry imports plugin modules, and a plugin
module needs these helpers). One source of truth, no cycle.

Two ways a call is authorised, exactly as before:
* **HTTP** — ``_McpAuthMiddleware`` sets these contextvars from the request principal.
* **stdio** — no contextvars; the tenant comes from ``RAG_MCP_TENANT`` (+ optional
  ``RAG_MCP_TOKEN``), and the operator-launched binding is full-trust for role checks.
"""

from __future__ import annotations

from contextvars import ContextVar

from svc_control_plane.widget import owns_kb

from rag_common import ConfigError, UnauthorizedError, authenticate, get_settings

# Per-request principal (HTTP transport). Empty acl = tenant-public spans only (A6).
ctx_tenant: ContextVar[str | None] = ContextVar("rag_mcp_tenant", default=None)
ctx_acl: ContextVar[tuple[str, ...]] = ContextVar("rag_mcp_acl", default=())
ctx_roles: ContextVar[tuple[str, ...]] = ContextVar("rag_mcp_roles", default=())


def current_tenant() -> str:
    """The tenant this call is authorised for — request principal (HTTP) or the
    launch-time binding (stdio). Fail-closed when neither is present."""
    from_request = ctx_tenant.get()
    if from_request:
        return from_request
    settings = get_settings()
    tenant = settings.mcp_tenant
    if not tenant:
        raise ConfigError("no tenant bound: authenticate over HTTP, or set RAG_MCP_TENANT for stdio")
    if settings.mcp_token:
        principal = authenticate(f"Bearer {settings.mcp_token}")
        if principal.tenant_id != tenant:
            raise ConfigError("RAG_MCP_TOKEN tenant does not match RAG_MCP_TENANT")
    return tenant


def scoped_tenant(kb_id: str | None) -> str:
    """The namespace a tool operates on: a specific ``kb_id`` the caller's org OWNS, or the
    org's own default namespace. Never lets a credential reach another org's KB."""
    base = current_tenant()
    if not kb_id:
        return base
    if not owns_kb(base, kb_id):
        raise UnauthorizedError("credential may not access this knowledge base")
    return kb_id


def require_role(role: str) -> None:
    """Enforce a role for write tools, mirroring the REST ``_require_role`` gate exactly.

    HTTP requests are ALWAYS bound to a request context (``_McpAuthMiddleware`` sets
    ``ctx_tenant`` from the authenticated principal for every request, even one whose
    principal happens to hold zero roles) — so ``ctx_tenant.get() is not None`` is the
    correct "am I on the HTTP transport" signal, and the role check must be strict there:
    a principal with no roles must be denied, exactly like REST. Only the stdio transport
    (no request context at all — ``ctx_tenant`` stays at its ``None`` default) is the
    operator-launched, full-trust binding that skips the check.

    Previously this branched on ``roles`` being non-empty, which conflated "stdio, no
    context" with "HTTP, authenticated principal with zero roles" — both produce an empty
    ``ctx_roles`` tuple — and silently let a zero-role HTTP caller through. Gate on the
    transport signal instead so an authenticated-but-role-less HTTP principal is denied."""
    if ctx_tenant.get() is None:
        return  # stdio: operator-launched, full-trust binding — no per-request context
    roles = ctx_roles.get()
    if role not in roles:
        raise UnauthorizedError(f"this tool requires the '{role}' role")
