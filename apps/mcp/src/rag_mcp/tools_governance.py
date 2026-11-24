"""Built-in governance tool pack (registered through the plugin framework).

These tools expose EXISTING control-plane governance surfaces to agents over MCP —
no new pipeline behaviour, just agent-callable read access to what the console already
shows. Each runs under the same per-request tenancy as the core tools, so an agent only
ever sees its own organisation's posture:

* ``rag.attestation`` — sovereignty/residency + protection report for a KB (H8).
* ``rag.usage``       — usage counts + indicative showback over a window (H4).
* ``rag.list_grants`` — Federated RAG federation grants issued/received (needs federation).

They are read-only and additive: enabling them changes no stored state and no default
behaviour of the query path.
"""

from __future__ import annotations

from typing import Any

from svc_control_plane import wiring
from svc_control_plane.compliance import attestation
from svc_control_plane.usage import usage_report

from rag_common import UnauthorizedError, get_settings, tenant_scope
from rag_mcp.context import current_tenant, scoped_tenant
from rag_mcp.registry import ToolRegistry


async def attestation_tool(kb_id: str | None = None) -> dict[str, Any]:
    """Sovereignty & data-protection attestation for a knowledge base: egress posture,
    model origins, encryption, audit-chain integrity, and retention. ``kb_id`` targets one
    of the org's KBs (default: the org's own namespace)."""
    tenant = scoped_tenant(kb_id)
    with tenant_scope(tenant):
        report = await attestation(tenant)
    return report.model_dump(mode="json")


async def usage_tool(window: str = "30d", kb_id: str | None = None) -> dict[str, Any]:
    """Usage metering + indicative showback over ``window`` (one of ``24h``/``7d``/``30d``/
    ``all``): query/ingest/abstain/guardrail counts and a nominal cost breakdown."""
    tenant = scoped_tenant(kb_id)
    with tenant_scope(tenant):
        report = await usage_report(tenant, window)
    return report.model_dump(mode="json")


async def list_grants_tool() -> dict[str, Any]:
    """List the Federated RAG federation grants this organisation has ISSUED to peers and
    RECEIVED from them (grant id, parties, scope, validity, revocation). Requires
    ``RAG_FEDERATION_ENABLED``."""
    if not get_settings().federation_enabled:
        raise UnauthorizedError("federation is not enabled")
    org = current_tenant()
    store = wiring.grant_store()
    issued = await store.list_issued(org)
    received = await store.list_received(org)
    return {
        "issued": [g.model_dump(mode="json") for g in issued],
        "received": [g.model_dump(mode="json") for g in received],
    }


# (name, description, fn) — the governance pack surface.
_GOVERNANCE_TOOLS = (
    ("rag.attestation", "Sovereignty & data-protection attestation report for a knowledge base.", attestation_tool),
    ("rag.usage", "Usage metering + indicative showback over a time window.", usage_tool),
    ("rag.list_grants", "List Federated RAG federation grants issued/received (needs federation).", list_grants_tool),
)


def register(registry: ToolRegistry) -> None:
    """Framework entrypoint: contribute the governance tools to ``registry``."""
    for name, description, fn in _GOVERNANCE_TOOLS:
        registry.register(name, description, fn)
