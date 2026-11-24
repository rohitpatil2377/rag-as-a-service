"""Agentic-orchestration tool pack — expose the plan-execute-critique agent over MCP.

``rag.agent_run`` runs the platform's autonomous agent: an LLM plans over the in-process
tool catalog (list_kbs / retrieve / answer / sql_query / ingest), an independent critic
reviews the plan, CRITICAL (write) steps PAUSE for dual-control human approval, safe steps
run tenant-scoped + audited, and the step observations are synthesised into an answer.

Same guarantees as the REST ``/agent/run`` route it wraps (``svc_control_plane.agent_orchestrator``);
disabled unless ``RAG_AGENT_ORCHESTRATOR_ENABLED``. Human-in-the-loop is intrinsic: a run that
needs a critical action returns with a ``pending_approval_id`` instead of acting.
"""

from __future__ import annotations

from typing import Any

from svc_control_plane import wiring
from svc_control_plane.agent_orchestrator import run_agent

from rag_common import UnauthorizedError, get_settings, tenant_scope
from rag_mcp.context import current_tenant
from rag_mcp.registry import ToolRegistry


async def agent_run_tool(goal: str, max_steps: int = 6) -> dict[str, Any]:
    """Run the plan-execute-critique agent toward ``goal``. Returns the plan, the critic
    review, per-step observations, any ``pending_approval_id`` (a critical step paused for
    human sign-off), and the synthesised answer. Requires RAG_AGENT_ORCHESTRATOR_ENABLED."""
    if not get_settings().agent_orchestrator_enabled:
        raise UnauthorizedError("agent orchestrator is disabled")
    tenant = current_tenant()
    gateway = await wiring.resolve_tenant_gateway(tenant)
    with tenant_scope(tenant):
        result = await run_agent(
            goal,
            tenant_id=tenant,
            actor="mcp-agent",
            gateway=gateway,
            store=wiring.approval_store(),
            max_steps=max_steps,
        )
    return result.model_dump(mode="json")


_AGENT_TOOLS = (
    (
        "rag.agent_run",
        "Run the plan-execute-critique agent: LLM plans over platform tools, critic-reviewed, HITL-gated on writes.",
        agent_run_tool,
    ),
)


def register(registry: ToolRegistry) -> None:
    """Framework entrypoint: contribute the agent-orchestration tools to ``registry``."""
    for name, description, fn in _AGENT_TOOLS:
        registry.register(name, description, fn)
