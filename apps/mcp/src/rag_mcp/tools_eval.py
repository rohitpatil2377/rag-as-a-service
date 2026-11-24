"""Evaluation tool pack — expose the sovereign Ragas-style RAG-quality eval over MCP.

``rag.ragas_eval`` scores a knowledge base's golden set on the platform's OWN models
(self-hosted Sarvam judge + BGE-M3 embedder) — answer relevancy, context precision, and the
blind-spot metric context recall — and records the run as an EVAL_RUN audit event. It runs
the real retrieval + answer pipeline, so the numbers reflect production behaviour. Read-only
to stored data; the only side effect is the audit record. Requires the ``admin`` role.
"""

from __future__ import annotations

from typing import Any

from svc_control_plane.ragas_eval import run_ragas_eval_for_kb

from rag_common import tenant_scope
from rag_mcp.context import require_role, scoped_tenant
from rag_mcp.registry import ToolRegistry


async def ragas_eval_tool(kb_id: str | None = None) -> dict[str, Any]:
    """Run the sovereign Ragas-style quality evaluation over the KB's golden set and return
    the aggregate scores (answer_relevancy / context_precision / context_recall /
    faithfulness) plus per-question rows. ``kb_id`` targets one of the org's knowledge bases
    (default: the org's own namespace). Records an EVAL_RUN audit event. Admin only."""
    require_role("admin")
    tenant = scoped_tenant(kb_id)
    with tenant_scope(tenant):
        report = await run_ragas_eval_for_kb(tenant, actor="mcp-agent")
    return report.model_dump(mode="json")


_EVAL_TOOLS = (
    (
        "rag.ragas_eval",
        "Score a KB's golden set on our own models (Ragas-style: relevancy/precision/recall), audited.",
        ragas_eval_tool,
    ),
)


def register(registry: ToolRegistry) -> None:
    """Framework entrypoint: contribute the evaluation tools to ``registry``."""
    for name, description, fn in _EVAL_TOOLS:
        registry.register(name, description, fn)
