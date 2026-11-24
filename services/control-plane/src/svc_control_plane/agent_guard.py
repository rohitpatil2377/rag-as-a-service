"""Action guard for the agent orchestrator: classify a planned tool call as safe or CRITICAL,
and gate CRITICAL steps behind human approval (dual-control / maker-checker) before the executor
runs them.

Fail-closed: :func:`gate_step` never lets a critical step run un-gated. If opening the approval
request raises, the exception propagates (it is NOT swallowed) — the caller must treat any
exception from :func:`gate_step` as "blocked, do not execute this step". Only a clean ``None``
return means "safe to run immediately"; any non-``None`` return is an approval id the executor
must pause on, and any exception means the step must not run at all.
"""

from __future__ import annotations

from typing import Any

from svc_control_plane import dual_control
from svc_control_plane.agent_types import PlanStep

# Substrings (case-insensitive) that mark a tool as mutating/egress — and therefore CRITICAL.
CRITICAL_TOOL_MARKERS: tuple[str, ...] = (
    "ingest",
    "delete",
    "write",
    "update",
    "create",
    "execute",
    "send",
    "approve",
    "provision",
    "erase",
    "revoke",
    "grant",
)

# dual_control.Action is a closed Literal (publish_kb/ingest_classified/purge/erasure/security_change);
# arbitrary tool names can't be used as the Action value, so every critical agent tool call is filed
# under this generic bucket. The concrete tool + args are preserved in the approval's ``params``.
_GENERIC_CRITICAL_ACTION: dual_control.Action = "security_change"


def tool_criticality(tool_name: str) -> str:
    """ "critical" if ``tool_name`` contains any :data:`CRITICAL_TOOL_MARKERS` substring, else "safe"."""
    lowered = tool_name.lower()
    if any(marker in lowered for marker in CRITICAL_TOOL_MARKERS):
        return "critical"
    return "safe"


async def gate_step(
    step: PlanStep,
    *,
    actor: str,
    tenant_id: str,
    store: dual_control.ApprovalStore,
) -> str | None:
    """Gate ``step`` if its tool is CRITICAL. Returns the pending approval request id (the
    executor must pause on it) or ``None`` if the step is safe to run immediately.

    Fail-closed: this never returns normally without either gating (non-``None``) or clearing
    (``None``) the step. If :func:`dual_control.submit` raises, the exception propagates —
    callers must treat that as "blocked", not as permission to run the step.
    """
    if tool_criticality(step.tool) != "critical":
        return None

    params: dict[str, Any] = {"tool": step.tool, "args": step.args}
    req = await dual_control.submit(
        store,
        action=_GENERIC_CRITICAL_ACTION,
        tenant_id=tenant_id,
        kb_id=None,
        params=params,
        summary=f"Critical agent tool call: {step.tool}",
        requested_by=actor,
        reason=step.rationale,
    )
    return req.id
