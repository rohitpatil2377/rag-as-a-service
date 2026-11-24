"""Agentic-workflow orchestration (FR-F, v1.1) — concrete step handlers + storage + runner.

Binds the pure runtime in :mod:`rag_core.workflow` to the platform's real capabilities: a
step can run a grounded RAG answer, a governed NL→SQL query, or a raw retrieval, or set/route on
the accumulated context via the rules engine. Because handlers call the existing services, every
workflow step inherits grounding, guardrails, permission-aware retrieval, and audit. The whole
run is recorded as a decision trace on the audit chain (``AuditAction.WORKFLOW_RUN``), so an
auditor can see exactly which branch fired and why.

Definitions live per-KB in ``tenant_config`` (JSON) — no new table, migration-free.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from rag_clients.factory import get_catalog
from rag_common import get_logger, get_settings
from rag_common.errors import ConfigError, ForbiddenError, NotFoundError, ValidationError
from rag_core.workflow import Step, WorkflowDef, WorkflowResult, run_workflow
from rag_schemas import RetrievalQuery
from rag_schemas.common import AuditAction
from svc_control_plane import pipeline, sql_query, wiring

_log = get_logger(__name__)

_WORKFLOWS_KEY = "workflows"  # tenant_config → JSON {id: WorkflowDef}

# Step types whose direct REST endpoint is capability-role-gated (e.g. POST /kb/{id}/sql/query
# requires "ingestor" — see api.py's `_require_role(principal, "ingestor")`). A workflow must
# never let the calling principal do more than they could by calling the underlying endpoint
# directly, so gated step types are re-checked here against the REAL caller's roles before the
# step's handler runs. "answer"/"retrieve" have no such gate on their direct endpoints (open to
# any authenticated tenant member, like /v1/ask), so they are intentionally absent here.
_STEP_ROLE_REQUIREMENTS: dict[str, str] = {
    "sql": "ingestor",
}


def _check_step_role(step_type: str, roles: frozenset[str]) -> None:
    required = _STEP_ROLE_REQUIREMENTS.get(step_type)
    if required is None:
        return
    if required in roles or "admin" in roles:  # admin-inclusive, matching api._require_role
        return
    raise ForbiddenError(
        f"this workflow step requires the '{required}' role", details={"step_type": step_type, "have": sorted(roles)}
    )


# --- storage -----------------------------------------------------------------
async def _load_all(kb_id: str) -> dict[str, dict[str, Any]]:
    raw = (await get_catalog().get_tenant_config(kb_id)).get(_WORKFLOWS_KEY)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


async def save(kb_id: str, defn: WorkflowDef) -> WorkflowDef:
    """Register/replace a workflow definition (validated by the WorkflowDef model)."""
    if not any(s.id == defn.start for s in defn.steps):
        raise ValidationError("start step is not defined", details={"start": defn.start})
    stored = await _load_all(kb_id)
    stored[defn.id] = defn.model_dump(mode="json")
    await get_catalog().set_tenant_config(kb_id, _WORKFLOWS_KEY, json.dumps(stored))
    _log.info("workflow_saved", kb_id=kb_id, workflow_id=defn.id, steps=len(defn.steps))
    return defn


async def list_defs(kb_id: str) -> list[WorkflowDef]:
    return [WorkflowDef(**d) for d in (await _load_all(kb_id)).values()]


async def get_def(kb_id: str, workflow_id: str) -> WorkflowDef:
    stored = await _load_all(kb_id)
    if workflow_id not in stored:
        raise NotFoundError("workflow not found", details={"workflow_id": workflow_id})
    return WorkflowDef(**stored[workflow_id])


async def delete(kb_id: str, workflow_id: str) -> None:
    stored = await _load_all(kb_id)
    if stored.pop(workflow_id, None) is not None:
        await get_catalog().set_tenant_config(kb_id, _WORKFLOWS_KEY, json.dumps(stored))


# --- step handlers (compose existing services) -------------------------------
def _resolve_text(config: dict[str, Any], context: dict[str, Any], key: str) -> str:
    """A step input is either a literal (``key``) or a reference to a context var (``key_from``)."""
    from rag_core.workflow import _MISSING, get_field  # noqa: PLC0415 — reuse the dotted resolver

    if config.get(key) is not None:
        return str(config[key])
    ref = config.get(f"{key}_from")
    if ref:
        val = get_field(context, ref)
        if val is not _MISSING:
            return str(val)
    raise ValidationError(f"step needs '{key}' or '{key}_from'", details={"config": list(config)})


def _make_handlers(kb_id: str, actor: str, groups: list[str], clearance: int, roles: frozenset[str]) -> dict[str, Any]:
    async def _answer(step: Step, context: dict[str, Any]) -> dict[str, Any]:
        _check_step_role("answer", roles)
        question = _resolve_text(step.config, context, "question")
        query = RetrievalQuery(
            tenant_id=kb_id, text=question, top_k=int(step.config.get("top_k", 5)), acl=groups, clearance=clearance
        )
        ans = await pipeline.answer_query(query, actor=actor)
        out = step.config.get("output", "answer")
        return {
            out: {
                "status": ans.status.value,
                "answer": ans.answer,
                "citations": len(ans.citations),
                "grounded": ans.status.value == "grounded",
            }
        }

    async def _sql(step: Step, context: dict[str, Any]) -> dict[str, Any]:
        _check_step_role("sql", roles)
        question = _resolve_text(step.config, context, "question")
        # A KB can wire several databases; ``config.connection`` (a connection NAME or id) targets a
        # specific one so a step isn't stuck on the KB's default. Omitted → the default, as before.
        connection_id = step.config.get("connection") or None
        if connection_id:
            match = next(
                (c for c in await sql_query.list_connections(kb_id) if connection_id in (c.conn_id, c.name)),
                None,
            )
            connection_id = match.conn_id if match else connection_id
        result = await sql_query.ask(kb_id, question, connection_id=connection_id, actor=actor)
        out = step.config.get("output", "sql_result")
        return {out: {"row_count": result.row_count, "rows": result.rows, "columns": result.columns, "sql": result.sql}}

    async def _retrieve(step: Step, context: dict[str, Any]) -> dict[str, Any]:
        _check_step_role("retrieve", roles)
        question = _resolve_text(step.config, context, "query")
        query = RetrievalQuery(
            tenant_id=kb_id, text=question, top_k=int(step.config.get("top_k", 5)), acl=groups, clearance=clearance
        )
        result = await (await wiring.retrieval_service_for(kb_id)).retrieve(query)
        out = step.config.get("output", "retrieved")
        return {out: {"hits": len(result.spans)}}

    async def _set(step: Step, context: dict[str, Any]) -> dict[str, Any]:
        return dict(step.config.get("vars", {}))

    async def _decision(step: Step, context: dict[str, Any]) -> dict[str, Any]:
        return {}  # pure routing — the rules on this step choose the branch

    async def _output(step: Step, context: dict[str, Any]) -> dict[str, Any]:
        return {}

    return {
        "answer": _answer,
        "sql": _sql,
        "retrieve": _retrieve,
        "set": _set,
        "decision": _decision,
        "output": _output,
    }


def _summarize(step_type: str, produced: dict[str, Any]) -> str:
    if not produced:
        return step_type
    key = next(iter(produced))
    val = produced[key]
    if isinstance(val, dict):
        keep = {k: val[k] for k in ("status", "grounded", "row_count", "hits") if k in val}
        return f"{key}={keep}" if keep else key
    return f"{key}={val}"


async def run(
    kb_id: str,
    workflow_id: str,
    inputs: dict[str, Any],
    *,
    actor: str,
    groups: list[str],
    clearance: int = 0,
    roles: frozenset[str] | list[str] = frozenset(),
) -> WorkflowResult:
    """Execute a registered workflow and record its decision trace on the audit chain.

    ``groups``/``clearance``/``roles`` must be the REAL calling principal's own attributes
    (forwarded by the caller from the authenticated request) — a workflow step can never be
    allowed to see more, or do more, than the person running it could by calling the underlying
    endpoint directly.
    """
    if not get_settings().agentic_workflows_enabled:
        raise ConfigError("agentic workflows are not enabled (RAG_AGENTIC_WORKFLOWS_ENABLED)")
    defn = await get_def(kb_id, workflow_id)
    context: dict[str, Any] = {"inputs": inputs, **inputs}
    handlers = _make_handlers(kb_id, actor, groups, clearance, frozenset(roles))
    result = await run_workflow(defn, handlers, context, summarize=_summarize)
    await wiring.audit_service().record(
        tenant_id=kb_id,
        trace_id=str(uuid.uuid4()),
        action=AuditAction.WORKFLOW_RUN,
        actor=actor,
        payload={
            "workflow_id": workflow_id,
            "status": result.status.value,
            "steps": [t.model_dump() for t in result.trace],
            "error": result.error,
        },
    )
    _log.info("workflow_run", kb_id=kb_id, workflow_id=workflow_id, status=result.status.value, steps=len(result.trace))
    return result
