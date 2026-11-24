"""Agent orchestrator (Agentic RAG with MCP + AI agents) — plan, review, gate, execute.

Ties the role modules into one dynamic agent run:

1. **Planner** (:mod:`agent_planner`) turns a goal into an ordered plan of tool calls, chosen from
   the tool catalog (the same capabilities the MCP server exposes, callable in-process here so the
   run is tenant-scoped and audited).
2. **Critic** (:mod:`agent_critic`) reviews the plan before anything runs — multi-agent reasoning;
   a blocked plan halts (advisory: fail-open).
3. **Action guard** (:mod:`agent_guard`) gates every CRITICAL step behind a dual-control human
   approval; the run pauses on the approval id (the hard safety gate: fail-closed).
4. The executor runs each safe step against its in-process tool, threading the observations as the
   run's contextual memory, then synthesises the final answer.

The whole run is recorded as an ``AGENT_RUN`` audit event; each critical-action approval is its own
dual-control APPROVAL. Tools are INJECTABLE (``run_agent(tools=...)``) so the loop is testable
without the real services; :func:`default_tools` binds the live read tools + one gated write tool.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from rag_common import get_logger
from rag_schemas.common import AuditAction
from svc_control_plane import agent_critic, agent_guard, agent_planner
from svc_control_plane.agent_types import AgentRunResult, StepObservation

_log = get_logger(__name__)

# A tool the agent may call: (tenant_id, actor, args) -> JSON-serialisable result.
ToolFn = Callable[[str, str, dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True)
class ToolSpec:
    """One agent tool: a human ``description`` (shown to the planner) + the async ``fn`` to run."""

    description: str
    fn: ToolFn


# --- built-in tools (in-process, tenant-scoped) --------------------------------------


async def _list_kbs(tenant_id: str, _actor: str, _args: dict[str, Any]) -> Any:
    from svc_control_plane import wiring  # noqa: PLC0415

    return [{"kb_id": k.kb_id, "name": k.name} for k in await wiring.kb_store().list(tenant_id)]


async def _retrieve(tenant_id: str, _actor: str, args: dict[str, Any]) -> Any:
    from rag_schemas import RetrievalQuery  # noqa: PLC0415
    from svc_control_plane import wiring  # noqa: PLC0415

    kb = str(args.get("kb_id") or tenant_id)
    svc = await wiring.retrieval_service_for(kb)
    res = await svc.retrieve(RetrievalQuery(tenant_id=kb, text=str(args.get("query", "")), top_k=5, acl=[]))
    return [{"doc_id": s.doc_id, "text": s.text[:400]} for s in res.spans]


async def _answer(tenant_id: str, actor: str, args: dict[str, Any]) -> Any:
    from rag_schemas import RetrievalQuery  # noqa: PLC0415
    from svc_control_plane import pipeline  # noqa: PLC0415

    kb = str(args.get("kb_id") or tenant_id)
    ans = await pipeline.answer_query(
        RetrievalQuery(tenant_id=kb, text=str(args.get("query", "")), top_k=5, acl=[]), actor=actor
    )
    return {"answer": ans.answer, "status": ans.status.value, "citations": len(ans.citations)}


async def _sql_query(tenant_id: str, actor: str, args: dict[str, Any]) -> Any:
    from svc_control_plane import sql_query  # noqa: PLC0415

    kb = str(args.get("kb_id") or tenant_id)
    ans = await sql_query.ask(kb, str(args.get("question", "")), actor=actor)
    return {"sql": ans.sql, "row_count": ans.row_count, "rows": ans.rows[:10]}


async def _ingest_document(_tenant_id: str, _actor: str, _args: dict[str, Any]) -> Any:
    # CRITICAL + gated: the agent always PAUSES for human approval before this can run (its name
    # trips agent_guard). Executing an approved write step (build the DocumentEnvelope + resume the
    # run) is a documented follow-up; today the human-approval gate itself is the deliverable, so
    # reaching execution here is not expected in a normal run.
    raise NotImplementedError("ingest is a gated write; approved-run resume is a follow-up")


def default_tools() -> dict[str, ToolSpec]:
    """The live tool catalog: read tools run immediately; ``ingest_document`` is CRITICAL (its name
    trips the action guard) so it is gated behind human approval before it can run."""
    return {
        "list_kbs": ToolSpec("List the knowledge bases available to the tenant.", _list_kbs),
        "retrieve": ToolSpec("Retrieve passages for a query. args: {query, kb_id?}.", _retrieve),
        "answer": ToolSpec("Answer a question with grounded RAG. args: {query, kb_id?}.", _answer),
        "sql_query": ToolSpec("Ask an NL question over a connected DB. args: {question, kb_id?}.", _sql_query),
        "ingest_document": ToolSpec("Ingest a document (WRITE). args: {kb_id, doc_id, text}.", _ingest_document),
    }


# --- synthesis + audit ---------------------------------------------------------------


async def _synthesize(goal: str, observations: list[StepObservation], *, gateway: Any) -> str:
    """Final answer from the goal + the tool observations (the run's contextual memory). Fail-safe:
    on a model error, fall back to a compact rendering of the results."""
    facts = "\n".join(
        f"- {o.tool}: {'OK ' + str(o.result)[:400] if o.ok else 'ERROR ' + o.error}" for o in observations
    )
    prompt = (
        f"You are an assistant completing a goal using tool results.\nGoal: {goal}\n\nTool results:\n{facts}\n\n"
        "Write the final answer to the goal, grounded ONLY in the results above. Be concise."
    )
    try:
        answer: str = (await gateway.complete(prompt, max_tokens=600)).strip()
    except Exception:
        return f"Completed {sum(o.ok for o in observations)}/{len(observations)} step(s) for: {goal}."
    return answer


async def _audit_run(tenant_id: str, actor: str, result: AgentRunResult) -> None:
    import uuid  # noqa: PLC0415

    from svc_control_plane import wiring  # noqa: PLC0415

    try:
        await wiring.audit_service().record(
            tenant_id=tenant_id,
            trace_id=str(uuid.uuid4()),
            action=AuditAction.AGENT_RUN,
            actor=actor,
            payload={
                "goal": result.goal,
                "steps": [s.tool for s in result.plan.steps],
                "ran": [o.tool for o in result.observations if o.ok],
                "halted_reason": result.halted_reason,
                "pending_approval_id": result.pending_approval_id,
            },
        )
    except Exception as exc:
        _log.warning("agent_run_audit_failed", tenant_id=tenant_id, error=str(exc)[:160])


# --- the run loop --------------------------------------------------------------------


async def run_agent(
    goal: str,
    *,
    tenant_id: str,
    actor: str,
    gateway: Any,
    store: Any,
    tools: dict[str, ToolSpec] | None = None,
    max_steps: int = 6,
    run_critic: bool = True,
) -> AgentRunResult:
    """Plan → critic review → for each step: approval-gate then execute → synthesise. The run PAUSES
    (returns with ``pending_approval_id``) the moment a critical step needs human approval; resume it
    once the dual-control request is approved. Every failure mode is captured in ``halted_reason``."""
    tools = tools if tools is not None else default_tools()
    catalog = [{"name": name, "description": spec.description} for name, spec in tools.items()]

    plan = await agent_planner.plan(goal, catalog, gateway=gateway)
    result = AgentRunResult(goal=goal, plan=plan)
    if not plan.steps:
        result.halted_reason = "no_plan"
        await _audit_run(tenant_id, actor, result)
        return result

    if run_critic:
        result.review = await agent_critic.review_plan(plan, gateway=gateway)
        if not result.review.approved:
            result.halted_reason = "critic_blocked"
            await _audit_run(tenant_id, actor, result)
            return result

    observations: list[StepObservation] = []
    for step in plan.steps[:max_steps]:
        if step.tool not in tools:
            observations.append(StepObservation(tool=step.tool, ok=False, error="unknown tool"))
            continue
        # Hard safety gate: a critical tool call needs human approval BEFORE it runs.
        try:
            approval_id = await agent_guard.gate_step(step, actor=actor, tenant_id=tenant_id, store=store)
        except Exception as exc:
            observations.append(StepObservation(tool=step.tool, ok=False, error=f"blocked: {str(exc)[:200]}"))
            result.halted_reason = "approval_error"
            break
        if approval_id is not None:
            observations.append(
                StepObservation(tool=step.tool, ok=False, approval_id=approval_id, error="awaiting human approval")
            )
            result.pending_approval_id = approval_id
            result.halted_reason = "awaiting_approval"
            break
        # Safe step: run it, observe the result (contextual memory for later steps).
        try:
            out = await tools[step.tool].fn(tenant_id, actor, step.args)
            observations.append(StepObservation(tool=step.tool, ok=True, result=out))
        except Exception as exc:
            observations.append(StepObservation(tool=step.tool, ok=False, error=str(exc)[:300]))

    result.observations = observations
    if not result.pending_approval_id and result.halted_reason == "":
        result.answer = await _synthesize(goal, observations, gateway=gateway)
    await _audit_run(tenant_id, actor, result)
    return result
