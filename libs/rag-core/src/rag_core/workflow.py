"""Agentic-workflow runtime — deterministic orchestration + rules engine + decision trace.

This is the "explainable agent" the regulated buyers ask for: NOT a black-box LLM agent that
decides its own control flow, but a DECLARATIVE workflow of steps whose branching is decided by
a transparent rules engine over the run's accumulated context, and whose every step is recorded
in an ordered decision trace. The same inputs always produce the same path — auditable and
reproducible.

This module is pure: it defines the workflow/rule/trace data model, the rules engine, and an
executor that drives steps via INJECTED async handlers. The concrete handlers (RAG answer,
governed SQL, retrieval, …) live in the control plane and compose the existing services, so a
workflow step inherits grounding, guardrails, and audit for free. Loops are bounded by
``max_steps`` so a mis-authored workflow can never run away.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Operator(StrEnum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    CONTAINS = "contains"
    IN = "in"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    MATCHES = "matches"  # regex


class Condition(BaseModel):
    field: str = Field(description="Dotted path into the run context, e.g. 'answer.status'.")
    op: Operator
    value: Any = None


class Rule(BaseModel):
    """A branch: if ALL of ``all`` and ANY of ``any`` hold, control jumps to ``goto``."""

    all: list[Condition] = Field(default_factory=list)
    any: list[Condition] = Field(default_factory=list)
    goto: str


class Step(BaseModel):
    id: str
    type: str = Field(description="Handler name, e.g. 'answer' | 'sql' | 'set' | 'decision' | 'output'.")
    config: dict[str, Any] = Field(default_factory=dict)
    rules: list[Rule] = Field(default_factory=list, description="Evaluated in order; first match wins.")
    next: str | None = Field(default=None, description="Fallback next step when no rule matches.")


class WorkflowDef(BaseModel):
    id: str
    name: str
    start: str
    steps: list[Step]
    max_steps: int = Field(default=50, ge=1, le=500)


class StepTrace(BaseModel):
    step_id: str
    type: str
    summary: str = ""
    goto: str | None = None
    matched_rule: bool = False


class WorkflowStatus(StrEnum):
    COMPLETED = "completed"
    MAX_STEPS = "max_steps_exceeded"
    ERROR = "error"


class WorkflowResult(BaseModel):
    workflow_id: str
    status: WorkflowStatus
    trace: list[StepTrace] = Field(default_factory=list)
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


# --- rules engine (pure) -----------------------------------------------------
_MISSING = object()


def get_field(context: dict[str, Any], dotted: str) -> Any:
    """Resolve a dotted path into nested dicts; returns ``_MISSING`` if absent."""
    cur: Any = context
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return _MISSING
    return cur


# Binary operators as (actual, value) → bool. EXISTS/NOT_EXISTS handled separately (missing check).
_BINARY_OPS: dict[Operator, Callable[[Any, Any], bool]] = {
    Operator.EQ: lambda a, v: bool(a == v),
    Operator.NE: lambda a, v: bool(a != v),
    Operator.GT: lambda a, v: bool(a > v),
    Operator.GTE: lambda a, v: bool(a >= v),
    Operator.LT: lambda a, v: bool(a < v),
    Operator.LTE: lambda a, v: bool(a <= v),
    Operator.CONTAINS: lambda a, v: bool(v in a),
    Operator.IN: lambda a, v: bool(a in v),
    Operator.MATCHES: lambda a, v: re.search(str(v), str(a)) is not None,
}


def eval_condition(cond: Condition, context: dict[str, Any]) -> bool:
    actual = get_field(context, cond.field)
    if cond.op == Operator.EXISTS:
        return actual is not _MISSING
    if cond.op == Operator.NOT_EXISTS:
        return actual is _MISSING
    if actual is _MISSING:
        return False
    try:
        return _BINARY_OPS[cond.op](actual, cond.value)
    except TypeError:
        return False  # incomparable types → condition simply does not hold


def eval_rule(rule: Rule, context: dict[str, Any]) -> bool:
    all_ok = all(eval_condition(c, context) for c in rule.all)
    any_ok = not rule.any or any(eval_condition(c, context) for c in rule.any)
    return all_ok and any_ok


def next_step(step: Step, context: dict[str, Any]) -> tuple[str | None, bool]:
    """The next step id and whether a rule (vs the fallback) chose it."""
    for rule in step.rules:
        if eval_rule(rule, context):
            return rule.goto, True
    return step.next, False


# --- executor (pure control flow; async handlers injected) -------------------
StepHandler = Callable[[Step, dict[str, Any]], Awaitable[dict[str, Any]]]


async def run_workflow(
    defn: WorkflowDef,
    handlers: dict[str, StepHandler],
    context: dict[str, Any],
    *,
    summarize: Callable[[str, dict[str, Any]], str] | None = None,
) -> WorkflowResult:
    """Drive the workflow from ``defn.start``, following rule/next control flow, until a step
    has no successor or ``max_steps`` is hit. ``context`` is mutated in place with each step's
    output. Handlers do the I/O; this function owns only the deterministic control flow."""
    by_id = {s.id: s for s in defn.steps}
    trace: list[StepTrace] = []
    current: str | None = defn.start
    steps_run = 0

    while current is not None:
        if steps_run >= defn.max_steps:
            return WorkflowResult(workflow_id=defn.id, status=WorkflowStatus.MAX_STEPS, trace=trace, output=context)
        step = by_id.get(current)
        if step is None:
            return WorkflowResult(
                workflow_id=defn.id,
                status=WorkflowStatus.ERROR,
                trace=trace,
                output=context,
                error=f"unknown step '{current}'",
            )
        handler = handlers.get(step.type)
        if handler is None:
            return WorkflowResult(
                workflow_id=defn.id,
                status=WorkflowStatus.ERROR,
                trace=trace,
                output=context,
                error=f"no handler for step type '{step.type}'",
            )
        try:
            produced = await handler(step, context)
        except Exception as exc:  # a failing step ends the run with a recorded error
            trace.append(StepTrace(step_id=step.id, type=step.type, summary=f"error: {str(exc)[:160]}"))
            return WorkflowResult(
                workflow_id=defn.id, status=WorkflowStatus.ERROR, trace=trace, output=context, error=str(exc)[:200]
            )
        context.update(produced)
        goto, matched = next_step(step, context)
        summary = summarize(step.type, produced) if summarize else ""
        trace.append(StepTrace(step_id=step.id, type=step.type, summary=summary, goto=goto, matched_rule=matched))
        steps_run += 1
        current = goto

    return WorkflowResult(workflow_id=defn.id, status=WorkflowStatus.COMPLETED, trace=trace, output=context)
