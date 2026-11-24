"""Pure workflow runtime: rules engine + deterministic executor + decision trace."""

from __future__ import annotations

from typing import Any

from rag_core.workflow import (
    Condition,
    Operator,
    Rule,
    Step,
    WorkflowDef,
    WorkflowStatus,
    eval_condition,
    next_step,
    run_workflow,
)


def test_condition_operators() -> None:
    ctx = {"a": {"status": "grounded", "n": 3}, "tags": ["x", "y"]}
    assert eval_condition(Condition(field="a.status", op=Operator.EQ, value="grounded"), ctx)
    assert eval_condition(Condition(field="a.n", op=Operator.GTE, value=3), ctx)
    assert not eval_condition(Condition(field="a.n", op=Operator.LT, value=3), ctx)
    assert eval_condition(Condition(field="tags", op=Operator.CONTAINS, value="x"), ctx)
    assert eval_condition(Condition(field="a.status", op=Operator.IN, value=["grounded", "flagged"]), ctx)
    assert eval_condition(Condition(field="a.missing", op=Operator.NOT_EXISTS), ctx)
    assert eval_condition(Condition(field="a.status", op=Operator.MATCHES, value="^ground"), ctx)


def test_missing_field_is_false_not_error() -> None:
    assert not eval_condition(Condition(field="nope.deep", op=Operator.EQ, value=1), {})
    assert not eval_condition(Condition(field="a", op=Operator.GT, value="notcomparable"), {"a": 5})


def test_next_step_first_matching_rule_wins() -> None:
    step = Step(
        id="s",
        type="decision",
        rules=[
            Rule(all=[Condition(field="score", op=Operator.GTE, value=0.8)], goto="high"),
            Rule(all=[Condition(field="score", op=Operator.LT, value=0.8)], goto="low"),
        ],
        next="fallback",
    )
    assert next_step(step, {"score": 0.9}) == ("high", True)
    assert next_step(step, {"score": 0.1}) == ("low", True)
    assert next_step(step, {}) == ("fallback", False)  # no rule matched → fallback, not a rule


async def _handlers() -> dict[str, Any]:
    async def _set(step: Step, ctx: dict[str, Any]) -> dict[str, Any]:
        return dict(step.config.get("vars", {}))

    async def _noop(step: Step, ctx: dict[str, Any]) -> dict[str, Any]:
        return {}

    return {"set": _set, "decision": _noop, "output": _noop}


async def test_run_follows_branch() -> None:
    defn = WorkflowDef(
        id="wf",
        name="branch",
        start="init",
        steps=[
            Step(id="init", type="set", config={"vars": {"score": 0.9}}, next="route"),
            Step(
                id="route",
                type="decision",
                rules=[Rule(all=[Condition(field="score", op=Operator.GTE, value=0.8)], goto="win")],
                next="lose",
            ),
            Step(id="win", type="output"),
            Step(id="lose", type="output"),
        ],
    )
    result = await run_workflow(defn, await _handlers(), {})
    assert result.status == WorkflowStatus.COMPLETED
    assert [t.step_id for t in result.trace] == ["init", "route", "win"]
    assert result.trace[1].matched_rule is True


async def test_max_steps_guard() -> None:
    defn = WorkflowDef(
        id="loop",
        name="loop",
        start="a",
        max_steps=5,
        steps=[Step(id="a", type="decision", next="a")],  # deliberate infinite loop
    )
    result = await run_workflow(defn, await _handlers(), {})
    assert result.status == WorkflowStatus.MAX_STEPS
    assert len(result.trace) == 5


async def test_handler_error_ends_run() -> None:
    async def _boom(step: Step, ctx: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("kaboom")

    defn = WorkflowDef(id="e", name="e", start="a", steps=[Step(id="a", type="boom")])
    result = await run_workflow(defn, {"boom": _boom}, {})
    assert result.status == WorkflowStatus.ERROR
    assert "kaboom" in (result.error or "")


async def test_unknown_step_type_is_handled() -> None:
    defn = WorkflowDef(id="u", name="u", start="a", steps=[Step(id="a", type="nonexistent")])
    result = await run_workflow(defn, {}, {})
    assert result.status == WorkflowStatus.ERROR
    assert "no handler" in (result.error or "")
