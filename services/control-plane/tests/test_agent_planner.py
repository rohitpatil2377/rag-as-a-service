"""Agent planner — prompt build, tolerant JSON parsing, fail-safe empty plan on error."""

from __future__ import annotations

from svc_control_plane.agent_planner import build_plan_prompt, parse_plan, plan
from svc_control_plane.agent_types import Plan

_CATALOG = [
    {"name": "search_kb", "description": "Search a knowledge base."},
    {"name": "send_email", "description": "Send an email."},
]


class _Gateway:
    def __init__(self, out: str) -> None:
        self._out = out

    async def complete(self, prompt: str, *, max_tokens: int | None = None) -> str:
        return self._out


def test_build_plan_prompt_lists_tools_and_goal() -> None:
    prompt = build_plan_prompt("find the invoice", _CATALOG)
    assert "find the invoice" in prompt
    assert "search_kb" in prompt
    assert "send_email" in prompt
    assert "STRICT JSON" in prompt


def test_parse_plan_keeps_steps_in_order_with_args() -> None:
    data = {
        "steps": [
            {"tool": "search_kb", "args": {"query": "invoice"}, "rationale": "find it"},
            {"tool": "send_email", "args": {"to": "a@b.com"}, "rationale": "notify"},
        ]
    }
    result = parse_plan(data, "goal", allowed_tools={"search_kb", "send_email"})
    assert result.goal == "goal"
    assert [s.tool for s in result.steps] == ["search_kb", "send_email"]
    assert result.steps[0].args == {"query": "invoice"}
    assert result.steps[1].rationale == "notify"


def test_parse_plan_drops_step_with_tool_not_in_catalog() -> None:
    data = {
        "steps": [
            {"tool": "search_kb", "args": {}, "rationale": "ok"},
            {"tool": "delete_everything", "args": {}, "rationale": "not allowed"},
        ]
    }
    result = parse_plan(data, "goal", allowed_tools={"search_kb", "send_email"})
    assert [s.tool for s in result.steps] == ["search_kb"]


async def test_plan_parses_fenced_json() -> None:
    gw = _Gateway('```json\n{"steps":[{"tool":"search_kb","args":{"query":"q"},"rationale":"r"}]}\n```')
    result = await plan("find q", _CATALOG, gateway=gw)
    assert [s.tool for s in result.steps] == ["search_kb"]
    assert result.steps[0].args == {"query": "q"}


async def test_plan_failsafe_on_gateway_error() -> None:
    class _Boom:
        async def complete(self, prompt: str, *, max_tokens: int | None = None) -> str:
            raise RuntimeError("model down")

    result = await plan("goal", _CATALOG, gateway=_Boom())
    assert result == Plan(goal="goal")


async def test_plan_failsafe_on_non_json() -> None:
    result = await plan("goal", _CATALOG, gateway=_Gateway("sorry, I cannot help"))
    assert result == Plan(goal="goal")
    assert result.steps == []
