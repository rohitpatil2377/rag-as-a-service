"""Agent planner — goal + MCP tool catalog -> an ordered :class:`Plan` of tool calls.

The planner role of the agent orchestrator: an LLM is prompted with the goal and the available
tools, asked to output a STRICT JSON plan, and the response is tolerantly parsed back into a typed
:class:`Plan`. Pure + gateway-injected (any object with ``async complete(prompt, *, max_tokens) ->
str``) so it is testable without a model. Fail-safe: an error or a non-JSON response yields an
EMPTY plan, never an exception — planning degrades, the request lives.
"""

from __future__ import annotations

import json
from typing import Any

from svc_control_plane.agent_types import Plan, PlanStep


def build_plan_prompt(goal: str, tool_catalog: list[dict[str, str]]) -> str:
    """Prompt an LLM to plan tool calls for ``goal`` as STRICT JSON, choosing only from
    ``tool_catalog`` (each entry a ``{"name", "description"}`` pair)."""
    tools = "\n".join(f"- {t.get('name', '')}: {t.get('description', '')}" for t in tool_catalog)
    return (
        "You are a planning agent. Given a GOAL and a TOOL CATALOG, produce a plan to accomplish "
        "the goal as STRICT JSON with this shape:\n"
        '{"steps": [{"tool": "<name>", "args": {...}, "rationale": "..."}]}\n'
        "Choose tools ONLY from the TOOL CATALOG below (use the exact name). Use the minimal number "
        "of steps needed. args MUST match what the chosen tool expects. Output ONLY the JSON.\n\n"
        f"GOAL:\n{goal}\n\nTOOL CATALOG:\n{tools}"
    )


def _extract_json(raw: str) -> dict[str, Any]:
    """Best-effort JSON object from a model response — tolerates ```json fences and prose around the
    object by taking the outermost ``{ ... }``. Returns ``{}`` on anything unparseable."""
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        obj = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return {}
    return obj if isinstance(obj, dict) else {}


def parse_plan(data: dict[str, Any], goal: str, *, allowed_tools: set[str]) -> Plan:
    """Build a validated :class:`Plan` from loosely-typed planner output. DROPS any step whose
    ``tool`` is not in ``allowed_tools`` (never invents or keeps a call to an unknown tool), coerces
    ``args`` to a ``dict`` and ``rationale`` to ``str``, and keeps step order."""
    steps: list[PlanStep] = []
    for raw in data.get("steps", []) or []:
        if not isinstance(raw, dict):
            continue
        tool = str(raw.get("tool", ""))
        if tool not in allowed_tools:
            continue
        args = raw.get("args", {})
        if not isinstance(args, dict):
            args = {}
        rationale = str(raw.get("rationale", ""))
        steps.append(PlanStep(tool=tool, args=args, rationale=rationale))
    return Plan(goal=goal, steps=steps)


async def plan(goal: str, tool_catalog: list[dict[str, str]], *, gateway: Any, max_tokens: int = 1000) -> Plan:
    """Plan tool calls for ``goal`` over ``tool_catalog`` via the tenant's LLM ``gateway``. Fail-safe:
    a gateway error or a non-JSON response yields an EMPTY :class:`Plan`, never an exception."""
    try:
        raw = await gateway.complete(build_plan_prompt(goal, tool_catalog), max_tokens=max_tokens)
    except Exception:
        return Plan(goal=goal)
    allowed_tools = {t["name"] for t in tool_catalog}
    return parse_plan(_extract_json(raw), goal, allowed_tools=allowed_tools)
