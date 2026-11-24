"""Agent critic — an independent review of a :class:`Plan` before it runs.

The critic role of the agent orchestrator: multi-agent reasoning/collaboration where the planner
proposes a plan and a second, independent LLM call checks it for unsafe or irreversible steps,
missing steps, and wrong tool choices before execution. Pure + gateway-injected (any object with
``async complete(prompt, *, max_tokens) -> str``) so it is testable without a model.

FAIL-OPEN BY DESIGN: a critic error or a non-JSON response yields an APPROVED :class:`Review`
flagging the outage, never an exception and never a block. The action guard (dual-control on
CRITICAL steps) is the hard safety gate; this critic is advisory, and an outage of the critic must
never itself stop legitimate work.
"""

from __future__ import annotations

import json
from typing import Any

from svc_control_plane.agent_types import Plan, Review


def build_review_prompt(plan: Plan) -> str:
    """Prompt an LLM to review ``plan`` (goal + steps) as STRICT JSON before it is executed."""
    body = json.dumps(plan.model_dump(), separators=(",", ":"))
    return (
        "You are a critic reviewing an AI agent's PLAN before it is allowed to run — an independent "
        "check for unsafe or irreversible actions, missing steps, and wrong tool choices. Set "
        "approved=false ONLY for a plan that is genuinely unsafe, irreversible, or incorrect; a plan "
        "that is merely suboptimal or could be phrased differently should still be approved. Output "
        "STRICT JSON with this shape:\n"
        '{"approved": true/false, "concerns": ["..."], "suggestion": "..."}\n'
        "concerns MUST be concrete and specific to this plan (name the step and the risk); leave it "
        'empty if there are none. suggestion is how to fix the plan when not approved, else "". '
        "Output ONLY the JSON.\n\nPLAN:\n" + body
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


def parse_review(data: dict[str, Any]) -> Review:
    """Build a validated :class:`Review` from loosely-typed critic output. ``approved`` defaults to
    True when absent (fail-open at the parse layer too, not just on a gateway error); ``concerns`` is
    coerced to a ``list[str]`` (a bare string becomes a single-item list), and ``suggestion`` to
    ``str``."""
    raw_approved = data.get("approved", True)
    if isinstance(raw_approved, str):
        approved = raw_approved.strip().lower() not in ("false", "no", "0", "")
    else:
        approved = bool(raw_approved)

    raw_concerns = data.get("concerns", [])
    concerns: list[str] = []
    if isinstance(raw_concerns, list):
        concerns = [str(c).strip() for c in raw_concerns if str(c).strip()]
    elif isinstance(raw_concerns, str) and raw_concerns.strip():
        concerns = [raw_concerns.strip()]

    suggestion = str(data.get("suggestion", "") or "").strip()

    return Review(approved=approved, concerns=concerns, suggestion=suggestion)


async def review_plan(plan: Plan, *, gateway: Any, max_tokens: int = 600) -> Review:
    """Review ``plan`` via the tenant's LLM ``gateway`` — the critic half of planner/critic
    multi-agent reasoning. FAIL-OPEN: a gateway error yields ``Review(approved=True,
    concerns=["critic unavailable"])`` rather than propagating, so a critic outage never blocks
    legitimate work; the action guard remains the hard safety gate."""
    try:
        raw = await gateway.complete(build_review_prompt(plan), max_tokens=max_tokens)
    except Exception:
        return Review(approved=True, concerns=["critic unavailable"])
    return parse_review(_extract_json(raw))
