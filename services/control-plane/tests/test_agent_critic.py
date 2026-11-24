from __future__ import annotations

from svc_control_plane.agent_critic import build_review_prompt, parse_review, review_plan
from svc_control_plane.agent_types import Plan, PlanStep, Review


def _plan() -> Plan:
    return Plan(
        goal="Rotate the tenant API key",
        steps=[
            PlanStep(tool="list_api_keys", args={}, rationale="find the current key"),
            PlanStep(tool="delete_api_key", args={"key_id": "*"}, rationale="revoke every key"),
        ],
    )


def test_build_review_prompt_contains_goal_and_steps() -> None:
    p = build_review_prompt(_plan())
    assert "STRICT JSON" in p
    assert "Rotate the tenant API key" in p
    assert "delete_api_key" in p


def test_parse_review_defaults_and_coercion() -> None:
    # approved defaults True when absent; concerns/suggestion coerced from loose shapes.
    review = parse_review({"concerns": "single concern", "suggestion": None})
    assert review.approved is True
    assert review.concerns == ["single concern"]
    assert review.suggestion == ""


# --- reviewer (fake gateways) ---------------------------------------------------------


class _Gateway:
    def __init__(self, out: str) -> None:
        self._out = out

    async def complete(self, prompt: str, *, max_tokens: int | None = None) -> str:
        return self._out


async def test_review_plan_approves() -> None:
    gw = _Gateway('{"approved":true,"concerns":[],"suggestion":""}')
    review = await review_plan(_plan(), gateway=gw)
    assert isinstance(review, Review)
    assert review.approved is True
    assert review.concerns == []


async def test_review_plan_rejects_with_concerns() -> None:
    gw = _Gateway(
        '{"approved":false,"concerns":["delete_api_key with key_id=* revokes every key with no '
        'confirmation step"],"suggestion":"scope key_id to the single key being rotated and add a '
        'confirmation step"}'
    )
    review = await review_plan(_plan(), gateway=gw)
    assert review.approved is False
    assert review.concerns == ["delete_api_key with key_id=* revokes every key with no confirmation step"]
    assert "confirmation" in review.suggestion


async def test_review_plan_parses_fenced_json() -> None:
    gw = _Gateway('```json\n{"approved": true, "concerns": [], "suggestion": ""}\n```')
    review = await review_plan(_plan(), gateway=gw)
    assert review.approved is True


async def test_review_plan_gateway_error_fails_open() -> None:
    class _Boom:
        async def complete(self, prompt: str, *, max_tokens: int | None = None) -> str:
            raise RuntimeError("model down")

    review = await review_plan(_plan(), gateway=_Boom())
    assert review.approved is True
    assert "critic unavailable" in review.concerns


async def test_review_plan_malformed_json_fails_open() -> None:
    gw = _Gateway("sorry, I cannot produce JSON for this request")
    review = await review_plan(_plan(), gateway=gw)
    assert review.approved is True
