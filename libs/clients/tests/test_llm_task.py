"""``complete_for_task``: the one seam that tells a gateway what a call is FOR.

The policy is a CAPABILITY, not a Protocol method, because `LLMGateway.complete` has two real
implementations, an in-memory twin and ~32 test stubs — and a keyword on the Protocol would oblige
every one of them to accept a policy only one adapter can act on. These tests pin both halves of
that: an adapter that advertises the capability is told the task, and anything else is called
exactly as it is called today.
"""

from __future__ import annotations

from typing import Any

from rag_clients.llm_task import complete_for_task
from rag_core import sampling


class _Plain:
    """A conforming gateway with NOTHING but `complete` — the shape most test stubs have, and the
    reason this helper exists rather than a Protocol parameter. Passing it `task=` would be a
    TypeError, which is exactly the regression this guards."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> str:
        self.calls.append({"max_tokens": max_tokens, "temperature": temperature})
        return "plain"

    # The rest of the Protocol, so this really is a CONFORMING gateway rather than a duck that
    # happens to have `complete` -- which is the whole claim these tests make.
    def stream(self, prompt: str, **_: Any) -> Any:  # pragma: no cover - unused by this helper
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        return "plain-test-model"

    async def list_models(self) -> list[str]:
        return ["plain-test-model"]


class _Metered(_Plain):
    """Reports usage but does not implement the sampling policy."""

    async def complete_with_usage(self, prompt: str, **kw: Any) -> tuple[str, dict[str, Any]]:
        self.calls.append(dict(kw))
        return "metered", {"completion_tokens": 7, "finish_reason": "stop"}


class _PolicyAware(_Metered):
    """What `OpenAICompatGateway` is: usage AND the per-task sampling policy."""

    honours_sampling_task = True


async def test_a_policy_aware_gateway_is_told_the_task() -> None:
    gw = _PolicyAware()
    reply, usage = await complete_for_task(gw, "p", max_tokens=11, temperature=0.0, task=sampling.STRUCTURED_EXTRACTION)

    assert reply == "metered" and usage["completion_tokens"] == 7
    assert gw.calls[-1]["task"] == sampling.STRUCTURED_EXTRACTION
    assert gw.calls[-1]["temperature"] == 0.0


async def test_a_gateway_that_does_not_advertise_the_policy_is_never_handed_a_task() -> None:
    """Not a cosmetic check: `task=` to a gateway whose signature has no such parameter is a
    TypeError at ingest time, and the sampling policy must never be the reason a document fails."""
    for gw in (_Plain(), _Metered()):
        reply, usage = await complete_for_task(
            gw, "p", max_tokens=11, temperature=0.0, task=sampling.STRUCTURED_EXTRACTION
        )
        assert "task" not in gw.calls[-1]
        assert reply in ("plain", "metered")
        assert usage == {} if reply == "plain" else usage["completion_tokens"] == 7


async def test_usage_is_empty_rather_than_absent_for_a_gateway_that_cannot_report_it() -> None:
    reply, usage = await complete_for_task(_Plain(), "p", task=sampling.GENERATION)
    assert reply == "plain" and usage == {}


async def test_a_gateway_error_is_not_swallowed_here() -> None:
    """The stages that call this have their own fail-safes -- the enricher scopes a failure to one
    section, the profiler to one document -- and a rescue here would hide the error from both."""

    class _Broken(_Plain):
        async def complete(self, prompt: str, **kw: Any) -> str:
            raise RuntimeError("endpoint down")

    try:
        await complete_for_task(_Broken(), "p", task=sampling.STRUCTURED_EXTRACTION)
    except RuntimeError as exc:
        assert "endpoint down" in str(exc)
    else:  # pragma: no cover - the assertion above is the test
        raise AssertionError("the gateway error was swallowed")
