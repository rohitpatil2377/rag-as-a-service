"""Decoding controls on the OpenAI-compatible gateway."""

from __future__ import annotations

import pytest


def test_greedy_with_reasoning_suppressed_is_nudged_off_zero_for_any_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loop comes from decoding greedily with no reasoning pass, whatever the model is called.

    Gating the nudge on one deployment id ("/model") left every other model -- gemma4-31b here --
    running greedy with RAG_LLM_REASONING_EFFORT=off: observed live emitting a page of its own
    deliberation, a repeated-word run, and this prompt's own "1,25,49,23,246" formatting example
    presented as an answer."""
    from rag_clients.llm_gateway import OpenAICompatGateway
    from rag_common.config import get_settings

    def _gateway(effort: str) -> OpenAICompatGateway:
        # localhost is on the default egress allowlist; the guard runs in the constructor and this
        # test never makes a call.
        monkeypatch.setenv("RAG_LLM_REASONING_EFFORT", effort)
        get_settings.cache_clear()
        return OpenAICompatGateway(url="http://localhost:9/v1", model="gemma4-31b")

    try:
        off = _gateway("off")
        assert off._eff_temp(0.0) == 0.6
        assert off._eff_temp(0.3) == 0.3, "an explicit temperature is still honoured"

        reasoning = _gateway("medium")
        assert reasoning._eff_temp(0.0) == 0.0, "a model that DOES reason is left greedy"
    finally:
        get_settings.cache_clear()


def test_a_read_timeout_is_not_reported_as_an_unreachable_endpoint() -> None:
    """A client-side read timeout and a dead endpoint used to produce the same sentence.

    102 map calls timed out at exactly the configured 120 s during a token-cap experiment and were
    first read as the endpoint flapping; the endpoint was healthy the whole time and the limit was
    ours. The message must name the limit, not send the operator to check a URL that is fine."""
    import httpx

    from rag_clients.llm_gateway import _raise_clean

    err = _raise_clean(
        httpx.ReadTimeout("timed out"),
        "https://model.example/v1",
        "gemma4-31b",
        timeout=120.0,
        elapsed=120.09,
    )
    assert err.details["failure_kind"] == "read_timeout"
    assert err.details["timeout_s"] == 120.0
    assert err.details["elapsed_s"] == 120.09
    assert err.details["exc_type"] == "ReadTimeout"
    assert "120s" in str(err) and "120.1s" in str(err)
    assert "RAG_LLM_TIMEOUT_SECONDS" in str(err)
    assert "reachable" not in str(err).lower().replace("the endpoint is reachable", "")

    dead = _raise_clean(httpx.ConnectError("refused"), "https://model.example/v1", "gemma4-31b")
    assert dead.details["failure_kind"] == "unreachable"
    assert "Check the URL" in str(dead)


def test_the_request_timeout_comes_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cap on max_tokens is really a cap on generation TIME, so the timeout has to be tunable
    without a code change: ~38 tok/s means a 4,000-token budget needs ~105 s, which had only ~15 s
    of margin under the old 120 s default and lost 8 of 54 map calls in one measured run."""
    from rag_clients.llm_gateway import OpenAICompatGateway
    from rag_common.config import get_settings

    try:
        get_settings.cache_clear()
        assert OpenAICompatGateway(url="http://localhost:9/v1", model="m")._timeout == 240.0, (
            "the shipped default: 120 s did not cover a 4,000-token generation reliably"
        )

        monkeypatch.setenv("RAG_LLM_TIMEOUT_SECONDS", "360")
        get_settings.cache_clear()
        assert OpenAICompatGateway(url="http://localhost:9/v1", model="m")._timeout == 360.0
        assert OpenAICompatGateway(url="http://localhost:9/v1", model="m", timeout=30.0)._timeout == 30.0
    finally:
        get_settings.cache_clear()


def test_structured_extraction_keeps_the_temperature_it_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    """The nudge above is right for generation and was wrong for extraction, silently.

    A facet id is Phase 8's facet-union join key and its label is a cross-encoder probe, so
    re-rolling the Phase-3 profile re-rolls section selection -- and three ingests of identical
    bytes produced three 10-facet vocabularies sharing ONE id. The nudge was in force on every one
    of those calls while the caller asked for greedy. See `rag_core.sampling`."""
    from rag_clients.llm_gateway import OpenAICompatGateway
    from rag_common.config import get_settings
    from rag_core import sampling

    monkeypatch.setenv("RAG_LLM_REASONING_EFFORT", "off")
    get_settings.cache_clear()
    try:
        gw = OpenAICompatGateway(url="http://localhost:9/v1", model="gemma4-31b")
        assert gw._eff_temp(0.0, sampling.STRUCTURED_EXTRACTION) == 0.0
        assert gw._eff_temp(0.0, sampling.GENERATION) == 0.6, "generation keeps the nudge"
        assert gw._eff_temp(0.0) == 0.6, "and so does a caller that never classified itself"
        assert gw._eff_temp(0.4, sampling.STRUCTURED_EXTRACTION) == 0.4
        assert gw._eff_temp(0.0, "a task this version does not know") == 0.6, (
            "an unknown task takes the conservative branch -- today's behaviour -- rather than raising"
        )
    finally:
        get_settings.cache_clear()


async def test_the_requested_temperature_is_what_the_request_body_carries(monkeypatch: pytest.MonkeyPatch) -> None:
    """THE ACCEPTANCE TEST FOR THE POLICY, and it is about the WIRE rather than the return value.

    The defect was a contract violation at this boundary: the caller asked for 0.0, believed it had
    asked for greedy, and the endpoint received 0.6. Only the request body can show that, which is
    why this asserts on the body and not on what comes back. It deliberately does NOT assert that
    two replies are byte-identical -- greedy sampling is not determinism (measured: at 0.0 two of
    three extractions of one section were distinct), and a test that demanded it would be
    asserting something the endpoint does not promise."""
    import httpx

    from rag_clients.llm_gateway import OpenAICompatGateway
    from rag_common.config import get_settings
    from rag_core import sampling

    monkeypatch.setenv("RAG_LLM_REASONING_EFFORT", "off")
    get_settings.cache_clear()
    sent: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    try:
        gw = OpenAICompatGateway(url="http://localhost:9/v1", model="gemma4-31b")
        await gw._client.aclose()
        gw._client = httpx.AsyncClient(base_url="http://localhost:9/v1", transport=httpx.MockTransport(handler))

        await gw.complete("p", max_tokens=99, temperature=0.0, task=sampling.STRUCTURED_EXTRACTION)
        await gw.complete_with_usage("p", max_tokens=99, temperature=0.0, task=sampling.STRUCTURED_EXTRACTION)
        await gw.complete("p", max_tokens=99, temperature=0.0)

        assert [b["temperature"] for b in sent] == [0.0, 0.0, 0.6]
        assert {b["max_tokens"] for b in sent} == {99}, "the budget is untouched by the policy"
        assert all("task" not in b for b in sent), "the task is a local policy, never sent to the server"
        await gw._client.aclose()
    finally:
        get_settings.cache_clear()
