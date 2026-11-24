"""How the vision gateway authenticates (FR-B multimodal).

Reported live: a JPG ingest failed with 401 Unauthorized against
``https://models-uat.example.internal/openai/v1/chat/completions`` while ordinary text answers
from the SAME endpoint worked. RAG_LLM_API_KEY was set; RAG_VISION_LLM_API_KEY was not, and
the vision gateway has its own key setting — so the one endpoint was called twice, once with a
credential and once without.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

from rag_clients.vision import OpenAICompatVisionGateway


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,models.sovereign.local,other.sovereign.local")
    monkeypatch.setenv("RAG_VLLM_URL", "https://models.sovereign.local/openai/v1")
    monkeypatch.setenv("RAG_LLM_API_KEY", "llm-secret")
    monkeypatch.setenv("RAG_LLM_API_KEY_HEADER", "Authorization")
    monkeypatch.setenv("RAG_VISION_LLM_API_KEY", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _auth(gw: OpenAICompatVisionGateway) -> dict[str, str]:
    return {k.lower(): v for k, v in gw._client.headers.items()}


def test_the_llm_key_is_reused_when_vision_points_at_the_same_endpoint() -> None:
    """Same host, same header, key configured once. Requiring it to be pasted into a second
    variable is a footgun that fails only on the image path, long after the model was proved
    working by every text answer."""
    gw = OpenAICompatVisionGateway(url="https://models.sovereign.local/openai/v1")
    assert _auth(gw)["authorization"] == "Bearer llm-secret"


def test_a_different_vision_endpoint_never_borrows_the_llm_key() -> None:
    """One provider's credential must never be sent to another host — the reuse is only safe
    because the endpoints are identical."""
    gw = OpenAICompatVisionGateway(url="https://other.sovereign.local/openai/v1")
    assert "authorization" not in _auth(gw)


def test_its_own_key_wins_and_is_bearer_prefixed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicitly configured vision key takes precedence, and an Authorization header gets
    the Bearer scheme the text gateway already applies — a bare token there is rejected."""
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_VISION_LLM_API_KEY", "vision-secret")
    get_settings.cache_clear()
    gw = OpenAICompatVisionGateway(url="https://other.sovereign.local/openai/v1")
    assert _auth(gw)["authorization"] == "Bearer vision-secret"


def test_a_non_authorization_header_takes_the_raw_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """AIOps-style 'apikey' headers take the token unchanged — Bearer would break them."""
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_VISION_LLM_API_KEY", "raw-secret")
    monkeypatch.setenv("RAG_VISION_LLM_API_KEY_HEADER", "apikey")
    get_settings.cache_clear()
    gw = OpenAICompatVisionGateway(url="https://other.sovereign.local/openai/v1")
    assert _auth(gw)["apikey"] == "raw-secret"


# --- the vision answer gets the same hygiene as the text answer ----------------------------------


def _reply(content: object) -> httpx.MockTransport:
    import json as _json

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    _ = _json
    return httpx.MockTransport(handler)


async def test_a_reasoning_preamble_is_stripped_from_a_caption() -> None:
    """Observed live: a scanned page ingested with the caption "[Image description] thought A
    plain white image containing three lines...". The text gateway strips chain-of-thought; the
    vision gateway returned the model's content untouched, so Gemma's planning preamble became
    part of the document's searchable text — and part of every citation quoting it."""
    gw = OpenAICompatVisionGateway(url="https://models.sovereign.local/openai/v1")
    gw._client = httpx.AsyncClient(
        base_url="http://x", transport=_reply("<think>planning</think>A diagram of two boxes.")
    )
    assert await gw.complete_with_images("p", [b"img"]) == "A diagram of two boxes."


async def test_a_null_content_is_empty_not_the_string_none() -> None:
    """A reasoning model may return content: null with its working in reasoning_content.
    str(None) is "None" — a non-empty, plausible-looking caption that would be indexed as the
    document's text. The text gateway already guards this; vision did not."""
    gw = OpenAICompatVisionGateway(url="https://models.sovereign.local/openai/v1")
    gw._client = httpx.AsyncClient(base_url="http://x", transport=_reply(None))
    assert await gw.complete_with_images("p", [b"img"]) == ""


async def test_reasoning_is_suppressed_at_the_source_for_vision_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stripping the preamble AFTER the fact does not recover the budget it consumed. Observed
    live on Gemma-31B: the same scanned page captioned fully on one call and returned six
    characters ("yours.") on the next, because the planning preamble ate max_tokens before the
    caption was written. The text gateway already suppresses this with a system turn when
    reasoning_effort is off; the vision gateway sent none, so the one path where the output IS
    the document's indexed text was the one left unprotected."""
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_LLM_REASONING_EFFORT", "off")
    get_settings.cache_clear()
    seen: dict[str, list[dict[str, object]]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen.update(_json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "a caption"}}]})

    gw = OpenAICompatVisionGateway(url="https://models.sovereign.local/openai/v1")
    gw._client = httpx.AsyncClient(base_url="http://x", transport=httpx.MockTransport(handler))
    await gw.complete_with_images("describe", [b"img"])

    roles = [m["role"] for m in seen["messages"]]
    assert roles[0] == "system", f"no reasoning-suppression system turn was sent: {roles}"
    assert roles[-1] == "user"
