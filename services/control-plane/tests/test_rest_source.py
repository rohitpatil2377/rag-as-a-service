"""REST-API connector — register a search endpoint, ask a question, get one evidence span.

Mirrors test_unified_kb.py's SQL-connector coverage: the connection lifecycle (set/get/clear,
never leaking the token), the fail-safe evidence-span contract (None on not-configured/error/
non-200/timeout/empty), and that the sovereignty guard is actually enforced — both at
registration and at query time. No test in this file touches the network: every HTTP call goes
through `rest_source._client_factory`, monkeypatched to build an httpx.AsyncClient over
httpx.MockTransport.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
from svc_control_plane import rest_source
from svc_control_plane.rest_source import RestConnectionInfo, render_rest_evidence, rest_evidence_span

from rag_common.config import get_settings
from rag_common.errors import SovereigntyViolationError
from rag_schemas.common import Modality


@pytest.fixture(autouse=True)
def _clean(inmemory: None) -> None:
    get_settings.cache_clear()


def _client_with(handler: Callable[[httpx.Request], httpx.Response]) -> Callable[[], httpx.AsyncClient]:
    """A `_client_factory`-shaped stub whose AsyncClient never touches the network."""

    def _factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return _factory


# --- render_rest_evidence ------------------------------------------------------------------


def test_render_rest_evidence_compact_and_truncated() -> None:
    text = render_rest_evidence({"answer": "42", "unit": "widgets"})
    assert '"answer":"42"' in text

    long_text = render_rest_evidence({"x": "y" * 5000}, max_chars=100)
    assert len(long_text) == 101  # 100 chars + the truncation marker
    assert long_text.endswith("…")


# --- connection lifecycle -------------------------------------------------------------------


async def test_set_get_clear_round_trip() -> None:
    info = await rest_source.set_rest_connection(
        "kb1", "http://localhost:9200", path="/api/search", query_param="query", token="s3cr3t"
    )
    assert info == RestConnectionInfo(
        configured=True, base_url="http://localhost:9200", path="/api/search", query_param="query", has_token=True
    )

    fetched = await rest_source.get_rest_connection("kb1")
    assert fetched == info

    await rest_source.clear_rest_connection("kb1")
    cleared = await rest_source.get_rest_connection("kb1")
    assert cleared == RestConnectionInfo(configured=False)


async def test_get_never_leaks_the_token() -> None:
    await rest_source.set_rest_connection("kb1", "http://localhost", token="super-secret-token")
    info = await rest_source.get_rest_connection("kb1")
    assert info.has_token is True
    assert not hasattr(info, "token")  # the model has no field that could carry it
    assert "super-secret-token" not in info.model_dump_json()


async def test_defaults_when_path_and_query_param_omitted() -> None:
    info = await rest_source.set_rest_connection("kb1", "http://localhost")
    assert info.path == "/search"
    assert info.query_param == "q"
    assert info.has_token is False


async def test_blank_token_on_update_keeps_existing_token() -> None:
    await rest_source.set_rest_connection("kb1", "http://localhost", token="keep-me")
    updated = await rest_source.set_rest_connection("kb1", "http://localhost", path="/v2/search")
    assert updated.has_token is True  # omitting token on update must not drop the existing one


# --- sovereignty guard ---------------------------------------------------------------------


async def test_set_rejects_non_allowlisted_host() -> None:
    with pytest.raises(SovereigntyViolationError):
        await rest_source.set_rest_connection("kb1", "https://evil.example.com")
    # And nothing was persisted by the rejected call.
    info = await rest_source.get_rest_connection("kb1")
    assert info.configured is False


async def test_span_fails_safe_when_allowlist_narrows_after_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    await rest_source.set_rest_connection("kb1", "http://localhost:9200")

    def _boom(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be called — sovereignty guard should reject before any request")

    monkeypatch.setattr(rest_source, "_client_factory", _client_with(_boom))
    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "some-other-host.local")
    get_settings.cache_clear()

    assert await rest_evidence_span("kb1", "how many widgets", actor="u") is None


# --- rest_evidence_span: fail-safe evidence contract ----------------------------------------


async def test_span_none_when_not_configured() -> None:
    assert await rest_evidence_span("unconfigured-kb", "how many widgets", actor="u") is None


async def test_span_built_from_200_json(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, httpx.Request] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(200, json={"answer": 42, "unit": "widgets"})

    await rest_source.set_rest_connection("kb1", "http://localhost:9200", path="/search", query_param="q")
    monkeypatch.setattr(rest_source, "_client_factory", _client_with(_handler))

    span = await rest_evidence_span("kb1", "how many widgets", actor="u")
    assert span is not None
    assert span.chunk_id == "rest::kb1"
    assert span.doc_id == "api://localhost"
    assert span.score == 1.0
    assert span.scorer == "exact"
    assert span.modality is Modality.API
    assert span.source_uri == "http://localhost:9200"
    assert span.category == "api"
    assert "42" in span.text

    req = seen["request"]
    assert req.url.params["q"] == "how many widgets"
    assert "Authorization" not in req.headers  # no token registered -> no bearer header


async def test_span_sends_bearer_token_when_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization", "")
        return httpx.Response(200, json={"ok": True})

    await rest_source.set_rest_connection("kb1", "http://localhost", token="tok-abc")
    monkeypatch.setattr(rest_source, "_client_factory", _client_with(_handler))

    span = await rest_evidence_span("kb1", "q", actor="u")
    assert span is not None
    assert captured["auth"] == "Bearer tok-abc"


@pytest.mark.parametrize("status", [404, 500, 503])
async def test_span_none_on_non_200(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status)

    await rest_source.set_rest_connection("kb1", "http://localhost")
    monkeypatch.setattr(rest_source, "_client_factory", _client_with(_handler))
    assert await rest_evidence_span("kb1", "q", actor="u") is None


async def test_span_none_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    await rest_source.set_rest_connection("kb1", "http://localhost")
    monkeypatch.setattr(rest_source, "_client_factory", _client_with(_handler))
    assert await rest_evidence_span("kb1", "q", actor="u") is None


async def test_span_none_on_empty_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    await rest_source.set_rest_connection("kb1", "http://localhost")
    monkeypatch.setattr(rest_source, "_client_factory", _client_with(_handler))
    assert await rest_evidence_span("kb1", "q", actor="u") is None


async def test_span_none_on_empty_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    await rest_source.set_rest_connection("kb1", "http://localhost")
    monkeypatch.setattr(rest_source, "_client_factory", _client_with(_handler))
    assert await rest_evidence_span("kb1", "q", actor="u") is None


async def test_span_none_on_oversized_response(monkeypatch: pytest.MonkeyPatch) -> None:
    huge = json.dumps({"data": "x" * (rest_source._MAX_RESPONSE_BYTES + 1)}).encode()

    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=huge)

    await rest_source.set_rest_connection("kb1", "http://localhost")
    monkeypatch.setattr(rest_source, "_client_factory", _client_with(_handler))
    assert await rest_evidence_span("kb1", "q", actor="u") is None


async def test_span_never_raises_on_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _handler(_request: httpx.Request) -> httpx.Response:
        raise RuntimeError("boom")

    await rest_source.set_rest_connection("kb1", "http://localhost")
    monkeypatch.setattr(rest_source, "_client_factory", _client_with(_handler))
    assert await rest_evidence_span("kb1", "q", actor="u") is None
