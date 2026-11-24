"""Unit tests: the Chandra client + vision gateway (mock HTTP transport, no real model)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from rag_clients.chandra import ChandraClient, ChandraVisionGateway

_IMG = b"\x89PNG\r\n fake image bytes"


def _stub_transport(reply: str, *, capture: dict[str, Any] | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture["body"] = json.loads(request.content)
            capture["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    return httpx.MockTransport(handler)


def _client_with(transport: httpx.MockTransport) -> ChandraClient:
    c = ChandraClient(url="http://localhost/v1", model="chandra-test")
    c._client = httpx.AsyncClient(transport=transport, base_url="http://localhost/v1")
    return c


async def test_run_sends_image_as_data_uri_and_returns_text(inmemory: None) -> None:
    capture: dict[str, Any] = {}
    c = _client_with(_stub_transport("transcribed markdown", capture=capture))
    out = await c.run("Transcribe this.", [_IMG], mime="image/png")
    assert out == "transcribed markdown"
    # The image is sent as an OpenAI image_url content part with a base64 data URI.
    content = capture["body"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Transcribe this."}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


async def test_vision_gateway_wraps_the_client(inmemory: None) -> None:
    c = _client_with(_stub_transport("a description of the chart"))
    gw = ChandraVisionGateway(client=c)
    assert gw.model_name == "chandra-test"
    out = await gw.complete_with_images("What does this show?", [_IMG])
    assert out == "a description of the chart"


async def test_http_error_becomes_model_gateway_error(inmemory: None) -> None:
    from rag_common import ModelGatewayError

    def boom(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream down")

    c = _client_with(httpx.MockTransport(boom))
    with pytest.raises(ModelGatewayError):
        await c.run("x", [_IMG])
