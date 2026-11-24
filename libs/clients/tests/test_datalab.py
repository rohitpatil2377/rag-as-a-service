"""Unit tests: the Datalab hosted /convert client (submit → poll), mock transport, no network."""

from __future__ import annotations

import httpx
import pytest

from rag_clients.datalab import DatalabClient, DatalabVisionGateway

_IMG = b"\x89PNG\r\n fake image bytes"


def _submit_then_complete(markdown: str) -> httpx.MockTransport:
    """First call (POST /convert) returns a check url; the poll (GET) returns complete."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                200,
                json={"success": True, "error": None, "request_check_url": "http://localhost/api/v1/convert/abc"},
            )
        return httpx.Response(200, json={"status": "complete", "markdown": markdown})

    return httpx.MockTransport(handler)


def _client_with(transport: httpx.MockTransport) -> DatalabClient:
    c = DatalabClient(url="http://localhost/api/v1", poll_seconds=0.0)  # poll_seconds 0 → fast test
    c._client = httpx.AsyncClient(transport=transport)
    return c


async def test_convert_submits_then_polls_to_completion(inmemory: None) -> None:
    c = _client_with(_submit_then_complete("# Invoice\n\n| A | B |\n|---|---|\n| 1 | 2 |"))
    out = await c.convert(_IMG, "image/png")
    assert "Invoice" in out and "| A | B |" in out


async def test_vision_gateway_returns_the_transcription(inmemory: None) -> None:
    c = _client_with(_submit_then_complete("chart shows Q3 revenue 42 crore"))
    gw = DatalabVisionGateway(client=c)
    assert gw.model_name == "datalab-chandra"
    out = await gw.complete_with_images("What does this show?", [_IMG])
    assert "42 crore" in out


async def test_submit_rejection_raises(inmemory: None) -> None:
    from rag_common import ModelGatewayError

    def reject(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"success": False, "error": "unsupported file"})

    c = _client_with(httpx.MockTransport(reject))
    with pytest.raises(ModelGatewayError):
        await c.convert(_IMG, "image/png")


async def test_failed_status_raises(inmemory: None) -> None:
    from rag_common import ModelGatewayError

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"success": True, "request_check_url": "http://localhost/c/1"})
        return httpx.Response(200, json={"status": "failed", "error": "bad page"})

    c = _client_with(httpx.MockTransport(handler))
    with pytest.raises(ModelGatewayError):
        await c.convert(_IMG, "image/png")
