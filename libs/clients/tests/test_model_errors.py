"""Unit tests: raw model-endpoint failures become actionable ModelGatewayErrors.

The user must never see a raw ``httpx.HTTPStatusError`` with the internal URL — a
misconfigured model connection should read as "check Models", not a stack trace.
"""

from __future__ import annotations

import httpx

from rag_clients.llm_gateway import _raise_clean
from rag_common import ModelGatewayError


def _status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://models-dev.example.internal/openai/v1/chat/completions")
    response = httpx.Response(code, request=request)
    return httpx.HTTPStatusError(f"{code}", request=request, response=response)


def test_404_points_at_url_or_model() -> None:
    err = _raise_clean(_status_error(404), "https://models-dev.example.internal/openai/v1", "qwen")
    assert isinstance(err, ModelGatewayError)
    assert "404" in err.message
    assert "Models" in err.message
    # The clean message must not leak the full completions URL.
    assert "chat/completions" not in err.message


def test_401_points_at_key() -> None:
    err = _raise_clean(_status_error(401), "https://x/v1", "m")
    assert "key" in err.message.lower()


def test_connection_error_is_actionable() -> None:
    err = _raise_clean(httpx.ConnectError("refused"), "https://x/v1", "m")
    assert isinstance(err, ModelGatewayError)
    assert "reach" in err.message.lower()
