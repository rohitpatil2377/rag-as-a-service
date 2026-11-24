"""OpenAI-compatible embedder adapter (per-KB embedding attach).

Drives a stubbed ``/embeddings`` endpoint — no model, no network. The contracts under
test: the API key travels RAW in the configured header (never with a Bearer prefix),
vectors come back in ``index`` order regardless of response order, and a response of
the wrong width is refused loudly (the KB's vector collection was created against the
stored dimension, so a warning would let a mismatched vector poison it).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator

import httpx
import pytest

from rag_clients.openai_embedder import OpenAICompatEmbedder, probe_embedding_dim
from rag_common import ConfigError, ModelGatewayError, SovereigntyViolationError


@pytest.fixture(autouse=True)
def _allowlist_the_endpoint(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,models.sovereign.local")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


_URL = "https://models.sovereign.local/openai/v1"


def _transport(dim: int = 4, *, seen: list[httpx.Request] | None = None, shuffle: bool = False) -> httpx.MockTransport:
    """An /embeddings endpoint returning ``dim``-wide vectors whose first component encodes
    the input's batch position — so ordering bugs change values, not just metadata."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        inputs = json.loads(request.content)["input"]
        data = [{"index": i, "embedding": [float(i)] + [0.0] * (dim - 1)} for i in range(len(inputs))]
        if shuffle:
            data = list(reversed(data))
        return httpx.Response(200, json={"object": "list", "data": data})

    return httpx.MockTransport(handler)


def _embedder(
    dim: int = 4, *, transport: httpx.MockTransport, key: str = "", header: str = "apikey"
) -> OpenAICompatEmbedder:
    e = OpenAICompatEmbedder(_URL, "bge-m3", dim=dim, api_key=key, api_key_header=header)
    e._client = httpx.AsyncClient(base_url=_URL, transport=transport, headers={header: key} if key else {})
    return e


def test_the_key_travels_raw_in_the_configured_header() -> None:
    """AIOps authenticates with ``apikey: <key>`` — no Bearer prefix, no Authorization."""
    seen: list[httpx.Request] = []
    e = _embedder(transport=_transport(seen=seen), key="model-secret", header="apikey")
    asyncio.run(e.embed(["hello"]))
    assert seen[0].headers["apikey"] == "model-secret"
    assert "authorization" not in seen[0].headers


def test_vectors_come_back_in_index_order_not_response_order() -> None:
    e = _embedder(transport=_transport(shuffle=True))
    out = asyncio.run(e.embed(["a", "b", "c"]))
    assert [v[0] for v in out] == [0.0, 1.0, 2.0]


def test_batches_beyond_the_server_limit_are_all_embedded() -> None:
    seen: list[httpx.Request] = []
    e = _embedder(transport=_transport(seen=seen))
    out = asyncio.run(e.embed([f"text {i}" for i in range(70)]))
    assert len(out) == 70
    assert len(seen) == 3  # 32 + 32 + 6


def test_a_wrong_width_response_is_refused_not_warned() -> None:
    """The stored dimension is the contract the Qdrant collection was created against."""
    e = _embedder(dim=1024, transport=_transport(dim=4))
    with pytest.raises(ConfigError, match="1024"):
        asyncio.run(e.embed(["hello"]))


def test_an_unreachable_endpoint_names_the_models_page() -> None:
    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    e = _embedder(transport=httpx.MockTransport(dead))
    with pytest.raises(ModelGatewayError, match="Models"):
        asyncio.run(e.embed(["hello"]))


def test_a_foreign_endpoint_is_refused_before_any_call() -> None:
    with pytest.raises(SovereigntyViolationError):
        OpenAICompatEmbedder("https://api.openai.com/v1", "text-embedding-3-small", dim=1536)


def test_the_probe_reports_the_dimension_the_endpoint_actually_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Test button's contract: ask the running server, never trust a form field."""
    transport = _transport(dim=1024)
    real_client = httpx.AsyncClient

    class PatchedClient(real_client):  # type: ignore[valid-type,misc]
        def __init__(self, **kw: object) -> None:
            kw["transport"] = transport
            super().__init__(**kw)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedClient)
    assert asyncio.run(probe_embedding_dim(_URL, "bge-m3", api_key="k")) == 1024
