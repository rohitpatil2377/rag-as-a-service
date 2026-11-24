"""bge-m3 multi-vector embedder adapter (Slice 2b).

Drives a stubbed ``/embed`` endpoint — no model, no network. Contracts under test: dense is parsed
1024-wide, sparse comes back as an int-keyed weight map and ColBERT as a list of per-token vectors;
the key travels as ``Authorization: Bearer <key>`` (NOT the raw apikey header); vectors are ordered
by ``index`` not response order; a wrong dense width is refused loudly; a foreign host is rejected
before any call; and ``embed()`` returns dense only (drop-in Embedder). Dense-space equivalence
(``/embed`` dense == ``/embeddings`` dense) is asserted structurally against a twin here — the live
cosine≈1 check is the separate integration gate.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator

import httpx
import pytest

from rag_clients.bge_m3_embedder import BgeM3MultiEmbedder
from rag_common import ConfigError, ModelGatewayError, SovereigntyViolationError
from rag_core import MultiVectorEmbedder


@pytest.fixture(autouse=True)
def _allowlist_the_endpoint(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,models.sovereign.local")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


_URL = "https://models.sovereign.local/openai/v1"


def _transport(dim: int = 4, *, seen: list[httpx.Request] | None = None, shuffle: bool = False) -> httpx.MockTransport:
    """An /embed endpoint returning dense/sparse/colbert per requested flag. The dense first
    component encodes the input's batch position, so an ordering bug changes values."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        body = json.loads(request.content)
        inputs = body["input"]
        data = []
        for i in range(len(inputs)):
            row: dict[str, object] = {"index": i}
            if body.get("dense"):
                row["dense"] = [float(i)] + [0.0] * (dim - 1)
            if body.get("sparse"):
                row["sparse"] = {"101": 0.5, "202": 0.25}
            if body.get("colbert"):
                row["colbert"] = [[float(i)] + [0.0] * (dim - 1), [0.1] * dim]
            data.append(row)
        if shuffle:
            data = list(reversed(data))
        return httpx.Response(200, json={"object": "list", "data": data})

    return httpx.MockTransport(handler)


def _embedder(
    dim: int = 4, *, transport: httpx.MockTransport, key: str = "", auth: str = "bearer"
) -> BgeM3MultiEmbedder:
    e = BgeM3MultiEmbedder(_URL, "bge-m3", dim=dim, api_key=key, auth=auth)
    headers = {}
    if key:
        headers = {"Authorization": f"Bearer {key}"} if auth == "bearer" else {"apikey": key}
    e._client = httpx.AsyncClient(base_url=_URL, transport=transport, headers=headers)
    return e


def test_it_is_a_multivector_embedder() -> None:
    e = _embedder(transport=_transport())
    assert isinstance(e, MultiVectorEmbedder)


def test_the_key_travels_as_a_bearer_token() -> None:
    seen: list[httpx.Request] = []
    e = _embedder(transport=_transport(seen=seen), key="model-secret", auth="bearer")
    asyncio.run(e.embed(["hello"]))
    assert seen[0].headers["authorization"] == "Bearer model-secret"
    assert "apikey" not in seen[0].headers


def test_embed_returns_dense_only() -> None:
    e = _embedder(dim=1024, transport=_transport(dim=1024))
    out = asyncio.run(e.embed(["a", "b"]))
    assert len(out) == 2
    assert len(out[0]) == 1024


def test_embed_multi_parses_all_three_signals() -> None:
    e = _embedder(dim=1024, transport=_transport(dim=1024))
    out = asyncio.run(e.embed_multi(["a"]))
    mv = out[0]
    assert len(mv.dense) == 1024
    assert mv.sparse == {101: 0.5, 202: 0.25}
    assert len(mv.colbert) == 2
    assert all(len(tok) == 1024 for tok in mv.colbert)


def test_embed_multi_honours_requested_signals() -> None:
    e = _embedder(transport=_transport())
    out = asyncio.run(e.embed_multi(["a"], dense=True, sparse=False, colbert=False))
    assert out[0].dense and out[0].sparse == {} and out[0].colbert == []


def test_dense_is_the_same_from_embed_and_embed_multi() -> None:
    """Dense-space equivalence (structural twin of the live cosine≈1 gate): the dense vector the
    drop-in ``embed`` returns is the SAME one ``embed_multi`` carries, so nothing about attaching
    the multivector adapter moves the dense space the abstain threshold was calibrated against."""
    e = _embedder(dim=1024, transport=_transport(dim=1024))
    dense_only = asyncio.run(e.embed(["hello"]))[0]
    multi = asyncio.run(e.embed_multi(["hello"]))[0]
    assert dense_only == multi.dense


def test_vectors_come_back_in_index_order_not_response_order() -> None:
    e = _embedder(transport=_transport(shuffle=True))
    out = asyncio.run(e.embed(["a", "b", "c"]))
    assert [v[0] for v in out] == [0.0, 1.0, 2.0]


def test_a_wrong_width_dense_response_is_refused_not_warned() -> None:
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
        BgeM3MultiEmbedder("https://api.openai.com/v1", "bge-m3", dim=1024)
