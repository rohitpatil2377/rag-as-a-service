"""Cross-encoder reranking (FR-D2).

The reranker decides FINAL order and the confidence the abstain gate reads, so it has to
score relevance — not vocabulary overlap. The lexical reranker cannot pass
``test_semantic_match_outranks_a_lexical_decoy`` by construction: the decoy shares more
query words than the correct passage does.

The unit tests drive a stubbed cross-encoder so they run with no model. The live test
against a real container is marked `integration`.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator

import httpx
import pytest

from rag_clients.reranker import (
    CrossEncoderReranker,
    LexicalReranker,
    OpenAICompatReranker,
    build_reranker,
    parse_api_shape,
    probe_rerank_shape,
)
from rag_common import ConfigError
from rag_schemas import RetrievedSpan


@pytest.fixture(autouse=True)
def _allowlist_the_reranker(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The adapter refuses a non-allowlisted endpoint before making any call (sovereignty
    guard), so the stubbed host must be allowlisted exactly as the real one is in compose."""
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,reranker")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# The question shares almost no vocabulary with the passage that answers it.
_QUERY = "How long before I can take time off after joining?"
_CORRECT = "A new joiner becomes eligible for annual leave once the probation period of six months ends."
_DECOY = "Time off the premises is not permitted during a shift; joining another team requires approval."


def _spans() -> list[RetrievedSpan]:
    """Candidates as the retriever hands them over: unscored, carrying only their fusion rank.
    The decoy fused first — which is exactly the ordering the reranker has to overturn."""
    return [
        RetrievedSpan(chunk_id="decoy", doc_id="d1", text=_DECOY, fusion_score=1.0),
        RetrievedSpan(chunk_id="correct", doc_id="d1", text=_CORRECT, fusion_score=0.4),
    ]


_CONFIGURED_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _stub_transport(scores: dict[str, float], *, serves: str = _CONFIGURED_MODEL) -> httpx.MockTransport:
    """A cross-encoder that scores by chunk content, not by word overlap. ``serves`` is the
    model the endpoint reports on /info — the adapter checks it before trusting any score."""

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        if request.url.path == "/info":
            return httpx.Response(200, json={"model_id": serves})
        body = json.loads(request.content)
        out = [{"index": i, "score": scores[t[:20]]} for i, t in enumerate(body["texts"])]
        out.sort(key=lambda r: r["score"], reverse=True)
        return httpx.Response(200, json=out)

    return httpx.MockTransport(handler)


def _reranker(scores: dict[str, float], *, serves: str = _CONFIGURED_MODEL, **kw: object) -> CrossEncoderReranker:
    r = CrossEncoderReranker(url="http://reranker:80", model=_CONFIGURED_MODEL, **kw)  # type: ignore[arg-type]
    r._client = httpx.AsyncClient(base_url="http://reranker:80", transport=_stub_transport(scores, serves=serves))
    return r


_SCORES = {_CORRECT[:20]: 0.91, _DECOY[:20]: 0.02}


# --- the case the lexical reranker cannot pass ----------------------------------------------


def test_the_lexical_reranker_gets_this_wrong() -> None:
    """Pinned so the failure being fixed is visible, not asserted from memory: the decoy shares
    'time off' and 'joining' with the query; the correct passage shares almost nothing."""
    out = asyncio.run(LexicalReranker().rerank(_QUERY, _spans(), 2))
    assert out[0].chunk_id == "decoy"


def test_semantic_match_outranks_a_lexical_decoy() -> None:
    out = asyncio.run(_reranker(_SCORES).rerank(_QUERY, _spans(), 2))
    assert [s.chunk_id for s in out] == ["correct", "decoy"]
    assert out[0].score == pytest.approx(0.91)


# --- what the score means downstream ---------------------------------------------------------


def test_the_fusion_score_survives_untouched_beside_the_relevance_score() -> None:
    """Two-stage: RRF answers 'which candidates', the cross-encoder answers 'how relevant'.
    Both are kept because they are different questions — in different fields, so neither can
    be mistaken for the other, and the gate reads only the second."""
    out = asyncio.run(_reranker(_SCORES).rerank(_QUERY, _spans(), 2))
    by_id = {s.chunk_id: s for s in out}
    assert by_id["correct"].score == pytest.approx(0.91)  # relevance, absolute
    assert by_id["correct"].scorer == "cross_encoder"  # ...and it says who judged it
    assert by_id["correct"].fusion_score == pytest.approx(0.4)  # rank fusion, set-relative
    assert by_id["decoy"].fusion_score == pytest.approx(1.0)  # the loser keeps its rank too


def test_a_query_of_only_stopwords_leaves_the_spans_unscored() -> None:
    """`rerank` short-circuits when the query has no scoreable tokens. It must not hand back
    spans that look scored: whatever they arrived carrying is a fusion value, and the top one
    is 1.0 by construction. Unscored is the only honest answer, and it fails safe at the gate."""
    spans = [RetrievedSpan(chunk_id="c", doc_id="d", text=_CORRECT, fusion_score=1.0)]
    out = asyncio.run(LexicalReranker().rerank("...", spans, 5))
    assert [s.score for s in out] == [0.0]
    assert [s.scorer for s in out] == ["unscored"]


def test_the_score_is_not_renormalised_within_the_result_set() -> None:
    """min-max normalisation would force the top span to exactly 1.0 for EVERY query, making a
    threshold against it meaningless. The cross-encoder score must survive unscaled."""
    weak = {_CORRECT[:20]: 0.11, _DECOY[:20]: 0.02}
    out = asyncio.run(_reranker(weak).rerank(_QUERY, _spans(), 2))
    assert out[0].score == pytest.approx(0.11)  # NOT 1.0
    assert max(s.score for s in out) < 0.5  # a whole result set can be uniformly weak


def test_a_logit_returning_server_is_refused_by_name() -> None:
    """TEI sigmoids by default, but `raw_scores=true`, a version change or a different server
    (Infinity, vLLM /score) returns unbounded logits. A threshold calibrated on probabilities
    means nothing against those, and left alone the first such value would surface as a bare
    pydantic error and 500 every query."""
    out_of_range = {_CORRECT[:20]: 7.4, _DECOY[:20]: -3.1}
    with pytest.raises(ConfigError, match="probability"):
        asyncio.run(_reranker(out_of_range).rerank(_QUERY, _spans(), 2))


def test_candidates_beyond_the_server_batch_limit_are_all_scored() -> None:
    """TEI caps a request at 32 texts; the candidate pool is larger. Batching must not drop
    candidates or order them within-batch only."""
    texts = {f"chunk number {i:03d}"[:20]: i / 100.0 for i in range(50)}
    spans = [
        RetrievedSpan(chunk_id=f"c{i}", doc_id="d", text=f"chunk number {i:03d}", score=0.5, scorer="cross_encoder")
        for i in range(50)
    ]
    out = asyncio.run(_reranker(texts, batch_size=8).rerank("q", spans, 3))
    assert [s.chunk_id for s in out] == ["c49", "c48", "c47"]  # global order, not per-batch


# --- the score has to come from the model the threshold was calibrated against ----------------


def test_a_different_served_model_is_refused() -> None:
    """RAG_ABSTAIN_THRESHOLD is a number about a specific score distribution. Swapping the
    reranker container — switching to the GPU tier is a one-line compose change — moves the
    distribution while the threshold stays put, so the gate silently means something else.
    The endpoint declares what it serves; a mismatch fails closed."""
    with pytest.raises(ConfigError, match="bge-reranker-v2-m3"):
        asyncio.run(_reranker(_SCORES, serves="BAAI/bge-reranker-v2-m3").rerank(_QUERY, _spans(), 2))


def test_the_served_model_is_matched_case_insensitively() -> None:
    """Hugging Face ids are case-sensitive; operators are not. Refusing on case alone would be
    a self-inflicted outage, not a safety property."""
    out = asyncio.run(_reranker(_SCORES, serves=_CONFIGURED_MODEL.upper()).rerank(_QUERY, _spans(), 2))
    assert out[0].chunk_id == "correct"


def test_the_model_is_verified_once_not_per_query() -> None:
    """The check is lazy because the constructor cannot await, but it must not add a round trip
    to every query."""
    r = _reranker(_SCORES)
    calls: list[str] = []
    inner = r._client
    original = inner.send

    async def counting(request: httpx.Request, **kw: object) -> httpx.Response:
        calls.append(request.url.path)
        return await original(request, **kw)  # type: ignore[arg-type]

    inner.send = counting  # type: ignore[method-assign]
    asyncio.run(r.rerank(_QUERY, _spans(), 2))
    asyncio.run(r.rerank(_QUERY, _spans(), 2))
    assert calls.count("/info") == 1


# --- selection is configuration, never the environment ----------------------------------------


def test_the_backend_is_chosen_by_config_not_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The old factory returned the lexical reranker whenever env != prod, so no environment
    a developer could run ever exercised a real model."""
    from rag_common.config import get_settings

    for env in ("dev", "staging", "prod"):
        monkeypatch.setenv("RAG_ENV", env)
        monkeypatch.setenv("RAG_RERANK_BACKEND", "cross_encoder")
        get_settings.cache_clear()
        assert isinstance(build_reranker(), CrossEncoderReranker), env

    monkeypatch.setenv("RAG_RERANK_BACKEND", "lexical")
    get_settings.cache_clear()
    assert isinstance(build_reranker(), LexicalReranker)
    get_settings.cache_clear()


def test_there_is_no_silently_disabled_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    """`none` used to be an accepted value that `build_reranker` did not handle, so an operator
    who asked for no reranker silently got a cross-encoder. It is now rejected at startup: with
    nothing writing `score`, every span would reach the gate unscored and the KB would abstain
    on every query — a mode that must be impossible to select by accident."""
    import pydantic

    from rag_common.config import Settings, get_settings

    monkeypatch.setenv("RAG_RERANK_BACKEND", "none")
    get_settings.cache_clear()
    with pytest.raises(pydantic.ValidationError, match="cross_encoder"):
        Settings()
    get_settings.cache_clear()


def test_an_unreachable_reranker_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never a silent downgrade to word counting: the caller gets an error naming the endpoint.

    502, not 500 — a permitted upstream that is down is not our bug, and it is the same shape
    ConnectorError and ModelGatewayError already use. A misconfiguration (wrong model, scores
    off-scale) stays a 500, because that one IS the deployment's fault."""
    from rag_common import RetrievalError

    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    r = CrossEncoderReranker(url="http://reranker:80")
    r._client = httpx.AsyncClient(base_url="http://reranker:80", transport=httpx.MockTransport(dead))
    with pytest.raises(RetrievalError, match="reranker") as raised:
        asyncio.run(r.rerank(_QUERY, _spans(), 2))
    assert raised.value.http_status == 502


# --- against the real model -------------------------------------------------------------------


@pytest.mark.integration
def test_a_real_cross_encoder_ranks_the_semantic_match_first() -> None:
    """The claim that motivates the whole change, against an actual model."""
    url = os.environ.get("RAG_RERANK_URL", "http://localhost:8086")
    try:
        httpx.get(f"{url}/info", timeout=5).raise_for_status()
    except Exception:
        pytest.skip(f"no reranker serving at {url}")

    r = CrossEncoderReranker(url=url)
    out = asyncio.run(r.rerank(_QUERY, _spans(), 2))
    assert [s.chunk_id for s in out] == ["correct", "decoy"]
    assert out[0].score > out[1].score


# --- the per-KB OpenAI-compat reranker (probe-and-store wire shapes) ---------------------------


def _openai_reranker(
    api_shape: str,
    *,
    wire_scores: dict[str, float],
    respond_as: str = "cohere",
    seen: list[httpx.Request] | None = None,
    key: str = "",
) -> OpenAICompatReranker:
    """A stubbed endpoint speaking ``respond_as``; the adapter is configured with
    ``api_shape``. Scores are looked up by text prefix, like the TEI stub above."""
    import json

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        body = json.loads(request.content)
        texts = body.get("documents") or body.get("texts")
        if respond_as == "cohere":
            results = [{"index": i, "relevance_score": wire_scores[t[:20]]} for i, t in enumerate(texts)]
            return httpx.Response(200, json={"results": results})
        return httpx.Response(200, json=[{"index": i, "score": wire_scores[t[:20]]} for i, t in enumerate(texts)])

    r = OpenAICompatReranker(
        "http://reranker:80", "bge-reranker", api_shape=api_shape, api_key=key, api_key_header="apikey"
    )
    r._client = httpx.AsyncClient(
        base_url="http://reranker:80", transport=httpx.MockTransport(handler), headers={"apikey": key} if key else {}
    )
    return r


def test_the_cohere_wire_scores_and_reorders() -> None:
    r = _openai_reranker("cohere", wire_scores=_SCORES, respond_as="cohere")
    out = asyncio.run(r.rerank(_QUERY, _spans(), 2))
    assert [s.chunk_id for s in out] == ["correct", "decoy"]
    assert out[0].score == pytest.approx(0.91)
    assert out[0].scorer == "cross_encoder"


def test_the_tei_wire_scores_and_reorders() -> None:
    r = _openai_reranker("tei", wire_scores=_SCORES, respond_as="tei")
    out = asyncio.run(r.rerank(_QUERY, _spans(), 2))
    assert [s.chunk_id for s in out] == ["correct", "decoy"]


def test_sigmoid_turns_logits_into_probabilities() -> None:
    """bge-reranker behind vLLM/Infinity returns raw logits; the attach-time probe stores
    '+sigmoid' so the gate keeps reading a probability."""
    import math

    logits = {_CORRECT[:20]: 2.0, _DECOY[:20]: -4.0}
    r = _openai_reranker("cohere+sigmoid", wire_scores=logits)
    out = asyncio.run(r.rerank(_QUERY, _spans(), 2))
    assert out[0].chunk_id == "correct"
    assert out[0].score == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))


def test_logits_without_the_sigmoid_shape_are_refused() -> None:
    """A shape drift AFTER attach (server reconfigured to raw scores) fails loudly instead
    of feeding the abstain gate a new scale."""
    logits = {_CORRECT[:20]: 7.4, _DECOY[:20]: -3.1}
    r = _openai_reranker("cohere", wire_scores=logits)
    with pytest.raises(ConfigError, match="probability"):
        asyncio.run(r.rerank(_QUERY, _spans(), 2))


def test_the_key_travels_raw_in_the_configured_header() -> None:
    seen: list[httpx.Request] = []
    r = _openai_reranker("cohere", wire_scores=_SCORES, seen=seen, key="model-secret")
    asyncio.run(r.rerank(_QUERY, _spans(), 2))
    assert seen[0].headers["apikey"] == "model-secret"
    assert "authorization" not in seen[0].headers


def test_an_unknown_api_shape_is_refused_at_construction() -> None:
    for bad in ("jina", "cohere+softmax", "", "tei+"):
        with pytest.raises(ConfigError, match="api_shape"):
            parse_api_shape(bad)


def test_the_probe_detects_wire_and_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Test button's contract: the endpoint itself decides what gets stored."""
    import json as _json
    import math
    from collections.abc import Callable

    def cohere_logits(request: httpx.Request) -> httpx.Response:
        body = _json.loads(request.content)
        if "documents" not in body:
            return httpx.Response(422)
        return httpx.Response(
            200, json={"results": [{"index": i, "relevance_score": [3.7, -5.0][i]} for i in range(2)]}
        )

    def tei_probabilities(request: httpx.Request) -> httpx.Response:
        body = _json.loads(request.content)
        if "texts" not in body:
            return httpx.Response(422)
        return httpx.Response(200, json=[{"index": i, "score": [0.93, 0.04][i]} for i in range(2)])

    real_client = httpx.AsyncClient

    def run(handler: Callable[[httpx.Request], httpx.Response]) -> tuple[str, float]:
        class PatchedClient(real_client):  # type: ignore[valid-type,misc]
            def __init__(self, **kw: object) -> None:
                kw["transport"] = httpx.MockTransport(handler)
                super().__init__(**kw)

        monkeypatch.setattr(httpx, "AsyncClient", PatchedClient)
        try:
            return asyncio.run(probe_rerank_shape("http://reranker:80", "bge-reranker"))
        finally:
            monkeypatch.setattr(httpx, "AsyncClient", real_client)

    shape, sample = run(cohere_logits)
    assert shape == "cohere+sigmoid"
    assert sample == pytest.approx(1.0 / (1.0 + math.exp(-3.7)))

    shape, sample = run(tei_probabilities)
    assert shape == "tei"
    assert sample == pytest.approx(0.93)
