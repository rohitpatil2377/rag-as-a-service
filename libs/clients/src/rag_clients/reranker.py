"""Reranker adapters (implement :class:`Reranker`) — FR-D2.

The reranker is not a tie-breaker. It decides the FINAL order of what reaches the model, and
its top score becomes ``RetrievalResult.max_score``, which the grounded-or-abstain gate reads.
So it has to score *relevance*.

**Two-stage, not blended.** Dense + BM25 + RRF answer "which candidates are worth looking at":
a rank-derived, set-relative signal (``normalize_scores`` maps the best candidate of every
result set to exactly 1.0, so it carries no information about whether that candidate is any
good). The cross-encoder answers "how relevant is this passage to this question": it reads the
query and the passage together and returns a calibrated probability. Averaging the two would
make the gate's input depend on what else happened to be retrieved — a query whose whole pool
is irrelevant would still show a strong blended score, which is exactly the failure the gate
exists to catch. So the two never share a field: candidates arrive carrying ``fusion_score``
only, and a reranker is the ONLY thing that ever writes ``score`` — declaring itself in
``scorer`` as it does so. A span that reaches the gate unscored therefore abstains, which is
what makes skipping the cross-encoder a visible failure rather than a silent 1.0.

* :class:`CrossEncoderReranker` — a self-hosted TEI ``/rerank`` server. The real path, in every
  environment.
* :class:`LexicalReranker` — term-overlap counting. Retained only so an operator can run with
  no reranker container at all, and only when explicitly selected: it is chosen by
  ``RAG_RERANK_BACKEND``, never by the environment name. Its scores are overlap fractions and
  are NOT comparable to a cross-encoder threshold.
"""

from __future__ import annotations

import math
import re

import httpx

from rag_common import (
    ConfigError,
    RetrievalError,
    assert_sovereign,
    get_logger,
    get_settings,
    model_tls_verify,
)
from rag_core import Reranker
from rag_schemas import RetrievedSpan

_log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Wire shapes an OpenAI-compat rerank endpoint may speak. "cohere" also covers Jina —
# same request body, same ``results[].relevance_score`` response.
_WIRES = ("tei", "cohere")


def parse_api_shape(shape: str) -> tuple[str, bool]:
    """``"<wire>[+sigmoid]"`` → (wire, apply_sigmoid). Raises :class:`ConfigError` on an
    unknown grammar — a connection whose stored shape we cannot speak must fail loudly,
    not fall back to a guess (the probe at attach time is where guessing happens)."""
    wire, plus, transform = shape.partition("+")
    if wire not in _WIRES or (plus and transform != "sigmoid"):
        raise ConfigError(
            f"unknown rerank api_shape {shape!r} — expected one of "
            f"{', '.join(_WIRES)} optionally suffixed with '+sigmoid'.",
            details={"api_shape": shape},
        )
    return wire, bool(plus)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _probability(value: float, url: str) -> float:
    """A cross-encoder score the gate can be calibrated against must be a probability.

    TEI's ``/rerank`` sigmoids by default, but ``raw_scores=true``, a TEI version change, or a
    different server (Infinity, vLLM ``/score``) returns unbounded logits. Left alone, the first
    such value fails the schema bound as a bare pydantic error and every query 500s. Refuse it
    here instead, naming the endpoint and the value — the abstain threshold is meaningless
    against a scale nobody chose."""
    if not 0.0 <= value <= 1.0:
        raise ConfigError(
            f"reranker at {url} returned {value}, which is not a relevance probability in [0,1]. "
            "The abstain threshold is calibrated against a probability, so this scorer cannot be "
            "used as-is — check that the server is not configured to return raw logits.",
            details={"url": url, "score": value},
        )
    return value


class LexicalReranker:
    """Term-overlap scoring, no model. Explicitly selected only — see the module docstring."""

    async def rerank(self, query: str, spans: list[RetrievedSpan], top_k: int) -> list[RetrievedSpan]:
        q = _tokens(query)
        if not q:
            # Nothing to score against. Hand the spans back exactly as they arrived — unscored —
            # rather than letting their fusion rank masquerade as a judgement.
            return spans[:top_k]
        scored = [
            s.model_copy(update={"score": round(len(q & _tokens(s.text)) / len(q), 6), "scorer": "lexical"})
            for s in spans
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]


class CrossEncoderReranker:
    """Cross-encoder reranker via a self-hosted TEI ``/rerank`` server (sovereign, allowlisted).

    Runs in every environment. If the server is unreachable the query FAILS — there is no
    fallback to lexical scoring, because a silent downgrade would leave the abstain gate
    reading a number on a different scale from the threshold it is compared against."""

    def __init__(
        self,
        url: str | None = None,
        *,
        model: str | None = None,
        batch_size: int | None = None,
        timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        self._url = (url or settings.rerank_url).rstrip("/")
        assert_sovereign(self._url)
        self._model = model or settings.rerank_model
        self._model_verified = False
        self._batch = batch_size or settings.rerank_batch_size
        self._client = httpx.AsyncClient(base_url=self._url, timeout=timeout or settings.rerank_timeout_seconds)

    async def _verify_model(self) -> None:
        """Confirm the endpoint serves the model the abstain threshold was calibrated against.

        A threshold is a claim about one model's score distribution. Repointing the reranker —
        switching to the GPU tier is a one-line compose change — moves the distribution while
        the threshold stays put, so the gate quietly starts meaning something else. Same bug
        class as a re-index silently changing chunking strategy.

        Lazy, because __init__ cannot await; once per instance, so it costs one round trip per
        process rather than one per query. NOTE: `get_reranker` is lru_cached, so a container
        swapped underneath a running API is not re-detected."""
        if self._model_verified:
            return
        try:
            resp = await self._client.get("/info")
            resp.raise_for_status()
            served = str(resp.json().get("model_id", ""))
        except httpx.HTTPError as exc:
            # Unreachable upstream, not a bad request and not our bug: 502, the same shape
            # ConnectorError and ModelGatewayError already use for this class of failure.
            raise RetrievalError(
                f"reranker at {self._url} did not answer /info ({type(exc).__name__}: {exc}), so the "
                "model behind the abstain threshold cannot be confirmed.",
                details={"url": self._url},
            ) from exc
        if served.casefold() != self._model.casefold():
            raise ConfigError(
                f"reranker at {self._url} serves {served!r}, but RAG_RERANK_MODEL says "
                f"{self._model!r}. RAG_ABSTAIN_THRESHOLD is calibrated against one model's score "
                "distribution and is not transferable — point the config at the model actually "
                "being served and re-calibrate the threshold for it.",
                details={"url": self._url, "served": served, "configured": self._model},
            )
        self._model_verified = True

    async def rerank(self, query: str, spans: list[RetrievedSpan], top_k: int) -> list[RetrievedSpan]:
        if not spans:
            return []
        await self._verify_model()
        # TEI caps a request at max_client_batch_size (32 by default) — below our candidate
        # pool. Batch here rather than depending on server flags, and sort GLOBALLY afterwards:
        # each response is ordered within its own batch only.
        scored: list[tuple[float, RetrievedSpan]] = []
        for start in range(0, len(spans), self._batch):
            batch = spans[start : start + self._batch]
            try:
                resp = await self._client.post(
                    "/rerank", json={"query": query, "texts": [s.text for s in batch], "return_text": False}
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise RetrievalError(
                    f"reranker at {self._url} is unavailable ({type(exc).__name__}: {exc}). Retrieval cannot "
                    "produce a comparable confidence without it; start the reranker service or set "
                    "RAG_RERANK_BACKEND explicitly to accept a different scorer.",
                    details={"url": self._url, "candidates": len(spans)},
                ) from exc
            for result in resp.json():
                scored.append((_probability(float(result["score"]), self._url), batch[int(result["index"])]))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        # `score` gets the cross-encoder's absolute relevance and says so. The set-relative
        # fusion score the span arrived with is left untouched in its own field.
        return [s.model_copy(update={"score": value, "scorer": "cross_encoder"}) for value, s in scored[:top_k]]

    async def aclose(self) -> None:
        await self._client.aclose()


class OpenAICompatReranker:
    """Reranker behind a KB's *reranking* model connection (per-KB attach, sovereign).

    Speaks the wire shape probed at attach time and stored on the connection
    (``api_shape``) — fully deterministic at query time, no negotiation. There is no
    TEI ``/info`` check here (OpenAI-compat servers don't expose it); the model id
    travels in every request body instead, which is the equivalent pinning. Scores
    stay under the same ``_probability`` contract as the platform reranker: an
    endpoint that returns raw logits gets ``+sigmoid`` from the probe, and a shape
    drift AFTER attach fails loudly instead of feeding the gate a new scale."""

    def __init__(
        self,
        url: str,
        model: str,
        *,
        api_shape: str,
        api_key: str = "",
        api_key_header: str = "apikey",
        batch_size: int | None = None,
        timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        self._url = url.rstrip("/")
        assert_sovereign(self._url)
        self._model = model
        self._wire, self._sigmoid = parse_api_shape(api_shape)
        self._batch = batch_size or settings.rerank_batch_size
        headers = {api_key_header: api_key} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=self._url,
            timeout=timeout or settings.rerank_timeout_seconds,
            headers=headers,
            verify=model_tls_verify(),
        )

    def _score(self, value: float) -> float:
        if self._sigmoid:
            value = 1.0 / (1.0 + math.exp(-value))
        return _probability(value, self._url)

    async def rerank(self, query: str, spans: list[RetrievedSpan], top_k: int) -> list[RetrievedSpan]:
        if not spans:
            return []
        scored: list[tuple[float, RetrievedSpan]] = []
        for start in range(0, len(spans), self._batch):
            batch = spans[start : start + self._batch]
            texts = [s.text for s in batch]
            try:
                if self._wire == "cohere":
                    resp = await self._client.post(
                        "/rerank", json={"model": self._model, "query": query, "documents": texts}
                    )
                else:
                    resp = await self._client.post(
                        "/rerank", json={"query": query, "texts": texts, "return_text": False}
                    )
                resp.raise_for_status()
                payload = resp.json()
                results = payload["results"] if self._wire == "cohere" else payload
                for result in results:
                    raw = result["relevance_score"] if self._wire == "cohere" else result["score"]
                    scored.append((self._score(float(raw)), batch[int(result["index"])]))
            except httpx.HTTPError as exc:
                raise RetrievalError(
                    f"reranker at {self._url} is unavailable ({type(exc).__name__}: {exc}). Retrieval "
                    "cannot produce a comparable confidence without it — check this KB's reranking "
                    "connection in Models.",
                    details={"url": self._url, "candidates": len(spans)},
                ) from exc

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [s.model_copy(update={"score": value, "scorer": "cross_encoder"}) for value, s in scored[:top_k]]

    async def aclose(self) -> None:
        await self._client.aclose()


async def probe_rerank_shape(
    url: str,
    model: str,
    *,
    api_key: str = "",
    api_key_header: str = "apikey",
    timeout_seconds: float = 15.0,
) -> tuple[str, float]:
    """Learn which wire shape the endpoint speaks — asked of the RUNNING server at attach
    time (the Test button), then stored on the connection so the runtime never guesses.

    Tries the cohere body first (the likely shape behind an ``/openai/v1`` base), then
    TEI's. A relevant/irrelevant probe pair decides whether scores are already
    probabilities or raw logits (``+sigmoid``). Returns ``(api_shape, sample_score)``
    where the sample is the relevant document's post-transform score. Raises the last
    failure when neither shape works; the test endpoint reports it verbatim."""
    assert_sovereign(url)
    query = "What is the capital of France?"
    docs = ["Paris is the capital of France.", "Photosynthesis converts light into chemical energy."]
    headers = {api_key_header: api_key} if api_key else {}
    last_error: Exception | None = None
    async with httpx.AsyncClient(
        base_url=url.rstrip("/"), timeout=timeout_seconds, headers=headers, verify=model_tls_verify()
    ) as client:
        for wire in ("cohere", "tei"):
            try:
                if wire == "cohere":
                    resp = await client.post("/rerank", json={"model": model, "query": query, "documents": docs})
                    resp.raise_for_status()
                    results = resp.json()["results"]
                    raw = {int(r["index"]): float(r["relevance_score"]) for r in results}
                else:
                    resp = await client.post("/rerank", json={"query": query, "texts": docs, "return_text": False})
                    resp.raise_for_status()
                    raw = {int(r["index"]): float(r["score"]) for r in resp.json()}
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                last_error = exc
                continue
            if not raw:
                last_error = ConfigError(f"reranker at {url} returned no scores", details={"url": url})
                continue
            sigmoid = any(not 0.0 <= v <= 1.0 for v in raw.values())
            shape = f"{wire}+sigmoid" if sigmoid else wire
            sample = raw.get(0, next(iter(raw.values())))
            if sigmoid:
                sample = 1.0 / (1.0 + math.exp(-sample))
            return shape, sample
    raise last_error if last_error is not None else ConfigError("rerank probe failed", details={"url": url})


class FallbackReranker:
    """A PRIMARY reranker with a SECONDARY fallback (implements the Reranker Protocol).

    Tries the primary; if it raises :class:`RetrievalError` (timeout, 5xx, or unreachable — the
    only failure a reranker adapter raises) it reranks with the secondary instead, so retrieval
    still yields a comparable confidence when the primary (e.g. a hosted jina endpoint) is briefly
    down. Non-failure results pass straight through. The gate is calibrated to the primary's scale,
    so a fallback window is a documented degraded mode — see ``rerank_fallback_*`` in config."""

    def __init__(self, primary: Reranker, secondary: Reranker) -> None:
        self.primary = primary
        self.secondary = secondary

    async def rerank(self, query: str, spans: list[RetrievedSpan], top_k: int) -> list[RetrievedSpan]:
        try:
            return await self.primary.rerank(query, spans, top_k)
        except RetrievalError as exc:
            _log.warning("reranker_primary_failed_fallback_secondary", error=str(exc)[:200])
            return await self.secondary.rerank(query, spans, top_k)

    async def aclose(self) -> None:
        for r in (self.primary, self.secondary):
            close = getattr(r, "aclose", None)
            if close is not None:
                await close()


def _build_one_reranker(
    backend: str, *, url: str, model: str, api_shape: str, api_key: str, api_key_header: str
) -> LexicalReranker | CrossEncoderReranker | OpenAICompatReranker:
    """Build ONE reranker from an explicit backend + params (no wrapping)."""
    if backend == "lexical":
        _log.warning(
            "reranker_lexical_selected",
            detail="term-overlap scoring: final ranking is bag-of-words and max_score is an overlap "
            "fraction, NOT comparable to a cross-encoder abstain threshold",
        )
        return LexicalReranker()
    if backend == "openai_compat":
        # An OpenAI-compatible /rerank endpoint (e.g. jina, or AIOps bge-reranker-v2-m3 on GPU).
        return OpenAICompatReranker(url, model, api_shape=api_shape, api_key=api_key, api_key_header=api_key_header)
    return CrossEncoderReranker()


def build_reranker() -> LexicalReranker | CrossEncoderReranker | OpenAICompatReranker | FallbackReranker:
    """Select the reranker from CONFIG. Never from the environment name.

    The previous factory returned the lexical reranker whenever ``env != "prod"``, so no
    environment a developer or a staging deployment could actually run ever exercised a model.
    If ``rerank_fallback_backend`` is set, the primary is wrapped in a :class:`FallbackReranker`
    with a secondary built from the ``rerank_fallback_*`` settings (e.g. jina primary + bge
    secondary)."""
    settings = get_settings()
    primary = _build_one_reranker(
        settings.rerank_backend,
        url=settings.rerank_url,
        model=settings.rerank_model,
        api_shape=settings.rerank_api_shape,
        api_key=settings.rerank_api_key,
        api_key_header=settings.rerank_api_key_header,
    )
    if not settings.rerank_fallback_backend:
        return primary
    secondary = _build_one_reranker(
        settings.rerank_fallback_backend,
        url=settings.rerank_fallback_url,
        model=settings.rerank_fallback_model,
        api_shape=settings.rerank_fallback_api_shape,
        api_key=settings.rerank_fallback_api_key,
        api_key_header=settings.rerank_fallback_api_key_header,
    )
    _log.info(
        "reranker_fallback_configured",
        primary=settings.rerank_backend,
        secondary=settings.rerank_fallback_backend,
    )
    return FallbackReranker(primary, secondary)
