"""OpenAI-compatible embedder adapter (implements :class:`Embedder`) — per-KB attach.

Talks to any OpenAI-compatible ``POST /embeddings`` endpoint (vLLM, Infinity, TEI's
OpenAI facade, the AIOps platform). This is the adapter behind a KB's *embedding* model
connection; the platform-wide default remains :class:`TeiEmbedder` over TEI's native API.

Two contracts differ from TEI deliberately:

* **Auth** — the key travels in a configurable header sent RAW (``apikey: <key>`` for
  AIOps; no ``Bearer`` prefix is ever added), mirroring :class:`OpenAICompatGateway`.
* **Dimension is a stored contract, not a discovery.** ``dim`` comes from the
  connection's ``embed_dim`` (probed once at attach time). The KB's vector collection
  was created against it, so a response of any other width is refused loudly
  (:class:`ConfigError`) instead of warned about — a mismatched vector would poison
  the collection, not merely score oddly.

``token_budget``/``token_offsets`` are intentionally absent: the OpenAI embeddings API
has no tokenize endpoint, and chunking degrades gracefully to its character cap
(bge-m3's 8k context makes that safe).
"""

from __future__ import annotations

import httpx

from rag_common import (
    ConfigError,
    ModelGatewayError,
    assert_sovereign,
    get_logger,
    get_settings,
    model_tls_verify,
)

_log = get_logger(__name__)

# Same rationale as TEI: servers cap a request's batch, so embed in bounded batches.
_BATCH_SIZE = 32

_PROBE_TEXT = "sovereign embedding probe"


class OpenAICompatEmbedder:
    """Embedder backed by an OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(
        self,
        url: str,
        model: str,
        *,
        dim: int,
        api_key: str = "",
        api_key_header: str = "apikey",
        timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        self._url = url.rstrip("/")
        self._model = model
        self._dim = dim
        # Sovereignty: the endpoint host must be egress-allowlisted. Never loosened here.
        assert_sovereign(self._url)
        headers = {api_key_header: api_key} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=self._url,
            timeout=timeout if timeout is not None else settings.embed_timeout_seconds,
            headers=headers,
            verify=model_tls_verify(),
        )

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            try:
                resp = await self._client.post("/embeddings", json={"model": self._model, "input": batch})
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                _log.warning("openai_embed_error", url=self._url, model=self._model, error=str(exc)[:200])
                raise ModelGatewayError(
                    f"Embedding endpoint failed ({type(exc).__name__}). Check this KB's embedding "
                    "connection in Models.",
                    details={"endpoint": self._url, "model": self._model},
                ) from exc
            # Order is the request's, but by contract only via ``index`` — sort, don't assume.
            data = sorted(resp.json()["data"], key=lambda d: int(d["index"]))
            vectors.extend([list(map(float, d["embedding"])) for d in data])
        if vectors and len(vectors[0]) != self._dim:
            raise ConfigError(
                f"embedding endpoint {self._url} returned {len(vectors[0])}-dimensional vectors, but "
                f"this connection is attached at {self._dim} — the KB's vector collection was created "
                "against that width. Re-run Test on the Models page and re-attach.",
                details={"url": self._url, "expected": self._dim, "got": len(vectors[0])},
            )
        return vectors

    async def aclose(self) -> None:
        await self._client.aclose()


async def probe_embedding_dim(
    url: str,
    model: str,
    *,
    api_key: str = "",
    api_key_header: str = "apikey",
    timeout_seconds: float = 15.0,
) -> int:
    """The dimension the endpoint actually returns — asked of the RUNNING server at attach
    time (the Test button), then stored on the connection as the contract :class:`OpenAICompatEmbedder`
    enforces. Raises on any failure; the test endpoint reports it, nothing is stored."""
    assert_sovereign(url)
    headers = {api_key_header: api_key} if api_key else {}
    async with httpx.AsyncClient(
        base_url=url.rstrip("/"), timeout=timeout_seconds, headers=headers, verify=model_tls_verify()
    ) as client:
        resp = await client.post("/embeddings", json={"model": model, "input": [_PROBE_TEXT]})
        resp.raise_for_status()
        return len(resp.json()["data"][0]["embedding"])
