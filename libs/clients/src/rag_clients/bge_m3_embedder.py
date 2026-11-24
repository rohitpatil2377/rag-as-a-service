"""bge-m3 multi-vector embedder adapter (Slice 2b) — dense + learned-sparse + ColBERT.

Talks to the bge-m3 ``POST /embed`` endpoint, which returns THREE representations for a text in
ONE call: a dense 1024-d vector (the same space the ``/embeddings`` facade returns — verified
live, cosine 1.000000), a learned-sparse map (token id → weight), and a ColBERT late-interaction
multivector (N_tokens x dim).

It implements the base :class:`Embedder` Protocol (``dim`` + ``embed`` returning DENSE ONLY), so it
is a drop-in dense embedder for the unchanged retrieval/index paths and for ``pipeline_fingerprint``.
It ADDS the :class:`MultiVectorEmbedder` capability (``embed_multi``) that the multivector path
duck-types — mirroring the optional ``token_budget``/``token_offsets`` pattern.

Two contracts differ from :class:`OpenAICompatEmbedder` deliberately:

* **Auth** — bge-m3 ``/embed`` expects ``Authorization: Bearer <key>`` (``embed_api_auth="bearer"``),
  NOT the raw ``apikey`` header the ``/embeddings`` adapter uses. The existing openai_compat path is
  untouched. The key value lives ONLY in the gitignored ``docker-compose.override.yml``.
* **Dimension is a stored contract** — a dense response of any other width is refused loudly
  (:class:`ConfigError`), exactly like :class:`OpenAICompatEmbedder`: a wrong-width dense vector
  would poison the collection, not merely score oddly.
"""

from __future__ import annotations

import hashlib
from typing import Any

import httpx

from rag_common import (
    ConfigError,
    ModelGatewayError,
    assert_sovereign,
    get_logger,
    get_settings,
    model_tls_verify,
)
from rag_core import MultiVectorOutput

_log = get_logger(__name__)

# The server caps a request's batch, so embed in bounded batches (same rationale as TEI/openai).
_BATCH_SIZE = 32
# Coerce a non-integer sparse key to a stable, bounded positive index for Qdrant SparseVector.
_SPARSE_KEY_SPACE = 2**31 - 1


class BgeM3MultiEmbedder:
    """Embedder backed by a bge-m3 ``/embed`` endpoint (dense + sparse + ColBERT)."""

    def __init__(
        self,
        url: str,
        model: str,
        *,
        dim: int,
        api_key: str = "",
        api_key_header: str = "apikey",
        auth: str = "bearer",
        timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        self._url = url.rstrip("/")
        self._model = model
        self._dim = dim
        # Sovereignty: the endpoint host must be egress-allowlisted. Never loosened here.
        assert_sovereign(self._url)
        headers: dict[str, str] = {}
        if api_key:
            # Bearer for the bge-m3 /embed endpoint; "raw" keeps the openai_compat apikey shape.
            headers = {"Authorization": f"Bearer {api_key}"} if auth == "bearer" else {api_key_header: api_key}
        self._client = httpx.AsyncClient(
            base_url=self._url,
            timeout=timeout if timeout is not None else settings.embed_timeout_seconds,
            headers=headers,
            verify=model_tls_verify(),
        )

    @property
    def dim(self) -> int:
        return self._dim

    async def _post(self, batch: list[str], *, dense: bool, sparse: bool, colbert: bool) -> list[dict[str, Any]]:
        try:
            resp = await self._client.post(
                "/embed",
                json={"input": batch, "dense": dense, "sparse": sparse, "colbert": colbert},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            _log.warning("bge_m3_embed_error", url=self._url, model=self._model, error=str(exc)[:200])
            raise ModelGatewayError(
                f"Embedding endpoint failed ({type(exc).__name__}). Check this KB's embedding connection in Models.",
                details={"endpoint": self._url, "model": self._model},
            ) from exc
        # Order is the request's, but by contract only via ``index`` — sort, don't assume.
        return sorted(resp.json()["data"], key=lambda d: int(d["index"]))

    def _check_dense_width(self, vec: list[float]) -> None:
        if vec and len(vec) != self._dim:
            raise ConfigError(
                f"embedding endpoint {self._url} returned {len(vec)}-dimensional dense vectors, but "
                f"this connection is attached at {self._dim} — the KB's vector collection was created "
                "against that width. Re-run Test on the Models page and re-attach.",
                details={"url": self._url, "expected": self._dim, "got": len(vec)},
            )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """DENSE vectors only — the drop-in :class:`Embedder` contract (index/query/fingerprint)."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            data = await self._post(batch, dense=True, sparse=False, colbert=False)
            vectors.extend([list(map(float, d["dense"])) for d in data])
        if vectors:
            self._check_dense_width(vectors[0])
        return vectors

    async def embed_multi(
        self, texts: list[str], *, dense: bool = True, sparse: bool = True, colbert: bool = True
    ) -> list[MultiVectorOutput]:
        """Dense + learned-sparse + ColBERT for each text (one /embed call per batch)."""
        if not texts:
            return []
        out: list[MultiVectorOutput] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            data = await self._post(batch, dense=dense, sparse=sparse, colbert=colbert)
            for d in data:
                dv = [float(x) for x in d["dense"]] if dense and d.get("dense") is not None else []
                if dv:
                    self._check_dense_width(dv)
                out.append(
                    MultiVectorOutput(
                        dense=dv,
                        sparse=_parse_sparse(d.get("sparse")) if sparse else {},
                        colbert=_parse_colbert(d.get("colbert")) if colbert else [],
                    )
                )
        return out

    async def aclose(self) -> None:
        await self._client.aclose()


def _sparse_index(key: Any) -> int:
    """A bge-m3 sparse key → a non-negative integer index for Qdrant SparseVector.

    bge-m3 lexical weights are keyed by token id (an integer). A server that returns token STRINGS
    is coerced to a stable, bounded positive index by hashing, so the sparse dot-product still
    matches identical tokens between query and document."""
    try:
        return abs(int(key))
    except (TypeError, ValueError):
        h = int(hashlib.md5(str(key).encode()).hexdigest(), 16)  # noqa: S324 - non-crypto hashing
        return h % _SPARSE_KEY_SPACE


def _parse_sparse(raw: Any) -> dict[int, float]:
    if not isinstance(raw, dict):
        return {}
    return {_sparse_index(k): float(v) for k, v in raw.items()}


def _parse_colbert(raw: Any) -> list[list[float]]:
    if not isinstance(raw, list):
        return []
    return [[float(x) for x in tok] for tok in raw]
