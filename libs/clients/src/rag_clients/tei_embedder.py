"""TEI embedder adapter (implements :class:`Embedder`).

Talks to a self-hosted Hugging Face Text-Embeddings-Inference server over its native
``POST /embed`` API. Sovereign: the endpoint is validated against the egress allowlist
on every call, so the platform cannot be pointed at a foreign embedding API.

Dev default: ``BAAI/bge-small-en-v1.5`` (384-dim) on CPU — see docker-compose.
"""

from __future__ import annotations

import httpx

from rag_common import assert_sovereign, get_logger, get_settings

_log = get_logger(__name__)

# TEI rejects a request whose batch (count or total payload) exceeds its server limit, so a
# large document's chunks must be embedded in bounded batches rather than one giant call.
_BATCH_SIZE = 32

# "not asked yet" — distinct from None, which means "asked, and the server could not say".
_UNKNOWN: object = object()

# Characters per /tokenize request. TEI refuses a large input (measured on the running server:
# 20 000 characters fine, 200 000 -> 422), so a long document is tokenized in windows. Well
# under the observed failure point rather than at it: the limit is the server's, not ours, and
# a different model or TEI build will place it somewhere else.
_TOKENIZE_WINDOW_CHARS = 8000


def _windows(text: str, size: int) -> list[tuple[int, str]]:
    """``(base offset, window)`` covering ``text``, each cut at whitespace where possible.

    A word split across two windows tokenizes differently from a whole one, so the offsets
    would stop describing what the embedder will actually see."""
    out: list[tuple[int, str]] = []
    at = 0
    while at < len(text):
        if len(text) - at <= size:
            out.append((at, text[at:]))
            break
        cut = text.rfind(" ", at, at + size)
        end = cut if cut > at else at + size  # no space in the whole window: cut anyway
        out.append((at, text[at:end]))
        # Step PAST the separator. Leaving it to lead the next window makes the tokenizer see
        # " word" instead of "word", which adds a phantom token at every boundary — six on a
        # 48 000-character document, each one shifting nothing but inflating the count.
        at = end + 1 if end < len(text) and text[end] == " " else end
    return out


class TeiEmbedder:
    """Embedder backed by a self-hosted TEI server."""

    def __init__(self, url: str | None = None, dim: int | None = None, timeout: float | None = None) -> None:
        settings = get_settings()
        self._url = (url or settings.tei_url).rstrip("/")
        self._dim = dim or settings.embed_dim
        assert_sovereign(self._url)
        self._client = httpx.AsyncClient(
            base_url=self._url, timeout=timeout if timeout is not None else settings.embed_timeout_seconds
        )
        self._budget: int | None | object = _UNKNOWN  # discovered once per process

    @property
    def dim(self) -> int:
        return self._dim

    async def token_budget(self) -> int | None:
        """The model's input limit in TOKENS, minus the special tokens it adds itself.

        Asked of the RUNNING server rather than taken from config: the limit that truncates a
        chunk is the one the container serves, and a stale setting would let a chunker believe
        it is inside a budget it is not. ``None`` when the server cannot say, which the caller
        must treat as "no token bound available" rather than as unlimited.

        The specials are discovered, not assumed to be two: a model with a different template
        adds a different number, and they consume the same budget as content."""
        if self._budget is not _UNKNOWN:
            return self._budget if isinstance(self._budget, int) else None
        self._budget = None
        try:
            limit = int((await self._client.get("/info")).json()["max_input_length"])
            probe = await self._client.post("/tokenize", json={"inputs": "a"})
            specials = sum(1 for t in probe.json()[0] if t.get("special"))
            self._budget = max(1, limit - specials)
        except Exception as exc:
            _log.warning(
                "embedder_token_budget_unavailable",
                url=self._url,
                error_type=type(exc).__name__,
                error=str(exc),
                detail="chunking falls back to the character cap, which cannot bound a token budget",
            )
        return self._budget if isinstance(self._budget, int) else None

    async def token_offsets(self, text: str) -> list[int] | None:
        """Character offset of each token the embedder would produce for ``text``.

        Amortised over the DOCUMENT, not per chunk: the chunker slices the result with a binary
        search, so capping every chunk costs a handful of round trips for the whole document.
        Special tokens have no offsets and are excluded — they are counted in the budget instead.

        Windowed because TEI rejects a large input outright (measured: 20 000 characters is
        fine, 200 000 returns 422), and each window is cut at whitespace so a word never
        straddles a boundary — a split word tokenizes differently from a whole one, which would
        make the offsets disagree with what the embedder will actually see."""
        if not text:
            return []
        try:
            offsets: list[int] = []
            for base, window in _windows(text, _TOKENIZE_WINDOW_CHARS):
                resp = await self._client.post("/tokenize", json={"inputs": window})
                resp.raise_for_status()
                offsets.extend(base + t["start"] for t in resp.json()[0] if t.get("start") is not None)
            return offsets
        except Exception as exc:
            _log.warning(
                "embedder_tokenize_failed",
                url=self._url,
                error_type=type(exc).__name__,
                error=str(exc),
                chars=len(text),
                detail="this document is chunked on characters alone and may exceed the token limit",
            )
            return None

    async def count_tokens(self, texts: list[str]) -> list[int] | None:
        """How many tokens the embedder produces for each of ``texts`` — its own count, not a guess.

        The honest source for ``chunks.token_count``. A characters/4 rule is not close on the inputs
        that matter: measured through this adapter against the running bge-small server, 1 000
        characters of table-of-contents dot leaders tokenize to **1 000 tokens** — twice the
        510-token budget, where characters/4 would have recorded 250 and reported a chunk as
        comfortably inside a limit it is double. (Devanagari runs ~1.6 chars/token and prose ~5.1,
        so the error has no safe direction either.) A token count that is wrong is worse than one
        that is absent, because a reader cannot tell which it has. So this asks the RUNNING server,
        for the same reason :meth:`token_budget` does, and returns ``None`` — meaning "write NULL" —
        when it cannot answer. Optional on the Protocol: an adapter without a tokenizer endpoint
        simply lacks it.

        Special tokens are EXCLUDED, as in :meth:`token_offsets`: they are the model's own framing
        rather than content of the chunk, and :meth:`token_budget` already subtracts them from the
        limit this count is compared against.

        Unlike ``/embed``, ``/tokenize`` does NOT truncate at ``max_input_length`` — measured on the
        running server, a 4 718-token input came back whole against a 512-token limit — so an
        over-budget chunk reports its REAL size, which is the case actually worth seeing.

        Batched at the server's ``max_client_batch_size``; the bound is on the COUNT, not the
        payload (both measured on the running server: a batch of 40 is refused outright with
        ``batch size 40 > maximum allowed batch size 32``, while 32 inputs of 20 000 characters
        each — 640 000 in one request — are tokenized fine. A single 200 000-character input is
        refused, which is a per-input limit and well above any chunk).
        """
        if not texts:
            return []
        counts: list[int] = []
        try:
            for start in range(0, len(texts), _BATCH_SIZE):
                resp = await self._client.post("/tokenize", json={"inputs": texts[start : start + _BATCH_SIZE]})
                resp.raise_for_status()
                counts.extend(sum(1 for t in tokens if t.get("start") is not None) for tokens in resp.json())
        except Exception as exc:
            _log.warning(
                "embedder_count_tokens_failed",
                url=self._url,
                error_type=type(exc).__name__,
                error=str(exc),
                texts=len(texts),
                detail="token_count is left unset rather than estimated",
            )
            return None
        if len(counts) != len(texts):
            # One count per input or nothing: a short response would silently pair each count with
            # the WRONG text, which is a wrong number presented as a measured one.
            _log.warning("embedder_count_tokens_mismatch", url=self._url, asked=len(texts), returned=len(counts))
            return None
        return counts

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            # ``truncate`` lets TEI clip an over-long chunk to the model's max sequence
            # length instead of 413/422-ing the whole ingest.
            resp = await self._client.post("/embed", json={"inputs": batch, "truncate": True})
            resp.raise_for_status()
            vectors.extend(resp.json())
        if vectors and len(vectors[0]) != self._dim:
            _log.warning("embed_dim_mismatch", expected=self._dim, got=len(vectors[0]))
        return vectors

    async def aclose(self) -> None:
        await self._client.aclose()
