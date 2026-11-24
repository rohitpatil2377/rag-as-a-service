"""Asking the embedder how it tokenizes (P49).

The embedder truncates over-long input silently — `truncate: true` on /embed means HTTP 200,
a normal-looking vector, no header, no log. Content past the limit stays in the stored text and
in BM25 but is invisible to dense search. The chunker can only prevent that if it knows the
budget in the units the embedder actually enforces, so it asks the RUNNING server rather than
trusting config or a local tokenizer copy that could drift from what is served.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import httpx
import pytest

from rag_clients.tei_embedder import _TOKENIZE_WINDOW_CHARS, TeiEmbedder, _windows


@pytest.fixture(autouse=True)
def _allowlist(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,tei")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _server(
    *,
    limit: int = 512,
    specials: int = 2,
    reject_over: int | None = None,
    max_batch: int = 32,
    batches: list[int] | None = None,
    drop_last: bool = False,
) -> httpx.MockTransport:
    """A stand-in TEI, faithful to the three behaviours measured on the running server (bge-small,
    `max_input_length` 512, `max_client_batch_size` 32): `/tokenize` takes EITHER one string or a
    list and answers one token list per input; a batch over the server's limit is refused outright;
    and it does NOT truncate at `max_input_length` (a 4 718-token input came back whole).

    ``batches`` records the size of each request, for asserting how the client batches.
    ``drop_last`` makes it answer one list short — the mispairing `count_tokens` must refuse.
    """

    def _tokens_for(text: str) -> list[dict[str, object]]:
        tokens: list[dict[str, object]] = [
            {"id": 0, "special": True, "start": None, "stop": None} for _ in range(specials // 2)
        ]
        at = 0
        for word in text.split(" "):
            tokens.append({"id": 1, "special": False, "start": at, "stop": at + len(word)})
            at += len(word) + 1
        tokens += [{"id": 2, "special": True, "start": None, "stop": None} for _ in range(specials - specials // 2)]
        return tokens

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        if request.url.path == "/info":
            return httpx.Response(200, json={"max_input_length": limit, "model_id": "stub"})
        body = json.loads(request.content)
        raw = body["inputs"]
        batch = raw if isinstance(raw, list) else [raw]
        if batches is not None:
            batches.append(len(batch))
        if len(batch) > max_batch:
            return httpx.Response(
                422, json={"error": f"batch size {len(batch)} > maximum allowed batch size {max_batch}"}
            )
        if reject_over is not None and any(len(t) > reject_over for t in batch):
            return httpx.Response(422, json={"error": "input too long"})
        answered = [_tokens_for(t) for t in batch]
        return httpx.Response(200, json=answered[:-1] if drop_last else answered)

    return httpx.MockTransport(handler)


def _embedder(transport: httpx.MockTransport) -> TeiEmbedder:
    e = TeiEmbedder(url="http://tei")
    e._client = httpx.AsyncClient(base_url="http://tei", transport=transport)
    return e


def test_the_budget_excludes_the_special_tokens_the_model_adds() -> None:
    """Specials consume the same budget as content, so a chunker told "512" would build chunks
    the embedder then truncates. Discovered, not assumed to be two."""
    assert asyncio.run(_embedder(_server(limit=512, specials=2)).token_budget()) == 510
    assert asyncio.run(_embedder(_server(limit=512, specials=4)).token_budget()) == 508


def test_an_unreachable_tokenizer_degrades_visibly() -> None:
    """No token bound is a real state — it must read as "unknown", never as "unlimited"."""

    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    e = _embedder(httpx.MockTransport(dead))
    assert asyncio.run(e.token_budget()) is None
    assert asyncio.run(e.token_offsets("some text")) is None


def test_a_long_document_is_tokenized_in_windows() -> None:
    """TEI refuses a large input outright (measured: 20k chars fine, 200k -> 422), so a whole
    document cannot go in one call."""
    text = " ".join(f"word{i}" for i in range(6000))  # comfortably over one window
    assert len(text) > _TOKENIZE_WINDOW_CHARS
    offsets = asyncio.run(_embedder(_server(reject_over=_TOKENIZE_WINDOW_CHARS + 1)).token_offsets(text))
    assert offsets is not None
    assert len(offsets) == 6000  # every token, across every window
    assert offsets == sorted(offsets)  # ...with offsets that keep ascending across boundaries


def test_window_offsets_are_absolute() -> None:
    """A window's offsets are relative to itself; unadjusted they would all point near the
    start of the document and the chunker would cap the wrong regions."""
    text = " ".join(f"word{i}" for i in range(6000))
    offsets = asyncio.run(_embedder(_server(reject_over=_TOKENIZE_WINDOW_CHARS + 1)).token_offsets(text))
    assert offsets is not None
    for offset, word in zip(offsets[:200], text.split(" ")[:200], strict=True):
        assert text[offset : offset + len(word)] == word


def test_windows_never_split_a_word() -> None:
    """A word split across two windows tokenizes differently from a whole one, so the offsets
    would stop describing what the embedder will actually see."""
    text = " ".join(f"word{i:05d}" for i in range(4000))
    for base, window in _windows(text, 1000):
        assert text[base : base + len(window)] == window
        assert not window.startswith(" ")
        if base + len(window) < len(text):
            assert text[base + len(window)] == " ", "window boundary landed inside a word"


# --- count_tokens: the honest source for `chunks.token_count` (ADR-0024 / discovery §8.10) -------
#
# A v2 chunk is assembled from markup blocks and has no span into the plain-text layer, so the
# document-level `token_offsets` cannot be sliced for it. The embedder's own tokenizer is the only
# thing that knows, and an estimate is not an acceptable substitute: characters/4 is out by a factor
# of four on table-of-contents dot leaders. So the contract is "the server's count, or nothing".


def test_the_count_is_the_servers_own_and_excludes_specials() -> None:
    """Specials are the model's framing, not content of the chunk — and `token_budget` already
    subtracts them from the limit this count is compared against, so counting them here would
    double-count them."""
    assert asyncio.run(_embedder(_server(specials=2)).count_tokens(["one two three", "a"])) == [3, 1]
    assert asyncio.run(_embedder(_server(specials=6)).count_tokens(["one two three"])) == [3]


def test_counting_is_batched_at_the_servers_limit() -> None:
    """Measured: 40 inputs in one request is refused with "batch size 40 > maximum allowed batch
    size 32". The bound is on the COUNT, not the payload, so batching by count is what is needed."""
    batches: list[int] = []
    counts = asyncio.run(_embedder(_server(batches=batches)).count_tokens([f"t{i}" for i in range(70)]))
    assert counts == [1] * 70
    assert batches == [32, 32, 6]


def test_an_unanswerable_tokenizer_leaves_the_count_unset_rather_than_guessing() -> None:
    """None means "the embedder could not say", which is what makes a NULL token_count honest.
    Returning an estimate here is the one outcome nothing downstream could detect."""

    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    assert asyncio.run(_embedder(httpx.MockTransport(dead)).count_tokens(["anything"])) is None


def test_a_short_answer_is_refused_rather_than_mispaired() -> None:
    """One count per input or nothing. A response one list short would silently pair every count
    after the gap with the WRONG chunk — a wrong number presented as a measured one."""
    assert asyncio.run(_embedder(_server(drop_last=True)).count_tokens(["one two", "three"])) is None


def test_nothing_to_count_asks_nothing() -> None:
    batches: list[int] = []
    assert asyncio.run(_embedder(_server(batches=batches)).count_tokens([])) == []
    assert batches == []
