"""`chunks.token_count` comes from the embedder, or not at all (RAG v2; discovery §8.10).

The plan gives `chunks` a `token_count`, and until this it had no honest source: the document-level
`token_offsets` cannot supply one, because a v2 chunk is assembled from markup blocks and so has no
span into the plain-text layer to slice. These tests pin the three properties that make the column
trustworthy rather than merely populated:

  * a v2 chunk carries the EMBEDDER's count of its own stored text;
  * an embedder with no tokenizer leaves it unset — never estimated;
  * the v1 path neither carries a count nor makes the call, so the default path is unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from rag_core.layout import LayoutTree, parse_layout
from rag_schemas import DocumentEnvelope
from rag_schemas.common import ChunkStrategy, SourceKind

_DOC_ID = "fixture-token-count.pdf"

_MARKUP = """
<h1>Attachment-1</h1>
<h2>13. Video Wall Technical Requirements</h2>
<p>The controller shall comply with IEC 62443-4-2.</p>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Brightness</td><td>700 nits</td></tr>
</table>
"""


def _tree() -> LayoutTree:
    overlay = {
        "schema_version": 2,
        "extractor": "chandra",
        "content_hash": "",
        "pages": [{"page_no": 1, "markup": _MARKUP, "markup_format": "html", "degradations": []}],
    }
    return parse_layout(overlay, _DOC_ID)


def _doc(tenant: str = "tokens") -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id=tenant,
        doc_id=_DOC_ID,
        source_id="s1",
        source_kind=SourceKind.UPLOAD,
        mime="application/pdf",
        ingested_at=datetime(2026, 10, 21, tzinfo=UTC),
    )


class _Counting:
    """A `LocalHashEmbedder` that can also tokenize, like TEI and unlike bge-m3's `/embed` facade.

    One token per whitespace-separated word, so an expected count is readable from the text itself.
    Every call is recorded: "the v1 path does not even ask" is part of the contract.
    """

    def __init__(self, *, answer: bool = True) -> None:
        from rag_clients.memory import LocalHashEmbedder

        self._inner = LocalHashEmbedder()
        self.asked: list[list[str]] = []
        self._answer = answer

    @property
    def dim(self) -> int:
        return self._inner.dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._inner.embed(texts)

    async def count_tokens(self, texts: list[str]) -> list[int] | None:
        self.asked.append(list(texts))
        return [len(t.split()) for t in texts] if self._answer else None


def _svc(embedder: Any) -> tuple[Any, Any]:
    from svc_indexing import IndexingService

    from rag_clients.memory import InMemoryCatalog, InMemorySparseIndex, InMemoryVectorStore

    catalog = InMemoryCatalog()
    return IndexingService(embedder, InMemoryVectorStore(), InMemorySparseIndex(), catalog), catalog


async def test_a_v2_chunk_carries_the_embedders_count_of_its_own_text(inmemory: None) -> None:
    embedder = _Counting()
    svc, catalog = _svc(embedder)
    await svc.index_document(_doc(), "irrelevant", layout=_tree(), mask=lambda t: t)

    chunks = await catalog.document_chunks("tokens", _DOC_ID)
    assert chunks, "the layout path produced no chunks"
    assert all(c.level is not None for c in chunks)
    # The count describes `text`, the stored and citable body — not the contextualized string that
    # is actually embedded, which is per-KB config and, with an LLM contextualizer, not even
    # reproducible. So it is checkable straight off the row.
    for c in chunks:
        assert c.token_count == len(c.text.split()), c.text
    assert len(embedder.asked) == 1, "one batched ask per document, not one per chunk"


async def test_an_embedder_that_cannot_tokenize_leaves_it_unset(inmemory: None) -> None:
    """NULL reads as "not known". An estimate would read as a measurement, and nothing downstream
    could tell the difference — which is why there is no fallback here at all."""
    from rag_clients.memory import LocalHashEmbedder

    svc, catalog = _svc(LocalHashEmbedder())  # no `count_tokens` at all, like bge-m3
    await svc.index_document(_doc(), "irrelevant", layout=_tree(), mask=lambda t: t)
    chunks = await catalog.document_chunks("tokens", _DOC_ID)
    assert chunks and all(c.token_count is None for c in chunks)

    # ...and the same when it HAS the method but the server cannot answer.
    svc, catalog = _svc(_Counting(answer=False))
    await svc.index_document(_doc(), "irrelevant", layout=_tree(), mask=lambda t: t)
    assert all(c.token_count is None for c in await catalog.document_chunks("tokens", _DOC_ID))


async def test_the_v1_path_carries_no_count_and_does_not_even_ask(inmemory: None) -> None:
    """`token_count` is a plan §1.1 v2 column. Writing it onto a v1 chunk would also put a new key
    in every record this build serialises, which the previous image's `extra="forbid"` Chunk cannot
    read — the same one-way door the `exclude_none` fix closed."""
    embedder = _Counting()
    svc, catalog = _svc(embedder)
    text = "1. Scope. The contractor shall supply the system. 2. Delivery. Within ninety days."
    await svc.index_document(_doc(), text, strategy=ChunkStrategy.CLAUSE)

    chunks = await catalog.document_chunks("tokens", _DOC_ID)
    assert chunks
    assert all(c.level is None and c.token_count is None for c in chunks)
    assert embedder.asked == [], "the v1 path must make no tokenizer call at all"
