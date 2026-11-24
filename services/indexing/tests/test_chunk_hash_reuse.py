"""Idempotency by `chunk_hash`: unchanged text is not embedded twice (RAG v2 Phase 6).

Content-derived ids already make the WRITES idempotent — text that survives an edit keeps its id
and is re-upserted in place. What they do not do is stop the embedder being called again for it,
and embedding is the dominant cost of an index. So a caller that knows the vectors came from the
same pipeline may skip it, which these tests pin in both directions:

  * opted in, only the changed chunks reach the embedder, and the unchanged ones keep their
    vectors and still get their catalog rows refreshed;
  * not opted in (the default for every existing caller), every chunk is embedded exactly as
    before.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from rag_common import vector_collection
from rag_schemas import DocumentEnvelope
from rag_schemas.common import ChunkStrategy, SourceKind

_DOC_ID = "night-shift-policy"
_TENANT = "reuse"

# Three clauses, each long enough to be its own chunk under the configured clause packing
# (`chunk_max_chars` 700, so two of these cannot share one), because the whole point is that an
# edit to ONE of them leaves the other two byte-identical. Padded with distinct prose rather than
# filler so a chunk can be identified by what it says.
_PAD_1 = (
    "Shift rosters are published a fortnight in advance and are circulated to every supervisor, "
    "who confirms receipt. Staff on rotating shifts are counted against the roster of the unit "
    "they report to at the start of the shift, not the unit that raised the requisition. Changes "
    "requested after publication require the written approval of the operations director and are "
    "recorded in the roster register with the reason for the change."
)
_PAD_2 = (
    "The allowance is computed per completed shift and appears as a separate line on the payslip, "
    "so it can be reconciled against the roster register without reference to any other record. "
    "Partial shifts are not counted, and a shift abandoned on medical grounds is treated as "
    "completed for the purpose of this clause provided the duty doctor certifies it in writing."
)
_PAD_3 = (
    "Continuous service is counted from the date of joining and is not broken by approved leave, "
    "by a transfer between units, or by a change of designation within the same grade. A break in "
    "service of more than thirty days restarts the count, and the employee is notified in writing "
    "when that happens so the eligibility date is never a matter of inference."
)
_CLAUSE_1 = f"1. Scope. This policy applies to all operations staff on rotating shifts. {_PAD_1}"
_CLAUSE_2_OLD = f"2. Allowance. The night allowance is Rs. 200 per shift, paid monthly. {_PAD_2}"
_CLAUSE_2_NEW = f"2. Allowance. The night allowance is Rs. 400 per shift, paid monthly. {_PAD_2}"
_CLAUSE_3 = f"3. Eligibility. Six months of continuous service is required. {_PAD_3}"

_V1 = f"{_CLAUSE_1}\n{_CLAUSE_2_OLD}\n{_CLAUSE_3}\n"
_V2 = f"{_CLAUSE_1}\n{_CLAUSE_2_NEW}\n{_CLAUSE_3}\n"

# The v2 (layout) shape of the same question: two sections, then the same document with a THIRD
# section inserted ahead of them. Discrete blocks and no overlap, so the two original sections stay
# byte-identical while their position in the document moves.
_PAGE_1 = "<h1>Attachment-1</h1><h2>13. Video wall</h2><p>The controller shall comply with IEC 62443-4-2.</p>"
_PAGE_1_WITH_PREAMBLE = (
    "<h1>Attachment-1</h1>"
    "<h2>12. Preamble</h2><p>Issued under corrigendum five of this tender.</p>"
    "<h2>13. Video wall</h2><p>The controller shall comply with IEC 62443-4-2.</p>"
)
_PAGE_2 = "<h2>14. Power</h2><p>Dual redundant feeds shall be provided to each rack.</p>"


def _tree(page_1: str) -> Any:
    from rag_core.layout import parse_layout

    return parse_layout(
        {
            "schema_version": 2,
            "extractor": "chandra",
            "content_hash": "",
            "pages": [
                {"page_no": 1, "markup": page_1, "markup_format": "html", "degradations": []},
                {"page_no": 2, "markup": _PAGE_2, "markup_format": "html", "degradations": []},
            ],
        },
        _DOC_ID,
    )


class _CountingEmbedder:
    """Records every batch it is asked to embed, so "was this text embedded again?" is a fact."""

    def __init__(self) -> None:
        from rag_clients.memory import LocalHashEmbedder

        self._inner = LocalHashEmbedder()
        self.batches: list[list[str]] = []

    @property
    def dim(self) -> int:
        return self._inner.dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return await self._inner.embed(texts)

    @property
    def embedded(self) -> list[str]:
        return [t for batch in self.batches for t in batch]


def _doc() -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id=_TENANT,
        doc_id=_DOC_ID,
        source_id="s1",
        source_kind=SourceKind.UPLOAD,
        mime="text/plain",
        ingested_at=datetime(2026, 10, 21, tzinfo=UTC),
    )


def _svc(embedder: Any) -> tuple[Any, Any, Any, Any]:
    from svc_indexing import IndexingService

    from rag_clients.memory import InMemoryCatalog, InMemorySparseIndex, InMemoryVectorStore

    vs, sparse, catalog = InMemoryVectorStore(), InMemorySparseIndex(), InMemoryCatalog()
    return IndexingService(embedder, vs, sparse, catalog), vs, sparse, catalog


async def test_only_the_edited_region_is_embedded_again(inmemory: None) -> None:
    """A mid-document edit: one of the three chunks survives it byte-identical and is not embedded
    again.

    ONE, not two, and that is a property of the v1 clause chunker rather than a shortfall here: it
    packs prose up to `chunk_max_chars` across clause boundaries and carries a 150-character
    overlap, so an edit perturbs the chunk holding it AND its neighbour. Measured on this fixture,
    not assumed. The v2 layout path has no overlap and discrete blocks, which is why its reuse is
    strictly better — see the sequence test below, where 2 of 3 survive a bigger change.
    """
    embedder = _CountingEmbedder()
    svc, vs, _sparse, catalog = _svc(embedder)

    first = await svc.index_document(_doc(), _V1, strategy=ChunkStrategy.CLAUSE)
    assert first.chunk_count == 3
    assert first.reused_chunks == 0, "a first index has nothing to reuse"
    assert len(embedder.embedded) == 3

    embedder.batches.clear()
    second = await svc.index_document(_doc(), _V2, strategy=ChunkStrategy.CLAUSE, reuse_unchanged=True)

    assert second.chunk_count == 3
    assert second.reused_chunks == 1
    # Asserted on CONTENT, not just count: the edited figure must be in what was embedded, so a
    # change that re-embedded the wrong chunks could not pass this.
    assert len(embedder.embedded) == 2
    assert any("Rs. 400" in t for t in embedder.embedded)
    assert not any("Rs. 200" in t for t in embedder.embedded)

    # Nothing was lost: the chunk that was NOT re-embedded still has its vector, which is the
    # whole safety question. (The collection also still holds the two superseded points — purging
    # those is the pipeline's write-then-swap, a layer above this one, so the count here is 5.)
    [reused_id] = set(first.chunk_ids) & set(second.chunk_ids)
    collection = vs._c[vector_collection(_TENANT)]
    assert reused_id in collection
    assert len(collection) == 5, "index_document never purges; the swap above it does"
    stored = await catalog.get_chunks(_TENANT, second.chunk_ids)
    assert any("Rs. 400" in c.text for c in stored.values())


async def test_an_appended_clause_leaves_the_earlier_chunks_alone(inmemory: None) -> None:
    """The corrigendum shape — text added at the END — which is where reuse pays most on a v1
    corpus: the earlier chunks are untouched, so two thirds of this document is not re-embedded."""
    embedder = _CountingEmbedder()
    svc, _vs, _sparse, _catalog = _svc(embedder)

    await svc.index_document(_doc(), _V1, strategy=ChunkStrategy.CLAUSE)
    embedder.batches.clear()
    appended = _V1 + (
        "4. Review. This policy is reviewed annually by the operations committee together with the "
        "human resources department, and any change is notified in writing to every affected "
        "employee before it takes effect.\n"
    )
    result = await svc.index_document(_doc(), appended, strategy=ChunkStrategy.CLAUSE, reuse_unchanged=True)

    assert result.reused_chunks == 2
    assert len(embedder.embedded) == result.chunk_count - 2


async def test_a_reused_chunk_still_has_its_record_refreshed(inmemory: None) -> None:
    """Unchanged TEXT is not an unchanged RECORD, and this is the case that proves it: a new section
    inserted AHEAD of two others leaves their chunks byte-identical — same ids, no re-embed — while
    their `sequence`, the only ordering key a v2 chunk has, moves 0->1 and 1->2. Had the reused
    chunks' rows been skipped along with their vectors, the document would read back in the wrong
    order for the citation viewer, whole-document answering and neighbour expansion alike.
    """
    embedder = _CountingEmbedder()
    svc, _vs, _sparse, catalog = _svc(embedder)

    await svc.index_document(_doc(), "unused", layout=_tree(_PAGE_1), mask=lambda t: t)
    before = {c.chunk_id: c.sequence for c in await catalog.document_chunks(_TENANT, _DOC_ID)}
    assert sorted(before.values()) == [0, 1]

    embedder.batches.clear()
    result = await svc.index_document(
        _doc(), "unused", layout=_tree(_PAGE_1_WITH_PREAMBLE), mask=lambda t: t, reuse_unchanged=True
    )

    assert result.reused_chunks == 2, "both original sections are word for word what they were"
    assert len(embedder.embedded) == 1, "only the inserted section reached the embedder"
    after = await catalog.get_chunks(_TENANT, result.chunk_ids)
    shifted = {cid: after[cid].sequence for cid in result.chunk_ids if cid in before}
    assert sorted(shifted.values()) == [1, 2], "a reused chunk's record was NOT refreshed"


async def test_without_opting_in_everything_is_embedded_exactly_as_before(inmemory: None) -> None:
    """The default path must be byte-identical: no catalog read for hashes, no skip, every chunk
    embedded. This is what keeps the feature from changing any caller that has not asked for it."""
    embedder = _CountingEmbedder()
    svc, _vs, _sparse, _catalog = _svc(embedder)

    await svc.index_document(_doc(), _V1, strategy=ChunkStrategy.CLAUSE)
    embedder.batches.clear()
    again = await svc.index_document(_doc(), _V1, strategy=ChunkStrategy.CLAUSE)

    assert again.reused_chunks == 0
    assert len(embedder.embedded) == 3, "re-embedded, as it always was"


async def test_a_document_with_no_stored_hashes_reuses_nothing(inmemory: None) -> None:
    """Rows written before `chunk_hash` existed come back without one. "No hash" must read as
    "cannot be reused" — the safe direction, costing a re-embed rather than a stale vector."""
    embedder = _CountingEmbedder()
    svc, _vs, _sparse, catalog = _svc(embedder)

    class _Hashless:
        def __getattr__(self, name: str) -> Any:
            return getattr(catalog, name)

        async def chunk_hashes(self, tenant_id: str, doc_id: str) -> dict[str, str]:
            return {}

    await svc.index_document(_doc(), _V1, strategy=ChunkStrategy.CLAUSE)
    svc._catalog = _Hashless()
    embedder.batches.clear()
    result = await svc.index_document(_doc(), _V1, strategy=ChunkStrategy.CLAUSE, reuse_unchanged=True)

    assert result.reused_chunks == 0
    assert len(embedder.embedded) == 3
