"""HybridRetriever cross-modal leg — a wired VisionEmbedder + vision store adds a THIRD
fusion leg, so an image chunk retrievable only by visual similarity reaches the candidate
pool. Default-off: with no vision seam the retriever behaves exactly as the text-only path,
and a vision-backend fault degrades to text-only rather than taking retrieval down.
"""

from __future__ import annotations

import asyncio

from rag_core.hybrid import ChunkLookup, HybridRetriever
from rag_schemas import Chunk, RetrievalQuery, RetrievedSpan
from rag_schemas.chunk import Span

_TENANT = "acme"


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d1", tenant_id=_TENANT, text=text, span=Span(start=0, end=len(text)))


# c0..c1 are text hits; v0 is an image chunk ONLY the cross-modal leg surfaces.
_CHUNKS = {f"c{i}": _chunk(f"c{i}", f"passage number {i}") for i in range(2)}
_CHUNKS["v0"] = _chunk("v0", "[Image description] a red invoice with a total in the corner")


class _Embedder:
    dim = 3

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class _Store:
    def __init__(self, hits: list[tuple[str, float]]) -> None:
        self._hits = hits

    async def search(self, tenant_id: str, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        return self._hits


class _VisionEmbedder:
    """Records the query it was asked to embed cross-modally."""

    dim = 4

    def __init__(self) -> None:
        self.text_calls: list[list[str]] = []

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        self.text_calls.append(list(texts))
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    async def embed_image(self, images: list[bytes]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in images]


class _Boom:
    async def search(self, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        raise RuntimeError("vision store down")


async def _lookup(tenant_id: str, ids: list[str]) -> dict[str, Chunk]:
    return {cid: _CHUNKS[cid] for cid in ids if cid in _CHUNKS}


class _Rerank:
    async def rerank(self, query: str, spans: list[RetrievedSpan], top_k: int) -> list[RetrievedSpan]:
        return [s.model_copy(update={"score": 0.7, "scorer": "cross_encoder"}) for s in spans][:top_k]


def _query() -> RetrievalQuery:
    return RetrievalQuery(tenant_id=_TENANT, text="which invoice is red?", top_k=5)


def test_cross_modal_leg_surfaces_an_image_only_chunk() -> None:
    """With the vision seam wired, an image chunk the text legs never return is retrieved by
    the visual leg — and the query IS embedded cross-modally to do it."""
    text_hits = [("c0", 0.9), ("c1", 0.8)]
    vemb = _VisionEmbedder()
    r = HybridRetriever(
        _Embedder(),
        _Store(text_hits),  # type: ignore[arg-type]
        _Store(text_hits),  # type: ignore[arg-type]
        _Rerank(),
        ChunkLookup(_lookup),
        vision_embedder=vemb,
        vision_store=_Store([("v0", 0.95)]),  # type: ignore[arg-type]
    )
    spans, max_conf = asyncio.run(r.retrieve(_query()))
    ids = {s.chunk_id for s in spans}
    assert "v0" in ids, "the image-only chunk must be surfaced by the cross-modal leg"
    assert vemb.text_calls == [["which invoice is red?"]], "the query must be embedded cross-modally"
    assert max_conf == 0.7


def test_without_vision_seam_the_image_only_chunk_is_not_found() -> None:
    """Default-off: no vision seam → the text-only path never sees the image chunk."""
    text_hits = [("c0", 0.9), ("c1", 0.8)]
    r = HybridRetriever(
        _Embedder(),
        _Store(text_hits),  # type: ignore[arg-type]
        _Store(text_hits),  # type: ignore[arg-type]
        _Rerank(),
        ChunkLookup(_lookup),
    )
    spans, _ = asyncio.run(r.retrieve(_query()))
    assert "v0" not in {s.chunk_id for s in spans}


def test_a_broken_vision_leg_degrades_to_text_only() -> None:
    """Cross-modal recall is additive: a vision-backend fault must not take down text retrieval."""
    text_hits = [("c0", 0.9), ("c1", 0.8)]
    r = HybridRetriever(
        _Embedder(),
        _Store(text_hits),  # type: ignore[arg-type]
        _Store(text_hits),  # type: ignore[arg-type]
        _Rerank(),
        ChunkLookup(_lookup),
        vision_embedder=_VisionEmbedder(),
        vision_store=_Boom(),  # type: ignore[arg-type]
    )
    spans, _ = asyncio.run(r.retrieve(_query()))
    assert {s.chunk_id for s in spans} == {"c0", "c1"}, "text results must still return when the vision leg errors"
