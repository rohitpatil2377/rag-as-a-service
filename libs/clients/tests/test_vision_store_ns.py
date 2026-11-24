"""The cross-modal image index is the SAME VectorStore adapter, namespaced to a SEPARATE
per-tenant collection — so image vectors never collide with the text collection."""

from __future__ import annotations

import asyncio

from rag_clients.memory import InMemoryVectorStore
from rag_common import vector_collection, vision_collection
from rag_schemas import Chunk
from rag_schemas.chunk import Span

_TENANT = "acme"


def _chunk(cid: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d1", tenant_id=_TENANT, text="x", span=Span(start=0, end=1))


def test_vision_collection_is_a_distinct_namespace() -> None:
    assert vision_collection(_TENANT) != vector_collection(_TENANT)
    assert vision_collection(_TENANT).endswith("_vchunks")


def test_namespaced_store_isolates_image_vectors_from_text() -> None:
    """A vision-namespaced store and a text-namespaced store over the same class do not see each
    other's vectors, even for the same tenant + chunk id — separate collections."""

    async def _run() -> None:
        text = InMemoryVectorStore()  # default namespace = vector_collection
        vision = InMemoryVectorStore(namespace=vision_collection)
        await text.ensure_collection(_TENANT, 3)
        await vision.ensure_collection(_TENANT, 4)
        await text.upsert(_TENANT, [_chunk("d1::0")], [[1.0, 0.0, 0.0]])
        await vision.upsert(_TENANT, [_chunk("d1::0")], [[0.0, 1.0, 0.0, 0.0]])
        # Each store returns only its own vector for the shared chunk id.
        assert await text.search(_TENANT, [1.0, 0.0, 0.0], 5) == [("d1::0", 1.0)]
        assert (await vision.search(_TENANT, [0.0, 1.0, 0.0, 0.0], 5))[0][0] == "d1::0"
        # Purging one leaves the other intact.
        await vision.delete_chunks(_TENANT, ["d1::0"])
        assert await vision.search(_TENANT, [0.0, 1.0, 0.0, 0.0], 5) == []
        assert await text.search(_TENANT, [1.0, 0.0, 0.0], 5) == [("d1::0", 1.0)]

    asyncio.run(_run())
