"""Qdrant upsert batching — one document must not become one oversized request.

The regression: ``upsert`` sent every point of a document in a single call. Qdrant caps a
request at ``max_request_size_mb`` (32 by default) and answers an oversized one with a 400
while the client is still uploading, so httpx raises a bare ``ReadError``, ingest returns 500,
and NOTHING is indexed — with no mention of size in any error. Measured on a real 80-page
tender: RAG v2 ingest produced 1801 points (1425 of them table rows) at ~20 KB each = 36.9 MB
in the one and only upsert call, so that document could never be ingested at all.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_clients.qdrant_store import QdrantVectorStore
from rag_schemas import Chunk


class _RecordingClient:
    """Stands in for AsyncQdrantClient and records the size of each upsert request."""

    def __init__(self) -> None:
        self.batches: list[int] = []

    async def upsert(self, collection_name: str, points: list[Any]) -> None:
        self.batches.append(len(points))


def _chunks(n: int) -> list[Chunk]:
    return [Chunk(chunk_id=f"c{i}", doc_id="d", tenant_id="t", text=f"chunk {i}", sequence=i) for i in range(n)]


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> tuple[QdrantVectorStore, _RecordingClient]:
    monkeypatch.setenv("RAG_QDRANT_UPSERT_BATCH", "256")
    from rag_common import config

    config.get_settings.cache_clear()
    st = QdrantVectorStore(url="http://localhost:6333")
    client = _RecordingClient()
    st._client = client  # type: ignore[assignment]
    return st, client


async def test_a_large_document_is_split_into_several_requests(
    store: tuple[QdrantVectorStore, _RecordingClient],
) -> None:
    st, client = store
    n = 1801  # the real count from the tender that exposed this
    await st.upsert("t", _chunks(n), [[0.1] * 8 for _ in range(n)])

    assert sum(client.batches) == n, "every point must still be sent exactly once"
    assert max(client.batches) <= 256, f"a request carried {max(client.batches)} points"
    assert len(client.batches) == 8  # 7 full + remainder


async def test_a_small_document_is_still_one_request(
    store: tuple[QdrantVectorStore, _RecordingClient],
) -> None:
    """Batching must not add round-trips to the ordinary case."""
    st, client = store
    await st.upsert("t", _chunks(10), [[0.1] * 8 for _ in range(10)])
    assert client.batches == [10]


async def test_an_empty_upsert_sends_nothing(
    store: tuple[QdrantVectorStore, _RecordingClient],
) -> None:
    st, client = store
    await st.upsert("t", [], [])
    assert client.batches == []
