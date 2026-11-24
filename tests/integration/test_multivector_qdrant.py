"""Slice-2b real-Qdrant round-trip (design §9) — runs against a LIVE Qdrant only.

Marked ``integration`` so the default unit gate skips it; the reviewer runs it against the
docker-compose Qdrant on the box. Proves the named-dict upsert is accepted, that each leg
(dense / learned-sparse / ColBERT Prefetch+MaxSim) returns the planted best chunk, that a reindex
into the named collection preserves content-derived ids without purging, and that a named upsert
against a legacy UNNAMED collection raises the clear "not migrated" error (not a raw Qdrant 400).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable

import pytest

from rag_common import ConfigError, get_settings
from rag_core import MultiVectorOutput
from rag_schemas import Chunk
from rag_schemas.chunk import Span

pytestmark = pytest.mark.integration

_DIM = 8


def _ns(suffix: str) -> Callable[[str], str]:
    # A unique collection name per test run so repeats don't collide; ignores tenant_id (isolation
    # is exercised by test_tenant_isolation — here we just need a private collection).
    name = f"mv_it_{suffix}_{uuid.uuid4().hex[:8]}"
    return lambda _tenant: name


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d1", tenant_id="t1", text=text, span=Span(start=0, end=len(text)))


def _out(dense: list[float], sparse: dict[int, float], colbert: list[list[float]]) -> MultiVectorOutput:
    return MultiVectorOutput(dense=dense, sparse=sparse, colbert=colbert)


@pytest.fixture
async def store() -> AsyncIterator[object]:
    """A QdrantVectorStore against the configured live Qdrant; skip if it is unreachable."""
    from rag_clients.qdrant_store import QdrantVectorStore

    vs = QdrantVectorStore(namespace=_ns("dense"))
    try:
        await vs._client.get_collections()
    except Exception as exc:
        await vs.aclose()
        pytest.skip(f"live Qdrant not reachable ({type(exc).__name__}); integration test skipped")
    yield vs
    try:
        await vs.delete_tenant("t1")
    finally:
        await vs.aclose()


async def test_named_upsert_and_each_leg_round_trips(store: object) -> None:
    await store.ensure_multivector_collection("t1", _DIM)  # type: ignore[attr-defined]
    # c0 is closest on dense AND shares the sparse token AND shares ColBERT tokens.
    c0 = _out([1.0] + [0.0] * (_DIM - 1), {101: 0.9}, [[1.0] + [0.0] * (_DIM - 1)])
    c1 = _out([0.0, 1.0] + [0.0] * (_DIM - 2), {202: 0.9}, [[0.0, 1.0] + [0.0] * (_DIM - 2)])
    await store.upsert_multi("t1", [_chunk("c0", "alpha"), _chunk("c1", "beta")], [c0, c1])  # type: ignore[attr-defined]

    dense = await store.search_dense("t1", [1.0] + [0.0] * (_DIM - 1), 5)  # type: ignore[attr-defined]
    assert dense and dense[0][0] == "c0"
    sparse = await store.search_sparse("t1", {101: 1.0}, 5)  # type: ignore[attr-defined]
    assert sparse and sparse[0][0] == "c0"
    colbert = await store.search_colbert(  # type: ignore[attr-defined]
        "t1", [1.0] + [0.0] * (_DIM - 1), [[1.0] + [0.0] * (_DIM - 1)], 5, prefetch=10
    )
    assert colbert and colbert[0][0] == "c0"


async def test_reindex_into_named_collection_preserves_ids_without_purge(store: object) -> None:
    await store.ensure_multivector_collection("t1", _DIM)  # type: ignore[attr-defined]
    c0 = _out([1.0] + [0.0] * (_DIM - 1), {101: 0.9}, [[1.0] + [0.0] * (_DIM - 1)])
    await store.upsert_multi("t1", [_chunk("c0", "alpha")], [c0])  # type: ignore[attr-defined]
    # Re-upsert the SAME content-derived id (a write-then-swap reindex): it updates in place, and
    # "old - fresh" is empty so nothing is deleted — the chunk stays retrievable throughout.
    await store.upsert_multi("t1", [_chunk("c0", "alpha")], [c0])  # type: ignore[attr-defined]
    dense = await store.search_dense("t1", [1.0] + [0.0] * (_DIM - 1), 5)  # type: ignore[attr-defined]
    assert [cid for cid, _ in dense] == ["c0"]


async def test_named_upsert_against_legacy_collection_raises_clear_error() -> None:
    """A named-vector upsert on an UNNAMED (legacy dense-only) collection must raise the clear
    'not migrated' error, not a raw Qdrant 400."""
    from rag_clients.qdrant_store import QdrantVectorStore

    vs = QdrantVectorStore(namespace=_ns("legacy"))
    try:
        await vs._client.get_collections()
    except Exception as exc:
        await vs.aclose()
        pytest.skip(f"live Qdrant not reachable ({type(exc).__name__})")
    try:
        await vs.ensure_collection("t1", _DIM)  # UNNAMED, single-vector (legacy shape)
        out = _out([1.0] + [0.0] * (_DIM - 1), {101: 0.9}, [[1.0] + [0.0] * (_DIM - 1)])
        with pytest.raises(ConfigError, match="not migrated"):
            await vs.upsert_multi("t1", [_chunk("c0", "alpha")], [out])
    finally:
        await vs.delete_tenant("t1")
        await vs.aclose()


def test_settings_dim_is_available() -> None:
    # Sanity that settings load in the integration env (no assertion on the value).
    assert get_settings().embed_dim > 0
