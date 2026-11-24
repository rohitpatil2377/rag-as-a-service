"""Slice-2b named dense+sparse+colbert store — the pure builder + the in-memory round-trip.

The Qdrant named-collection kwargs are asserted structurally (no live client — the real
upsert/query round-trip needs a live Qdrant and is an integration test). The in-memory twin
proves the sparse dot-product + ColBERT MaxSim rescore behave, so fusion/retrieval can be tested
in CI without infra.
"""

from __future__ import annotations

import asyncio

import pytest
from qdrant_client import models

from rag_clients.memory import InMemoryVectorStore, LocalHashEmbedder
from rag_clients.qdrant_store import _multivector_collection_kwargs
from rag_common import Settings
from rag_core import MultiVectorOutput
from rag_schemas import Chunk
from rag_schemas.chunk import Span


def _settings(**over: object) -> Settings:
    base = {"qdrant_quantization": "none", "qdrant_on_disk": False, "qdrant_shards": 1, "qdrant_replication": 1}
    base.update(over)
    return Settings(**base)  # type: ignore[arg-type]


def _chunk(cid: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d1", tenant_id="t1", text=cid, span=Span(start=0, end=1))


def test_upsert_multi_on_legacy_collection_raises_clear_error() -> None:
    """HIGH#2 mismatch guard: a named-vector upsert against an unnamed/legacy collection surfaces a
    clear 'not migrated' ConfigError, not a raw Qdrant 4xx — without needing a live Qdrant."""
    from httpx import Headers
    from qdrant_client.http.exceptions import UnexpectedResponse

    from rag_clients.qdrant_store import QdrantVectorStore
    from rag_common import ConfigError

    vs = QdrantVectorStore(namespace=lambda t: "legacy_col")

    class _Stub:
        async def upsert(self, **kw: object) -> None:
            raise UnexpectedResponse(400, "Bad Request", b"Wrong input: not existing vector name: dense", Headers({}))

    vs._client = _Stub()  # type: ignore[assignment]
    out = MultiVectorOutput(dense=[0.1, 0.2], sparse={1: 1.0}, colbert=[[0.1, 0.2]])
    with pytest.raises(ConfigError, match="not migrated"):
        asyncio.run(vs.upsert_multi("t1", [_chunk("c1")], [out]))


def test_named_collection_has_dense_colbert_maxsim_and_idf_sparse() -> None:
    kw = _multivector_collection_kwargs(1024, _settings())
    vc = kw["vectors_config"]
    assert set(vc) == {"dense", "colbert"}
    assert vc["dense"].size == 1024 and vc["dense"].distance == models.Distance.COSINE
    # ColBERT: MaxSim multivector, rescore-only (m=0), storage-bounded datatype + on_disk.
    assert vc["colbert"].multivector_config.comparator == models.MultiVectorComparator.MAX_SIM
    assert vc["colbert"].hnsw_config.m == 0
    assert vc["colbert"].on_disk is True
    assert vc["colbert"].datatype == models.Datatype.FLOAT16
    assert kw["sparse_vectors_config"]["sparse"].modifier == models.Modifier.IDF


def test_twin_dense_space_equivalence() -> None:
    """The twin's dense from ``embed`` equals its dense from ``embed_multi`` — the CI stand-in for
    the /embed vs /embeddings dense equivalence the live integration gate proves."""
    emb = LocalHashEmbedder(dim=64)
    [dense_only] = asyncio.run(emb.embed(["alpha beta gamma"]))
    [multi] = asyncio.run(emb.embed_multi(["alpha beta gamma"]))
    assert dense_only == multi.dense


def test_twin_dense_search_is_the_named_equivalent() -> None:
    vs = InMemoryVectorStore()
    emb = LocalHashEmbedder(dim=64)
    outs = asyncio.run(emb.embed_multi(["alpha beta", "gamma delta"]))
    asyncio.run(vs.upsert_multi("t1", [_chunk("c1"), _chunk("c2")], outs))
    [q] = asyncio.run(emb.embed(["alpha beta"]))
    ranked = asyncio.run(vs.search_dense("t1", q, 5))
    assert ranked[0][0] == "c1"


def test_twin_sparse_dot_product_matches_shared_tokens() -> None:
    vs = InMemoryVectorStore()
    emb = LocalHashEmbedder(dim=64)
    outs = asyncio.run(emb.embed_multi(["alpha beta gamma", "zeta eta theta"]))
    asyncio.run(vs.upsert_multi("t1", [_chunk("c1"), _chunk("c2")], outs))
    [q] = asyncio.run(emb.embed_multi(["alpha beta"]))
    ranked = asyncio.run(vs.search_sparse("t1", q.sparse, 5))
    assert ranked and ranked[0][0] == "c1"
    assert all(cid != "c2" for cid, _ in ranked)  # no shared tokens → no match


def test_twin_colbert_maxsim_rescores_the_dense_pool() -> None:
    vs = InMemoryVectorStore()
    emb = LocalHashEmbedder(dim=64)
    outs = asyncio.run(emb.embed_multi(["alpha beta gamma delta", "alpha zeta eta theta"]))
    asyncio.run(vs.upsert_multi("t1", [_chunk("c1"), _chunk("c2")], outs))
    [q] = asyncio.run(emb.embed_multi(["alpha beta gamma"]))
    ranked = asyncio.run(vs.search_colbert("t1", q.dense, q.colbert, 5, prefetch=10))
    assert ranked[0][0] == "c1"  # shares 3 query tokens vs c2's 1


def test_twin_multivector_survives_delete_and_isolation() -> None:
    vs = InMemoryVectorStore()
    emb = LocalHashEmbedder(dim=64)
    outs = asyncio.run(emb.embed_multi(["alpha beta"]))
    asyncio.run(vs.upsert_multi("t1", [_chunk("c1")], outs))
    asyncio.run(vs.upsert_multi("t2", [_chunk("c1")], outs))
    asyncio.run(vs.delete_chunks("t1", ["c1"]))
    [q] = asyncio.run(emb.embed_multi(["alpha beta"]))
    assert asyncio.run(vs.search_sparse("t1", q.sparse, 5)) == []
    assert asyncio.run(vs.search_sparse("t2", q.sparse, 5))  # tenant t2 untouched


def test_empty_signals_no_op() -> None:
    vs = InMemoryVectorStore()
    assert asyncio.run(vs.search_sparse("t1", {}, 5)) == []
    assert asyncio.run(vs.search_colbert("t1", [0.0], [], 5, prefetch=10)) == []


def test_empty_unnamed_collection_is_recreated_named_then_upsert_multi_succeeds() -> None:
    """Fresh-KB path: an EMPTY unnamed collection (as ``ensure_namespaces`` makes at KB creation)
    is recreated to the named multivector schema by ``ensure_multivector_collection``, so the first
    ``upsert_multi`` succeeds instead of hitting the 'not migrated' ConfigError."""
    vs = InMemoryVectorStore()
    emb = LocalHashEmbedder(dim=64)
    # KB creation: unnamed single-vector collection, empty.
    asyncio.run(vs.ensure_collection("t1", 64))
    # Attaching bge_m3_multi as the FIRST embedder → next ingest flips to the multivector path.
    outs = asyncio.run(emb.embed_multi(["alpha beta"]))
    asyncio.run(vs.ensure_multivector_collection("t1", 64))  # empty unnamed → recreated named
    asyncio.run(vs.upsert_multi("t1", [_chunk("c1")], outs))  # must NOT raise
    [q] = asyncio.run(emb.embed_multi(["alpha beta"]))
    assert asyncio.run(vs.search_dense("t1", q.dense, 5))[0][0] == "c1"
    assert asyncio.run(vs.search_sparse("t1", q.sparse, 5))[0][0] == "c1"


def test_populated_unnamed_collection_is_not_dropped_and_still_requires_reindex() -> None:
    """Data-loss safety: a POPULATED unnamed collection (>=1 indexed point) is NEVER auto-dropped.
    ``ensure_multivector_collection`` leaves it intact and ``upsert_multi`` still raises the clear
    reindex-required ConfigError — the populated migration is the deliberate embedder_switch.apply."""
    from rag_common import ConfigError

    vs = InMemoryVectorStore()
    emb = LocalHashEmbedder(dim=64)
    # Legacy dense-only KB with real indexed content.
    asyncio.run(vs.ensure_collection("t1", 64))
    [vec] = asyncio.run(emb.embed(["existing indexed content"]))
    asyncio.run(vs.upsert("t1", [_chunk("legacy1")], [vec]))
    assert asyncio.run(vs.search("t1", vec, 5))[0][0] == "legacy1"  # point exists

    # Flag flips on but the populated collection was never migrated.
    asyncio.run(vs.ensure_multivector_collection("t1", 64))  # must NOT drop the collection

    # The indexed point survived (no data loss).
    survived = asyncio.run(vs.search("t1", vec, 5))
    assert survived and survived[0][0] == "legacy1"

    # And a named-vector upsert still raises the clear reindex-required error.
    outs = asyncio.run(emb.embed_multi(["new content"]))
    with pytest.raises(ConfigError, match="not migrated"):
        asyncio.run(vs.upsert_multi("t1", [_chunk("c2")], outs))
    # The pre-existing point is STILL there after the refused upsert.
    assert asyncio.run(vs.search("t1", vec, 5))[0][0] == "legacy1"


def test_already_named_collection_is_a_no_op() -> None:
    """An already-named multivector collection is left untouched by ``ensure_multivector_collection``
    (no drop/recreate), and its indexed content survives."""
    vs = InMemoryVectorStore()
    emb = LocalHashEmbedder(dim=64)
    asyncio.run(vs.ensure_multivector_collection("t1", 64))  # create named
    outs = asyncio.run(emb.embed_multi(["alpha beta"]))
    asyncio.run(vs.upsert_multi("t1", [_chunk("c1")], outs))
    asyncio.run(vs.ensure_multivector_collection("t1", 64))  # no-op — must not drop c1
    [q] = asyncio.run(emb.embed_multi(["alpha beta"]))
    assert asyncio.run(vs.search_dense("t1", q.dense, 5))[0][0] == "c1"
