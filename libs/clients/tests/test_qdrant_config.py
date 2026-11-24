"""Qdrant at-scale collection config — pure builders (no live client).

Covers the quantization / on-disk / sharding wiring: defaults must reproduce the original
single-node full-precision collection exactly; each setting must map to the right Qdrant
model. See docs/production-sizing.md.
"""

from __future__ import annotations

from qdrant_client import models

from rag_clients.qdrant_store import _collection_kwargs, _quantization_config, _search_params
from rag_common import Settings


def _settings(**over: object) -> Settings:
    base = {
        "qdrant_quantization": "none",
        "qdrant_on_disk": False,
        "qdrant_shards": 1,
        "qdrant_replication": 1,
    }
    base.update(over)
    return Settings(**base)  # type: ignore[arg-type]


def test_defaults_reproduce_original_collection() -> None:
    kw = _collection_kwargs(384, _settings())
    vp = kw["vectors_config"]
    assert isinstance(vp, models.VectorParams)
    assert vp.size == 384
    assert vp.distance == models.Distance.COSINE
    assert vp.on_disk is False
    assert kw["quantization_config"] is None
    assert kw["shard_number"] == 1
    assert kw["replication_factor"] == 1
    # No search-param override when quantization is off.
    assert _search_params(_settings()) is None


def test_scalar_quantization() -> None:
    q = _quantization_config("scalar")
    assert isinstance(q, models.ScalarQuantization)
    assert q.scalar.type == models.ScalarType.INT8
    assert q.scalar.always_ram is True
    kw = _collection_kwargs(1024, _settings(qdrant_quantization="scalar"))
    assert isinstance(kw["quantization_config"], models.ScalarQuantization)
    sp = _search_params(_settings(qdrant_quantization="scalar"))
    assert isinstance(sp, models.SearchParams)
    assert sp.quantization is not None and sp.quantization.rescore is True


def test_binary_quantization() -> None:
    q = _quantization_config("binary")
    assert isinstance(q, models.BinaryQuantization)
    assert q.binary.always_ram is True
    sp = _search_params(_settings(qdrant_quantization="binary"))
    assert isinstance(sp, models.SearchParams)
    assert sp.quantization is not None and sp.quantization.rescore is True


def test_on_disk_and_sharding() -> None:
    kw = _collection_kwargs(1024, _settings(qdrant_on_disk=True, qdrant_shards=6, qdrant_replication=2))
    assert kw["vectors_config"].on_disk is True
    assert kw["shard_number"] == 6
    assert kw["replication_factor"] == 2


def test_none_quantization_is_none() -> None:
    assert _quantization_config("none") is None
