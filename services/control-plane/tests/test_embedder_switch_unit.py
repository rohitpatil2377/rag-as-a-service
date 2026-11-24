"""Unit: the embedder-switch record helpers (pure logic, no stores)."""

from __future__ import annotations

from svc_control_plane.embedder_switch import is_same_embedder, parse_record, record_for
from svc_control_plane.model_connection import StoredModelConnection


def _conn(**overrides: object) -> StoredModelConnection:
    base: dict[str, object] = {
        "tenant_id": "t",
        "conn_id": "c1",
        "name": "bge-m3",
        "endpoint": "https://models.sovereign.local/v1",
        "model": "bge-m3",
        "role": "embedding",
        "embed_dim": 1024,
    }
    base.update(overrides)
    return StoredModelConnection(**base)


def test_the_signature_covers_what_defines_the_vector_space() -> None:
    """Endpoint, model and dimension define the space; name, key and conn_id do not —
    an edit that changes only those must read as the SAME embedder (no re-embed)."""
    a = record_for(_conn())
    same_space = record_for(_conn(conn_id="c2", name="renamed", key_ciphertext="env:new"))
    assert a["sig"] == same_space["sig"]
    for field, value in (("model", "bge-large"), ("endpoint", "https://other.sovereign.local/v1"), ("embed_dim", 384)):
        assert a["sig"] != record_for(_conn(**{field: value}))["sig"], field


def test_is_same_embedder_matches_on_signature_not_identity() -> None:
    record = record_for(_conn())
    assert is_same_embedder(record, _conn(conn_id="other"))
    assert not is_same_embedder(record, _conn(embed_dim=384))
    assert not is_same_embedder(None, _conn())


def test_parse_record_never_raises_on_junk() -> None:
    assert parse_record("") is None
    assert parse_record("not json") is None
    assert parse_record('"a string"') is None
    assert parse_record('{"sig": "x"}') == {"sig": "x"}
