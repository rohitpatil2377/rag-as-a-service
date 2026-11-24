"""Unified KB — NoSQL (MongoDB) connector: same governance/shape contract as test_unified_kb.py's
SQL twin, but exercised against a fake NoSqlClient so no driver install / live DB is required."""

from __future__ import annotations

from typing import Any

import pytest
from svc_control_plane import nosql_query
from svc_control_plane.nosql_query import (
    NoSqlConnectionInfo,
    clear_nosql_connection,
    get_nosql_connection,
    nosql_evidence_span,
    render_nosql_evidence,
    set_nosql_connection,
)

from rag_common.errors import ValidationError
from rag_schemas.common import Modality


class _FakeCatalog:
    """Minimal in-memory stand-in for CatalogAdapter's tenant_config surface."""

    def __init__(self) -> None:
        self._cfg: dict[str, dict[str, str]] = {}

    async def get_tenant_config(self, tenant_id: str) -> dict[str, str]:
        return dict(self._cfg.get(tenant_id, {}))

    async def set_tenant_config(self, tenant_id: str, key: str, value: str) -> None:
        self._cfg.setdefault(tenant_id, {})[key] = value


class _FakeClient:
    """A NoSqlClient that never imports/needs motor. `docs_by_collection` maps collection name ->
    documents it "contains"; `find` does a naive keyword filter, mirroring the real client's
    read-only, bounded contract."""

    def __init__(self, uri: str, docs_by_collection: dict[str, list[dict[str, Any]]]) -> None:
        self.uri = uri
        self._docs = docs_by_collection

    async def find(self, database: str, collection: str, terms: list[str], limit: int) -> list[dict[str, Any]]:
        docs = self._docs.get(collection, [])
        if terms:
            docs = [d for d in docs if any(t.lower() in str(d).lower() for t in terms)]
        return docs[:limit]


class _BoomClient:
    async def find(self, database: str, collection: str, terms: list[str], limit: int) -> list[dict[str, Any]]:
        raise RuntimeError("mongo down")


@pytest.fixture(autouse=True)
def _fake_catalog(monkeypatch: pytest.MonkeyPatch) -> _FakeCatalog:
    catalog = _FakeCatalog()
    monkeypatch.setattr(nosql_query, "get_catalog", lambda: catalog)
    return catalog


@pytest.fixture(autouse=True)
def _fake_crypto(monkeypatch: pytest.MonkeyPatch) -> None:
    """tenant_crypto round-trips as an identity no-op here — the real envelope-encryption
    behaviour is covered by tenant_crypto's own tests; this module only needs "store secret,
    get secret back"."""

    async def _encrypt(_kb: str, plaintext: str) -> str:
        return f"enc:{plaintext}"

    async def _decrypt(_kb: str, ciphertext: str) -> str:
        return ciphertext.removeprefix("enc:")

    monkeypatch.setattr(nosql_query.tenant_crypto, "encrypt", _encrypt)  # type: ignore[attr-defined]
    monkeypatch.setattr(nosql_query.tenant_crypto, "decrypt", _decrypt)  # type: ignore[attr-defined]


# --- render_nosql_evidence --------------------------------------------------------------------


def test_render_nosql_evidence_truncation() -> None:
    docs = [{"_id": i, "name": f"row{i}"} for i in range(25)]
    text = render_nosql_evidence(docs, max_docs=20)
    assert "25 document(s)" in text
    assert "... (5 more document(s))" in text


# --- set / get / clear round-trip ---------------------------------------------------------------


async def test_set_requires_at_least_one_collection() -> None:
    with pytest.raises(ValidationError):
        await set_nosql_connection("kb1", "mongodb://localhost:27017", "salesdb", [])


async def test_set_get_clear_round_trip() -> None:
    before = await get_nosql_connection("kb1")
    assert before == NoSqlConnectionInfo(configured=False)

    info = await set_nosql_connection("kb1", "mongodb://localhost:27017", "salesdb", ["deals", "accounts"])
    assert info.configured is True
    assert info.database == "salesdb"
    assert info.collections == ["accounts", "deals"]
    assert info.host == "localhost"

    fetched = await get_nosql_connection("kb1")
    assert fetched == info

    await clear_nosql_connection("kb1")
    after = await get_nosql_connection("kb1")
    assert after == NoSqlConnectionInfo(configured=False)


# --- nosql_evidence_span ------------------------------------------------------------------------


async def test_span_none_when_not_configured() -> None:
    assert await nosql_evidence_span("kb1", "how many deals", actor="u") is None


async def test_span_built_from_matching_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    await set_nosql_connection("kb1", "mongodb://localhost:27017", "salesdb", ["deals"])
    fake = _FakeClient("mongodb://localhost:27017", {"deals": [{"_id": "1", "name": "Acme deal"}]})
    monkeypatch.setattr(nosql_query, "_client_factory", lambda _uri: fake)

    span = await nosql_evidence_span("kb1", "Acme", actor="u")
    assert span is not None
    assert span.scorer == "exact"
    assert span.score == 1.0
    assert span.modality is Modality.DATABASE
    assert span.doc_id == "nosql://kb1"
    assert span.chunk_id == "nosql::kb1"
    assert span.source_uri == "nosql://kb1"
    assert span.category == "database"
    assert "Acme deal" in span.text


async def test_span_none_on_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    await set_nosql_connection("kb1", "mongodb://localhost:27017", "salesdb", ["deals"])
    fake = _FakeClient("mongodb://localhost:27017", {"deals": []})
    monkeypatch.setattr(nosql_query, "_client_factory", lambda _uri: fake)

    assert await nosql_evidence_span("kb1", "nonexistent thing", actor="u") is None


async def test_span_failsafe_on_driver_error(monkeypatch: pytest.MonkeyPatch) -> None:
    await set_nosql_connection("kb1", "mongodb://localhost:27017", "salesdb", ["deals"])
    monkeypatch.setattr(nosql_query, "_client_factory", lambda _uri: _BoomClient())

    assert await nosql_evidence_span("kb1", "q", actor="u") is None  # degrades, no raise
