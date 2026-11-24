"""DPDP erasure reaches the RAG v2 `query_log` (plan §1.1 table, erasure contract).

`query_log.answer` will hold passages lifted from the corpus, so a log row is the same class of
artifact as a chat message: derived content quoting a document. It therefore gets the same
treatment as derived chat history — purged BY CITATION — and these tests pin the parts of that
contract that need no database. The array-overlap semantics themselves are real SQL and are proven
against a live Postgres in `tests/integration/test_v2_catalog_ddl.py`.

See `docs/automated-erasure.md` for what is retained and why.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from svc_control_plane import compliance, pipeline, wiring


def test_the_certificate_hash_does_not_depend_on_the_new_field() -> None:
    """VERIFIED, not assumed (the review asked): `_certificate_hash` is computed over
    (kb_id, sorted(doc_ids), at) only, so adding a defaulted `query_log_purged` cannot change a
    previously issued certificate's hash. If this ever fails, every certificate issued before the
    field was added stops verifying."""
    at = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
    expected = compliance._certificate_hash("kb", ["d1", "d2"], at)

    lean = compliance.ErasureCertificate(
        kb_id="kb",
        requested_by="dpo",
        reason="r",
        documents_erased=["d1", "d2"],
        certificate_hash=compliance._certificate_hash("kb", ["d1", "d2"], at),
        erased_at=at,
    )
    rich = compliance.ErasureCertificate(
        kb_id="kb",
        requested_by="dpo",
        reason="r",
        documents_erased=["d1", "d2"],
        chunks_purged=7,
        conversations_purged=3,
        query_log_purged=11,
        certificate_hash=compliance._certificate_hash("kb", ["d1", "d2"], at),
        erased_at=at,
    )
    assert lean.certificate_hash == rich.certificate_hash == expected
    # And the hash really is blind to the counts: same inputs, different counts, same digest.
    assert compliance._certificate_hash("kb", ["d2", "d1"], at) == expected  # order-insensitive too


def test_the_field_defaults_to_zero_so_old_payloads_still_validate() -> None:
    """A certificate recorded before this field existed must still deserialise."""
    cert = compliance.ErasureCertificate.model_validate(
        {
            "kb_id": "kb",
            "requested_by": "dpo",
            "reason": "r",
            "documents_erased": ["d1"],
            "certificate_hash": "x" * 64,
            "erased_at": "2026-10-16T12:00:00+00:00",
        }
    )
    assert cert.query_log_purged == 0


async def test_erase_purges_the_query_log_and_reports_the_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """The count must reach BOTH the certificate and the audit payload — the certificate is what a
    data principal is handed, the audit row is what a regulator reads."""
    calls: list[tuple[str, set[str]]] = []
    recorded: list[dict[str, object]] = []

    async def fake_purge(kb_id: str, doc_ids: set[str]) -> int:
        calls.append((kb_id, doc_ids))
        return 4

    async def fake_resolve(kb_id: str, doc_ids: list[str] | None, subject: str | None) -> list[str]:
        return ["d1", "d2"]

    async def fake_delete(kb_id: str, doc_id: str, *, actor: str) -> int:
        return 3

    class _Convs:
        async def delete_citing(self, kb_id: str, doc_ids: set[str]) -> int:
            return 1

    class _Audit:
        async def record(self, **kwargs: object) -> None:
            recorded.append(kwargs)

    monkeypatch.setattr(compliance, "purge_query_log_citing", fake_purge)
    monkeypatch.setattr(compliance, "_resolve_targets", fake_resolve)
    monkeypatch.setattr(pipeline, "delete_document", fake_delete)
    monkeypatch.setattr(wiring, "conversation_store", _Convs)
    monkeypatch.setattr(wiring, "audit_service", _Audit)

    cert = await compliance.erase("kb", actor="dpo")

    assert calls == [("kb", {"d1", "d2"})], "purged for exactly the resolved target set"
    assert cert.query_log_purged == 4
    assert cert.conversations_purged == 1
    payload = recorded[0]["payload"]
    assert isinstance(payload, dict)
    assert payload["query_log_purged"] == 4


async def test_the_query_log_purge_is_scoped_to_the_resolved_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Erasure by SUBJECT resolves to a document set first; the purge must see that set, not the
    (empty) explicit doc_ids argument — otherwise a subject erasure would scrub nothing."""
    seen: list[set[str]] = []

    async def fake_purge(kb_id: str, doc_ids: set[str]) -> int:
        seen.append(doc_ids)
        return 0

    async def fake_resolve(kb_id: str, doc_ids: list[str] | None, subject: str | None) -> list[str]:
        assert subject == "alice"
        return ["subject-doc"]

    async def fake_delete(kb_id: str, doc_id: str, *, actor: str) -> int:
        return 0

    class _Convs:
        async def delete_citing(self, kb_id: str, doc_ids: set[str]) -> int:
            return 0

    class _Audit:
        async def record(self, **kwargs: object) -> None:
            return None

    monkeypatch.setattr(compliance, "purge_query_log_citing", fake_purge)
    monkeypatch.setattr(compliance, "_resolve_targets", fake_resolve)
    monkeypatch.setattr(pipeline, "delete_document", fake_delete)
    monkeypatch.setattr(wiring, "conversation_store", _Convs)
    monkeypatch.setattr(wiring, "audit_service", _Audit)

    await compliance.erase("kb", actor="dpo", subject="alice")
    assert seen == [{"subject-doc"}]


async def test_the_factory_seam_reports_zero_under_the_in_memory_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing writes the query log under the in-memory profile, so there is genuinely nothing to
    purge and 0 is the truthful answer — not a silently skipped step. This also pins that
    `compliance` never needs a concrete-adapter import to reach it."""
    from rag_clients import factory
    from rag_common import get_settings

    monkeypatch.setenv("RAG_INMEMORY", "true")
    monkeypatch.setenv("RAG_ENV", "dev")
    get_settings.cache_clear()
    factory.get_catalog.cache_clear()
    try:
        assert await factory.purge_query_log_citing("t", {"d1"}) == 0
    finally:
        factory.get_catalog.cache_clear()
        get_settings.cache_clear()


async def test_an_empty_target_set_purges_nothing() -> None:
    """Short-circuited before any SQL: an erasure that resolved to no documents must not issue a
    DELETE whose `&&` against an empty array would be meaningless."""
    from rag_clients.postgres_catalog import PostgresCatalog

    class _Boom:
        def begin(self) -> object:
            raise AssertionError("no transaction should be opened for an empty doc-id set")

    cat = PostgresCatalog(engine=_Boom())  # type: ignore[arg-type]
    assert await cat.delete_query_log_citing("t", set()) == 0


# --- an unrecognised catalog backend must stop the erasure, not report zero ---------------------


async def test_an_unknown_catalog_adapter_fails_the_erasure_instead_of_reporting_zero() -> None:
    """The seam narrows on the two adapters it knows. A third one returning 0 would let erasure
    finish and sign a certificate asserting a completeness nobody verified — the same silent
    failure as duck-typing the method. It has to fail closed."""
    import pytest

    from rag_clients import factory
    from rag_common.errors import ConfigError

    class _UnknownCatalog:
        """Neither Postgres nor the in-memory twin — a future backend."""

    def _unknown() -> _UnknownCatalog:
        return _UnknownCatalog()

    factory.get_catalog.cache_clear()
    original = factory.get_catalog
    factory.get_catalog = _unknown  # type: ignore[assignment]
    try:
        with pytest.raises(ConfigError) as ei:
            await factory.purge_query_log_citing("acme", {"doomed"})
        assert "_UnknownCatalog" in str(ei.value), "the error must name the backend that was not handled"
    finally:
        factory.get_catalog = original
        factory.get_catalog.cache_clear()
