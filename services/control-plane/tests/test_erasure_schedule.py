"""Retention-driven automated erasure scheduler — due-document selection and the injected
``erase_fn`` seam (``svc_control_plane.erasure_schedule``)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from svc_control_plane.compliance import ErasureCertificate
from svc_control_plane.erasure_schedule import (
    RetainedDoc,
    RetentionPolicy,
    due_document_ids,
    run_scheduled_erasure,
)

_NOW = datetime(2026, 8, 5, tzinfo=UTC)


def _doc(doc_id: str, age_days: int) -> RetainedDoc:
    return RetainedDoc(doc_id=doc_id, created_at=_NOW - timedelta(days=age_days))


def _cert(kb_id: str, doc_ids: list[str]) -> ErasureCertificate:
    return ErasureCertificate(
        kb_id=kb_id,
        requested_by="scheduler",
        reason="retention_policy",
        documents_erased=doc_ids,
        certificate_hash="deadbeef",
        erased_at=_NOW,
    )


class _FakeEraser:
    """Records calls instead of touching any store — the seam under test."""

    def __init__(self, cert: ErasureCertificate) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self._cert = cert

    async def __call__(self, kb_id: str, doc_ids: list[str]) -> ErasureCertificate:
        self.calls.append((kb_id, doc_ids))
        return self._cert


def test_due_document_ids_selects_only_over_age_docs() -> None:
    docs = [_doc("young", age_days=10), _doc("old", age_days=31), _doc("older", age_days=100)]
    assert due_document_ids(now=_NOW, retention_days=30, docs=docs) == ["old", "older"]


def test_due_document_ids_boundary_exactly_retention_days_is_not_yet_due() -> None:
    docs = [_doc("exact", age_days=30)]
    assert due_document_ids(now=_NOW, retention_days=30, docs=docs) == []


def test_due_document_ids_empty_docs() -> None:
    assert due_document_ids(now=_NOW, retention_days=30, docs=[]) == []


def test_due_document_ids_result_is_sorted() -> None:
    docs = [_doc("zeta", age_days=100), _doc("alpha", age_days=100)]
    assert due_document_ids(now=_NOW, retention_days=30, docs=docs) == ["alpha", "zeta"]


async def test_disabled_policy_returns_none_and_erase_fn_not_called() -> None:
    policy = RetentionPolicy(kb_id="acme__hr", retention_days=30, enabled=False)
    docs = [_doc("old", age_days=100)]
    eraser = _FakeEraser(_cert("acme__hr", ["old"]))
    result = await run_scheduled_erasure(policy=policy, now=_NOW, docs=docs, erase_fn=eraser)
    assert result is None
    assert eraser.calls == []


async def test_no_due_docs_returns_none_and_erase_fn_not_called() -> None:
    policy = RetentionPolicy(kb_id="acme__hr", retention_days=30)
    docs = [_doc("young", age_days=5)]
    eraser = _FakeEraser(_cert("acme__hr", []))
    result = await run_scheduled_erasure(policy=policy, now=_NOW, docs=docs, erase_fn=eraser)
    assert result is None
    assert eraser.calls == []


async def test_empty_docs_returns_none_and_erase_fn_not_called() -> None:
    policy = RetentionPolicy(kb_id="acme__hr", retention_days=30)
    eraser = _FakeEraser(_cert("acme__hr", []))
    result = await run_scheduled_erasure(policy=policy, now=_NOW, docs=[], erase_fn=eraser)
    assert result is None
    assert eraser.calls == []


async def test_due_docs_calls_erase_fn_once_with_right_ids_and_returns_its_certificate() -> None:
    policy = RetentionPolicy(kb_id="acme__hr", retention_days=30)
    docs = [_doc("young", age_days=5), _doc("old", age_days=31), _doc("older", age_days=90)]
    cert = _cert("acme__hr", ["old", "older"])
    eraser = _FakeEraser(cert)
    result = await run_scheduled_erasure(policy=policy, now=_NOW, docs=docs, erase_fn=eraser)
    assert result is cert
    assert eraser.calls == [("acme__hr", ["old", "older"])]
