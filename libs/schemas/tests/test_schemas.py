"""Unit tests for the data contracts — focus on invariants that guard correctness."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError as PydanticValidationError

from rag_schemas import AnswerStatus, AuditAction, AuditEvent, Citation, GroundedAnswer


def _citation() -> Citation:
    return Citation(chunk_id="c1", doc_id="d1", quote="uptime target is 99.5%", score=0.9)


def test_grounded_answer_requires_citation() -> None:
    with pytest.raises(PydanticValidationError):
        GroundedAnswer(tenant_id="t", query="q", status=AnswerStatus.GROUNDED, answer="a", confidence=0.9)


def test_abstained_answer_forbids_citations() -> None:
    with pytest.raises(PydanticValidationError):
        GroundedAnswer(
            tenant_id="t",
            query="q",
            status=AnswerStatus.ABSTAINED,
            answer="I don't know",
            citations=[_citation()],
            confidence=0.1,
        )


def test_grounded_answer_ok_with_citation() -> None:
    ans = GroundedAnswer(
        tenant_id="t",
        query="q",
        status=AnswerStatus.GROUNDED,
        answer="99.5% baseline",
        citations=[_citation()],
        confidence=0.9,
    )
    assert ans.citations[0].quote


def test_audit_hash_chain_is_tamper_evident() -> None:
    e1 = AuditEvent(
        event_id="e1",
        tenant_id="t",
        trace_id="tr",
        action=AuditAction.QUERY,
        actor="officer",
        at=datetime(2026, 7, 2, tzinfo=UTC),
        payload={"q": "uptime?"},
    ).sealed()
    # Any change to content yields a different hash -> tamper is detectable.
    tampered = e1.model_copy(update={"payload": {"q": "changed"}})
    assert tampered.compute_hash() != e1.event_hash


# --- `filters` is the metadata channel, not a way to name a record field ------------------------


def test_a_record_field_is_rejected_as_a_metadata_filter_key() -> None:
    """`filters={"doc_id": ...}` used to be accepted and then SILENTLY under-match: the sparse leg
    resolves every filter key to `meta.<key>`, so it matched nothing there while the dense leg's
    flat payload matched the record — document-scoped retrieval ran dense-only with no BM25 and no
    error. It is now a validation failure that names the right field instead."""
    from rag_schemas import RetrievalQuery

    with pytest.raises(PydanticValidationError) as ei:
        RetrievalQuery(tenant_id="t", text="q", filters={"doc_id": "a.pdf"})
    assert "doc_ids" in str(ei.value), "the error must point the caller at the dedicated field"


@pytest.mark.parametrize("key", ["chunk_id", "doc_id", "tenant_id", "page", "acl"])
def test_every_record_field_is_rejected(key: str) -> None:
    from rag_schemas import RetrievalQuery

    with pytest.raises(PydanticValidationError):
        RetrievalQuery(tenant_id="t", text="q", filters={key: "x"})


def test_ordinary_metadata_filters_still_pass() -> None:
    """The FR-D8/B7 tag channel is untouched — only the five record fields are refused."""
    from rag_schemas import RetrievalQuery

    q = RetrievalQuery(tenant_id="t", text="q", filters={"doc_type": "policy", "lang": "en", "category": "HR"})
    assert q.filters == {"doc_type": "policy", "lang": "en", "category": "HR"}


def test_a_document_scope_uses_its_own_field() -> None:
    from rag_schemas import RetrievalQuery

    q = RetrievalQuery(tenant_id="t", text="q", doc_ids=("a.pdf", "b.pdf"), filters={"lang": "en"})
    assert q.doc_ids == ("a.pdf", "b.pdf") and q.filters == {"lang": "en"}
