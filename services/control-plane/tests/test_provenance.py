"""Provider-side provenance signing — SBOM, residency attestation, receipt verify wiring."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from svc_control_plane import provenance
from svc_control_plane.compliance import verify_node_attestation

from rag_common import get_settings
from rag_common.model_sbom import verify_sbom
from rag_common.trust_receipt import CitedSpan, Contributor, TrustReceipt, build_receipt, sign_receipt
from rag_schemas import GroundedAnswer
from rag_schemas.citation import Citation
from rag_schemas.common import AnswerStatus, AuditAction

_AT = datetime(2026, 1, 1, tzinfo=UTC)
_MODELS = [
    {
        "name": "bge-small",
        "role": "embed",
        "source_origin": "India",
        "license": "MIT",
        "weight_sha256": "a" * 64,
        "revision": "1",
    }
]


@pytest.fixture(autouse=True)
def _clear_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_dev_key_is_deterministic() -> None:
    assert provenance.signing_key_b64() == provenance.signing_key_b64()
    assert provenance.public_key_b64()  # derivable from the private half


def test_signed_sbom_none_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MODEL_SBOM_MODELS", "")
    get_settings.cache_clear()
    assert provenance.signed_sbom() is None


def test_signed_sbom_signs_and_verifies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MODEL_SBOM_MODELS", json.dumps(_MODELS))
    get_settings.cache_clear()
    sbom = provenance.signed_sbom(now=_AT)
    assert sbom is not None
    assert sbom.models[0].name == "bge-small"
    assert verify_sbom(sbom, provenance.public_key_b64())
    # Tamper a license -> signature no longer verifies.
    bad = sbom.model_copy(update={"models": [sbom.models[0].model_copy(update={"license": "GPL"})]})
    assert not verify_sbom(bad, provenance.public_key_b64())


def test_bad_sbom_entry_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MODEL_SBOM_MODELS", json.dumps([*_MODELS, {"name": "x"}]))  # 2nd missing fields
    get_settings.cache_clear()
    sbom = provenance.signed_sbom(now=_AT)
    assert sbom is not None
    assert [m.name for m in sbom.models] == ["bge-small"]  # invalid entry dropped, valid kept


async def test_signed_residency_signs_and_verifies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_RESIDENCY_NODE_ID", "node-7")
    monkeypatch.setenv("RAG_RESIDENCY_REGION", "India")
    get_settings.cache_clear()
    att = await provenance.signed_residency(kb_ids=["bank__deals"], now=_AT)
    assert att.node_id == "node-7"
    assert att.no_foreign_egress is True
    assert verify_node_attestation(att, provenance.public_key_b64())
    bad = att.model_copy(update={"region": "US"})
    assert not verify_node_attestation(bad, provenance.public_key_b64())


def test_verify_supplied_receipt_roundtrip() -> None:
    receipt = build_receipt(
        tenant_id="bank",
        answer_id="a1",
        query="what is the rate",
        answer_text="the rate is 7 percent",
        model_id="sarvam-105b",
        model_weight_sha256="b" * 64,
        contributors=[Contributor(dept_id="bank", grant_id=None, kb_ids=["bank__deals"])],
        cited_spans=[CitedSpan(text="rate 7%", chunk_id="c1")],
        prev_audit_hash=None,
        issued_at=_AT,
        signer_kid=provenance.signer_kid(),
        receipt_id="r1",
    )
    signed = sign_receipt(receipt, provenance.signing_key_b64())
    assert provenance.verify_supplied_receipt(signed)
    tampered = signed.model_copy(update={"answer_sha256": "0" * 64})
    assert not provenance.verify_supplied_receipt(tampered)


# --- Hot-path emit -------------------------------------------------------------------


class _StubAudit:
    """Captures audit.record calls without a real sink."""

    def __init__(self) -> None:
        self.events: list[tuple[AuditAction, dict[str, Any]]] = []

    async def record(
        self,
        *,
        tenant_id: str,
        trace_id: str | None,
        action: AuditAction,
        actor: str,
        payload: dict[str, Any],
    ) -> str:
        self.events.append((action, payload))
        return "event-hash-123"


def _grounded() -> GroundedAnswer:
    return GroundedAnswer(
        tenant_id="bank",
        query="what is the rate",
        status=AnswerStatus.GROUNDED,
        answer="the rate is 7 percent",
        citations=[Citation(chunk_id="c1", doc_id="d1", quote="rate 7%", score=0.9)],
        confidence=0.9,
        trace_id="trace-1",
        model="sarvam-105b",
    )


async def test_emit_disabled_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_TRUST_RECEIPTS_ENABLED", "false")
    get_settings.cache_clear()
    audit = _StubAudit()
    assert await provenance.emit_answer_receipt(audit, _grounded(), actor="u") is None  # type: ignore[arg-type]
    assert audit.events == []


async def test_emit_records_verifiable_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_TRUST_RECEIPTS_ENABLED", "true")
    get_settings.cache_clear()
    audit = _StubAudit()
    result = await provenance.emit_answer_receipt(audit, _grounded(), actor="u")  # type: ignore[arg-type]
    assert result == "event-hash-123"
    assert len(audit.events) == 1
    action, payload = audit.events[0]
    assert action is AuditAction.RECEIPT
    receipt = TrustReceipt(**payload["receipt"])
    assert provenance.verify_supplied_receipt(receipt)
    assert receipt.answer_id == "trace-1"
    assert receipt.contributors[0].dept_id == "bank"
    assert receipt.model_id == "sarvam-105b"


async def test_emit_skips_abstain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_TRUST_RECEIPTS_ENABLED", "true")
    get_settings.cache_clear()
    audit = _StubAudit()
    abstained = GroundedAnswer(
        tenant_id="bank",
        query="q",
        status=AnswerStatus.ABSTAINED,
        answer="cannot answer",
        citations=[],
        confidence=0.0,
        trace_id="trace-2",
        model=None,
    )
    assert await provenance.emit_answer_receipt(audit, abstained, actor="u") is None  # type: ignore[arg-type]
    assert audit.events == []
