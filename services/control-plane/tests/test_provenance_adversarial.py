"""ADVERSARIAL provenance tests — trying to BREAK trust receipts / SBOM / residency for go-live.

Not shipped intentionally; run with:
    uv run pytest services/control-plane/tests/test_provenance_adversarial.py -q
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat
from svc_control_plane import provenance
from svc_control_plane.compliance import (
    NodeResidencyAttestation,
    _node_canonical_bytes,
    node_residency_attestation,
    verify_node_attestation,
)

from rag_common import get_settings
from rag_common.model_sbom import ModelEntry, ModelSBOM, sign_sbom, verify_sbom
from rag_common.trust_receipt import (
    CitedSpan,
    Contributor,
    TrustReceipt,
    build_receipt,
    sha256_hex,
    sign_receipt,
    verify_receipt,
)
from rag_schemas import GroundedAnswer
from rag_schemas.citation import Citation
from rag_schemas.common import AnswerStatus, AuditAction

_AT = datetime(2026, 8, 5, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _clear_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _keypair() -> tuple[str, str]:
    k = Ed25519PrivateKey.generate()
    priv = base64.b64encode(k.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())).decode()
    pub = base64.b64encode(k.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()
    return priv, pub


def _m(name: str, role: str, wh: str = "a") -> dict[str, str]:
    """A compact SBOM model dict for env config."""
    return {"name": name, "role": role, "source_origin": "India", "license": "MIT", "weight_sha256": wh * 64}


def _receipt(**kw: object) -> TrustReceipt:
    base: dict[str, object] = {
        "tenant_id": "acme",
        "answer_id": "ans-1",
        "query": "q",
        "answer_text": "a",
        "model_id": "m",
        "model_weight_sha256": "a" * 64,
        "contributors": [Contributor(dept_id="hr")],
        "cited_spans": [CitedSpan(text="t", chunk_id="c1")],
        "prev_audit_hash": None,
        "issued_at": _AT,
        "signer_kid": "kid-1",
        "receipt_id": "r1",
    }
    base.update(kw)
    return build_receipt(**base)  # type: ignore[arg-type]


# ============================================================================
# A. JSON round-trip fidelity — the classic break: a signed receipt sent over
#    the wire as JSON, parsed back, must STILL verify. This is exactly what
#    /receipts/verify and /receipts/{id} do (TrustReceipt(**payload)).
# ============================================================================


def test_receipt_survives_full_json_roundtrip() -> None:
    """A signed receipt serialized to JSON and reparsed (the endpoint path) must verify."""
    priv, pub = _keypair()
    signed = sign_receipt(_receipt(), priv)
    wire = json.loads(signed.model_dump_json())
    reparsed = TrustReceipt(**wire)
    assert verify_receipt(reparsed, pub) is True


def test_receipt_survives_roundtrip_with_prev_audit_hash_and_grant() -> None:
    priv, pub = _keypair()
    signed = sign_receipt(
        _receipt(
            prev_audit_hash="b" * 64,
            contributors=[Contributor(dept_id="hr", grant_id="g1", kb_ids=["k1", "k2"])],
        ),
        priv,
    )
    reparsed = TrustReceipt(**json.loads(signed.model_dump_json()))
    assert verify_receipt(reparsed, pub) is True


def test_sbom_survives_full_json_roundtrip() -> None:
    priv, pub = _keypair()
    sbom = sign_sbom(
        ModelSBOM(
            sbom_id="s1",
            deployment_id="d1",
            models=[ModelEntry(name="m", role="embed", source_origin="India", license="MIT", weight_sha256="a" * 64)],
            generated_at=_AT,
            signer_kid="k1",
        ),
        priv,
    )
    reparsed = ModelSBOM(**json.loads(sbom.model_dump_json()))
    assert verify_sbom(reparsed, pub) is True


async def test_residency_survives_full_json_roundtrip() -> None:
    priv, pub = _keypair()
    att = await node_residency_attestation(
        node_id="n1", region="India", kb_ids=["a", "b"], private_key_b64=priv, signer_kid="k1", now=_AT
    )
    reparsed = NodeResidencyAttestation(**json.loads(att.model_dump_json()))
    assert verify_node_attestation(reparsed, pub) is True


# ============================================================================
# B. TIMEZONE / datetime canonicalization — same instant, different tz repr.
#    If a receipt's issued_at is expressed in +05:30 vs Z it is the SAME instant
#    but a DIFFERENT string; sha/JSON commits to the string. Does a receipt
#    signed in UTC survive being re-expressed in another offset for the same instant?
# ============================================================================


def test_same_instant_different_tz_offset_changes_canonical_bytes() -> None:
    """DOCUMENTS a sharp edge: signature commits to the tz STRING, not the instant.
    A receipt re-expressed in +05:30 (same instant) will NOT verify — verifiers must
    preserve the exact serialized datetime, not just the instant."""
    priv, pub = _keypair()
    ist = timezone(timedelta(hours=5, minutes=30))
    utc_receipt = _receipt(issued_at=datetime(2026, 8, 5, 0, 0, tzinfo=UTC))
    signed = sign_receipt(utc_receipt, priv)
    # same instant, expressed in IST
    ist_same_instant = signed.model_copy(update={"issued_at": datetime(2026, 8, 5, 5, 30, tzinfo=ist)})
    # It is the same moment in time...
    assert signed.issued_at == ist_same_instant.issued_at
    # ...but the signature does NOT survive the reserialization (string differs).
    assert verify_receipt(ist_same_instant, pub) is False


def test_naive_datetime_receipt_signs_and_verifies() -> None:
    """A naive (tz-less) issued_at is accepted and round-trips — no crash, still verifies."""
    priv, pub = _keypair()
    signed = sign_receipt(_receipt(issued_at=datetime(2026, 8, 5, 12, 0)), priv)
    reparsed = TrustReceipt(**json.loads(signed.model_dump_json()))
    assert verify_receipt(reparsed, pub) is True


# ============================================================================
# C. Tamper on EVERY committed field of a receipt (task: any field tampered -> False)
# ============================================================================


@pytest.mark.parametrize(
    "field,newval",
    [
        ("answer_sha256", "0" * 64),
        ("query_sha256", "0" * 64),
        ("model_id", "evil-model"),
        ("model_weight_sha256", "0" * 64),
        ("tenant_id", "attacker"),
        ("answer_id", "other-answer"),
        ("receipt_id", "other-receipt"),
        ("signer_kid", "other-kid"),
        ("prev_audit_hash", "c" * 64),
    ],
)
def test_tamper_any_scalar_field_breaks_receipt(field: str, newval: str) -> None:
    priv, pub = _keypair()
    signed = sign_receipt(_receipt(prev_audit_hash="b" * 64), priv)
    tampered = signed.model_copy(update={field: newval})
    assert verify_receipt(tampered, pub) is False


def test_tamper_citations_list_breaks_receipt() -> None:
    priv, pub = _keypair()
    signed = sign_receipt(_receipt(), priv)
    # add a forged span digest
    tampered = signed.model_copy(update={"span_digests": [*signed.span_digests, sha256_hex("forged|cX")]})
    assert verify_receipt(tampered, pub) is False
    # drop a span
    tampered2 = signed.model_copy(update={"span_digests": []})
    assert verify_receipt(tampered2, pub) is False


def test_tamper_contributors_list_breaks_receipt() -> None:
    priv, pub = _keypair()
    signed = sign_receipt(_receipt(), priv)
    tampered = signed.model_copy(update={"contributors": [Contributor(dept_id="attacker-dept")]})
    assert verify_receipt(tampered, pub) is False


# ============================================================================
# D. Signature malleability / empty-token — verify_receipt (unlike verify_sbom)
#    has NO explicit `if not signed_token` guard. Prove empty/garbage still False.
# ============================================================================


def test_empty_signed_token_fails_receipt() -> None:
    priv, pub = _keypair()
    signed = sign_receipt(_receipt(), priv)
    assert verify_receipt(signed.model_copy(update={"signed_token": ""}), pub) is False


def test_truncated_signature_fails_receipt() -> None:
    priv, pub = _keypair()
    signed = sign_receipt(_receipt(), priv)
    raw = base64.b64decode(signed.signed_token)
    truncated = base64.b64encode(raw[:-1]).decode()
    assert verify_receipt(signed.model_copy(update={"signed_token": truncated}), pub) is False


def test_all_zero_signature_fails_receipt() -> None:
    _priv, pub = _keypair()
    zero = base64.b64encode(b"\x00" * 64).decode()
    assert verify_receipt(_receipt().model_copy(update={"signed_token": zero}), pub) is False


def test_bitflip_in_signature_fails_receipt() -> None:
    priv, pub = _keypair()
    signed = sign_receipt(_receipt(), priv)
    raw = bytearray(base64.b64decode(signed.signed_token))
    raw[0] ^= 0x01
    flipped = base64.b64encode(bytes(raw)).decode()
    assert verify_receipt(signed.model_copy(update={"signed_token": flipped}), pub) is False


# ============================================================================
# E. Cross-artifact / cross-key confusion.
# ============================================================================


def test_receipt_signed_by_one_key_rejected_by_another() -> None:
    priv1, _pub1 = _keypair()
    _priv2, pub2 = _keypair()
    signed = sign_receipt(_receipt(), priv1)
    assert verify_receipt(signed, pub2) is False


def test_signing_with_public_key_bytes_produces_non_verifying_receipt() -> None:
    """A raw Ed25519 public key is also 32 bytes, so from_private_bytes ACCEPTS it (no raise).
    Security property that actually matters: the resulting receipt does NOT verify under the
    real signer's public key — an attacker cannot forge by feeding the wrong key material."""
    _priv, pub = _keypair()
    forged = sign_receipt(_receipt(), pub)  # misuse: pub bytes used as a private seed
    assert verify_receipt(forged, pub) is False


# ============================================================================
# F. Canonicalization equivalence across the THREE verifiers.
#    receipt/sbom use model_dump(mode=json).pop(); node uses exclude={signed_token}.
#    They must all strip the token identically (no divergence a re-serializer trips on).
# ============================================================================


def test_node_canonical_excludes_signed_token_like_the_others() -> None:
    priv, _pub = _keypair()
    att = NodeResidencyAttestation(
        attestation_id="a1",
        node_id="n1",
        region="India",
        kb_ids=["k"],
        data_residency="India",
        no_foreign_egress=True,
        generated_at=_AT,
        signer_kid="k1",
    )
    before = _node_canonical_bytes(att)
    signed = sign_node_attestation_local(att, priv)
    after = _node_canonical_bytes(signed)
    assert before == after  # signing must not change what is signed


def sign_node_attestation_local(att: NodeResidencyAttestation, priv: str) -> NodeResidencyAttestation:
    from svc_control_plane.compliance import sign_node_attestation

    return sign_node_attestation(att, priv)


# ============================================================================
# G. Hot-path emit: FLAGGED_FOR_REVIEW MUST emit a receipt (task requirement),
#    ABSTAINED must not. Plus contributors from federation coverage.
# ============================================================================


class _StubAudit:
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
        return "evt-1"


async def test_emit_receipt_for_flagged_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_TRUST_RECEIPTS_ENABLED", "true")
    get_settings.cache_clear()
    audit = _StubAudit()
    flagged = GroundedAnswer(
        tenant_id="bank",
        query="q",
        status=AnswerStatus.FLAGGED_FOR_REVIEW,
        answer="maybe 7 percent",
        citations=[Citation(chunk_id="c1", doc_id="d1", quote="rate 7%", score=0.6)],
        confidence=0.5,
        trace_id="trace-flag",
        model="sarvam",
    )
    h = await provenance.emit_answer_receipt(audit, flagged, actor="u")  # type: ignore[arg-type]
    assert h == "evt-1"
    assert len(audit.events) == 1
    action, payload = audit.events[0]
    assert action is AuditAction.RECEIPT
    receipt = TrustReceipt(**payload["receipt"])
    assert provenance.verify_supplied_receipt(receipt)


async def test_emit_skips_grounded_with_no_citations_is_impossible_but_abstain_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        trace_id="trace-ab",
        model=None,
    )
    assert await provenance.emit_answer_receipt(audit, abstained, actor="u") is None  # type: ignore[arg-type]
    assert audit.events == []


# ============================================================================
# H. TRUST HOLE probe: an answer whose model is NOT in the SBOM gets
#    weight "unverified" but the receipt STILL signs+verifies as genuine.
#    Document whether a verifier can tell weights were never attested.
# ============================================================================


async def test_receipt_for_unlisted_model_has_unverified_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_TRUST_RECEIPTS_ENABLED", "true")
    monkeypatch.setenv("RAG_MODEL_SBOM_MODELS", json.dumps([_m("approved-only", "generate")]))
    get_settings.cache_clear()
    audit = _StubAudit()
    ans = GroundedAnswer(
        tenant_id="bank",
        query="q",
        status=AnswerStatus.GROUNDED,
        answer="x",
        citations=[Citation(chunk_id="c1", doc_id="d1", quote="q", score=0.9)],
        confidence=0.9,
        trace_id="t",
        model="some-unlisted-model",
    )
    await provenance.emit_answer_receipt(audit, ans, actor="u")  # type: ignore[arg-type]
    receipt = TrustReceipt(**audit.events[0][1]["receipt"])
    # A genuine, verifiable receipt whose weight hash is the literal string "unverified":
    assert receipt.model_weight_sha256 == "unverified"
    assert provenance.verify_supplied_receipt(receipt) is True


# ============================================================================
# I. SBOM: invalid entries dropped; ALL-invalid -> None (not an empty signed SBOM).
# ============================================================================


def test_sbom_all_entries_invalid_yields_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MODEL_SBOM_MODELS", json.dumps([{"name": "x"}, {"bogus": 1}]))
    get_settings.cache_clear()
    assert provenance.signed_sbom(now=_AT) is None


def test_sbom_malformed_json_yields_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MODEL_SBOM_MODELS", "{not json")
    get_settings.cache_clear()
    assert provenance.signed_sbom(now=_AT) is None


def test_sbom_bad_role_entry_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MODEL_SBOM_MODELS", json.dumps([_m("good", "embed"), _m("bad", "translate", "b")]))
    get_settings.cache_clear()
    sbom = provenance.signed_sbom(now=_AT)
    assert sbom is not None
    assert [m.name for m in sbom.models] == ["good"]
    assert verify_sbom(sbom, provenance.public_key_b64())


# ============================================================================
# J. residency no_foreign_egress is signed — flipping it breaks verification.
# ============================================================================


async def test_flip_no_foreign_egress_breaks_residency() -> None:
    priv, pub = _keypair()
    att = await node_residency_attestation(
        node_id="n1", region="India", kb_ids=["a"], private_key_b64=priv, signer_kid="k1", now=_AT
    )
    tampered = att.model_copy(update={"no_foreign_egress": False})
    assert verify_node_attestation(tampered, pub) is False


async def test_flip_data_residency_string_breaks_residency() -> None:
    priv, pub = _keypair()
    att = await node_residency_attestation(
        node_id="n1", region="India", kb_ids=["a"], private_key_b64=priv, signer_kid="k1", now=_AT
    )
    tampered = att.model_copy(update={"data_residency": "US (self-hosted)"})
    assert verify_node_attestation(tampered, pub) is False
