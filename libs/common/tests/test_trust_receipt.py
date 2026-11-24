"""Trust receipts — determinism, sign/verify round-trip, tamper-evidence, fail-closed verify."""

from __future__ import annotations

import base64
from datetime import UTC, datetime

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

from rag_common.trust_receipt import (
    CitedSpan,
    Contributor,
    TrustReceipt,
    build_receipt,
    canonical_bytes,
    sha256_hex,
    sign_receipt,
    verify_receipt,
)


def _keypair() -> tuple[str, str]:
    k = Ed25519PrivateKey.generate()
    priv = base64.b64encode(k.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())).decode()
    pub = base64.b64encode(k.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()
    return priv, pub


def _receipt() -> TrustReceipt:
    return build_receipt(
        tenant_id="acme",
        answer_id="ans-1",
        query="what is the leave policy?",
        answer_text="Employees accrue 1.5 days of leave per month.",
        model_id="approved-llm:1.0",
        model_weight_sha256="a" * 64,
        contributors=[
            Contributor(dept_id="hr", grant_id="g-42", kb_ids=["hr-handbook"]),
            Contributor(dept_id="legal", grant_id=None, kb_ids=["policies"]),
        ],
        cited_spans=[
            CitedSpan(text="Leave accrues at 1.5 days/month.", chunk_id="c1"),
            CitedSpan(text="Unused leave carries over.", chunk_id="c2"),
        ],
        prev_audit_hash="b" * 64,
        issued_at=datetime(2026, 8, 5, tzinfo=UTC),
        signer_kid="kid-1",
        receipt_id="rcpt-1",
    )


def test_build_is_deterministic() -> None:
    assert canonical_bytes(_receipt()) == canonical_bytes(_receipt())


def test_build_computes_digests() -> None:
    r = _receipt()
    assert r.query_sha256 == sha256_hex("what is the leave policy?")
    assert r.answer_sha256 == sha256_hex("Employees accrue 1.5 days of leave per month.")
    assert r.span_digests == [
        sha256_hex("Leave accrues at 1.5 days/month.|c1"),
        sha256_hex("Unused leave carries over.|c2"),
    ]
    assert r.signed_token == ""


def test_sign_then_verify_roundtrips() -> None:
    priv, pub = _keypair()
    signed = sign_receipt(_receipt(), priv)
    assert signed.signed_token != ""
    assert verify_receipt(signed, pub) is True


def test_signature_does_not_cover_itself() -> None:
    # canonical_bytes must exclude signed_token, so signing does not change what is signed.
    priv, _pub = _keypair()
    r = _receipt()
    signed = sign_receipt(r, priv)
    assert canonical_bytes(signed) == canonical_bytes(r)


def test_tampering_answer_fails_verification() -> None:
    priv, pub = _keypair()
    signed = sign_receipt(_receipt(), priv)
    tampered = signed.model_copy(update={"answer_sha256": sha256_hex("a forged answer")})
    assert verify_receipt(tampered, pub) is False


def test_tampering_contributor_grant_fails_verification() -> None:
    priv, pub = _keypair()
    signed = sign_receipt(_receipt(), priv)
    forged = [
        Contributor(dept_id="hr", grant_id="g-999", kb_ids=["hr-handbook"]),
        Contributor(dept_id="legal", grant_id=None, kb_ids=["policies"]),
    ]
    tampered = signed.model_copy(update={"contributors": forged})
    assert verify_receipt(tampered, pub) is False


def test_tampering_a_span_fails_verification() -> None:
    priv, pub = _keypair()
    signed = sign_receipt(_receipt(), priv)
    tampered = signed.model_copy(update={"span_digests": [sha256_hex("swapped|c1"), *signed.span_digests[1:]]})
    assert verify_receipt(tampered, pub) is False


def test_wrong_public_key_fails_verification() -> None:
    priv, _pub = _keypair()
    _priv2, pub2 = _keypair()
    signed = sign_receipt(_receipt(), priv)
    assert verify_receipt(signed, pub2) is False


def test_verify_never_raises_on_unsigned_receipt() -> None:
    _priv, pub = _keypair()
    assert verify_receipt(_receipt(), pub) is False


def test_verify_never_raises_on_garbage_inputs() -> None:
    signed = sign_receipt(_receipt(), _keypair()[0])
    assert verify_receipt(signed, "not-base64!!") is False
    assert verify_receipt(signed.model_copy(update={"signed_token": "not-base64!!"}), _keypair()[1]) is False
    assert verify_receipt(signed, "") is False
