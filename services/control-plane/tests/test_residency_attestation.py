"""Per-node residency attestation — signature round-trip, tamper detection, and a smoke check
that the existing per-KB (unsigned) attestation is untouched."""

from __future__ import annotations

import base64
from datetime import UTC, datetime

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat
from svc_control_plane.compliance import (
    Attestation,
    NodeResidencyAttestation,
    attestation,
    node_residency_attestation,
    sign_node_attestation,
    verify_node_attestation,
)


def _keypair() -> tuple[str, str]:
    k = Ed25519PrivateKey.generate()
    priv = base64.b64encode(k.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())).decode()
    pub = base64.b64encode(k.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()
    return priv, pub


async def _node_attestation(priv: str, **kw: object) -> NodeResidencyAttestation:
    base: dict[str, object] = {
        "node_id": "node-blr-01",
        "region": "India",
        "kb_ids": ["acme__hr", "acme__finance"],
        "private_key_b64": priv,
        "signer_kid": "acme-release-2026",
        "now": datetime(2026, 8, 5, tzinfo=UTC),
    }
    base.update(kw)
    return await node_residency_attestation(**base)  # type: ignore[arg-type]


async def test_sign_then_verify_roundtrips() -> None:
    priv, pub = _keypair()
    att = await _node_attestation(priv)
    assert att.signed_token != ""
    assert att.no_foreign_egress is True
    assert att.kb_ids == ["acme__finance", "acme__hr"]  # sorted
    assert verify_node_attestation(att, pub) is True


async def test_a_tampered_node_id_fails_verification() -> None:
    priv, pub = _keypair()
    att = await _node_attestation(priv)
    tampered = att.model_copy(update={"node_id": "node-evil-99"})
    assert verify_node_attestation(tampered, pub) is False


async def test_a_tampered_region_fails_verification() -> None:
    priv, pub = _keypair()
    att = await _node_attestation(priv)
    tampered = att.model_copy(update={"region": "Elsewhere"})
    assert verify_node_attestation(tampered, pub) is False


async def test_a_tampered_kb_id_fails_verification() -> None:
    priv, pub = _keypair()
    att = await _node_attestation(priv)
    tampered = att.model_copy(update={"kb_ids": [*att.kb_ids, "acme__legal"]})
    assert verify_node_attestation(tampered, pub) is False


async def test_a_signature_from_the_wrong_key_fails() -> None:
    priv, _pub = _keypair()
    _other_priv, other_pub = _keypair()
    att = await _node_attestation(priv)
    assert verify_node_attestation(att, other_pub) is False  # not the signer's key


def test_verify_never_raises_on_garbage_input() -> None:
    _priv, pub = _keypair()
    garbage = NodeResidencyAttestation(
        attestation_id="a1",
        node_id="n1",
        region="India",
        kb_ids=["kb1"],
        data_residency="India (self-hosted)",
        no_foreign_egress=True,
        generated_at=datetime(2026, 8, 5, tzinfo=UTC),
        signer_kid="k1",
        signed_token="not-base64!!",
    )
    assert verify_node_attestation(garbage, pub) is False
    assert verify_node_attestation(garbage, "also-not-base64!!") is False


async def test_sign_node_attestation_is_pure_and_returns_a_copy() -> None:
    priv, pub = _keypair()
    unsigned = await _node_attestation(priv)
    unsigned = unsigned.model_copy(update={"signed_token": ""})
    signed = sign_node_attestation(unsigned, priv)
    assert unsigned.signed_token == ""  # original untouched
    assert signed.signed_token != ""
    assert verify_node_attestation(signed, pub) is True


async def test_existing_per_kb_attestation_is_unaffected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke check: the pre-existing, unsigned per-KB ``attestation()`` still works untouched."""

    class _Settings:
        sovereign_enforce = True
        egress_allowlist: list[str] = ["*.acme.example"]  # noqa: RUF012 - test double, not a model
        federation_enabled = False

    class _Catalog:
        async def list_documents(self, kb_id: str) -> list[object]:
            return []

        async def get_tenant_config(self, kb_id: str) -> dict[str, str]:
            return {}

    import svc_control_plane.compliance as compliance_mod

    monkeypatch.setattr(compliance_mod, "get_settings", _Settings)
    monkeypatch.setattr(compliance_mod, "get_catalog", _Catalog)

    class _AuditService:
        async def list(self, kb_id: str, cursor: object, limit: int) -> list[object]:
            return []

    class _Wiring:
        @staticmethod
        def audit_service() -> _AuditService:
            return _AuditService()

    monkeypatch.setattr(compliance_mod, "wiring", _Wiring())

    result = await attestation("acme__hr", now=datetime(2026, 8, 5, tzinfo=UTC))
    assert isinstance(result, Attestation)
    assert result.data_residency == "India (self-hosted; no foreign/managed AI or cloud calls at runtime)"
    assert result.federation.enabled is False
