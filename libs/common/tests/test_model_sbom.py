"""Model SBOM — signature and tamper-evidence guards for the sovereignty claim."""

from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat
from pydantic import ValidationError

from rag_common.model_sbom import (
    ModelEntry,
    ModelSBOM,
    entry_for_file,
    sign_sbom,
    verify_sbom,
)


def _keypair() -> tuple[str, str]:
    k = Ed25519PrivateKey.generate()
    priv = base64.b64encode(k.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())).decode()
    pub = base64.b64encode(k.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()
    return priv, pub


def _entries() -> list[ModelEntry]:
    return [
        ModelEntry(
            name="bge-m3",
            role="embed",
            source_origin="India",
            license="MIT",
            weight_sha256="a" * 64,
            revision="v1",
        ),
        ModelEntry(
            name="krutrim-7b",
            role="generate",
            source_origin="India",
            license="Apache-2.0",
            weight_sha256="b" * 64,
        ),
    ]


def _sbom(**kw: object) -> ModelSBOM:
    base: dict[str, object] = {
        "sbom_id": "sbom-1",
        "deployment_id": "dep-1",
        "models": _entries(),
        "generated_at": datetime.now(UTC),
        "signer_kid": "acme-release-2026",
    }
    base.update(kw)
    return ModelSBOM(**base)


def test_sign_then_verify_roundtrips() -> None:
    priv, pub = _keypair()
    sbom = sign_sbom(_sbom(), priv)
    assert sbom.signed_token != ""
    assert verify_sbom(sbom, pub) is True


def test_unsigned_sbom_fails_verification() -> None:
    _priv, pub = _keypair()
    assert verify_sbom(_sbom(), pub) is False


def test_a_tampered_weight_hash_fails_verification() -> None:
    priv, pub = _keypair()
    sbom = sign_sbom(_sbom(), priv)
    tampered_models = [sbom.models[0].model_copy(update={"weight_sha256": "f" * 64}), sbom.models[1]]
    tampered = sbom.model_copy(update={"models": tampered_models})
    assert verify_sbom(tampered, pub) is False


def test_a_tampered_license_fails_verification() -> None:
    priv, pub = _keypair()
    sbom = sign_sbom(_sbom(), priv)
    tampered_models = [sbom.models[0], sbom.models[1].model_copy(update={"license": "GPL-3.0"})]
    tampered = sbom.model_copy(update={"models": tampered_models})
    assert verify_sbom(tampered, pub) is False


def test_a_signature_from_the_wrong_key_fails() -> None:
    priv, _pub = _keypair()
    _other_priv, other_pub = _keypair()
    sbom = sign_sbom(_sbom(), priv)
    assert verify_sbom(sbom, other_pub) is False


def test_verify_never_raises_on_garbage_input() -> None:
    priv, _pub = _keypair()
    sbom = sign_sbom(_sbom(), priv)
    assert verify_sbom(sbom, "not-base64-at-all!!") is False
    garbage = sbom.model_copy(update={"signed_token": "not-base64-sig!!"})
    assert verify_sbom(garbage, base64.b64encode(b"short").decode()) is False


def test_role_pattern_rejects_a_bad_role() -> None:
    with pytest.raises(ValidationError):
        ModelEntry(
            name="mystery-model",
            role="translate",
            source_origin="India",
            license="MIT",
            weight_sha256="c" * 64,
        )


def test_models_must_be_non_empty() -> None:
    with pytest.raises(ValidationError):
        _sbom(models=[])


def test_entry_for_file_computes_weight_sha256(tmp_path: Path) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"pretend weight bytes")
    entry = entry_for_file(
        name="bge-m3",
        role="embed",
        source_origin="India",
        license="MIT",
        weight_path=weights,
        revision="v2",
    )
    assert entry.weight_sha256 == hashlib.sha256(b"pretend weight bytes").hexdigest()
    assert entry.revision == "v2"


def test_canonical_bytes_are_stable_regardless_of_field_order() -> None:
    priv, pub = _keypair()
    ts = datetime.now(UTC)
    sbom_a = sign_sbom(_sbom(sbom_id="x", deployment_id="y", generated_at=ts), priv)
    sbom_b = sign_sbom(_sbom(deployment_id="y", sbom_id="x", generated_at=ts), priv)
    assert sbom_a.signed_token == sbom_b.signed_token
    assert verify_sbom(sbom_a, pub) is True
    assert verify_sbom(sbom_b, pub) is True
