"""Offline-update manifest — signature, checksum, and monotonicity guards."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

from rag_common.update_manifest import (
    Artifact,
    ImageRef,
    IncompatibleUpdateError,
    UpdateManifest,
    check_upgrade,
    sha256_file,
    sign_manifest,
    verify_artifacts,
    verify_manifest,
)


def _keypair() -> tuple[str, str]:
    k = Ed25519PrivateKey.generate()
    priv = base64.b64encode(k.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())).decode()
    pub = base64.b64encode(k.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()
    return priv, pub


def _manifest(**kw: object) -> UpdateManifest:
    base: dict[str, object] = {
        "bundle_id": "b1",
        "product_version": "0.2.0",
        "build_number": 2,
        "min_from_build": 1,
        "created_at": "2026-08-05T00:00:00+00:00",
        "migration_head": "0007_tenants",
        "images": [ImageRef(ref="reg/api:0.2.0", digest="sha256:abc")],
        "artifacts": [],
    }
    base.update(kw)
    return UpdateManifest(**base)


def test_sign_then_verify_roundtrips() -> None:
    priv, pub = _keypair()
    m = _manifest()
    sig = sign_manifest(m, priv)
    assert verify_manifest(m, sig, pub) is True


def test_a_tampered_manifest_fails_verification() -> None:
    priv, pub = _keypair()
    m = _manifest()
    sig = sign_manifest(m, priv)
    # An attacker swaps the image digest but keeps the old signature.
    tampered = m.model_copy(update={"images": [ImageRef(ref="reg/api:0.2.0", digest="sha256:EVIL")]})
    assert verify_manifest(tampered, sig, pub) is False


def test_a_signature_from_the_wrong_key_fails() -> None:
    priv, _pub = _keypair()
    _other_priv, other_pub = _keypair()
    m = _manifest()
    sig = sign_manifest(m, priv)
    assert verify_manifest(m, sig, other_pub) is False  # not the Acme release key
    assert verify_manifest(m, "not-base64!!", other_pub) is False  # garbage sig -> fail closed


def test_artifact_checksums_catch_a_swapped_file(tmp_path: Path) -> None:
    good = tmp_path / "images.tar"
    good.write_bytes(b"the real image tarball")
    m = _manifest(artifacts=[Artifact(path="images.tar", sha256=sha256_file(good), bytes=good.stat().st_size)])
    assert verify_artifacts(m, tmp_path) == []  # intact

    good.write_bytes(b"a swapped, malicious tarball")  # same name, different bytes
    assert verify_artifacts(m, tmp_path) == ["images.tar"]

    missing = _manifest(artifacts=[Artifact(path="gone.tar", sha256="0" * 64, bytes=1)])
    assert verify_artifacts(missing, tmp_path) == ["gone.tar"]


def test_upgrade_must_be_a_step_forward() -> None:
    # newer build onto a compatible base -> ok
    check_upgrade(current_build=1, manifest=_manifest(build_number=2, min_from_build=1))
    # downgrade / replay -> refused
    with pytest.raises(IncompatibleUpdateError):
        check_upgrade(current_build=2, manifest=_manifest(build_number=2, min_from_build=1))
    with pytest.raises(IncompatibleUpdateError):
        check_upgrade(current_build=5, manifest=_manifest(build_number=3, min_from_build=1))
    # too-old base (must apply intermediate first) -> refused
    with pytest.raises(IncompatibleUpdateError):
        check_upgrade(current_build=1, manifest=_manifest(build_number=5, min_from_build=3))
