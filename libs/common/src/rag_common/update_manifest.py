"""Offline-update bundle manifest — signed, verifiable, monotonic (pure, testable).

An air-gapped site cannot ``docker pull`` or ``git pull``. Updates arrive as a BUNDLE on vetted
physical media. Integrity checksums (SHA256SUMS) prove the bytes did not rot; they do NOT prove
the bundle came from Acme and was not swapped — that is what a SIGNATURE adds. This is the
security spine of the offline-update flow:

* the provider builds a bundle, writes a **manifest** (versions, the exact image digests, the DB
  migration head, and a SHA-256 for every artifact), and **signs** the manifest with the Acme
  release private key (Ed25519, detached signature);
* the air-gapped site **verifies** the signature with the pre-installed release PUBLIC key, then
  verifies every artifact's checksum against the manifest, then checks the update is a real
  step FORWARD (monotonic ``build_number``, and its ``min_from_build`` floor is met) before
  loading images / running migrations.

A tampered bundle (any byte of an image or the manifest changed, or a downgrade/replay) is
refused BEFORE anything is loaded. Pure + I/O-free except :func:`verify_artifacts` (which only
reads + hashes files); everything is deterministic and unit-tested.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import BaseModel, Field


class ImageRef(BaseModel):
    ref: str = Field(..., description="Image reference, e.g. registry.internal.acme/ragaas/api:0.2.0")
    digest: str = Field(..., description="sha256:... content digest pinned at build time")


class Artifact(BaseModel):
    path: str = Field(..., description="Path INSIDE the bundle, e.g. images.tar")
    sha256: str
    bytes: int


class UpdateManifest(BaseModel):
    """The signed description of one offline-update bundle."""

    bundle_id: str
    product_version: str
    # Monotonic release counter: an update must be STRICTLY newer than the running build, and the
    # site must be at least `min_from_build` (older sites apply the intermediate updates first).
    build_number: int = Field(..., ge=1)
    min_from_build: int = Field(..., ge=0)
    created_at: str = Field(..., description="ISO-8601 UTC, stamped by the builder")
    migration_head: str = Field(..., description="Alembic revision the DB reaches after this update")
    images: list[ImageRef] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)


class IncompatibleUpdateError(ValueError):
    """The bundle is not a valid step forward for this site (downgrade/replay or too-old base)."""


def canonical_bytes(manifest: UpdateManifest) -> bytes:
    """Deterministic bytes signed by the provider. Sorted keys, no whitespace — a verifier that
    re-serialises the manifest the same way gets byte-identical input, so the signature is stable
    regardless of field order in transit."""
    return json.dumps(manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_manifest(manifest: UpdateManifest, private_key_b64: str) -> str:
    """Detached base64 Ed25519 signature over the manifest's canonical bytes (provider side)."""
    key = Ed25519PrivateKey.from_private_bytes(base64.b64decode(private_key_b64))
    return base64.b64encode(key.sign(canonical_bytes(manifest))).decode("ascii")


def verify_manifest(manifest: UpdateManifest, signature_b64: str, public_key_b64: str) -> bool:
    """True iff ``signature`` is a valid Acme-release signature over ``manifest`` (customer side).
    False on ANY problem (bad key, bad signature, tamper) — fail closed, never raise."""
    try:
        pub = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key_b64))
        pub.verify(base64.b64decode(signature_b64), canonical_bytes(manifest))
    except (InvalidSignature, ValueError, TypeError):
        return False
    return True


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_artifacts(manifest: UpdateManifest, bundle_dir: Path) -> list[str]:
    """Recompute each artifact's SHA-256 under ``bundle_dir`` and compare to the (signed) manifest.
    Returns the list of artifact paths that are MISSING or MISMATCHED — empty means all intact.
    Because the manifest is signed, a matching checksum here means the byte came from Acme."""
    failures: list[str] = []
    for art in manifest.artifacts:
        f = bundle_dir / art.path
        if not f.is_file() or sha256_file(f) != art.sha256:
            failures.append(art.path)
    return failures


def check_upgrade(current_build: int, manifest: UpdateManifest) -> None:
    """Refuse a bundle that is not a genuine step forward for a site at ``current_build``:
    a downgrade/replay (``build_number <= current``), or a jump that skips a required base
    (``current < min_from_build``). Raises :class:`IncompatibleUpdateError`."""
    if manifest.build_number <= current_build:
        raise IncompatibleUpdateError(
            f"bundle build {manifest.build_number} is not newer than the running build {current_build} "
            "(downgrade or replay refused)"
        )
    if current_build < manifest.min_from_build:
        raise IncompatibleUpdateError(
            f"this bundle requires build >= {manifest.min_from_build}; the site is at {current_build} "
            "— apply the intermediate update(s) first"
        )
