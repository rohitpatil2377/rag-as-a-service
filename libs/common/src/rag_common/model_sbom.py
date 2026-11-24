"""Model SBOM (Software Bill of Materials for AI weights) — signed, verifiable (pure, testable).

The platform's "India-origin, self-hosted" sovereignty claim is only as good as it is checkable.
This module lets the provider assert, in one signed document, EXACTLY which model weights a
deployment runs: name, role, origin, license, and a SHA-256 of the weight file — then lets a
customer or auditor verify that assertion offline, without trusting a dashboard or a vendor's
say-so. Same shape as :mod:`rag_common.update_manifest`: canonical sorted-keys JSON, detached
Ed25519 signature, fail-closed verification that never raises. Pure + I/O-free except
:func:`entry_for_file` (which only reads + hashes a weight file).
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import BaseModel, Field

try:
    from rag_common.update_manifest import sha256_file
except ImportError:  # pragma: no cover - fallback if update_manifest is ever unavailable
    import hashlib

    def sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()


ModelRole = Literal["embed", "rerank", "generate", "ocr"]


class ModelEntry(BaseModel):
    """One model weight backing the deployment."""

    name: str
    role: ModelRole
    source_origin: str = Field(..., description="Where the weights come from, e.g. 'India' or a vendor name")
    license: str
    weight_sha256: str = Field(..., description="sha256 of the weight file/dir manifest, hex-encoded")
    revision: str = ""


class ModelSBOM(BaseModel):
    """The signed bill of materials for every model a deployment runs."""

    sbom_id: str
    deployment_id: str
    models: list[ModelEntry] = Field(..., min_length=1)
    provenance: str = "India-origin, self-hosted"
    generated_at: datetime
    signer_kid: str
    signed_token: str = ""


def canonical_bytes(sbom: ModelSBOM) -> bytes:
    """Deterministic bytes signed by the provider. Sorted keys, no whitespace, ``signed_token``
    excluded — a verifier that re-serialises the SBOM the same way gets byte-identical input."""
    data = sbom.model_dump(mode="json")
    data.pop("signed_token", None)
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_sbom(sbom: ModelSBOM, private_key_b64: str) -> ModelSBOM:
    """Return a copy of ``sbom`` with ``signed_token`` set to a base64 Ed25519 signature over its
    canonical bytes (provider side)."""
    key = Ed25519PrivateKey.from_private_bytes(base64.b64decode(private_key_b64))
    token = base64.b64encode(key.sign(canonical_bytes(sbom))).decode("ascii")
    return sbom.model_copy(update={"signed_token": token})


def verify_sbom(sbom: ModelSBOM, public_key_b64: str) -> bool:
    """True iff ``sbom.signed_token`` is a valid signature over ``sbom``'s canonical bytes
    (customer/auditor side). False on ANY problem (bad key, bad/missing signature, tamper) —
    fail closed, never raise."""
    try:
        if not sbom.signed_token:
            return False
        pub = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key_b64))
        pub.verify(base64.b64decode(sbom.signed_token), canonical_bytes(sbom))
    except (InvalidSignature, ValueError, TypeError):
        return False
    return True


def entry_for_file(
    *,
    name: str,
    role: ModelRole,
    source_origin: str,
    license: str,
    weight_path: Path,
    revision: str = "",
) -> ModelEntry:
    """Build a :class:`ModelEntry`, computing ``weight_sha256`` from the file at ``weight_path``."""
    return ModelEntry(
        name=name,
        role=role,
        source_origin=source_origin,
        license=license,
        weight_sha256=sha256_file(weight_path),
        revision=revision,
    )
