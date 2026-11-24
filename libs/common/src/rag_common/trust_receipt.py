"""Cryptographic trust receipts — signed, portable, offline-verifiable answer provenance (pure).

A grounded answer is only trustworthy if you TRUST the platform that produced it. A trust receipt
removes that assumption: it is a small, signed proof — emitted per answer — that lets ANYONE verify,
OFFLINE and without contacting the platform, that a specific answer was:

* authorised for a tenant (``tenant_id``) and bound to a specific query and answer text
  (``query_sha256`` / ``answer_sha256`` — the receipt commits to the exact bytes, not a paraphrase);
* grounded in specific source spans (``span_digests`` — one SHA-256 per cited span, so a verifier
  holding the sources can confirm the answer cited THOSE spans and not others);
* produced by an approved model on approved weights (``model_id`` / ``model_weight_sha256``);
* contributed to by named departments/grants across a federation (``contributors``);
* chained into the tamper-evident audit log (``prev_audit_hash`` links this receipt to the prior
  audit entry, so a receipt cannot be back-dated or replayed out of order).

The whole thing is signed with the issuer's Ed25519 key (``signer_kid`` names which key). Verification
mirrors :mod:`rag_common.update_manifest` exactly: canonical, sorted-keys JSON (no whitespace) is the
signed input, the signature is base64 detached, and :func:`verify_receipt` is fail-closed — it returns
``bool`` and never raises. Everything here is PURE: keys and data come in as arguments, values go out.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class CitedSpan:
    """One cited source span, as fed to :func:`build_receipt`.

    ``text`` is the exact supporting quote; ``chunk_id`` identifies the source chunk it came from.
    The receipt commits to ``sha256(text|chunk_id)`` so a verifier holding the sources can confirm
    the answer was grounded in THIS span of THIS chunk — without the platform disclosing the text."""

    text: str
    chunk_id: str


class Contributor(BaseModel):
    """A department (and optional grant) whose knowledge bases fed a federated answer."""

    dept_id: str = Field(..., description="Department/peer whose evidence contributed to the answer.")
    grant_id: str | None = Field(default=None, description="Cross-department grant that authorised the access, if any.")
    kb_ids: list[str] = Field(default_factory=list, description="Knowledge bases the contributor supplied.")


class TrustReceipt(BaseModel):
    """A signed, portable provenance proof for one grounded answer (offline-verifiable)."""

    receipt_id: str = Field(..., description="Stable id for this receipt.")
    tenant_id: str = Field(..., description="Tenant the answer was authorised for.")
    answer_id: str = Field(..., description="Id of the grounded answer this receipt attests.")
    query_sha256: str = Field(..., description="SHA-256 hex of the exact query bytes.")
    answer_sha256: str = Field(..., description="SHA-256 hex of the exact answer text bytes.")
    model_id: str = Field(..., description="Approved model/deployment that produced the answer.")
    model_weight_sha256: str = Field(..., description="SHA-256 of the approved model weights.")
    contributors: list[Contributor] = Field(
        default_factory=list, description="Departments/grants whose evidence fed the answer."
    )
    span_digests: list[str] = Field(
        default_factory=list, description="One SHA-256 per cited span (sha256 of text|chunk_id)."
    )
    prev_audit_hash: str | None = Field(
        default=None, description="SHA-256 of the prior audit entry — links this receipt into the audit hash-chain."
    )
    issued_at: datetime = Field(..., description="When the receipt was issued (UTC).")
    signer_kid: str = Field(..., description="Key id of the Ed25519 signer — tells a verifier which public key to use.")
    signed_token: str = Field(default="", description="Base64 detached Ed25519 signature — filled by sign_receipt.")


def sha256_hex(text: str) -> str:
    """SHA-256 hex digest of ``text`` (UTF-8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_bytes(receipt: TrustReceipt) -> bytes:
    """Deterministic bytes the signature covers. Sorted keys, no whitespace, and ``signed_token``
    EXCLUDED — a signature cannot cover itself, so it is stripped before signing and before verifying,
    giving both sides byte-identical input regardless of field order in transit."""
    payload: dict[str, Any] = receipt.model_dump(mode="json")
    payload.pop("signed_token", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_receipt(
    *,
    tenant_id: str,
    answer_id: str,
    query: str,
    answer_text: str,
    model_id: str,
    model_weight_sha256: str,
    contributors: Sequence[Contributor],
    cited_spans: Sequence[CitedSpan],
    prev_audit_hash: str | None,
    issued_at: datetime,
    signer_kid: str,
    receipt_id: str,
) -> TrustReceipt:
    """Assemble an UNSIGNED receipt from raw answer inputs (pure, deterministic).

    Hashes the query and answer bytes and digests each cited span (``sha256(text|chunk_id)``) so the
    receipt commits to the exact query/answer/spans without disclosing them. Same inputs → identical
    receipt. Sign it with :func:`sign_receipt`; ``signed_token`` is empty until then."""
    return TrustReceipt(
        receipt_id=receipt_id,
        tenant_id=tenant_id,
        answer_id=answer_id,
        query_sha256=sha256_hex(query),
        answer_sha256=sha256_hex(answer_text),
        model_id=model_id,
        model_weight_sha256=model_weight_sha256,
        contributors=list(contributors),
        span_digests=[sha256_hex(f"{s.text}|{s.chunk_id}") for s in cited_spans],
        prev_audit_hash=prev_audit_hash,
        issued_at=issued_at,
        signer_kid=signer_kid,
    )


def sign_receipt(receipt: TrustReceipt, private_key_b64: str) -> TrustReceipt:
    """Return a COPY of ``receipt`` with ``signed_token`` set to a base64 detached Ed25519 signature
    over its canonical bytes (issuer side). The input receipt is left unchanged."""
    key = Ed25519PrivateKey.from_private_bytes(base64.b64decode(private_key_b64))
    signature = base64.b64encode(key.sign(canonical_bytes(receipt))).decode("ascii")
    return receipt.model_copy(update={"signed_token": signature})


def verify_receipt(receipt: TrustReceipt, public_key_b64: str) -> bool:
    """True iff ``receipt.signed_token`` is a valid signature over ``receipt`` under ``public_key_b64``.
    Recomputes the canonical bytes (excluding ``signed_token``) and checks the signature. False on ANY
    problem — bad key, bad/empty signature, or a tampered field — fail closed, never raise."""
    try:
        pub = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key_b64))
        pub.verify(base64.b64decode(receipt.signed_token), canonical_bytes(receipt))
    except (InvalidSignature, ValueError, TypeError):
        return False
    return True
