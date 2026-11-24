"""Provider-side provenance signing — trust receipts, model SBOM, residency attestation.

ONE provider Ed25519 key backs every signed provenance artifact the platform emits: a
per-answer trust receipt (:mod:`rag_common.trust_receipt`), the model SBOM
(:mod:`rag_common.model_sbom`), and the per-node residency attestation
(:mod:`svc_control_plane.compliance`). Centralising the key here keeps the API layer thin and
gives one place to swap in a real HSM-held key in production.

The key comes from ``provenance_signing_key`` (base64 raw 32-byte Ed25519 private). Empty →
a deterministic dev key derived from ``dev_jwt_secret``, so the in-memory/CI profile signs and
verifies with zero setup — NEVER for production (mirrors ``grants._load_signing_key``).
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from svc_guardrails_eval import AuditService

from rag_common import get_logger, get_settings
from rag_common.model_sbom import ModelEntry, ModelSBOM, sign_sbom
from rag_common.trust_receipt import (
    CitedSpan,
    Contributor,
    TrustReceipt,
    build_receipt,
    sign_receipt,
    verify_receipt,
)
from rag_schemas import GroundedAnswer
from rag_schemas.common import AnswerStatus, AuditAction
from svc_control_plane import compliance

_log = get_logger(__name__)


# --- Key material --------------------------------------------------------------------


def signing_key_b64() -> str:
    """The provider signing key (base64 raw Ed25519 private). Dev fallback derives a
    deterministic 32-byte key from ``dev_jwt_secret`` when unset — CI/in-memory only."""
    settings = get_settings()
    if settings.provenance_signing_key:
        return settings.provenance_signing_key
    seed = hashlib.sha256(f"provenance-dev::{settings.dev_jwt_secret}".encode()).digest()
    return base64.b64encode(seed).decode("ascii")


def signer_kid() -> str:
    """Key id (kid) stamped on every artifact so a verifier knows which public key to use."""
    return get_settings().provenance_signing_key_id or "dev"


def public_key_b64() -> str:
    """Base64 raw Ed25519 PUBLIC key matching :func:`signing_key_b64` — handed to verifiers."""
    priv = Ed25519PrivateKey.from_private_bytes(base64.b64decode(signing_key_b64()))
    raw = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return base64.b64encode(raw).decode("ascii")


# --- Model SBOM ----------------------------------------------------------------------


def _configured_models() -> list[ModelEntry]:
    """Parse the operator-declared model list (``model_sbom_models`` JSON). A malformed entry
    is skipped, not fatal — a bad line must not deny the whole SBOM."""
    raw = get_settings().model_sbom_models.strip()
    if not raw:
        return []
    try:
        items = json.loads(raw)
    except (ValueError, TypeError):
        _log.warning("model_sbom_models_parse_failed")
        return []
    entries: list[ModelEntry] = []
    for item in items if isinstance(items, list) else []:
        try:
            entries.append(ModelEntry(**item))
        except (ValueError, TypeError):
            _log.warning("model_sbom_entry_invalid")
    return entries


def signed_sbom(*, now: datetime | None = None) -> ModelSBOM | None:
    """Assemble + sign the deployment's model SBOM from config, or None if none is declared."""
    entries = _configured_models()
    if not entries:
        return None
    at = now or datetime.now(UTC)
    deployment_id = get_settings().residency_node_id or "default"
    sbom = ModelSBOM(
        sbom_id=f"sbom-{signer_kid()}-{at.strftime('%Y%m%dT%H%M%SZ')}",
        deployment_id=deployment_id,
        models=entries,
        generated_at=at,
        signer_kid=signer_kid(),
    )
    return sign_sbom(sbom, signing_key_b64())


# --- Per-node residency attestation --------------------------------------------------


async def signed_residency(
    *, kb_ids: list[str] | None = None, now: datetime | None = None
) -> compliance.NodeResidencyAttestation:
    """A signed per-node data-residency attestation for the regulator (see
    :func:`compliance.node_residency_attestation`)."""
    settings = get_settings()
    return await compliance.node_residency_attestation(
        node_id=settings.residency_node_id or "node-local",
        region=settings.residency_region or "India",
        kb_ids=kb_ids or [],
        private_key_b64=signing_key_b64(),
        signer_kid=signer_kid(),
        now=now,
    )


# --- Trust-receipt verification ------------------------------------------------------


def verify_supplied_receipt(receipt: TrustReceipt) -> bool:
    """Verify a receipt against the provider's public key — fail-closed (never raises)."""
    return verify_receipt(receipt, public_key_b64())


# --- Per-answer trust receipts (hot path) --------------------------------------------


def _contributors_from(answer: GroundedAnswer) -> list[Contributor]:
    """Departments whose evidence fed the answer. A federated answer names every reached peer;
    a single-tenant answer has exactly one contributor — the answering tenant."""
    cov = answer.federation_coverage
    if cov and cov.reached:
        return [Contributor(dept_id=dept) for dept in cov.reached]
    return [Contributor(dept_id=answer.tenant_id)]


def _weight_for(model: str | None) -> str:
    """The declared weight hash for ``model`` from the configured SBOM, or ``"unverified"`` when
    no SBOM entry matches (dev / weights not declared) — never fails receipt issuance."""
    if not model:
        return "unverified"
    for entry in _configured_models():
        if entry.name == model:
            return entry.weight_sha256
    return "unverified"


async def emit_answer_receipt(
    audit: AuditService,
    answer: GroundedAnswer,
    *,
    actor: str,
    contributors: list[Contributor] | None = None,
) -> str | None:
    """Build, sign, and record a trust receipt for a grounded answer — a no-op unless
    ``trust_receipts_enabled`` and the answer is a cited, generated one (GROUNDED /
    FLAGGED_FOR_REVIEW; abstains carry no citations, so no receipt). Recorded as a RECEIPT audit
    event keyed by the answer's ``trace_id`` (so it is chained + SIEM-forwarded like any audit
    event, and pullable by answer id). Returns the receipt's audit event hash, or None when skipped."""
    if not get_settings().trust_receipts_enabled:
        return None
    if answer.status not in (AnswerStatus.GROUNDED, AnswerStatus.FLAGGED_FOR_REVIEW):
        return None
    receipt = build_receipt(
        tenant_id=answer.tenant_id,
        answer_id=answer.trace_id or "",
        query=answer.query,
        answer_text=answer.answer,
        model_id=answer.model or "unknown",
        model_weight_sha256=_weight_for(answer.model),
        contributors=contributors if contributors is not None else _contributors_from(answer),
        cited_spans=[CitedSpan(text=c.quote, chunk_id=c.chunk_id) for c in answer.citations],
        prev_audit_hash=None,
        issued_at=datetime.now(UTC),
        signer_kid=signer_kid(),
        receipt_id=f"receipt-{answer.trace_id or answer.tenant_id}",
    )
    signed = sign_receipt(receipt, signing_key_b64())
    event_hash: str = await audit.record(
        tenant_id=answer.tenant_id,
        trace_id=answer.trace_id or "",
        action=AuditAction.RECEIPT,
        actor=actor,
        payload={"receipt": signed.model_dump(mode="json")},
    )
    return event_hash
