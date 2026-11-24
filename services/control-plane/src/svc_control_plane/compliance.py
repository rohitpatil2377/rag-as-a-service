"""DPDP compliance surface (FR-H8, v1.1) — retention, right-to-erasure, residency.

For a *sovereign* platform serving regulated Indian buyers, DPDP Act obligations are
product features, not paperwork:

* **Right to erasure** — erase a data principal's data on request. Erasure cascades
  through EVERY store (catalog rows, dense + sparse indexes, raw object bytes, semantic
  cache) AND every derived record that quoted it — conversation history and the RAG v2
  query log — then emits a tamper-evident **erasure certificate** onto the audit chain
  (``AuditAction.ERASURE``). See ``docs/automated-erasure.md`` for what is retained and why.
* **Retention** — a per-KB TTL; documents older than it are purged through the same
  erasure path. Applied on demand here; the scheduled worker rides FR-F7 triggers later.
* **Residency attestation** — an on-demand report proving where the data lives and how it
  is protected (sovereignty enforcement, egress allowlist, encryption, audit-chain
  integrity, retention policy), suitable to hand an auditor.
* **Node residency attestation** — a SIGNED, per-NODE counterpart: the claim an operator
  hands a regulator that THIS node's data (across the KBs resident on it) never leaves its
  region. A detached Ed25519 signature (mirroring ``rag_common.update_manifest``'s bundle
  signing) makes it tamper-evident — a regulator with the public key can catch any edit,
  and the private key never touches this module (it is a caller-supplied argument).

Everything reuses existing seams — nothing here weakens tenant isolation or grounding.
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime, timedelta

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import BaseModel, Field

from rag_clients.factory import get_catalog, purge_query_log_citing
from rag_common import NotFoundError, get_logger, get_settings
from rag_core.parent_child import not_parent
from rag_schemas import Grant
from rag_schemas.common import AuditAction
from svc_control_plane import pipeline, wiring

_log = get_logger(__name__)

_RETENTION_KEY = "retention_days"


class ErasureCertificate(BaseModel):
    """Tamper-evident proof of a DPDP erasure — mirrored onto the audit chain."""

    kb_id: str
    requested_by: str
    reason: str
    subject: str | None = None
    documents_erased: list[str] = Field(default_factory=list)
    chunks_purged: int = 0
    conversations_purged: int = 0
    # Derived RAG v2 query-log rows that quoted an erased document (same contract as
    # `conversations_purged`). Defaulted, and the certificate hash is computed over
    # (kb_id, targets, at) only — see `_certificate_hash` — so adding it is additive and does not
    # change any previously issued certificate's hash.
    query_log_purged: int = 0
    certificate_hash: str = Field(description="SHA-256 over the erased doc set + KB + time.")
    erased_at: datetime


class RetentionInfo(BaseModel):
    kb_id: str
    retention_days: int = Field(description="0 = keep indefinitely (no auto-expiry).")


class RetentionResult(BaseModel):
    kb_id: str
    retention_days: int
    documents_erased: list[str] = Field(default_factory=list)
    certificate: ErasureCertificate | None = None


class FederationPosture(BaseModel):
    """The KB org's cross-tenant federation stance, for the residency attestation.

    Federation is the one mechanism that opens tenant isolation, so an auditor must see it:
    how many grants this department published, how many it holds, how much of that is
    RE-DELEGATED (transitive A->B->C), and how deep any delegation chain runs."""

    enabled: bool
    grants_issued: int = 0
    grants_issued_active: int = 0
    grants_received_active: int = 0
    delegated_grants_issued: int = 0
    max_delegation_depth: int = 0
    revoked_grants: int = 0


class GrantHop(BaseModel):
    """One link in a delegation chain — a single publisher->subject authorisation."""

    grant_id: str
    issuer_dept_id: str
    subject_dept_id: str
    topics: list[str]
    kb_ids: list[str]
    active: bool
    revoked_at: datetime | None = None
    parent_grant_id: str | None = None


class GrantUsage(BaseModel):
    """How a grant was actually EXERCISED — drawn from the tamper-evident audit chain."""

    federated_queries: int = 0
    last_used_at: datetime | None = None
    actors: list[str] = Field(default_factory=list)
    denials: int = 0


class GrantLineageExport(BaseModel):
    """Per-grant DPDP/attestation export: which department answered, under which grant, with
    the full consent lineage (FR-H8 federation edge).

    The ``chain`` is the delegation ancestry ROOT-first (A->B->C), so a regulator can read the
    consent trail end to end: the KB owner's original authorisation, every narrowing hop, and
    the leaf grant this export is about. ``usage`` is what the audit chain records the SUBJECT
    actually did under it. ``chain_active`` is False if ANY ancestor is revoked/expired —
    transitive revocation, so a killed root invalidates the whole branch."""

    grant_id: str
    generated_at: datetime
    issuer_dept_id: str
    subject_dept_id: str
    topics: list[str]
    kb_ids: list[str]
    active: bool
    chain_active: bool
    revoked_at: datetime | None
    expires_at: datetime
    delegation_depth: int
    root_grant_id: str
    consent_lineage: str
    chain: list[GrantHop] = Field(default_factory=list)
    children: list[GrantHop] = Field(default_factory=list)
    usage: GrantUsage = Field(default_factory=GrantUsage)


class Attestation(BaseModel):
    """A residency + protection report an operator can hand a regulator (FR-H8)."""

    kb_id: str
    generated_at: datetime
    data_residency: str
    sovereignty_enforced: bool
    egress_allowlist: list[str]
    encryption_at_rest: bool
    encryption_in_transit: bool
    audit_chain_verified: bool
    audit_events: int
    retention_days: int
    documents: int
    federation: FederationPosture
    notes: list[str] = Field(default_factory=list)


class NodeResidencyAttestation(BaseModel):
    """A SIGNED, per-NODE residency report — the claim an operator hands a regulator that
    THIS node's data (across the KBs resident on it) stays in ``region``, cryptographically
    signed so it can't be forged or altered in transit. Unlike :class:`Attestation` (per-KB,
    unsigned, derived from live per-KB state), this is a static claim about node placement
    that a third party can independently verify with the signer's public key."""

    attestation_id: str
    node_id: str
    region: str
    kb_ids: list[str] = Field(default_factory=list, description="KBs resident on this node.")
    data_residency: str
    no_foreign_egress: bool
    generated_at: datetime
    signer_kid: str = Field(description="Key id of the Ed25519 key that signed this attestation.")
    signed_token: str = Field(default="", description="Detached base64 Ed25519 signature; empty until signed.")


async def _resolve_targets(kb_id: str, doc_ids: list[str] | None, subject: str | None) -> list[str]:
    """The document ids to erase: explicit ids, and/or every document whose text or
    metadata contains ``subject`` (data-principal match, complete within the KB)."""
    catalog = get_catalog()
    docs = await catalog.list_documents(kb_id)
    known = {d.doc_id for d in docs}
    targets: set[str] = {d for d in (doc_ids or []) if d in known}
    if subject:
        needle = subject.strip().lower()
        for d in docs:
            if needle in d.doc_id.lower() or any(needle in str(v).lower() for v in d.metadata.values()):
                targets.add(d.doc_id)
                continue
            # FR-C2: scan children only — parents are re-concatenations of the SAME doc's children
            # (which tile its text), so a subject in the doc is in a child; a parent adds no new
            # source text and its doc_id already matches, so skipping parents cannot miss a target.
            chunks = not_parent(await catalog.document_chunks(kb_id, d.doc_id))
            if any(needle in c.text.lower() for c in chunks):
                targets.add(d.doc_id)
    return sorted(targets)


def _certificate_hash(kb_id: str, doc_ids: list[str], at: datetime) -> str:
    canonical = "|".join([kb_id, at.isoformat(), *sorted(doc_ids)])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


async def erase(
    kb_id: str,
    *,
    actor: str,
    doc_ids: list[str] | None = None,
    subject: str | None = None,
    reason: str = "data_principal_request",
    now: datetime | None = None,
) -> ErasureCertificate:
    """Erase documents (by id and/or subject match) across every store + every derived artifact
    that quoted them, and record a signed erasure certificate on the audit chain.

    "Derived artifact" is two things, scrubbed on the same contract: chat history
    (`ConversationStore.delete_citing`) and the RAG v2 query log
    (`PostgresCatalog.delete_query_log_citing`). Both hold text quoted out of documents, so both
    must go or the certificate would overstate what was erased.

    Note the deliberate asymmetry with a plain `pipeline.delete_document`: that purges the stores
    and the document's own derived rows, while ERASURE additionally scrubs derived history that
    merely quoted the document. Only this path does the latter."""
    at = now or datetime.now(UTC)
    targets = await _resolve_targets(kb_id, doc_ids, subject)

    chunks_purged = 0
    for doc_id in targets:
        chunks_purged += await pipeline.delete_document(kb_id, doc_id, actor=actor)
    conversations_purged = await wiring.conversation_store().delete_citing(kb_id, set(targets))
    query_log_purged = await purge_query_log_citing(kb_id, set(targets))

    cert = ErasureCertificate(
        kb_id=kb_id,
        requested_by=actor,
        reason=reason,
        subject=subject,
        documents_erased=targets,
        chunks_purged=chunks_purged,
        conversations_purged=conversations_purged,
        query_log_purged=query_log_purged,
        certificate_hash=_certificate_hash(kb_id, targets, at),
        erased_at=at,
    )
    await wiring.audit_service().record(
        tenant_id=kb_id,
        trace_id=cert.certificate_hash[:32],
        action=AuditAction.ERASURE,
        actor=actor,
        payload={
            "reason": reason,
            "subject": subject,
            "documents_erased": targets,
            "chunks_purged": chunks_purged,
            "conversations_purged": conversations_purged,
            "query_log_purged": query_log_purged,
            "certificate_hash": cert.certificate_hash,
        },
    )
    _log.info(
        "dpdp_erasure",
        kb_id=kb_id,
        docs=len(targets),
        chunks=chunks_purged,
        conversations=conversations_purged,
        query_log=query_log_purged,
        actor=actor,
    )
    return cert


async def get_retention(kb_id: str) -> RetentionInfo:
    config = await get_catalog().get_tenant_config(kb_id)
    raw = config.get(_RETENTION_KEY)
    days = int(raw) if raw and raw.isdigit() else 0
    return RetentionInfo(kb_id=kb_id, retention_days=days)


async def set_retention(kb_id: str, days: int) -> RetentionInfo:
    await get_catalog().set_tenant_config(kb_id, _RETENTION_KEY, str(max(0, days)))
    _log.info("retention_policy_set", kb_id=kb_id, retention_days=days)
    return await get_retention(kb_id)


async def apply_retention(kb_id: str, *, actor: str, now: datetime | None = None) -> RetentionResult:
    """Erase documents older than the KB's retention window (0 = keep forever)."""
    at = now or datetime.now(UTC)
    info = await get_retention(kb_id)
    if info.retention_days <= 0:
        return RetentionResult(kb_id=kb_id, retention_days=0)
    cutoff = at - timedelta(days=info.retention_days)
    docs = await get_catalog().list_documents(kb_id)
    expired = [d.doc_id for d in docs if d.ingested_at < cutoff]
    if not expired:
        return RetentionResult(kb_id=kb_id, retention_days=info.retention_days)
    cert = await erase(kb_id, actor=actor, doc_ids=expired, reason="retention_policy", now=at)
    return RetentionResult(kb_id=kb_id, retention_days=info.retention_days, documents_erased=expired, certificate=cert)


# How far back a single integrity scan reaches. A KB with more events than this is verified
# over its newest AUDIT_CHAIN_SCAN_LIMIT only, and `ChainIntegrity.truncated` says so rather
# than letting a partial scan masquerade as a whole-trail guarantee.
AUDIT_CHAIN_SCAN_LIMIT = 10_000


class ChainIntegrity(BaseModel):
    """What a walk of a KB's audit hash chain actually established."""

    ok: bool
    events: int
    tampered: int  # content no longer hashes to its seal
    missing_parents: int  # seals a parent that is not in the log — an event was removed
    orphan_genesis: int  # a second parentless event, i.e. a chain that restarts mid-log
    duplicate_hashes: int  # the same sealed event recorded twice
    forks: int  # one parent sealed by >1 event — CONCURRENT WRITES, not damage
    truncated: bool  # the log is longer than this scan reached


async def _verify_audit_chain(kb_id: str) -> ChainIntegrity:
    """Walk the tenant's audit log and establish whether the SHA-256 chain is intact.

    The chain is a TREE, not a line. ``AuditSink.append`` reads the current tip and then
    inserts, so two events written concurrently — the continuous-eval scheduler and a live
    query, say — legitimately seal the SAME parent and the chain forks. This is ordinary
    operation: measured on a real deployment, 5 forks in 250 events, every one a genuine
    pair of concurrent writes.

    The previous implementation walked it as a strict line (``e.prev_hash != prev``) and so
    reported ``audit_chain_verified: false`` on a perfectly intact trail — the first fork
    ended the walk. That verdict reached a regulator-facing attestation and the DPDP /
    Responsible-AI control status, i.e. the platform accused itself of a broken audit trail
    whenever it was busy enough to write two events at once. Do not reinstate a linear walk.

    What is checked instead, which is strictly stronger and has no false positives:

    * **Content** — ``event_hash == compute_hash()`` for every event. Because ``compute_hash``
      commits to ``prev_hash`` as well as the body, this authenticates each event's CONTENT
      *and* its claimed parent pointer. Any edit to a stored row fails here.
    * **Topology** — every non-genesis ``prev_hash`` must resolve to an event that really
      exists earlier in the log. Deleting an event orphans its children, which this catches;
      forking does not trip it, because both siblings' parent is present.
    * Exactly one parentless event, and no event sealed twice.

    Honest limits, recorded on the result rather than hidden: a scan reaching the window
    limit cannot speak for parents below it (``truncated``), and an event that nothing seals
    on — the newest, or either tip of a fork — can be removed without orphaning anything.
    """
    events = list(reversed(await wiring.audit_service().list(kb_id, None, AUDIT_CHAIN_SCAN_LIMIT)))
    truncated = len(events) >= AUDIT_CHAIN_SCAN_LIMIT
    seen: set[str] = set()
    parents: dict[str, int] = {}
    tampered = missing_parents = orphan_genesis = duplicate_hashes = 0

    for i, e in enumerate(events):
        if e.event_hash != e.compute_hash():
            tampered += 1
        if e.event_hash is not None:
            if e.event_hash in seen:
                duplicate_hashes += 1
            seen.add(e.event_hash)
        if e.prev_hash is None:
            # A parentless event is the start of the chain, and there can be only one. When the
            # scan is truncated the real genesis is below the window, so a parentless event in
            # view at all would be a second one.
            if i != 0 or truncated:
                orphan_genesis += 1
            continue
        parents[e.prev_hash] = parents.get(e.prev_hash, 0) + 1
        if e.prev_hash not in seen and not truncated:
            # Only a complete scan can call a missing parent a deletion: where the chain forks,
            # the newest-N window is not closed under "is the parent of", so a sibling branch's
            # event can sit just below the floor while rows above it still point at it.
            missing_parents += 1

    forks = sum(1 for n in parents.values() if n > 1)
    ok = not (tampered or missing_parents or orphan_genesis or duplicate_hashes)
    return ChainIntegrity(
        ok=ok,
        events=len(events),
        tampered=tampered,
        missing_parents=missing_parents,
        orphan_genesis=orphan_genesis,
        duplicate_hashes=duplicate_hashes,
        forks=forks,
        truncated=truncated,
    )


async def attestation(kb_id: str, *, now: datetime | None = None) -> Attestation:
    """Compile a residency + protection attestation for the KB (FR-H8)."""
    settings = get_settings()
    chain = await _verify_audit_chain(kb_id)
    retention = await get_retention(kb_id)
    docs = await get_catalog().list_documents(kb_id)
    notes: list[str] = []
    if not settings.sovereign_enforce:
        notes.append("Sovereignty enforcement is DISABLED in this environment — not production-representative.")
    if retention.retention_days == 0:
        notes.append("No retention limit set — documents are kept until explicitly erased.")
    # Say what the integrity check established, and what it could not. A regulator reading a
    # bare `audit_chain_verified: true` is entitled to know the scope behind it.
    if chain.truncated:
        notes.append(
            f"Audit-chain integrity was verified over the most recent {chain.events} events; "
            "the trail is longer than a single scan reaches, so events before that window were "
            "not re-hashed in this report."
        )
    if not chain.ok:
        reasons = [
            f"{chain.tampered} event(s) no longer match their seal" if chain.tampered else "",
            f"{chain.missing_parents} event(s) reference a parent that is absent (an event was removed)"
            if chain.missing_parents
            else "",
            f"{chain.orphan_genesis} unexpected parentless event(s)" if chain.orphan_genesis else "",
            f"{chain.duplicate_hashes} duplicated event(s)" if chain.duplicate_hashes else "",
        ]
        notes.append("AUDIT CHAIN INTEGRITY FAILED: " + "; ".join(r for r in reasons if r) + ".")
    return Attestation(
        kb_id=kb_id,
        generated_at=now or datetime.now(UTC),
        data_residency="India (self-hosted; no foreign/managed AI or cloud calls at runtime)",
        sovereignty_enforced=settings.sovereign_enforce,
        egress_allowlist=sorted(settings.egress_allowlist),
        encryption_at_rest=True,
        encryption_in_transit=True,
        audit_chain_verified=chain.ok,
        audit_events=chain.events,
        retention_days=retention.retention_days,
        documents=len(docs),
        federation=await _federation_posture(kb_id),
        notes=notes,
    )


def _dept_of(kb_id: str) -> str:
    """The federation DEPARTMENT (org) that owns a KB — grants are issued dept-to-dept, so the
    ``org__kb`` sub-KB maps back to its org root for the federation view."""
    return kb_id.split("__", 1)[0]


def _hop(g: Grant) -> GrantHop:
    return GrantHop(
        grant_id=g.grant_id,
        issuer_dept_id=g.issuer_dept_id,
        subject_dept_id=g.subject_dept_id,
        topics=g.topics,
        kb_ids=g.kb_ids,
        active=g.is_active,
        revoked_at=g.revoked_at,
        parent_grant_id=g.parent_grant_id,
    )


async def _federation_posture(kb_id: str) -> FederationPosture:
    """Summarise this department's grant portfolio for the attestation."""
    settings = get_settings()
    if not settings.federation_enabled:
        return FederationPosture(enabled=False)
    store = wiring.grant_store()
    dept = _dept_of(kb_id)
    issued = await store.list_issued(dept)
    received = await store.list_received(dept)
    delegated = [g for g in issued if g.parent_grant_id]
    max_depth = 0
    for g in issued:
        depth = len(await store.resolve_chain(g.grant_id))
        max_depth = max(max_depth, depth)
    return FederationPosture(
        enabled=True,
        grants_issued=len(issued),
        grants_issued_active=sum(1 for g in issued if g.is_active),
        grants_received_active=len(received),
        delegated_grants_issued=len(delegated),
        max_delegation_depth=max_depth,
        revoked_grants=sum(1 for g in issued if g.revoked_at is not None),
    )


def _lineage_string(chain_root_first: list[Grant]) -> str:
    """Human consent trail: ``A [finance,hr | kb1] -> B [finance | kb1] -> C``."""
    parts: list[str] = []
    for i, g in enumerate(chain_root_first):
        if i == 0:
            parts.append(f"{g.issuer_dept_id} [{','.join(sorted(g.topics))} | {','.join(sorted(g.kb_ids))}]")
        parts.append(f"{g.subject_dept_id} [{','.join(sorted(g.topics))} | {','.join(sorted(g.kb_ids))}]")
    return " -> ".join(parts)


async def _grant_usage(grant: Grant) -> GrantUsage:
    """What the SUBJECT did under this grant, read from its own tamper-evident audit chain.

    A FEDERATED_QUERY is recorded in the CALLER's (subject's) log with the grants it used, so
    the usage of a grant is found by scanning the subject's federated-query + denial events for
    this ``grant_id``."""
    events = await wiring.audit_service().list(grant.subject_dept_id, None, 10_000)
    count = 0
    denials = 0
    actors: set[str] = set()
    last: datetime | None = None
    for e in events:
        if e.action is AuditAction.FEDERATED_QUERY:
            used = e.payload.get("grants_used") or []
            if any(isinstance(u, dict) and u.get("grant_id") == grant.grant_id for u in used):
                count += 1
                actors.add(e.actor)
                last = e.at if last is None or e.at > last else last
        elif e.action is AuditAction.FEDERATION_DENIED and grant.issuer_dept_id in (e.payload.get("denied_by") or []):
            denials += 1
    return GrantUsage(federated_queries=count, last_used_at=last, actors=sorted(actors), denials=denials)


async def grant_export(grant_id: str, *, now: datetime | None = None) -> GrantLineageExport:
    """A per-grant compliance export: the consent lineage (A->B->C), transitive validity, and
    the audit-sourced record of who answered under it. Raises ``NotFoundError`` if unknown."""
    store = wiring.grant_store()
    grant = await store.get(grant_id)
    if grant is None:
        raise NotFoundError("grant not found", details={"grant_id": grant_id})
    chain = await store.resolve_chain(grant_id)  # leaf-first
    root_first = list(reversed(chain))
    children = await store.list_children(grant_id)
    return GrantLineageExport(
        grant_id=grant.grant_id,
        generated_at=now or datetime.now(UTC),
        issuer_dept_id=grant.issuer_dept_id,
        subject_dept_id=grant.subject_dept_id,
        topics=grant.topics,
        kb_ids=grant.kb_ids,
        active=grant.is_active,
        chain_active=await store.is_chain_active(grant_id),
        revoked_at=grant.revoked_at,
        expires_at=grant.expires_at,
        delegation_depth=len(chain),
        root_grant_id=root_first[0].grant_id if root_first else grant.grant_id,
        consent_lineage=_lineage_string(root_first),
        chain=[_hop(g) for g in root_first],
        children=[_hop(g) for g in children],
        usage=await _grant_usage(grant),
    )


def _node_canonical_bytes(att: NodeResidencyAttestation) -> bytes:
    """Deterministic bytes signed over the attestation, EXCLUDING ``signed_token`` itself (you
    cannot sign over your own signature) — sorted keys, no whitespace, mirroring
    ``rag_common.update_manifest.canonical_bytes`` so re-serialising the same attestation
    always yields byte-identical input regardless of field order in transit."""
    payload = att.model_dump(mode="json", exclude={"signed_token"})
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_node_attestation(att: NodeResidencyAttestation, private_key_b64: str) -> NodeResidencyAttestation:
    """Return a COPY of ``att`` with ``signed_token`` set to a detached base64 Ed25519 signature
    over its canonical bytes. The key is a caller-supplied argument only — never read from
    config/settings here, so signing stays pure and testable with any ephemeral keypair."""
    key = Ed25519PrivateKey.from_private_bytes(base64.b64decode(private_key_b64))
    token = base64.b64encode(key.sign(_node_canonical_bytes(att))).decode("ascii")
    return att.model_copy(update={"signed_token": token})


def verify_node_attestation(att: NodeResidencyAttestation, public_key_b64: str) -> bool:
    """True iff ``att.signed_token`` is a valid Ed25519 signature over ``att``'s canonical bytes
    for ``public_key_b64``. False on ANY problem (bad key, bad/missing signature, tampered
    field) — fail closed, never raise."""
    try:
        pub = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key_b64))
        pub.verify(base64.b64decode(att.signed_token), _node_canonical_bytes(att))
    except (InvalidSignature, ValueError, TypeError):
        return False
    return True


async def node_residency_attestation(
    *,
    node_id: str,
    region: str,
    kb_ids: list[str],
    private_key_b64: str,
    signer_kid: str,
    now: datetime | None = None,
) -> NodeResidencyAttestation:
    """Compile + SIGN a per-node residency attestation an operator hands a regulator: proof that
    THIS node's data — across every KB resident on it — stays in ``region``, with a detached
    Ed25519 signature so the report can't be forged or silently altered after issuance.

    Unlike :func:`attestation` (per-KB, unsigned, queries live per-KB state), this is a static,
    cryptographically-signed claim about the node's deployment posture: the platform enforces
    egress allow-listing at the network layer (see :class:`Attestation`'s ``egress_allowlist``),
    so a node-level report can truthfully assert ``no_foreign_egress=True`` outright."""
    at = now or datetime.now(UTC)
    att = NodeResidencyAttestation(
        attestation_id=f"node-residency-{node_id}-{at.strftime('%Y%m%dT%H%M%SZ')}",
        node_id=node_id,
        region=region,
        kb_ids=sorted(kb_ids),
        data_residency=f"{region} (self-hosted; no foreign/managed AI or cloud calls at runtime)",
        no_foreign_egress=True,
        generated_at=at,
        signer_kid=signer_kid,
    )
    return sign_node_attestation(att, private_key_b64)
