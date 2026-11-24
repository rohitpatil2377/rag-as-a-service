"""Federation grants (Federated RAG, FR cross-tenant P2P) — publisher-approved access.

A grant is a cryptographic authorisation: an ISSUER department signs permission for a
SUBJECT department to query specific ``topics`` from specific ``kb_ids``, bounded by
``not_before``/``expires_at`` and revocable. It is the ONLY thing that opens a hole in
the platform's hard tenant isolation — no grant, no cross-dept read (the default 403
stays for un-granted federation).

Grants are inherently CROSS-tenant, so — like API keys, eval- and sync-schedules — they
live in one small GLOBAL table in the ``rag_identity`` schema, not a per-tenant schema.

The token is a detached Ed25519 signature over the grant's canonical claims. The
issuer signs with a per-process signing key; any peer validates with the public key
BEFORE executing a local search. A tampered or expired grant fails closed (empty
result), never an error. Signing/verification is pure and self-contained here so the
Go SovereignNode (Phase C) can reproduce the exact same canonical form + Ed25519 check.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from rag_common import get_logger, get_settings
from rag_schemas import Grant

_log = get_logger(__name__)

_IDENTITY_SCHEMA = "rag_identity"


# --- Ed25519 signing / verification (pure; mirrors the Go node's canonical form) -----


def _canonical_claims(grant: Grant) -> bytes:
    """Deterministic byte form signed by the issuer.

    Sorted keys, no whitespace, ISO-8601 UTC timestamps — a Go implementation must
    produce byte-identical output to validate a grant issued here (and vice versa).
    """
    claims = {
        "grant_id": grant.grant_id,
        "issuer_dept_id": grant.issuer_dept_id,
        "subject_dept_id": grant.subject_dept_id,
        "topics": sorted(grant.topics),
        "kb_ids": sorted(grant.kb_ids),
        "max_results_per_query": grant.max_results_per_query,
        "not_before": grant.not_before.astimezone(UTC).isoformat(),
        "expires_at": grant.expires_at.astimezone(UTC).isoformat(),
    }
    # A delegated grant binds its parent into the signature so the chain link is tamper-evident.
    # Added ONLY when present, so a root grant's canonical bytes are unchanged and the Go
    # SovereignNode still validates direct grants byte-for-byte (sorted keys make order moot).
    if grant.parent_grant_id:
        claims["parent_grant_id"] = grant.parent_grant_id
    return json.dumps(claims, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _load_signing_key() -> Ed25519PrivateKey:
    """The issuer's Ed25519 private key from config (base64 raw 32 bytes).

    Empty config → a deterministic dev key derived from the dev-JWT secret, so the
    in-memory profile signs and verifies with zero setup (never for production).
    """
    settings = get_settings()
    raw_b64 = getattr(settings, "grant_signing_key", "") or ""
    if raw_b64:
        return Ed25519PrivateKey.from_private_bytes(base64.b64decode(raw_b64))
    # Dev fallback: derive 32 deterministic bytes from the dev secret.
    import hashlib  # noqa: PLC0415

    seed = hashlib.sha256(f"grant-dev::{settings.dev_jwt_secret}".encode()).digest()
    return Ed25519PrivateKey.from_private_bytes(seed)


def sign_grant(grant: Grant) -> str:
    """Return a base64 detached Ed25519 signature over the grant's canonical claims."""
    key = _load_signing_key()
    sig = key.sign(_canonical_claims(grant))
    return base64.b64encode(sig).decode("ascii")


def _retired_public_keys() -> list[Ed25519PublicKey]:
    """Retired-but-still-trusted verification keys for a rotation overlap.

    Parsed from ``grant_retired_verify_keys`` (a JSON list of base64 raw 32-byte Ed25519 PUBLIC
    keys). A malformed entry is skipped, not fatal — a bad retired key must never take down
    verification of grants signed by the ACTIVE key."""
    raw = get_settings().grant_retired_verify_keys.strip()
    if not raw:
        return []
    try:
        items = json.loads(raw)
    except (ValueError, TypeError):
        _log.warning("grant_retired_keys_parse_failed")
        return []
    keys: list[Ed25519PublicKey] = []
    for item in items if isinstance(items, list) else []:
        try:
            keys.append(Ed25519PublicKey.from_public_bytes(base64.b64decode(str(item))))
        except (ValueError, TypeError):
            _log.warning("grant_retired_key_invalid")
    return keys


def verifying_keys() -> list[Ed25519PublicKey]:
    """The full keyring a local verifier trusts: the ACTIVE signing key's public half plus any
    RETIRED keys still inside their rotation overlap. New grants are signed by the active key;
    grants signed before a rotation keep verifying against a retired key until they expire."""
    return [_load_signing_key().public_key(), *_retired_public_keys()]


def current_key_id() -> str:
    """The active signing key's id (kid). Recorded on grant-issue audit events so a rotation is
    itself auditable and an operator can tell when a retired key is safe to drop."""
    return get_settings().grant_signing_key_id or "dev"


def verify_grant(grant: Grant, public_key: Ed25519PublicKey | None = None) -> bool:
    """True if ``grant.signed_token`` is a valid signature over the grant's claims.

    ``public_key`` given (a real P2P mesh: the issuer peer's known key) → check ONLY that key.
    Otherwise check the local KEYRING — the active signing key PLUS any retired keys still in
    their rotation overlap — so rotating the signing key never invalidates already-signed,
    unexpired grants. Verification is signature-only; expiry/revocation are checked separately
    by :meth:`Grant.is_active`."""
    try:
        sig = base64.b64decode(grant.signed_token)
    except (ValueError, TypeError):
        return False
    claims = _canonical_claims(grant)
    candidates = [public_key] if public_key is not None else verifying_keys()
    for pub in candidates:
        try:
            pub.verify(sig, claims)
            return True
        except InvalidSignature:
            continue
    return False


# --- Transitive delegation: bounded trust (A->B->C) ----------------------------------


class DelegationError(ValueError):
    """A re-delegation that would WIDEN access (or otherwise break the trust bound)."""


def attenuate(parent: Grant, child: Grant) -> None:
    """Enforce that ``child`` only NARROWS ``parent`` — the bounded-trust rule for A->B->C.

    A re-delegation may never grant more than the delegator itself holds: the issuer must be
    the parent's SUBJECT (you can only re-share what was shared with YOU), the topics and KBs
    must be subsets, the result cap no larger, and the validity window no wider. Raises
    :class:`DelegationError` on any widening — the store refuses to persist such a grant."""
    if child.issuer_dept_id != parent.subject_dept_id:
        raise DelegationError(
            f"only the grant's subject ({parent.subject_dept_id}) may re-delegate it, not {child.issuer_dept_id}"
        )
    extra_topics = set(child.topics) - set(parent.topics)
    if extra_topics:
        raise DelegationError(f"delegated topics exceed the parent grant: {sorted(extra_topics)}")
    extra_kbs = set(child.kb_ids) - set(parent.kb_ids)
    if extra_kbs:
        raise DelegationError(f"delegated kb_ids exceed the parent grant: {sorted(extra_kbs)}")
    if child.max_results_per_query > parent.max_results_per_query:
        raise DelegationError("delegated max_results_per_query exceeds the parent grant")
    if child.not_before < parent.not_before:
        raise DelegationError("delegated grant starts before the parent grant")
    if child.expires_at > parent.expires_at:
        raise DelegationError("delegated grant outlives the parent grant")


# --- Store (global rag_identity table + in-memory twin) -----------------------------


@runtime_checkable
class GrantStore(Protocol):
    async def ensure(self) -> None: ...
    async def create(self, grant: Grant) -> None: ...
    async def get(self, grant_id: str) -> Grant | None: ...
    async def list_issued(self, issuer_dept_id: str) -> list[Grant]: ...
    async def list_received(self, subject_dept_id: str) -> list[Grant]: ...
    async def revoke(self, grant_id: str, issuer_dept_id: str) -> bool: ...
    async def get_active(self, issuer_dept_id: str, subject_dept_id: str) -> list[Grant]: ...
    async def resolve_chain(self, grant_id: str) -> list[Grant]: ...
    async def is_chain_active(self, grant_id: str) -> bool: ...
    async def list_children(self, grant_id: str) -> list[Grant]: ...


class InMemoryGrantStore:
    def __init__(self) -> None:
        self._rows: dict[str, Grant] = {}

    async def ensure(self) -> None:
        return None

    async def create(self, grant: Grant) -> None:
        if grant.grant_id in self._rows:
            raise ValueError(f"grant {grant.grant_id} already exists")
        self._rows[grant.grant_id] = grant

    async def get(self, grant_id: str) -> Grant | None:
        return self._rows.get(grant_id)

    async def list_issued(self, issuer_dept_id: str) -> list[Grant]:
        return [g for g in self._rows.values() if g.issuer_dept_id == issuer_dept_id]

    async def list_received(self, subject_dept_id: str) -> list[Grant]:
        return [g for g in self._rows.values() if g.subject_dept_id == subject_dept_id and g.is_active]

    async def revoke(self, grant_id: str, issuer_dept_id: str) -> bool:
        g = self._rows.get(grant_id)
        if g is None:
            return False
        if g.issuer_dept_id != issuer_dept_id:
            raise PermissionError("only the issuer can revoke a grant")
        if g.revoked_at is None:
            self._rows[grant_id] = g.model_copy(update={"revoked_at": datetime.now(UTC)})
        return True

    async def get_active(self, issuer_dept_id: str, subject_dept_id: str) -> list[Grant]:
        return [
            g
            for g in self._rows.values()
            if g.issuer_dept_id == issuer_dept_id and g.subject_dept_id == subject_dept_id and g.is_active
        ]

    async def resolve_chain(self, grant_id: str) -> list[Grant]:
        chain: list[Grant] = []
        seen: set[str] = set()
        cur = self._rows.get(grant_id)
        while cur is not None and cur.grant_id not in seen:
            seen.add(cur.grant_id)
            chain.append(cur)
            cur = self._rows.get(cur.parent_grant_id) if cur.parent_grant_id else None
        return chain

    async def is_chain_active(self, grant_id: str) -> bool:
        chain = await self.resolve_chain(grant_id)
        return bool(chain) and all(g.is_active for g in chain)

    async def list_children(self, grant_id: str) -> list[Grant]:
        return [g for g in self._rows.values() if g.parent_grant_id == grant_id]


_DDL = (
    f'CREATE SCHEMA IF NOT EXISTS "{_IDENTITY_SCHEMA}"',
    f'''CREATE TABLE IF NOT EXISTS "{_IDENTITY_SCHEMA}".federation_grants (
        grant_id TEXT PRIMARY KEY,
        issuer_dept_id TEXT NOT NULL,
        subject_dept_id TEXT NOT NULL,
        topics TEXT[] NOT NULL,
        kb_ids TEXT[] NOT NULL,
        max_results_per_query INTEGER NOT NULL DEFAULT 5,
        not_before TIMESTAMPTZ NOT NULL,
        expires_at TIMESTAMPTZ NOT NULL,
        signed_token TEXT NOT NULL,
        revoked_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        parent_grant_id TEXT REFERENCES "{_IDENTITY_SCHEMA}".federation_grants (grant_id)
    )''',
    # ADD COLUMN IF NOT EXISTS: this table predates parent_grant_id on already-provisioned
    # deployments, so the CREATE TABLE IF NOT EXISTS above silently no-ops and never adds it —
    # mirror the migration-safe pattern used elsewhere (e.g. DocumentEnvelope.content_hash).
    f'ALTER TABLE "{_IDENTITY_SCHEMA}".federation_grants ADD COLUMN IF NOT EXISTS parent_grant_id '
    f'TEXT REFERENCES "{_IDENTITY_SCHEMA}".federation_grants (grant_id)',
    f'CREATE INDEX IF NOT EXISTS ix_grants_issuer ON "{_IDENTITY_SCHEMA}".federation_grants (issuer_dept_id)',
    f'CREATE INDEX IF NOT EXISTS ix_grants_subject ON "{_IDENTITY_SCHEMA}".federation_grants (subject_dept_id)',
    f'CREATE INDEX IF NOT EXISTS ix_grants_parent ON "{_IDENTITY_SCHEMA}".federation_grants (parent_grant_id)',
)

_COLS = (
    "grant_id, issuer_dept_id, subject_dept_id, topics, kb_ids, max_results_per_query, "
    "not_before, expires_at, signed_token, revoked_at, created_at, parent_grant_id"
)


class PostgresGrantStore:
    def __init__(self, engine: AsyncEngine | None = None) -> None:
        self._engine = engine or create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)

    async def ensure(self) -> None:
        async with self._engine.begin() as c:
            for stmt in _DDL:
                await c.execute(text(stmt))

    async def create(self, grant: Grant) -> None:
        await self.ensure()
        async with self._engine.begin() as c:
            await c.execute(
                text(
                    f'INSERT INTO "{_IDENTITY_SCHEMA}".federation_grants ({_COLS}) VALUES '
                    "(:grant_id, :issuer_dept_id, :subject_dept_id, :topics, :kb_ids, "
                    ":max_results_per_query, :not_before, :expires_at, :signed_token, "
                    ":revoked_at, :created_at, :parent_grant_id)"
                ),
                {
                    **grant.model_dump(),
                    "created_at": grant.created_at or datetime.now(UTC),
                },
            )

    async def get(self, grant_id: str) -> Grant | None:
        await self.ensure()
        sql = f'SELECT {_COLS} FROM "{_IDENTITY_SCHEMA}".federation_grants WHERE grant_id = :g'
        async with self._engine.connect() as c:
            row = (await c.execute(text(sql), {"g": grant_id})).mappings().first()
        return Grant(**dict(row)) if row else None

    async def list_issued(self, issuer_dept_id: str) -> list[Grant]:
        await self.ensure()
        sql = (
            f'SELECT {_COLS} FROM "{_IDENTITY_SCHEMA}".federation_grants '
            "WHERE issuer_dept_id = :d ORDER BY created_at DESC"
        )
        async with self._engine.connect() as c:
            rows = (await c.execute(text(sql), {"d": issuer_dept_id})).mappings().all()
        return [Grant(**dict(r)) for r in rows]

    async def list_received(self, subject_dept_id: str) -> list[Grant]:
        await self.ensure()
        sql = (
            f'SELECT {_COLS} FROM "{_IDENTITY_SCHEMA}".federation_grants '
            "WHERE subject_dept_id = :d AND revoked_at IS NULL "
            "AND now() >= not_before AND now() < expires_at ORDER BY created_at DESC"
        )
        async with self._engine.connect() as c:
            rows = (await c.execute(text(sql), {"d": subject_dept_id})).mappings().all()
        return [Grant(**dict(r)) for r in rows]

    async def revoke(self, grant_id: str, issuer_dept_id: str) -> bool:
        await self.ensure()
        existing = await self.get(grant_id)
        if existing is None:
            return False
        if existing.issuer_dept_id != issuer_dept_id:
            raise PermissionError("only the issuer can revoke a grant")
        async with self._engine.begin() as c:
            await c.execute(
                text(
                    f'UPDATE "{_IDENTITY_SCHEMA}".federation_grants SET revoked_at = :at '
                    "WHERE grant_id = :g AND revoked_at IS NULL"
                ),
                {"at": datetime.now(UTC), "g": grant_id},
            )
        return True

    async def get_active(self, issuer_dept_id: str, subject_dept_id: str) -> list[Grant]:
        await self.ensure()
        sql = (
            f'SELECT {_COLS} FROM "{_IDENTITY_SCHEMA}".federation_grants '
            "WHERE issuer_dept_id = :i AND subject_dept_id = :s AND revoked_at IS NULL "
            "AND now() >= not_before AND now() < expires_at ORDER BY created_at DESC"
        )
        async with self._engine.connect() as c:
            rows = (await c.execute(text(sql), {"i": issuer_dept_id, "s": subject_dept_id})).mappings().all()
        return [Grant(**dict(r)) for r in rows]

    async def resolve_chain(self, grant_id: str) -> list[Grant]:
        await self.ensure()
        chain: list[Grant] = []
        seen: set[str] = set()
        cur = await self.get(grant_id)
        while cur is not None and cur.grant_id not in seen:
            seen.add(cur.grant_id)
            chain.append(cur)
            cur = await self.get(cur.parent_grant_id) if cur.parent_grant_id else None
        return chain

    async def is_chain_active(self, grant_id: str) -> bool:
        chain = await self.resolve_chain(grant_id)
        return bool(chain) and all(g.is_active for g in chain)

    async def list_children(self, grant_id: str) -> list[Grant]:
        await self.ensure()
        sql = (
            f'SELECT {_COLS} FROM "{_IDENTITY_SCHEMA}".federation_grants '
            "WHERE parent_grant_id = :g ORDER BY created_at DESC"
        )
        async with self._engine.connect() as c:
            rows = (await c.execute(text(sql), {"g": grant_id})).mappings().all()
        return [Grant(**dict(r)) for r in rows]
