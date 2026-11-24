"""Human-in-the-Loop gates (TDD §4.6 · FR-F6, CAG POC A4) + FR-E4 faithfulness flags.

Typed review/approval workflow over grounded answers. No AI output is auto-finalised:
an answer enters as ``pending`` and a reviewer must act — view / accept / modify / flag
/ override — with every action written to the tamper-evident audit log.

FR-E4 reuses this exact flow: a faithfulness-borderline answer opens a PENDING review
with ``reason="faithfulness_warning"`` and the ``warnings`` (unsupported claims) attached,
so the reviewer sees precisely what wasn't supported. No new status, no parallel path.

Persistence is behind :class:`HitlStore` (Postgres for real deployments, in-memory for
the zero-infra dev/test profile), consistent with every other store.

Schema note (per-tenant ALTER caveat): the ``reason``/``warnings`` columns are added to
``t_<slug>.hitl_reviews`` via ``CREATE TABLE IF NOT EXISTS``, so they apply cleanly to
fresh tenants; EXISTING per-tenant tables in a live DB need an ``ALTER TABLE`` — a real
per-tenant migration is Phase-1 catalog work (see ADR-0005).
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from svc_guardrails_eval import AuditService

from rag_common import NotFoundError, get_logger, get_settings, pg_schema
from rag_schemas import GroundedAnswer
from rag_schemas.common import AuditAction

_log = get_logger(__name__)


class HitlStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    MODIFIED = "modified"
    FLAGGED = "flagged"
    OVERRIDDEN = "overridden"


class HitlAction(StrEnum):
    VIEW = "view"
    ACCEPT = "accept"
    MODIFY = "modify"
    FLAG = "flag"
    OVERRIDE = "override"


_ACTION_TO_STATUS: dict[HitlAction, HitlStatus] = {
    HitlAction.ACCEPT: HitlStatus.ACCEPTED,
    HitlAction.MODIFY: HitlStatus.MODIFIED,
    HitlAction.FLAG: HitlStatus.FLAGGED,
    HitlAction.OVERRIDE: HitlStatus.OVERRIDDEN,
}
_ACTION_TO_AUDIT: dict[HitlAction, AuditAction] = {
    HitlAction.VIEW: AuditAction.HITL_VIEW,
    HitlAction.ACCEPT: AuditAction.HITL_ACCEPT,
    HitlAction.MODIFY: AuditAction.HITL_MODIFY,
    HitlAction.FLAG: AuditAction.HITL_FLAG,
    HitlAction.OVERRIDE: AuditAction.HITL_OVERRIDE,
}

# Review reason opened by the FR-E4 faithfulness pass.
REASON_FAITHFULNESS = "faithfulness_warning"


class HitlReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str
    tenant_id: str
    trace_id: str
    status: HitlStatus
    query: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    reason: str | None = Field(default=None, description="Why the answer is in review (e.g. faithfulness_warning).")
    warnings: list[str] = Field(default_factory=list, description="Unsupported claims / reviewer-facing warnings.")
    reviewer: str | None = None
    note: str | None = None
    updated_at: datetime
    # --- the verification context a reviewer needs, and the identity that keeps this review
    # unique. All nullable: every review written before this existed has them NULL, and the
    # FAITHFULNESS audit event remains the authoritative provenance record — these are a
    # convenient projection for whoever opens the queue, not a second source of truth.
    faithfulness_score: float | None = Field(default=None, description="The batch's measured score.")
    faithfulness_threshold: float | None = Field(
        default=None, description="The gate applied when the decision was made, so an old review stays readable."
    )
    model_version: str | None = Field(default=None, description="Model that produced the material under review.")
    #: Stable identity of the EVIDENCE, not of the answer: tenant, trace, decision kind and the
    #: sorted chunk ids. Deliberately excludes the score — a fluctuation must not mint a second
    #: review for the same evidence. A partial unique index on it makes dedup durable rather
    #: than merely request-local; NULL means "not deduplicated", which is every legacy row.
    evidence_fingerprint: str | None = Field(default=None, description="Dedup identity of the reviewed evidence.")


# --- Persistence -------------------------------------------------------------
@runtime_checkable
class HitlStore(Protocol):
    async def ensure(self, tenant_id: str) -> None: ...
    async def insert(self, review: HitlReview) -> HitlReview:
        """Insert, or return the existing review for the same evidence fingerprint."""
        ...

    async def fetch(self, tenant_id: str, review_id: str) -> HitlReview | None: ...
    async def update(self, tenant_id: str, review: HitlReview) -> None: ...
    async def list(self, tenant_id: str, status: HitlStatus | None, limit: int) -> list[HitlReview]: ...


def _ddl(schema: str) -> list[str]:
    return [
        f'CREATE SCHEMA IF NOT EXISTS "{schema}"',
        f'''CREATE TABLE IF NOT EXISTS "{schema}".hitl_reviews (
            review_id TEXT PRIMARY KEY,
            trace_id TEXT NOT NULL,
            status TEXT NOT NULL,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            citations JSONB NOT NULL DEFAULT '[]',
            reason TEXT,
            warnings JSONB NOT NULL DEFAULT '[]',
            reviewer TEXT,
            note TEXT,
            updated_at TIMESTAMPTZ NOT NULL,
            faithfulness_score DOUBLE PRECISION,
            faithfulness_threshold DOUBLE PRECISION,
            model_version TEXT,
            evidence_fingerprint TEXT
        )''',
        # Declared in the CREATE above AND as ALTERs here, exactly like `documents.content_hash`
        # and `federation_grants.parent_grant_id`: a deployment provisioned before this commit
        # already HAS the table, and `CREATE TABLE IF NOT EXISTS` silently no-ops on it. That
        # trap has cost this repo two columns that never reached a running deployment.
        f'ALTER TABLE "{schema}".hitl_reviews ADD COLUMN IF NOT EXISTS faithfulness_score DOUBLE PRECISION',
        f'ALTER TABLE "{schema}".hitl_reviews ADD COLUMN IF NOT EXISTS faithfulness_threshold DOUBLE PRECISION',
        f'ALTER TABLE "{schema}".hitl_reviews ADD COLUMN IF NOT EXISTS model_version TEXT',
        f'ALTER TABLE "{schema}".hitl_reviews ADD COLUMN IF NOT EXISTS evidence_fingerprint TEXT',
        # PARTIAL, so legacy rows — which all have a NULL fingerprint — do not collide with each
        # other. The schema is already per-tenant, so (trace_id, fingerprint) here IS
        # (tenant, trace, fingerprint). This is what makes dedup survive a retried request: a
        # pre-insert SELECT would still race two concurrent coverage passes.
        f'''CREATE UNIQUE INDEX IF NOT EXISTS hitl_reviews_evidence_uniq
            ON "{schema}".hitl_reviews (trace_id, evidence_fingerprint)
            WHERE evidence_fingerprint IS NOT NULL''',
    ]


class PostgresHitlStore:
    """Durable HITL persistence, schema-per-tenant."""

    def __init__(self, engine: AsyncEngine | None = None) -> None:
        self._engine = engine or create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)

    async def ensure(self, tenant_id: str) -> None:
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as conn:
            for stmt in _ddl(schema):
                await conn.execute(text(stmt))

    async def insert(self, review: HitlReview) -> HitlReview:
        """Insert, or return the review that already covers this evidence.

        ATOMIC, not read-then-write. Two coverage passes over the same flagged batch, or two
        retries of the same request, arrive concurrently; a pre-insert SELECT would let both
        see nothing and both write. `ON CONFLICT DO NOTHING RETURNING` returns no row when the
        partial unique index rejects the duplicate, and the follow-up SELECT then hands back
        the review that won — so the caller always gets a real review and never a second one
        for the same evidence."""
        await self.ensure(review.tenant_id)
        schema = pg_schema(review.tenant_id)
        async with self._engine.begin() as conn:
            inserted = (
                await conn.execute(
                    text(
                        f'''INSERT INTO "{schema}".hitl_reviews
                            (review_id, trace_id, status, query, answer, citations, reason, warnings,
                             updated_at, faithfulness_score, faithfulness_threshold, model_version,
                             evidence_fingerprint)
                            VALUES (:rid, :tid, :status, :q, :a, CAST(:cits AS JSONB), :reason,
                                    CAST(:warn AS JSONB), :ts, :fscore, :fthresh, :model, :fp)
                            ON CONFLICT DO NOTHING
                            RETURNING review_id'''
                    ),
                    {
                        "rid": review.review_id,
                        "tid": review.trace_id,
                        "status": review.status.value,
                        "q": review.query,
                        "a": review.answer,
                        "cits": json.dumps(review.citations),
                        "reason": review.reason,
                        "warn": json.dumps(review.warnings),
                        "ts": review.updated_at,
                        "fscore": review.faithfulness_score,
                        "fthresh": review.faithfulness_threshold,
                        "model": review.model_version,
                        "fp": review.evidence_fingerprint,
                    },
                )
            ).first()
            if inserted is not None:
                return review
            existing = (
                (
                    await conn.execute(
                        text(
                            f'''SELECT * FROM "{schema}".hitl_reviews
                                WHERE trace_id = :tid AND evidence_fingerprint = :fp'''
                        ),
                        {"tid": review.trace_id, "fp": review.evidence_fingerprint},
                    )
                )
                .mappings()
                .first()
            )
        # A conflict with no recoverable row would mean the index fired on something other than
        # the fingerprint; returning the unsaved review would then claim a persistence that did
        # not happen, so say so instead.
        if existing is None:
            raise RuntimeError("hitl insert conflicted but no existing review could be read back")
        return _row_to_review(review.tenant_id, dict(existing))

    async def fetch(self, tenant_id: str, review_id: str) -> HitlReview | None:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text(f'SELECT * FROM "{schema}".hitl_reviews WHERE review_id = :rid'), {"rid": review_id}
                    )
                )
                .mappings()
                .first()
            )
        return None if row is None else _row_to_review(tenant_id, dict(row))

    async def update(self, tenant_id: str, review: HitlReview) -> None:
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f'''UPDATE "{schema}".hitl_reviews
                        SET status=:s, reviewer=:r, note=:n, answer=:a, updated_at=:ts WHERE review_id=:rid'''
                ),
                {
                    "s": review.status.value,
                    "r": review.reviewer,
                    "n": review.note,
                    "a": review.answer,
                    "ts": review.updated_at,
                    "rid": review.review_id,
                },
            )

    async def list(self, tenant_id: str, status: HitlStatus | None, limit: int) -> list[HitlReview]:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        clause = "WHERE status = :status " if status is not None else ""
        params: dict[str, Any] = {"limit": limit}
        if status is not None:
            params["status"] = status.value
        async with self._engine.connect() as conn:
            rows = (
                (
                    await conn.execute(
                        text(f'SELECT * FROM "{schema}".hitl_reviews {clause}ORDER BY updated_at DESC LIMIT :limit'),
                        params,
                    )
                )
                .mappings()
                .all()
            )
        return [_row_to_review(tenant_id, dict(r)) for r in rows]


class InMemoryHitlStore:
    """Zero-infra HITL persistence for the in-memory dev/test profile."""

    def __init__(self) -> None:
        self._db: dict[str, dict[str, HitlReview]] = defaultdict(dict)

    async def ensure(self, tenant_id: str) -> None:
        self._db[tenant_id]

    async def insert(self, review: HitlReview) -> HitlReview:
        """Same dedup contract as Postgres, so the two backends cannot disagree.

        Without this the in-memory profile would happily write a second review for identical
        evidence and every dedup test would pass only against a real database."""
        if review.evidence_fingerprint is not None:
            for existing in self._db[review.tenant_id].values():
                if (
                    existing.evidence_fingerprint == review.evidence_fingerprint
                    and existing.trace_id == review.trace_id
                ):
                    return existing
        self._db[review.tenant_id][review.review_id] = review
        return review

    async def fetch(self, tenant_id: str, review_id: str) -> HitlReview | None:
        return self._db[tenant_id].get(review_id)

    async def update(self, tenant_id: str, review: HitlReview) -> None:
        self._db[tenant_id][review.review_id] = review

    async def list(self, tenant_id: str, status: HitlStatus | None, limit: int) -> list[HitlReview]:
        reviews = [r for r in self._db[tenant_id].values() if status is None or r.status is status]
        reviews.sort(key=lambda r: r.updated_at, reverse=True)
        return reviews[:limit]


# --- Service -----------------------------------------------------------------
class HitlService:
    """Durable HITL gate. Reviews start pending; reviewer actions are audited."""

    def __init__(self, audit: AuditService, store: HitlStore) -> None:
        self._audit = audit
        self._store = store

    async def create(
        self,
        answer: GroundedAnswer,
        *,
        actor: str,
        reason: str | None = None,
        warnings: list[str] | None = None,
        faithfulness_threshold: float | None = None,
        model_version: str | None = None,
        evidence_fingerprint: str | None = None,
    ) -> tuple[HitlReview, str]:
        """Open a pending review for an answer (no auto-finalisation). FR-E4 passes reason/warnings.

        Returns the review AND which of two things happened — ``"created"`` or
        ``"deduplicated"``. The caller needs to know: a composite answer may only tell its
        reader that material is "pending review" if a review really is, and an audit record
        that cannot distinguish a fresh review from a repeat of one is not a provenance trail.
        (The third outcome, ``"failed"``, is the caller's to record — this raises.)

        ``evidence_fingerprint`` identifies the EVIDENCE rather than the answer, so a second
        pass over the same flagged batch returns the first review instead of minting another."""
        review = HitlReview(
            review_id=str(uuid.uuid4()),
            tenant_id=answer.tenant_id,
            trace_id=answer.trace_id or "",
            status=HitlStatus.PENDING,
            query=answer.query,
            answer=answer.answer,
            citations=[c.chunk_id for c in answer.citations],
            reason=reason,
            warnings=warnings or [],
            updated_at=datetime.now(UTC),
            faithfulness_score=answer.faithfulness_score,
            faithfulness_threshold=faithfulness_threshold,
            model_version=model_version,
            evidence_fingerprint=evidence_fingerprint,
        )
        effective = await self._store.insert(review)
        outcome = "created" if effective.review_id == review.review_id else "deduplicated"
        _log.info("hitl_created", review_id=effective.review_id, reason=reason, outcome=outcome)
        return effective, outcome

    async def get(self, tenant_id: str, review_id: str) -> HitlReview:
        review = await self._store.fetch(tenant_id, review_id)
        if review is None:
            raise NotFoundError("hitl review not found", details={"review_id": review_id})
        return review

    async def list(self, tenant_id: str, status: HitlStatus | None = None, limit: int = 100) -> list[HitlReview]:
        """Review queue for the console (defaults to newest-first, all statuses)."""
        return await self._store.list(tenant_id, status, limit)

    async def act(
        self,
        tenant_id: str,
        review_id: str,
        action: HitlAction,
        *,
        reviewer: str,
        note: str | None = None,
        modified_answer: str | None = None,
    ) -> HitlReview:
        """Apply a reviewer action; update status (except VIEW) and audit it (FR-F6/G4)."""
        review = await self.get(tenant_id, review_id)
        if action is not HitlAction.VIEW:
            new_answer = modified_answer if (action is HitlAction.MODIFY and modified_answer) else review.answer
            review = review.model_copy(
                update={
                    "status": _ACTION_TO_STATUS.get(action, review.status),
                    "reviewer": reviewer,
                    "note": note,
                    "answer": new_answer,
                    "updated_at": datetime.now(UTC),
                }
            )
            await self._store.update(tenant_id, review)
        await self._audit.record(
            tenant_id=tenant_id,
            trace_id=review.trace_id,
            action=_ACTION_TO_AUDIT[action],
            actor=reviewer,
            payload={"review_id": review_id, "status": review.status.value, "note": note},
        )
        _log.info("hitl_action", review_id=review_id, action=action, status=review.status)
        return review


def _row_to_review(tenant_id: str, m: Mapping[str, Any]) -> HitlReview:
    def _list(v: object) -> list[str]:
        return v if isinstance(v, list) else json.loads(v)  # type: ignore[arg-type]

    return HitlReview(
        review_id=m["review_id"],
        tenant_id=tenant_id,
        trace_id=m["trace_id"],
        status=HitlStatus(m["status"]),
        query=m["query"],
        answer=m["answer"],
        citations=_list(m["citations"]),
        reason=m["reason"],
        warnings=_list(m["warnings"]),
        reviewer=m["reviewer"],
        note=m["note"],
        updated_at=m["updated_at"],
        # `.get`, not `[...]`: a row read back through a path that does not SELECT * — or a
        # test fixture built before these columns existed — simply has no key, and a review
        # that predates them legitimately has none of this context.
        faithfulness_score=m.get("faithfulness_score"),
        faithfulness_threshold=m.get("faithfulness_threshold"),
        model_version=m.get("model_version"),
        evidence_fingerprint=m.get("evidence_fingerprint"),
    )
