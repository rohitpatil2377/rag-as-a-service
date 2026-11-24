"""Retention-driven automated erasure scheduler (DPDP Act, FR-H8 companion) — finds documents
past their per-KB retention window and erases them without a human trigger.

``compliance.erase`` / ``compliance.apply_retention`` are the ON-DEMAND half: a data principal
(or an operator running a manual sweep) asks explicitly. This module is the missing
RETENTION-DRIVEN half: a background loop is meant to call :func:`run_scheduled_erasure` for
every enrolled KB on a timer, so time-based expiry happens even if nobody asks.

Kept deliberately PURE and DECOUPLED from the rest of the control plane: no catalog client, no
DB, no audit chain, no HTTP here. The actual erasure (cascade delete + signed certificate +
audit-chain record) is a CALLER-SUPPLIED async callback (``erase_fn``) — this module only
decides WHICH documents are due and WHETHER to call it. That seam is what makes the scheduling
decision unit-testable with a fake in milliseconds, and what lets the parent wire it to
whatever erasure implementation it likes (today: ``svc_control_plane.compliance.erase``)
without this module importing pipeline/wiring/catalog itself.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from svc_control_plane.compliance import ErasureCertificate


class RetentionPolicy(BaseModel):
    """A KB's automated-erasure enrolment: erase documents older than ``retention_days`` on
    the next scheduled sweep. ``enabled=False`` lets an operator pause auto-erasure for a KB
    without losing the configured window."""

    kb_id: str
    retention_days: int = Field(ge=1, description="Documents older than this are due for erasure.")
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class RetainedDoc:
    """The minimal shape the scheduler needs to know about a document: its id and when it was
    created. Deliberately NOT the full catalog document model — this module has no catalog
    dependency, so the parent projects whatever it has into this before calling in."""

    doc_id: str
    created_at: datetime


def due_document_ids(*, now: datetime, retention_days: int, docs: Sequence[RetainedDoc]) -> list[str]:
    """Doc ids whose age strictly EXCEEDS ``retention_days`` — pure, deterministic, sorted.

    The boundary is exclusive: a document created exactly ``retention_days`` ago is NOT yet
    due (mirrors ``compliance.apply_retention``'s ``ingested_at < cutoff`` check, so on-demand
    and scheduled retention agree on the edge case).
    """
    cutoff = now - timedelta(days=retention_days)
    return sorted(d.doc_id for d in docs if d.created_at < cutoff)


async def run_scheduled_erasure(
    *,
    policy: RetentionPolicy,
    now: datetime,
    docs: Sequence[RetainedDoc],
    erase_fn: Callable[[str, list[str]], Awaitable[ErasureCertificate]],
) -> ErasureCertificate | None:
    """Run one scheduled sweep for ``policy``'s KB.

    Computes the due document ids; if the policy is disabled or nothing is due, returns
    ``None`` WITHOUT calling ``erase_fn`` (no-op sweeps must not touch the injected callback,
    so tests can assert it was never invoked). Otherwise calls ``erase_fn(kb_id, due_ids)``
    once and returns its certificate.

    ``erase_fn`` is the only place this module touches the outside world — the parent supplies
    a callable, typically a thin wrapper around
    ``compliance.erase(kb_id, actor=..., doc_ids=doc_ids, reason="retention_policy", now=now)``,
    so the scheduling DECISION (this module) stays fully decoupled and unit-testable from the
    erasure MECHANICS (owned by ``compliance``).
    """
    if not policy.enabled:
        return None
    due_ids = due_document_ids(now=now, retention_days=policy.retention_days, docs=docs)
    if not due_ids:
        return None
    return await erase_fn(policy.kb_id, due_ids)
