"""rag_identity: version the global (non-tenant) schema

Revision ID: 0002_rag_identity
Revises: 0001_baseline
Create Date: 2026-07-13

Brings the GLOBAL ``rag_identity`` schema under version control. These tables are fixed
(not per-tenant): organisations, users, service-account API keys, the continuous-eval and
connector-sync schedules, runtime feature-flag overrides, egress-approval requests, and the
async bulk-ingestion job queue.

To keep a SINGLE source of truth (and avoid drift), this migration executes the very same
idempotent DDL the store adapters use at runtime — it imports each module's ``_DDL`` and runs
it. The runtime ``ensure()`` remains as a self-healing safety net for deployments that skip
``make migrate``; because every statement is ``CREATE … IF NOT EXISTS``, the two never conflict.

Per-tenant ``t_<slug>`` schemas are created at runtime by design (schema-per-tenant, FR-H1) and
are deliberately NOT managed here — see 0001_baseline and ADR-0003.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002_rag_identity"
down_revision: str | None = "0001_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _global_ddl() -> list[str]:
    """The idempotent DDL for every global rag_identity table, sourced from the store modules."""
    # Imported here (not at module top) so the module list stays the source of truth for the DDL.
    from svc_control_plane.api_keys import _DDL as API_KEYS_DDL  # noqa: PLC0415
    from svc_control_plane.dual_control import _DDL as DUAL_CONTROL_DDL  # noqa: PLC0415
    from svc_control_plane.egress_requests import _DDL as EGRESS_REQUESTS_DDL  # noqa: PLC0415
    from svc_control_plane.eval_schedule import _DDL as EVAL_SCHEDULE_DDL  # noqa: PLC0415
    from svc_control_plane.feature_flags import _DDL as FEATURE_FLAGS_DDL  # noqa: PLC0415
    from svc_control_plane.identity import _DDL as IDENTITY_DDL  # noqa: PLC0415
    from svc_control_plane.ingest_jobs import _DDL as INGEST_JOBS_DDL  # noqa: PLC0415
    from svc_control_plane.sync_jobs import _DDL as SYNC_JOBS_DDL  # noqa: PLC0415

    statements: list[str] = []
    for ddl in (
        IDENTITY_DDL,
        API_KEYS_DDL,
        EVAL_SCHEDULE_DDL,
        SYNC_JOBS_DDL,
        FEATURE_FLAGS_DDL,
        EGRESS_REQUESTS_DDL,
        INGEST_JOBS_DDL,
        DUAL_CONTROL_DDL,
    ):
        statements.extend(ddl)
    return statements


def upgrade() -> None:
    for stmt in _global_ddl():
        op.execute(stmt)


def downgrade() -> None:
    op.execute('DROP SCHEMA IF EXISTS "rag_identity" CASCADE')
