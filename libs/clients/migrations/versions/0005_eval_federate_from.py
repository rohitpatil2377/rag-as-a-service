"""eval_schedules.federate_from: continuous eval across the federation mesh

Revision ID: 0005_eval_federate_from
Revises: 0004_grant_delegation
Create Date: 2026-08-05

Adds ``federate_from`` to ``rag_identity.eval_schedules`` so a KB's continuous-evaluation
enrolment can name peer departments to also evaluate across the mesh (FR-G9 per-grant quality
attribution). Fresh installs get the column from ``eval_schedule._DDL``; this ALTERs an
already-created table. ``ADD COLUMN IF NOT EXISTS`` makes applying twice a no-op.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0005_eval_federate_from"
down_revision: str | None = "0004_grant_delegation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SCHEMA = "rag_identity"


def upgrade() -> None:
    op.execute(
        f'ALTER TABLE "{_SCHEMA}".eval_schedules '
        f"ADD COLUMN IF NOT EXISTS federate_from TEXT[] NOT NULL DEFAULT '{{}}'"
    )


def downgrade() -> None:
    op.execute(f'ALTER TABLE "{_SCHEMA}".eval_schedules DROP COLUMN IF EXISTS federate_from')
