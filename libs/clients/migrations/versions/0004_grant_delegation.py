"""federation_grants.parent_grant_id: transitive delegation (A->B->C)

Revision ID: 0004_grant_delegation
Revises: 0003_federation_grants
Create Date: 2026-08-05

Adds ``parent_grant_id`` to ``rag_identity.federation_grants`` so a grant can record the
grant it was RE-DELEGATED from — the link that makes a bounded A->B->C trust chain (each
hop only narrows access; revoking any ancestor kills the descendants). Fresh installs get
the column from ``grants._DDL`` directly; this migration ALTERs an already-created table.
``ADD COLUMN IF NOT EXISTS`` makes applying twice a no-op.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0004_grant_delegation"
down_revision: str | None = "0003_federation_grants"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SCHEMA = "rag_identity"


def upgrade() -> None:
    op.execute(
        f'ALTER TABLE "{_SCHEMA}".federation_grants '
        f"ADD COLUMN IF NOT EXISTS parent_grant_id TEXT "
        f'REFERENCES "{_SCHEMA}".federation_grants (grant_id)'
    )
    op.execute(f'CREATE INDEX IF NOT EXISTS ix_grants_parent ON "{_SCHEMA}".federation_grants (parent_grant_id)')


def downgrade() -> None:
    op.execute(f'DROP INDEX IF EXISTS "{_SCHEMA}".ix_grants_parent')
    op.execute(f'ALTER TABLE "{_SCHEMA}".federation_grants DROP COLUMN IF EXISTS parent_grant_id')
