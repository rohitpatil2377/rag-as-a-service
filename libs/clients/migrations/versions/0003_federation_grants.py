"""federation_grants: Federated RAG cross-tenant grant registry

Revision ID: 0003_federation_grants
Revises: 0002_rag_identity
Create Date: 2026-08-03

Adds the global ``rag_identity.federation_grants`` table (Federated RAG, cross-tenant
P2P RAG). A grant is a publisher-approved, Ed25519-signed authorisation for one
department to query specific topics from another's KBs — the only mechanism that opens
the platform's hard tenant-isolation wall, and always revocable/auditable.

Like the other global tables (0002), this executes the very same idempotent DDL the
store adapter uses at runtime — it imports ``grants._DDL`` so there is a SINGLE source
of truth and the runtime ``ensure()`` never drifts from the migration. Every statement
is ``CREATE … IF NOT EXISTS``, so applying twice is a no-op.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0003_federation_grants"
down_revision: str | None = "0002_rag_identity"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    from svc_control_plane.grants import _DDL  # noqa: PLC0415 — module _DDL is the source of truth

    for stmt in _DDL:
        op.execute(stmt)


def downgrade() -> None:
    op.execute('DROP TABLE IF EXISTS "rag_identity".federation_grants')
