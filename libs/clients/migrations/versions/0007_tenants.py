"""tenants: managed-cloud tenant registry (provider control plane)

Revision ID: 0007_tenants
Revises: 0006_org_units
Create Date: 2026-08-05

Adds the global ``rag_identity.tenants`` table — the provider's fleet record for managed-cloud
tenants (status, plan, owner). Runs the same idempotent DDL the store uses at runtime (imports
``tenant_control._DDL``), so the migration and runtime ``ensure()`` never drift.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0007_tenants"
down_revision: str | None = "0006_org_units"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    from svc_control_plane.tenant_control import _DDL  # noqa: PLC0415 — module _DDL is the source of truth

    for stmt in _DDL:
        op.execute(stmt)


def downgrade() -> None:
    op.execute('DROP TABLE IF EXISTS "rag_identity".tenants')
