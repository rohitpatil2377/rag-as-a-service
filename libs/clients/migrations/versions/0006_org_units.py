"""org_units: per-tenant nested RBAC-scope tree

Revision ID: 0006_org_units
Revises: 0005_eval_federate_from
Create Date: 2026-08-05

Adds the global ``rag_identity.org_units`` table — the nested departmental tree each tenant's
hierarchical RBAC scope hangs on (a user/KB belongs to a unit; access flows down the tree). Runs
the same idempotent DDL the store uses at runtime (imports ``org_units._DDL``), so the migration
and runtime ``ensure()`` never drift.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0006_org_units"
down_revision: str | None = "0005_eval_federate_from"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    from svc_control_plane.org_units import _DDL  # noqa: PLC0415 — module _DDL is the source of truth

    for stmt in _DDL:
        op.execute(stmt)


def downgrade() -> None:
    op.execute('DROP TABLE IF EXISTS "rag_identity".org_units')
