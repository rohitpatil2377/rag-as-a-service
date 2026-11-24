"""baseline: shared tenant registry

Revision ID: 0001_baseline
Revises:
Create Date: 2026-07-02

Creates the shared ``public.tenants`` registry. Per-tenant data schemas (``t_<slug>``)
with documents/chunks/tenant_config/audit_log/hitl_reviews tables are created by the
store adapters at runtime (schema-per-tenant, FR-H1) — this migration deliberately
manages only the shared baseline.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_baseline"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tenants",
        sa.Column("tenant_id", sa.Text(), primary_key=True),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column("schema_name", sa.Text(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        schema="public",
    )


def downgrade() -> None:
    op.drop_table("tenants", schema="public")
