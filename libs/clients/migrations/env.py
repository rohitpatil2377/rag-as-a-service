"""Alembic environment.

The DB URL comes from ``RAG_POSTGRES_DSN`` (12-factor; no URL in code). Migrations
here manage the SHARED baseline (the ``public.tenants`` registry). Per-tenant schemas
(``t_<slug>``) and their tables are created idempotently by the store adapters at
runtime (schema-per-tenant isolation, FR-H1) — a documented, deliberate split.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine

from rag_common import get_settings


def _sync_url() -> str:
    # psycopg3 dialect works for sync Alembic runs too.
    return get_settings().postgres_dsn


def run_migrations_offline() -> None:
    context.configure(url=_sync_url(), literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_sync_url(), pool_pre_ping=True)
    with engine.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
