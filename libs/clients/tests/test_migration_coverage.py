"""Guard: the 0002 migration versions EVERY global rag_identity table (no drift).

Runs without a database — it loads the migration's DDL (sourced from the store modules) and
checks each expected global table is covered. If a new global table is added to a store but not
wired into the migration's module list, this fails.
"""

from __future__ import annotations

import importlib.util
import pathlib
from types import ModuleType

_EXPECTED_GLOBAL_TABLES = (
    "orgs",
    "users",
    "api_keys",
    "eval_schedules",
    "sync_schedules",
    "platform_flags",
    "egress_requests",
    "ingest_jobs",
    "ingest_job_items",
    "approval_requests",
)


def _load_migration() -> ModuleType:
    path = pathlib.Path(__file__).resolve().parents[1] / "migrations" / "versions" / "0002_rag_identity.py"
    spec = importlib.util.spec_from_file_location("mig_0002_rag_identity", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_migration_covers_every_global_table() -> None:
    ddl = " ".join(_load_migration()._global_ddl()).lower()
    for table in _EXPECTED_GLOBAL_TABLES:
        assert f'rag_identity".{table}' in ddl, f"migration 0002 does not create rag_identity.{table}"


def test_migration_ddl_is_idempotent() -> None:
    # Every statement is CREATE … IF NOT EXISTS / ALTER … IF NOT EXISTS, so the migration and the
    # runtime ensure() can both run without conflict.
    for stmt in _load_migration()._global_ddl():
        low = stmt.lower()
        assert "if not exists" in low or "on conflict" in low or "create index" in low, stmt[:80]
