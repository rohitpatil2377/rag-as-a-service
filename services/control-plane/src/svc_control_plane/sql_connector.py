"""Read-only RDBMS connector (FR-A1/D10) — introspect + execute governed SELECTs.

A tenant registers a database connection (DSN + an allowlist of tables). This connector
introspects the allowlisted tables' columns to ground NL→SQL, and executes only the
validated SELECT produced by :mod:`nl2sql` — never model-generated SQL directly. Execution
opens a NON-transactional read connection and never commits, so even a hypothetical
side-effect can't persist; on Postgres it also sets a per-statement timeout.

SQLAlchemy's engine is synchronous (the driver — psycopg for Postgres, the stdlib for
SQLite — needs no async extra), so calls are offloaded with ``asyncio.to_thread`` to keep the
event loop free. The same code path serves SQLite in dev/CI and Postgres/MySQL in production
by DSN alone.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from sqlalchemy import create_engine, inspect, text

from rag_common import get_logger
from rag_common.sovereignty import assert_sovereign

_log = get_logger(__name__)

# Columns whose NAME suggests a small enum/category — the ones NL→SQL filters on and most often
# guesses wrong. We sample only these (not every column) so the value-lookup stays cheap.
_CATEGORICAL_HINT = re.compile(
    r"(status|stage|state|type|category|kind|priority|level|mode|flag|role|tier|class|label|"
    r"currency|country|region|dept|department|active|enabled|source|gender|segment|channel|grade)",
    re.I,
)

# Longest sample value kept when profiling a categorical column; longer values are free text, not enums.
_MAX_SAMPLE_VALUE_LEN = 40


@runtime_checkable
class SqlConnector(Protocol):
    async def schema(self) -> dict[str, list[str]]: ...
    async def column_samples(
        self, schema: dict[str, list[str]], *, max_distinct: int = 15, max_cols_per_table: int = 12
    ) -> dict[str, dict[str, list[str]]]: ...
    async def run(self, sql: str, *, max_rows: int) -> tuple[list[str], list[list[Any]]]: ...
    async def aclose(self) -> None: ...


def assert_dsn_sovereign(dsn: str) -> None:
    """A networked database is an external dependency → its host must be egress-allowlisted.
    Local SQLite files (no host) are exempt."""
    parts = urlsplit(dsn)
    if parts.hostname:
        assert_sovereign(f"{parts.scheme.split('+')[0]}://{parts.hostname}")


# One SQLAlchemy DBAPI driver per dialect — the pure-python, sovereign-friendly one the image
# ships (psycopg v3, PyMySQL, pymssql, python-oracledb thin mode). SQLAlchemy's *default* driver
# for a bare scheme is often a different, UNINSTALLED one (postgresql→psycopg2, mysql→mysqldb,
# mssql→pyodbc), so a bare DSN would crash with ModuleNotFoundError. Pinning the driver is what
# lets ANY of these database types connect without a per-DB code change.
_DEFAULT_DRIVERS = {
    "postgresql": ("postgresql", "psycopg"),
    "postgres": ("postgresql", "psycopg"),
    "mysql": ("mysql", "pymysql"),
    "mariadb": ("mysql", "pymysql"),
    "mssql": ("mssql", "pymssql"),
    "oracle": ("oracle", "oracledb"),
}

# SQLAlchemy's historical default drivers that this image does NOT ship (they need C libraries /
# system ODBC). A DSN copied from a vendor doc often names one of these explicitly — map it to the
# installed pure-python equivalent for the same dialect so it connects instead of crashing.
_DRIVER_ALIAS = {
    "psycopg2": "psycopg",
    "pyodbc": "pymssql",
    "mysqldb": "pymysql",
    "mysqlconnector": "pymysql",
    "cx_oracle": "oracledb",
}


def _pin_driver(dsn: str) -> str:
    """Resolve a DSN to a driver this image actually ships, so ANY supported dialect connects
    without a per-DB code change: a bare ``scheme://…`` gets the dialect's installed driver; an
    explicit-but-uninstalled driver (``…+pyodbc://``) is swapped for its installed equivalent; an
    already-installed or unknown/local (``sqlite://``) driver is left untouched."""
    scheme, sep, rest = dsn.partition("://")
    if not sep:
        return dsn
    if "+" in scheme:
        base, _, drv = scheme.partition("+")
        alt = _DRIVER_ALIAS.get(drv.lower())
        return f"{base}+{alt}://{rest}" if alt else dsn
    mapped = _DEFAULT_DRIVERS.get(scheme.lower())
    if not mapped:
        return dsn
    base, driver = mapped
    return f"{base}+{driver}://{rest}"


class SqlAlchemyConnector:
    """Read-only connector over any SQLAlchemy-supported DSN, scoped to allowlisted tables."""

    def __init__(
        self, dsn: str, allowed_tables: set[str], *, statement_timeout_ms: int = 5000, connect_timeout_s: int = 4
    ) -> None:
        self._allowed = {t.lower() for t in allowed_tables}
        dsn = _pin_driver(dsn)
        self._is_pg = dsn.startswith(("postgresql", "postgres"))
        connect_args: dict[str, Any] = {}
        # A CONNECT timeout is mandatory: without it, a connected source whose host is unreachable
        # (e.g. a DB on a network the API can't route to) hangs the driver's TCP connect for minutes,
        # and because that's a block — not a raised exception — it stalls the WHOLE query (unified-KB
        # gather_evidence can't skip a hang). Bound it so a dead source fails fast and is skipped.
        if self._is_pg:
            connect_args["options"] = f"-c statement_timeout={statement_timeout_ms} -c default_transaction_read_only=on"
            connect_args["connect_timeout"] = connect_timeout_s  # libpq (psycopg)
        elif dsn.startswith(("mysql", "mariadb")):
            connect_args["connect_timeout"] = connect_timeout_s
        elif dsn.startswith("mssql"):
            connect_args["timeout"] = connect_timeout_s  # pyodbc login timeout
        self._engine = create_engine(dsn, connect_args=connect_args, pool_pre_ping=True)

    def _schema_sync(self) -> dict[str, list[str]]:
        inspector = inspect(self._engine)
        available = {t.lower() for t in inspector.get_table_names()}
        out: dict[str, list[str]] = {}
        for table in sorted(self._allowed & available):
            out[table] = [c["name"] for c in inspector.get_columns(table)]
        return out

    def _run_sync(self, sql: str, max_rows: int) -> tuple[list[str], list[list[Any]]]:
        with self._engine.connect() as conn:  # no begin()/commit() — read-only by construction
            result = conn.execute(text(sql))
            cols = list(result.keys())
            rows = [list(r) for r in result.fetchmany(max_rows)]
        return cols, rows

    def _samples_sync(
        self, schema: dict[str, list[str]], max_distinct: int, max_cols_per_table: int
    ) -> dict[str, dict[str, list[str]]]:
        """The distinct values of each low-cardinality, category-like column — the ground truth NL→SQL
        needs so it filters with REAL values (``stage IN ('won','proposal')``) instead of guessing.
        Only heuristic-categorical columns are sampled, up to ``max_cols_per_table`` PER TABLE (a
        per-table budget so a wide first table can't starve later ones), each read read-only; a column
        with more than ``max_distinct`` values (high cardinality) or non-text values is skipped."""
        prep = self._engine.dialect.identifier_preparer
        out: dict[str, dict[str, list[str]]] = {}
        with self._engine.connect() as conn:
            for table, cols in schema.items():
                budget = max_cols_per_table
                for col in cols:
                    if budget <= 0:
                        break
                    if not _CATEGORICAL_HINT.search(col):
                        continue
                    budget -= 1
                    try:
                        # S608 is a false positive here: `col` and `table` are IDENTIFIERS, which
                        # cannot be bound as parameters, and both come from SQLAlchemy's own
                        # introspection rather than user input. They are quoted with the dialect's
                        # preparer, which is the correct defence for an identifier.
                        q = text(
                            f"SELECT DISTINCT {prep.quote(col)} AS v "  # noqa: S608
                            f"FROM {prep.quote(table)} WHERE {prep.quote(col)} IS NOT NULL"
                        )
                        vals = [r[0] for r in conn.execute(q).fetchmany(max_distinct + 1)]
                    except Exception as exc:
                        # Profiling is best-effort: a column we cannot read simply is not profiled.
                        _log.debug("column_profile_skipped", table=table, column=col, error=str(exc)[:120])
                        continue
                    picked = [
                        str(v).strip()
                        for v in vals
                        if isinstance(v, str) and 0 < len(str(v).strip()) <= _MAX_SAMPLE_VALUE_LEN
                    ]
                    if picked and len(vals) <= max_distinct:
                        out.setdefault(table, {})[col] = picked
        return out

    async def column_samples(
        self, schema: dict[str, list[str]], *, max_distinct: int = 15, max_cols_per_table: int = 12
    ) -> dict[str, dict[str, list[str]]]:
        return await asyncio.to_thread(self._samples_sync, schema, max_distinct, max_cols_per_table)

    async def schema(self) -> dict[str, list[str]]:
        return await asyncio.to_thread(self._schema_sync)

    async def run(self, sql: str, *, max_rows: int) -> tuple[list[str], list[list[Any]]]:
        return await asyncio.to_thread(self._run_sync, sql, max_rows)

    async def aclose(self) -> None:
        await asyncio.to_thread(self._engine.dispose)
