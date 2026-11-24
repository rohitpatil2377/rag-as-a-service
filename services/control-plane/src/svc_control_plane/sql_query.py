"""Governed NL→SQL service (FR-D10) — register one or more DBs, ask in English, read safely.

A KB may register MANY read-only DB connections (``sql_connection.SqlConnectionStore``),
one marked default. Query-time routing is EXPLICIT: ``ask``/``execute_sql`` take an
optional ``connection_id`` and fall back to the KB's default when omitted — never
schema-sniffed/auto-picked across connections (see sql_connection.py's module docstring
for why). Two entry points: ``execute_sql`` runs an analyst-supplied SQL through the
governance layer (useful and testable without a model), and ``ask`` does the full
NL→SQL→govern→run flow. Both audit an ``SQL_QUERY`` event carrying the connection used,
the question, the model SQL, the validated SQL, and the row count, so every database
read is attributable.
"""

from __future__ import annotations

import uuid
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from rag_common import get_logger, get_settings
from rag_common.errors import ConfigError, NotFoundError, ValidationError
from rag_schemas import RetrievedSpan
from rag_schemas.common import AuditAction, Modality
from svc_control_plane import nl2sql, tenant_crypto, wiring
from svc_control_plane.sql_connection import StoredSqlConnection
from svc_control_plane.sql_connector import SqlAlchemyConnector, SqlConnector, assert_dsn_sovereign

_log = get_logger(__name__)


class SqlConnectionInfo(BaseModel):
    """Non-secret view of a KB's DEFAULT DB connection (back-compat single-connection
    shape — Dashboard/quick checks). Use SqlConnectionItem/list_connections for the
    full multi-connection picture."""

    configured: bool
    conn_id: str | None = None
    name: str | None = None
    dialect: str = "postgres"
    tables: list[str] = Field(default_factory=list)
    # Non-secret DSN parts so the console can show WHICH database is connected.
    host: str | None = None
    port: int | None = None
    database: str | None = None


class SqlConnectionItem(BaseModel):
    """Non-secret view of one registered connection, for the connection list/picker."""

    conn_id: str
    name: str
    dialect: str
    tables: list[str]
    is_default: bool
    host: str | None = None
    port: int | None = None
    database: str | None = None


class SqlAnswer(BaseModel):
    question: str | None = None
    connection_id: str | None = None
    connection_name: str | None = None
    sql: str = Field(description="The validated, bounded SQL that actually ran.")
    generated_sql: str | None = Field(default=None, description="The model's raw SQL (NL→SQL path).")
    columns: list[str] = Field(default_factory=list)
    rows: list[list[object]] = Field(default_factory=list)
    row_count: int = 0


def _dialect_from_dsn(dsn: str) -> str | None:
    """Derive the SQL dialect from the DSN scheme, so a wrong manual pick can't break
    prompting/parsing (e.g. MySQL selected for a postgresql:// DSN)."""
    scheme = urlsplit(dsn).scheme.split("+")[0].lower()
    if scheme.startswith("postgres"):
        return "postgres"
    if scheme.startswith("mysql"):
        return "mysql"
    if scheme.startswith("sqlite"):
        return "sqlite"
    # SQL Server and Oracle were missing, so their DSNs fell through to None and the connection
    # was stored as postgres. The platform then generated Postgres SQL for a SQL Server, and the
    # Postgres row-cap wrap (a derived table) was rejected outright — MSSQL forbids ORDER BY inside
    # one (error 1033), which broke every "break down by" / "top N" question against it.
    if scheme.startswith(("mssql", "sqlserver")):
        return "tsql"
    if scheme.startswith("oracle"):
        return "oracle"
    return None


async def _dsn_parts(conn: StoredSqlConnection) -> tuple[str | None, int | None, str | None]:
    try:
        parts = urlsplit(await tenant_crypto.decrypt(conn.tenant_id, conn.dsn_ciphertext))
        return parts.hostname, parts.port, (parts.path.lstrip("/") or None)
    except Exception as exc:  # a corrupt DSN shouldn't break the page; it just shows less detail
        _log.warning("sql_dsn_parse_failed", kb_id=conn.tenant_id, conn_id=conn.conn_id, error=str(exc)[:120])
        return None, None, None


async def _item(conn: StoredSqlConnection) -> SqlConnectionItem:
    host, port, database = await _dsn_parts(conn)
    return SqlConnectionItem(
        conn_id=conn.conn_id,
        name=conn.name,
        dialect=conn.dialect,
        tables=conn.tables,
        is_default=conn.is_default,
        host=host,
        port=port,
        database=database,
    )


async def list_connections(kb_id: str) -> list[SqlConnectionItem]:
    return [await _item(c) for c in await wiring.sql_connection_store().list(kb_id)]


async def get_default_info(kb_id: str) -> SqlConnectionInfo:
    """The KB's default connection (no secret) — back-compat single-connection view."""
    conn = await wiring.sql_connection_store().get_default(kb_id)
    if conn is None:
        return SqlConnectionInfo(configured=False)
    host, port, database = await _dsn_parts(conn)
    return SqlConnectionInfo(
        configured=True,
        conn_id=conn.conn_id,
        name=conn.name,
        dialect=conn.dialect,
        tables=conn.tables,
        host=host,
        port=port,
        database=database,
    )


async def get_connection(kb_id: str) -> SqlConnectionInfo:
    """Back-compat alias for :func:`get_default_info` (the KB's DEFAULT connection view). Kept as a
    distinct, patchable name because phase-3 consumers — the unified-KB evidence span and
    graph_sync's connector introspection (and their tests) — call/monkeypatch it directly."""
    return await get_default_info(kb_id)


async def add_connection(
    kb_id: str, name: str, dsn: str, tables: list[str], dialect: str = "postgres", make_default: bool = True
) -> SqlConnectionItem:
    """Register a new read-only DB connection. DSN host must be egress-allowlisted. The
    FIRST connection on a KB always becomes the default, regardless of make_default."""
    if not tables:
        raise ValidationError("at least one allowlisted table is required")
    assert_dsn_sovereign(dsn)  # networked DB is external → sovereignty-gated
    # The DSN scheme is authoritative for the dialect — a manual pick can't override it
    # (a MySQL pick on a postgresql:// DSN used to mis-prompt the model and mis-parse SQL).
    dialect = _dialect_from_dsn(dsn) or dialect
    store = wiring.sql_connection_store()
    existing = await store.list(kb_id)
    conn = StoredSqlConnection(
        tenant_id=kb_id,
        conn_id=str(uuid.uuid4()),
        name=name,
        dsn_ciphertext=await tenant_crypto.encrypt(kb_id, dsn),
        tables=sorted({t.strip() for t in tables if t.strip()}),
        dialect=dialect,
        is_default=make_default or not existing,
    )
    await store.put(conn)
    _log.info("sql_connection_added", kb_id=kb_id, conn_id=conn.conn_id, tables=len(tables), dialect=dialect)
    return await _item(conn)


async def update_connection(
    kb_id: str, conn_id: str, name: str, dsn: str | None, tables: list[str], dialect: str = "postgres"
) -> SqlConnectionItem:
    """Update a connection's label/table allowlist, and optionally rotate its DSN."""
    store = wiring.sql_connection_store()
    existing = await store.get(kb_id, conn_id)
    if existing is None:
        raise NotFoundError("SQL connection not found", details={"conn_id": conn_id})
    if not tables:
        raise ValidationError("at least one allowlisted table is required")
    dsn_ciphertext = existing.dsn_ciphertext
    if dsn:
        assert_dsn_sovereign(dsn)
        dialect = _dialect_from_dsn(dsn) or dialect
        dsn_ciphertext = await tenant_crypto.encrypt(kb_id, dsn)
    else:
        dialect = existing.dialect
    updated = existing.model_copy(
        update={
            "name": name,
            "dsn_ciphertext": dsn_ciphertext,
            "tables": sorted({t.strip() for t in tables if t.strip()}),
            "dialect": dialect,
        }
    )
    await store.put(updated)
    _log.info("sql_connection_updated", kb_id=kb_id, conn_id=conn_id)
    return await _item(updated)


async def set_default_connection(kb_id: str, conn_id: str) -> SqlConnectionItem:
    store = wiring.sql_connection_store()
    conn = await store.get(kb_id, conn_id)
    if conn is None:
        raise NotFoundError("SQL connection not found", details={"conn_id": conn_id})
    await store.set_default(kb_id, conn_id)
    # Re-seed the knowledge graph with the connected sources' tables — additive + best-effort, so a
    # graph hiccup never blocks the default switch. Local import breaks the graph_sync<->sql_query cycle.
    try:
        from svc_control_plane import graph_sync  # noqa: PLC0415

        await graph_sync.sync_kb_sources(kb_id)
    except Exception as exc:
        _log.warning("sql_default_graph_sync_failed", kb_id=kb_id, error=str(exc)[:160])
    return await _item((await store.get(kb_id, conn_id)) or conn)


async def delete_connection(kb_id: str, conn_id: str) -> None:
    await wiring.sql_connection_store().delete(kb_id, conn_id)
    _log.info("sql_connection_deleted", kb_id=kb_id, conn_id=conn_id)


async def _resolve(kb_id: str, connection_id: str | None) -> StoredSqlConnection:
    store = wiring.sql_connection_store()
    conn = await store.get(kb_id, connection_id) if connection_id else await store.get_default(kb_id)
    if conn is None:
        # Client-actionable resource state (register/pick a connection), not a server
        # misconfiguration — a 404, like model-connection's "model connection not found".
        detail = (
            "no SQL connection is registered for this knowledge base"
            if not connection_id
            else ("SQL connection not found")
        )
        raise NotFoundError(detail, details={"conn_id": connection_id})
    return conn


async def _connector(kb_id: str, connection_id: str | None) -> tuple[SqlConnector, StoredSqlConnection]:
    if not get_settings().nl2sql_enabled:
        raise ConfigError("NL→SQL is not enabled (RAG_NL2SQL_ENABLED)")
    stored = await _resolve(kb_id, connection_id)
    dsn = await tenant_crypto.decrypt(kb_id, stored.dsn_ciphertext)
    try:
        conn = SqlAlchemyConnector(
            dsn, set(stored.tables), statement_timeout_ms=get_settings().nl2sql_statement_timeout_ms
        )
    except ModuleNotFoundError as exc:
        # The dialect's DBAPI driver isn't installed — a clean, actionable 422 instead of a 500.
        raise ValidationError(
            f"no database driver is available for the '{stored.dialect}' connection "
            "(supported: postgresql, mysql/mariadb, mssql, oracle, sqlite)",
            details={"dialect": stored.dialect, "missing_module": str(exc)},
        ) from exc
    except Exception as exc:
        raise ValidationError(
            f"could not open the database connection: {str(exc).splitlines()[0][:200]}",
            details={"dialect": stored.dialect},
        ) from exc
    return conn, stored


async def _run_governed(
    kb_id: str, raw_sql: str, stored: StoredSqlConnection, conn: SqlConnector, *, actor: str, question: str | None
) -> SqlAnswer:
    s = get_settings()
    safe_sql = nl2sql.validate_sql(
        raw_sql, allowed_tables=set(stored.tables), dialect=stored.dialect, max_rows=s.nl2sql_max_rows
    )
    try:
        cols, rows = await conn.run(safe_sql, max_rows=s.nl2sql_max_rows)
    except Exception as exc:
        # The generated SQL is syntactically valid but the DATABASE rejected it (a bad enum value,
        # a type mismatch, an unknown column…). That is a bad model query, not a server fault —
        # surface it as a clean 422 with the DB's own message instead of a 500.
        detail = str(getattr(exc, "orig", exc)).splitlines()[0][:200]
        raise ValidationError(f"the generated SQL could not run against the database: {detail}") from exc
    await wiring.audit_service().record(
        tenant_id=kb_id,
        trace_id=str(uuid.uuid4()),
        action=AuditAction.SQL_QUERY,
        actor=actor,
        payload={
            "connection_id": stored.conn_id,
            "connection_name": stored.name,
            "question": question,
            "generated_sql": raw_sql,
            "sql": safe_sql,
            "row_count": len(rows),
        },
    )
    _log.info("sql_query", kb_id=kb_id, conn_id=stored.conn_id, rows=len(rows))
    return SqlAnswer(
        question=question,
        connection_id=stored.conn_id,
        connection_name=stored.name,
        sql=safe_sql,
        generated_sql=raw_sql if question is not None else None,
        columns=cols,
        rows=rows,
        row_count=len(rows),
    )


async def execute_sql(kb_id: str, raw_sql: str, *, connection_id: str | None = None, actor: str) -> SqlAnswer:
    """Run an analyst-supplied SQL through the governance layer (validated + bounded + audited)."""
    conn, stored = await _connector(kb_id, connection_id)
    try:
        return await _run_governed(kb_id, raw_sql, stored, conn, actor=actor, question=None)
    finally:
        await conn.aclose()


async def ask(kb_id: str, question: str, *, connection_id: str | None = None, actor: str) -> SqlAnswer:
    """Full NL→SQL: introspect schema → tenant model writes SQL → govern → run → audit."""
    conn, stored = await _connector(kb_id, connection_id)
    try:
        try:
            schema = await conn.schema()
        except Exception as exc:
            detail = str(getattr(exc, "orig", exc)).splitlines()[0][:200]
            raise ValidationError(
                f"could not connect to the database: {detail}", details={"dialect": stored.dialect}
            ) from exc
        if not schema:
            raise ConfigError("none of the allowlisted tables exist in the database")
        # Ground the prompt in the REAL values of categorical columns so the model filters with them
        # (e.g. stage IN ('Won','Lost')) instead of guessing — best-effort, never blocks the answer.
        samples: dict[str, dict[str, list[str]]] = {}
        try:
            samples = await conn.column_samples(schema)
        except Exception as exc:
            _log.warning("nl2sql_column_sampling_failed", kb_id=kb_id, error=str(exc)[:160])
        gateway = await wiring.resolve_tenant_gateway(kb_id)
        prompt = nl2sql.build_sql_prompt(question, schema, dialect=stored.dialect, samples=samples)
        # A REASONING model spends most of its budget inside <think>…</think> before the SQL, and
        # sometimes returns NO final answer (content=null) or a transient endpoint error — a
        # STOCHASTIC miss, not a deterministic one. Retry a few times so a single flaky generation
        # doesn't surface to the user as "empty SQL".
        max_tokens = get_settings().nl2sql_max_tokens
        raw = ""
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                raw = nl2sql.extract_sql(await gateway.complete(prompt, max_tokens=max_tokens))
                if raw.strip():
                    break
                _log.warning("nl2sql_empty_generation", attempt=attempt, kb_id=kb_id)
            except Exception as exc:
                last_error = exc
                _log.warning("nl2sql_generation_error", attempt=attempt, error=str(exc)[:160])
        if not raw.strip():
            if last_error is not None:
                raise last_error
            raise ValidationError(
                "the SQL model returned no query after several attempts — rephrase the question, "
                "or narrow the allowlisted tables"
            )
        return await _run_governed(kb_id, raw, stored, conn, actor=actor, question=question)
    finally:
        await conn.aclose()


# --- Unified KB: structured DB result as a retrieved evidence span --------------------


def render_sql_evidence(answer: SqlAnswer, *, max_rows: int = 20) -> str:
    """Render a SqlAnswer as compact, model-readable evidence text (a bounded markdown table +
    the exact SQL that ran, for provenance). Truncated at ``max_rows`` so a wide result never
    blows the prompt."""
    lines = [f"Structured database result — {answer.row_count} row(s) from SQL: {answer.sql}", ""]
    if answer.columns:
        lines.append(" | ".join(answer.columns))
        lines.append(" | ".join("---" for _ in answer.columns))
    for row in answer.rows[:max_rows]:
        lines.append(" | ".join("" if v is None else str(v) for v in row))
    if answer.row_count > max_rows:
        lines.append(f"... ({answer.row_count - max_rows} more row(s))")
    return "\n".join(lines)


async def sql_evidence_span(kb_id: str, question: str, *, actor: str) -> list[RetrievedSpan]:
    """One authoritative evidence span PER connected DB, for UNIFIED (documents + database) answers.

    Runs NL→SQL for ``question`` against EVERY SQL connection on the KB (a KB may hold several —
    FR-D10) and renders each connection's rows as its own cited span the generator grounds on
    alongside document spans. A connection that errors, has NL→SQL disabled, or returns no rows is
    skipped; an EMPTY list means no connected DB answered and the caller falls back to
    documents-only. Each span is scored 1.0 / ``sql`` (exact, authoritative) and badged
    ``Modality.DATABASE`` so its citation is flagged as a database source."""
    spans: list[RetrievedSpan] = []
    for conn in await list_connections(kb_id):
        try:
            answer = await ask(kb_id, question, connection_id=conn.conn_id, actor=actor)
        except Exception as exc:
            _log.info("sql_evidence_skipped", kb_id=kb_id, conn_id=conn.conn_id, error=str(exc)[:160])
            continue
        if answer.row_count == 0:
            continue
        label = conn.name or conn.conn_id
        spans.append(
            RetrievedSpan(
                chunk_id=f"sql::{kb_id}::{conn.conn_id}",
                doc_id=f"db://{kb_id}/{label}",
                text=f"[{label}] {render_sql_evidence(answer)}",
                score=1.0,
                scorer="sql",
                modality=Modality.DATABASE,
                source_uri=f"sql://{kb_id}/{conn.conn_id}",
                category="database",
            )
        )
    return spans
