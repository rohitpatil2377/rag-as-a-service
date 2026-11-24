"""Governed NoSQL (MongoDB) connector — the unified KB's second evidence producer (FR-D10 twin).

Same shape as :mod:`sql_query`, for a per-KB read-only MongoDB connection instead of a
relational one: a connection URI + a collection allowlist, envelope-encrypted in
``tenant_config`` via :mod:`tenant_crypto`, with the host(s) egress-allowlisted via
:func:`rag_common.sovereignty.assert_sovereign`. ``nosql_evidence_span`` is the producer the
unified KB (:mod:`unified`) calls alongside :func:`svc_control_plane.sql_query.sql_evidence_span`:
given a KB id and a question it returns ONE authoritative :class:`RetrievedSpan` or ``None``, so
a bad or unconfigured NoSQL source never fails the query.

No NL→query translation here (that is a SQL-only path with sqlglot governance behind it) — the
find is a BOUNDED, read-only keyword match over the allowlisted collections, which sidesteps
NoSQL query-injection entirely because the question text never becomes operator syntax.

The Mongo driver (``motor``) is imported lazily, INSIDE the default client's ``find``, so this
module — and the rest of the control plane — imports fine on a host where the driver isn't
installed; only a KB that actually registers a NoSQL connection ever needs it. Tests inject a
fake :class:`NoSqlClient` through the ``_client_factory`` seam and never touch the driver.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from rag_clients.factory import get_catalog
from rag_common import get_logger
from rag_common.errors import ValidationError
from rag_common.sovereignty import assert_sovereign
from rag_schemas import RetrievedSpan
from rag_schemas.common import Modality
from svc_control_plane import tenant_crypto

_log = get_logger(__name__)

_URI_KEY = "nosql_uri"  # envelope-encrypted Mongo URI in tenant_config
_DB_KEY = "nosql_db"
_COLLECTIONS_KEY = "nosql_collections"  # comma-separated allowlist

_RESULT_LIMIT = 20  # hard bound on documents unioned across all allowlisted collections


class NoSqlConnectionInfo(BaseModel):
    """Non-secret view of a KB's registered NoSQL connection."""

    configured: bool
    database: str = ""
    collections: list[str] = Field(default_factory=list)
    # Non-secret host (never the full URI/creds) so the console can show WHICH store is connected.
    host: str | None = None


def _hosts(uri: str) -> tuple[str, set[str]]:
    """Scheme (mongodb/mongodb+srv normalised to its base) and every host in the URI's
    authority — a replica-set URI can list more than one, and each one is an external
    dependency that must be egress-allowlisted."""
    parts = urlsplit(uri)
    scheme = parts.scheme.split("+")[0] or "mongodb"
    authority = parts.netloc.rsplit("@", 1)[-1]  # drop userinfo, keep host[:port][,host[:port]...]
    hosts = {h.split(":")[0] for h in authority.split(",") if h.split(":")[0]}
    return scheme, hosts


def assert_nosql_uri_sovereign(uri: str) -> None:
    """A networked MongoDB is an external dependency → every host in the URI must be
    egress-allowlisted."""
    scheme, hosts = _hosts(uri)
    for host in hosts:
        assert_sovereign(f"{scheme}://{host}")


async def _sync_graph_best_effort(kb_id: str) -> None:
    """Seed the knowledge graph with this KB's connected sources' structural entities the moment a
    source is registered — additive + best-effort, so a graph hiccup never blocks the connection.
    Local import breaks the graph_sync<->source-module cycle."""
    try:
        from svc_control_plane import graph_sync  # noqa: PLC0415

        await graph_sync.sync_kb_sources(kb_id)
    except Exception as exc:
        _log.warning("source_graph_sync_failed", kb_id=kb_id, error=str(exc)[:160])


async def set_nosql_connection(kb_id: str, uri: str, database: str, collections: list[str]) -> NoSqlConnectionInfo:
    """Register/replace a KB's read-only MongoDB connection. URI host(s) must be
    egress-allowlisted; at least one allowlisted collection is required."""
    if not collections:
        raise ValidationError("at least one allowlisted collection is required")
    assert_nosql_uri_sovereign(uri)
    catalog = get_catalog()
    await catalog.set_tenant_config(kb_id, _URI_KEY, await tenant_crypto.encrypt(kb_id, uri))
    await catalog.set_tenant_config(kb_id, _DB_KEY, database)
    await catalog.set_tenant_config(
        kb_id, _COLLECTIONS_KEY, ",".join(sorted({c.strip() for c in collections if c.strip()}))
    )
    _log.info("nosql_connection_registered", kb_id=kb_id, database=database, collections=len(collections))
    await _sync_graph_best_effort(kb_id)
    return await get_nosql_connection(kb_id)


async def clear_nosql_connection(kb_id: str) -> None:
    """Disconnect: remove the KB's registered NoSQL connection (URI, database, allowlist)."""
    catalog = get_catalog()
    for key in (_URI_KEY, _DB_KEY, _COLLECTIONS_KEY):
        await catalog.set_tenant_config(kb_id, key, "")
    _log.info("nosql_connection_cleared", kb_id=kb_id)


async def get_nosql_connection(kb_id: str) -> NoSqlConnectionInfo:
    cfg = await get_catalog().get_tenant_config(kb_id)
    uri_ct = cfg.get(_URI_KEY)
    if not uri_ct:
        return NoSqlConnectionInfo(configured=False)
    host: str | None = None
    try:
        _, hosts = _hosts(await tenant_crypto.decrypt(kb_id, uri_ct))
        host = sorted(hosts)[0] if hosts else None
    except Exception as exc:  # a corrupt/undecryptable URI shouldn't break the page
        _log.warning("nosql_uri_parse_failed", kb_id=kb_id, error=str(exc)[:120])
    return NoSqlConnectionInfo(
        configured=True,
        database=cfg.get(_DB_KEY, ""),
        collections=sorted(c for c in cfg.get(_COLLECTIONS_KEY, "").split(",") if c),
        host=host,
    )


# --- Driver seam ------------------------------------------------------------------------------


class NoSqlClient(Protocol):
    """What the query path needs from a Mongo client. Small and read-only on purpose — tests
    implement this directly with an in-memory fake, no driver required."""

    async def find(self, database: str, collection: str, terms: list[str], limit: int) -> list[dict[str, Any]]: ...


class _MotorClient:
    """Default motor-backed :class:`NoSqlClient`. The driver import lives INSIDE ``find`` (not at
    module scope) so importing this module never requires motor to be installed — only a KB with
    an actual NoSQL connection reaches this line."""

    def __init__(self, uri: str) -> None:
        self._uri = uri

    async def find(self, database: str, collection: str, terms: list[str], limit: int) -> list[dict[str, Any]]:
        from motor.motor_asyncio import AsyncIOMotorClient  # type: ignore[import-not-found]  # noqa: PLC0415

        client: Any = AsyncIOMotorClient(self._uri)
        try:
            coll = client[database][collection]
            # Bounded, read-only KEYWORD match — never an NL->query translation, so the
            # question text can never become Mongo operator syntax (no injection surface).
            query: dict[str, Any] = {"$text": {"$search": " ".join(terms)}} if terms else {}
            cursor = coll.find(query).limit(limit)
            return [doc async for doc in cursor]
        finally:
            client.close()


# Seam: production wiring can swap in a pooled/pre-authenticated client, and tests inject a fake
# NoSqlClient here so the driver never needs to be installed to exercise this module.
_client_factory: Callable[[str], NoSqlClient] = _MotorClient


# --- Unified KB: NoSQL result as a retrieved evidence span ------------------------------------


def render_nosql_evidence(docs: list[dict[str, Any]], *, max_docs: int = 20) -> str:
    """Render matched Mongo documents as compact, model-readable evidence text (bounded at
    ``max_docs`` so a wide result never blows the prompt)."""
    lines = [f"Structured NoSQL result — {len(docs)} document(s)", ""]
    for doc in docs[:max_docs]:
        lines.append(json.dumps(doc, default=str, ensure_ascii=False, sort_keys=True))
    if len(docs) > max_docs:
        lines.append(f"... ({len(docs) - max_docs} more document(s))")
    return "\n".join(lines)


async def nosql_evidence_span(kb_id: str, question: str, *, actor: str) -> RetrievedSpan | None:
    """A retrieved-span view of the KB's connected MongoDB store, for UNIFIED (documents + NoSQL)
    answers. Runs a BOUNDED, read-only keyword ``find`` (the question's terms — never an
    arbitrary NL->query, so the question text can't become query syntax) over each allowlisted
    collection, unions up to ``_RESULT_LIMIT`` documents, and renders them as ONE authoritative
    evidence span. FAIL-SAFE BY CONSTRUCTION: returns ``None`` when the KB has no NoSQL
    connection, the driver/network/query errors for ANY reason, or nothing matches — in every
    case the caller simply falls back to a documents-only answer, and a bad source can never fail
    the query. ``actor`` is accepted for shape-symmetry with the other evidence producers (e.g.
    :func:`svc_control_plane.sql_query.sql_evidence_span`); this connector issues no audit call
    of its own — the parent wires that. The span is scored 1.0 / ``exact`` (authoritative) and
    badged ``Modality.DATABASE`` so its citation is flagged as a database source."""
    del actor  # unused here: no audit call in this module (parent wires it); kept for shape parity
    try:
        info = await get_nosql_connection(kb_id)
        if not info.configured or not info.collections:
            return None
        cfg = await get_catalog().get_tenant_config(kb_id)
        uri_ct = cfg.get(_URI_KEY)
        if not uri_ct:
            return None
        uri = await tenant_crypto.decrypt(kb_id, uri_ct)
        terms = [t for t in question.split() if t]
        client = _client_factory(uri)
        docs: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for collection in info.collections:
            remaining = _RESULT_LIMIT - len(docs)
            if remaining <= 0:
                break
            for doc in await client.find(info.database, collection, terms, remaining):
                key = str(doc.get("_id", id(doc)))
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                docs.append(doc)
        if not docs:
            return None
        return RetrievedSpan(
            chunk_id=f"nosql::{kb_id}",
            doc_id=f"nosql://{kb_id}",
            text=render_nosql_evidence(docs, max_docs=_RESULT_LIMIT),
            score=1.0,
            scorer="exact",
            modality=Modality.DATABASE,
            source_uri=f"nosql://{kb_id}",
            category="database",
        )
    except Exception as exc:  # a bad NoSQL source must never fail the query
        _log.info("nosql_evidence_skipped", kb_id=kb_id, error=str(exc)[:160])
        return None
