"""Connection-time graph sync — seed a KB graph with its connected sources' structural entities.

The graph bridge (:mod:`graph_bridge`) links a source's live ANSWER entities into the knowledge
graph at query time. This is the complementary CONNECTION-time half: it reads each connected
source's structure — SQL table names and NoSQL collection names straight from the stored
connection config (no live query), and for REST / SharePoint a bounded, read-only introspection
of the live source (its OpenAPI paths / its non-hidden lists, falling back to just the host when
introspection finds nothing) — and upserts them as graph nodes. So the moment a source is
connected, a table named ``orders`` already shares a node with any document that mentions orders:
the sources are in contact in GraphRAG up front, not only after a question happens to touch them.

Additive + fail-safe: a source that can't be read is skipped; the graph is never a hard dependency.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from rag_clients.factory import get_graph_store
from rag_common import get_logger, get_settings
from svc_control_plane import graph_bridge, nosql_query, rest_source, sharepoint_source, sql_query

_log = get_logger(__name__)


async def source_schema_entities(kb_id: str) -> dict[str, list[str]]:
    """The structural entities of every source connected to ``kb_id``: SQL table names and NoSQL
    collection names straight from the stored config (no live read), and for REST / SharePoint a
    live, read-only introspection of the source (its OpenAPI spec / its non-hidden lists — see
    :func:`rest_source.introspect_entities` and :func:`sharepoint_source.introspect_entities`)
    when that succeeds, falling back to the host alone when it doesn't (source unreachable, no
    spec published, introspection erred — introspection is itself fail-safe and never raises, but
    the call is wrapped here too as defence in depth)."""
    out: dict[str, list[str]] = {}
    # Every SQL connection on the KB (not just the default) — a KB can wire several databases and
    # each one's tables belong in the graph. De-duped, order-preserving across connections.
    sql_tables: list[str] = []
    seen: set[str] = set()
    for c in await sql_query.list_connections(kb_id):
        for t in c.tables:
            if t not in seen:
                seen.add(t)
                sql_tables.append(t)
    if sql_tables:
        out["sql"] = sql_tables
    nosql = await nosql_query.get_nosql_connection(kb_id)
    if nosql.configured and nosql.collections:
        out["nosql"] = list(nosql.collections)
    rest = await rest_source.get_rest_connection(kb_id)
    if rest.configured and rest.base_url:
        host_entity = [urlsplit(rest.base_url).hostname or rest.base_url]
        try:
            entities = await rest_source.introspect_entities(kb_id)
        except Exception as exc:
            _log.warning("graph_sync_rest_introspect_failed", kb_id=kb_id, error=str(exc)[:160])
            entities = []
        out["rest"] = entities or host_entity
    sharepoint = await sharepoint_source.get_sharepoint_connection(kb_id)
    if sharepoint.configured and sharepoint.site_url:
        host_entity = [urlsplit(sharepoint.site_url).hostname or sharepoint.site_url]
        try:
            entities = await sharepoint_source.introspect_entities(kb_id)
        except Exception as exc:
            _log.warning("graph_sync_sharepoint_introspect_failed", kb_id=kb_id, error=str(exc)[:160])
            entities = []
        out["sharepoint"] = entities or host_entity
    return out


async def sync_kb_sources(kb_id: str) -> dict[str, int]:
    """Upsert every connected source's structural entities into the KB graph. No-op (empty result)
    unless ``graphrag_enabled``. Returns ``{kind: entities_synced}``; a source that fails is skipped,
    never blocking the others."""
    if not get_settings().graphrag_enabled:
        return {}
    store = get_graph_store()
    await store.ensure(kb_id)
    result: dict[str, int] = {}
    for kind, names in (await source_schema_entities(kb_id)).items():
        try:
            synced = await graph_bridge.sync_entities_to_graph(kb_id, f"{kind}-schema::{kb_id}", names, store=store)
        except Exception as exc:
            _log.warning("graph_sync_source_failed", kb_id=kb_id, kind=kind, error=str(exc)[:160])
            continue
        if synced:
            result[kind] = synced
    return result
