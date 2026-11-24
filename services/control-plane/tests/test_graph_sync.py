"""Connection-time graph sync — a connected source's declared schema seeds the KB graph up front."""

from __future__ import annotations

import pytest
from svc_control_plane import graph_bridge, graph_sync
from svc_control_plane.nosql_query import NoSqlConnectionInfo
from svc_control_plane.rest_source import RestConnectionInfo
from svc_control_plane.sharepoint_source import SharePointConnectionInfo
from svc_control_plane.sql_query import SqlConnectionItem

from rag_clients.graph_store import InMemoryGraphStore
from rag_common import get_settings
from rag_core.graph import cooccurrence_edges


async def test_sync_entities_puts_schema_in_contact_with_docs() -> None:
    store = InMemoryGraphStore()
    await store.ensure("k")
    await store.add_chunk("k", "doc::1", ["Orders", "Report"], cooccurrence_edges(["Orders", "Report"]))
    n = await graph_bridge.sync_entities_to_graph("k", "sql-schema::k", ["Orders", "Customers"], store=store)
    assert n == 2
    # Up-front contact: the SQL table 'Orders' shares the Orders node with the document — BEFORE any query.
    reached = await store.linked_chunks("k", ["Orders"], hops=1)
    assert "doc::1" in reached
    assert "sql-schema::k" in reached


async def test_sync_entities_empty_is_zero() -> None:
    store = InMemoryGraphStore()
    await store.ensure("k")
    assert await graph_bridge.sync_entities_to_graph("k", "c", ["", "  "], store=store) == 0


async def test_source_schema_entities_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _sql_list(_kb: str) -> list[SqlConnectionItem]:
        return [
            SqlConnectionItem(
                conn_id="c1", name="db", dialect="sqlite", tables=["orders", "customers"], is_default=True
            )
        ]

    async def _nosql(_kb: str) -> NoSqlConnectionInfo:
        return NoSqlConnectionInfo(configured=True, collections=["events"])

    async def _rest(_kb: str) -> RestConnectionInfo:
        return RestConnectionInfo(configured=True, base_url="https://api.acme.local/search")

    async def _sp(_kb: str) -> SharePointConnectionInfo:
        return SharePointConnectionInfo(configured=False)

    monkeypatch.setattr(graph_sync.sql_query, "list_connections", _sql_list)  # type: ignore[attr-defined]
    monkeypatch.setattr(graph_sync.nosql_query, "get_nosql_connection", _nosql)  # type: ignore[attr-defined]
    monkeypatch.setattr(graph_sync.rest_source, "get_rest_connection", _rest)  # type: ignore[attr-defined]
    monkeypatch.setattr(graph_sync.sharepoint_source, "get_sharepoint_connection", _sp)  # type: ignore[attr-defined]
    ents = await graph_sync.source_schema_entities("k")
    assert ents == {"sql": ["orders", "customers"], "nosql": ["events"], "rest": ["api.acme.local"]}


async def test_sync_kb_sources_noop_when_graphrag_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_GRAPHRAG_ENABLED", "false")
    get_settings.cache_clear()
    assert await graph_sync.sync_kb_sources("k") == {}
    get_settings.cache_clear()


async def test_sync_kb_sources_populates_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_GRAPHRAG_ENABLED", "true")
    get_settings.cache_clear()
    store = InMemoryGraphStore()
    monkeypatch.setattr(graph_sync, "get_graph_store", lambda: store)

    async def _sql_list(_kb: str) -> list[SqlConnectionItem]:
        return [SqlConnectionItem(conn_id="c1", name="db", dialect="sqlite", tables=["orders"], is_default=True)]

    async def _off_nosql(_kb: str) -> NoSqlConnectionInfo:
        return NoSqlConnectionInfo(configured=False)

    async def _off_rest(_kb: str) -> RestConnectionInfo:
        return RestConnectionInfo(configured=False)

    async def _off_sp(_kb: str) -> SharePointConnectionInfo:
        return SharePointConnectionInfo(configured=False)

    monkeypatch.setattr(graph_sync.sql_query, "list_connections", _sql_list)  # type: ignore[attr-defined]
    monkeypatch.setattr(graph_sync.nosql_query, "get_nosql_connection", _off_nosql)  # type: ignore[attr-defined]
    monkeypatch.setattr(graph_sync.rest_source, "get_rest_connection", _off_rest)  # type: ignore[attr-defined]
    monkeypatch.setattr(graph_sync.sharepoint_source, "get_sharepoint_connection", _off_sp)  # type: ignore[attr-defined]
    result = await graph_sync.sync_kb_sources("bank__deals")
    assert result == {"sql": 1}
    reached = await store.linked_chunks("bank__deals", ["orders"], hops=1)
    assert "sql-schema::bank__deals" in reached
    get_settings.cache_clear()
