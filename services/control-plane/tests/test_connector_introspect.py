"""Connection-time introspection — REST/SharePoint entities discovered by a live, read-only probe
of the source, layered on top of the config-only entities the connectors already exposed.

Mirrors ``test_rest_source.py``/``test_sharepoint_source.py``'s style: every HTTP call goes
through the connector's ``_client_factory`` seam (monkeypatched to an ``httpx.MockTransport``),
so no test here ever touches the network. Covers ``rest_source.introspect_entities`` against a
sample OpenAPI spec, ``sharepoint_source.introspect_entities`` against a sample lists response,
the fail-safe contract (non-200 / malformed / not-connected -> ``[]``) for both, and that
``graph_sync.source_schema_entities`` prefers introspected names when present but falls back to
the host-only entity when introspection returns ``[]``.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from svc_control_plane import graph_sync, rest_source, sharepoint_source
from svc_control_plane.nosql_query import NoSqlConnectionInfo
from svc_control_plane.rest_source import RestConnectionInfo
from svc_control_plane.sharepoint_source import SharePointConnectionInfo
from svc_control_plane.sql_query import SqlConnectionInfo

_KB = "acme__eng"
_REST_URL = "http://localhost:9200"
_SITE = "https://localhost/sites/eng"  # "localhost" is in the default egress allowlist


def _mock_client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _openapi_spec() -> dict[str, Any]:
    """A minimal but realistic OpenAPI document: two paths, one with an operationId + tags, one
    identified only by its path segment."""
    return {
        "openapi": "3.0.0",
        "paths": {
            "/orders": {
                "get": {"operationId": "listOrders", "tags": ["Orders"]},
                "post": {"operationId": "createOrder", "tags": ["Orders"]},
            },
            "/customers/{id}": {
                "get": {"tags": ["Customers"]},
            },
        },
    }


def _lists_json() -> dict[str, Any]:
    """A realistic ``/_api/web/lists`` (odata=nometadata) payload: two visible lists, one hidden."""
    return {
        "value": [
            {"Title": "Documents", "Hidden": False},
            {"Title": "Site Pages", "Hidden": False},
            {"Title": "Workflow History", "Hidden": True},
        ]
    }


# --- rest_source.introspect_entities: happy path ---------------------------------------------


async def test_rest_introspect_parses_openapi_spec(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=_openapi_spec())
        return httpx.Response(404)

    await rest_source.set_rest_connection(_KB, _REST_URL, token="tok")
    monkeypatch.setattr(rest_source, "_client_factory", lambda: _mock_client(handler))

    entities = await rest_source.introspect_entities(_KB)
    assert entities == ["listOrders", "Orders", "createOrder", "customers", "Customers"]


async def test_rest_introspect_tries_fallback_spec_paths(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """``/openapi.json`` 404s; the next conventional path (``/swagger.json``) is tried next."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/swagger.json":
            return httpx.Response(200, json=_openapi_spec())
        return httpx.Response(404)

    await rest_source.set_rest_connection(_KB, _REST_URL)
    monkeypatch.setattr(rest_source, "_client_factory", lambda: _mock_client(handler))

    entities = await rest_source.introspect_entities(_KB)
    assert entities  # found on the second attempt
    assert seen == ["/openapi.json", "/swagger.json"]


# --- rest_source.introspect_entities: fail-safe -------------------------------------------------


async def test_rest_introspect_not_connected_returns_empty(inmemory: None) -> None:
    assert await rest_source.introspect_entities("unconfigured-kb") == []


async def test_rest_introspect_no_spec_found_returns_empty(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    await rest_source.set_rest_connection(_KB, _REST_URL)
    monkeypatch.setattr(rest_source, "_client_factory", lambda: _mock_client(handler))
    assert await rest_source.introspect_entities(_KB) == []


async def test_rest_introspect_malformed_spec_returns_empty(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all", headers={"content-type": "text/plain"})

    await rest_source.set_rest_connection(_KB, _REST_URL)
    monkeypatch.setattr(rest_source, "_client_factory", lambda: _mock_client(handler))
    assert await rest_source.introspect_entities(_KB) == []


async def test_rest_introspect_spec_without_paths_returns_empty(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"openapi": "3.0.0"})

    await rest_source.set_rest_connection(_KB, _REST_URL)
    monkeypatch.setattr(rest_source, "_client_factory", lambda: _mock_client(handler))
    assert await rest_source.introspect_entities(_KB) == []


async def test_rest_introspect_transport_error_returns_empty(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    await rest_source.set_rest_connection(_KB, _REST_URL)
    monkeypatch.setattr(rest_source, "_client_factory", lambda: _mock_client(handler))
    assert await rest_source.introspect_entities(_KB) == []


# --- sharepoint_source.introspect_entities: happy path ------------------------------------------


async def test_sharepoint_introspect_returns_non_hidden_titles(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        assert request.url.path.endswith("/_api/web/lists")
        return httpx.Response(200, json=_lists_json())

    await sharepoint_source.set_sharepoint_connection(_KB, _SITE, token="tok")
    monkeypatch.setattr(sharepoint_source, "_client_factory", lambda: _mock_client(handler))

    entities = await sharepoint_source.introspect_entities(_KB)
    assert entities == ["Documents", "Site Pages"]  # hidden list excluded
    assert seen["headers"]["authorization"] == "Bearer tok"
    assert seen["headers"]["accept"] == "application/json;odata=nometadata"
    assert "$select=Title" in seen["url"] or "%24select=Title" in seen["url"]


async def test_sharepoint_introspect_handles_results_wrapper(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Some SharePoint deployments nest the collection under ``d.results`` (verbose OData)."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"d": {"results": _lists_json()["value"]}})

    await sharepoint_source.set_sharepoint_connection(_KB, _SITE)
    monkeypatch.setattr(sharepoint_source, "_client_factory", lambda: _mock_client(handler))

    entities = await sharepoint_source.introspect_entities(_KB)
    assert entities == ["Documents", "Site Pages"]


# --- sharepoint_source.introspect_entities: fail-safe --------------------------------------------


async def test_sharepoint_introspect_not_connected_returns_empty(inmemory: None) -> None:
    assert await sharepoint_source.introspect_entities("unconfigured-kb") == []


async def test_sharepoint_introspect_non_200_returns_empty(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    await sharepoint_source.set_sharepoint_connection(_KB, _SITE, token="tok")
    monkeypatch.setattr(sharepoint_source, "_client_factory", lambda: _mock_client(handler))
    assert await sharepoint_source.introspect_entities(_KB) == []


async def test_sharepoint_introspect_malformed_json_returns_empty(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json", headers={"content-type": "text/plain"})

    await sharepoint_source.set_sharepoint_connection(_KB, _SITE, token="tok")
    monkeypatch.setattr(sharepoint_source, "_client_factory", lambda: _mock_client(handler))
    assert await sharepoint_source.introspect_entities(_KB) == []


async def test_sharepoint_introspect_unrecognised_shape_returns_empty(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    await sharepoint_source.set_sharepoint_connection(_KB, _SITE, token="tok")
    monkeypatch.setattr(sharepoint_source, "_client_factory", lambda: _mock_client(handler))
    assert await sharepoint_source.introspect_entities(_KB) == []


async def test_sharepoint_introspect_transport_error_returns_empty(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    await sharepoint_source.set_sharepoint_connection(_KB, _SITE, token="tok")
    monkeypatch.setattr(sharepoint_source, "_client_factory", lambda: _mock_client(handler))
    assert await sharepoint_source.introspect_entities(_KB) == []


# --- graph_sync.source_schema_entities: prefer-introspect-else-host -----------------------------


async def test_graph_sync_uses_introspected_names_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _sql(_kb: str) -> SqlConnectionInfo:
        return SqlConnectionInfo(configured=False)

    async def _nosql(_kb: str) -> NoSqlConnectionInfo:
        return NoSqlConnectionInfo(configured=False)

    async def _rest(_kb: str) -> RestConnectionInfo:
        return RestConnectionInfo(configured=True, base_url="https://api.acme.local/search")

    async def _sp(_kb: str) -> SharePointConnectionInfo:
        return SharePointConnectionInfo(configured=True, site_url="https://acme.sharepoint.com/sites/eng")

    async def _rest_introspect(_kb: str) -> list[str]:
        return ["listOrders", "Orders"]

    async def _sp_introspect(_kb: str) -> list[str]:
        return ["Documents", "Site Pages"]

    monkeypatch.setattr(graph_sync.sql_query, "get_connection", _sql)  # type: ignore[attr-defined]
    monkeypatch.setattr(graph_sync.nosql_query, "get_nosql_connection", _nosql)  # type: ignore[attr-defined]
    monkeypatch.setattr(graph_sync.rest_source, "get_rest_connection", _rest)  # type: ignore[attr-defined]
    monkeypatch.setattr(graph_sync.sharepoint_source, "get_sharepoint_connection", _sp)  # type: ignore[attr-defined]
    monkeypatch.setattr(graph_sync.rest_source, "introspect_entities", _rest_introspect)  # type: ignore[attr-defined]
    monkeypatch.setattr(graph_sync.sharepoint_source, "introspect_entities", _sp_introspect)  # type: ignore[attr-defined]

    ents = await graph_sync.source_schema_entities("k")
    assert ents == {"rest": ["listOrders", "Orders"], "sharepoint": ["Documents", "Site Pages"]}


async def test_graph_sync_falls_back_to_host_when_introspection_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _sql(_kb: str) -> SqlConnectionInfo:
        return SqlConnectionInfo(configured=False)

    async def _nosql(_kb: str) -> NoSqlConnectionInfo:
        return NoSqlConnectionInfo(configured=False)

    async def _rest(_kb: str) -> RestConnectionInfo:
        return RestConnectionInfo(configured=True, base_url="https://api.acme.local/search")

    async def _sp(_kb: str) -> SharePointConnectionInfo:
        return SharePointConnectionInfo(configured=True, site_url="https://acme.sharepoint.com/sites/eng")

    async def _empty(_kb: str) -> list[str]:
        return []

    monkeypatch.setattr(graph_sync.sql_query, "get_connection", _sql)  # type: ignore[attr-defined]
    monkeypatch.setattr(graph_sync.nosql_query, "get_nosql_connection", _nosql)  # type: ignore[attr-defined]
    monkeypatch.setattr(graph_sync.rest_source, "get_rest_connection", _rest)  # type: ignore[attr-defined]
    monkeypatch.setattr(graph_sync.sharepoint_source, "get_sharepoint_connection", _sp)  # type: ignore[attr-defined]
    monkeypatch.setattr(graph_sync.rest_source, "introspect_entities", _empty)  # type: ignore[attr-defined]
    monkeypatch.setattr(graph_sync.sharepoint_source, "introspect_entities", _empty)  # type: ignore[attr-defined]

    ents = await graph_sync.source_schema_entities("k")
    assert ents == {"rest": ["api.acme.local"], "sharepoint": ["acme.sharepoint.com"]}


async def test_graph_sync_falls_back_to_host_when_introspection_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defence in depth: even if a future ``introspect_entities`` bug ever raised, graph_sync
    itself must not blow up the whole connection-time sync — it degrades to the host entity."""

    async def _sql(_kb: str) -> SqlConnectionInfo:
        return SqlConnectionInfo(configured=False)

    async def _nosql(_kb: str) -> NoSqlConnectionInfo:
        return NoSqlConnectionInfo(configured=False)

    async def _rest(_kb: str) -> RestConnectionInfo:
        return RestConnectionInfo(configured=True, base_url="https://api.acme.local/search")

    async def _sp(_kb: str) -> SharePointConnectionInfo:
        return SharePointConnectionInfo(configured=False)

    async def _boom(_kb: str) -> list[str]:
        raise RuntimeError("boom")

    monkeypatch.setattr(graph_sync.sql_query, "get_connection", _sql)  # type: ignore[attr-defined]
    monkeypatch.setattr(graph_sync.nosql_query, "get_nosql_connection", _nosql)  # type: ignore[attr-defined]
    monkeypatch.setattr(graph_sync.rest_source, "get_rest_connection", _rest)  # type: ignore[attr-defined]
    monkeypatch.setattr(graph_sync.sharepoint_source, "get_sharepoint_connection", _sp)  # type: ignore[attr-defined]
    monkeypatch.setattr(graph_sync.rest_source, "introspect_entities", _boom)  # type: ignore[attr-defined]

    ents = await graph_sync.source_schema_entities("k")
    assert ents == {"rest": ["api.acme.local"]}
