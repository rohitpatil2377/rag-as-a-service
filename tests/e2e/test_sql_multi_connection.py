"""E2E: a KB may register MANY read-only SQL connections (FR-D10 extended).

Query-time routing is EXPLICIT (connection_id), not auto-detected — this proves a KB
with two connections answers each question against the ONE named, never guessing or
merging across both, plus the default-fallback and CRUD lifecycle (add/list/rename/
rotate-DSN/set-default/delete). See test_nl2sql.py for single-connection governance
(injection blocking, table allowlist) — unchanged by this feature.
"""

from __future__ import annotations

import sqlite3

import httpx
import pytest

pytestmark = pytest.mark.e2e


def _seed(path: str, table: str, rows: str) -> None:
    con = sqlite3.connect(path)
    con.executescript(f"CREATE TABLE {table} (id INTEGER, name TEXT, amount INTEGER); {rows}")
    con.commit()
    con.close()


async def _client(tenant: str, monkeypatch: pytest.MonkeyPatch) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_NL2SQL_ENABLED", "true")
    get_settings.cache_clear()
    app = create_app()
    token = issue_dev_jwt("analyst", tenant, roles=["reader", "admin", "ingestor"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def test_two_connections_query_independently(inmemory: None, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    sales_db = tmp_path / "sales.db"
    inventory_db = tmp_path / "inventory.db"
    _seed(str(sales_db), "deals", "INSERT INTO deals VALUES (1,'Acme',5000);")
    _seed(str(inventory_db), "stock", "INSERT INTO stock VALUES (1,'Widgets',200);")
    tenant = "cag"
    c = await _client(tenant, monkeypatch)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})

        sales = (
            await c.post(
                f"/v1/kb/{tenant}/sql/connections",
                json={"name": "Sales DB", "dsn": f"sqlite:///{sales_db}", "tables": ["deals"], "dialect": "sqlite"},
            )
        ).json()
        assert sales["is_default"] is True  # first connection on a KB is always the default

        inventory = (
            await c.post(
                f"/v1/kb/{tenant}/sql/connections",
                json={
                    "name": "Inventory DB",
                    "dsn": f"sqlite:///{inventory_db}",
                    "tables": ["stock"],
                    "dialect": "sqlite",
                    "make_default": False,
                },
            )
        ).json()
        assert inventory["is_default"] is False  # explicit make_default:false honoured

        # Full list shows both, default first.
        listed = (await c.get(f"/v1/kb/{tenant}/sql/connections")).json()
        assert [c["name"] for c in listed] == ["Sales DB", "Inventory DB"]

        # No connection_id -> default (Sales).
        r = await c.post(f"/v1/kb/{tenant}/sql/execute", json={"sql": "SELECT name FROM deals"})
        assert r.status_code == 200 and r.json()["rows"] == [["Acme"]]
        assert r.json()["connection_name"] == "Sales DB"

        # Explicit connection_id -> Inventory, NOT the default.
        r = await c.post(
            f"/v1/kb/{tenant}/sql/execute",
            json={"sql": "SELECT name FROM stock", "connection_id": inventory["conn_id"]},
        )
        assert r.status_code == 200 and r.json()["rows"] == [["Widgets"]]
        assert r.json()["connection_name"] == "Inventory DB"

        # Sales' table isn't in Inventory's allowlist — proves connections don't bleed
        # into each other's governance (each has its OWN table allowlist).
        r = await c.post(
            f"/v1/kb/{tenant}/sql/execute",
            json={"sql": "SELECT name FROM deals", "connection_id": inventory["conn_id"]},
        )
        assert r.status_code == 422

        # Flip the default, then confirm the no-connection_id path follows it.
        r = await c.post(f"/v1/kb/{tenant}/sql/connections/{inventory['conn_id']}/default")
        assert r.status_code == 200 and r.json()["is_default"] is True
        r = await c.post(f"/v1/kb/{tenant}/sql/execute", json={"sql": "SELECT name FROM stock"})
        assert r.status_code == 200 and r.json()["connection_name"] == "Inventory DB"

        # Delete Sales; it's gone from the list, and asking it directly now 404s.
        r = await c.delete(f"/v1/kb/{tenant}/sql/connections/{sales['conn_id']}")
        assert r.status_code == 204
        listed = (await c.get(f"/v1/kb/{tenant}/sql/connections")).json()
        assert [c["conn_id"] for c in listed] == [inventory["conn_id"]]
        r = await c.post(
            f"/v1/kb/{tenant}/sql/execute",
            json={"sql": "SELECT 1", "connection_id": sales["conn_id"]},
        )
        assert r.status_code == 404


async def test_rename_and_rotate_dsn_without_losing_data(
    inmemory: None, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    db1 = tmp_path / "old.db"
    db2 = tmp_path / "new.db"
    _seed(str(db1), "t", "INSERT INTO t VALUES (1,'old',1);")
    _seed(str(db2), "t", "INSERT INTO t VALUES (1,'new',2);")
    tenant = "cag"
    c = await _client(tenant, monkeypatch)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        conn = (
            await c.post(
                f"/v1/kb/{tenant}/sql/connections",
                json={"name": "Original", "dsn": f"sqlite:///{db1}", "tables": ["t"], "dialect": "sqlite"},
            )
        ).json()

        # Rename only — DSN untouched, still reads the old DB.
        r = await c.put(
            f"/v1/kb/{tenant}/sql/connections/{conn['conn_id']}",
            json={"name": "Renamed", "tables": ["t"]},
        )
        assert r.status_code == 200 and r.json()["name"] == "Renamed"
        r = await c.post(f"/v1/kb/{tenant}/sql/execute", json={"sql": "SELECT name FROM t"})
        assert r.json()["rows"] == [["old"]]

        # Rotate the DSN — same conn_id, now reads the new DB.
        r = await c.put(
            f"/v1/kb/{tenant}/sql/connections/{conn['conn_id']}",
            json={"name": "Renamed", "dsn": f"sqlite:///{db2}", "tables": ["t"]},
        )
        assert r.status_code == 200
        r = await c.post(f"/v1/kb/{tenant}/sql/execute", json={"sql": "SELECT name FROM t"})
        assert r.json()["rows"] == [["new"]]


async def test_no_connection_registered_is_404_not_500(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    tenant = "cag"
    c = await _client(tenant, monkeypatch)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        r = await c.post(f"/v1/kb/{tenant}/sql/execute", json={"sql": "SELECT 1"})
        assert r.status_code == 404
        info = (await c.get(f"/v1/kb/{tenant}/sql/connection")).json()
        assert info["configured"] is False
