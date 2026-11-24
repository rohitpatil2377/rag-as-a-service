"""E2E: governed NL→SQL over a real SQLite database (FR-D10).

Exercises the full path — register a read-only connection, run a governed analyst SQL, block
an unsafe one, and drive NL→SQL end-to-end with a stub model — with NO external DB server.
"""

from __future__ import annotations

import sqlite3

import httpx
import pytest

pytestmark = pytest.mark.e2e


def _seed_db(path: str) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE employees (id INTEGER, name TEXT, dept TEXT, salary INTEGER);
        INSERT INTO employees VALUES (1,'Asha','Eng',120), (2,'Ravi','Eng',110), (3,'Meena','Ops',90);
        CREATE TABLE secrets (id INTEGER, token TEXT);
        INSERT INTO secrets VALUES (1,'do-not-read');
        """
    )
    con.commit()
    con.close()


async def _client(tenant: str, monkeypatch: pytest.MonkeyPatch) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_NL2SQL_ENABLED", "true")
    get_settings.cache_clear()
    app = create_app()
    token = issue_dev_jwt("analyst", tenant, roles=["reader", "admin"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def test_register_execute_and_block(inmemory: None, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    db = tmp_path / "hr.db"
    _seed_db(str(db))
    tenant = "cag"
    c = await _client(tenant, monkeypatch)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        # Register: only the employees table is allowlisted (secrets is NOT).
        r = await c.post(
            f"/v1/kb/{tenant}/sql/connections",
            json={"name": "HR DB", "dsn": f"sqlite:///{db}", "tables": ["employees"], "dialect": "sqlite"},
        )
        assert r.status_code == 201
        assert r.json()["tables"] == ["employees"]
        assert r.json()["is_default"] is True

        # Governed execute of a legit SELECT returns rows.
        sql = "SELECT name, salary FROM employees WHERE dept='Eng'"
        r = await c.post(f"/v1/kb/{tenant}/sql/execute", json={"sql": sql})
        assert r.status_code == 200
        body = r.json()
        assert body["row_count"] == 2
        assert "limit" in body["sql"].lower()  # bounded
        assert {row[0] for row in body["rows"]} == {"Asha", "Ravi"}

        # A write is refused by governance (422), never reaches the DB.
        r = await c.post(f"/v1/kb/{tenant}/sql/execute", json={"sql": "DELETE FROM employees"})
        assert r.status_code == 422

        # Querying a non-allowlisted table is refused even though it exists.
        r = await c.post(f"/v1/kb/{tenant}/sql/execute", json={"sql": "SELECT token FROM secrets"})
        assert r.status_code == 422
        # ... and the secret was never read.
        assert "do-not-read" not in r.text


async def test_nl_to_sql_with_stub_model(inmemory: None, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    db = tmp_path / "hr.db"
    _seed_db(str(db))
    tenant = "cag"

    class _StubGateway:
        model_name = "stub"

        async def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
            return "```sql\nSELECT dept, COUNT(*) FROM employees GROUP BY dept\n```"

        async def aclose(self) -> None:
            return None

    from svc_control_plane import sql_query

    async def _fake_gateway(kb_id: str, model: str | None = None) -> _StubGateway:
        return _StubGateway()

    monkeypatch.setattr(sql_query.wiring, "resolve_tenant_gateway", _fake_gateway)

    c = await _client(tenant, monkeypatch)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.post(
            f"/v1/kb/{tenant}/sql/connections",
            json={"name": "HR DB", "dsn": f"sqlite:///{db}", "tables": ["employees"], "dialect": "sqlite"},
        )
        r = await c.post(f"/v1/kb/{tenant}/sql/query", json={"question": "how many employees per department?"})
        assert r.status_code == 200
        body = r.json()
        assert body["question"]
        assert "group by" in body["sql"].lower()
        assert body["row_count"] == 2  # Eng, Ops


def test_tsql_row_cap_does_not_break_ordered_questions() -> None:
    """SQL Server rejects a derived table that carries its own ORDER BY (error 1033).

    Observed live: "How many tickets are there, and how do they break down by status?" produced
    perfectly good SQL, and the row-cap wrap then made it illegal — so every ranked question
    ("break down by", "top 5") failed against MSSQL while the identical shape worked on Postgres.
    T-SQL takes the cap as TOP on the statement, leaving ORDER BY at the top level where it is legal.
    """
    from svc_control_plane.nl2sql import validate_sql

    ordered = (
        "SELECT st.STATUSNAME, COUNT(*) AS n FROM WorkOrderStates s "
        "JOIN StatusDefinition st ON st.STATUSID = s.STATUSID GROUP BY st.STATUSNAME ORDER BY n DESC"
    )
    out = validate_sql(ordered, allowed_tables={"workorderstates", "statusdefinition"}, dialect="tsql", max_rows=1000)
    assert out.upper().startswith("SELECT TOP 1000"), "T-SQL must cap with TOP, not a derived table"
    assert "_rag" not in out, "no subquery wrap on T-SQL — that is what error 1033 rejects"
    assert "ORDER BY" in out.upper(), "the ranking must survive the cap"

    # The wrap is still used everywhere else, so the bound holds whatever the inner shape.
    pg = validate_sql(
        "SELECT status, COUNT(*) AS n FROM leads GROUP BY status ORDER BY n DESC",
        allowed_tables={"leads"},
        dialect="postgres",
        max_rows=1000,
    )
    assert "_rag" in pg and "LIMIT 1000" in pg.upper()
