"""E2E: agentic-workflow runtime over real services (FR-F).

Builds a workflow that runs a governed SQL step, then branches on the row count via the rules
engine — proving deterministic orchestration + decision trace, composing the NL→SQL connector,
with no external model needed.
"""

from __future__ import annotations

import sqlite3

import httpx
import pytest

pytestmark = pytest.mark.e2e


def _seed_db(path: str) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE tickets (id INTEGER, status TEXT);"
        "INSERT INTO tickets VALUES (1,'open'), (2,'open'), (3,'closed');"
    )
    con.commit()
    con.close()


async def _client(tenant: str, monkeypatch: pytest.MonkeyPatch) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_AGENTIC_WORKFLOWS_ENABLED", "true")
    monkeypatch.setenv("RAG_NL2SQL_ENABLED", "true")
    get_settings.cache_clear()
    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["reader", "admin"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


_WORKFLOW = {
    "id": "triage",
    "name": "Ticket triage",
    "start": "count",
    "steps": [
        {"id": "count", "type": "sql", "config": {"question": "count open tickets", "output": "q"}, "next": "route"},
        {
            "id": "route",
            "type": "decision",
            "rules": [{"all": [{"field": "q.row_count", "op": "gte", "value": 1}], "goto": "has_rows"}],
            "next": "empty",
        },
        {"id": "has_rows", "type": "output"},
        {"id": "empty", "type": "output"},
    ],
}


async def test_register_and_run_workflow(inmemory: None, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    db = tmp_path / "ops.db"
    _seed_db(str(db))
    tenant = "cag"

    # The SQL step runs a fixed governed query rather than a model (question ignored by the stub).
    from svc_control_plane import sql_query

    class _StubGateway:
        model_name = "stub"

        async def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
            return "SELECT status, COUNT(*) FROM tickets WHERE status='open' GROUP BY status"

        async def aclose(self) -> None:
            return None

    async def _fake_gateway(kb_id: str, model: str | None = None) -> _StubGateway:
        return _StubGateway()

    monkeypatch.setattr(sql_query.wiring, "resolve_tenant_gateway", _fake_gateway)

    c = await _client(tenant, monkeypatch)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.post(
            f"/v1/kb/{tenant}/sql/connections",
            json={"name": "Tickets DB", "dsn": f"sqlite:///{db}", "tables": ["tickets"], "dialect": "sqlite"},
        )
        # Register the workflow.
        r = await c.put(f"/v1/kb/{tenant}/workflows/triage", json=_WORKFLOW)
        assert r.status_code == 200

        assert len((await c.get(f"/v1/kb/{tenant}/workflows")).json()) == 1

        # Run it: SQL finds an 'open' group (row_count>=1) → branches to has_rows.
        r = await c.post(f"/v1/kb/{tenant}/workflows/triage/run", json={"inputs": {}})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "completed"
        path = [t["step_id"] for t in body["trace"]]
        assert path == ["count", "route", "has_rows"]
        assert body["trace"][1]["matched_rule"] is True
        assert body["output"]["q"]["row_count"] == 1  # one 'open' group


async def test_run_disabled_without_flag(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt
    from rag_common.config import get_settings

    tenant = "cag"
    monkeypatch.setenv("RAG_AGENTIC_WORKFLOWS_ENABLED", "false")  # feature OFF
    get_settings.cache_clear()
    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["reader", "admin"])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"}
    ) as c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        simple = {"id": "w", "name": "w", "start": "s", "steps": [{"id": "s", "type": "output"}]}
        await c.put(f"/v1/kb/{tenant}/workflows/w", json=simple)  # registering is allowed
        r = await c.post(f"/v1/kb/{tenant}/workflows/w/run", json={"inputs": {}})
        assert r.status_code >= 400  # running is feature-gated
