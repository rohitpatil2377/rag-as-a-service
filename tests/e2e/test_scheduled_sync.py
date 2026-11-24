"""E2E: FR-A8 scheduled & delta connector sync — capture source, re-sync skips unchanged."""

from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.e2e

_ENDPOINT = "s3.sovereign.local"
_BUCKET = "acme-docs"
_CRED_REF = "acmesrc"


def _source() -> dict[str, object]:
    return {"endpoint": _ENDPOINT, "bucket": _BUCKET, "secure": True, "credentials_ref": _CRED_REF}


def _allow(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", f"localhost,{_ENDPOINT}")
    monkeypatch.setenv("RAG_ACMESRC_ACCESS_KEY", "AKIA-dev")
    monkeypatch.setenv("RAG_ACMESRC_SECRET_KEY", "shh-dev")
    get_settings.cache_clear()


def _seed(*files: tuple[str, bytes]) -> None:
    from rag_clients.factory import get_external_object_fetcher

    fetcher = get_external_object_fetcher()
    for key, data in files:
        fetcher.seed(_ENDPOINT, _BUCKET, key, data)  # type: ignore[attr-defined]


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    token = issue_dev_jwt("admin", tenant, roles=["reader", "ingestor", "admin"])
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()),
        base_url="http://t",
        headers={"Authorization": f"Bearer {token}"},
    )


async def test_scheduled_delta_resync(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _allow(monkeypatch)
    _seed(("a.txt", b"Annual leave is 24 days."), ("b.txt", b"Sick leave is 12 days."))
    kb = "syncco"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        # 1) Initial folder sync — captures the source; both files created.
        r1 = (
            await c.post(
                f"/v1/kb/{kb}/ingest/sync", json={"external_source": _source(), "prefix": "", "strategy": "clause"}
            )
        ).json()
        assert r1["ingested"] == 2 and r1["unchanged"] == 0

        # 2) Enrol for scheduled sync (source was captured).
        sched = (await c.put(f"/v1/kb/{kb}/sync/schedule", json={"enabled": True, "interval_hours": 24})).json()
        assert sched["enabled"] is True and sched["source_configured"] is True

        # 3) Re-sync now — nothing changed, so the delta skips both (cheap).
        r2 = (await c.post(f"/v1/kb/{kb}/sync/run-now")).json()
        assert r2["found"] == 2 and r2["unchanged"] == 2 and r2["updated"] == 0

        # 4) One file changes -> only it re-indexes.
        _seed(("a.txt", b"Annual leave is now 30 days."))
        r3 = (await c.post(f"/v1/kb/{kb}/sync/run-now")).json()
        assert r3["updated"] == 1 and r3["unchanged"] == 1

        # 5) The scheduler tick runs the same due job.
        from svc_control_plane import sync_jobs

        ran = await sync_jobs.sync_tick()
        assert ran >= 1


async def test_cron_schedule_calendar_precise(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _allow(monkeypatch)
    _seed(("a.txt", b"Annual leave is 24 days."))
    kb = "syncron"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await c.post(f"/v1/kb/{kb}/ingest/sync", json={"external_source": _source(), "prefix": ""})

        # Enrol with a cron: every Monday 02:00 IST. It is validated, stored, and anchored.
        r = await c.put(
            f"/v1/kb/{kb}/sync/schedule",
            json={"enabled": True, "cron": "0 2 * * 1", "timezone": "Asia/Kolkata"},
        )
        assert r.status_code == 200
        v = r.json()
        assert v["enabled"] is True and v["cron"] == "0 2 * * 1" and v["timezone"] == "Asia/Kolkata"
        assert v["next_run_at"] is not None  # computed next fire time
        # Anchored at enable -> it must NOT be immediately due.
        from svc_control_plane import sync_jobs

        sched = await sync_jobs.sync_schedule_store().get(kb)
        assert sched is not None and sync_jobs.is_due(sched, sched.last_run_at) is False  # type: ignore[arg-type]

        # A bad cron is rejected with a clear error.
        bad = await c.put(f"/v1/kb/{kb}/sync/schedule", json={"enabled": True, "cron": "99 * * * *"})
        assert bad.status_code in (400, 422)


async def test_schedule_requires_captured_source(inmemory: None) -> None:
    kb = "syncnosrc"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        resp = await c.put(f"/v1/kb/{kb}/sync/schedule", json={"enabled": True, "interval_hours": 24})
        assert resp.status_code in (400, 422)  # no source captured yet
        assert (await c.post(f"/v1/kb/{kb}/sync/run-now")).status_code == 404
