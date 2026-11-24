"""E2E: knowledge-base deletion (full CRUD) — the KB is really gone, not just delisted.

A world-class platform lets you remove what you created. Deleting a KB purges its
documents from every store, drops its conversations, and de-registers it — and because
the tenant guard authorises ``{org}__*`` ids by prefix, the purge (not just delisting) is
what makes the data actually unreachable. The default KB is protected.
"""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e


async def _client(org: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("owner", org, roles=["reader", "ingestor", "admin"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def test_delete_kb_purges_everything(inmemory: None, lexical_gate: None) -> None:
    org = "acme"
    c = await _client(org)
    async with c:
        kb = (await c.post("/v1/knowledge-bases", json={"display_name": "HR Policies"})).json()
        kb_id = kb["kb_id"]
        await c.put(f"/v1/kb/{kb_id}/settings", json={"faithful_threshold": 0.1})
        doc = base64.b64encode(b"The leave policy grants 12 casual days.").decode()
        await c.post(f"/v1/kb/{kb_id}/ingest", json={"doc_id": "leave.txt", "mime": "text/plain", "content_b64": doc})

        # A conversation in the KB, then confirm the doc is answerable.
        conv = (await c.post(f"/v1/kb/{kb_id}/conversations", json={"query": "leave?"})).json()
        await c.post(
            f"/v1/kb/{kb_id}/conversations/{conv['conversation_id']}/ask", json={"query": "How many casual days?"}
        )
        assert (await c.post(f"/v1/kb/{kb_id}/query", json={"query": "How many casual days?"})).json()[
            "status"
        ] == "grounded"

        # DELETE is a soft-delete (Trash): hidden from the active list, data retained — but since
        # 82f1d5d ("trashed KBs are inaccessible and can't be silently re-created") a trashed KB is
        # also UNREACHABLE: _assert_kb_access, the tenant_guard chokepoint, 404s it. Retention is
        # therefore proven by restoring it, not by reading through the guard.
        res = await c.delete(f"/v1/knowledge-bases/{kb_id}")
        assert res.status_code == 200 and res.json()["status"] == "trashed"
        assert kb_id not in {k["kb_id"] for k in (await c.get("/v1/knowledge-bases")).json()}
        assert (await c.get(f"/v1/kb/{kb_id}/documents")).status_code == 404  # inaccessible while trashed
        assert (await c.post(f"/v1/kb/{kb_id}/query", json={"query": "How many casual days?"})).status_code == 404
        assert kb_id in {k["kb_id"] for k in (await c.get("/v1/knowledge-bases/trash")).json()}  # retained

        # Restore proves the data really was retained (restore/purge use get_principal, not
        # tenant_guard, so they still work on a trashed KB).
        restored = await c.post(f"/v1/knowledge-bases/{kb_id}/restore")
        assert restored.status_code == 200
        assert len((await c.get(f"/v1/kb/{kb_id}/documents")).json()) == 1  # the document survived the trash

        # Permanent purge (typed confirmation) actually removes everything.
        await c.delete(f"/v1/knowledge-bases/{kb_id}")
        purged = await c.post(f"/v1/knowledge-bases/{kb_id}/purge", json={"confirm_name": "HR Policies"})
        assert purged.status_code == 200 and purged.json()["documents"] == 1
        # …and the KB is de-registered: gone from both the active list and the trash.
        assert kb_id not in {k["kb_id"] for k in (await c.get("/v1/knowledge-bases")).json()}
        assert kb_id not in {k["kb_id"] for k in (await c.get("/v1/knowledge-bases/trash")).json()}
        # Note the deliberate asymmetry with the TRASHED state above: a purged id still resolves
        # past the tenant guard (which authorises `{org}__*` by prefix — see this module's
        # docstring), so the guarantee here is "nothing left to read", not 404. That is why the
        # purge must empty every store rather than rely on the id being rejected.
        assert (await c.get(f"/v1/kb/{kb_id}/documents")).json() == []
        after = (await c.post(f"/v1/kb/{kb_id}/query", json={"query": "How many casual days?"})).json()
        assert after["status"] == "abstained"
        # …and its conversations are gone.
        assert (await c.get(f"/v1/kb/{kb_id}/conversations")).json() == []


async def test_cannot_delete_default_kb(inmemory: None) -> None:
    org = "acme"
    c = await _client(org)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})  # ensures org provisioned
        # The default KB id == org id; the delete guard refuses it.
        r = await c.delete(f"/v1/knowledge-bases/{org}")
        assert r.status_code in (403, 422)


async def test_cannot_delete_another_orgs_kb(inmemory: None) -> None:
    a = await _client("orga")
    b = await _client("orgb")
    async with a, b:
        kb = (await a.post("/v1/knowledge-bases", json={"display_name": "Secret"})).json()
        r = await b.delete(f"/v1/knowledge-bases/{kb['kb_id']}")
        assert r.status_code == 403
