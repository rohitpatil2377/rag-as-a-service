"""E2E: hierarchical KB access enforced at the chokepoint + discovery + ingest level.

A KB above the caller's clearance is neither LISTED nor ACCESSIBLE (404, existence hidden); a
cleared caller sees it. And a caller cannot ingest a document classified ABOVE its own clearance
even into a KB it can write to (the exfiltration guard)."""

from __future__ import annotations

import base64
from collections.abc import Iterator

import httpx
import pytest

pytestmark = pytest.mark.e2e


@pytest.fixture
def publish_off(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_DUAL_CONTROL_ACTIONS", "")  # no publish gate — KBs answer immediately
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _client(tenant: str, roles: list[str], clearance: int | str = 0) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    token = issue_dev_jwt("officer", tenant, roles=roles, attrs={"clearance": clearance})
    transport = httpx.ASGITransport(app=create_app())
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def test_a_confidential_kb_is_hidden_and_blocked_below_clearance(
    inmemory: None, lexical_gate: None, publish_off: None
) -> None:
    from svc_control_plane import wiring

    tenant = "bank"
    owner = _client(tenant, ["owner", "ingestor"], clearance="restricted")
    async with owner:
        r = await owner.post("/v1/knowledge-bases", json={"display_name": "Exec"})
        assert r.status_code == 201, r.text
        kb_id = r.json()["kb_id"]
    # Classify the KB Confidential (floor 2) — done via the store (the management API is Sub-step D).
    await wiring.kb_store().set_governance(tenant, kb_id, unit_id=None, classification_floor=2)

    # A cleared (Confidential) reader sees it.
    high = _client(tenant, ["reader"], clearance="confidential")
    async with high:
        listing = (await high.get("/v1/knowledge-bases")).json()
        assert any(k["kb_id"] == kb_id for k in listing), "a cleared reader must see the KB"
        assert (await high.post(f"/v1/kb/{kb_id}/query", json={"query": "hi"})).status_code == 200

    # A below-clearance (Internal) reader: NOT listed, and access is 404 (existence hidden).
    low = _client(tenant, ["reader"], clearance="internal")
    async with low:
        listing = (await low.get("/v1/knowledge-bases")).json()
        assert all(k["kb_id"] != kb_id for k in listing), "a too-high KB must not appear in discovery"
        assert (await low.post(f"/v1/kb/{kb_id}/query", json={"query": "hi"})).status_code == 404


async def test_cannot_ingest_above_your_clearance(inmemory: None, publish_off: None) -> None:
    tenant = "corp"
    owner = _client(tenant, ["owner", "ingestor"], clearance="restricted")
    async with owner:
        kb_id = (await owner.post("/v1/knowledge-bases", json={"display_name": "Docs"})).json()["kb_id"]

    # An Internal-cleared ingestor may write to this Public-floor KB, but NOT at Restricted level.
    ingestor = _client(tenant, ["reader", "ingestor"], clearance="internal")
    async with ingestor:
        body = {
            "doc_id": "d1",
            "mime": "text/plain",
            "content_b64": base64.b64encode(b"some content").decode(),
            "classification": "restricted",
        }
        r = await ingestor.post(f"/v1/kb/{kb_id}/ingest", json=body)
        assert r.status_code == 403, r.text
        # ...but ingesting at its own (Internal) level is allowed.
        body["classification"] = "internal"
        assert (await ingestor.post(f"/v1/kb/{kb_id}/ingest", json=body)).status_code == 200
