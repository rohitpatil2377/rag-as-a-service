"""E2E: the ``doc_ids`` document scope on the query API.

WHY THE FIELD EXISTS. ``filters={"doc_id": ...}`` has always been refused with the hint "Use
`doc_ids` for a document scope" — a field the request model did not have, so the hint named
nothing and a caller had no way to scope a question to one document. It is also the only way to
reach the Phase-8 section pass: that pass reads EVERY section of the documents in scope, so it
declines an unscoped question rather than load a whole KB onto the latency path, and the router
cannot infer a scope because the ``doc`` chunk level has no producer.

WHAT IS TESTED HERE is the API contract, not the section pass (which needs the v2 flags and a
real model): the scope narrows the answer, an absent scope changes nothing, and a named document
the caller may not read is refused the SAME way as one that does not exist — the field must not
become an oracle for the titles of classified files.
"""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e

_SLA = "The platform uptime target is 99.5 percent baseline and 99.9 percent at peak."
_DIESEL = "The diesel generator must sustain the full connected load for 8 hours continuously."
_SECRET = "The classified fallback site for Project Trishul is located at Sector 9."


async def _client(tenant: str, groups: list[str] | None = None, clearance: int | None = None) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    attrs: dict = {"groups": groups or []}
    if clearance is not None:
        attrs["clearance"] = clearance
    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["reader", "ingestor", "admin"], attrs=attrs)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://t",
        headers={"Authorization": f"Bearer {token}"},
    )


def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


async def _ingest(
    client: httpx.AsyncClient,
    tenant: str,
    doc_id: str,
    text: str,
    acl: list[str] | None = None,
    classification: int | None = None,
) -> int:
    body: dict = {"doc_id": doc_id, "mime": "text/plain", "content_b64": _b64(text)}
    if acl:
        body["acl"] = acl
    if classification is not None:
        body["classification"] = classification
    return (await client.post(f"/v1/kb/{tenant}/ingest", json=body)).status_code


async def _seed(client: httpx.AsyncClient, tenant: str) -> None:
    await client.post("/v1/knowledge-bases", json={"display_name": "KB"})
    assert await _ingest(client, tenant, "sla", _SLA) == 200
    assert await _ingest(client, tenant, "diesel", _DIESEL) == 200


async def test_a_scoped_query_only_cites_the_named_document(inmemory: None) -> None:
    """The point of the field: the same question, scoped, must not reach the other document."""
    tenant = "scope-a"
    client = await _client(tenant)
    async with client:
        await _seed(client, tenant)
        r = await client.post(
            f"/v1/kb/{tenant}/retrieve", json={"query": "what is required", "top_k": 5, "doc_ids": ["diesel"]}
        )
        assert r.status_code == 200, r.text
        docs = {s["doc_id"] for s in r.json()["spans"]}
        assert docs == {"diesel"}, f"the scope leaked: {docs}"


async def test_an_absent_scope_is_unchanged(inmemory: None) -> None:
    """Default-empty must be byte-identical to the behaviour before the field existed, which is
    what makes this additive: every existing caller sends no `doc_ids`."""
    tenant = "scope-b"
    client = await _client(tenant)
    async with client:
        await _seed(client, tenant)
        both = await client.post(f"/v1/kb/{tenant}/retrieve", json={"query": "what is required", "top_k": 5})
        assert both.status_code == 200
        docs = {s["doc_id"] for s in both.json()["spans"]}
        assert docs == {"sla", "diesel"}, f"an unscoped query should see the whole KB, saw {docs}"


async def test_an_unknown_document_is_refused_not_silently_empty(inmemory: None) -> None:
    """A typo would otherwise come back as a confident "the sources do not cover that", which
    reads exactly like the document genuinely not saying it."""
    tenant = "scope-c"
    client = await _client(tenant)
    async with client:
        await _seed(client, tenant)
        r = await client.post(f"/v1/kb/{tenant}/query", json={"query": "what is required", "doc_ids": ["no-such-doc"]})
        assert r.status_code == 422, r.text
        assert "no-such-doc" in r.text


async def test_a_document_the_caller_may_not_read_fails_like_one_that_does_not_exist(inmemory: None) -> None:
    """NO EXISTENCE ORACLE. Distinguishing "not yours" from "no such document" would let an
    unprivileged caller enumerate classified titles by watching which ids are accepted, so both
    must produce the same refusal."""
    tenant = "scope-d"
    owner = await _client(tenant, groups=["intel"])
    async with owner:
        await owner.post("/v1/knowledge-bases", json={"display_name": "KB"})
        assert await _ingest(owner, tenant, "sla", _SLA) == 200
        assert await _ingest(owner, tenant, "fallback", _SECRET, acl=["intel"]) == 200

    outsider = await _client(tenant, groups=["facilities"])
    async with outsider:
        restricted = await outsider.post(
            f"/v1/kb/{tenant}/query", json={"query": "where is the fallback site", "doc_ids": ["fallback"]}
        )
        missing = await outsider.post(
            f"/v1/kb/{tenant}/query", json={"query": "where is the fallback site", "doc_ids": ["ghost"]}
        )
        assert restricted.status_code == missing.status_code == 422, (restricted.text, missing.text)
        # Same CODE and same SHAPE of message — only the echoed id differs, which the caller
        # already supplied and so learns nothing from.
        assert restricted.json()["code"] == missing.json()["code"]
        assert _SECRET not in restricted.text


async def test_a_scope_naming_another_tenants_document_is_refused(inmemory: None) -> None:
    """Defence in depth. `tenant_guard` and the per-tenant namespace already make this
    unreachable — a foreign id narrows the search to nothing rather than crossing a boundary —
    but the field must still refuse it rather than answer emptily."""
    tenant, other = "scope-e", "scope-f"
    mine = await _client(tenant)
    async with mine:
        await _seed(mine, tenant)
    theirs = await _client(other)
    async with theirs:
        await theirs.post("/v1/knowledge-bases", json={"display_name": "KB"})
        assert await _ingest(theirs, other, "their-secret", _SECRET) == 200

    mine2 = await _client(tenant)
    async with mine2:
        r = await mine2.post(f"/v1/kb/{tenant}/query", json={"query": "what is required", "doc_ids": ["their-secret"]})
        assert r.status_code == 422, r.text
        assert _SECRET not in r.text
