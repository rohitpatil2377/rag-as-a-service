"""Adversarial auth/RBAC/MCP boundary audit (production go-live gate).

Covers, per the audit brief:
  (a) API key scopes — read key REJECTED for ingest/admin ops
  (b) revoked key -> 401 everywhere incl /mcp/
  (c) expired/malformed/no JWT -> 401
  (d) MCP get_principal fix — API key AND JWT both work; bogus/revoked 401; tenant-X key
      cannot drive MCP tools against tenant Y
  (e) platform-admin allowlist — non-listed 403 on /v1/platform/*
  (f) hierarchical RBAC — clearance x nested unit scope; escalation-gated assignment APIs
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator

import httpx
import pytest

pytestmark = pytest.mark.e2e

_HDR = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


# --- harness -----------------------------------------------------------------


def _app():
    from rag_api.app import create_app

    return create_app()


def _client(headers: dict[str, str] | None = None, app=None) -> httpx.AsyncClient:
    app = app or _app()
    c = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", headers=headers or {})
    c._rag_app = app  # type: ignore[attr-defined]
    return c


def _jwt(tenant: str, roles: list[str], attrs: dict | None = None, *, ttl: int | None = None) -> str:
    from rag_common import issue_dev_jwt

    return issue_dev_jwt("officer", tenant, roles=roles, attrs=attrs or {}, ttl_seconds=ttl)


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def _rpc(method: str, rpc_id: int, params: dict | None = None) -> dict:
    body: dict = {"jsonrpc": "2.0", "id": rpc_id, "method": method}
    if params is not None:
        body["params"] = params
    return body


@pytest.fixture
def publish_off(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_DUAL_CONTROL_ACTIONS", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# --- (a) API key scopes ------------------------------------------------------


async def test_read_key_rejected_for_ingest_and_admin_write(inmemory: None, publish_off: None) -> None:
    """A 'read' scope key: can query, but 403 on ingest AND on admin-only writes (mint key,
    add member). Also confirms the create path itself is 201 for the admin owner."""
    org = "zzzscope"
    owner = _client(_bearer(_jwt(org, ["owner", "admin", "ingestor", "reader"])))
    async with owner:
        assert (await owner.post("/v1/knowledge-bases", json={"display_name": "KB"})).status_code == 201
        assert (
            await owner.post(
                f"/v1/kb/{org}/ingest",
                json={"doc_id": "sla", "mime": "text/plain", "content_b64": _b64("The uptime target is 99.5 percent.")},
            )
        ).status_code == 200
        created = await owner.post("/v1/api-keys", json={"name": "ro", "scope": "read"})
        assert created.status_code in (200, 201), created.text
        read_key = created.json()["key"]

    bot = _client(_bearer(read_key))
    async with bot:
        # read works
        q = await bot.post(f"/v1/kb/{org}/query", json={"query": "What is the uptime target?"})
        assert q.status_code == 200, q.text
        # ingest write -> 403 (no ingestor role)
        ing = await bot.post(
            f"/v1/kb/{org}/ingest", json={"doc_id": "x", "mime": "text/plain", "content_b64": _b64("x")}
        )
        assert ing.status_code == 403, ing.text
        # admin-only: mint another key -> 403 (no admin role)
        km = await bot.post("/v1/api-keys", json={"name": "sneaky", "scope": "admin"})
        assert km.status_code == 403, km.text
        # admin-only: add member -> 403
        am = await bot.post("/v1/org/members", json={"email": "e@x.com", "name": "E", "roles": ["reader"]})
        assert am.status_code == 403, am.text


async def test_ingest_key_cannot_do_admin(inmemory: None, publish_off: None) -> None:
    """An 'ingest' scope key can ingest but still cannot mint keys (admin)."""
    org = "zzzingest"
    owner = _client(_bearer(_jwt(org, ["owner", "admin"])))
    async with owner:
        await owner.post("/v1/knowledge-bases", json={"display_name": "KB"})
        ik = (await owner.post("/v1/api-keys", json={"name": "ci", "scope": "ingest"})).json()["key"]
    bot = _client(_bearer(ik))
    async with bot:
        ing = await bot.post(
            f"/v1/kb/{org}/ingest", json={"doc_id": "d", "mime": "text/plain", "content_b64": _b64("hello world")}
        )
        assert ing.status_code == 200, ing.text
        assert (await bot.post("/v1/api-keys", json={"name": "z", "scope": "admin"})).status_code == 403


# --- (b) revoked key -> 401 everywhere incl /mcp/ ----------------------------


async def test_revoked_key_401_on_v1_and_mcp(inmemory: None, publish_off: None) -> None:
    org = "zzzrevoke"
    app = _app()
    owner = _client(_bearer(_jwt(org, ["owner", "admin"])), app=app)
    async with owner:
        await owner.post("/v1/knowledge-bases", json={"display_name": "KB"})
        created = (await owner.post("/v1/api-keys", json={"name": "rev", "scope": "read"})).json()
        full, key_id = created["key"], created["key_id"]
        # works before revoke
        assert (await owner.get("/v1/api-keys")).status_code == 200
        bot = _client(_bearer(full), app=app)
        async with bot:
            assert (await bot.post(f"/v1/kb/{org}/query", json={"query": "hi"})).status_code == 200
        # revoke it
        rv = await owner.delete(f"/v1/api-keys/{key_id}")
        assert rv.status_code in (200, 204), rv.text

    # after revoke: 401 on /v1 (no lifespan needed for a rejected request — auth runs first).
    # The revoked-key-on-/mcp/ assertion lives in the single MCP lifespan test below (the MCP
    # session manager is a module-level singleton: only one test per process may run it).
    bot2 = _client(_bearer(full), app=app)
    async with bot2:
        r = await bot2.post(f"/v1/kb/{org}/query", json={"query": "hi"})
        assert r.status_code == 401, r.text


# --- (c) expired / malformed / no JWT -> 401 ---------------------------------


async def test_missing_malformed_expired_jwt_401(inmemory: None) -> None:
    app = _app()
    cases = {
        "no-header": {},
        "not-bearer": {"Authorization": "Basic abc"},
        "garbage-token": _bearer("not.a.jwt"),
        "empty-bearer": {"Authorization": "Bearer "},
        "wrong-secret": _bearer(_forge_wrong_secret("t")),
        "expired": _bearer(_jwt("t", ["reader"], ttl=-10)),  # exp in the past
    }
    for name, hdr in cases.items():
        c = _client(hdr, app=app)
        async with c:
            r = await c.post("/v1/kb/t/query", json={"query": "hi"})
            assert r.status_code == 401, f"{name}: expected 401, got {r.status_code} {r.text}"


def _forge_wrong_secret(tenant: str) -> str:
    import jwt as _pyjwt

    return _pyjwt.encode({"sub": "x", "tenant_id": tenant, "roles": ["admin"]}, "attacker-secret", algorithm="HS256")


# --- (d) MCP get_principal fix ----------------------------------------------


async def test_mcp_accepts_api_key_and_jwt_rejects_bogus(inmemory: None, publish_off: None) -> None:
    app = _app()
    org = "zzzmcp"
    c = _client(app=app)
    async with c, app.router.lifespan_context(app):
        # seed data + mint a read key via the API (same app/lifespan as the MCP calls)
        owner_hdr = {**_HDR, **_bearer(_jwt(org, ["owner", "admin", "reader"]))}
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"}, headers=owner_hdr)
        await c.post(
            f"/v1/kb/{org}/ingest",
            headers=owner_hdr,
            json={"doc_id": "sla", "mime": "text/plain", "content_b64": _b64("The escalation window is 4 hours.")},
        )
        api_key = (await c.post("/v1/api-keys", json={"name": "mcpbot", "scope": "read"}, headers=owner_hdr)).json()[
            "key"
        ]

        # JWT works on /mcp/
        jwt_hdr = {**_HDR, **_bearer(_jwt(org, ["reader"]))}
        r = await c.post("/mcp/", json=_rpc("tools/list", 1), headers=jwt_hdr)
        assert r.status_code == 200, r.text

        # API KEY works on /mcp/ (the get_principal fix) — initialize + tools/list
        key_hdr = {**_HDR, **_bearer(api_key)}
        ini = await c.post(
            "/mcp/",
            json=_rpc(
                "initialize",
                2,
                {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
            ),
            headers=key_hdr,
        )
        assert ini.status_code == 200, ini.text
        tl = await c.post("/mcp/", json=_rpc("tools/list", 3), headers=key_hdr)
        assert tl.status_code == 200, tl.text

        # API key can drive a tool call scoped to its own org
        call = _rpc("tools/call", 4, {"name": "rag.query", "arguments": {"query": "What is the escalation window?"}})
        tc = await c.post("/mcp/", json=call, headers=key_hdr)
        assert tc.status_code == 200, tc.text
        payload = json.loads(tc.json()["result"]["content"][0]["text"])
        assert payload["tenant_id"] == org

        # bogus API key -> 401
        b = await c.post("/mcp/", json=_rpc("tools/list", 5), headers={**_HDR, **_bearer("ek_deadbeef_notreal")})
        assert b.status_code == 401, b.text
        # bogus JWT -> 401
        b2 = await c.post("/mcp/", json=_rpc("tools/list", 6), headers={**_HDR, **_bearer("garbage")})
        assert b2.status_code == 401, b2.text

        # --- (d cont.) cross-tenant: org X's key cannot drive MCP against org Y's KB -----
        # (These MCP assertions live together because they share one seeded fixture, not because
        # the transport can only start once — each app now builds its own MCP server.)
        x, y = "zzzx", "zzzy"
        yh = {**_HDR, **_bearer(_jwt(y, ["owner", "admin", "ingestor", "reader"]))}
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"}, headers=yh)
        await c.post(
            f"/v1/kb/{y}/ingest",
            headers=yh,
            json={"doc_id": "s", "mime": "text/plain", "content_b64": _b64("Y secret alpha is 12345.")},
        )
        xh = {**_HDR, **_bearer(_jwt(x, ["owner", "admin", "reader"]))}
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"}, headers=xh)
        xkey = (await c.post("/v1/api-keys", json={"name": "b", "scope": "read"}, headers=xh)).json()["key"]
        xkey_hdr = {**_HDR, **_bearer(xkey)}
        # X's key targeting Y's KB explicitly via kb_id -> must NOT surface Y's data.
        xcall = _rpc(
            "tools/call", 7, {"name": "rag.query", "arguments": {"query": "What is Y secret alpha?", "kb_id": y}}
        )
        xr = await c.post("/mcp/", json=xcall, headers=xkey_hdr)
        assert xr.status_code == 200, xr.text
        result = xr.json().get("result", {})
        if result.get("isError"):
            assert "12345" not in result["content"][0]["text"]
        else:
            xpayload = json.loads(result["content"][0]["text"])
            assert xpayload.get("tenant_id") != y, f"cross-tenant leak: got tenant {xpayload.get('tenant_id')}"
            assert "12345" not in json.dumps(xpayload), "cross-tenant data leak via kb_id"

        # --- (b cont.) a REVOKED key must 401 on /mcp/ (checked in this single lifespan) ----
        rk = (await c.post("/v1/api-keys", json={"name": "rev", "scope": "read"}, headers=owner_hdr)).json()
        rk_full, rk_id = rk["key"], rk["key_id"]
        # works on /mcp/ before revoke
        pre = await c.post("/mcp/", json=_rpc("tools/list", 8), headers={**_HDR, **_bearer(rk_full)})
        assert pre.status_code == 200, pre.text
        # revoke, then it must 401 on /mcp/
        assert (await c.delete(f"/v1/api-keys/{rk_id}", headers=owner_hdr)).status_code in (200, 204)
        post = await c.post("/mcp/", json=_rpc("tools/list", 9), headers={**_HDR, **_bearer(rk_full)})
        assert post.status_code == 401, post.text


# --- (e) platform-admin allowlist -------------------------------------------


async def test_platform_endpoints_403_for_non_allowlisted(inmemory: None) -> None:
    """With an empty allowlist NOBODY is a platform admin -> 403 even for an org owner/admin."""
    c = _client(_bearer(_jwt("zzzplat", ["owner", "admin"])))
    async with c:
        r = await c.get("/v1/platform/tenants")
        assert r.status_code == 403, r.text


async def test_platform_allowlisted_email_permitted(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """When RAG_PLATFORM_ADMIN_EMAILS lists the subject, the platform endpoint is reachable;
    a different subject is still 403 (proves the allowlist actually gates)."""
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_PLATFORM_ADMIN_EMAILS", "ops@acme.example")
    get_settings.cache_clear()
    from rag_common import issue_dev_jwt

    app = _app()
    admin_tok = issue_dev_jwt("ops@acme.example", "zzzp", roles=["owner"])
    other_tok = issue_dev_jwt("nobody@evil.com", "zzzp", roles=["owner", "admin"])
    allowed = _client(_bearer(admin_tok), app=app)
    async with allowed:
        r = await allowed.get("/v1/platform/tenants")
        assert r.status_code == 200, r.text
    denied = _client(_bearer(other_tok), app=app)
    async with denied:
        r = await denied.get("/v1/platform/tenants")
        assert r.status_code == 403, r.text
    get_settings.cache_clear()


# --- (f) hierarchical RBAC: clearance x nested scope + escalation ------------


async def test_sibling_unit_kb_not_readable(inmemory: None, publish_off: None) -> None:
    """A member of unit A (viewer/member) cannot read a KB living in sibling unit B — even at
    equal clearance. Scope (need-to-know) is independent of clearance."""
    from svc_control_plane import wiring
    from svc_control_plane.org_units import OrgUnit

    tenant = "zzzunit"
    store = wiring.org_unit_store()
    await store.ensure_root(tenant)
    await store.create(OrgUnit(unit_id="unit-a", tenant_id=tenant, name="A", parent_id=tenant))
    await store.create(OrgUnit(unit_id="unit-b", tenant_id=tenant, name="B", parent_id=tenant))

    owner = _client(_bearer(_jwt(tenant, ["owner", "admin"], {"clearance": "restricted"})))
    async with owner:
        kb = (await owner.post("/v1/knowledge-bases", json={"display_name": "B Docs"})).json()["kb_id"]
    # Put the KB in sibling unit B, Internal floor.
    await wiring.kb_store().set_governance(tenant, kb, unit_id="unit-b", classification_floor=1)

    # A reader in unit A, cleared Internal (>= floor) but WRONG unit -> hidden (404).
    a_reader = _client(_bearer(_jwt(tenant, ["reader"], {"clearance": "internal", "unit": "unit-a"})))
    async with a_reader:
        listing = (await a_reader.get("/v1/knowledge-bases")).json()
        assert all(k["kb_id"] != kb for k in listing), "sibling-unit KB must not be discoverable"
        r = await a_reader.post(f"/v1/kb/{kb}/query", json={"query": "hi"})
        assert r.status_code == 404, r.text

    # A reader in unit B, same clearance -> can reach it (proves it's scope, not a blanket deny).
    b_reader = _client(_bearer(_jwt(tenant, ["reader"], {"clearance": "internal", "unit": "unit-b"})))
    async with b_reader:
        r = await b_reader.post(f"/v1/kb/{kb}/query", json={"query": "hi"})
        assert r.status_code == 200, r.text


async def test_child_unit_admin_cannot_read_parent_higher_classified(inmemory: None, publish_off: None) -> None:
    """A UNIT_ADMIN of a child unit governs its subtree DOWNWARD only; a higher-classified KB in
    the PARENT unit above its clearance stays hidden (ceiling checked first)."""
    from svc_control_plane import wiring
    from svc_control_plane.org_units import OrgUnit

    tenant = "zzzchild"
    store = wiring.org_unit_store()
    await store.ensure_root(tenant)
    await store.create(OrgUnit(unit_id="child", tenant_id=tenant, name="Child", parent_id=tenant))

    owner = _client(_bearer(_jwt(tenant, ["owner", "admin"], {"clearance": "restricted"})))
    async with owner:
        kb = (await owner.post("/v1/knowledge-bases", json={"display_name": "Exec"})).json()["kb_id"]
    # KB in ROOT (parent) unit, Restricted floor (3).
    await wiring.kb_store().set_governance(tenant, kb, unit_id=None, classification_floor=3)

    # child-unit admin cleared only Internal -> ceiling blocks (404).
    child_admin = _client(_bearer(_jwt(tenant, ["admin"], {"clearance": "internal", "unit": "child"})))
    async with child_admin:
        r = await child_admin.post(f"/v1/kb/{kb}/query", json={"query": "hi"})
        assert r.status_code == 404, r.text


async def test_assignment_rejects_clearance_escalation(inmemory: None) -> None:
    """update_member: an admin cleared Internal cannot grant a member a Confidential clearance
    (no handing out access you don't hold)."""
    tenant = "zzzassign"
    # bootstrap: owner (restricted) adds two members
    owner = _client(_bearer(_jwt(tenant, ["owner", "admin"], {"clearance": "restricted"})))
    async with owner:
        r = await owner.post("/v1/org/members", json={"email": "m@x.com", "name": "M", "roles": ["reader"]})
        assert r.status_code == 201, r.text

    # A LOW-clearance admin tries to grant Confidential -> 403.
    low_admin = _client(_bearer(_jwt(tenant, ["admin"], {"clearance": "internal"})))
    async with low_admin:
        r = await low_admin.patch(
            "/v1/org/members/m@x.com",
            json={"roles": ["reader"], "kb_access": [], "clearance": "confidential"},
        )
        assert r.status_code == 403, r.text


async def test_assignment_rejects_ungoverned_unit(inmemory: None) -> None:
    """update_member: a unit admin cannot place a member into a unit it does not govern (sibling)."""
    from svc_control_plane import wiring
    from svc_control_plane.org_units import OrgUnit

    tenant = "zzzgov"
    store = wiring.org_unit_store()
    await store.ensure_root(tenant)
    await store.create(OrgUnit(unit_id="ga", tenant_id=tenant, name="GA", parent_id=tenant))
    await store.create(OrgUnit(unit_id="gb", tenant_id=tenant, name="GB", parent_id=tenant))

    owner = _client(_bearer(_jwt(tenant, ["owner", "admin"], {"clearance": "restricted"})))
    async with owner:
        assert (
            await owner.post("/v1/org/members", json={"email": "u@x.com", "name": "U", "roles": ["reader"]})
        ).status_code == 201

    # admin scoped to unit "ga" (UNIT_ADMIN of ga) tries to assign member into sibling "gb".
    ga_admin = _client(_bearer(_jwt(tenant, ["admin"], {"clearance": "restricted", "unit": "ga"})))
    async with ga_admin:
        r = await ga_admin.patch(
            "/v1/org/members/u@x.com",
            json={"roles": ["reader"], "kb_access": [], "clearance": "public", "unit_id": "gb"},
        )
        assert r.status_code == 403, r.text
