"""E2E: self-serve accounts + organisations (multi-tenant SaaS auth)."""

from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.e2e


def _app() -> object:
    from rag_api.app import create_app

    return create_app()


async def _c() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()), base_url="http://t")  # type: ignore[arg-type]


async def _signup(c: httpx.AsyncClient, org: str, name: str, email: str, pw: str) -> httpx.Response:
    return await c.post("/v1/auth/signup", json={"org_name": org, "name": name, "email": email, "password": pw})


def _bearer(resp_json: dict) -> dict[str, str]:
    return {"Authorization": f"Bearer {resp_json['access_token']}"}


async def test_signup_login_me_and_isolation(inmemory: None) -> None:
    async with await _c() as c:
        # Signup creates an org + owner and returns a usable token.
        r = await _signup(c, "Acme Bank", "Rohit", "Rohit@Acme.com", "s3cretpw!!")
        assert r.status_code == 201, r.text
        s = r.json()
        assert s["org_id"] == "acme-bank" and s["role"] == "owner" and s["email"] == "rohit@acme.com"
        auth = _bearer(s)

        # /auth/me reflects the account; the org's default KB works.
        me = (await c.get("/v1/auth/me", headers=auth)).json()
        assert me["org_name"] == "Acme Bank" and me["org_id"] == "acme-bank"
        assert (await c.get("/v1/kb/acme-bank/documents", headers=auth)).status_code == 200

        # Duplicate email is rejected; wrong password fails login.
        assert (await _signup(c, "X", "Y", "rohit@acme.com", "another88")).status_code == 422
        bad = await c.post("/v1/auth/login", json={"email": "rohit@acme.com", "password": "wrong"})
        assert bad.status_code == 401

        # Correct login returns a token for the same org.
        li = (await c.post("/v1/auth/login", json={"email": "rohit@acme.com", "password": "s3cretpw!!"})).json()
        assert li["org_id"] == "acme-bank" and li["role"] == "owner"


async def test_second_org_is_isolated_and_unique(inmemory: None) -> None:
    async with await _c() as c:
        a = (await _signup(c, "Globex", "A", "a@g.com", "pwpwpwpw")).json()
        b = (await _signup(c, "Globex", "B", "b@g.com", "pwpwpwpw")).json()
        assert a["org_id"] == "globex" and b["org_id"] == "globex-2"  # names collide → unique ids

        # Org A cannot read Org B's KB.
        r = await c.get(f"/v1/kb/{b['org_id']}/documents", headers=_bearer(a))
        assert r.status_code == 403


async def test_owner_can_add_member_viewer_cannot(inmemory: None) -> None:
    async with await _c() as c:
        owner = (await _signup(c, "Initech", "Own", "own@i.com", "pwpwpwpw")).json()
        oauth = _bearer(owner)

        r = await c.post(
            "/v1/org/members",
            headers=oauth,
            json={"name": "Member", "email": "m@i.com", "password": "memberpw1", "role": "member"},
        )
        assert r.status_code == 201

        members = {m["email"] for m in (await c.get("/v1/org/members", headers=oauth)).json()}
        assert members == {"own@i.com", "m@i.com"}

        # A new member starts INVITED and cannot sign in until the account is activated —
        # a temporary password is not a working credential on its own.
        blocked = await c.post("/v1/auth/login", json={"email": "m@i.com", "password": "memberpw1"})
        assert blocked.status_code == 401, "an invited member must not be able to sign in"

        link = (await c.post("/v1/org/members/m@i.com/activate", headers=oauth)).json()
        token = link["link"].rsplit("token=", 1)[-1] if "link" in link else link["token"]
        assert (
            await c.post("/v1/auth/set-password", json={"token": token, "password": "memberpw1"})
        ).status_code == 204

        # The activated member (no admin/owner cap) still can't add members.
        li = await c.post("/v1/auth/login", json={"email": "m@i.com", "password": "memberpw1"})
        assert li.status_code == 200, li.text
        mtok = li.json()["access_token"]
        r = await c.post(
            "/v1/org/members",
            headers={"Authorization": f"Bearer {mtok}"},
            json={"name": "X", "email": "x@i.com", "password": "pwpwpwpw", "role": "member"},
        )
        assert r.status_code == 403
