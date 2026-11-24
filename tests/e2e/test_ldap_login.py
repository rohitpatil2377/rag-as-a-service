"""E2E: signing in via on-prem AD/LDAP provisions the user and issues a platform JWT."""

from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.e2e


async def test_login_via_ldap_provisions_user_with_mapped_role(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_api.app import create_app
    from rag_common import get_settings, ldap_auth
    from rag_common.ldap_auth import LdapIdentity

    monkeypatch.setenv("RAG_LDAP_ENABLED", "true")
    monkeypatch.setenv("RAG_LDAP_SERVER", "ldap://localhost:389")
    monkeypatch.setenv("RAG_LDAP_DEFAULT_ORG", "acmebank")
    monkeypatch.setenv("RAG_LDAP_ROLE_MAP", "Domain Admins:admin,Staff:reader")
    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost")
    get_settings.cache_clear()

    # Stand in for a real directory bind: valid creds return the AD identity + groups.
    def _fake(username: str, password: str, *, binder: object = None) -> LdapIdentity | None:
        if password == "AdPass1":
            return LdapIdentity(email="alex@bank.local", name="Alex Mercer", groups=["Staff", "Domain Admins"])
        return None

    monkeypatch.setattr(ldap_auth, "authenticate_ldap", _fake)

    app = create_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        # Wrong AD password → 401.
        bad = await c.post("/v1/auth/login", json={"email": "rohit", "password": "nope"})
        assert bad.status_code == 401

        # Correct AD password → provisioned + token, role mapped from "Domain Admins".
        ok = await c.post("/v1/auth/login", json={"email": "rohit", "password": "AdPass1"})
        assert ok.status_code == 200
        body = ok.json()
        assert body["org_id"] == "acmebank"
        assert body["role"] == "admin"
        token = body["access_token"]

        me = await c.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200
        assert me.json()["email"] == "alex@bank.local"

    get_settings.cache_clear()
