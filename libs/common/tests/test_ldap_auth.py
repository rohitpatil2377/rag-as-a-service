"""On-prem AD/LDAP bind authentication (banking pack)."""

from __future__ import annotations

import pytest

from rag_common import get_settings
from rag_common.ldap_auth import LdapIdentity, authenticate_ldap, role_for_groups


def test_role_for_groups_highest_privilege_wins() -> None:
    role_map = "Domain Admins:admin,RAG-Reviewers:reviewer,Staff:reader"
    assert role_for_groups(["Staff", "RAG-Reviewers"], role_map) == "reviewer"
    assert role_for_groups(["Staff", "Domain Admins", "RAG-Reviewers"], role_map) == "admin"
    assert role_for_groups(["Unknown"], role_map) == "reader"  # default
    assert role_for_groups([], role_map, default="ingestor") == "ingestor"


def _enable_ldap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LDAP_ENABLED", "true")
    monkeypatch.setenv("RAG_LDAP_SERVER", "ldap://localhost:389")
    monkeypatch.setenv("RAG_LDAP_USER_DN_TEMPLATE", "{username}@bank.local")
    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost")
    get_settings.cache_clear()


def test_bind_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_ldap(monkeypatch)

    def _binder(server: str, user_dn: str, password: str, base_dn: str) -> LdapIdentity | None:
        assert server == "ldap://localhost:389" and user_dn == "rohit@bank.local"
        if password == "correct":
            return LdapIdentity(email="rohit@bank.local", name="Rohit", groups=["Domain Admins"])
        return None  # bad bind

    assert authenticate_ldap("rohit", "correct", binder=_binder).email == "rohit@bank.local"  # type: ignore[union-attr]
    assert authenticate_ldap("rohit", "wrong", binder=_binder) is None
    assert authenticate_ldap("rohit", "", binder=_binder) is None  # never allow empty-password bind
    get_settings.cache_clear()


def test_disabled_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_LDAP_ENABLED", raising=False)
    get_settings.cache_clear()
    assert authenticate_ldap("rohit", "correct", binder=lambda *_: LdapIdentity("e", "n")) is None
    get_settings.cache_clear()


def test_non_allowlisted_server_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LDAP_ENABLED", "true")
    monkeypatch.setenv("RAG_LDAP_SERVER", "ldap://exfil.example.com:389")
    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost")
    monkeypatch.setenv("RAG_ENV", "prod")  # sovereign_enforce on
    get_settings.cache_clear()
    called = False

    def _binder(*_: str) -> LdapIdentity | None:
        nonlocal called
        called = True
        return LdapIdentity("e", "n")

    assert authenticate_ldap("rohit", "correct", binder=_binder) is None  # gate blocks before bind
    assert called is False
    get_settings.cache_clear()
