"""Principal -> Subject mapping (Sub-step B): tier from roles, clearance/unit from claims."""

from __future__ import annotations

import pytest
from svc_control_plane.rbac_context import subject_of

from rag_common.auth import Principal
from rag_common.rbac import Clearance, Tier


def _p(roles: set[str], *, subject: str = "u@x", tenant: str = "acme", **attrs: object) -> Principal:
    return Principal(subject=subject, tenant_id=tenant, roles=frozenset(roles), attrs=dict(attrs))


def test_role_maps_to_tier() -> None:
    assert subject_of(_p({"owner", "admin"})).tier is Tier.TENANT_ADMIN
    assert subject_of(_p({"admin"})).tier is Tier.TENANT_ADMIN
    assert subject_of(_p({"reader", "ingestor"})).tier is Tier.MEMBER
    assert subject_of(_p({"reader"})).tier is Tier.VIEWER


def test_a_unit_scoped_admin_is_a_unit_admin() -> None:
    # admin whose token is scoped to a non-root unit -> UNIT_ADMIN (governs that subtree only).
    assert subject_of(_p({"admin"}, unit="branch")).tier is Tier.UNIT_ADMIN
    # admin at the tenant root stays TENANT_ADMIN.
    assert subject_of(_p({"admin"}, unit="acme")).tier is Tier.TENANT_ADMIN


def test_clearance_defaults_public_and_is_fail_closed() -> None:
    assert subject_of(_p({"reader"})).clearance is Clearance.PUBLIC  # no claim -> least privilege
    assert subject_of(_p({"reader"}, clearance=2)).clearance is Clearance.CONFIDENTIAL  # int level
    assert subject_of(_p({"reader"}, clearance="restricted")).clearance is Clearance.RESTRICTED  # name
    assert subject_of(_p({"reader"}, clearance="bad")).clearance is Clearance.PUBLIC  # unparsable -> fail closed


def test_unit_defaults_to_tenant_root() -> None:
    assert subject_of(_p({"reader"})).unit_id == "acme"
    assert subject_of(_p({"reader"}, unit="teller")).unit_id == "teller"


def test_platform_admin_allowlist_is_super_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_PLATFORM_ADMIN_EMAILS", "ops@acme.example, root@acme.example")
    get_settings.cache_clear()
    try:
        assert subject_of(_p({"reader"}, subject="ops@acme.example")).tier is Tier.SUPER_ADMIN
        assert subject_of(_p({"owner"}, subject="someone@else.com")).tier is Tier.TENANT_ADMIN
    finally:
        get_settings.cache_clear()
