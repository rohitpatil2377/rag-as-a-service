"""Egress-host approval workflow: request → admin approve → allowlisted (guard never disabled)."""

from __future__ import annotations

import os

import pytest
from svc_control_plane import egress_requests
from svc_control_plane.egress_requests import InMemoryEgressRequestStore, normalize_host

from rag_common.config import get_settings
from rag_common.errors import NotFoundError, ValidationError


@pytest.fixture(autouse=True)
def _clean(inmemory: None) -> None:
    get_settings.cache_clear()


def test_normalize_host_from_url_or_bare() -> None:
    assert normalize_host("https://docs.example.com/latest/") == "docs.example.com"
    assert normalize_host("docs.example.com") == "docs.example.com"
    assert normalize_host("obj.sovereign.local:9000/bucket") == "obj.sovereign.local"
    with pytest.raises(ValidationError):
        normalize_host("")


async def test_request_then_approve_allowlists_host() -> None:
    store = InMemoryEgressRequestStore()
    req = await egress_requests.submit(
        store,
        host_or_url="https://docs.example.com/latest/",
        tenant_id="acme",
        requested_by="user@acme.example",
        kb_id="acme__docs",
        reason="product docs",
    )
    assert req.status == "pending" and req.host == "docs.example.com"
    assert "docs.example.com" not in get_settings().egress_allowlist  # not yet

    decided = await egress_requests.decide(store, req_id=req.id, approve=True, decided_by="owner@acme.example")
    assert decided.status == "approved" and decided.decided_by == "owner@acme.example"
    assert "docs.example.com" in get_settings().egress_allowlist  # now allowlisted, guard still on


async def test_reject_does_not_allowlist() -> None:
    store = InMemoryEgressRequestStore()
    req = await egress_requests.submit(
        store,
        host_or_url="evil.example.com",
        tenant_id="acme",
        requested_by="u",
        kb_id=None,
        reason="",
    )
    await egress_requests.decide(store, req_id=req.id, approve=False, decided_by="owner")
    assert "evil.example.com" not in get_settings().egress_allowlist


async def test_duplicate_request_returns_existing() -> None:
    store = InMemoryEgressRequestStore()
    a = await egress_requests.submit(
        store, host_or_url="dup.local", tenant_id="t", requested_by="u", kb_id=None, reason=""
    )
    b = await egress_requests.submit(
        store, host_or_url="https://dup.local/x", tenant_id="t", requested_by="u", kb_id=None, reason=""
    )
    assert a.id == b.id  # same active request, not a second row
    assert len(await store.list()) == 1


async def test_decide_unknown_request_raises() -> None:
    with pytest.raises(NotFoundError):
        await egress_requests.decide(InMemoryEgressRequestStore(), req_id="nope", approve=True, decided_by="owner")


async def test_load_approved_reapplies_hosts() -> None:
    store = InMemoryEgressRequestStore()
    req = await egress_requests.submit(
        store, host_or_url="approved.local", tenant_id="t", requested_by="u", kb_id=None, reason=""
    )
    await egress_requests.decide(store, req_id=req.id, approve=True, decided_by="owner")
    # Simulate a restart: clear the env override, then reload from the store.
    os.environ.pop("RAG_ALLOWED_EGRESS_HOSTS", None)
    get_settings.cache_clear()
    await egress_requests.load_approved(store)
    assert "approved.local" in get_settings().egress_allowlist


async def test_revoke_removes_host_from_allowlist() -> None:
    store = InMemoryEgressRequestStore()
    req = await egress_requests.submit(
        store, host_or_url="temp.example.com", tenant_id="t", requested_by="u", kb_id=None, reason=""
    )
    await egress_requests.decide(store, req_id=req.id, approve=True, decided_by="owner")
    assert "temp.example.com" in get_settings().egress_allowlist
    revoked = await egress_requests.revoke(store, req_id=req.id, decided_by="owner")
    assert revoked.status == "revoked"
    assert "temp.example.com" not in get_settings().egress_allowlist  # blocked again
