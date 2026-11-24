"""Unit tests: Federated RAG grant signing/verification + the in-memory grant store."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from svc_control_plane.grants import (
    InMemoryGrantStore,
    sign_grant,
    verify_grant,
)

from rag_schemas import Grant


def _grant(**over: object) -> Grant:
    now = datetime(2026, 8, 3, tzinfo=UTC)
    base: dict[str, object] = {
        "grant_id": "grant_test01",
        "issuer_dept_id": "hr",
        "subject_dept_id": "finance",
        "topics": ["leave-policy", "headcount"],
        "kb_ids": ["hr__policies"],
        "max_results_per_query": 5,
        "not_before": now - timedelta(days=1),
        "expires_at": now + timedelta(days=180),
        "signed_token": "",
        "created_at": now,
    }
    base.update(over)
    return Grant(**base)


def test_sign_then_verify_roundtrips() -> None:
    g = _grant()
    g = g.model_copy(update={"signed_token": sign_grant(g)})
    assert verify_grant(g) is True


def test_tampered_claims_fail_verification() -> None:
    g = _grant()
    g = g.model_copy(update={"signed_token": sign_grant(g)})
    # Widen the granted topics after signing — the signature must no longer match.
    tampered = g.model_copy(update={"topics": ["leave-policy", "headcount", "salaries"]})
    assert verify_grant(tampered) is False


def test_tampered_kb_ids_fail_verification() -> None:
    g = _grant()
    g = g.model_copy(update={"signed_token": sign_grant(g)})
    tampered = g.model_copy(update={"kb_ids": ["hr__policies", "finance__ledger"]})
    assert verify_grant(tampered) is False


def test_garbage_token_fails_verification() -> None:
    g = _grant(signed_token="not-base64-!!")
    assert verify_grant(g) is False


def test_topic_order_does_not_affect_signature() -> None:
    """Canonical form sorts topics/kb_ids, so a differently-ordered but equal grant verifies."""
    g = _grant(topics=["leave-policy", "headcount"])
    token = sign_grant(g)
    reordered = g.model_copy(update={"topics": ["headcount", "leave-policy"], "signed_token": token})
    assert verify_grant(reordered) is True


def test_is_active_expiry_and_revocation() -> None:
    now = datetime.now(UTC)
    live = _grant(not_before=now - timedelta(days=1), expires_at=now + timedelta(days=1))
    assert live.is_active is True

    expired = _grant(not_before=now - timedelta(days=2), expires_at=now - timedelta(days=1))
    assert expired.is_active is False

    not_yet = _grant(not_before=now + timedelta(days=1), expires_at=now + timedelta(days=2))
    assert not_yet.is_active is False

    revoked = live.model_copy(update={"revoked_at": now})
    assert revoked.is_active is False


async def test_store_create_and_get() -> None:
    store = InMemoryGrantStore()
    g = _grant()
    await store.create(g)
    assert await store.get("grant_test01") == g


async def test_store_rejects_duplicate_id() -> None:
    store = InMemoryGrantStore()
    await store.create(_grant())
    with pytest.raises(ValueError, match="already exists"):
        await store.create(_grant())


async def test_store_list_issued_and_received() -> None:
    store = InMemoryGrantStore()
    await store.create(_grant(grant_id="g1", issuer_dept_id="hr", subject_dept_id="finance"))
    await store.create(_grant(grant_id="g2", issuer_dept_id="hr", subject_dept_id="legal"))
    await store.create(_grant(grant_id="g3", issuer_dept_id="legal", subject_dept_id="finance"))

    issued_by_hr = await store.list_issued("hr")
    assert {g.grant_id for g in issued_by_hr} == {"g1", "g2"}

    received_by_finance = await store.list_received("finance")
    assert {g.grant_id for g in received_by_finance} == {"g1", "g3"}


async def test_store_revoke_only_by_issuer() -> None:
    store = InMemoryGrantStore()
    await store.create(_grant(grant_id="g1", issuer_dept_id="hr", subject_dept_id="finance"))

    # A different department cannot revoke HR's grant.
    with pytest.raises(PermissionError):
        await store.revoke("g1", "finance")

    # The issuer can.
    assert await store.revoke("g1", "hr") is True
    g = await store.get("g1")
    assert g is not None and g.revoked_at is not None and g.is_active is False


async def test_store_revoke_unknown_returns_false() -> None:
    store = InMemoryGrantStore()
    assert await store.revoke("nope", "hr") is False


async def test_store_get_active_excludes_revoked_and_expired() -> None:
    store = InMemoryGrantStore()
    now = datetime.now(UTC)
    await store.create(_grant(grant_id="live", issuer_dept_id="hr", subject_dept_id="finance"))
    await store.create(
        _grant(
            grant_id="expired",
            issuer_dept_id="hr",
            subject_dept_id="finance",
            not_before=now - timedelta(days=3),
            expires_at=now - timedelta(days=1),
        )
    )
    await store.create(_grant(grant_id="revoked", issuer_dept_id="hr", subject_dept_id="finance"))
    await store.revoke("revoked", "hr")

    active = await store.get_active("hr", "finance")
    assert {g.grant_id for g in active} == {"live"}
