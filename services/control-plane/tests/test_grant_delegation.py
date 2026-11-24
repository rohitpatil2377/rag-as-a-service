"""Transitive federation grants (A->B->C) — bounded trust + chain resolution.

Unit-level: the pure attenuation rule (a re-delegation may only NARROW) and the store's chain
walk / transitive revocation, with no app or stores beyond the in-memory grant twin.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from svc_control_plane.grants import (
    DelegationError,
    InMemoryGrantStore,
    attenuate,
    sign_grant,
    verify_grant,
)

from rag_schemas import Grant

_NOW = datetime(2026, 8, 5, tzinfo=UTC)


def _grant(
    gid: str,
    iss: str,
    sub: str,
    topics: list[str],
    kbs: list[str],
    *,
    mx: int = 5,
    parent: str | None = None,
    days: int = 180,
) -> Grant:
    g = Grant(
        grant_id=gid,
        issuer_dept_id=iss,
        subject_dept_id=sub,
        topics=topics,
        kb_ids=kbs,
        max_results_per_query=mx,
        not_before=_NOW,
        expires_at=_NOW + timedelta(days=days),
        signed_token="",
        created_at=_NOW,
        parent_grant_id=parent,
    )
    return g.model_copy(update={"signed_token": sign_grant(g)})


_PARENT = _grant("gAB", "A", "B", ["finance", "hr"], ["A__kb1", "A__kb2"], mx=10)


def test_a_narrowing_delegation_is_accepted() -> None:
    child = _grant("gBC", "B", "C", ["finance"], ["A__kb1"], mx=5, parent="gAB")
    attenuate(_PARENT, child)  # no raise


def test_the_parent_link_is_signed_and_direct_grants_stay_byte_identical() -> None:
    child = _grant("gBC", "B", "C", ["finance"], ["A__kb1"], parent="gAB")
    assert verify_grant(child), "a delegated grant's signature must cover its parent link"
    # A direct grant's signature is unchanged by the feature (parent omitted from canonical form).
    assert verify_grant(_PARENT)


@pytest.mark.parametrize(
    ("child", "why"),
    [
        (_grant("x", "B", "C", ["finance", "legal"], ["A__kb1"], parent="gAB"), "extra topic"),
        (_grant("x", "B", "C", ["finance"], ["A__kb1", "A__kb9"], parent="gAB"), "extra kb"),
        (_grant("x", "B", "C", ["finance"], ["A__kb1"], mx=20, parent="gAB"), "bigger cap"),
        (_grant("x", "C", "C", ["finance"], ["A__kb1"], parent="gAB"), "issuer is not the parent subject"),
        (_grant("x", "B", "C", ["finance"], ["A__kb1"], parent="gAB", days=999), "outlives the parent"),
    ],
)
def test_any_widening_is_rejected(child: Grant, why: str) -> None:
    with pytest.raises(DelegationError):
        attenuate(_PARENT, child)


def test_chain_resolves_root_last_and_transitive_revocation_kills_descendants() -> None:
    async def _run() -> None:
        store = InMemoryGrantStore()
        ab = _PARENT
        bc = _grant("gBC", "B", "C", ["finance"], ["A__kb1"], parent="gAB")
        cd = _grant("gCD", "C", "D", ["finance"], ["A__kb1"], mx=3, parent="gBC")
        for g in (ab, bc, cd):
            await store.create(g)

        assert [g.grant_id for g in await store.resolve_chain("gCD")] == ["gCD", "gBC", "gAB"]
        assert await store.is_chain_active("gCD")
        assert [g.grant_id for g in await store.list_children("gAB")] == ["gBC"]

        # Revoke the ROOT — every descendant's chain goes inactive (transitive revocation),
        # even though the leaf grant itself was never touched.
        await store.revoke("gAB", "A")
        assert not await store.is_chain_active("gCD")
        assert not await store.is_chain_active("gBC")
        # The leaf's OWN active flag is still true — only the chain check catches the root kill.
        leaf = await store.get("gCD")
        assert leaf is not None and leaf.is_active

    asyncio.run(_run())
