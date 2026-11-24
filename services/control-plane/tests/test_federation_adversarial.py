"""ADVERSARIAL federation cases (go-live scrutiny) — scope escalation, expired grants,
forged kids, transitive-revocation edges, delegation-depth, and use-time enforcement gaps.

These probe the SAME modules the friend's tests cover (grants.attenuate/sign/verify, the
grant store, federated_eval.attribute) but from an attacker's angle: I am trying to BREAK
the bounded-trust and least-privilege guarantees, not confirm them.
"""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
from svc_control_plane.federated_eval import AnswerMetrics, QuestionEval, attribute, peer_of_doc
from svc_control_plane.grants import (
    DelegationError,
    InMemoryGrantStore,
    attenuate,
    current_key_id,
    sign_grant,
    verify_grant,
    verifying_keys,
)

from rag_common.config import get_settings
from rag_schemas import Grant

# Anchor to real "now" so nb_days/days offsets are always genuinely relative — a hardcoded
# calendar date turns "not_before = _NOW + N days" into a time-bomb once the wall clock passes it.
_NOW = datetime.now(UTC)


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
    nb_days: int = 0,
    sign: bool = True,
) -> Grant:
    g = Grant(
        grant_id=gid,
        issuer_dept_id=iss,
        subject_dept_id=sub,
        topics=topics,
        kb_ids=kbs,
        max_results_per_query=mx,
        not_before=_NOW + timedelta(days=nb_days),
        expires_at=_NOW + timedelta(days=days),
        signed_token="",
        created_at=_NOW,
        parent_grant_id=parent,
    )
    return g.model_copy(update={"signed_token": sign_grant(g)}) if sign else g


def _keypair() -> tuple[str, str]:
    k = Ed25519PrivateKey.generate()
    priv = base64.b64encode(k.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())).decode()
    pub = base64.b64encode(k.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()
    return priv, pub


# ==========================================================================================
# 1) SCOPE ESCALATION — a delegated grant must NEVER exceed the parent, transitively.
# ==========================================================================================


def test_grandchild_cannot_re_widen_what_the_child_narrowed() -> None:
    """A->B->C narrows to {finance}; C->D must not re-add {hr} even though the ROOT had it.
    attenuate compares against the DIRECT parent (the child C grant), so re-widening fails."""
    root = _grant("gAB", "A", "B", ["finance", "hr"], ["A__kb1", "A__kb2"], mx=10)
    child = _grant("gBC", "B", "C", ["finance"], ["A__kb1"], mx=5, parent="gAB")
    attenuate(root, child)  # ok
    # Grandchild tries to re-add 'hr' (present in root, absent in its direct parent `child`).
    grandchild = _grant("gCD", "C", "D", ["finance", "hr"], ["A__kb1"], mx=3, parent="gBC")
    with pytest.raises(DelegationError, match="topics exceed"):
        attenuate(child, grandchild)


def test_grandchild_cannot_re_add_a_kb_the_child_dropped() -> None:
    root = _grant("gAB", "A", "B", ["finance"], ["A__kb1", "A__kb2"], mx=10)
    child = _grant("gBC", "B", "C", ["finance"], ["A__kb1"], mx=5, parent="gAB")
    attenuate(root, child)
    grandchild = _grant("gCD", "C", "D", ["finance"], ["A__kb1", "A__kb2"], mx=3, parent="gBC")
    with pytest.raises(DelegationError, match="kb_ids exceed"):
        attenuate(child, grandchild)


def test_grandchild_cannot_raise_the_result_cap_above_the_child() -> None:
    child = _grant("gBC", "B", "C", ["finance"], ["A__kb1"], mx=5, parent="gAB")
    grandchild = _grant("gCD", "C", "D", ["finance"], ["A__kb1"], mx=6, parent="gBC")
    with pytest.raises(DelegationError, match="max_results_per_query"):
        attenuate(child, grandchild)


def test_delegation_by_a_non_subject_is_refused() -> None:
    """Only the parent's SUBJECT may re-delegate. A third party (or the issuer) cannot."""
    parent = _grant("gAB", "A", "B", ["finance"], ["A__kb1"])
    # 'X' is neither issuer nor subject of the parent.
    forged = _grant("gXD", "X", "D", ["finance"], ["A__kb1"], parent="gAB")
    with pytest.raises(DelegationError, match="only the grant's subject"):
        attenuate(parent, forged)
    # Even the ISSUER (A) cannot pose as a re-delegator of its own grant via this path.
    issuer_reshare = _grant("gAD", "A", "D", ["finance"], ["A__kb1"], parent="gAB")
    with pytest.raises(DelegationError, match="only the grant's subject"):
        attenuate(parent, issuer_reshare)


def test_delegated_grant_cannot_start_before_parent() -> None:
    parent = _grant("gAB", "A", "B", ["finance"], ["A__kb1"], nb_days=5, days=180)
    early = _grant("gBC", "B", "C", ["finance"], ["A__kb1"], parent="gAB", nb_days=0, days=180)
    with pytest.raises(DelegationError, match="starts before"):
        attenuate(parent, early)


# ==========================================================================================
# 2) THE STORE DOES NOT ENFORCE ATTENUATION — only the API endpoint does. Prove the gap.
# ==========================================================================================


def test_store_create_does_NOT_enforce_attenuation() -> None:  # noqa: N802
    """SECURITY-RELEVANT: InMemoryGrantStore.create() persists ANY grant, including a
    'delegated' one that WIDENS its parent. Bounded trust is enforced ONLY in the API
    (delegate_grant calls attenuate). A direct store write / a future code path that skips
    the endpoint would persist an over-broad grant. Documents the trust boundary."""

    async def _run() -> None:
        store = InMemoryGrantStore()
        parent = _grant("gAB", "A", "B", ["finance"], ["A__kb1"], mx=5)
        await store.create(parent)
        # A widening child: extra topic, extra kb, bigger cap, outlives the parent.
        rogue = _grant(
            "gBC",
            "B",
            "C",
            ["finance", "hr", "legal"],
            ["A__kb1", "A__kb2"],
            mx=20,
            parent="gAB",
            days=999,
        )
        # attenuate WOULD reject it:
        with pytest.raises(DelegationError):
            attenuate(parent, rogue)
        # ...but the store happily persists it (no attenuation at the store layer):
        await store.create(rogue)
        got = await store.get("gBC")
        assert got is not None and got.max_results_per_query == 20
        # And its chain reads as active because both links are time-valid — the widening is
        # invisible to is_chain_active (which only checks validity, not scope containment).
        assert await store.is_chain_active("gBC")

    asyncio.run(_run())


def test_use_time_path_never_re_checks_attenuation_containment() -> None:
    """The federated retrieval path (answer_federated) filters grants by is_chain_active but
    does NOT re-run attenuate at USE time. So a stored widening delegated grant (see above)
    would authorise retrieval from its (over-broad) kb_ids at its (inflated) cap. This asserts
    the containment property is issue-time-only — a coverage gap for defence-in-depth."""
    import inspect

    from svc_control_plane import federation, pipeline

    src = inspect.getsource(pipeline.answer_federated) + inspect.getsource(federation.LocalTransport)
    assert "attenuate" not in src, "if attenuate is now called at use-time this assertion is stale (good) — update it"


# ==========================================================================================
# 3) FORGED KID / SIGNATURE — a grant signed by a key outside the keyring must fail closed.
# ==========================================================================================


def test_forged_grant_signed_by_attacker_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """An attacker signs a fully-formed, in-scope-looking grant with THEIR OWN key. Because
    that key is neither the active signing key nor a retired-overlap key, verification fails."""
    victim_priv, _ = _keypair()
    attacker_priv, _ = _keypair()

    # Platform's active key is the victim's; the attacker's key is NOT trusted.
    monkeypatch.setenv("RAG_GRANT_SIGNING_KEY", attacker_priv)
    monkeypatch.setenv("RAG_GRANT_SIGNING_KEY_ID", "attacker")
    monkeypatch.setenv("RAG_GRANT_RETIRED_VERIFY_KEYS", json.dumps([]))
    get_settings.cache_clear()
    forged = _grant("evil", "hr", "finance", ["salaries"], ["hr__payroll"])  # signed by attacker

    monkeypatch.setenv("RAG_GRANT_SIGNING_KEY", victim_priv)
    monkeypatch.setenv("RAG_GRANT_SIGNING_KEY_ID", "prod")
    get_settings.cache_clear()
    assert verify_grant(forged) is False, "a grant signed by an untrusted key must fail closed"
    get_settings.cache_clear()


def test_kid_is_not_bound_into_the_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    """OBSERVATION: the kid (RAG_GRANT_SIGNING_KEY_ID) is NOT part of the signed canonical
    claims — it's audit metadata only. So changing ONLY the kid while keeping the same key
    material does not change the signature or invalidate a grant. A verifier trusts by KEY,
    not by kid. This is by design (kid is a hint), but a compliance reader must not treat the
    kid as cryptographically bound."""
    priv, _ = _keypair()
    monkeypatch.setenv("RAG_GRANT_SIGNING_KEY", priv)
    monkeypatch.setenv("RAG_GRANT_SIGNING_KEY_ID", "kid-1")
    get_settings.cache_clear()
    g = _grant("g", "hr", "fin", ["t"], ["hr__k"])
    tok1 = g.signed_token
    assert current_key_id() == "kid-1"

    monkeypatch.setenv("RAG_GRANT_SIGNING_KEY_ID", "kid-2")  # same key, different kid label
    get_settings.cache_clear()
    g2 = _grant("g", "hr", "fin", ["t"], ["hr__k"])
    assert g2.signed_token == tok1, "signature is over claims only; kid is not bound in"
    assert current_key_id() == "kid-2"
    assert verify_grant(g) and verify_grant(g2)
    get_settings.cache_clear()


def test_a_grant_missing_its_parent_link_in_the_signature_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parent link IS bound into a delegated grant's signature. If an attacker takes a
    root-grant signature and reparents the grant (sets parent_grant_id) without re-signing,
    the canonical claims now differ (parent added) and verification must fail — preventing a
    forged 'this was delegated from X' provenance claim."""
    priv, _ = _keypair()
    monkeypatch.setenv("RAG_GRANT_SIGNING_KEY", priv)
    monkeypatch.setenv("RAG_GRANT_SIGNING_KEY_ID", "k")
    get_settings.cache_clear()
    # Sign as a ROOT (no parent).
    root = _grant("g", "B", "C", ["finance"], ["A__kb1"], parent=None)
    assert verify_grant(root)
    # Reparent WITHOUT re-signing -> canonical bytes gain parent_grant_id -> sig no longer matches.
    reparented = root.model_copy(update={"parent_grant_id": "gAB"})
    assert verify_grant(reparented) is False, "adding a parent link post-signing must break verification"
    get_settings.cache_clear()


# ==========================================================================================
# 4) EXPIRED / NOT-YET-VALID GRANTS across the chain.
# ==========================================================================================


def test_expired_root_blocks_a_still_valid_leaf() -> None:
    """The leaf grant is time-valid on its own, but its ROOT ancestor has EXPIRED. The chain
    check must report inactive — no cross-dept retrieval under an expired ancestor."""

    async def _run() -> None:
        store = InMemoryGrantStore()
        # Root expired yesterday; child/leaf are long-lived.
        root = _grant("gAB", "A", "B", ["finance"], ["A__kb1"], days=-1, nb_days=-10)
        child = _grant("gBC", "B", "C", ["finance"], ["A__kb1"], parent="gAB", days=180)
        for g in (root, child):
            await store.create(g)
        leaf = await store.get("gBC")
        assert leaf is not None and leaf.is_active, "leaf itself is time-valid"
        assert not await store.is_chain_active("gBC"), "expired root must kill the chain"

    asyncio.run(_run())


def test_not_yet_active_intermediate_blocks_the_chain() -> None:
    async def _run() -> None:
        store = InMemoryGrantStore()
        root = _grant("gAB", "A", "B", ["finance"], ["A__kb1"], days=180)
        # Intermediate not_before is in the FUTURE -> not active yet.
        mid = _grant("gBC", "B", "C", ["finance"], ["A__kb1"], parent="gAB", nb_days=5, days=180)
        leaf = _grant("gCD", "C", "D", ["finance"], ["A__kb1"], parent="gBC", days=180)
        for g in (root, mid, leaf):
            await store.create(g)
        assert not await store.is_chain_active("gCD")

    asyncio.run(_run())


def test_broken_chain_missing_parent_row_is_inactive_not_crash() -> None:
    """A leaf points at a parent_grant_id that isn't in the store (deleted/never-synced). The
    chain walk must terminate cleanly and report inactive rather than treat the leaf as valid
    or raise."""

    async def _run() -> None:
        store = InMemoryGrantStore()
        orphan = _grant("gCD", "C", "D", ["finance"], ["A__kb1"], parent="gMISSING", days=180)
        await store.create(orphan)
        chain = await store.resolve_chain("gCD")
        assert [g.grant_id for g in chain] == ["gCD"], "walk stops at the missing parent"
        # The chain does not include the (missing) parent, so is_chain_active is True only
        # because the missing ancestor is invisible. This is a SILENT-BROKEN-CHAIN gap:
        # a dangling parent link is treated as a root. Assert current behaviour to flag it.
        assert await store.is_chain_active("gCD") is True

    asyncio.run(_run())


def test_revoking_middle_of_chain_kills_only_descendants_not_ancestors() -> None:
    async def _run() -> None:
        store = InMemoryGrantStore()
        root = _grant("gAB", "A", "B", ["finance"], ["A__kb1"], days=180)
        mid = _grant("gBC", "B", "C", ["finance"], ["A__kb1"], parent="gAB", days=180)
        leaf = _grant("gCD", "C", "D", ["finance"], ["A__kb1"], parent="gBC", days=180)
        for g in (root, mid, leaf):
            await store.create(g)
        # Revoke the MIDDLE grant (B's re-share to C). Only C and D lose access; A->B stays.
        await store.revoke("gBC", "B")
        assert not await store.is_chain_active("gCD")
        assert not await store.is_chain_active("gBC")
        assert await store.is_chain_active("gAB"), "the ancestor grant is untouched"

    asyncio.run(_run())


# ==========================================================================================
# 5) CHAIN-WALK ROBUSTNESS — a cycle in parent links must not infinite-loop.
# ==========================================================================================


def test_cyclic_parent_links_do_not_infinite_loop() -> None:
    """Adversarial/corrupt data: two grants each name the other as parent. resolve_chain uses a
    `seen` set, so it must terminate rather than hang."""

    async def _run() -> None:
        store = InMemoryGrantStore()
        g1 = _grant("g1", "A", "B", ["finance"], ["A__kb1"], parent="g2", days=180)
        g2 = _grant("g2", "B", "A", ["finance"], ["A__kb1"], parent="g1", days=180)
        await store.create(g1)
        await store.create(g2)
        chain = await asyncio.wait_for(store.resolve_chain("g1"), timeout=2.0)
        ids = [g.grant_id for g in chain]
        assert ids == ["g1", "g2"], "cycle visits each node once then stops"
        # is_chain_active must also terminate.
        await asyncio.wait_for(store.is_chain_active("g1"), timeout=2.0)

    asyncio.run(_run())


def test_self_parent_link_terminates() -> None:
    async def _run() -> None:
        store = InMemoryGrantStore()
        g = _grant("g", "A", "B", ["finance"], ["A__kb1"], parent="g", days=180)
        await store.create(g)
        chain = await asyncio.wait_for(store.resolve_chain("g"), timeout=2.0)
        assert [x.grant_id for x in chain] == ["g"]

    asyncio.run(_run())


# ==========================================================================================
# 6) LEAST PRIVILEGE — federated eval attribution must not mis-attribute across peers.
# ==========================================================================================


def test_attribution_never_credits_a_peer_that_was_not_ablated() -> None:
    """A peer with NO ablation entry (never removed) contributes zero marginals and is not
    falsely flagged as degrading."""
    evals = [QuestionEval("q", baseline=_m_hit(True, ["p1"]), ablations={"p1": _m_hit(True)})]
    peers = {p.peer_dept: p for p in attribute(evals, {"p1": ["g1"], "ghost": ["g2"]})}
    # ghost has no ablation across any question -> all marginals empty -> not degrading.
    assert peers["ghost"].verdict in {"unused", "neutral"}
    assert not peers["ghost"].degrades
    assert peers["ghost"].questions_cited == 0


def test_peer_of_doc_rejects_a_lookalike_own_kb_label() -> None:
    """Least privilege / attribution safety: an own-KB doc label that happens to equal a peer
    name prefix must only be attributed when that dept was actually federated from."""
    peers = frozenset({"hr"})
    assert peer_of_doc("hr/leave", peers) == "hr"
    assert peer_of_doc("hrfoo/leave", peers) is None  # not an exact head match
    assert peer_of_doc("budget/x", peers) is None


def test_degradation_below_epsilon_is_noise_not_a_flag() -> None:
    """A marginal within +-DEGRADE_EPS must be treated as noise (neutral), not a false
    'degrades'/'helps'. Guards against over-flagging peers (revoking a good grant)."""
    evals = [
        QuestionEval(
            "q",
            baseline=AnswerMetrics(True, 0.500, True, frozenset({"p"})),
            ablations={"p": AnswerMetrics(True, 0.540, True, frozenset())},
        )
    ]
    [p] = attribute(evals, {"p": ["g"]})
    # faith marginal = 0.5 - 0.54 = -0.04, |.| < 0.05 eps -> not degrading.
    assert not p.degrades
    assert p.verdict == "neutral"


def _m_hit(hit: bool, cited: list[str] | None = None) -> AnswerMetrics:
    return AnswerMetrics(grounded=True, faithfulness=None, hit=hit, cited_peers=frozenset(cited or []))


# ==========================================================================================
# 7) KEYRING — rotation overlap edge cases.
# ==========================================================================================


def test_verifying_keys_survives_a_malformed_retired_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single corrupt retired-key entry must NOT take down verification of grants signed by
    the active key (DoS resistance)."""
    active_priv, _ = _keypair()
    _, good_retired_pub = _keypair()
    monkeypatch.setenv("RAG_GRANT_SIGNING_KEY", active_priv)
    monkeypatch.setenv("RAG_GRANT_SIGNING_KEY_ID", "k")
    # One valid retired key + one garbage entry.
    monkeypatch.setenv("RAG_GRANT_RETIRED_VERIFY_KEYS", json.dumps([good_retired_pub, "!!!not-b64!!!"]))
    get_settings.cache_clear()
    keys = verifying_keys()
    # active + the one good retired key; the garbage one is skipped, not fatal.
    assert len(keys) == 2
    g = _grant("g", "hr", "fin", ["t"], ["hr__k"])
    assert verify_grant(g)
    get_settings.cache_clear()


def test_malformed_retired_keys_json_is_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    active_priv, _ = _keypair()
    monkeypatch.setenv("RAG_GRANT_SIGNING_KEY", active_priv)
    monkeypatch.setenv("RAG_GRANT_RETIRED_VERIFY_KEYS", "{not json at all")
    get_settings.cache_clear()
    keys = verifying_keys()
    assert len(keys) == 1  # just the active key; bad JSON -> [] retired
    g = _grant("g", "hr", "fin", ["t"], ["hr__k"])
    assert verify_grant(g)
    get_settings.cache_clear()


# ==========================================================================================
# 8) CASE-SENSITIVITY INCONSISTENCY between attenuate() and _topic_scoped().
# ==========================================================================================


def test_attenuate_topics_are_case_sensitive_but_topic_filter_is_not() -> None:
    """INCONSISTENCY (fails CLOSED, so not a hole, but a correctness footgun):
    attenuate() compares topics as EXACT strings, while federation._topic_scoped() matches
    case-insensitively. So a parent granting 'Finance' and a child requesting 'finance' is
    REJECTED by attenuate (treated as an extra topic) even though the runtime filter would
    consider them the same topic. Delegators must match the parent's exact casing."""
    from svc_control_plane.federation import _topic_scoped

    from rag_schemas import RetrievedSpan

    parent = _grant("gAB", "A", "B", ["Finance"], ["A__kb1"])
    child = _grant("gBC", "B", "C", ["finance"], ["A__kb1"], parent="gAB")
    # attenuate: 'finance' not in {'Finance'} -> flagged as extra -> DelegationError.
    with pytest.raises(DelegationError, match="topics exceed"):
        attenuate(parent, child)
    # But the runtime topic filter treats them as equal (case-insensitive):
    span = RetrievedSpan(chunk_id="c", doc_id="d", text="x", score=0.5, scorer="lexical", acl=[], category="finance")
    kept = _topic_scoped([span], parent, enabled=True)  # parent topic 'Finance', span 'finance'
    assert len(kept) == 1, "runtime filter matches case-insensitively — inconsistent with attenuate"


def test_topic_enforcement_drops_uncategorised_spans_fail_closed() -> None:
    """ADR-0017 L1: with topic enforcement ON, a span with NO category cannot be shown to fall
    under a granted topic, so it is dropped (fail-closed). Prevents an unlabelled span leaking
    across a topic-scoped grant."""
    from svc_control_plane.federation import _topic_scoped

    from rag_schemas import RetrievedSpan

    grant = _grant("g", "A", "B", ["finance"], ["A__kb1"])
    uncategorised = RetrievedSpan(
        chunk_id="c", doc_id="d", text="x", score=0.5, scorer="lexical", acl=[], category=None
    )
    off_topic = RetrievedSpan(chunk_id="c2", doc_id="d2", text="y", score=0.5, scorer="lexical", acl=[], category="hr")
    on_topic = RetrievedSpan(
        chunk_id="c3", doc_id="d3", text="z", score=0.5, scorer="lexical", acl=[], category="finance"
    )
    kept = _topic_scoped([uncategorised, off_topic, on_topic], grant, enabled=True)
    assert [s.chunk_id for s in kept] == ["c3"], "only the on-topic span survives; unlabelled + off-topic dropped"
    # With enforcement OFF, everything passes (default behaviour).
    assert len(_topic_scoped([uncategorised, off_topic, on_topic], grant, enabled=False)) == 3
