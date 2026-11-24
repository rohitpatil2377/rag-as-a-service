"""Semantic-cache unit tests — focus on cross-replica invalidation coherence (scale/HA)."""

from __future__ import annotations

import pytest
from svc_control_plane import semantic_cache
from svc_control_plane.semantic_cache import SemanticCache, still_passes

from rag_common.coordination import InMemoryCoordinator
from rag_schemas import Citation, GroundedAnswer
from rag_schemas.common import AnswerStatus


def _grounded() -> GroundedAnswer:
    return GroundedAnswer(
        tenant_id="kb1",
        query="q",
        status=AnswerStatus.GROUNDED,
        answer="42",
        confidence=0.9,
        citations=[Citation(chunk_id="c0", doc_id="d0", quote="x", score=0.9)],
    )


def test_hit_and_acl_scoping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(semantic_cache, "get_coordinator", InMemoryCoordinator)
    c = SemanticCache()
    c.put("kb1", "q", [1.0, 0.0], acl=["hr"], answer=_grounded())
    assert c.get("kb1", [1.0, 0.0], acl=["hr"]) is not None
    assert c.get("kb1", [1.0, 0.0], acl=["finance"]) is None  # different visibility → never replayed


def test_cross_replica_invalidation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replica A invalidates a KB; replica B must stop serving that KB's cached answer."""
    shared = InMemoryCoordinator()  # one coordinator == a shared Redis in production
    monkeypatch.setattr(semantic_cache, "get_coordinator", lambda: shared)
    a, b = SemanticCache(), SemanticCache()
    a.put("kb1", "q", [1.0, 0.0], acl=[], answer=_grounded())
    b.put("kb1", "q", [1.0, 0.0], acl=[], answer=_grounded())
    assert b.get("kb1", [1.0, 0.0], acl=[]) is not None

    a.invalidate("kb1")  # replica A ingests → bumps the shared version
    assert b.get("kb1", [1.0, 0.0], acl=[]) is None  # replica B sees the newer version and drops it


def test_never_caches_abstain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(semantic_cache, "get_coordinator", InMemoryCoordinator)
    c = SemanticCache()
    abstained = GroundedAnswer(
        tenant_id="kb1", query="q", status=AnswerStatus.ABSTAINED, answer="no basis", confidence=0.0
    )
    c.put("kb1", "q", [1.0, 0.0], acl=[], answer=abstained)
    assert c.get("kb1", [1.0, 0.0], acl=[]) is None


# --- a cache hit must still pass today's gate ---------------------------------------------------


def _answer(confidence: float, faithfulness: float | None = None) -> GroundedAnswer:
    from rag_schemas import AnswerStatus, Citation

    return GroundedAnswer(
        tenant_id="t",
        query="q",
        status=AnswerStatus.GROUNDED,
        answer="cached answer [1]",
        citations=[Citation(chunk_id="d::1", doc_id="d", quote="q", score=confidence)],
        confidence=confidence,
        faithfulness_score=faithfulness,
    )


def test_an_answer_below_todays_abstain_threshold_is_not_served() -> None:
    """The cache stores a GROUNDED answer and returns it verbatim, so the grounding gate is
    skipped on a hit. An operator who RAISES their KB's abstain threshold — the one control they
    have for "stop answering marginal questions" — would keep receiving the marginal answers it
    was meant to stop, for as long as the entry lived."""
    assert still_passes(_answer(0.20), abstain=0.15, faithful=0.7, flag=0.4)
    assert not still_passes(_answer(0.20), abstain=0.50, faithful=0.7, flag=0.4)


def test_an_answer_below_todays_faithfulness_floor_is_not_served() -> None:
    """Same for the faithfulness thresholds: the routing decision (grounded / flagged for review
    / abstained) was made when the answer was cached, against numbers that have since moved."""
    assert still_passes(_answer(0.9, 0.95), abstain=0.15, faithful=0.7, flag=0.4)
    assert not still_passes(_answer(0.9, 0.55), abstain=0.15, faithful=0.7, flag=0.4)  # now FLAGGED
    assert not still_passes(_answer(0.9, 0.30), abstain=0.15, faithful=0.7, flag=0.4)  # now ABSTAINED


def test_an_answer_with_no_faithfulness_score_is_judged_on_confidence_alone() -> None:
    """Faithfulness may be disabled for the KB, so a missing score is normal, not a failure."""
    assert still_passes(_answer(0.9, None), abstain=0.15, faithful=0.7, flag=0.4)


def test_a_different_top_k_is_a_different_question(monkeypatch: pytest.MonkeyPatch) -> None:
    """top_k decides how much evidence the generator is shown, so it changes the answer — an
    enumerated list answered from 5 passages is genuinely less complete than the same question
    answered from 15. Keyed only on (embedding, acl, clearance), the cache replayed the 5-passage
    answer to the 15-passage request: measured live on a real handbook, "what are the core values"
    returned a byte-identical 4-of-6 answer at both settings, which also silently defeats any
    attempt to tune top_k. Same bug class as the recency and metadata-filter leaks already closed
    in the pipeline."""
    monkeypatch.setattr(semantic_cache, "get_coordinator", InMemoryCoordinator)
    c = SemanticCache()
    c.put("kb1", "q", [1.0, 0.0], acl=["hr"], answer=_grounded(), top_k=5)
    assert c.get("kb1", [1.0, 0.0], acl=["hr"], top_k=5) is not None
    assert c.get("kb1", [1.0, 0.0], acl=["hr"], top_k=15) is None


def test_clearance_and_top_k_are_both_part_of_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adding top_k must not weaken the clearance scoping it sits beside: a Confidential answer
    still must never reach a lower-clearance caller, whatever top_k either of them used."""
    monkeypatch.setattr(semantic_cache, "get_coordinator", InMemoryCoordinator)
    c = SemanticCache()
    c.put("kb1", "q", [1.0, 0.0], acl=["hr"], answer=_grounded(), clearance=3, top_k=5)
    assert c.get("kb1", [1.0, 0.0], acl=["hr"], clearance=3, top_k=5) is not None
    assert c.get("kb1", [1.0, 0.0], acl=["hr"], clearance=0, top_k=5) is None
