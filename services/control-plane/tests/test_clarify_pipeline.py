"""Gate tests for grounded query disambiguation (Layer-2 item 4) — the pipeline attach path.

Covers the risk-mitigation contract: flag-OFF is byte-identical; options are attached only on
genuine ambiguity (abstain or low-confidence grounded) with >= min distinct clusters; a confident
single-topic hit and a single-cluster retrieval both get NO options; and attaching options never
touches the grounded-or-abstain invariant.
"""

from __future__ import annotations

import pytest
from svc_control_plane.pipeline import _attach_clarifications

from rag_common.config import get_settings
from rag_schemas import Citation, GroundedAnswer, RetrievalQuery, RetrievalResult, RetrievedSpan
from rag_schemas.common import AnswerStatus


@pytest.fixture
def clarify_on(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable the (default-OFF) clarify flag with the heuristic backend for a test."""
    monkeypatch.setenv("RAG_CLARIFY_ENABLED", "true")
    get_settings.cache_clear()


def _span(doc_id: str, text: str, score: float, section: str | None = None) -> RetrievedSpan:
    return RetrievedSpan(
        chunk_id=f"{doc_id}-c", doc_id=doc_id, text=text, score=score, scorer="lexical", section=section
    )


def _two_clusters() -> list[RetrievedSpan]:
    return [
        _span("doc-leave", "Annual leave accrues at 1.5 days per month.", 0.7, section="Leave Policy"),
        _span("doc-sla", "The uptime target is 99.9 percent.", 0.5, section="Service Levels"),
    ]


def _retrieval(spans: list[RetrievedSpan], max_score: float) -> RetrievalResult:
    return RetrievalResult(query=RetrievalQuery(tenant_id="t", text="policy"), spans=spans, max_score=max_score)


def _abstained() -> GroundedAnswer:
    return GroundedAnswer(
        tenant_id="t",
        query="policy",
        status=AnswerStatus.ABSTAINED,
        answer="I can't answer that.",
        citations=[],
        confidence=0.4,
        abstain_reason="low_confidence",
    )


def _grounded() -> GroundedAnswer:
    return GroundedAnswer(
        tenant_id="t",
        query="policy",
        status=AnswerStatus.GROUNDED,
        answer="Leave accrues monthly.[1]",
        citations=[Citation(chunk_id="doc-leave-c", doc_id="doc-leave", quote="leave", score=0.7)],
        confidence=0.7,
    )


async def test_flag_off_is_byte_identical(inmemory: None) -> None:
    # Default (flag OFF): no generator call, no field set, no change — the answer is untouched.
    answer = _abstained()
    before = answer.model_dump(mode="json")
    r = _retrieval(_two_clusters(), 0.4)
    out, n = await _attach_clarifications(answer, r.query, r)
    assert n == 0
    assert out is answer  # same object — nothing was copied
    assert out.model_dump(mode="json") == before
    assert out.clarification_options == []


async def test_abstained_with_two_clusters_offers_options(clarify_on: None) -> None:
    answer = _abstained()
    r = _retrieval(_two_clusters(), 0.4)
    out, n = await _attach_clarifications(answer, r.query, r)
    assert n >= 2
    assert out.clarification_options and all(o.endswith("?") for o in out.clarification_options)
    # Grounded-or-abstain invariant untouched: still ABSTAINED with zero citations + the message.
    assert out.status is AnswerStatus.ABSTAINED
    assert out.citations == []
    assert out.answer == answer.answer


async def test_grounded_confident_gets_no_options(clarify_on: None) -> None:
    # A confident grounded hit (max_score >= ceiling, default 0.75) is a clear single-topic query.
    r = _retrieval(_two_clusters(), 0.9)
    out, n = await _attach_clarifications(_grounded(), r.query, r)
    assert n == 0
    assert out.clarification_options == []


async def test_grounded_low_confidence_offers_options(clarify_on: None) -> None:
    answer = _grounded()
    r = _retrieval(_two_clusters(), 0.5)
    out, n = await _attach_clarifications(answer, r.query, r)
    assert n >= 2
    # Grounded answer still carries its real citation — options are supplementary, not a citation.
    assert out.status is AnswerStatus.GROUNDED
    assert out.citations == answer.citations


async def test_single_cluster_gets_no_options(clarify_on: None) -> None:
    spans = [
        _span("doc-sla", "The uptime target is 99.9 percent.", 0.5, section="Service Levels"),
        _span("doc-sla", "Credits apply below the target.", 0.45, section="Service Levels"),
    ]
    r = _retrieval(spans, 0.5)
    out, n = await _attach_clarifications(_abstained(), r.query, r)
    assert n == 0
    assert out.clarification_options == []


async def test_runaway_top_hit_gets_no_options(clarify_on: None) -> None:
    # A dominant top cluster (2nd cluster far weaker) is not genuinely ambiguous → no options.
    spans = [
        _span("doc-leave", "Annual leave accrues monthly.", 0.9, section="Leave Policy"),
        _span("doc-sla", "The uptime target is 99.9 percent.", 0.2, section="Service Levels"),
    ]
    r = _retrieval(spans, 0.5)
    out, n = await _attach_clarifications(_abstained(), r.query, r)
    assert n == 0
    assert out.clarification_options == []
