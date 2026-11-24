"""E2E faithfulness three-way routing (FR-E4), deterministic, no model in CI.

Uses a stub LLM gateway to control the generated answer text, plus the deterministic
LexicalFaithfulnessChecker, to exercise all three outcomes and confirm the FLAGGED path
lands in the existing HITL flow with the unsupported claims attached.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime

import pytest

from rag_schemas import DocumentEnvelope, RetrievalQuery, RetrievalResult, RetrievedSpan
from rag_schemas.common import SourceKind

pytestmark = pytest.mark.e2e

_SRC = "The platform uptime target is 99.5 percent baseline and 99.9 percent at peak."


class _StubGateway:
    """Returns a fixed answer regardless of prompt (controls the faithfulness input)."""

    def __init__(self, text: str) -> None:
        self._text = text

    @property
    def model_name(self) -> str:
        return "stub"

    async def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        return self._text


def _retrieval(tenant: str) -> RetrievalResult:
    # `scorer` alongside `score`, as the I11 invariant requires: only a reranker writes a
    # score, and it names itself in the same breath. Without it the span reads as unscored and
    # the gate abstains — correctly, since a bare number belongs to no known scale.
    span = RetrievedSpan(chunk_id="c1", doc_id="brief", text=_SRC, score=0.9, scorer="cross_encoder")
    q = RetrievalQuery(tenant_id=tenant, text="what is the uptime target?", top_k=3)
    return RetrievalResult(query=q, spans=[span], max_score=0.9)


def _gen(answer_text: str):  # type: ignore[no-untyped-def]
    from svc_control_plane import wiring
    from svc_generation import GenerationService

    from rag_clients.faithfulness import LexicalFaithfulnessChecker

    return GenerationService(_StubGateway(answer_text), wiring.audit_service(), LexicalFaithfulnessChecker())


async def test_faithful_answer_is_grounded(inmemory: None) -> None:
    ans = await _gen("The uptime target is 99.5 percent baseline. [1]").answer(
        _retrieval("t"), actor="u", trace_id="tr"
    )
    assert ans.status == "grounded"
    assert ans.citations and ans.faithfulness_score is not None and ans.faithfulness_score >= 0.7


async def test_fabricated_answer_abstains(inmemory: None) -> None:
    ans = await _gen("Mars has two moons named Phobos and Deimos.").answer(_retrieval("t"), actor="u", trace_id="tr")
    assert ans.status == "abstained"
    assert ans.citations == []
    assert ans.abstain_reason == "faithfulness_below_threshold"


async def test_borderline_answer_is_flagged(inmemory: None) -> None:
    ans = await _gen("The uptime target is 99.5 percent baseline. Refunds take twelve weeks.").answer(
        _retrieval("t"), actor="u", trace_id="tr"
    )
    assert ans.status == "flagged_for_review"
    assert ans.citations  # still cited
    assert any("Refunds" in c for c in ans.unsupported_claims)


async def test_flagged_answer_lands_in_hitl(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from svc_control_plane import pipeline, wiring

    tenant = "cag"
    with __import__("rag_common").tenant_scope(tenant):
        doc = DocumentEnvelope(
            tenant_id=tenant,
            doc_id="brief",
            source_id="s",
            source_kind=SourceKind.UPLOAD,
            mime="text/plain",
            content_b64=base64.b64encode(_SRC.encode()).decode(),
            ingested_at=datetime(2026, 7, 2, tzinfo=UTC),
        )
        await pipeline.ingest_document(doc, actor="loader")

        borderline = _gen("The uptime target is 99.5 percent baseline. Refunds take twelve weeks.")
        monkeypatch.setattr(wiring, "generation_service", lambda: borderline)

        ans = await pipeline.answer_query(
            RetrievalQuery(tenant_id=tenant, text="uptime target", top_k=3), actor="officer"
        )
        assert ans.status == "flagged_for_review"
        assert ans.review_id

        review = await wiring.hitl_service().get(tenant, ans.review_id)
        assert review.status == "pending"
        assert review.reason == "faithfulness_warning"
        assert review.warnings  # the unsupported claims are visible to the reviewer


class _TwoAnswerGateway:
    """First call returns an over-reaching answer; the retry returns a grounded one."""

    def __init__(self, first: str, second: str) -> None:
        self._answers = [first, second]
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "stub"

    async def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        self.calls += 1
        return self._answers[min(self.calls - 1, len(self._answers) - 1)]


async def test_unfaithful_answer_is_retried_and_the_better_one_kept(inmemory: None) -> None:
    """An answer that fails verification gets ONE bounded second attempt.

    Before this, the loop had no path back: a verified-unfaithful answer went straight to a human
    or to refusal, even when the sources fully supported a narrower answer. Live, this turned an
    abstention into a grounded answer — first attempt 0.0 with two unsupported claims, retry 0.59.
    """
    from svc_control_plane import wiring
    from svc_generation import GenerationService

    from rag_clients.faithfulness import LexicalFaithfulnessChecker

    gw = _TwoAnswerGateway(
        "The uptime target is 99.5 percent baseline. Refunds take twelve weeks.",
        "The uptime target is 99.5 percent baseline.",
    )
    svc = GenerationService(gw, wiring.audit_service(), LexicalFaithfulnessChecker())
    ans = await svc.answer(_retrieval("t"), actor="u", trace_id="tr")

    assert gw.calls == 2, "the model must be re-asked exactly once when verification fails"
    assert "Refunds" not in (ans.answer or ""), "the unsupported claim must be gone from the kept answer"
    assert ans.status == "grounded"


async def test_self_correction_never_keeps_a_worse_answer(inmemory: None) -> None:
    """The retry is a safety net, not a gamble — a worse second attempt is discarded."""
    from svc_control_plane import wiring
    from svc_generation import GenerationService

    from rag_clients.faithfulness import LexicalFaithfulnessChecker

    gw = _TwoAnswerGateway(
        "The uptime target is 99.5 percent baseline. Refunds take twelve weeks.",
        "Mars has two moons named Phobos and Deimos.",  # strictly worse
    )
    svc = GenerationService(gw, wiring.audit_service(), LexicalFaithfulnessChecker())
    ans = await svc.answer(_retrieval("t"), actor="u", trace_id="tr")

    assert gw.calls == 2
    assert "Mars" not in (ans.answer or ""), "a worse retry must be discarded, not published"
