"""What the whole-document read does with a batch that FAILS grounding (FR-E4 at composite level).

The incumbent answers an aggregate question by splitting the document into batches, running each
through the unchanged grounded generation path, and assembling the GROUNDED ones. Every batch is
faithfulness-checked and routed individually — that part is not in doubt.

What is in doubt is the COMPOSITE. The reduce loop drops any batch that is not GROUNDED
(``pipeline.py`` has four such ``continue`` branches), so the delivered answer is the surviving
subset. The question these tests settle is whether the reader and the audit trail are told:

    is the answer GROUNDED, or is it the part of it that happened to survive grounding?

Those are not the same claim, and a system must never report the second as the first.

DETERMINISTIC, no model: a stub gateway returns a faithful extract for one batch and an extract
with one unsupported sentence for the other, and the deterministic LexicalFaithfulnessChecker
turns that into a real FLAGGED outcome. Batch size is pinned to 1 chunk so the split is exact.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime

import pytest

from rag_schemas import DocumentEnvelope
from rag_schemas.common import SourceKind

pytestmark = pytest.mark.e2e

# Long enough that the clause chunker really produces two chunks — a two-sentence document
# becomes ONE chunk, and then there is no second batch to flag and the test proves nothing.
_CLEAN = " ".join(
    [
        "The platform uptime target is 99.5 percent baseline and 99.9 percent at peak.",
        *[
            f"Availability clause {i} requires the operator to record downtime minutes and "
            f"publish the monthly uptime attestation to the purchaser."
            for i in range(40)
        ],
    ]
)
_ALSO = " ".join(
    [
        "The recovery time objective is 2 hours for every production workload.",
        *[
            f"Continuity clause {i} requires the operator to rehearse failover and retain the "
            f"drill report for audit inspection."
            for i in range(40)
        ],
    ]
)
# The sentence no chunk supports. Lexically absent from both chunks, so the checker's verdict is
# earned rather than stubbed.
_UNSUPPORTED = "Refunds take twelve weeks to process."


class _PerBatchGateway:
    """A faithful extract for the first batch; a MOSTLY-faithful one for the second.

    The second batch must land in the FLAG band, not the abstain band, and that is the whole
    difficulty: an extract that shares no wording with its chunks scores near 0 and ABSTAINS,
    which is a different outcome with different (correct) handling — nothing to review, so
    nothing is queued. A flagged extract is mostly supported with one unsupported sentence, so
    this echoes real sentences out of the batch's OWN prompt and appends the single claim no
    chunk makes.
    """

    def __init__(self) -> None:
        self.prompts: list[str] = []

    @property
    def model_name(self) -> str:
        return "stub"

    @staticmethod
    def _echo(prompt: str, n: int) -> str:
        """n real sentences lifted from the batch's own excerpts, each cited."""
        sentences = [x.strip() for x in prompt.split(".") if "clause" in x or "uptime" in x or "recovery" in x]
        return " ".join(f"{x}. [1]" for x in sentences[:n]) or "No content. [1]"

    async def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        self.prompts.append(prompt)
        if "uptime" in prompt:
            return self._echo(prompt, 4)
        # Mostly supported, one sentence that no chunk contains -> FLAGGED, not abstained.
        return f"{self._echo(prompt, 6)} {_UNSUPPORTED}"


def _gen(gateway: object):  # type: ignore[no-untyped-def]
    from svc_control_plane import wiring
    from svc_generation import GenerationService

    from rag_clients.faithfulness import LexicalFaithfulnessChecker

    return GenerationService(gateway, wiring.audit_service(), LexicalFaithfulnessChecker())


async def _two_batch_answer(monkeypatch: pytest.MonkeyPatch, tenant: str, gateway=None):  # type: ignore[no-untyped-def]
    """Ingest a two-chunk document, force one chunk per batch, answer it whole-document."""
    from svc_control_plane import pipeline, wiring

    import rag_common
    from rag_common.config import get_settings

    with rag_common.tenant_scope(tenant):
        body = f"{_CLEAN}\n\n{_ALSO}"
        doc = DocumentEnvelope(
            tenant_id=tenant,
            doc_id="brief",
            source_id="s",
            source_kind=SourceKind.UPLOAD,
            mime="text/plain",
            content_b64=base64.b64encode(body.encode()).decode(),
            ingested_at=datetime(2026, 7, 2, tzinfo=UTC),
        )
        await pipeline.ingest_document(doc, actor="loader")

        # One chunk per batch, and the convergence re-read off: this test is about what the
        # reduce does with a flagged batch, not about a later loop recovering its chunks.
        # `Settings` is frozen, so this goes through the environment and the cache, as every
        # other settings-dependent e2e test does.
        monkeypatch.setenv("RAG_WHOLE_DOC_SUMMARY_BATCH_CHUNKS", "10")
        # Matched to the LIVE deployment, which has both of these on. The guard defaults OFF in
        # code, and leaving it there would have made `completeness is None` look like a finding
        # when it is only this harness not reproducing production.
        monkeypatch.setenv("RAG_COMPLETENESS_GUARD_ENABLED", "true")
        monkeypatch.setenv("RAG_COVERAGE_VERIFY_ENABLED", "true")
        monkeypatch.setenv("RAG_WHOLE_DOC_FULL_READ_TOKEN_BUDGET", "1000")
        get_settings.cache_clear()

        gateway = gateway or _PerBatchGateway()
        monkeypatch.setattr(wiring, "generation_service", lambda: _gen(gateway))

        from rag_schemas import RetrievalQuery

        query = RetrievalQuery(tenant_id=tenant, text="summarise this document", top_k=5)
        try:
            answer = await pipeline._answer_document_full_summary(query, "brief", actor="officer")
        finally:
            # monkeypatch restores the ENV; nothing restores the process-wide lru_cache, so a
            # later test in the same session would inherit this one's Settings.
            get_settings.cache_clear()
        return answer, gateway


async def test_a_flagged_batch_is_excluded_from_the_composite(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented behaviour, pinned so a later change cannot alter it silently."""
    answer, gateway = await _two_batch_answer(monkeypatch, "wd-a")
    assert len(gateway.prompts) >= 2, "the document did not split into two batches"
    assert _UNSUPPORTED not in answer.answer, "an unsupported claim reached the reader"


async def test_the_composite_declares_the_coverage_it_actually_achieved(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dropped batch's chunks are uncited, so completeness must show partial coverage. This is
    the reader's only signal that the answer is a SUBSET — without it, a composite assembled from
    half the document reads exactly like one assembled from all of it."""
    answer, _gw = await _two_batch_answer(monkeypatch, "wd-b")
    completeness = answer.completeness
    assert completeness is not None, "the composite carries no completeness payload at all"
    assert completeness.covered < completeness.expected, (
        f"coverage claims {completeness.covered}/{completeness.expected} — the flagged batch's "
        f"chunk is uncited and must not count as covered"
    )


async def test_the_flagged_batch_created_a_review_entry(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Routing must still happen for evidence the composite discarded. If the batch is dropped
    AND no review is queued, the failure is invisible to everyone: absent from the answer, absent
    from the queue, and the only trace is a log line."""
    from svc_control_plane import wiring

    await _two_batch_answer(monkeypatch, "wd-c")
    pending = await wiring.hitl_service().list("wd-c", None, 50)
    assert pending, "a batch was flagged for review and nothing reached the review queue"
    assert any(r.reason == "faithfulness_warning" for r in pending), [r.reason for r in pending]


async def test_the_composite_status_does_not_claim_more_than_it_earned(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE GOVERNANCE QUESTION. One batch was flagged for review and dropped. The composite is
    the surviving subset, and reporting it as plainly GROUNDED with no review pointer tells the
    reader that everything in the document was grounded — which is the one thing that was not
    established.

    Either the status carries the review (`flagged_for_review`/`review_id`), or the answer says
    in its own text that part of the document was withheld pending review."""
    answer, _gw = await _two_batch_answer(monkeypatch, "wd-d")
    signalled = (
        answer.status.value == "flagged_for_review"
        or answer.review_id is not None
        or "review" in (answer.answer or "").casefold()
    )
    assert signalled, (
        f"status={answer.status.value} review_id={answer.review_id} — a batch was flagged and "
        f"dropped, and the delivered answer signals neither"
    )


async def test_an_abstained_batch_creates_no_review_and_no_exclusion_event(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ABSTAIN IS NOT A FLAG, and conflating them is how a review queue becomes useless. An
    abstained batch asserted nothing, so there is no claim for a human to adjudicate; queueing
    it would bury the real flags under "we don't know". This is the case my first harness hit
    by accident — 19 abstained batches — and it must stay distinguishable."""
    from svc_control_plane import wiring

    class _AbstainGateway(_PerBatchGateway):
        async def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
            self.prompts.append(prompt)
            # Shares no wording with any chunk -> scores near zero -> ABSTAINED, not flagged.
            return "Mars has two moons named Phobos and Deimos. [1]"

    await _two_batch_answer(monkeypatch, "wd-abstain", gateway=_AbstainGateway())
    assert await wiring.hitl_service().list("wd-abstain", None, 50) == []
    events = [
        e
        for e in await wiring.audit_service().list("wd-abstain", limit=200)
        if e.action.value == "faithfulness_batch_excluded"
    ]
    assert events == [], f"an abstained batch produced an exclusion event: {events}"


async def test_repeated_flagged_evidence_does_not_create_a_second_review(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coverage loop re-reads uncited chunks, so before this fix one flagged batch became
    four reviews. Two things now prevent it: the chunks are withheld rather than left uncited,
    so the loop stops re-reading them, and the evidence fingerprint deduplicates any repeat
    that still occurs. The fingerprint excludes the score deliberately — a model scoring 0.6511
    then 0.6489 on identical evidence is the same failure, not a new one."""
    from svc_control_plane import wiring

    _answer, _gw = await _two_batch_answer(monkeypatch, "wd-dedup")
    reviews = await wiring.hitl_service().list("wd-dedup", None, 50)
    fingerprints = {r.evidence_fingerprint for r in reviews}
    assert len(reviews) == len(fingerprints), (
        f"{len(reviews)} reviews for {len(fingerprints)} distinct evidence fingerprint(s)"
    )
    assert all(r.evidence_fingerprint for r in reviews), "a review was written with no fingerprint"


async def test_the_review_carries_the_verification_context_a_reviewer_needs(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Score AND the threshold it was judged against. The score alone is unreadable a month
    later, once the gate has moved."""
    from svc_control_plane import wiring

    await _two_batch_answer(monkeypatch, "wd-meta")
    reviews = await wiring.hitl_service().list("wd-meta", None, 50)
    assert reviews, "nothing to inspect"
    r = reviews[0]
    assert r.faithfulness_score is not None and 0.0 < r.faithfulness_score < 1.0, r.faithfulness_score
    assert r.faithfulness_threshold is not None, "the gate applied was not recorded"
    assert r.warnings, "the unsupported claims did not reach the reviewer"


async def test_every_excluded_batch_is_recoverable_from_the_audit_trail(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The invariant the whole change exists for: a flagged batch never disappears without a
    recorded disposition. The FAITHFULNESS event says verification happened; only this one says
    what became of the evidence afterwards."""
    from svc_control_plane import wiring

    answer, _gw = await _two_batch_answer(monkeypatch, "wd-audit")
    events = [
        e
        for e in await wiring.audit_service().list("wd-audit", limit=200)
        if e.action.value == "faithfulness_batch_excluded"
    ]
    assert events, "evidence was withheld with no exclusion event"
    withheld = int((answer.structured or {}).get("withheld_batches") or 0)
    assert len(events) == withheld, f"{withheld} batches withheld, {len(events)} exclusion events"
    for e in events:
        p = e.payload
        assert p["chunk_ids"], "the excluded evidence cannot be recovered"
        assert p["evidence_fingerprint"], "no dedup identity recorded"
        assert p["persistence_outcome"] in ("created", "deduplicated", "failed"), p["persistence_outcome"]
        assert p["exclusion_reason"] == "below_faithfulness_threshold"


async def test_a_persistence_failure_still_withholds_and_still_audits(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE CASE THE FIX EXISTS TO MAKE SAFE. If the review cannot be stored, the answer must
    still not claim to be fully grounded, the evidence must still be withheld, and the trail
    must still record it — and the disclosure must NOT promise a review that nobody received."""
    from svc_control_plane import wiring

    service = wiring.hitl_service()

    async def _boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("hitl store unavailable")

    monkeypatch.setattr(service, "create", _boom)
    monkeypatch.setattr(wiring, "hitl_service", lambda: service)

    answer, _gw = await _two_batch_answer(monkeypatch, "wd-fail")
    assert answer.status.value == "flagged_for_review", "a failed review let the answer read as grounded"
    assert _UNSUPPORTED not in answer.answer, "the unverified claim reached the reader anyway"
    assert "could NOT be queued" in answer.answer, "the disclosure promised a review that never persisted"
    events = [
        e
        for e in await wiring.audit_service().list("wd-fail", limit=200)
        if e.action.value == "faithfulness_batch_excluded"
    ]
    assert events and all(e.payload["persistence_outcome"] == "failed" for e in events)
    assert all(e.payload["review_id"] is None for e in events)
