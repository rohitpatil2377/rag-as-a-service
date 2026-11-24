"""FR-E4 over a map-reduce answer, on BOTH paths (RAG v2 Phase 12).

`_map_reduce_answer` short-circuits ``generation_service().answer()`` and the streamed reduce
never reaches it either, so neither inherited the faithfulness stage. For a while only the
streaming path had a version of it — which meant the SAME question routed to human review
through the console and not through the API, with nothing saying so.

So what is pinned here is PARITY, not merely presence: the two paths run the same check through
the same function and reach the same status, they both record the distinct FAITHFULNESS audit
event the ordinary path records, and a flagged answer reaches the caller's existing HITL hook.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
from svc_control_plane import pipeline, wiring

from rag_clients.factory import get_audit_sink, get_catalog
from rag_clients.memory import InMemoryCatalog
from rag_common import get_settings, tenant_scope
from rag_core.doc_profile import DocProfile, ProfileFacet
from rag_core.faithfulness import FaithfulnessVerdict
from rag_core.layout import LayoutBlock, LayoutSection, LayoutTree
from rag_core.section_enrich import SectionEnrichment
from rag_schemas import DocumentEnvelope, RetrievalQuery, RetrievedSpan
from rag_schemas.common import AnswerStatus, AuditAction, SourceKind

_TENANT = "mrfaithkb"
_DOC = "corrigendum-v.txt"
_SWITCH_TEXT = "Switches. The switches shall comply with UL 62368 and FCC Part 15 Class A certifications."
_FIRE_TEXT = "Firewalls. Each firewall solution shall be supplied with an HBOM and an SBOM."
_TEXT = f"{_SWITCH_TEXT}\n\n{_FIRE_TEXT}"
_QUERY = "list all the compliance details from this document"


def _twin() -> InMemoryCatalog:
    catalog = get_catalog()
    assert isinstance(catalog, InMemoryCatalog), "these are unit tests on the in-memory profile"
    return catalog


def _enable(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    for flag in ("RAG_V2_ROUTER", "RAG_V2_RETRIEVAL", "RAG_V2_GENERATE"):
        monkeypatch.setenv(flag, "true")
    monkeypatch.setenv("RAG_FAITHFULNESS_ENABLED", "true")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


class _Checker:
    """A checker that returns a fixed score, and records what it was asked to judge."""

    def __init__(self, score: float, claims: list[str] | None = None) -> None:
        self._score = score
        self._claims = claims or []
        self.calls: list[tuple[str, int, str]] = []

    async def check(self, answer: str, spans: list[RetrievedSpan], query: str = "") -> FaithfulnessVerdict:
        self.calls.append((answer, len(spans), query))
        return FaithfulnessVerdict(
            score=self._score,
            faithful=not self._claims,
            unsupported_claims=list(self._claims),
            # A double that leaves `method` unset is not standing in for a real checker:
            # every shipped one names itself, and `verification_block` refuses to invent
            # provenance for a verdict that did not record any.
            method="nli",
        )


_SWITCH_SECTION = "- The switches shall comply with UL 62368 [m1]"
_FIRE_SECTION = "- Each firewall is supplied with an HBOM and an SBOM [m2]"


class _Gateway:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def _reduce(self, prompt: str) -> str:
        topic = prompt.rsplit("## Topic", 1)[-1]
        return _SWITCH_SECTION if "certifications" in topic else _FIRE_SECTION

    async def complete(self, prompt: str, **_: Any) -> str:
        self.prompts.append(prompt)
        if "## Extracted items for this topic" in prompt:
            return self._reduce(prompt)
        return _map_reply(prompt)

    async def stream(self, prompt: str, **_: Any) -> AsyncIterator[str]:
        self.prompts.append(prompt)
        for token in self._reduce(prompt).split(" "):
            yield token + " "

    @property
    def model_name(self) -> str:
        return "test-model"

    async def list_models(self) -> list[str]:
        return ["test-model"]


def _map_reply(prompt: str) -> str:
    import json

    quote = "UL 62368" if "Switches" in prompt else "HBOM and an SBOM"
    facet = "certifications" if "Switches" in prompt else "supply_chain"
    return json.dumps([{"facet_id": facet, "statement": f"The document requires {quote}.", "quote": quote}])


def _tree() -> LayoutTree:
    def section(sid: str, crumb: str, ordinal: int) -> LayoutSection:
        return LayoutSection(
            section_id=sid,
            breadcrumb=crumb,
            title=crumb,
            depth=0,
            ordinal=ordinal,
            text="body",
            text_hash="h",
            blocks=(LayoutBlock(block_id=f"{sid}-b0", block_type="paragraph", text="body", page=1, ordinal=0),),
        )

    return LayoutTree(
        doc_id=_DOC,
        sections=(section("sec-switch", "21. Switches", 0), section("sec-fire", "22. Firewalls", 1)),
        source="markup",
    )


async def _seed() -> None:
    with tenant_scope(_TENANT):
        await pipeline.ingest_document(
            DocumentEnvelope(
                tenant_id=_TENANT,
                doc_id=_DOC,
                source_id="s",
                source_kind=SourceKind.UPLOAD,
                mime="text/plain",
                content_b64=base64.b64encode(_TEXT.encode()).decode(),
                ingested_at=datetime(2026, 10, 22, tzinfo=UTC),
            ),
            actor="loader",
        )
    catalog = _twin()
    await catalog.replace_document_profile(
        _TENANT,
        _DOC,
        DocProfile(
            doc_type="corrigendum",
            purpose="Amends the bidding document.",
            title=_DOC,
            facets=(ProfileFacet("certifications", "Certifications", "d"), ProfileFacet("supply_chain", "SC", "d")),
        ),
    )
    await catalog.replace_document_structure(_TENANT, _tree())
    await catalog.replace_section_enrichment(
        _TENANT,
        _DOC,
        {
            "sec-switch": SectionEnrichment(
                section_id="sec-switch", summary="Switch certifications.", facet_ids=("certifications",)
            ),
            "sec-fire": SectionEnrichment(
                section_id="sec-fire", summary="Firewall supply chain.", facet_ids=("supply_chain",)
            ),
        },
    )
    existing = await catalog.document_chunks(_TENANT, _DOC)
    base = existing[0]
    await catalog.record_chunks(
        _TENANT,
        [
            base.model_copy(update={"chunk_id": "ck-switch", "section_id": "sec-switch", "text": _SWITCH_TEXT}),
            base.model_copy(update={"chunk_id": "ck-fire", "section_id": "sec-fire", "text": _FIRE_TEXT}),
        ],
    )


def _patch(monkeypatch: pytest.MonkeyPatch, checker: _Checker) -> _Gateway:
    gw = _Gateway()

    async def _gateway(_tenant: str, model: str | None = None) -> Any:
        return gw

    async def _faith(_tenant: str) -> Any:
        return checker

    monkeypatch.setattr(wiring, "resolve_tenant_gateway", _gateway)
    monkeypatch.setattr(wiring, "resolve_tenant_faithfulness", _faith)
    return gw


async def _ask(monkeypatch: pytest.MonkeyPatch, checker: _Checker) -> Any:
    _patch(monkeypatch, checker)
    with tenant_scope(_TENANT):
        return await pipeline.answer_query(
            RetrievalQuery(tenant_id=_TENANT, text=_QUERY, top_k=5, doc_ids=(_DOC,)), actor="asker"
        )


async def _stream(monkeypatch: pytest.MonkeyPatch, checker: _Checker) -> Any:
    _patch(monkeypatch, checker)
    final = None
    with tenant_scope(_TENANT):
        async for event in pipeline.stream_answer(
            RetrievalQuery(tenant_id=_TENANT, text=_QUERY, top_k=5, doc_ids=(_DOC,)), actor="asker"
        ):
            if event["type"] == "done":
                final = event["answer"]
    return final


# --- the gap this closes ------------------------------------------------------------------------------


async def test_the_non_streaming_answer_carries_a_faithfulness_score(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It carried none before this slice — so a tenant relying on the review queue had it
    silently off for aggregate questions asked through the API."""
    _enable(monkeypatch)
    await _seed()

    answer = await _ask(monkeypatch, _Checker(0.95))

    assert answer.faithfulness_score == pytest.approx(0.95)
    assert answer.status is AnswerStatus.GROUNDED


async def test_both_paths_reach_the_same_status_and_score(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PARITY is the point. Two implementations of a safety control is one implementation and
    one liability, so both paths call the same function."""
    _enable(monkeypatch)
    await _seed()

    direct = await _ask(monkeypatch, _Checker(0.5, ["an unsupported sentence"]))
    streamed = await _stream(monkeypatch, _Checker(0.5, ["an unsupported sentence"]))

    assert streamed is not None
    assert direct.status.value == streamed["status"]
    assert direct.faithfulness_score == pytest.approx(streamed["faithfulness_score"])


# --- the three-way routing ----------------------------------------------------------------------------


async def test_a_high_score_stays_grounded(inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch, RAG_FAITHFUL_THRESHOLD="0.7", RAG_FLAG_THRESHOLD="0.4")
    await _seed()

    answer = await _ask(monkeypatch, _Checker(0.9))

    assert answer.status is AnswerStatus.GROUNDED and answer.citations


async def test_a_middling_score_is_flagged_for_review(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And it reaches the caller's EXISTING HITL hook, which keys on the status — so the review
    queue works for a map-reduce answer without a second code path."""
    _enable(monkeypatch, RAG_FAITHFUL_THRESHOLD="0.7", RAG_FLAG_THRESHOLD="0.4")
    await _seed()

    answer = await _ask(monkeypatch, _Checker(0.55, ["a doubtful sentence"]))

    assert answer.status is AnswerStatus.FLAGGED_FOR_REVIEW
    assert answer.unsupported_claims == ["a doubtful sentence"]
    assert answer.review_id, "the existing hook opened a review"


async def test_a_low_score_abstains(inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch, RAG_FAITHFUL_THRESHOLD="0.7", RAG_FLAG_THRESHOLD="0.4")
    await _seed()

    answer = await _ask(monkeypatch, _Checker(0.1, ["unsupported"]))

    assert answer.status is AnswerStatus.ABSTAINED
    assert answer.citations == [], "grounded-or-abstain: an abstain carries none"


async def test_per_kb_thresholds_win_over_the_platform_defaults(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parity with the generation service, which reads the same overrides. A KB that raised its
    floor must not keep receiving the answers it raised it to stop."""
    _enable(monkeypatch, RAG_FAITHFUL_THRESHOLD="0.5", RAG_FLAG_THRESHOLD="0.2")
    await _seed()
    await _twin().set_tenant_config(_TENANT, "faithful_threshold", "0.95")

    answer = await _ask(monkeypatch, _Checker(0.6))

    assert answer.status is not AnswerStatus.GROUNDED, "0.6 clears the platform floor but not the KB's"


# --- what the checker is given -------------------------------------------------------------------------


async def test_the_checker_judges_the_assembled_answer_against_its_own_cited_spans(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch)
    await _seed()
    checker = _Checker(0.95)

    await _ask(monkeypatch, checker)

    assert len(checker.calls) == 1
    judged, span_count, asked = checker.calls[0]
    assert "UL 62368" in judged and "HBOM" in judged, "the whole assembled answer"
    assert span_count >= 2, "its own resolved citation spans"
    assert asked == _QUERY, "and the question, for the relevance gate"


# --- metering ------------------------------------------------------------------------------------------


async def test_the_distinct_faithfulness_audit_event_is_recorded(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NOT the inflation a second QUERY event would be: the ordinary path records one
    FAITHFULNESS event per answer, so recording it here is what makes the meters AGREE."""
    _enable(monkeypatch)
    await _seed()

    await _ask(monkeypatch, _Checker(0.9))

    events = await get_audit_sink().list_events(_TENANT, None, 500)
    faith = [e for e in events if e.action is AuditAction.FAITHFULNESS]
    assert len(faith) == 1
    assert faith[0].payload["score"] == pytest.approx(0.9)
    assert faith[0].payload["outcome"] == AnswerStatus.GROUNDED.value
    assert faith[0].payload["path"] == "rag_v2_map_reduce"


async def test_the_query_event_count_is_unchanged(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`count_actions` meters QUERY for FinOps showback, so the faithfulness stage must not add
    one of those."""
    _enable(monkeypatch)
    await _seed()

    await _ask(monkeypatch, _Checker(0.9))
    events = await get_audit_sink().list_events(_TENANT, None, 500)
    first = sum(1 for e in events if e.action is AuditAction.QUERY)

    await _ask(monkeypatch, _Checker(0.9))
    events = await get_audit_sink().list_events(_TENANT, None, 500)

    assert sum(1 for e in events if e.action is AuditAction.QUERY) == first * 2


# --- fail-safe -----------------------------------------------------------------------------------------


async def test_a_checker_fault_leaves_the_answer_grounded(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every statement in this answer is already tied to a verbatim quote that was checked
    against the section it came from. Ungrounding it because the NLI service was down would
    refuse an answer on the strength of evidence that is still there."""
    _enable(monkeypatch)
    await _seed()

    class _Broken(_Checker):
        async def check(self, answer: str, spans: list[RetrievedSpan], query: str = "") -> FaithfulnessVerdict:
            raise RuntimeError("nli unavailable")

    answer = await _ask(monkeypatch, _Broken(0.0))

    assert answer.status is AnswerStatus.GROUNDED
    assert answer.faithfulness_score is None, "and it does not claim a score it never got"


async def test_the_gate_being_off_leaves_the_answer_unscored(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, RAG_FAITHFULNESS_ENABLED="false")
    await _seed()

    answer = await _ask(monkeypatch, _Checker(0.1))

    assert answer.status is AnswerStatus.GROUNDED and answer.faithfulness_score is None


# --- verification provenance on the aggregate (SPEC, not yet implemented) -------------------------
#
# The checker-level half of this is `libs/clients/tests/test_nli_verification_integrity.py`. This
# is the half that matters to a reader: whatever the checker records has to REACH the answer, or
# a governance process has to read a server log to find out how the answer was verified.


async def test_the_answer_discloses_how_it_was_verified(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A score is not provenance. `GroundedAnswer` carries `faithfulness_score` and
    `unsupported_claims` and nothing that says which checker produced either, so a lexically
    scored answer and an NLI-scored one are the same artifact to the console, the API and the
    audit reader."""
    _enable(monkeypatch)
    await _seed()

    answer = await _ask(monkeypatch, _Checker(0.95))

    assert answer.verification is not None
    assert answer.verification["outcome"] == "scored"
    assert answer.verification["degraded"] is False
    assert answer.verification["methods"] == [{"method": "nli", "count": 1}]


async def test_one_degraded_batch_makes_the_whole_answer_degraded(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE INVARIANT THIS EXISTS FOR. Different batches can be verified by different methods --
    the real run that prompted this had 12 batches scored by NLI and 1 by the lexical checker
    after a ReadError. An aggregate that reports the majority method, or just a mean score, tells
    the reader the whole answer was NLI-verified. It was not, and the one batch that was not is
    exactly the one a reviewer needs to look at.

    So the aggregate reports the DISTRIBUTION and takes the WEAKEST outcome in it."""
    _enable(monkeypatch)
    await _seed()

    class _Degraded(_Checker):
        async def check(self, answer: str, spans: list[RetrievedSpan], query: str = "") -> FaithfulnessVerdict:
            v = await super().check(answer, spans, query)
            object.__setattr__(v, "method", "lexical_fallback")
            object.__setattr__(v, "degraded", True)
            return v

    answer = await _ask(monkeypatch, _Degraded(0.95))

    assert answer.verification["degraded"] is True
    assert answer.verification["outcome"] == "scored", (
        "one method throughout, so its scores are comparable -- `degraded` is the separate fact "
        "that the intended mechanism was not the one that ran"
    )
    assert answer.verification["methods"] == [{"method": "lexical_fallback", "count": 1}]
    # AND THE STATUS DID NOT MOVE. The grounded-or-abstain invariant is about CITATIONS, and
    # every statement here is still tied to a quote checked verbatim against its section --
    # `test_a_checker_fault_leaves_the_answer_grounded` argues that case and this must not
    # contradict it. "The aggregate reflects the weakest verification outcome" is a statement
    # about the VERIFICATION metadata, not about `AnswerStatus`. Whether a degraded verification
    # should additionally route to review is a separate decision with its own cost, and is
    # deliberately NOT asserted here.
    assert answer.status is AnswerStatus.GROUNDED


# THE MIXED-METHOD CASE IS NOT ON THIS PATH, and a test for it here would have passed
# vacuously. Probed on the real pipeline: the map-reduce route calls the faithfulness checker
# EXACTLY ONCE, over the assembled answer, so its method distribution can only ever have one
# entry. Several batches verified by different methods is the WHOLE-DOCUMENT route's shape -- a
# real incumbent run produced 13 batch verifications, one of which fell back to the lexical
# checker. The aggregation itself is specified where it lives, as a unit test of
# `rag_core.faithfulness.verification_block`.
async def test_a_grounded_answer_still_discloses_what_the_checks_found(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GROUNDED must not be readable as "every check passed".

    `unsupported_claims` is populated in exactly two places, both inside `flag_for_review`;
    `ground()` has no such parameter. So a grounded answer carries an EMPTY claims list however
    much was found -- and things are found. Measured on a real run: the numeric guard reported 18
    ungrounded figures, built their carrying sentences, set `faithful=False`, and routing read the
    SCORE rather than that flag, so the answer shipped GROUNDED with nothing to show. The audit
    payload has them; the answer, the API and the console do not.

    The status is correct and stays GROUNDED -- the citations are intact, which is what the status
    is about. What must change is that the finding is reachable without reading a server log."""
    _enable(monkeypatch)
    await _seed()

    answer = await _ask(monkeypatch, _Checker(0.95, claims=["The total is 42 units."]))

    assert answer.status is AnswerStatus.GROUNDED
    assert answer.unsupported_claims == ["The total is 42 units."], (
        "a grounded answer must be able to carry what a check flagged, or GROUNDED silently "
        "means 'nothing was flagged' when it only means 'the citations hold'"
    )
