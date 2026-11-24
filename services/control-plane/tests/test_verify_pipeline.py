"""Answer verification through the real query pipeline (RAG v2 Phase 10).

``libs/rag-core/tests/test_verification.py`` carries the phase's acceptance — the seeded error set
and the false-positive measurement. This file pins the WIRING, and in particular the three things
this feature must not do:

1. **Flag off must be byte-identical.** Verification EDITS THE ANSWER, so it is the one thing in
   this phase an operator has to be able to decline completely.
2. **It must not change the answer's STATUS.** The plan's remedy for a failing claim is editorial
   — footnote it or drop it — and routing a deterministic check into ABSTAIN would make one false
   positive withhold a correct answer.
3. **It must not be able to fail an answer** that is already grounded and already paid for.

The fixture is built so a failure is UNAMBIGUOUS: the document states one figure, the generator is
replaced by a stub that states a different one, and the citation quote is real. Nothing here
depends on what a model would actually say.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from typing import Any

import pytest
from svc_control_plane import pipeline

from rag_clients.factory import get_audit_sink, get_catalog
from rag_clients.memory import InMemoryCatalog
from rag_common import get_settings, tenant_scope
from rag_schemas import DocumentEnvelope, RetrievalQuery
from rag_schemas.common import AnswerStatus, AuditAction, SourceKind

_TENANT = "verify10kb"
_DOC = "emd-policy.txt"
_TEXT = (
    "Earnest Money Deposit. Every bidder shall furnish an Earnest Money Deposit of Rs. 5 lakh "
    "along with the technical bid. The deposit is refundable within 30 days of award."
)
_QUESTION = "what Earnest Money Deposit must a bidder furnish?"


def _twin() -> InMemoryCatalog:
    catalog = get_catalog()
    assert isinstance(catalog, InMemoryCatalog), "these are unit tests on the in-memory profile"
    return catalog


def _enable(monkeypatch: pytest.MonkeyPatch, *, on: bool, action: str = "footnote", **env: str) -> None:
    monkeypatch.setenv("RAG_V2_VERIFY", "true" if on else "false")
    monkeypatch.setenv("RAG_V2_VERIFY_ACTION", action)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


async def _ingest() -> None:
    with tenant_scope(_TENANT):
        await pipeline.ingest_document(
            DocumentEnvelope(
                tenant_id=_TENANT,
                doc_id=_DOC,
                source_id="s",
                source_kind=SourceKind.UPLOAD,
                mime="text/plain",
                content_b64=base64.b64encode(_TEXT.encode()).decode(),
                ingested_at=datetime(2026, 10, 21, tzinfo=UTC),
            ),
            actor="loader",
        )


async def _ask() -> Any:
    with tenant_scope(_TENANT):
        return await pipeline.answer_query(RetrievalQuery(tenant_id=_TENANT, text=_QUESTION, top_k=5), actor="asker")


def _corrupt_the_answer(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    """Replace the generated answer text with ``text``, keeping its real citations.

    Patched at ``_verify_answer``'s input rather than at the gateway: what Phase 10 verifies is an
    answer plus the quotes it cites, and driving a small dev model into saying a specific wrong
    number is not something a test can rely on. The citations stay REAL, which is the part that
    matters — the verifier's premise is the stored ``Citation.quote``."""
    original = pipeline._verify_answer

    def _patched(answer: Any, *, trace_id: str) -> Any:
        return original(answer.model_copy(update={"answer": text}), trace_id=trace_id)

    monkeypatch.setattr(pipeline, "_verify_answer", _patched)


async def _quotes() -> list[str]:
    answer = await _ask()
    return [c.quote for c in answer.citations]


# --- the switch ------------------------------------------------------------------------------------


async def test_flag_off_leaves_the_answer_exactly_as_generated(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, on=False)
    await _ingest()
    _corrupt_the_answer(monkeypatch, "The EMD is Rs. 5 Crores [1].")

    answer = await _ask()

    assert answer.answer == "The EMD is Rs. 5 Crores [1]."
    assert "Could not be verified" not in answer.answer


async def test_a_clean_answer_is_untouched_with_the_flag_on(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The false-positive direction, through the real pipeline: a faithful restatement of the real
    citation quote must come back exactly as written."""
    _enable(monkeypatch, on=True)
    await _ingest()
    if not await _quotes():
        pytest.skip("the twin abstained, so there is no citation quote to verify against")
    _corrupt_the_answer(monkeypatch, "The EMD is Rs. 5 lakh [1].")

    answer = await _ask()

    assert answer.answer == "The EMD is Rs. 5 lakh [1]."


# --- the actions -----------------------------------------------------------------------------------


async def test_a_failing_claim_is_footnoted(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, on=True, action="footnote")
    await _ingest()
    if not await _quotes():
        pytest.skip("the twin abstained, so there is no citation quote to verify against")
    _corrupt_the_answer(monkeypatch, "The EMD is Rs. 5 Crores [1].")

    answer = await _ask()

    assert "Could not be verified" in answer.answer
    assert "Rs. 5 Crores" in answer.answer, "the reader keeps what the model said"


async def test_drop_removes_the_failing_claim_and_keeps_the_rest(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, on=True, action="drop")
    await _ingest()
    if not await _quotes():
        pytest.skip("the twin abstained, so there is no citation quote to verify against")
    _corrupt_the_answer(monkeypatch, "The EMD is Rs. 5 lakh [1].\nThe deposit is refundable within 90 days [1].")

    answer = await _ask()

    body = answer.answer.split("Could not be verified")[0]
    assert "Rs. 5 lakh" in body
    assert "90 days" not in body
    assert "Could not be verified" in answer.answer


async def test_report_records_the_verdict_and_changes_nothing(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For measuring the check against a live corpus before letting it touch an answer, which is
    how a guard like this should be adopted."""
    _enable(monkeypatch, on=True, action="report", RAG_QUERY_LOG_ENABLED="true")
    await _ingest()
    if not await _quotes():
        pytest.skip("the twin abstained, so there is no citation quote to verify against")
    _corrupt_the_answer(monkeypatch, "The EMD is Rs. 5 Crores [1].")

    answer = await _ask()

    assert answer.answer == "The EMD is Rs. 5 Crores [1]."
    rows = await _twin().query_log(_TENANT)
    assert rows and rows[-1]["verification"], "the verdict is still recorded"


# --- what this feature must not do -------------------------------------------------------------------


async def test_the_status_is_never_changed_by_verification(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Routing a DETERMINISTIC check into ABSTAIN would make one false positive withhold a correct
    answer. The plan's remedy is editorial, and footnoting or dropping a claim is already a
    stronger consequence than a score nudge."""
    _enable(monkeypatch, on=False)
    await _ingest()
    _corrupt_the_answer(monkeypatch, "The EMD is Rs. 5 Crores [1].")
    before = (await _ask()).status

    _enable(monkeypatch, on=True)
    after = await _ask()

    assert after.status is before
    assert after.status is not AnswerStatus.ABSTAINED or before is AnswerStatus.ABSTAINED


async def test_no_extra_audit_event_is_recorded(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same reason as the Phase-12 timing slice: ``count_actions`` meters QUERY events for FinOps
    showback, so a verification event per question would inflate every tenant's bill."""
    _enable(monkeypatch, on=False)
    await _ingest()
    await _ask()
    events = await get_audit_sink().list_events(_TENANT, None, 500)
    without = sum(1 for e in events if e.action is AuditAction.QUERY)

    _enable(monkeypatch, on=True)
    await _ask()
    events = await get_audit_sink().list_events(_TENANT, None, 500)
    with_verify = sum(1 for e in events if e.action is AuditAction.QUERY)

    assert with_verify - without == without, "the second question added exactly as many events as the first"


async def test_a_failing_verifier_does_not_fail_the_answer(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, on=True)
    await _ingest()

    def _boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("verifier exploded")

    monkeypatch.setattr(pipeline, "verify_claims", _boom)
    answer = await _ask()

    assert answer.query, "the answer was still returned"
    assert "Could not be verified" not in answer.answer


async def test_an_abstain_is_not_verified(inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """An abstain carries no citations by contract, and "every claim failed" would be a false
    reading of "there was nothing to check against"."""
    _enable(monkeypatch, on=True, RAG_QUERY_LOG_ENABLED="true")
    await _ingest()

    with tenant_scope(_TENANT):
        answer = await pipeline.answer_query(
            RetrievalQuery(tenant_id=_TENANT, text="what is the capital of Peru?", top_k=5), actor="asker"
        )

    assert "Could not be verified" not in answer.answer
    if answer.status is AnswerStatus.ABSTAINED:
        rows = await _twin().query_log(_TENANT)
        assert rows and rows[-1]["verification"] is None


# --- the stored report -------------------------------------------------------------------------------


async def test_the_report_reaches_the_query_log(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan §10 item 6. The column has existed since Phase 1; Phase 12 gave it a writer and this
    is the first thing that fills it."""
    import json

    _enable(monkeypatch, on=True, RAG_QUERY_LOG_ENABLED="true")
    await _ingest()
    if not await _quotes():
        pytest.skip("the twin abstained, so there is no citation quote to verify against")
    _corrupt_the_answer(monkeypatch, "The EMD is Rs. 5 Crores [1].")

    await _ask()

    stored = json.loads((await _twin().query_log(_TENANT))[-1]["verification"])
    assert stored["checked"] >= 1 and stored["failed"] >= 1
    assert stored["failures"][0]["reasons"] == ["value_not_in_quote"]


async def test_the_column_stays_null_when_verification_is_off(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NULL is the honest distinction between "verified and clean" and "not verified"."""
    _enable(monkeypatch, on=False, RAG_QUERY_LOG_ENABLED="true")
    await _ingest()
    await _ask()

    rows = await _twin().query_log(_TENANT)
    assert rows and rows[-1]["verification"] is None
