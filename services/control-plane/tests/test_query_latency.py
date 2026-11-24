"""Per-stage query timing and the ``query_log`` row, through the real pipeline (Phase 12).

Discovery §8.2 asks for this slice BEFORE the phases it is meant to measure, because the plan's
§0 success criteria are latency targets and the query path recorded one number — the round trip —
so a slow answer could not be attributed to routing, retrieval or generation. That is also why
the Phase-0 baseline was never run.

What is pinned here is the wiring and, more importantly, the three things this feature must NOT
do:

1. **It must not add an audit event.** ``AuditSink.count_actions`` meters QUERY events for FinOps
   showback (FR-H4), so one extra event per question would silently inflate every tenant's bill.
   The timings go to a structured log and to ``query_log``, never to a new audit row.
2. **It must not store anything by default.** The row carries the question and the answer, which
   quotes the corpus — the same class of derived record as a chat turn, which is why the DPDP
   erasure path for it existed before anything wrote it.
3. **It must not be able to fail an answer.** A correct, grounded, already-paid-for answer that
   500s because its latency could not be logged would be a worse defect than having no timings.
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
from rag_schemas.common import AuditAction, SourceKind

_TENANT = "qlogkb"
_DOC = "emd-policy.txt"
_TEXT = (
    "Earnest Money Deposit. Every bidder shall furnish an Earnest Money Deposit of Rs. 5 lakh "
    "along with the technical bid. The deposit is refundable within thirty days of award. "
    "A bid submitted without the deposit shall be treated as non-responsive."
)


def _twin() -> InMemoryCatalog:
    catalog = get_catalog()
    assert isinstance(catalog, InMemoryCatalog), "these are unit tests on the in-memory profile"
    return catalog


def _enable(monkeypatch: pytest.MonkeyPatch, *, on: bool, **env: str) -> None:
    monkeypatch.setenv("RAG_QUERY_LOG_ENABLED", "true" if on else "false")
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


async def _ask(question: str = "what Earnest Money Deposit must a bidder furnish?") -> Any:
    with tenant_scope(_TENANT):
        return await pipeline.answer_query(RetrievalQuery(tenant_id=_TENANT, text=question, top_k=5), actor="asker")


async def _query_actions() -> int:
    events = await get_audit_sink().list_events(_TENANT, None, 500)
    return sum(1 for e in events if e.action is AuditAction.QUERY)


# --- the switch -----------------------------------------------------------------------------------


async def test_no_row_is_written_by_default(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default OFF, and not because it is unfinished — see the module docstring."""
    _enable(monkeypatch, on=False)
    await _ingest()
    await _ask()

    assert await _twin().query_log(_TENANT) == []


async def test_the_timing_is_logged_even_with_the_row_switched_off(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The log line carries no question text — stage durations, a trace id and a status — so it is
    always on. The ROW is the part that stores the question, and that is what the flag gates.

    Read off STDOUT rather than through ``caplog``: this platform logs through structlog's own
    printer, so the stdlib handlers ``caplog`` installs never see the event."""
    _enable(monkeypatch, on=False)
    await _ingest()
    capsys.readouterr()  # drop the ingest's own lines so the assertion is about the QUERY
    await _ask()

    out = capsys.readouterr().out
    assert "query_stage_timing" in out
    assert '"retrieve"' in out and '"generate"' in out and '"total"' in out
    # The line is safe to ship anywhere: it names no question and no answer.
    assert "Earnest Money Deposit" not in out.split("query_stage_timing")[1].split("\n")[0]


# --- what the row holds ----------------------------------------------------------------------------


async def test_a_row_carries_the_per_stage_latency(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, on=True)
    await _ingest()
    await _ask()

    rows = await _twin().query_log(_TENANT)
    assert len(rows) == 1
    latency = rows[0]["latency_ms"]
    # The stages the plan's §0 targets are about, each a DURATION and not an offset.
    assert {"guard", "cache", "route", "retrieve", "generate", "post", "total"} <= set(latency)
    assert all(isinstance(v, int) and v >= 0 for v in latency.values())
    assert latency["total"] >= max(v for k, v in latency.items() if k != "total")


async def test_a_row_is_attributable_to_the_documents_it_quoted(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``cited_doc_ids`` is the ERASURE KEY: it is the only thing that lets a document-scoped
    delete reach a row whose answer quotes that document."""
    _enable(monkeypatch, on=True)
    await _ingest()
    answer = await _ask()

    row = (await _twin().query_log(_TENANT))[0]
    if answer.citations:
        assert row["cited_doc_ids"] == [_DOC]
    else:  # an abstain cites nothing, and a row that cited nothing must not claim otherwise
        assert row["cited_doc_ids"] == []


async def test_the_question_is_the_users_own_text(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retrieval may run on glossary-expanded text; the row records what was ASKED."""
    _enable(monkeypatch, on=True)
    await _ingest()
    await _ask("what Earnest Money Deposit must a bidder furnish?")

    assert (await _twin().query_log(_TENANT))[0]["query"] == "what Earnest Money Deposit must a bidder furnish?"


async def test_the_free_text_columns_are_bounded_and_say_they_were_cut(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row is an operations record, not a transcript. Truncation is marked, never silent."""
    _enable(monkeypatch, on=True, RAG_QUERY_LOG_MAX_QUERY_CHARS="120", RAG_QUERY_LOG_MAX_ANSWER_CHARS="100")
    await _ingest()
    await _ask("what Earnest Money Deposit must a bidder furnish, and " + "and why ".join([""] * 60))

    row = (await _twin().query_log(_TENANT))[0]
    assert len(row["query"]) <= 120 and row["query"].endswith("…")
    assert len(row["answer"]) <= 100


async def test_the_phase_seven_to_ten_columns_stay_null(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing routes, selects sections or verifies per claim yet. A dashboard reading "no
    sections selected" where the truth is "nothing selects sections" is worse than an empty cell."""
    _enable(monkeypatch, on=True)
    await _ingest()
    await _ask()

    row = (await _twin().query_log(_TENANT))[0]
    assert row["router_json"] is None and row["selected_sections"] is None and row["verification"] is None


# --- what this feature must not do -------------------------------------------------------------------


async def test_no_extra_audit_event_is_recorded(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`count_actions` meters QUERY events for FinOps showback, so an extra event per question
    would silently inflate every tenant's bill. Turning the query log ON must not change the
    audit trail at all."""
    _enable(monkeypatch, on=False)
    await _ingest()
    await _ask()
    without = await _query_actions()

    _enable(monkeypatch, on=True)
    await _ask()
    with_log = await _query_actions()

    assert with_log - without == without, "the second question added exactly as many events as the first"


async def test_a_failing_log_write_does_not_fail_the_answer(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The answer is already correct, grounded and paid for."""
    _enable(monkeypatch, on=True)
    await _ingest()

    async def _boom(*_a: object, **_k: object) -> int:
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr(pipeline, "record_query_log", _boom)
    answer = await _ask()

    assert answer.query, "the answer was still returned"
    assert await _twin().query_log(_TENANT) == []


async def test_erasing_the_document_takes_the_rows_that_quoted_it(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end, which is the whole point: before Phase 12 nothing wrote the table, so the gap
    `PostgresCatalog.delete_document` flagged could not leak. Now it could."""
    _enable(monkeypatch, on=True)
    await _ingest()
    answer = await _ask()
    if not answer.citations:
        pytest.skip("the twin abstained, so no row cites the document and there is nothing to erase")
    assert await _twin().query_log(_TENANT)

    await get_catalog().delete_document(_TENANT, _DOC)

    assert await _twin().query_log(_TENANT) == []
