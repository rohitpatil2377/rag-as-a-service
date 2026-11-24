"""Unified KB (documents + database) — structured DB result fused as a cited evidence span."""

from __future__ import annotations

import pytest
from svc_control_plane import pipeline, sql_query, unified
from svc_control_plane.sql_query import (
    SqlAnswer,
    SqlConnectionItem,
    render_sql_evidence,
    sql_evidence_span,
)

from rag_common import get_settings
from rag_core.grounding import gate_for, should_abstain
from rag_schemas import RetrievalMode, RetrievalQuery, RetrievalResult, RetrievedSpan
from rag_schemas.common import AuditAction, Modality


def test_render_sql_evidence_table_and_truncation() -> None:
    ans = SqlAnswer(sql="SELECT id FROM t", columns=["id"], rows=[[i] for i in range(25)], row_count=25)
    text = render_sql_evidence(ans, max_rows=20)
    assert "SELECT id FROM t" in text  # provenance: the exact SQL
    assert "id" in text.splitlines()[2]  # header row
    assert "... (5 more row(s))" in text  # truncated beyond max_rows


def _conn_item() -> SqlConnectionItem:
    return SqlConnectionItem(conn_id="c1", name="Deals DB", dialect="postgres", tables=["deals"], is_default=True)


async def test_span_none_when_no_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _no_conns(_kb: str) -> list[SqlConnectionItem]:
        return []

    monkeypatch.setattr(sql_query, "list_connections", _no_conns)
    assert await sql_evidence_span("bank__deals", "how many deals", actor="u") == []


async def test_span_built_from_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _conns(_kb: str) -> list[SqlConnectionItem]:
        return [_conn_item()]

    async def _ask(_kb: str, _q: str, *, connection_id: str | None = None, actor: str) -> SqlAnswer:
        return SqlAnswer(sql="SELECT count(*) FROM deals", columns=["n"], rows=[[42]], row_count=1)

    monkeypatch.setattr(sql_query, "list_connections", _conns)
    monkeypatch.setattr(sql_query, "ask", _ask)
    spans = await sql_evidence_span("bank__deals", "how many deals", actor="u")
    assert len(spans) == 1
    span = spans[0]
    assert span.scorer == "sql"
    assert span.score == 1.0
    assert span.modality is Modality.DATABASE
    assert span.doc_id == "db://bank__deals/Deals DB"
    assert "42" in span.text


async def test_span_failsafe_on_db_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _conns(_kb: str) -> list[SqlConnectionItem]:
        return [_conn_item()]

    async def _boom(_kb: str, _q: str, *, connection_id: str | None = None, actor: str) -> SqlAnswer:
        raise RuntimeError("db down")

    monkeypatch.setattr(sql_query, "list_connections", _conns)
    monkeypatch.setattr(sql_query, "ask", _boom)
    assert await sql_evidence_span("bank__deals", "q", actor="u") == []  # degrades, no raise


async def test_span_none_on_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _conns(_kb: str) -> list[SqlConnectionItem]:
        return [_conn_item()]

    async def _empty(_kb: str, _q: str, *, connection_id: str | None = None, actor: str) -> SqlAnswer:
        return SqlAnswer(sql="SELECT 1 WHERE false", columns=["x"], rows=[], row_count=0)

    monkeypatch.setattr(sql_query, "list_connections", _conns)
    monkeypatch.setattr(sql_query, "ask", _empty)
    assert await sql_evidence_span("bank__deals", "q", actor="u") == []


async def test_spans_from_every_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    conns = [
        SqlConnectionItem(conn_id="c1", name="Sales DB", dialect="postgres", tables=["orders"], is_default=True),
        SqlConnectionItem(conn_id="c2", name="HR DB", dialect="postgres", tables=["staff"], is_default=False),
    ]

    async def _conns(_kb: str) -> list[SqlConnectionItem]:
        return conns

    async def _ask(_kb: str, _q: str, *, connection_id: str | None = None, actor: str) -> SqlAnswer:
        return SqlAnswer(sql=f"SELECT 1 -- {connection_id}", columns=["n"], rows=[[1]], row_count=1)

    monkeypatch.setattr(sql_query, "list_connections", _conns)
    monkeypatch.setattr(sql_query, "ask", _ask)
    spans = await sql_evidence_span("bank__deals", "q", actor="u")
    assert len(spans) == 2  # one authoritative span PER connected DB (FR-D10)
    assert {s.doc_id for s in spans} == {"db://bank__deals/Sales DB", "db://bank__deals/HR DB"}


def test_sql_span_is_confident_at_the_abstain_gate() -> None:
    span = RetrievedSpan(chunk_id="sql::k", doc_id="db://k", text="42", score=1.0, scorer="sql")
    gate = gate_for([span], cross_encoder=0.15, lexical=0.60)
    assert gate == 0.15  # exact DB hit judged on the cross-encoder scale, not UNSCORED (1.0)
    assert should_abstain(1.0, gate) is False  # an exact DB answer does NOT abstain


# --- Pipeline fusion (_fuse_sql_evidence) --------------------------------------------


def _rq() -> RetrievalQuery:
    return RetrievalQuery(tenant_id="bank__deals", text="how many deals", top_k=5, acl=[])


def _rr(spans: list[RetrievedSpan], max_score: float) -> RetrievalResult:
    return RetrievalResult(query=_rq(), spans=spans, max_score=max_score, mode_used=RetrievalMode.HYBRID)


async def test_fuse_disabled_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_UNIFIED_KB_ENABLED", "false")
    get_settings.cache_clear()
    rr = _rr([], 0.0)
    assert await pipeline._fuse_structured_evidence(_rq(), rr, actor="u") is rr  # unchanged
    get_settings.cache_clear()


async def test_fuse_unions_source_spans_and_lifts_score(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_UNIFIED_KB_ENABLED", "true")
    get_settings.cache_clear()
    db = RetrievedSpan(chunk_id="sql::k", doc_id="db://k", text="42", score=1.0, scorer="sql")

    async def _gather(_kb: str, _q: str, *, actor: str) -> list[RetrievedSpan]:
        return [db]

    monkeypatch.setattr(pipeline.unified, "gather_evidence", _gather)  # type: ignore[attr-defined]
    doc = RetrievedSpan(chunk_id="c1", doc_id="d1", text="doc", score=0.3, scorer="cross_encoder")
    out = await pipeline._fuse_structured_evidence(_rq(), _rr([doc], 0.3), actor="u")
    assert len(out.spans) == 2
    assert out.spans[0].scorer == "sql"  # structured span fused alongside the document span
    assert out.max_score == 1.0  # exact source hit lifts confidence past the gate
    get_settings.cache_clear()


async def test_fuse_failsafe_when_no_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_UNIFIED_KB_ENABLED", "true")
    get_settings.cache_clear()

    async def _empty(_kb: str, _q: str, *, actor: str) -> list[RetrievedSpan]:
        return []

    monkeypatch.setattr(pipeline.unified, "gather_evidence", _empty)  # type: ignore[attr-defined]
    rr = _rr([], 0.0)
    assert await pipeline._fuse_structured_evidence(_rq(), rr, actor="u") is rr  # documents-only fallback
    get_settings.cache_clear()


# --- audit parity: every contributing source records a SOURCE_QUERY event ------------


async def test_gather_evidence_audits_each_source(monkeypatch: pytest.MonkeyPatch) -> None:
    from svc_control_plane import wiring

    span = RetrievedSpan(chunk_id="rest::k", doc_id="api://acme", text="x", score=1.0, scorer="exact")

    async def _producer(_kb: str, _q: str, *, actor: str) -> RetrievedSpan:
        return span

    monkeypatch.setattr(unified, "_PRODUCERS", [_producer])

    events: list[tuple[AuditAction, dict[str, object]]] = []

    class _Audit:
        async def record(
            self, *, tenant_id: str, trace_id: str, action: AuditAction, actor: str, payload: dict[str, object]
        ) -> str:
            events.append((action, payload))
            return "h"

    monkeypatch.setattr(wiring, "audit_service", _Audit)
    spans = await unified.gather_evidence("k", "how many deals?", actor="u")
    assert len(spans) == 1
    assert len(events) == 1
    assert events[0][0] is AuditAction.SOURCE_QUERY
    assert events[0][1]["source"] == "api://acme"
    assert events[0][1]["question"] == "how many deals?"


async def test_source_audit_never_fails_the_query(monkeypatch: pytest.MonkeyPatch) -> None:
    span = RetrievedSpan(chunk_id="rest::k", doc_id="api://acme", text="x", score=1.0, scorer="exact")

    async def _producer(_kb: str, _q: str, *, actor: str) -> RetrievedSpan:
        return span

    monkeypatch.setattr(unified, "_PRODUCERS", [_producer])

    from svc_control_plane import wiring

    class _BadAudit:
        async def record(self, **_kw: object) -> str:
            raise RuntimeError("SIEM down")

    monkeypatch.setattr(wiring, "audit_service", _BadAudit)
    spans = await unified.gather_evidence("k", "q", actor="u")
    assert len(spans) == 1  # audit failure did not fail the query
