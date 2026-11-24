"""Unified KB — gather structured evidence from every source connected to a KB.

A knowledge base can carry more than documents: a SQL database (:mod:`sql_query`) and — through
the same seam — a NoSQL store, a REST API, or a SharePoint site. Each is an EVIDENCE PRODUCER:
given a KB id and a natural-language question it returns ONE authoritative
:class:`RetrievedSpan` (score 1.0, badged with its own modality) or ``None`` when it is not
connected to that KB or has no answer. ``answer_query`` fuses whatever the producers return
alongside document spans, so a single grounded answer can draw on documents AND any connected
structured source — each shown as its own citation.

Fail-safe by construction: a producer that is unconfigured, errors, or finds nothing contributes
nothing and NEVER fails the query. Producers are registered here (the SQL one below); a new
connector module implements the same ``(kb_id, question, *, actor) -> RetrievedSpan | None``
shape and is registered with :func:`register_producer`.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

from rag_common import get_logger
from rag_schemas import RetrievedSpan
from rag_schemas.common import AuditAction
from svc_control_plane import nosql_query, rest_source, sharepoint_source, sql_query

_log = get_logger(__name__)


async def _record_source_query(kb_id: str, question: str, span: RetrievedSpan, *, actor: str) -> None:
    """Audit that a connected structured source contributed to a fused answer (SOURCE_QUERY).
    Fail-safe — a SIEM/audit hiccup must never fail the query. wiring is imported locally to avoid a
    module-load import cycle."""
    from svc_control_plane import wiring  # noqa: PLC0415

    try:
        await wiring.audit_service().record(
            tenant_id=kb_id,
            trace_id=str(uuid.uuid4()),
            action=AuditAction.SOURCE_QUERY,
            actor=actor,
            payload={"source": span.doc_id, "modality": str(span.modality), "question": question},
        )
    except Exception as exc:
        _log.warning("source_query_audit_failed", kb_id=kb_id, error=str(exc)[:160])


# (kb_id, question, *, actor) -> an authoritative evidence span, or None when not connected/empty.
EvidenceProducer = Callable[..., Awaitable[RetrievedSpan | list[RetrievedSpan] | None]]

# The registered structured-evidence producers. Document retrieval is NOT here — it is the base
# path; these are the extra structured sources fused on top. The parent registers a connector's
# producer once its module is wired.
_PRODUCERS: list[EvidenceProducer] = [
    sql_query.sql_evidence_span,
    nosql_query.nosql_evidence_span,
    rest_source.rest_evidence_span,
    sharepoint_source.sharepoint_evidence_span,
]


def register_producer(producer: EvidenceProducer) -> None:
    """Add a structured-evidence producer (idempotent). Called once per connector at wiring time."""
    if producer not in _PRODUCERS:
        _PRODUCERS.append(producer)


async def gather_evidence(kb_id: str, question: str, *, actor: str) -> list[RetrievedSpan]:
    """Every connected source's authoritative span for this question (documents excluded). A
    producer that raises is logged and skipped — one bad source never denies the others or the
    query."""
    spans: list[RetrievedSpan] = []
    for producer in _PRODUCERS:
        try:
            result = await producer(kb_id, question, actor=actor)
        except Exception as exc:
            _log.warning(
                "unified_producer_failed",
                producer=getattr(producer, "__name__", "?"),
                error=str(exc)[:160],
            )
            continue
        # A producer may return one span, a LIST of spans (e.g. one per DB connection), or None.
        produced = result if isinstance(result, list) else [] if result is None else [result]
        for span in produced:
            spans.append(span)
            await _record_source_query(kb_id, question, span, actor=actor)
    return spans
