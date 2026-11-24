"""Tests for the ingestion DAG — static shape, Mermaid rendering, and the per-document audit
overlay (svc_control_plane.ingest_dag)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from svc_control_plane.ingest_dag import (
    DagNode,
    IngestDag,
    document_trace,
    pipeline_dag,
    to_mermaid,
    trust_summary,
)

from rag_schemas import AuditAction, AuditEvent


def _event(action: AuditAction, payload: dict[str, object], *, event_hash: str = "") -> AuditEvent:
    return AuditEvent(
        event_id="evt-1",
        tenant_id="kb-1",
        trace_id="trace-1",
        action=action,
        actor="tester",
        at=datetime(2026, 1, 1, tzinfo=UTC),
        payload=payload,
        event_hash=event_hash,
    )


class _FakeAudit:
    """Structural stand-in for the injected wiring audit service — only `list` is needed."""

    def __init__(self, events: list[AuditEvent]) -> None:
        self._events = events

    async def list(self, tenant_id: str, trace_id: str | None = None, limit: int = 100) -> list[AuditEvent]:
        return self._events


class _RaisingAudit:
    async def list(self, tenant_id: str, trace_id: str | None = None, limit: int = 100) -> list[AuditEvent]:
        raise RuntimeError("audit backend unreachable")


# --- pipeline_dag -------------------------------------------------------------------------------


def test_pipeline_dag_has_expected_stages() -> None:
    dag = pipeline_dag()
    ids = {n.id for n in dag.nodes}
    assert ids == {
        "ingest",
        "object",
        "doc_intel",
        "sanitize",
        "quarantine",
        "chunk",
        "contextualize",
        "embed",
        "vector",
        "sparse",
        "catalog",
        "index",
        "graphrag",
    }
    # every node starts "defined" with no fabricated evidence
    assert all(n.status == "defined" for n in dag.nodes)
    assert all(n.event_hash == "" for n in dag.nodes)
    assert all(n.metrics == {} for n in dag.nodes)


def test_pipeline_dag_real_order_start_to_graph() -> None:
    dag = pipeline_dag()
    adjacency: dict[str, set[str]] = {}
    for e in dag.edges:
        adjacency.setdefault(e.source, set()).add(e.target)

    def reachable(src: str, dst: str) -> bool:
        seen: set[str] = set()
        stack = [src]
        while stack:
            cur = stack.pop()
            if cur == dst:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(adjacency.get(cur, ()))
        return False

    # the spine: ingest -> doc-intelligence -> sanitize -> chunk -> contextualize -> embed
    assert reachable("ingest", "doc_intel")
    assert reachable("doc_intel", "sanitize")
    assert reachable("sanitize", "chunk")
    assert reachable("chunk", "contextualize")
    assert reachable("contextualize", "embed")
    # embed fans out to the three index stores, which converge on the index checkpoint
    for store in ("vector", "sparse", "catalog"):
        assert reachable("embed", store)
        assert reachable(store, "index")
    # graph runs only after the index checkpoint
    assert reachable("index", "graphrag")
    assert reachable("ingest", "graphrag")
    # the raw-bytes object write happens at ingest, not after indexing
    assert reachable("ingest", "object")
    assert not reachable("index", "object")


def test_pipeline_dag_no_dangling_edges() -> None:
    dag = pipeline_dag()
    ids = {n.id for n in dag.nodes}
    for e in dag.edges:
        assert e.source in ids, f"edge source {e.source!r} is not a node id"
        assert e.target in ids, f"edge target {e.target!r} is not a node id"


# --- to_mermaid -----------------------------------------------------------------------------


def test_to_mermaid_is_deterministic() -> None:
    dag = pipeline_dag()
    assert to_mermaid(dag) == to_mermaid(dag)
    assert to_mermaid(pipeline_dag()) == to_mermaid(pipeline_dag())


def test_to_mermaid_marks_ran_node_with_class() -> None:
    dag = pipeline_dag()
    for n in dag.nodes:
        if n.id == "ingest":
            n.status = "ran"
    rendered = to_mermaid(dag)
    assert "flowchart LR" in rendered
    assert "class ingest ran;" in rendered


def test_to_mermaid_has_classdefs_and_subgraphs() -> None:
    rendered = to_mermaid(pipeline_dag())
    assert "classDef ran fill:#dcfce7,stroke:#16a34a,color:#14532d;" in rendered
    assert "classDef failed fill:#fee2e2,stroke:#dc2626,color:#7f1d1d;" in rendered
    assert "classDef skipped stroke-dasharray:4;" in rendered
    assert 'subgraph grp_doc_intelligence ["Doc-Intelligence"]' in rendered
    assert 'subgraph grp_indexing ["Indexing fan-out"]' in rendered
    assert 'subgraph grp_graph ["Graph"]' in rendered


def test_to_mermaid_no_status_produces_no_class_lines() -> None:
    rendered = to_mermaid(pipeline_dag())
    assert "class " not in rendered  # nothing ran/failed/skipped -> nothing to style


# --- document_trace -------------------------------------------------------------------------


async def test_document_trace_marks_confirmed_stages_ran() -> None:
    events = [
        _event(
            AuditAction.INGEST,
            {"doc_id": "doc-1", "object_uri": "s3://x", "content_hash": "abc123"},
            event_hash="h_ingest",
        ),
        _event(
            AuditAction.INDEX,
            {"doc_id": "doc-1", "chunks": 42, "pii_masked": 3, "replaced_chunks": 0},
            event_hash="h_index",
        ),
        _event(AuditAction.GRAPH, {"doc_id": "doc-1", "chunks": 42}, event_hash="h_graph"),
    ]
    dag = await document_trace("kb-1", "doc-1", audit=_FakeAudit(events))
    by_id = {n.id: n for n in dag.nodes}

    # A successful INDEX proves the whole pre-index happy path ran (doc-intel + sanitize included);
    # the GRAPH event proves graphrag ran. Nothing is left falsely "still pending" on a done doc.
    for confirmed in (
        "ingest",
        "object",
        "doc_intel",
        "sanitize",
        "chunk",
        "contextualize",
        "embed",
        "vector",
        "sparse",
        "catalog",
        "index",
        "graphrag",
    ):
        assert by_id[confirmed].status == "ran", confirmed

    # a successful index proves the document was never quarantined
    assert by_id["quarantine"].status == "skipped"
    assert by_id["graphrag"].event_hash == "h_graph"

    assert dag.doc_id == "doc-1"
    assert by_id["index"].metrics.get("chunks") == 42


async def test_document_trace_quarantined_document_skips_downstream() -> None:
    events = [
        _event(
            AuditAction.INGEST,
            {"doc_id": "doc-2", "object_uri": "s3://y", "content_hash": "def456"},
            event_hash="h_ingest",
        ),
        _event(
            AuditAction.GUARDRAIL,
            {"doc_id": "doc-2", "stage": "ingest", "reason": "prompt_injection_suspected", "hits": "3"},
            event_hash="h_guardrail",
        ),
    ]
    dag = await document_trace("kb-1", "doc-2", audit=_FakeAudit(events))
    by_id = {n.id: n for n in dag.nodes}

    assert by_id["sanitize"].status == "ran"
    assert by_id["quarantine"].status == "ran"
    for skipped in ("chunk", "contextualize", "embed", "vector", "sparse", "catalog", "index", "graphrag"):
        assert by_id[skipped].status == "skipped", skipped


async def test_document_trace_unrelated_doc_id_leaves_static_shape() -> None:
    events = [_event(AuditAction.INGEST, {"doc_id": "someone-else"}, event_hash="h")]
    dag = await document_trace("kb-1", "doc-3", audit=_FakeAudit(events))
    assert all(n.status == "defined" for n in dag.nodes)


async def test_document_trace_fail_safe_on_audit_error() -> None:
    dag = await document_trace("kb-1", "doc-4", audit=_RaisingAudit())
    static = pipeline_dag()
    assert [n.id for n in dag.nodes] == [n.id for n in static.nodes]
    assert all(n.status == "defined" for n in dag.nodes)


# --- enrichment: metrics / event_hash / trust_summary -------------------------------------------


async def test_document_trace_populates_metrics_and_event_hash_from_fake_audit() -> None:
    events = [
        _event(
            AuditAction.INGEST,
            {"doc_id": "doc-5", "object_uri": "s3://z", "content_hash": "aa11bb22cc"},
            event_hash="h_ingest_abcdef12",
        ),
        _event(
            AuditAction.INDEX,
            {"doc_id": "doc-5", "chunks": 7, "pii_masked": 1, "replaced_chunks": 2},
            event_hash="h_index_98765432",
        ),
    ]
    dag = await document_trace("kb-1", "doc-5", audit=_FakeAudit(events))
    by_id = {n.id: n for n in dag.nodes}

    assert by_id["ingest"].event_hash == "h_ingest_abcdef12"
    assert by_id["ingest"].metrics["content_hash"] == "aa11bb22cc"
    assert by_id["index"].event_hash == "h_index_98765432"
    assert by_id["index"].metrics["chunks"] == 7
    assert by_id["index"].metrics["replaced_chunks"] == 2
    # doc-intel is inferred-ran from the successful INDEX and carries that event's hash (indexing
    # cannot happen without extraction). It surfaces the real doc-intelligence signal present in the
    # INDEX payload (here just pii_masked) — nothing is fabricated.
    assert by_id["doc_intel"].status == "ran"
    assert by_id["doc_intel"].event_hash == "h_index_98765432"
    assert by_id["doc_intel"].metrics == {"pii_masked": 1}
    # quarantine was provably NOT entered (indexing succeeded) -> skipped, no event hash.
    assert by_id["quarantine"].status == "skipped"
    assert by_id["quarantine"].event_hash == ""


def test_trust_summary_verified_when_all_ran_stages_have_hashes() -> None:
    dag = pipeline_dag()
    for n in dag.nodes:
        if n.id in ("ingest", "object", "chunk"):
            n.status = "ran"
            n.event_hash = f"hash-{n.id}"
    summary = trust_summary(dag, chain_intact=True)
    assert summary == {
        "stages_total": len(dag.nodes),
        "stages_ran": 3,
        "verified": True,
        "note": "All 3 ran stage(s) are backed by tamper-evident audit event hashes; chain intact.",
    }


def test_trust_summary_not_verified_when_chain_broken() -> None:
    dag = pipeline_dag()
    for n in dag.nodes:
        if n.id == "ingest":
            n.status = "ran"
            n.event_hash = "hash-ingest"
    summary = trust_summary(dag, chain_intact=False)
    assert summary["verified"] is False
    assert summary["stages_ran"] == 1


def test_trust_summary_not_verified_when_ran_stage_missing_hash() -> None:
    dag = pipeline_dag()
    for n in dag.nodes:
        if n.id == "ingest":
            n.status = "ran"  # event_hash left empty
    summary = trust_summary(dag, chain_intact=True)
    assert summary["verified"] is False


def test_trust_summary_no_ran_stages_never_verified() -> None:
    summary = trust_summary(pipeline_dag(), chain_intact=True)
    assert summary["verified"] is False
    assert summary["stages_ran"] == 0


def test_dag_node_new_fields_default_safely() -> None:
    node = DagNode(id="x", label="X", stage="x")
    assert node.metrics == {}
    assert node.duration_ms is None
    assert node.event_hash == ""


def test_ingest_dag_model_roundtrip() -> None:
    dag = pipeline_dag()
    assert isinstance(dag, IngestDag)
    dumped = dag.model_dump()
    restored = IngestDag.model_validate(dumped)
    assert restored == dag


def test_live_progress_overlay_marks_ran_and_running() -> None:
    from svc_control_plane import ingest_dag as d
    from svc_control_plane import ingest_progress

    ingest_progress.reset("k", "doc-live")
    ingest_progress.mark("k", "doc-live", "ingest", "object", "doc_intel", "sanitize")
    dag = d.pipeline_dag()
    nb = {n.id: n for n in dag.nodes}
    d._apply_live_progress(nb, dag.edges, "k", "doc-live")
    assert nb["ingest"].status == "ran"
    assert nb["doc_intel"].status == "ran"
    assert nb["chunk"].status == "running"  # next stage after sanitize(ran)
    rendered = d.to_mermaid(dag.model_copy(update={"nodes": list(nb.values())}))
    assert "classDef running" in rendered and "class chunk running;" in rendered
    ingest_progress.reset("k", "doc-live")


def test_real_timing_overlay_attaches_at_ms_and_group_durations() -> None:
    from svc_control_plane import ingest_dag as d

    dag = d.pipeline_dag()
    nb = {n.id: n for n in dag.nodes}
    # Completion epochs: ingest@t0, doc-intel/sanitize one step later, the chunk..index bulk step,
    # then graph. Stages recorded together share an epoch (one bulk step in pipeline.py).
    live = {
        "ingest": 100.0,
        "object": 100.0,
        "doc_intel": 100.2,
        "sanitize": 100.2,
        "chunk": 101.5,
        "contextualize": 101.5,
        "embed": 101.5,
        "vector": 101.5,
        "sparse": 101.5,
        "catalog": 101.5,
        "index": 101.5,
        "graphrag": 101.7,
    }
    d._apply_real_timing(nb, live)
    # at_ms = each stage's completion offset from ingest start (t0=100.0).
    assert nb["ingest"].at_ms == 0
    assert nb["sanitize"].at_ms == 200
    assert nb["index"].at_ms == 1500
    assert nb["graphrag"].at_ms == 1700
    # duration is attributed to the stage that CLOSES each real timestamp boundary (last in group).
    assert nb["sanitize"].duration_ms == 200
    assert nb["index"].duration_ms == 1300
    assert nb["graphrag"].duration_ms == 200
    # Intermediate grouped stages carry no fabricated per-stage number.
    assert nb["doc_intel"].duration_ms is None
    assert nb["chunk"].duration_ms is None


def test_graph_stage_skipped_when_disabled_and_no_graph_event(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from svc_control_plane import ingest_dag as d

    monkeypatch.setattr(d, "get_settings", lambda: SimpleNamespace(graphrag_enabled=False))
    nb = {n.id: n for n in d.pipeline_dag().nodes}
    index_event = _event(AuditAction.INDEX, {"doc_id": "x", "chunks": 3}, event_hash="h_idx")
    d._apply_overlay(nb, [index_event])
    assert nb["doc_intel"].status == "ran"  # inferred from successful INDEX
    assert nb["sanitize"].status == "ran"
    assert nb["graphrag"].status == "skipped"  # graph disabled -> it no-ops, so not "still pending"


def test_graph_stage_inferred_ran_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from svc_control_plane import ingest_dag as d

    monkeypatch.setattr(d, "get_settings", lambda: SimpleNamespace(graphrag_enabled=True))
    nb = {n.id: n for n in d.pipeline_dag().nodes}
    index_event = _event(AuditAction.INDEX, {"doc_id": "x", "chunks": 3}, event_hash="h_idx")
    d._apply_overlay(nb, [index_event])
    # graph runs in the same ingest pass as indexing when enabled -> inferred ran, INDEX-backed.
    assert nb["graphrag"].status == "ran"
    assert nb["graphrag"].event_hash == "h_idx"


async def test_document_trace_durations_from_persisted_stage_ms() -> None:
    # A completed document, days later, has no in-memory live marks — durations must come from the
    # stage_ms persisted in the durable INDEX audit payload.
    events = [
        _event(AuditAction.INGEST, {"doc_id": "d", "object_uri": "s", "content_hash": "h"}, event_hash="h_i"),
        _event(
            AuditAction.INDEX,
            {
                "doc_id": "d",
                "chunks": 5,
                "stage_ms": {
                    "ingest": 0,
                    "doc_intel": 200,
                    "sanitize": 200,
                    "chunk": 1500,
                    "contextualize": 1500,
                    "embed": 1500,
                    "index": 1500,
                },
            },
            event_hash="h_x",
        ),
    ]
    dag = await document_trace("kb", "d", audit=_FakeAudit(events))
    by = {n.id: n for n in dag.nodes}
    assert by["index"].at_ms == 1500
    assert by["sanitize"].duration_ms == 200
    assert by["index"].duration_ms == 1300


async def test_stage_metrics_from_payloads() -> None:
    # doc-intel / sanitize / graphrag show real signal from the audit payloads instead of empty cells.
    events = [
        _event(AuditAction.INGEST, {"doc_id": "m", "object_uri": "s", "content_hash": "h"}, event_hash="h_i"),
        _event(
            AuditAction.INDEX,
            {
                "doc_id": "m",
                "chunks": 4,
                "pii_masked": 2,
                "modality": "text",
                "classification": "internal",
                "injection": "clean",
            },
            event_hash="h_x",
        ),
        _event(AuditAction.GRAPH, {"doc_id": "m", "chunks": 4, "entities": 9, "edges": 12}, event_hash="h_g"),
    ]
    dag = await document_trace("kb", "m", audit=_FakeAudit(events))
    by = {n.id: n for n in dag.nodes}
    assert by["doc_intel"].metrics.get("modality") == "text"
    assert by["doc_intel"].metrics.get("classification") == "internal"
    assert by["sanitize"].metrics.get("injection") == "clean"
    assert by["graphrag"].metrics.get("entities") == 9
    assert by["graphrag"].metrics.get("edges") == 12
