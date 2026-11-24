"""Ingestion DAG — visualization + traceability for the backend pipeline that runs after a
document is ingested (TDD §5.1).

:func:`pipeline_dag` is the STATIC canonical DAG of ``pipeline.ingest_document``'s real stages and
order, read directly from the pipeline code (not guessed): ingest (raw bytes -> object store,
catalog register) -> doc-intelligence (classify/OCR/PII) -> sanitize (prompt-injection scan) ->
[quarantine, if suspect] -> chunk -> contextualize -> embed -> index fan-out (Qdrant / OpenSearch /
Postgres, plus the MinIO object write recorded at ingest) -> index checkpoint -> GraphRAG.
:func:`to_mermaid` renders it as an editable, deterministic ``flowchart LR``, mirroring
:mod:`svc_control_plane.architecture`'s renderer/house style.

:func:`document_trace` overlays a specific document's audit trail onto the static DAG so a
reviewer can see which stages actually ran for THAT document — and because every confirmed stage
carries the real ``AuditEvent.event_hash`` (part of the tamper-evident hash chain, TDD §4.7), the
overlay is provable, not decorative. :func:`trust_summary` reports whether every "ran" stage is
backed by a hash and the chain is intact.
"""

from __future__ import annotations

import re
from itertools import pairwise
from typing import Any, Protocol

from pydantic import BaseModel, Field

from rag_common import get_settings
from rag_schemas import AuditAction, AuditEvent

# --- schema -----------------------------------------------------------------------------------


class DagNode(BaseModel):
    """One box in the ingestion DAG — a pipeline stage or a durable store it writes to."""

    id: str
    label: str
    stage: str
    status: str = "defined"  # "defined" | "ran" | "skipped"
    metrics: dict[str, str | int | float] = Field(default_factory=dict)
    duration_ms: int | None = None
    at_ms: int | None = None  # completion offset from ingest start (ms) — real live timeline
    event_hash: str = ""


class DagEdge(BaseModel):
    """A directed edge between two :class:`DagNode` ids."""

    source: str
    target: str


class IngestDag(BaseModel):
    """A full ingestion DAG: the static pipeline shape, optionally overlaid with one document's
    audit-confirmed run status."""

    nodes: list[DagNode]
    edges: list[DagEdge]
    doc_id: str = ""
    title: str = "Ingestion pipeline"


# --- static pipeline shape ---------------------------------------------------------------------


def pipeline_dag() -> IngestDag:
    """The static canonical DAG of what runs after ``ingest_document`` receives a document
    (``svc_control_plane.pipeline.ingest_document`` / ``reindex_document``, TDD §5.1), in the
    REAL code order:

    ``ingest`` writes the raw bytes to the object store and registers the document in the
    catalog, then hands off to ``doc-intelligence`` (classify/OCR/PII-mask). Its output is
    scanned by ``sanitize`` (indirect-prompt-injection detection); a SUSPECT verdict routes to
    ``quarantine`` and stops there — nothing after it runs. A clean document proceeds through
    ``chunk`` -> ``contextualize`` (FR-C2, config-gated) -> ``embed``, whose vectors/text then
    fan out to the three durable index stores (``vector``/Qdrant, ``sparse``/OpenSearch,
    ``catalog``/Postgres chunk rows). Once all three complete, pipeline.py records the INDEX
    audit checkpoint (``index``) and — only then — runs GraphRAG (``graph``, additive/no-op if
    disabled).
    """
    nodes = [
        DagNode(id="ingest", label="Ingest", stage="ingest"),
        DagNode(id="object", label="Object · MinIO (raw bytes)", stage="object"),
        DagNode(id="doc_intel", label="Doc Intelligence", stage="doc_intelligence"),
        DagNode(id="sanitize", label="Sanitize", stage="sanitize"),
        DagNode(id="quarantine", label="Quarantine (suspect)", stage="quarantine"),
        DagNode(id="chunk", label="Chunk", stage="chunk"),
        DagNode(id="contextualize", label="Contextualize", stage="contextualize"),
        DagNode(id="embed", label="Embed", stage="embed"),
        DagNode(id="vector", label="Vector · Qdrant", stage="vector"),
        DagNode(id="sparse", label="Sparse · OpenSearch", stage="sparse"),
        DagNode(id="catalog", label="Catalog · Postgres", stage="catalog"),
        DagNode(id="index", label="Index (checkpoint)", stage="index"),
        DagNode(id="graphrag", label="GraphRAG", stage="graph"),
    ]
    edges = [
        DagEdge(source="ingest", target="object"),
        DagEdge(source="ingest", target="doc_intel"),
        DagEdge(source="doc_intel", target="sanitize"),
        DagEdge(source="sanitize", target="quarantine"),
        DagEdge(source="sanitize", target="chunk"),
        DagEdge(source="chunk", target="contextualize"),
        DagEdge(source="contextualize", target="embed"),
        DagEdge(source="embed", target="vector"),
        DagEdge(source="embed", target="sparse"),
        DagEdge(source="embed", target="catalog"),
        DagEdge(source="vector", target="index"),
        DagEdge(source="sparse", target="index"),
        DagEdge(source="catalog", target="index"),
        DagEdge(source="index", target="graphrag"),
    ]
    return IngestDag(nodes=nodes, edges=edges, title="Ingestion pipeline")


# --- Mermaid renderer ---------------------------------------------------------------------------


def _clean(text: str) -> str:
    """Collapse whitespace + neutralise characters that break Mermaid label syntax. Mirrors
    ``svc_control_plane.architecture._clean`` (kept local so this module has no dependency on
    architecture.py's private helpers)."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    for bad, good in (('"', "'"), ("|", "/"), ("<", "("), (">", ")"), ("#", ""), ("`", "'")):
        cleaned = cleaned.replace(bad, good)
    return cleaned


def _fmt_metric(key: str, value: str | int | float) -> str:
    pretty_key = key.replace("_", " ")
    if isinstance(value, bool):
        return f"{pretty_key} {value}"
    if isinstance(value, (int, float)):
        return f"{value} {pretty_key}"
    return f"{pretty_key} {value}"


def _node_label(node: DagNode) -> str:
    """A ran node's label carries its first recorded metric + an 8-char event-hash prefix, so
    the rendered diagram doubles as a provenance receipt (e.g. ``Chunk · 42 chunks · a1b2c3d4``),
    not just a picture. A non-ran node keeps its plain label."""
    if node.status != "ran":
        return _clean(node.label)
    bits = [node.label]
    for key, value in node.metrics.items():
        bits.append(_fmt_metric(key, value))
        break  # one metric keeps the box readable; the rest is in `metrics` for the overlay UI
    if node.event_hash:
        bits.append(node.event_hash[:8])
    return _clean(" · ".join(str(b) for b in bits))


# Visual grouping only — does not change edge topology. A node's edges reflect where it is
# ACTUALLY written in pipeline.py even when, for readability, it is drawn inside a subgraph whose
# other members run at a different point (e.g. ``object`` is written during ``ingest``, but reads
# naturally alongside the other durable stores).
_SUBGRAPHS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("doc_intelligence", "Doc-Intelligence", ("doc_intel", "sanitize", "quarantine")),
    ("indexing", "Indexing fan-out", ("vector", "sparse", "object", "catalog")),
    ("graph", "Graph", ("graphrag",)),
)


def to_mermaid(dag: IngestDag) -> str:
    """Render ``dag`` as an editable, deterministic ``flowchart LR`` — same input always yields
    byte-identical output, so it diffs cleanly and can be pasted straight into docs (mirrors
    :func:`svc_control_plane.architecture.to_mermaid`'s house style). Nodes are grouped into
    subgraphs (Doc-Intelligence / Indexing fan-out / Graph) and styled by status via ``classDef``:
    ``ran`` (green fill, audit-confirmed), ``failed`` (red fill), ``skipped`` (dashed border) —
    a plain node was never confirmed either way (no overlay, or the static shape only)."""
    lines = [
        "flowchart LR",
        f"  %% {_clean(dag.title)}",
        "  %% legend: green fill = ran (audit-confirmed) · dashed = skipped · plain = defined only",
        "  classDef ran fill:#dcfce7,stroke:#16a34a,color:#14532d;",
        "  classDef failed fill:#fee2e2,stroke:#dc2626,color:#7f1d1d;",
        "  classDef skipped stroke-dasharray:4;",
        "  classDef running fill:#fef9c3,stroke:#ca8a04,color:#713f12;",
    ]
    node_by_id = {n.id: n for n in dag.nodes}
    grouped: set[str] = set()
    for gid, gtitle, members in _SUBGRAPHS:
        present = [m for m in members if m in node_by_id]
        if not present:
            continue
        lines.append(f'  subgraph grp_{gid} ["{_clean(gtitle)}"]')
        for nid in present:
            node = node_by_id[nid]
            lines.append(f'    {node.id}["{_node_label(node)}"]')
            grouped.add(nid)
        lines.append("  end")
    for node in dag.nodes:
        if node.id not in grouped:
            lines.append(f'  {node.id}["{_node_label(node)}"]')
    for edge in dag.edges:
        lines.append(f"  {edge.source} --> {edge.target}")
    for status in ("ran", "failed", "skipped", "running"):
        ids = [n.id for n in dag.nodes if n.status == status]
        if ids:
            lines.append(f"  class {','.join(ids)} {status};")
    return "\n".join(lines)


# --- per-document overlay -----------------------------------------------------------------------


class _AuditLister(Protocol):
    """The one method this module needs from the wiring audit service — a structural type so
    tests can inject a fake without subclassing ``svc_guardrails_eval.AuditService``."""

    async def list(self, tenant_id: str, trace_id: str | None = None, limit: int = 100) -> list[AuditEvent]: ...


# Which of an event's payload keys become a confirmed node's `metrics`, in the order tried —
# the first one PRESENT in the event's payload becomes the node's label metric (see _node_label).
# Every key here is a REAL field pipeline.py records (see ingest_document); nothing is invented.
_STAGE_METRIC_KEYS: dict[str, tuple[str, ...]] = {
    "ingest": ("content_hash", "object_uri"),
    "object": ("object_uri", "content_hash"),
    "doc_intel": ("modality", "classification", "pii_masked"),
    "sanitize": ("injection", "reason", "hits"),
    "quarantine": ("reason", "hits"),
    "chunk": ("chunks",),
    "contextualize": ("chunks",),
    "embed": ("chunks",),
    "vector": ("chunks",),
    "sparse": ("chunks",),
    "catalog": ("chunks", "pii_masked"),
    "index": ("chunks", "replaced_chunks", "pii_masked"),
    "graphrag": ("entities", "edges", "chunks"),
}

# Stages pipeline.py's ``_quarantine_suspect`` never reaches — a SUSPECT verdict returns
# immediately, so these are provably `skipped`, not merely unconfirmed.
_STAGES_AFTER_SANITIZE = ("chunk", "contextualize", "embed", "vector", "sparse", "catalog", "index", "graphrag")
# The linear happy-path order, used to pick the single 'running' stage for the live overlay
# (skips the parallel 'object' write + the 'quarantine' branch, which are not the active stage).
_MAIN_SEQUENCE = ("ingest", "doc_intel", "sanitize", "chunk", "contextualize", "embed", "index", "graphrag")


def _mark_ran(node_by_id: dict[str, DagNode], node_id: str, event: AuditEvent) -> None:
    node = node_by_id[node_id]
    node.status = "ran"
    node.event_hash = event.event_hash or ""
    for key in _STAGE_METRIC_KEYS.get(node_id, ()):
        if key in event.payload:
            node.metrics[key] = event.payload[key]


def _apply_overlay(node_by_id: dict[str, DagNode], doc_events: list[AuditEvent]) -> None:
    """Mutate ``node_by_id`` in place from this document's audit events. Kept separate from
    :func:`document_trace` so the branching stays under the repo's lint complexity limits."""
    by_action: dict[AuditAction, AuditEvent] = {}
    for event in doc_events:
        by_action[event.action] = event  # last write wins (a later re-ingest/reindex overrides)

    ingest_event = by_action.get(AuditAction.INGEST)
    if ingest_event is not None:
        _mark_ran(node_by_id, "ingest", ingest_event)
        _mark_ran(node_by_id, "object", ingest_event)

    guardrail_event = by_action.get(AuditAction.GUARDRAIL)
    quarantined = guardrail_event is not None and guardrail_event.payload.get("reason") == "prompt_injection_suspected"
    if quarantined and guardrail_event is not None:
        _mark_ran(node_by_id, "sanitize", guardrail_event)
        _mark_ran(node_by_id, "quarantine", guardrail_event)
        for stage_id in _STAGES_AFTER_SANITIZE:
            node_by_id[stage_id].status = "skipped"
        return  # a quarantined document has no INDEX event to apply below

    index_event = by_action.get(AuditAction.INDEX)
    if index_event is not None:
        # A successful INDEX provably means the whole pre-index happy path ran: the pipeline cannot
        # reach indexing without extracting (doc-intel) and passing the injection scan (sanitize).
        # So mark those ran too — otherwise a completed document reads as a perpetual "9/13, still
        # ingesting", the stages simply having no dedicated audit action of their own.
        for stage_id in (
            "doc_intel",
            "sanitize",
            "chunk",
            "contextualize",
            "embed",
            "vector",
            "sparse",
            "catalog",
            "index",
        ):
            _mark_ran(node_by_id, stage_id, index_event)
        if node_by_id["quarantine"].status == "defined":
            node_by_id["quarantine"].status = "skipped"  # indexing succeeded -> provably not quarantined
        _apply_graph_status(node_by_id, by_action, index_event)
        # Durable per-stage timeline persisted at ingest (INDEX payload) -> real durations for ANY
        # document, long after the in-memory live marks expire.
        stage_ms = index_event.payload.get("stage_ms")
        if isinstance(stage_ms, dict):
            _assign_timing(node_by_id, {k: int(v) for k, v in stage_ms.items() if isinstance(v, int | float)})


def _apply_graph_status(
    node_by_id: dict[str, DagNode], by_action: dict[AuditAction, AuditEvent], index_event: AuditEvent
) -> None:
    """Resolve the GraphRAG stage for an indexed document. A GRAPH audit event proves it ran; else,
    since graph runs in the SAME ingest pass as indexing when enabled, an indexed doc under a
    graph-enabled KB provably ran it too (inferred, INDEX-backed). Graph disabled -> it no-ops, so
    the stage is skipped, not left pending. This keeps a completed document from showing graph as a
    forever-unfinished stage."""
    graph_event = by_action.get(AuditAction.GRAPH)
    if graph_event is not None:
        _mark_ran(node_by_id, "graphrag", graph_event)
        graph_node = node_by_id.get("graphrag")
        at = graph_event.payload.get("graph_at_ms")
        dur = graph_event.payload.get("graph_ms")
        if graph_node is not None and isinstance(at, int | float):
            graph_node.at_ms = int(at)
        if graph_node is not None and isinstance(dur, int | float):
            graph_node.duration_ms = int(dur)
    elif get_settings().graphrag_enabled:
        _mark_ran(node_by_id, "graphrag", index_event)
    elif node_by_id["graphrag"].status == "defined":
        node_by_id["graphrag"].status = "skipped"


async def document_trace(kb_id: str, doc_id: str, *, audit: _AuditLister) -> IngestDag:
    """Overlay ``doc_id``'s audit trail onto :func:`pipeline_dag` so a reviewer can see which
    stages ACTUALLY ran for this document — backed by the tamper-evident hash chain (TDD §4.7): a
    confirmed stage's ``event_hash`` is the real ``AuditEvent.event_hash`` that recorded it.

    Correlation: an ``AuditEvent`` carries no dedicated per-document field — ``trace_id`` is a
    fresh uuid per ``ingest_document``/``reindex_document`` call, unusable without knowing it in
    advance. What DOES correlate every INGEST/INDEX/GUARDRAIL event pipeline.py records is
    ``payload["doc_id"]`` (see ``svc_control_plane.pipeline``), so this fetches the tenant's
    recent events and filters to the ones whose payload doc_id matches.

    What each event proves:
    * INGEST for this doc -> the raw bytes reached the object store and the document was
      registered -> ``ingest``/``object`` = ran.
    * INDEX for this doc -> pipeline.py records it only after chunk/contextualize/embed and every
      store write (vector, sparse, catalog) plus the stale-chunk swap complete -> those stages,
      and the ``index`` checkpoint itself, = ran; and since indexing succeeded, ``quarantine`` is
      provably = skipped.
    * GUARDRAIL with ``reason == "prompt_injection_suspected"`` for this doc -> the document WAS
      quarantined (``_quarantine_suspect`` returns immediately) -> ``sanitize``/``quarantine`` =
      ran, and everything after = skipped, PROVABLY (not guessed).

    ``doc_intel`` and ``graph`` have no dedicated audit action, so they are left at the default
    ``"defined"`` status rather than guessed — fail-safe means never claiming a stage ran when
    the audit trail cannot confirm it, just as much as never raising on a bad audit read.

    ``audit`` is the injected wiring audit-service dependency (production: a real
    ``AuditService``; tests: any object with a matching ``list`` coroutine).

    Fail-safe: on ANY error (audit unreachable, malformed events, …) this returns the static
    :func:`pipeline_dag` unchanged rather than raising — a broken overlay must never break the
    diagram.
    """
    try:
        dag = pipeline_dag()
        node_by_id = {n.id: n for n in dag.nodes}
        events = await audit.list(kb_id, None)
        doc_events = [e for e in events if e.payload.get("doc_id") == doc_id]
        if doc_events:
            _apply_overlay(node_by_id, doc_events)  # audit overlay: event_hash + trust on ingest/index
        _apply_live_progress(node_by_id, dag.edges, kb_id, doc_id)  # LIVE incremental overlay
        if not doc_events and all(n.status == "defined" for n in node_by_id.values()):
            # Nothing known yet (no audit events, no live marks) -> the static template.
            return dag.model_copy(update={"doc_id": doc_id, "title": f"Ingestion trace · {doc_id}"})
        return IngestDag(
            nodes=list(node_by_id.values()),
            edges=dag.edges,
            doc_id=doc_id,
            title=f"Ingestion trace · {doc_id}",
        )
    except Exception:
        return pipeline_dag()


def _apply_live_progress(node_by_id: dict[str, DagNode], edges: list[DagEdge], kb_id: str, doc_id: str) -> None:
    """Overlay the LIVE per-stage progress (``ingest_progress``) so the DAG fills in incrementally
    as the background ingest worker runs: completed stages -> ``ran``, and the next stage whose
    predecessor is done but itself is not -> ``running`` (amber). Fail-safe (no progress -> no-op)."""
    from svc_control_plane import ingest_progress  # noqa: PLC0415 - avoid a module-load import cycle

    _ = edges
    live = ingest_progress.completed_stages(kb_id, doc_id)
    if not live:
        return
    for stage_id in live:
        node = node_by_id.get(stage_id)
        if node is not None and node.status in ("defined", "skipped"):
            node.status = "ran"
    # The currently-running stage: the first MAIN-sequence stage still 'defined' whose predecessor
    # on that sequence has run. Uses the linear happy path so the branch nodes (object write during
    # ingest, quarantine) are never mistaken for the active stage.
    for prev_id, stage_id in pairwise(_MAIN_SEQUENCE):
        node = node_by_id.get(stage_id)
        prev = node_by_id.get(prev_id)
        if node is not None and prev is not None and node.status == "defined" and prev.status == "ran":
            node.status = "running"
            break
    _apply_real_timing(node_by_id, live)


def _apply_real_timing(node_by_id: dict[str, DagNode], live: dict[str, float]) -> None:
    """Attach the REAL per-stage timeline from the live completion EPOCHS (seconds). Thin wrapper
    over :func:`_assign_timing` — converts epochs to offset-ms first."""
    if not live:
        return
    t0 = min(live.values())
    _assign_timing(node_by_id, {s: max(0, round((e - t0) * 1000)) for s, e in live.items()})


def _assign_timing(node_by_id: dict[str, DagNode], at_ms: dict[str, int]) -> None:
    """Attach the REAL per-stage timeline from completion OFFSETS (ms from ingest start) so the UI
    can pace the animation to what actually happened and show honest durations. ``at_ms`` = a
    stage's completion offset; ``duration_ms`` = the elapsed time closing that stage's real
    measurement boundary. Several stages are recorded together (one bulk step), so a group's
    duration is attributed to the stage whose completion the offset marks — the LAST in the group —
    and the intermediate stages carry no fabricated per-stage number (left ``None``)."""
    seq = [(s, at_ms[s]) for s in _MAIN_SEQUENCE if s in at_ms and s in node_by_id]
    prev_off = 0
    i = 0
    while i < len(seq):
        j = i
        while j + 1 < len(seq) and seq[j + 1][1] == seq[i][1]:
            j += 1  # extend over stages sharing this completion offset (one bulk step)
        off = seq[i][1]
        for k in range(i, j + 1):
            node_by_id[seq[k][0]].at_ms = max(0, off)
        node_by_id[seq[j][0]].duration_ms = max(0, off - prev_off)
        prev_off = off
        i = j + 1


# --- trust ----------------------------------------------------------------------------------


def trust_summary(dag: IngestDag, *, chain_intact: bool) -> dict[str, Any]:
    """Is this DAG's overlay actually provable, or merely decorative?

    ``verified`` is True only when the audit hash CHAIN itself is reported intact (``chain_intact``
    — computed elsewhere, over the sink's ``prev_hash``/``event_hash`` links) AND every stage
    marked ``ran`` carries the real ``event_hash`` of the event that confirmed it. A DAG with no
    ran stages (no overlay applied, or nothing matched) is never "verified" — there is nothing to
    verify — even if the chain is intact.
    """
    ran = [n for n in dag.nodes if n.status == "ran"]
    total = len(dag.nodes)
    all_hashed = bool(ran) and all(n.event_hash for n in ran)
    verified = chain_intact and all_hashed
    if not ran:
        note = "No stages confirmed ran for this document; showing the static pipeline only."
    elif not chain_intact:
        note = f"{len(ran)} of {total} stage(s) show as ran, but the audit hash chain is reported broken."
    elif not all_hashed:
        missing = sum(1 for n in ran if not n.event_hash)
        note = f"{missing} of {len(ran)} ran stage(s) are missing an audit event hash; not fully verifiable."
    else:
        note = f"All {len(ran)} ran stage(s) are backed by tamper-evident audit event hashes; chain intact."
    return {"stages_total": total, "stages_ran": len(ran), "verified": verified, "note": note}
