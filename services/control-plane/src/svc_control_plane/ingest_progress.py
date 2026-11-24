"""Live per-document ingestion progress — powers the INCREMENTAL pipeline DAG.

``pipeline.ingest_document`` records each stage here as it completes, so the DAG endpoint can show a
document moving through the pipeline in real time: the bulk-ingest worker processes uploads in the
background, so a client polling the per-document DAG watches stages turn green one by one instead of
seeing a finished, static overlay.

Process-local + TTL-bounded (a few minutes — long enough to watch one ingest finish). Multi-replica
note: the bulk worker holds a single leader lock, so its live progress lives in one process; a
client that lands on a different replica still gets the audit-confirmed overlay (ingest/index), just
not the fine-grained live steps. Back this with the Redis coordinator for cross-replica live
progress.
"""

from __future__ import annotations

import time

# (kb_id, doc_id) -> {stage_id: completed_at_epoch_seconds}
_PROGRESS: dict[tuple[str, str], dict[str, float]] = {}
_TTL_SECONDS = 600.0


def mark(kb_id: str, doc_id: str, *stages: str) -> None:
    """Record one or more pipeline stages as completed NOW for this document."""
    now = time.time()
    bucket = _PROGRESS.setdefault((kb_id, doc_id), {})
    for stage in stages:
        bucket.setdefault(stage, now)
    _gc(now)


def completed_stages(kb_id: str, doc_id: str) -> dict[str, float]:
    """Which stages have completed for this document (stage_id -> epoch seconds). Empty if none/expired."""
    return dict(_PROGRESS.get((kb_id, doc_id), {}))


def reset(kb_id: str, doc_id: str) -> None:
    """Clear a document's progress (a re-ingest starts fresh)."""
    _PROGRESS.pop((kb_id, doc_id), None)


def _gc(now: float) -> None:
    stale = [key for key, marks in _PROGRESS.items() if marks and now - max(marks.values()) > _TTL_SECONDS]
    for key in stale:
        _PROGRESS.pop(key, None)
