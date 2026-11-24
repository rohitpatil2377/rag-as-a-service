"""Corpus-wide and automatic re-indexing, driven by the pipeline fingerprint.

A document is indexed by whatever pipeline was running when it was ingested. Change the chunker,
the embedder, contextual retrieval or the readers, and every existing document is now STALE: it
still answers, but from chunks and vectors a different pipeline produced, and nothing says so.
Today that is invisible -- the platform reports a successful ingest, a healthy KB and a normal
answer, while the index underneath was built by code this deployment no longer runs.

Two mechanisms, both keyed on ``index_pipeline_version`` (stamped at index time):

* **On demand** -- :func:`reindex_kb` rebuilds a knowledge base's documents, or only the stale
  ones. Sequential, one document at a time: a re-index is one extraction + one embedding pass
  per document against shared model endpoints, and firing a whole corpus at them in parallel is
  how a re-index makes every user's live question time out.
* **Automatic** -- :func:`worker_loop` re-indexes stale documents on an interval, for the
  knowledge bases an operator EXPLICITLY listed (``RAG_AUTO_REINDEX_KBS``). Opt-in by name
  rather than "everything", because this is a write over tenant data on a schedule.

Reporting is always safe and always on; ACTING on the report is opt-in. A document that has no
stamp at all (ingested before this existed) reads as stale, which is the honest answer: nothing
recorded how it was built, so nothing can claim it matches the running pipeline.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field

from svc_indexing.service import META_PIPELINE_VERSION, pipeline_fingerprint

from rag_clients.factory import get_catalog
from rag_common import get_logger, get_settings
from rag_common.coordination import get_coordinator

_log = get_logger(__name__)

AUTO_REINDEX_ACTOR = "system:auto-reindex"

# --- Leader-gated auto-reindex (multi-worker safety) ---------------------------------------
# The auto-reindexer is a WRITE loop that runs in every uvicorn worker. With >1 worker and no
# gate, each worker reindexes the same KB at the same time: concurrent graph-on writes deadlock
# in Postgres and both event loops go CPU-bound, starving live traffic. The coordinator lease
# makes exactly ONE worker run a pass. InMemoryCoordinator (single process) always grants the
# lease, so a single-worker deployment behaves exactly as before.
_REINDEX_LEASE = "auto_reindex"
_REINDEX_TOKEN = secrets.token_hex(8)  # identifies THIS worker process for the lease


@dataclass(slots=True)
class ReindexStatus:
    """How much of a knowledge base was indexed by the pipeline that is running now."""

    kb_id: str
    current_version: str
    total: int
    stale: int
    stale_doc_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReindexKbResult:
    kb_id: str
    requested: int
    reindexed: int
    failed: list[dict[str, str]] = field(default_factory=list)
    skipped_current: int = 0


async def stale_report(kb_id: str) -> ReindexStatus:
    """Which documents carry a pipeline stamp other than the running one (or none at all)."""
    version = pipeline_fingerprint(get_settings())
    docs = await get_catalog().list_documents(kb_id)
    stale = [d.doc_id for d in docs if d.metadata.get(META_PIPELINE_VERSION) != version]
    return ReindexStatus(kb_id=kb_id, current_version=version, total=len(docs), stale=len(stale), stale_doc_ids=stale)


async def reindex_kb(
    kb_id: str,
    *,
    actor: str,
    stale_only: bool = True,
    limit: int = 50,
    heartbeat: Callable[[], bool] | None = None,
) -> ReindexKbResult:
    """Rebuild a knowledge base's documents ONE AT A TIME.

    Per-document failures are reported, never fatal: the underlying re-index is write-then-swap,
    so a document that fails keeps serving its previous index rather than being left with none.

    ``heartbeat`` (optional): checked before each document; if it returns ``False`` the pass
    stops at the current document boundary. The auto-reindexer passes a leadership check here so
    a worker that loses the lease yields promptly instead of racing another worker. Direct
    callers (the on-demand reindex endpoint) pass nothing and rebuild the whole batch as before.
    """
    from svc_control_plane import pipeline  # noqa: PLC0415 -- avoid an import cycle at module load

    report = await stale_report(kb_id)
    targets = report.stale_doc_ids if stale_only else [d.doc_id for d in await get_catalog().list_documents(kb_id)]
    targets = targets[: max(0, limit)]
    result = ReindexKbResult(
        kb_id=kb_id, requested=len(targets), reindexed=0, skipped_current=report.total - report.stale
    )
    for doc_id in targets:
        if heartbeat is not None and not heartbeat():
            _log.info("reindex_kb_yielded", kb_id=kb_id, reindexed=result.reindexed, requested=result.requested)
            break
        try:
            await pipeline.reindex_document(kb_id, doc_id, actor=actor)
            result.reindexed += 1
        except Exception as exc:  # one bad document must not stop the corpus
            result.failed.append({"doc_id": doc_id, "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
            _log.warning("reindex_kb_document_failed", kb_id=kb_id, doc_id=doc_id, error=str(exc)[:160])
    _log.info(
        "reindex_kb_done",
        kb_id=kb_id,
        requested=result.requested,
        reindexed=result.reindexed,
        failed=len(result.failed),
    )
    return result


def auto_reindex_kbs() -> list[str]:
    """The knowledge bases the operator opted in to automatic re-indexing, by name."""
    return [k.strip() for k in get_settings().auto_reindex_kbs.split(",") if k.strip()]


def _reindex_leader() -> bool:
    """Acquire or RENEW this process's auto-reindex leadership.

    Calling it repeatedly extends the lease (the coordinator's compare-and-extend), so a long
    pass keeps leadership rather than letting the TTL lapse mid-document and letting a second
    worker start a competing reindex. Single process (in-memory coordinator) ⇒ always ``True``.
    """
    return get_coordinator().acquire_leader(_REINDEX_LEASE, get_settings().leader_lock_ttl_seconds, _REINDEX_TOKEN)


async def _lease_keeper(lost: asyncio.Event) -> None:
    """Renew the lease on a timer (TTL/3) while we hold it; flag ``lost`` the moment we don't.

    A graph-on document can take minutes while the lease TTL is ~90s, so renewing only between
    documents would let it expire mid-document. This renews independently of per-document
    duration; if leadership slips (another worker, or a Redis blip that fails closed) it sets
    ``lost`` so the in-flight pass yields at the next document boundary.
    """
    interval = max(5, get_settings().leader_lock_ttl_seconds // 3)
    while not lost.is_set():
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return
        if not _reindex_leader():
            lost.set()
            return


async def _run_leader_pass(kbs: list[str], stop: asyncio.Event) -> None:
    """Reindex each listed KB's stale docs while a background keeper renews our lease.

    Yields at the next document boundary if we lose leadership or are asked to stop.
    """
    settings = get_settings()
    lost = asyncio.Event()
    keeper = asyncio.create_task(_lease_keeper(lost))
    try:
        for kb_id in kbs:
            if stop.is_set() or lost.is_set():
                return
            try:
                report = await stale_report(kb_id)
                if report.stale:
                    _log.info("auto_reindex_start", kb_id=kb_id, stale=report.stale, version=report.current_version)
                    await reindex_kb(
                        kb_id,
                        actor=AUTO_REINDEX_ACTOR,
                        stale_only=True,
                        limit=settings.reindex_batch_limit,
                        heartbeat=lambda: not (lost.is_set() or stop.is_set()),
                    )
            except Exception as exc:  # a background loop never takes the API down
                _log.warning("auto_reindex_failed", kb_id=kb_id, error=f"{type(exc).__name__}: {str(exc)[:160]}")
    finally:
        keeper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await keeper


async def _tick(stop: asyncio.Event) -> None:
    """One scheduler tick: if we hold the lease, reindex the listed KBs; otherwise skip."""
    kbs = auto_reindex_kbs()
    if kbs and _reindex_leader():
        await _run_leader_pass(kbs, stop)


async def worker_loop(stop: asyncio.Event) -> None:
    """Keep the listed knowledge bases current: every interval, re-index whatever is stale.

    Leader-gated — with more than one worker, only the replica holding the ``auto_reindex`` lease
    runs a pass; the rest skip. That is what lets the backfill run while the API serves: one
    worker reindexes, the others stay free for live traffic, instead of every worker racing the
    same KB (concurrent graph-on reindexes deadlock in Postgres).
    """
    every = max(1, get_settings().auto_reindex_interval_minutes) * 60
    while not stop.is_set():
        await _tick(stop)
        with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=every)
