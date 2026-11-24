"""Continuous evaluation (FR-G9) — scheduled quality runs + drift alerting.

Composes existing machinery: each run scores the KB's golden set AND a sample of recent
live QUERY traffic (re-run through the REAL pipeline via :func:`quality.run_eval`), then
records an ``EVAL_RUN`` audit event with the combined metrics. When grounded-rate or
average faithfulness falls below the KB's floor, it emits a distinct ``EVAL_ALERT`` audit
event + a warning log — surfaced in-platform on the Quality page. No external egress.

The scheduler is an in-process asyncio loop (:func:`scheduler_loop`) started from the app
lifespan; it reads the enrolment registry (:mod:`eval_schedule`) each tick and runs the KBs
whose interval is due. Across multiple replicas exactly ONE ticks: :func:`_is_leader` holds
a Redis leader lease via :mod:`rag_common.coordination` (single-process is always leader).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import uuid
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel

from rag_clients.factory import get_catalog
from rag_common import get_coordinator, get_logger, get_settings
from rag_schemas.common import AuditAction
from svc_control_plane import wiring
from svc_control_plane.eval_schedule import EvalSchedule
from svc_control_plane.quality import EvalQuestion, EvalReport, run_eval

_log = get_logger(__name__)

_GOLDEN_KEY = "golden_set"

# Identifies THIS replica when contending for the scheduler leader lease (uuid per process).
_SCHEDULER_TOKEN = uuid.uuid4().hex


class EvalRunResult(BaseModel):
    """The outcome of one continuous-evaluation run (also the EVAL_RUN audit payload)."""

    kb_id: str
    at: datetime
    golden_n: int
    sampled_n: int
    total: int
    grounded: int
    abstained: int
    flagged: int
    grounded_rate: float | None = None
    faithfulness_avg: float | None = None
    hit_rate: float | None = None  # golden-set only (needs expected facts)
    breached: bool = False
    breach_reasons: list[str] = []


# --- pure helpers (unit-tested; no I/O) --------------------------------------
def is_due(sched: EvalSchedule, now: datetime) -> bool:
    """True when a schedule has never run or its interval has elapsed."""
    if sched.last_run_at is None:
        return True
    return now - sched.last_run_at >= timedelta(hours=sched.interval_hours)


def select_sample(query_texts: list[str], rate: float, cap: int) -> list[str]:
    """Pick which recent queries to re-score: dedupe (first-seen order = most recent first),
    take ``ceil(rate * unique)`` of them, capped at ``cap``. Deterministic (recency sample,
    no RNG) so runs are reproducible and tests are stable."""
    seen: list[str] = []
    for q in query_texts:
        t = q.strip()
        if t and t not in seen:
            seen.append(t)
    if rate <= 0 or not seen:
        return []
    take = min(cap, max(1, math.ceil(rate * len(seen))))
    return seen[:take]


def breach_reasons(grounded_rate: float | None, faithfulness_avg: float | None, sched: EvalSchedule) -> list[str]:
    """Which floors (if any) the run's metrics fell below — empty means healthy."""
    reasons: list[str] = []
    if grounded_rate is not None and grounded_rate < sched.min_grounded_rate:
        reasons.append(f"grounded_rate {grounded_rate:.2f} < {sched.min_grounded_rate:.2f}")
    if faithfulness_avg is not None and faithfulness_avg < sched.min_faithfulness:
        reasons.append(f"faithfulness {faithfulness_avg:.2f} < {sched.min_faithfulness:.2f}")
    return reasons


# --- golden-set + live-traffic loading ---------------------------------------
async def _load_golden(kb_id: str) -> list[EvalQuestion]:
    raw = (await get_catalog().get_tenant_config(kb_id)).get(_GOLDEN_KEY)
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return [EvalQuestion(**q) for q in data.get("questions", [])]
    except (ValueError, TypeError):
        _log.warning("eval_golden_parse_failed", kb_id=kb_id)
        return []


async def _sample_live_queries(kb_id: str) -> list[EvalQuestion]:
    settings = get_settings()
    events = await wiring.audit_service().list(kb_id, limit=settings.continuous_eval_lookback_events)
    texts = [
        str(e.payload.get("query", "")).strip()
        for e in events
        if e.action == AuditAction.QUERY and not e.payload.get("cache_hit") and e.payload.get("query")
    ]
    picked = select_sample(texts, settings.continuous_eval_sample_rate, settings.continuous_eval_sample_max)
    return [EvalQuestion(question=t) for t in picked]


_Combined = tuple[int, int, int, int, int, "float | None", "float | None", "float | None"]


def _combine(golden: EvalReport | None, live: EvalReport | None) -> _Combined:
    reports = [r for r in (golden, live) if r is not None]
    total = sum(r.total for r in reports)
    grounded = sum(r.grounded for r in reports)
    abstained = sum(r.abstained for r in reports)
    flagged = sum(r.flagged for r in reports)
    scores = [row.faithfulness for r in reports for row in r.rows if row.faithfulness is not None]
    grounded_rate = round(grounded / total, 4) if total else None
    faithfulness_avg = round(sum(scores) / len(scores), 4) if scores else None
    hit_rate = golden.hit_rate if golden else None  # only the golden set carries expected facts
    return total, grounded, abstained, flagged, len(scores), grounded_rate, faithfulness_avg, hit_rate


# --- the run -----------------------------------------------------------------
async def run_once(kb_id: str, sched: EvalSchedule, *, actor: str = "scheduler") -> EvalRunResult:
    """Score golden set + sampled live traffic, record EVAL_RUN (+ EVAL_ALERT on breach)."""
    golden_qs = await _load_golden(kb_id)
    live_qs = await _sample_live_queries(kb_id)

    golden = await run_eval(kb_id, golden_qs, acl=[], actor=f"g9:{actor}") if golden_qs else None
    live = await run_eval(kb_id, live_qs, acl=[], actor=f"g9:{actor}") if live_qs else None

    total, grounded, abstained, flagged, _scored, grounded_rate, faithfulness_avg, hit_rate = _combine(golden, live)
    reasons = breach_reasons(grounded_rate, faithfulness_avg, sched)
    result = EvalRunResult(
        kb_id=kb_id,
        at=datetime.now(UTC),
        golden_n=len(golden_qs),
        sampled_n=len(live_qs),
        total=total,
        grounded=grounded,
        abstained=abstained,
        flagged=flagged,
        grounded_rate=grounded_rate,
        faithfulness_avg=faithfulness_avg,
        hit_rate=hit_rate,
        breached=bool(reasons),
        breach_reasons=reasons,
    )

    audit = wiring.audit_service()
    trace_id = str(uuid.uuid4())
    payload = result.model_dump(mode="json")
    await audit.record(tenant_id=kb_id, trace_id=trace_id, action=AuditAction.EVAL_RUN, actor=actor, payload=payload)
    if reasons:
        _log.warning("eval_breach", kb_id=kb_id, reasons=reasons, grounded_rate=grounded_rate)
        await audit.record(
            tenant_id=kb_id,
            trace_id=trace_id,
            action=AuditAction.EVAL_ALERT,
            actor=actor,
            payload={"reasons": reasons, "grounded_rate": grounded_rate, "faithfulness_avg": faithfulness_avg},
        )
    # Mesh edge (FR-G9): when this KB federates, ALSO attribute joint-answer quality per peer
    # and alert on any peer that DEGRADES it. Same golden; ablation-based. No-op without peers.
    if sched.federate_from and golden_qs:
        from svc_control_plane import federated_eval  # noqa: PLC0415 — avoid an import cycle

        caller_dept = kb_id.split("__", 1)[0]
        try:
            await federated_eval.run_federated_eval(
                caller_dept, [kb_id], sched.federate_from, golden_qs, actor=f"g9:{actor}"
            )
        except Exception as exc:  # mesh eval must never fail the own-KB run
            _log.warning("federated_eval_failed", kb_id=kb_id, error=str(exc)[:200])
    _log.info("eval_run", kb_id=kb_id, total=total, grounded_rate=grounded_rate, breached=bool(reasons))
    return result


async def recent_runs(kb_id: str, limit: int = 20) -> list[dict[str, object]]:
    """Recent EVAL_RUN / EVAL_ALERT events for the console (newest first)."""
    events = await wiring.audit_service().list(kb_id, limit=200)
    out: list[dict[str, object]] = []
    for e in events:
        if e.action in (AuditAction.EVAL_RUN, AuditAction.EVAL_ALERT):
            out.append({"action": e.action.value, "at": e.at.isoformat(), **e.payload})
        if len(out) >= limit:
            break
    return out


# --- scheduler (in-process asyncio loop, started from the app lifespan) -------
async def scheduler_tick(now: datetime | None = None) -> int:
    """Run every enrolled KB whose interval is due. Returns how many ran. Each KB is
    isolated: one KB's failure never aborts the tick."""
    now = now or datetime.now(UTC)
    store = wiring.eval_schedule_store()
    ran = 0
    for sched in await store.list_enabled():
        if not is_due(sched, now):
            continue
        try:
            await run_once(sched.kb_id, sched, actor="scheduler")
            await store.mark_run(sched.kb_id, now)
            ran += 1
        except Exception as exc:  # never let one KB kill the loop
            _log.warning("eval_tick_failed", kb_id=sched.kb_id, error=str(exc)[:200])
    return ran


def _is_leader() -> bool:
    """True when this replica may run scheduled jobs. Single-process (no RAG_REDIS_URL) is
    always the leader; when scaled, a Redis lease ensures EXACTLY ONE replica runs each tick
    so eval/sync jobs aren't duplicated N times."""
    settings = get_settings()
    return get_coordinator().acquire_leader("scheduler", settings.leader_lock_ttl_seconds, _SCHEDULER_TOKEN)


async def scheduler_loop(stop: asyncio.Event) -> None:
    """The platform's in-process scheduler: wakes every ``continuous_eval_tick_minutes`` and
    runs whichever due jobs are enabled — continuous evaluation (FR-G9) and/or delta connector
    sync (FR-A8) — until stopped. When multiple replicas run, only the Redis leader ticks."""
    from svc_control_plane import sync_jobs  # noqa: PLC0415 — local import avoids an import cycle

    settings = get_settings()
    interval = settings.continuous_eval_tick_minutes * 60
    _log.info(
        "scheduler_started",
        tick_minutes=settings.continuous_eval_tick_minutes,
        eval=settings.continuous_eval_enabled,
        sync=settings.sync_scheduler_enabled,
    )
    while not stop.is_set():
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=interval)
        if stop.is_set():
            break
        if not _is_leader():  # a non-leader replica does nothing but stay ready to take over
            continue
        if get_settings().continuous_eval_enabled:
            with contextlib.suppress(Exception):
                ran = await scheduler_tick()
                if ran:
                    _log.info("eval_scheduler_tick", ran=ran)
        if get_settings().sync_scheduler_enabled:
            with contextlib.suppress(Exception):
                synced = await sync_jobs.sync_tick()
                if synced:
                    _log.info("sync_scheduler_tick", ran=synced)
    _log.info("scheduler_stopped")
