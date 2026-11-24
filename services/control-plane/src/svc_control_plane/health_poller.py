"""Continuous 24/7 model-health self-check — the "no more clicking" poller.

Until now ``/system/model-health`` probed on the request path: the console had to click
"Re-check" to know whether a model backend was answering, and each click fanned real requests
out to real model endpoints. This module runs an in-process ``asyncio`` loop (mirroring
:mod:`continuous_eval`'s scheduler) that probes the PLATFORM backends every ``health_poll_seconds``
with the SAME :func:`preflight.health` prober the endpoint uses, and publishes the result so the
endpoint can serve it INSTANTLY without probing.

Three properties make it ops-grade:

* **Leader election** — only the Redis leader replica probes on a timer
  (:func:`_leader_tick`); the others stay ready to take over. Single-process is always leader.
* **Shared snapshot** — the leader publishes the snapshot to a Redis key all replicas read, so
  all API replicas show identical fresh status. No Redis ⇒ an in-process fallback (one process).
* **Debounce** — a backend flips online→offline only after ``health_poll_fail_threshold``
  CONSECUTIVE failed probes, so a single transient blip never flips the dot.

Alerting is observable, never silent: a state TRANSITION emits a structured WARN (down) / INFO
(recovered) log AND updates an OpenTelemetry gauge + counter.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from datetime import UTC, datetime

from opentelemetry import metrics as _otel_metrics
from pydantic import BaseModel, Field

from rag_common import get_coordinator, get_logger, get_redis_client, get_settings
from rag_common.config import Settings
from svc_control_plane.preflight import BackendHealth
from svc_control_plane.preflight import health as probe_backend_health

_log = get_logger(__name__)

# Identifies THIS replica when contending for the poll leader lease (uuid per process).
_POLLER_TOKEN = uuid.uuid4().hex
_LEASE_NAME = "model-health-poll"

# The shared snapshot lives under one key (platform slot); several intervals of TTL so a brief
# leader gap doesn't drop the snapshot the endpoint serves from.
_SNAPSHOT_KEY = "rag:health:snapshot:platform"
_SLOT = "platform"
# A snapshot older than this many intervals is reported ``stale`` — old enough that the leader
# has plainly missed ticks (crash / lost lease) and the numbers should not be trusted as live.
_STALE_INTERVALS = 3

# In-process fallback for the single-process (no Redis) deployment. Keyed by slot to mirror the
# endpoint's own cache shape.
_poller_snapshot: dict[str, HealthSnapshot] = {}

# OTel metrics. ``get_meter`` returns a no-op meter when no MeterProvider is configured (the
# unit/dev profile), so these are always safe to call; when the OTLP exporter is wired they carry
# real signal. The gauge is the live 1/0 state; the counter marks each transition.
_meter = _otel_metrics.get_meter("svc_control_plane.health_poller")
_ONLINE_GAUGE = _meter.create_gauge(
    "model_backend_online", description="1 when a model backend's last debounced probe was online, else 0."
)
_TRANSITIONS = _meter.create_counter(
    "model_backend_transitions_total", description="Model-backend online/offline state transitions."
)


class BackendSnapshot(BaseModel):
    """One backend's published state — the debounced health plus what the debounce tracks."""

    name: str
    role: str
    endpoint: str
    online: bool = Field(..., description="DEBOUNCED state: stays online through the failure grace window.")
    latency_ms: int | None = None
    detail: str = ""
    model: str = ""
    consecutive_failures: int = Field(0, description="Consecutive failed probes; the dot flips past the threshold.")
    last_online: datetime | None = Field(None, description="When this backend was last observed online. None if never.")


class HealthSnapshot(BaseModel):
    """The whole platform roster as of one poll, published for every replica to serve."""

    polled_at: datetime
    backends: list[BackendSnapshot] = Field(default_factory=list)


# --- pure helper (unit-tested; no I/O) ---------------------------------------
def apply_debounce(
    prev: BackendSnapshot | None, probe: BackendHealth, threshold: int, now: datetime
) -> tuple[BackendSnapshot, str | None]:
    """Fold one raw probe into the debounced per-backend state.

    Returns the new state and a transition marker: ``"down"`` when it flipped online→offline
    (>= ``threshold`` consecutive failures), ``"up"`` when it recovered, else ``None``. A single
    transient failure does NOT flip an online backend — it only increments the failure count.
    """
    if probe.online:
        state = BackendSnapshot(
            name=probe.name,
            role=probe.role,
            endpoint=probe.endpoint,
            online=True,
            latency_ms=probe.latency_ms,
            detail=probe.detail,
            model=probe.model,
            consecutive_failures=0,
            last_online=now,
        )
        transition = "up" if (prev is not None and not prev.online) else None
        return state, transition

    # A failing probe. Debounce guards an ESTABLISHED online state: it keeps a backend online
    # through the grace window so a single blip doesn't flip the dot. With NO prior state (first
    # probe ever — right after deploy) there is no online status to protect, so a failure reports
    # offline immediately (honesty over grace); it is not a "down" transition because it was never
    # observed online to fall from.
    if prev is None:
        state = BackendSnapshot(
            name=probe.name,
            role=probe.role,
            endpoint=probe.endpoint,
            online=False,
            latency_ms=None,
            detail=probe.detail,
            model=probe.model,
            consecutive_failures=1,
            last_online=None,
        )
        return state, None

    failures = prev.consecutive_failures + 1
    online = prev.online and failures < threshold  # flip offline only once the run reaches threshold
    state = BackendSnapshot(
        name=probe.name,
        role=probe.role,
        endpoint=probe.endpoint,
        online=online,
        latency_ms=None,  # a failed probe's elapsed time is the timeout, not a latency
        detail=probe.detail,
        model=probe.model,
        consecutive_failures=failures,
        last_online=prev.last_online,
    )
    transition = "down" if (prev.online and not online) else None
    return state, transition


def is_stale(snap: HealthSnapshot, settings: Settings) -> bool:
    """True when the snapshot is older than several poll intervals — the leader has missed ticks."""
    age = (datetime.now(UTC) - snap.polled_at).total_seconds()
    return age > settings.health_poll_seconds * _STALE_INTERVALS


# --- shared snapshot store (Redis when scaled, in-process otherwise) ----------
# Both callers (the async endpoint and the async poll loop) await these, so the blocking Redis
# get/set run in a worker thread (asyncio.to_thread): a Redis stall (2s socket timeout) must not
# block the event loop that is serving every other request. The in-process branch is pure memory.
async def _publish(snap: HealthSnapshot) -> None:
    r = get_redis_client()
    if r is not None:
        ttl = max(get_settings().health_poll_seconds * (_STALE_INTERVALS + 1), 10)
        try:
            await asyncio.to_thread(r.set, _SNAPSHOT_KEY, snap.model_dump_json(), ex=ttl)
        except Exception as exc:  # a publish blip must never crash the loop
            _log.warning("health_snapshot_publish_failed", error=str(exc)[:200])
        return
    _poller_snapshot[_SLOT] = snap


async def read_snapshot() -> HealthSnapshot | None:
    """The latest published snapshot: Redis (shared across replicas) first, else in-process.

    None when the poller has never published, OR when the reading is older than the stale window —
    so a wedged loop self-heals to on-demand probing instead of serving a frozen snapshot forever
    (the Redis path gets this free from key expiry; the in-process path enforces it here)."""
    r = get_redis_client()
    if r is not None:
        try:
            raw = await asyncio.to_thread(r.get, _SNAPSHOT_KEY)
        except Exception as exc:
            _log.warning("health_snapshot_read_failed", error=str(exc)[:200])
            return None
        if not raw:
            return None
        with contextlib.suppress(Exception):
            return HealthSnapshot.model_validate_json(raw)
        return None
    snap = _poller_snapshot.get(_SLOT)
    if snap is not None and is_stale(snap, get_settings()):
        return None  # the loop has plainly stopped ticking — fall back to an on-demand probe
    return snap


# --- one poll -----------------------------------------------------------------
def _emit_transition(state: BackendSnapshot, kind: str) -> None:
    """Make a state change observable: a structured log + an OTel counter increment."""
    if kind == "down":
        _log.warning(
            "model_backend_down",
            name=state.name,
            role=state.role,
            endpoint=state.endpoint,
            detail=state.detail,
            consecutive_failures=state.consecutive_failures,
        )
    else:
        _log.info("model_backend_recovered", name=state.name, role=state.role, endpoint=state.endpoint)
    _TRANSITIONS.add(1, {"name": state.name, "role": state.role, "transition": kind})


async def poll_once() -> HealthSnapshot:
    """Probe every platform backend, fold each into its debounced state, publish, and alert on any
    transition. The prober is the SAME one the endpoint and the boot preflight use."""
    settings = get_settings()
    probed = await probe_backend_health(settings)
    prev = await read_snapshot()
    prev_by_name = {b.name: b for b in prev.backends} if prev is not None else {}
    now = datetime.now(UTC)
    states: list[BackendSnapshot] = []
    threshold = settings.health_poll_fail_threshold
    for probe in probed:
        state, transition = apply_debounce(prev_by_name.get(probe.name), probe, threshold, now)
        states.append(state)
        _ONLINE_GAUGE.set(1 if state.online else 0, {"name": state.name, "role": state.role})
        if transition is not None:
            _emit_transition(state, transition)
    snap = HealthSnapshot(polled_at=now, backends=states)
    await _publish(snap)
    return snap


async def _leader_tick() -> bool:
    """One beat: probe + publish ONLY if this replica holds the lease. Returns whether it probed.

    The lease TTL is ~2 intervals so a crashed leader's lease expires and another replica takes
    over within a tick. Extracted from the loop so the leader gate is unit-testable."""
    settings = get_settings()
    ttl = max(2 * settings.health_poll_seconds, 2)
    if not get_coordinator().acquire_leader(_LEASE_NAME, ttl, _POLLER_TOKEN):
        return False  # a non-leader replica never probes on a timer
    await poll_once()
    return True


async def poller_loop(stop: asyncio.Event) -> None:
    """The continuous model-health self-check: wake every ``health_poll_seconds``, and on the
    leader replica probe the platform backends and publish the snapshot, until stopped."""
    from rag_common.coordination import InMemoryCoordinator  # noqa: PLC0415 — only to type-check the coordinator

    interval = get_settings().health_poll_seconds
    _log.info(
        "health_poller_started",
        interval_s=interval,
        fail_threshold=get_settings().health_poll_fail_threshold,
    )
    # No Redis lease means acquire_leader is always True here, so if this flag is ever set on more
    # than one process (docker compose --scale api=N, or the mesh x-api-env) EVERY process would
    # probe the paid/remote model endpoints on its own timer. Warn so that is diagnosable rather
    # than silently multiplying probe traffic (mirrors the bulk-ingest single-process mitigation).
    if isinstance(get_coordinator(), InMemoryCoordinator):
        _log.warning(
            "health_poller_single_process",
            detail="health poller running single-process only (no Redis lease); do not enable on multiple replicas",
        )
    while not stop.is_set():
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=interval)
        if stop.is_set():
            break
        try:
            await _leader_tick()
        except Exception as exc:  # one bad tick must never kill the loop — but it must not be silent
            _log.warning("health_poll_tick_failed", error=str(exc)[:200])
    _log.info("health_poller_stopped")
