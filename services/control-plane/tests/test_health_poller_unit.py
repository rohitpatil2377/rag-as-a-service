"""Unit tests for the continuous model-health poller (FR-H5 self-check).

Covers the four behaviours the feature turns on: debounce (a single blip does NOT flip a dot,
two consecutive failures do, recovery resets), leader gating (a non-leader replica never probes
on a timer), snapshot serialize/deserialize, and the endpoint serving the published snapshot
WITHOUT probing while still force-probing on ``refresh=true``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from svc_control_plane import api as api_module
from svc_control_plane import health_poller
from svc_control_plane.health_poller import (
    BackendSnapshot,
    HealthSnapshot,
    apply_debounce,
)
from svc_control_plane.preflight import BackendHealth

from rag_common.auth import Principal


def _probe(*, online: bool, name: str = "Generation LLM", role: str = "Generation") -> BackendHealth:
    return BackendHealth(
        name=name,
        role=role,
        endpoint="models.example",
        online=online,
        latency_ms=42 if online else None,
        detail="" if online else "HTTP 503",
        model="gemma4-31b",
    )


def _p(roles: set[str]) -> Principal:
    return Principal(subject="u@x", tenant_id="acme", roles=frozenset(roles), attrs={})


# --- debounce ----------------------------------------------------------------
def test_one_failure_does_not_flip_the_dot() -> None:
    """A single transient failure must NOT flip an online backend offline."""
    now = datetime.now(UTC)
    online = BackendSnapshot(name="Generation LLM", role="Generation", endpoint="e", online=True, last_online=now)
    state, transition = apply_debounce(online, _probe(online=False), threshold=2, now=now)
    assert state.online is True, "one failure under the threshold keeps the backend online"
    assert state.consecutive_failures == 1
    assert transition is None
    assert state.last_online == now, "last_online is carried through the grace window"


def test_two_consecutive_failures_flip_offline() -> None:
    now = datetime.now(UTC)
    after_one = BackendSnapshot(
        name="Generation LLM", role="Generation", endpoint="e", online=True, consecutive_failures=1, last_online=now
    )
    state, transition = apply_debounce(after_one, _probe(online=False), threshold=2, now=now)
    assert state.online is False, "the second consecutive failure flips it offline"
    assert state.consecutive_failures == 2
    assert transition == "down", "the online->offline flip is a reported transition"


def test_first_ever_probe_failing_reports_offline_immediately() -> None:
    """Right after deploy there is no prior snapshot; a backend that fails its very first probe must
    report offline at once (no established online status to protect), not green until a 2nd failure."""
    now = datetime.now(UTC)
    state, transition = apply_debounce(None, _probe(online=False), threshold=2, now=now)
    assert state.online is False, "a first-sight failure is offline immediately"
    assert state.consecutive_failures == 1
    assert transition is None, "never observed online, so this is not a down-transition"


def test_recovery_resets_failures_and_marks_up() -> None:
    now = datetime.now(UTC)
    offline = BackendSnapshot(
        name="Generation LLM", role="Generation", endpoint="e", online=False, consecutive_failures=3, last_online=None
    )
    state, transition = apply_debounce(offline, _probe(online=True), threshold=2, now=now)
    assert state.online is True
    assert state.consecutive_failures == 0, "a successful probe resets the failure run"
    assert state.last_online == now
    assert transition == "up", "the offline->online recovery is a reported transition"


def test_healthy_probe_is_not_a_transition() -> None:
    now = datetime.now(UTC)
    online = BackendSnapshot(name="Generation LLM", role="Generation", endpoint="e", online=True, last_online=now)
    _state, transition = apply_debounce(online, _probe(online=True), threshold=2, now=now)
    assert transition is None


# --- snapshot serialize / deserialize ----------------------------------------
def test_snapshot_roundtrips_through_json() -> None:
    now = datetime.now(UTC)
    snap = HealthSnapshot(
        polled_at=now,
        backends=[
            BackendSnapshot(
                name="Generation LLM",
                role="Generation",
                endpoint="models.example",
                online=True,
                latency_ms=42,
                detail="",
                model="gemma4-31b",
                consecutive_failures=0,
                last_online=now,
            ),
            BackendSnapshot(
                name="Reranker",
                role="Reranking",
                endpoint="rr.example",
                online=False,
                latency_ms=None,
                detail="HTTP 401",
                model="bge-reranker",
                consecutive_failures=2,
                last_online=None,
            ),
        ],
    )
    restored = HealthSnapshot.model_validate_json(snap.model_dump_json())
    assert restored == snap, "a snapshot survives the Redis JSON round trip byte-for-byte"
    assert restored.backends[1].last_online is None
    assert restored.backends[1].consecutive_failures == 2


def test_is_stale_tracks_age(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_common.config import Settings

    settings = Settings(health_poll_seconds=60)
    fresh = HealthSnapshot(polled_at=datetime.now(UTC), backends=[])
    old = HealthSnapshot(polled_at=datetime.now(UTC) - timedelta(seconds=60 * 10), backends=[])
    assert health_poller.is_stale(fresh, settings) is False
    assert health_poller.is_stale(old, settings) is True


# --- leader gating -----------------------------------------------------------
class _Coordinator:
    def __init__(self, *, leader: bool) -> None:
        self._leader = leader

    def acquire_leader(self, name: str, ttl_s: int, token: str) -> bool:
        return self._leader


async def test_non_leader_never_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A replica that does not hold the lease must not probe the model endpoints on a timer."""
    calls = {"n": 0}

    async def _spy(_settings: object) -> list[BackendHealth]:
        calls["n"] += 1
        return [_probe(online=True)]

    monkeypatch.setattr(health_poller, "probe_backend_health", _spy)
    monkeypatch.setattr(health_poller, "get_coordinator", lambda: _Coordinator(leader=False))

    probed = await health_poller._leader_tick()
    assert probed is False
    assert calls["n"] == 0, "the non-leader replica did not probe"


async def test_leader_probes_and_publishes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    async def _spy(_settings: object) -> list[BackendHealth]:
        calls["n"] += 1
        return [_probe(online=True)]

    monkeypatch.setattr(health_poller, "probe_backend_health", _spy)
    monkeypatch.setattr(health_poller, "get_coordinator", lambda: _Coordinator(leader=True))
    monkeypatch.setattr(health_poller, "get_redis_client", lambda: None)
    health_poller._poller_snapshot.clear()

    probed = await health_poller._leader_tick()
    assert probed is True
    assert calls["n"] == 1
    snap = await health_poller.read_snapshot()
    assert snap is not None and snap.backends[0].online is True


async def test_in_process_snapshot_expires_past_stale_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wedged loop must not serve a frozen in-process snapshot forever: past the stale window the
    read returns None so the endpoint self-heals to an on-demand probe (matching Redis key expiry)."""
    from rag_common.config import Settings

    monkeypatch.setattr(health_poller, "get_redis_client", lambda: None)
    monkeypatch.setattr(health_poller, "get_settings", lambda: Settings(health_poll_seconds=60))

    fresh = HealthSnapshot(polled_at=datetime.now(UTC), backends=[])
    health_poller._poller_snapshot.clear()
    health_poller._poller_snapshot["platform"] = fresh
    assert await health_poller.read_snapshot() is not None, "a fresh in-process snapshot is served"

    # 60s interval, _STALE_INTERVALS=3 -> a snapshot older than 180s is stale.
    health_poller._poller_snapshot["platform"] = HealthSnapshot(
        polled_at=datetime.now(UTC) - timedelta(seconds=60 * 10), backends=[]
    )
    assert await health_poller.read_snapshot() is None, "a stale in-process snapshot reads as None"
    health_poller._poller_snapshot.clear()


# --- endpoint serves the snapshot without probing ----------------------------
async def test_endpoint_serves_snapshot_without_probing(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """When a poller snapshot is published, the endpoint returns it WITHOUT probing (source=poller);
    refresh=true still forces a live probe (source=live)."""
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_HEALTH_POLL_ENABLED", "true")
    get_settings.cache_clear()

    live_calls = {"n": 0}

    async def _live_probe(_settings: object) -> list[BackendHealth]:
        live_calls["n"] += 1
        return [_probe(online=True, name="Generation LLM"), _probe(online=True, name="Embedder", role="Embedding")]

    monkeypatch.setattr(api_module, "probe_backend_health", _live_probe)
    monkeypatch.setattr(api_module, "_health_cache", {})
    # No Redis in the in-memory profile, so the endpoint reads the in-process snapshot.
    monkeypatch.setattr(health_poller, "get_redis_client", lambda: None)
    now = datetime.now(UTC)
    health_poller._poller_snapshot.clear()
    health_poller._poller_snapshot["platform"] = HealthSnapshot(
        polled_at=now,
        backends=[
            BackendSnapshot(name="Generation LLM", role="Generation", endpoint="e", online=True, last_online=now),
            BackendSnapshot(name="Embedder", role="Embedding", endpoint="e2", online=False, consecutive_failures=2),
        ],
    )

    served = await api_module.system_model_health(_p({"admin"}), refresh=False, kb_id=None)
    assert served.source == "poller"
    assert served.cached is True
    assert live_calls["n"] == 0, "serving the snapshot must not probe on the request path"
    assert {b.name for b in served.backends} == {"Generation LLM", "Embedder"}
    embedder = next(b for b in served.backends if b.name == "Embedder")
    assert embedder.online is False and embedder.consecutive_failures == 2

    forced = await api_module.system_model_health(_p({"admin"}), refresh=True, kb_id=None)
    assert forced.source == "live"
    assert live_calls["n"] == 1, "refresh=true forces a live probe"

    get_settings.cache_clear()


async def test_flag_off_never_serves_snapshot(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the poller disabled the endpoint probes on-demand exactly as before — a stray published
    snapshot must be ignored."""
    from rag_common.config import get_settings

    get_settings.cache_clear()  # health_poll_enabled defaults False

    live_calls = {"n": 0}

    async def _live_probe(_settings: object) -> list[BackendHealth]:
        live_calls["n"] += 1
        return [_probe(online=True)]

    monkeypatch.setattr(api_module, "probe_backend_health", _live_probe)
    monkeypatch.setattr(api_module, "_health_cache", {})
    health_poller._poller_snapshot["platform"] = HealthSnapshot(
        polled_at=datetime.now(UTC),
        backends=[BackendSnapshot(name="Generation LLM", role="Generation", endpoint="e", online=True)],
    )

    report = await api_module.system_model_health(_p({"admin"}), refresh=False, kb_id=None)
    assert report.source == "live", "flag off = on-demand probe, never the snapshot"
    assert live_calls["n"] == 1
    health_poller._poller_snapshot.clear()
