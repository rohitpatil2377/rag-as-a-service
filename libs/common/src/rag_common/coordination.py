"""Distributed coordination seam (scale/HA hardening).

The modular monolith runs as one process by default; three pieces of shared state only
stay correct on a single replica: the per-org **rate limiter** (a sliding-window counter),
the **semantic-cache** invalidation signal, and the background **scheduler** (continuous
eval + connector sync). Scale to N replicas and each hazard appears — a tenant sprays
requests across replicas to beat the cap, replica B serves an answer replica A already
invalidated, every replica runs the same scheduled job.

This module is the single seam that removes all three. ``RAG_REDIS_URL`` empty ⇒
:class:`InMemoryCoordinator` (process-local primitives, exactly today's behaviour, zero
deps). Set it ⇒ :class:`RedisCoordinator` backs the same three primitives with Redis so
the platform scales horizontally with no call-site change — the identical
default-safe / production-plugs-in pattern used for the LLM gateway and OCR engine.

Redis is a self-hosted internal store (like Postgres/Qdrant/OpenSearch), not an AI or
cloud call, so it is deliberately NOT routed through ``assert_sovereign`` — the egress
guard governs outbound model/IdP/web traffic, not the platform's own datastore.

The primitives are synchronous by design: the rate limiter is checked from ~18 sync call
sites on the hot request path, and each op is a single pipelined round-trip to a
co-located Redis — awaiting it would ripple ``async`` through the whole request surface
for no latency win.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

from rag_common.config import get_settings
from rag_common.logging import get_logger

_log = get_logger(__name__)

_NS = "rag"

# Atomic sliding-window admission: drop expired, count, admit + record if under limit.
# KEYS[1]=zset  ARGV=[now, window_s, limit, member]  → {allowed(0|1), earliest_score}
_RL_LUA = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, ARGV[1] - ARGV[2])
local count = redis.call('ZCARD', KEYS[1])
if count >= tonumber(ARGV[3]) then
  local e = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
  return {0, e[2]}
end
redis.call('ZADD', KEYS[1], ARGV[1], ARGV[4])
redis.call('PEXPIRE', KEYS[1], math.ceil(ARGV[2] * 1000))
return {1, '0'}
"""

# Renew leadership only if we still hold it (compare-and-extend). KEYS[1]=lock key.
_RENEW_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  redis.call('PEXPIRE', KEYS[1], ARGV[2])
  return 1
end
return 0
"""


@runtime_checkable
class Coordinator(Protocol):
    """The three multi-replica primitives. One process or a hundred — same contract."""

    def allow(self, key: str, limit: int, window_s: float) -> tuple[bool, int]:
        """Record one event under a sliding window; return ``(allowed, retry_after_s)``.
        ``limit <= 0`` disables the limit (always allowed)."""
        ...

    def bump(self, key: str) -> int:
        """Increment a shared version counter (cache-coherence signal); return the new value."""
        ...

    def version(self, key: str) -> int:
        """Read a shared version counter (0 if never bumped)."""
        ...

    def acquire_leader(self, name: str, ttl_s: int, token: str) -> bool:
        """Try to take or renew leadership of ``name`` for ``ttl_s`` seconds. ``token``
        identifies this process. True ⇒ this replica is the leader for that lease."""
        ...


class InMemoryCoordinator:
    """Single-process default: leadership is trivially held; state is process-local.

    Behaviour is identical to the original in-process limiter/cache: within one process,
    a bump is immediately visible and there is exactly one leader.
    """

    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._versions: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window_s: float) -> tuple[bool, int]:
        if limit <= 0:
            return True, 0
        now = time.monotonic()
        with self._lock:
            q = self._events[key]
            while q and now - q[0] > window_s:
                q.popleft()
            if len(q) >= limit:
                return False, max(1, int(window_s - (now - q[0])) + 1)
            q.append(now)
            return True, 0

    def bump(self, key: str) -> int:
        with self._lock:
            self._versions[key] += 1
            return self._versions[key]

    def version(self, key: str) -> int:
        return self._versions[key]

    def acquire_leader(self, name: str, ttl_s: int, token: str) -> bool:
        return True  # a single process is always the leader

    def reset(self) -> None:
        """Clear all state (tests)."""
        with self._lock:
            self._events.clear()
            self._versions.clear()


class RedisCoordinator:
    """Redis-backed primitives for N horizontally-scaled replicas.

    Rate limiting is an atomic Lua sliding-window over a sorted set; the version counter
    is ``INCR``/``GET``; leadership is ``SET NX PX`` with a compare-and-extend renewal so
    a crashed leader's lease simply expires and another replica takes over.
    """

    def __init__(self, url: str) -> None:
        import redis  # noqa: PLC0415 — import-guarded optional dep (present only when scaled)

        self._r: Any = redis.Redis.from_url(url, decode_responses=True, socket_timeout=2.0)

    def allow(self, key: str, limit: int, window_s: float) -> tuple[bool, int]:
        if limit <= 0:
            return True, 0
        now = time.time()
        member = f"{now:.6f}:{time.monotonic_ns()}"  # unique per event within the window
        try:
            allowed, earliest = self._r.eval(_RL_LUA, 1, f"{_NS}:rl:{key}", now, window_s, limit, member)
        except Exception as exc:  # fail-open: a limiter outage must not take the platform down
            _log.warning("coordinator_rate_limit_degraded", error=str(exc)[:200])
            return True, 0
        if int(allowed) == 1:
            return True, 0
        retry = max(1, int(window_s - (now - float(earliest))) + 1) if earliest else 1
        return False, retry

    def bump(self, key: str) -> int:
        try:
            return int(self._r.incr(f"{_NS}:ver:{key}"))
        except Exception as exc:
            _log.warning("coordinator_bump_degraded", error=str(exc)[:200])
            return 0

    def version(self, key: str) -> int:
        try:
            v = self._r.get(f"{_NS}:ver:{key}")
            return int(v) if v is not None else 0
        except Exception as exc:
            _log.warning("coordinator_version_degraded", error=str(exc)[:200])
            return 0

    def acquire_leader(self, name: str, ttl_s: int, token: str) -> bool:
        lock = f"{_NS}:leader:{name}"
        try:
            if self._r.set(lock, token, nx=True, px=ttl_s * 1000):
                return True
            return bool(self._r.eval(_RENEW_LUA, 1, lock, token, ttl_s * 1000))
        except Exception as exc:  # fail-closed for leadership: don't run duplicate jobs on a blip
            _log.warning("coordinator_leader_degraded", error=str(exc)[:200])
            return False


@lru_cache(maxsize=1)
def get_coordinator() -> Coordinator:
    """The process-wide coordinator: Redis when ``RAG_REDIS_URL`` is set, else in-memory."""
    url = get_settings().redis_url
    if url:
        _log.info("coordinator_backend", backend="redis")
        return RedisCoordinator(url)
    return InMemoryCoordinator()


def get_redis_client() -> Any | None:
    """The SHARED Redis client the coordinator already built, or None single-process.

    Lets a caller stash a small shared blob (e.g. the continuous model-health snapshot) on the
    SAME connection pool the leader lease uses — no second client, no second URL, and it appears
    to every replica at once. Returns None with the in-memory coordinator (single process), where
    the caller keeps that state in-process instead."""
    coord = get_coordinator()
    return getattr(coord, "_r", None)


def reset_coordinator() -> None:
    """Drop the cached coordinator + clear in-memory state (tests / config reload)."""
    coord = get_coordinator()
    if isinstance(coord, InMemoryCoordinator):
        coord.reset()
    get_coordinator.cache_clear()
