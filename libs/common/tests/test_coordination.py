"""Unit tests for the scale/HA coordination seam (in-memory default primitives)."""

from __future__ import annotations

from rag_common.coordination import InMemoryCoordinator


def test_sliding_window_admits_then_blocks() -> None:
    c = InMemoryCoordinator()
    assert c.allow("org:query", limit=2, window_s=60)[0] is True
    assert c.allow("org:query", limit=2, window_s=60)[0] is True
    allowed, retry = c.allow("org:query", limit=2, window_s=60)
    assert allowed is False and retry >= 1


def test_zero_limit_disables() -> None:
    c = InMemoryCoordinator()
    for _ in range(100):
        assert c.allow("org:ingest", limit=0, window_s=60) == (True, 0)


def test_window_keys_are_independent() -> None:
    c = InMemoryCoordinator()
    assert c.allow("a:query", 1, 60)[0] is True
    assert c.allow("a:query", 1, 60)[0] is False
    assert c.allow("b:query", 1, 60)[0] is True  # different org unaffected


def test_version_bump_and_read() -> None:
    c = InMemoryCoordinator()
    assert c.version("semcache:kb1") == 0
    assert c.bump("semcache:kb1") == 1
    assert c.bump("semcache:kb1") == 2
    assert c.version("semcache:kb1") == 2
    assert c.version("semcache:kb2") == 0  # untouched key


def test_single_process_is_always_leader() -> None:
    c = InMemoryCoordinator()
    assert c.acquire_leader("scheduler", 90, "token-a") is True
    assert c.acquire_leader("scheduler", 90, "token-b") is True  # single process = always leader


def test_reset_clears_state() -> None:
    c = InMemoryCoordinator()
    c.allow("org:query", 1, 60)
    c.bump("semcache:kb1")
    c.reset()
    assert c.allow("org:query", 1, 60)[0] is True
    assert c.version("semcache:kb1") == 0
