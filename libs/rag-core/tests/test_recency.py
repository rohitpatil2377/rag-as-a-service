"""Unit tests for FR-D9 recency-aware retrieval — pure logic."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rag_core.recency import apply_recency, in_window, recency_factor
from rag_schemas import RetrievedSpan

_NOW = datetime(2026, 7, 7, tzinfo=UTC)


def _span(doc_id: str, score: float) -> RetrievedSpan:
    return RetrievedSpan(chunk_id=f"c-{doc_id}", doc_id=doc_id, text="x", score=score, scorer="cross_encoder")


def test_in_window_bounds() -> None:
    ts = datetime(2025, 6, 1, tzinfo=UTC)
    assert in_window(ts, None, None)
    assert in_window(ts, datetime(2025, 1, 1, tzinfo=UTC), None)
    assert not in_window(ts, datetime(2026, 1, 1, tzinfo=UTC), None)
    assert not in_window(ts, None, datetime(2025, 1, 1, tzinfo=UTC))
    # Unknown timestamp only admitted with no bounds.
    assert in_window(None, None, None)
    assert not in_window(None, datetime(2025, 1, 1, tzinfo=UTC), None)


def test_recency_factor_decays_with_age() -> None:
    fresh = recency_factor(_NOW, _NOW, 180)
    old = recency_factor(_NOW - timedelta(days=360), _NOW, 180)
    assert fresh == 1.0
    assert abs(old - 0.25) < 1e-6  # two half-lives
    assert recency_factor(None, _NOW, 180) == 0.5


def test_apply_recency_filters_window() -> None:
    spans = [_span("new", 0.9), _span("old", 0.8)]
    times = {"new": _NOW - timedelta(days=10), "old": _NOW - timedelta(days=800)}
    kept = apply_recency(spans, times, after=_NOW - timedelta(days=365), before=None, freshness=0.0, now=_NOW)
    assert [s.doc_id for s in kept] == ["new"]  # 'old' filtered out


def test_freshness_boost_reranks() -> None:
    # 'old' has a higher base score but 'new' wins once freshness dominates.
    spans = [_span("old", 0.90), _span("new", 0.80)]
    times = {"old": _NOW - timedelta(days=720), "new": _NOW - timedelta(days=1)}
    kept = apply_recency(spans, times, after=None, before=None, freshness=1.0, now=_NOW)
    assert [s.doc_id for s in kept] == ["new", "old"]


def test_apply_recency_noop_when_inactive() -> None:
    spans = [_span("a", 0.5), _span("b", 0.4)]
    assert apply_recency(spans, {}, after=None, before=None, freshness=0.0, now=_NOW) is spans
