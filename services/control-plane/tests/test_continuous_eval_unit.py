"""Unit tests for FR-G9 continuous-evaluation pure helpers (no I/O)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from svc_control_plane.continuous_eval import breach_reasons, is_due, select_sample
from svc_control_plane.eval_schedule import EvalSchedule


def test_is_due_when_never_run() -> None:
    s = EvalSchedule(kb_id="k", interval_hours=24)
    assert is_due(s, datetime(2026, 7, 7, tzinfo=UTC))


def test_is_due_respects_interval() -> None:
    now = datetime(2026, 7, 7, 12, tzinfo=UTC)
    recent = EvalSchedule(kb_id="k", interval_hours=24, last_run_at=now - timedelta(hours=1))
    stale = EvalSchedule(kb_id="k", interval_hours=24, last_run_at=now - timedelta(hours=25))
    assert not is_due(recent, now)
    assert is_due(stale, now)


def test_select_sample_dedupes_and_caps() -> None:
    texts = ["a", "a", "b", "c", "d", "e"]  # 5 unique
    # rate 0.5 of 5 -> ceil 3, cap 10 -> 3; recency order preserved
    assert select_sample(texts, 0.5, 10) == ["a", "b", "c"]
    # cap wins
    assert select_sample(texts, 1.0, 2) == ["a", "b"]
    # zero rate -> nothing
    assert select_sample(texts, 0.0, 10) == []
    # always at least one when rate>0 and there is traffic
    assert select_sample(["x", "y", "z"], 0.01, 10) == ["x"]


def test_select_sample_ignores_blanks() -> None:
    assert select_sample(["", "  ", "real"], 1.0, 10) == ["real"]


def test_breach_reasons_flags_each_floor() -> None:
    s = EvalSchedule(kb_id="k", min_grounded_rate=0.6, min_faithfulness=0.5)
    assert breach_reasons(0.9, 0.8, s) == []
    assert len(breach_reasons(0.4, 0.8, s)) == 1  # grounded floor only
    assert len(breach_reasons(0.4, 0.2, s)) == 2  # both
    # None metrics never breach (nothing measured)
    assert breach_reasons(None, None, s) == []
