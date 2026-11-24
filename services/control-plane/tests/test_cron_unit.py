"""Cron scheduling engine — parsing, next-occurrence, timezone, due-logic (banking pack)."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest
from svc_control_plane.cron import cron_is_due, cron_next, next_run, parse_cron, validate_cron

_IST = "Asia/Kolkata"


def _dt(y: int, mo: int, d: int, h: int, mi: int, tz: str = "UTC") -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=ZoneInfo(tz))


def test_parse_rejects_bad_expressions() -> None:
    for bad in ["* * * *", "60 * * * *", "* 24 * * *", "0 0 32 * *", "0 0 * 13 *", "a * * * *"]:
        with pytest.raises(ValueError):
            validate_cron(bad)


def test_every_monday_0200() -> None:
    spec = parse_cron("0 2 * * 1")  # Monday 02:00
    # 2026-07-14 is a Tuesday; next Monday is 2026-07-20.
    nxt = cron_next(spec, _dt(2026, 7, 14, 10, 0), ZoneInfo("UTC"))
    assert (nxt.year, nxt.month, nxt.day, nxt.hour, nxt.minute) == (2026, 7, 20, 2, 0)
    assert nxt.isoweekday() == 1  # Monday


def test_first_of_month_midnight() -> None:
    spec = parse_cron("0 0 1 * *")
    nxt = cron_next(spec, _dt(2026, 7, 14, 10, 0), ZoneInfo("UTC"))
    assert (nxt.month, nxt.day, nxt.hour, nxt.minute) == (8, 1, 0, 0)


def test_dom_and_dow_are_union() -> None:
    # "0 0 1 * 1" = midnight on the 1st OR any Monday (cron OR semantics).
    spec = parse_cron("0 0 1 * 1")
    # From Wed 2026-07-15: next is Monday 2026-07-20 (a Monday) — earlier than the 1st.
    nxt = cron_next(spec, _dt(2026, 7, 15, 9, 0), ZoneInfo("UTC"))
    assert nxt.day == 20 and nxt.isoweekday() == 1


def test_timezone_is_respected() -> None:
    # "0 0 * * *" daily midnight IST == 18:30 UTC the previous day.
    nxt = next_run("0 0 * * *", _IST, _dt(2026, 7, 14, 10, 0))
    assert nxt.astimezone(ZoneInfo(_IST)).hour == 0
    assert nxt.astimezone(UTC).hour == 18 and nxt.astimezone(UTC).minute == 30


def test_is_due_transition() -> None:
    anchor = _dt(2026, 7, 14, 10, 0)  # Tuesday
    # Monday 02:00 schedule: not due before next Monday, due at/after it.
    assert cron_is_due("0 2 * * 1", "UTC", anchor, _dt(2026, 7, 19, 23, 0)) is False  # Sunday night
    assert cron_is_due("0 2 * * 1", "UTC", anchor, _dt(2026, 7, 20, 2, 0)) is True  # Monday 02:00
    assert cron_is_due("0 2 * * 1", "UTC", anchor, _dt(2026, 7, 20, 1, 59)) is False


def test_no_anchor_is_not_due() -> None:
    # Without an anchor a cron schedule waits (it is anchored at enable time in the store).
    assert cron_is_due("* * * * *", "UTC", None, datetime.now(UTC)) is False


def test_step_and_list_syntax() -> None:
    spec = parse_cron("*/15 9-17 * * 1-5")  # every 15 min, 9-17h, Mon-Fri
    assert spec.minutes == frozenset({0, 15, 30, 45})
    assert spec.hours == frozenset(range(9, 18))
    assert spec.dows == frozenset({1, 2, 3, 4, 5})
