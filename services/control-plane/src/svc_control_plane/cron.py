"""Calendar-precise (cron) scheduling — pure, dependency-free (banking pack).

The sync scheduler supports either a simple hour interval OR a 5-field cron expression
(``minute hour day-of-month month day-of-week``) evaluated in a named timezone, so a bank can
say "every Monday 02:00" (``0 2 * * 1``) or "1st of each month at midnight" (``0 0 1 * *``) in
IST. Standard cron field syntax: ``*``, ``a``, ``a-b``, ``a,b``, ``*/n``, ``a-b/n``.

Hand-rolled (no croniter dependency, per ADR-0002) and pure so the due-logic is trivially
testable. Day-of-week: 0/7 = Sunday. If BOTH day-of-month and day-of-week are restricted, a day
matches if EITHER matches (standard cron semantics).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_MAX_SEARCH_DAYS = 366 * 5  # covers even a Feb-29-only cron (leap year)
_N_FIELDS = 5  # minute hour day-of-month month day-of-week
_SUNDAY_ALIAS = 7  # cron allows 7 as an alias for Sunday (0)


@dataclass(frozen=True, slots=True)
class CronSpec:
    minutes: frozenset[int]
    hours: frozenset[int]
    doms: frozenset[int]
    months: frozenset[int]
    dows: frozenset[int]
    dom_restricted: bool
    dow_restricted: bool


def _parse_field(field: str, lo: int, hi: int) -> set[int]:
    values: set[int] = set()
    for raw in field.split(","):
        part = raw.strip()
        step = 1
        if "/" in part:
            part, step_s = part.split("/", 1)
            step = int(step_s)
            if step < 1:
                raise ValueError(f"invalid step '{step_s}'")
        if part == "*":
            start, end = lo, hi
        elif "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
        else:
            start = end = int(part)
        if not (lo <= start <= hi and lo <= end <= hi and start <= end):
            raise ValueError(f"value out of range [{lo},{hi}]: '{raw}'")
        values.update(range(start, end + 1, step))
    return values


def parse_cron(expr: str) -> CronSpec:
    """Parse a 5-field cron expression, raising ValueError if malformed."""
    fields = expr.split()
    if len(fields) != _N_FIELDS:
        raise ValueError("cron must have 5 fields: 'minute hour day-of-month month day-of-week'")
    minutes = _parse_field(fields[0], 0, 59)
    hours = _parse_field(fields[1], 0, 23)
    doms = _parse_field(fields[2], 1, 31)
    months = _parse_field(fields[3], 1, 12)
    dows = {0 if d == _SUNDAY_ALIAS else d for d in _parse_field(fields[4], 0, 7)}
    return CronSpec(
        frozenset(minutes),
        frozenset(hours),
        frozenset(doms),
        frozenset(months),
        frozenset(dows),
        dom_restricted=fields[2] != "*",
        dow_restricted=fields[4] != "*",
    )


def validate_cron(expr: str) -> None:
    """Raise ValueError if the expression is not a valid 5-field cron."""
    parse_cron(expr)


def _day_matches(spec: CronSpec, d: datetime) -> bool:
    if d.month not in spec.months:
        return False
    dom_ok = d.day in spec.doms
    dow_ok = (d.isoweekday() % 7) in spec.dows  # isoweekday: Mon=1..Sun=7 -> cron Sun=0..Sat=6
    if spec.dom_restricted and spec.dow_restricted:
        return dom_ok or dow_ok
    if spec.dom_restricted:
        return dom_ok
    if spec.dow_restricted:
        return dow_ok
    return True


def cron_next(spec: CronSpec, after: datetime, tz: ZoneInfo) -> datetime:
    """The first cron occurrence strictly after ``after`` (tz-aware), in ``tz``."""
    start = after.astimezone(tz).replace(second=0, microsecond=0) + timedelta(minutes=1)
    day = start
    for _ in range(_MAX_SEARCH_DAYS):
        if _day_matches(spec, day):
            same_day = day.date() == start.date()
            floor = start.time() if same_day else time(0, 0)
            for hh in sorted(spec.hours):
                if hh < floor.hour:
                    continue
                for mm in sorted(spec.minutes):
                    if hh == floor.hour and mm < floor.minute:
                        continue
                    return datetime(day.year, day.month, day.day, hh, mm, tzinfo=tz)
        day = (day + timedelta(days=1)).replace(hour=0, minute=0)
    raise ValueError("no cron occurrence found within 5 years")


def resolve_tz(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name or "UTC")
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        return ZoneInfo("UTC")


def next_run(expr: str, tz_name: str, after: datetime) -> datetime:
    """Next scheduled run (UTC) after ``after`` for a cron expression."""
    tz = resolve_tz(tz_name)
    return cron_next(parse_cron(expr), after, tz).astimezone(after.tzinfo or tz)


def cron_is_due(expr: str, tz_name: str, last_run_at: datetime | None, now: datetime) -> bool:
    """True if a cron occurrence has passed since ``last_run_at`` (the anchor). When there is no
    anchor yet the schedule is not due — it is anchored at enable time so it fires at the NEXT
    occurrence, never immediately."""
    if last_run_at is None:
        return False
    tz = resolve_tz(tz_name)
    nxt = cron_next(parse_cron(expr), last_run_at, tz)
    return now.astimezone(tz) >= nxt
