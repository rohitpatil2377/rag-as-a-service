"""Billing — pure period bounds + rate-card invoice math."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from svc_control_plane.billing import build_invoice, period_bounds


def test_period_bounds_month_and_year_rollover() -> None:
    assert period_bounds("2026-08") == (
        datetime(2026, 8, 1, tzinfo=UTC),
        datetime(2026, 9, 1, tzinfo=UTC),
    )
    assert period_bounds("2026-12") == (
        datetime(2026, 12, 1, tzinfo=UTC),
        datetime(2027, 1, 1, tzinfo=UTC),
    )


@pytest.mark.parametrize("bad", ["2026", "2026-13", "2026-00", "nope", "2026-8x"])
def test_bad_period_is_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="YYYY-MM"):
        period_bounds(bad)


def test_invoice_prices_from_the_rate_card() -> None:
    inv = build_invoice(
        "acme",
        "2026-08",
        counts={"query": 10, "federated_query": 2, "ingest": 3},
        total_tokens=20000,
        currency="INR",
        q_rate=0.5,
        d_rate=2.0,
        t_rate=0.1,
        plan="standard",
        monthly_budget=100.0,
        generated_at=datetime(2026, 8, 5, tzinfo=UTC),
    )
    assert inv.queries == 12 and inv.documents == 3 and inv.total_tokens == 20000
    # 12*0.5 + 3*2.0 + (20000/1000)*0.1 = 6 + 6 + 2 = 14
    assert inv.subtotal == 14.0
    assert inv.over_budget is False


def test_over_budget_when_cost_exceeds_the_plan() -> None:
    inv = build_invoice(
        "acme",
        "2026-08",
        counts={"query": 1000},
        total_tokens=0,
        currency="INR",
        q_rate=0.5,
        d_rate=2.0,
        t_rate=0.1,
        plan="trial",
        monthly_budget=100.0,
        generated_at=datetime(2026, 8, 5, tzinfo=UTC),
    )
    assert inv.subtotal == 500.0 and inv.over_budget is True
    # A zero budget (unmetered) is never "over".
    unmetered = build_invoice(
        "acme",
        "2026-08",
        counts={"query": 1000},
        total_tokens=0,
        currency="INR",
        q_rate=0.5,
        d_rate=2.0,
        t_rate=0.1,
        plan="unmetered",
        monthly_budget=0.0,
        generated_at=datetime(2026, 8, 5, tzinfo=UTC),
    )
    assert unmetered.over_budget is False
