"""Per-KB token budgets & spend caps (FR-H7) — cost control + noisy-neighbour limits.

A knowledge base can carry a monthly token budget. Before every generation the current
calendar month's total tokens (from the tamper-evident audit meter, FR-H4) is compared to
the cap: over the hard cap → the request is refused (429) with a clear message; over the
soft-alert threshold → answered but flagged in the budget view. 0 = unlimited.

Per-KB (not per-org) so enforcement is a single aggregation on the hot path; an org-wide
rollup over a tenant's KBs is the follow-up.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from rag_clients.factory import get_catalog
from rag_common import QuotaExceededError, get_logger
from svc_control_plane import wiring

_log = get_logger(__name__)

_BUDGET_KEY = "budget_tokens"
_ALERT_KEY = "budget_alert_pct"
_DEFAULT_ALERT = 0.8


class BudgetInfo(BaseModel):
    kb_id: str
    budget_tokens: int = Field(description="Monthly hard cap in total tokens; 0 = unlimited.")
    alert_pct: float = Field(description="Soft-alert threshold as a fraction of the cap (0-1).")
    used_tokens: int
    remaining_tokens: int
    pct_used: float | None = Field(default=None, description="used / budget; null when unlimited.")
    over_alert: bool
    over_budget: bool
    period_start: datetime


def _month_start(now: datetime | None = None) -> datetime:
    n = now or datetime.now(UTC)
    return n.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


async def get_budget(kb_id: str, *, now: datetime | None = None) -> BudgetInfo:
    config = await get_catalog().get_tenant_config(kb_id)
    cap = int(config.get(_BUDGET_KEY) or 0)
    alert = float(config.get(_ALERT_KEY) or _DEFAULT_ALERT)
    start = _month_start(now)
    used = int((await wiring.audit_service().token_totals(kb_id, start)).get("total_tokens", 0))
    pct = (used / cap) if cap else None
    return BudgetInfo(
        kb_id=kb_id,
        budget_tokens=cap,
        alert_pct=alert,
        used_tokens=used,
        remaining_tokens=max(0, cap - used) if cap else 0,
        pct_used=round(pct, 4) if pct is not None else None,
        over_alert=bool(cap and pct is not None and pct >= alert),
        over_budget=bool(cap and used >= cap),
        period_start=start,
    )


async def set_budget(kb_id: str, budget_tokens: int, alert_pct: float) -> BudgetInfo:
    catalog = get_catalog()
    await catalog.set_tenant_config(kb_id, _BUDGET_KEY, str(max(0, budget_tokens)))
    await catalog.set_tenant_config(kb_id, _ALERT_KEY, str(min(max(alert_pct, 0.1), 1.0)))
    _log.info("budget_set", kb_id=kb_id, budget_tokens=budget_tokens, alert_pct=alert_pct)
    return await get_budget(kb_id)


async def enforce_budget(kb_id: str) -> None:
    """Refuse generation when this month's tokens have hit the KB's hard cap (FR-H7)."""
    info = await get_budget(kb_id)
    if info.over_budget:
        _log.warning("budget_exceeded", kb_id=kb_id, used=info.used_tokens, cap=info.budget_tokens)
        raise QuotaExceededError(
            "this knowledge base has reached its monthly token budget — raise it in Usage & Showback, "
            "or wait for the next billing month",
            details={"budget_tokens": info.budget_tokens, "used_tokens": info.used_tokens},
        )
