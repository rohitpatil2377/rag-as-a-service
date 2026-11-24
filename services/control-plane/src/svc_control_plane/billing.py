"""Billing hook — turn the FinOps usage meter into a per-tenant, per-period INVOICE (FR-H4+).

Reads the tamper-evident audit chain (the immutable source of metered usage), applies the
operator rate card, aggregates across ALL of a tenant's KBs, and produces a deterministic invoice
for a calendar month — plus a budget-exceeded signal against the tenant's plan. Every invoice is
recorded as a BILLING audit event, so an external billing system pulls an AUDITABLE statement (a
row on the tamper-evident chain) rather than a live, mutable feed. Same period in, same numbers
out.

Metering per period uses the existing ``count_actions(since)`` meter twice and subtracts:
``usage[start, end) = usage(since=start) - usage(since=end)`` — no new store method, and a
future ``end`` (the current month) simply yields ``usage(since=start)``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from rag_common import get_logger, get_settings
from rag_schemas.common import AuditAction
from svc_control_plane import wiring

_log = get_logger(__name__)
_MONTHS = 12


def period_bounds(period: str) -> tuple[datetime, datetime]:
    """``[start, end)`` UTC bounds of a calendar month ``'YYYY-MM'``."""
    try:
        year_s, month_s = period.split("-")
        year, month = int(year_s), int(month_s)
        if not 1 <= month <= _MONTHS:
            raise ValueError
        start = datetime(year, month, 1, tzinfo=UTC)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"period must be 'YYYY-MM', got {period!r}") from exc
    end = datetime(year + (month == _MONTHS), (month % _MONTHS) + 1, 1, tzinfo=UTC)
    return start, end


class InvoiceLine(BaseModel):
    unit: str
    count: int
    rate: float
    cost: float


class Invoice(BaseModel):
    org_id: str
    period: str
    currency: str
    plan: str
    monthly_budget: float = Field(description="The tenant plan's budget; 0 = unmetered / no cap.")
    queries: int
    documents: int
    total_tokens: int
    lines: list[InvoiceLine]
    subtotal: float
    over_budget: bool
    generated_at: datetime


def build_invoice(
    org_id: str,
    period: str,
    *,
    counts: dict[str, int],
    total_tokens: int,
    currency: str,
    q_rate: float,
    d_rate: float,
    t_rate: float,
    plan: str,
    monthly_budget: float,
    generated_at: datetime,
) -> Invoice:
    """Pure: apply the rate card to metered counts -> priced invoice + budget verdict."""
    queries = counts.get("query", 0) + counts.get("federated_query", 0)
    documents = counts.get("ingest", 0)
    lines = [
        InvoiceLine(unit="query", count=queries, rate=q_rate, cost=round(queries * q_rate, 2)),
        InvoiceLine(unit="document", count=documents, rate=d_rate, cost=round(documents * d_rate, 2)),
        InvoiceLine(unit="1k tokens", count=total_tokens, rate=t_rate, cost=round(total_tokens / 1000 * t_rate, 2)),
    ]
    subtotal = round(sum(line.cost for line in lines), 2)
    return Invoice(
        org_id=org_id,
        period=period,
        currency=currency,
        plan=plan,
        monthly_budget=monthly_budget,
        queries=queries,
        documents=documents,
        total_tokens=total_tokens,
        lines=lines,
        subtotal=subtotal,
        over_budget=bool(monthly_budget > 0 and subtotal > monthly_budget),
        generated_at=generated_at,
    )


async def _org_kb_ids(org_id: str) -> list[str]:
    """The org root plus every KB it owns — billing aggregates across the whole tenant."""
    kbs = {org_id}
    for kb in await wiring.kb_store().list(org_id):
        kbs.add(kb.kb_id)
    return sorted(kbs)


async def _period_meters(org_id: str, start: datetime, end: datetime) -> tuple[dict[str, int], int]:
    audit = wiring.audit_service()
    upper: datetime | None = None if end > datetime.now(UTC) else end  # current month -> open upper bound
    counts: dict[str, int] = {}
    tokens = 0
    for kb in await _org_kb_ids(org_id):
        start_counts = await audit.count_actions(kb, start)
        end_counts = await audit.count_actions(kb, upper) if upper else {}
        for k, v in start_counts.items():
            counts[k] = counts.get(k, 0) + v - end_counts.get(k, 0)
        start_tok = (await audit.token_totals(kb, start)).get("total_tokens", 0)
        end_tok = (await audit.token_totals(kb, upper)).get("total_tokens", 0) if upper else 0
        tokens += start_tok - end_tok
    return counts, tokens


async def invoice(org_id: str, period: str, *, actor: str = "billing", now: datetime | None = None) -> Invoice:
    """Compute the tenant's invoice for a calendar month, record it on the audit chain, and raise a
    budget-exceeded alert when the period cost passes the plan's monthly budget."""
    start, end = period_bounds(period)
    counts, tokens = await _period_meters(org_id, start, end)
    settings = get_settings()
    rec = await wiring.tenant_registry().get(org_id)
    inv = build_invoice(
        org_id,
        period,
        counts=counts,
        total_tokens=tokens,
        currency=settings.finops_currency,
        q_rate=settings.finops_cost_per_query,
        d_rate=settings.finops_cost_per_document,
        t_rate=settings.finops_cost_per_1k_tokens,
        plan=rec.plan if rec else "unmetered",
        monthly_budget=rec.monthly_budget if rec else 0.0,
        generated_at=now or datetime.now(UTC),
    )
    audit = wiring.audit_service()
    await audit.record(
        tenant_id=org_id,
        trace_id=f"invoice:{period}",
        action=AuditAction.BILLING,
        actor=actor,
        payload={
            "event": "invoice",
            "period": period,
            "subtotal": inv.subtotal,
            "currency": inv.currency,
            "over_budget": inv.over_budget,
        },
    )
    if inv.over_budget:
        _log.warning(
            "billing_over_budget", org_id=org_id, period=period, subtotal=inv.subtotal, budget=inv.monthly_budget
        )
        await audit.record(
            tenant_id=org_id,
            trace_id=f"invoice:{period}",
            action=AuditAction.BILLING,
            actor=actor,
            payload={
                "event": "budget_exceeded",
                "period": period,
                "subtotal": inv.subtotal,
                "budget": inv.monthly_budget,
            },
        )
    return inv
