"""Usage metering + FinOps showback (FR-H4).

Aggregates the tamper-evident audit trail into a per-tenant usage report over a time
window, and applies an operator-configurable rate card for showback. Metering reads only
countable audit actions — token-level accounting (capturing LLM usage on GENERATE events)
is a documented follow-up, not part of this cut.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, Field

from rag_common import get_settings
from svc_control_plane import wiring

# Window token -> lookback delta (None = all-time).
_WINDOWS: dict[str, timedelta | None] = {
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "all": None,
}


def window_since(window: str, *, now: datetime) -> datetime | None:
    """Resolve a window token to its ``since`` bound (raises on an unknown token)."""
    if window not in _WINDOWS:
        raise ValueError(f"unknown window '{window}'; expected one of {sorted(_WINDOWS)}")
    delta = _WINDOWS[window]
    return None if delta is None else now - delta


class UsageLine(BaseModel):
    """One priced line of the showback (a metered unit times its rate)."""

    unit: str = Field(..., description="Metered unit, e.g. 'query' or 'document'.")
    count: int
    rate: float = Field(..., description="Per-unit rate from the (indicative) rate card.")
    cost: float


class UsageReport(BaseModel):
    """Per-tenant usage + indicative showback over a window (FR-H4)."""

    tenant_id: str
    window: str
    since: datetime | None = Field(default=None, description="Lower time bound; null = all-time.")
    counts: dict[str, int] = Field(default_factory=dict, description="Raw audit-action counts.")
    queries: int
    documents: int
    abstains: int
    guardrail_blocks: int
    generations: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    currency: str
    rate_card_indicative: bool = Field(default=True, description="Defaults are nominal; operator sets real rates.")
    lines: list[UsageLine]
    total_cost: float


async def usage_report(tenant_id: str, window: str, *, now: datetime | None = None) -> UsageReport:
    """Build the usage report for ``tenant_id`` over ``window`` from the audit trail."""
    since = window_since(window, now=now or datetime.now(UTC))
    counts = await wiring.audit_service().count_actions(tenant_id, since)
    tokens = await wiring.audit_service().token_totals(tenant_id, since)
    s = get_settings()

    queries = counts.get("query", 0)
    documents = counts.get("ingest", 0)
    total_tokens = tokens.get("total_tokens", 0)
    q_rate, d_rate, t_rate = s.finops_cost_per_query, s.finops_cost_per_document, s.finops_cost_per_1k_tokens
    lines = [
        UsageLine(unit="query", count=queries, rate=q_rate, cost=round(queries * q_rate, 2)),
        UsageLine(unit="document", count=documents, rate=d_rate, cost=round(documents * d_rate, 2)),
        UsageLine(unit="1k tokens", count=total_tokens, rate=t_rate, cost=round(total_tokens / 1000 * t_rate, 2)),
    ]
    return UsageReport(
        tenant_id=tenant_id,
        window=window,
        since=since,
        counts=counts,
        queries=queries,
        documents=documents,
        abstains=counts.get("abstain", 0),
        guardrail_blocks=counts.get("guardrail", 0),
        generations=counts.get("generate", 0),
        prompt_tokens=tokens.get("prompt_tokens", 0),
        completion_tokens=tokens.get("completion_tokens", 0),
        total_tokens=total_tokens,
        currency=s.finops_currency,
        lines=lines,
        total_cost=round(sum(line.cost for line in lines), 2),
    )
