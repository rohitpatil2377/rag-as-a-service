"""Recency-aware retrieval (FR-D9) — pure logic, no framework.

Two optional, per-query behaviours over already-retrieved spans, keyed off each span's
document ingest time:
* **time-window filter** — drop spans whose document falls outside ``[after, before]``
  ("only documents from 2025");
* **freshness boost** — multiply a span's score by a recency factor (half-life decay), so
  fresher documents rank higher without discarding older-but-relevant ones.

Both default off (weight 0 / no bounds) → ranking is unchanged. Applied AFTER retrieval so
no index/schema change is needed; the caller over-fetches to preserve recall under a filter.
"""

from __future__ import annotations

from datetime import datetime

from rag_schemas import RetrievedSpan

_DAY_SECONDS = 86400.0


def recency_factor(ingested_at: datetime | None, now: datetime, half_life_days: float) -> float:
    """A 0-1 freshness weight: 1.0 for brand-new, halving every ``half_life_days``. Unknown
    timestamps get a neutral-ish 0.5 so they are neither boosted nor buried."""
    if ingested_at is None or half_life_days <= 0:
        return 0.5
    age_days = max(0.0, (now - ingested_at).total_seconds() / _DAY_SECONDS)
    return float(0.5 ** (age_days / half_life_days))


def in_window(ingested_at: datetime | None, after: datetime | None, before: datetime | None) -> bool:
    """True if the document's ingest time is within the (optional) bounds. An unknown
    timestamp is only admitted when no bounds are set."""
    if after is None and before is None:
        return True
    if ingested_at is None:
        return False
    if after is not None and ingested_at < after:
        return False
    return not (before is not None and ingested_at > before)


def apply_recency(
    spans: list[RetrievedSpan],
    doc_times: dict[str, datetime],
    *,
    after: datetime | None,
    before: datetime | None,
    freshness: float,
    now: datetime,
    half_life_days: float = 180.0,
) -> list[RetrievedSpan]:
    """Filter spans to the time window and (optionally) re-rank by a freshness-boosted score.
    Returns spans sorted by the effective score, highest first. A no-op when there is no
    window and ``freshness == 0``."""
    if after is None and before is None and freshness <= 0:
        return spans
    kept: list[tuple[float, RetrievedSpan]] = []
    for s in spans:
        ts = doc_times.get(s.doc_id)
        if not in_window(ts, after, before):
            continue
        score = s.score
        if freshness > 0:
            factor = recency_factor(ts, now, half_life_days)
            score = s.score * ((1.0 - freshness) + freshness * factor)
        kept.append((score, s))
    kept.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in kept]
