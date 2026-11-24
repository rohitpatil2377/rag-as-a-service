"""Abstain & feedback analytics — "every out-of-scope is a lead" (FR-G5 extension).

A read-only view over a KB's audit trail that surfaces the DEMAND signal a grounded-or-abstain
platform is uniquely able to see: what users asked that the KB could NOT answer.

* **abstain rate** — ABSTAINED / total QUERY over the window.
* **clustered abstained queries** — abstained question texts grouped by a cheap heuristic key
  (a normalized leading n-gram, else the salient terms), so many "list the …" abstains collapse
  into one missing-intent cluster with a count + examples.
* **thumbs-down queries** — FR-I2 down votes, correlated back to the question (and comment).
* **low-faithfulness answers** — FAITHFULNESS verdicts below the KB's flag threshold.

Every read goes through the KB-scoped UNION audit read (:meth:`AuditService.list_events_kb`) so
a multi-KB org sees its query-time events even if some reached the org schema — the same fix the
quality dashboard now relies on. Console wiring is OUT OF SCOPE (backend + endpoint only).
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from rag_common import get_logger, get_settings
from rag_schemas.common import AuditAction
from svc_control_plane import wiring
from svc_control_plane.pipeline import kb_threshold_overrides
from svc_control_plane.usage import window_since

_log = get_logger(__name__)

# Bound the union read: analytics lists top clusters + examples, so a windowed sample is enough
# and a runaway KB can never pull an unbounded result set into memory.
_MAX_EVENTS = 5000
_MAX_CLUSTERS = 10
_MAX_EXAMPLES = 3
_MAX_THUMBS_DOWN = 50
_MAX_LOW_FAITHFULNESS = 50
_LEADING_NGRAM = 3  # cluster key: the first N significant words of the question
_MIN_WORD_LEN = 3  # shorter tokens are noise for the heuristic cluster key

# Small stopword set for the heuristic cluster key (mirrors the conversation condenser's intent).
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "for",
        "to",
        "is",
        "are",
        "what",
        "which",
        "who",
        "one",
        "about",
        "how",
        "why",
        "do",
        "does",
        "can",
        "i",
        "me",
        "my",
        "in",
        "on",
        "at",
        "with",
        "please",
        "tell",
        "give",
        "show",
        "list",
        "all",
        "any",
    }
)
_WORD_RE = re.compile(r"[a-z0-9]+")


def _significant_words(text: str) -> list[str]:
    """Lowercase content words (stopwords + very short tokens dropped), in order."""
    return [w for w in _WORD_RE.findall(text.lower()) if len(w) >= _MIN_WORD_LEN and w not in _STOPWORDS]


def _cluster_key(text: str) -> str:
    """A cheap, deterministic grouping key for an abstained query. The first few significant
    words capture the shared intent of e.g. every "list the invoices for …" question; when a
    query has none (all stopwords), fall back to its normalized whole text so it still groups
    with identical asks rather than vanishing."""
    words = _significant_words(text)
    if words:
        return " ".join(words[:_LEADING_NGRAM])
    return " ".join(text.lower().split())


class AbstainCluster(BaseModel):
    """A group of abstained questions that share a leading-intent key (the missing-intent signal)."""

    key: str = Field(description="Normalized leading-intent key the queries share.")
    count: int
    examples: list[str] = Field(default_factory=list, description="Representative original query texts.")


class ThumbsDownItem(BaseModel):
    trace_id: str
    query: str | None = Field(default=None, description="The rated question (from the correlated QUERY event).")
    comment: str | None = None
    at: datetime | None = None


class LowFaithfulnessItem(BaseModel):
    trace_id: str
    query: str
    score: float
    outcome: str | None = None
    at: datetime | None = None


class AnalyticsGaps(BaseModel):
    """Abstain + feedback analytics for a KB over a window."""

    tenant_id: str
    window: str
    total_queries: int = Field(description="QUERY events over the window (the abstain-rate denominator).")
    abstained: int = Field(description="Abstained answers: retrieval/empty/guardrail ABSTAIN + faithfulness-abstained.")
    abstain_rate: float | None = Field(default=None, description="abstained / total_queries; null when no queries.")
    flag_threshold: float = Field(description="Faithfulness flag threshold used for low-faithfulness selection.")
    abstain_clusters: list[AbstainCluster] = Field(default_factory=list)
    thumbs_down: list[ThumbsDownItem] = Field(default_factory=list)
    low_faithfulness: list[LowFaithfulnessItem] = Field(default_factory=list)


async def analytics_gaps(tenant_id: str, window: str, *, now: datetime | None = None) -> AnalyticsGaps:
    """Build the abstain/feedback analytics for ``tenant_id`` over ``window`` (24h | 7d | 30d | all)."""
    since = window_since(window, now=now or datetime.now(UTC))
    audit = wiring.audit_service()
    overrides = await kb_threshold_overrides(tenant_id)
    flag_gate = overrides.get("flag_threshold", get_settings().flag_threshold)

    events = await audit.list_events_kb(
        tenant_id,
        since,
        actions=[
            AuditAction.QUERY.value,
            AuditAction.ABSTAIN.value,
            AuditAction.FAITHFULNESS.value,
            AuditAction.FEEDBACK.value,
        ],
        limit=_MAX_EVENTS,
    )

    total_queries = 0
    abstained = 0
    cluster_counts: dict[str, int] = defaultdict(int)
    cluster_examples: dict[str, list[str]] = defaultdict(list)
    thumbs_down: list[ThumbsDownItem] = []
    low_faithfulness: list[LowFaithfulnessItem] = []
    query_texts: dict[str, str] = {}  # trace_id -> question, to correlate feedback back to its query

    for e in events:
        payload = e.payload
        if e.action is AuditAction.QUERY:
            total_queries += 1
            q = payload.get("query")
            if isinstance(q, str):
                query_texts.setdefault(e.trace_id, q)
        elif e.action is AuditAction.ABSTAIN:
            abstained += 1
            _add_cluster(payload.get("query"), cluster_counts, cluster_examples)
        elif e.action is AuditAction.FAITHFULNESS:
            score = payload.get("score")
            outcome = payload.get("outcome")
            q = payload.get("query")
            if outcome == "abstained":
                abstained += 1
                _add_cluster(q, cluster_counts, cluster_examples)
            if (
                isinstance(score, int | float)
                and float(score) < flag_gate
                and isinstance(q, str)
                and len(low_faithfulness) < _MAX_LOW_FAITHFULNESS
            ):
                low_faithfulness.append(
                    LowFaithfulnessItem(
                        trace_id=e.trace_id,
                        query=q,
                        score=round(float(score), 4),
                        outcome=outcome if isinstance(outcome, str) else None,
                        at=e.at,
                    )
                )
        elif e.action is AuditAction.FEEDBACK and payload.get("rating") == "down":
            if len(thumbs_down) < _MAX_THUMBS_DOWN:
                comment = payload.get("comment")
                thumbs_down.append(
                    ThumbsDownItem(
                        trace_id=e.trace_id,
                        query=None,  # filled after the pass, once every QUERY is indexed
                        comment=comment if isinstance(comment, str) and comment else None,
                        at=e.at,
                    )
                )

    # Correlate each down-vote to its question by trace_id (QUERY may sort after the FEEDBACK).
    thumbs_down = [item.model_copy(update={"query": query_texts.get(item.trace_id)}) for item in thumbs_down]

    clusters = [
        AbstainCluster(key=key, count=cluster_counts[key], examples=cluster_examples[key])
        for key in sorted(cluster_counts, key=lambda k: cluster_counts[k], reverse=True)[:_MAX_CLUSTERS]
    ]

    return AnalyticsGaps(
        tenant_id=tenant_id,
        window=window,
        total_queries=total_queries,
        abstained=abstained,
        abstain_rate=round(abstained / total_queries, 4) if total_queries else None,
        flag_threshold=flag_gate,
        abstain_clusters=clusters,
        thumbs_down=thumbs_down,
        low_faithfulness=low_faithfulness,
    )


def _add_cluster(query: object, counts: dict[str, int], examples: dict[str, list[str]]) -> None:
    """Fold one abstained query text into its heuristic intent cluster."""
    if not isinstance(query, str) or not query.strip():
        return
    key = _cluster_key(query)
    counts[key] += 1
    if len(examples[key]) < _MAX_EXAMPLES and query not in examples[key]:
        examples[key].append(query)
