"""Reciprocal Rank Fusion (RRF) — combine dense + sparse result lists (FR-D1).

RRF is rank-based (not score-based), so it fuses heterogeneous scorers (cosine vs
BM25) without needing score normalisation across systems. Score for a doc ``d``:

    RRF(d) = sum_over_lists 1 / (k + rank_list(d))

with ``k`` a smoothing constant (60 is the well-established default from the
original Cormack et al. formulation).
"""

from __future__ import annotations

RRF_K = 60


def reciprocal_rank_fusion(ranked_lists: list[list[tuple[str, float]]], *, k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse multiple ranked ``[(id, score)]`` lists into one, sorted best-first.

    Input scores are used only to break ties within a list before ranking; the fused
    score is purely rank-derived. Returns ``[(id, rrf_score)]`` descending.
    """
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        # Ensure a stable rank order per list (desc by score).
        ordered = sorted(ranked, key=lambda x: x[1], reverse=True)
        for rank, (doc_id, _score) in enumerate(ordered, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def normalize_scores(scored: list[tuple[str, float]]) -> dict[str, float]:
    """Min-max normalise fused scores into [0,1] for the confidence/abstain gate."""
    if not scored:
        return {}
    values = [s for _, s in scored]
    lo, hi = min(values), max(values)
    if hi == lo:
        return {doc_id: 1.0 for doc_id, _ in scored}
    return {doc_id: (s - lo) / (hi - lo) for doc_id, s in scored}
