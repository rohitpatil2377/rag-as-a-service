"""Pure threshold-calibration math (no I/O, no stack) — feeds ``scripts/calibrate.py``.

A gate compares one score against one threshold and PASSES when ``score >= threshold``.
Calibrating it means seeing the two populations it must separate — *positives* that should
pass (e.g. answerable questions at the abstain gate; supported answers at the faithfulness
gate) and *negatives* that should be stopped — and choosing the threshold that best splits
them. This module computes the confusion at every candidate threshold and picks the optimum,
so the orchestrator script only has to gather the score populations from a running stack.

Nothing here writes a threshold: it produces evidence + a recommendation; a human decides
(same discipline as the existing ``calibrate_abstain.py`` / ``calibrate_faithfulness.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

_EPS = 1e-9  # threshold/score equality tolerance


@dataclass(frozen=True)
class ThresholdCost:
    """Confusion for a ``score >= threshold`` gate over labelled score populations."""

    threshold: float
    tp: int  # positive that passes (correct)
    fn: int  # positive that is stopped (miss)
    tn: int  # negative that is stopped (correct)
    fp: int  # negative that passes (leak)

    @property
    def recall(self) -> float:
        """Fraction of positives the gate lets through (sensitivity)."""
        n = self.tp + self.fn
        return self.tp / n if n else 0.0

    @property
    def specificity(self) -> float:
        """Fraction of negatives the gate stops."""
        n = self.tn + self.fp
        return self.tn / n if n else 0.0

    @property
    def precision(self) -> float:
        n = self.tp + self.fp
        return self.tp / n if n else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        n = self.tp + self.fn + self.tn + self.fp
        return (self.tp + self.tn) / n if n else 0.0

    @property
    def youden_j(self) -> float:
        """recall + specificity - 1 — maximised by the threshold that best separates the two
        populations, independent of their relative sizes (good for an imbalanced golden set)."""
        return self.recall + self.specificity - 1.0


def confusion_at(threshold: float, positives: list[float], negatives: list[float]) -> ThresholdCost:
    """Confusion for a single threshold over the two score populations."""
    tp = sum(1 for s in positives if s >= threshold)
    fp = sum(1 for s in negatives if s >= threshold)
    return ThresholdCost(threshold=threshold, tp=tp, fn=len(positives) - tp, tn=len(negatives) - fp, fp=fp)


def candidate_grid(lo: float = 0.0, hi: float = 1.0, step: float = 0.01) -> list[float]:
    """Inclusive threshold grid ``[lo, hi]`` at ``step`` spacing (rounded to avoid FP drift)."""
    if step <= 0:
        raise ValueError("step must be positive")
    n = round((hi - lo) / step)
    return [round(lo + i * step, 6) for i in range(n + 1)]


def sweep(positives: list[float], negatives: list[float], candidates: list[float]) -> list[ThresholdCost]:
    """Confusion at every candidate threshold."""
    return [confusion_at(t, positives, negatives) for t in candidates]


def best_threshold(costs: list[ThresholdCost], *, key: str = "youden_j") -> ThresholdCost | None:
    """The threshold maximising ``key`` (``youden_j`` by default; ``f1``/``accuracy`` also valid).
    Ties break toward the LOWER threshold (more permissive — a false-abstain is usually costlier
    than a marginal leak, and the caller can re-rank). None if there are no candidates."""
    if not costs:
        return None
    return max(costs, key=lambda c: (getattr(c, key), -c.threshold))


def separation_gap(positives: list[float], negatives: list[float]) -> float | None:
    """min(positives) - max(negatives): > 0 means the populations are cleanly separable (any
    threshold in the gap scores perfectly). Negative means they overlap. None if either empty."""
    if not positives or not negatives:
        return None
    return min(positives) - max(negatives)


def format_report(
    label: str,
    *,
    current: float,
    positives: list[float],
    negatives: list[float],
    candidates: list[float],
    key: str = "youden_j",
) -> str:
    """A markdown before/after block: where the two populations sit, what the CURRENT threshold
    costs, and what the swept OPTIMUM costs — so the delta of changing it is explicit."""
    lines = [f"### {label}", ""]
    if not positives and not negatives:
        return "\n".join([*lines, "_No samples gathered — cannot calibrate._", ""])
    gap = separation_gap(positives, negatives)
    lines += [
        f"- positives (should pass): n={len(positives)}"
        + (f", min={min(positives):.4f}, mean={_mean(positives):.4f}" if positives else ""),
        f"- negatives (should stop): n={len(negatives)}"
        + (f", max={max(negatives):.4f}, mean={_mean(negatives):.4f}" if negatives else ""),
        f"- separation gap (min_pos - max_neg): {gap:.4f}"
        if gap is not None
        else "- separation gap: n/a (one population empty)",
        "",
    ]
    costs = sweep(positives, negatives, candidates)
    cur = confusion_at(current, positives, negatives)
    best = best_threshold(costs, key=key)
    lines += [
        "| | threshold | recall | specificity | F1 | Youden J | leaks(fp) | misses(fn) |",
        "|---|---|---|---|---|---|---|---|",
        _row("current", cur),
    ]
    if best is not None:
        lines.append(_row(f"optimal ({key})", best))
        # Keep the current threshold when it ALREADY achieves the optimal score — even if some
        # other threshold ties on the metric, moving a working gate for no gain is just noise.
        if getattr(cur, key) >= getattr(best, key) - _EPS:
            lines += ["", f"**Recommendation:** keep `{current}` — it is already optimal on this set."]
        else:
            lines += [
                "",
                f"**Recommendation:** move `{current}` → `{best.threshold}` "
                f"(Youden J {cur.youden_j:.3f} → {best.youden_j:.3f}; "
                f"leaks {cur.fp}→{best.fp}, misses {cur.fn}→{best.fn}). "
                "Evidence only — a human sets the value.",
            ]
    lines.append("")
    return "\n".join(lines)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _row(name: str, c: ThresholdCost) -> str:
    return (
        f"| {name} | {c.threshold:.3f} | {c.recall:.3f} | {c.specificity:.3f} | "
        f"{c.f1:.3f} | {c.youden_j:.3f} | {c.fp} | {c.fn} |"
    )
