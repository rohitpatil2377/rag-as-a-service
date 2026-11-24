"""Coverage-aware generation — the direct completeness fix (pure logic, no I/O).

After the model produces a grounded answer, retrieval may have surfaced a distinct,
clearly-relevant point that the answer simply OMITTED — the generation-fidelity gap seen
live (retrieval had all six points, the model listed four). :func:`uncovered_spans` detects
those omitted-but-relevant spans so the generation service can do ONE targeted re-prompt to
add them back. This module makes ZERO decisions about whether to re-prompt or whether to keep
the result — it only reports which relevant spans the answer does not yet reflect.

Two guards against injecting noise (mirrored from the design):
* ONLY high-score spans count — a span the reranker scored below ``min_score`` is not clearly
  relevant, so an answer that omitted it omitted it correctly and must be left alone.
* a span already partially reflected in the answer is considered covered (``coverage_ratio``),
  so a paraphrased point is never mistaken for a dropped one.
"""

from __future__ import annotations

from rag_core.grounding import _content_terms
from rag_schemas import RetrievedSpan


def uncovered_spans(
    answer_text: str,
    spans: list[RetrievedSpan],
    *,
    min_score: float,
    coverage_ratio: float,
    max_spans: int,
) -> list[RetrievedSpan]:
    """The clearly-relevant retrieved spans whose point is largely ABSENT from ``answer_text``.

    For each span the reranker scored ``>= min_score`` (i.e. clearly relevant), measure the
    fraction of its salient :func:`~rag_core.grounding._content_terms` that appear in the
    answer's own content terms. When that fraction is below ``coverage_ratio`` the span's point
    is largely missing → "uncovered". The result is highest-score-first and capped at
    ``max_spans``; an EMPTY list means the answer already reflects the relevant evidence and no
    re-prompt is warranted.
    """
    answer_terms = _content_terms(answer_text)
    scored: list[tuple[float, RetrievedSpan]] = []
    for span in spans:
        # Only clearly-relevant spans are "expected". A low-score span the model omitted was
        # very likely omitted correctly; forcing it back in would inject noise.
        if span.score < min_score:
            continue
        span_terms = _content_terms(span.text)
        if not span_terms:
            continue  # nothing salient to be missing
        covered = len(span_terms & answer_terms) / len(span_terms)
        if covered < coverage_ratio:
            scored.append((span.score, span))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [span for _, span in scored[:max_spans]]
