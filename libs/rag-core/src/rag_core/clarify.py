"""Grounded query disambiguation — "Did you mean X / Y / Z?" (Layer-2 item 4) — pure logic.

The structural sibling of the FR-I8 follow-ups seam (:mod:`followups`). When a query is
AMBIGUOUS — it matches SEVERAL distinct KB topics shallowly, or would abstain while the KB
DOES hold candidate topics — offer 2-4 concrete "Did you mean …?" options phrased ONLY from
the RETRIEVED spans (never the model's parametric knowledge, never an invented topic), so a
user who knows their intent but not the exact term can pick one.

Options are GROUNDED-ONLY and NON-AUTHORITATIVE: they never change the answer text, the
citations, the status, or the grounded-or-abstain invariant. Two backends behind one seam
(heuristic default / LLM opt-in), exactly like follow-ups:
* :class:`HeuristicClarifyGenerator` — clusters the retrieved spans by ``doc_id`` (one
  document = one candidate interpretation) and phrases one candidate question per DISTINCT
  cluster from that cluster's salient terms; deterministic, model-free, CI-safe;
* the LLM backend (in ``rag_clients``) prompts the KB's model over the same spans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rag_core.grounding import _content_terms, _span_quote
from rag_schemas import RetrievedSpan

_MAX_SOURCES = 8  # cap spans fed to the prompt / clustered — enough to cover the distinct topics
_TOPIC_TERMS = 3  # salient terms used to name a topic when a cluster has no section heading
_MIN_CLUSTERS = 2  # fewer than two DISTINCT topics is not ambiguous — never fabricate an option
_MIN_OPTION_LEN = 8  # a real option has some length + a question mark (mirrors parse_followups)


@runtime_checkable
class ClarifyGenerator(Protocol):
    async def suggest(self, query: str, spans: list[RetrievedSpan], count: int) -> list[str]: ...


@dataclass(frozen=True)
class SpanCluster:
    """One distinct retrieved topic: all spans of a single ``doc_id``, with its best score."""

    doc_id: str
    best_score: float
    text: str
    section: str | None


def cluster_spans(spans: list[RetrievedSpan]) -> list[SpanCluster]:
    """Group retrieved spans into DISTINCT topic clusters by ``doc_id`` (one document = one
    candidate interpretation), highest-scoring cluster first. Pure + deterministic."""
    order: list[str] = []
    grouped: dict[str, list[RetrievedSpan]] = {}
    for span in spans[:_MAX_SOURCES]:
        if span.doc_id not in grouped:
            grouped[span.doc_id] = []
            order.append(span.doc_id)
        grouped[span.doc_id].append(span)
    clusters: list[SpanCluster] = []
    for doc_id in order:
        group = grouped[doc_id]
        clusters.append(
            SpanCluster(
                doc_id=doc_id,
                best_score=max(s.score for s in group),
                text="\n\n".join(s.text for s in group),
                section=next((s.section for s in group if s.section), None),
            )
        )
    clusters.sort(key=lambda c: c.best_score, reverse=True)
    return clusters


def _cluster_topic(cluster: SpanCluster, asked: set[str]) -> str:
    """A short, grounded label for a cluster's topic: its section heading when the document
    numbers its sections, else its most salient content terms not already named in the query.
    Derived ONLY from the cluster's own text — never invented."""
    if cluster.section and cluster.section.strip():
        return cluster.section.strip()
    lower = cluster.text.lower()
    terms = _content_terms(cluster.text) - asked
    ranked = sorted(terms, key=lambda t: (-lower.count(t), t))
    return " ".join(ranked[:_TOPIC_TERMS])


def build_clarify_prompt(query: str, spans: list[RetrievedSpan], count: int) -> str:
    """Prompt the LLM backend for disambiguation options grounded strictly in the spans."""
    sources = "\n\n".join(f"[{i + 1}] {_span_quote(span, query)}" for i, span in enumerate(spans[:_MAX_SOURCES]))
    return (
        f'A user asked an AMBIGUOUS question: "{query.strip()}".\n'
        f"The SOURCES below span SEVERAL distinct topics. Using ONLY these SOURCES (never outside "
        f'knowledge), write {count} short "Did you mean …?" options, each naming a DISTINCT topic that '
        "actually appears in the SOURCES and that the user might have meant. Do NOT invent topics. "
        "Return one option per line, no numbering, no preamble.\n\n"
        f"SOURCES:\n{sources}\n\nOPTIONS:"
    )


def parse_clarify(raw: str, count: int) -> list[str]:
    """Extract clean, deduped option lines from a model reply (capped at ``count``)."""
    out: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        opt = line.strip().lstrip("-*0123456789. )").strip()
        if len(opt) < _MIN_OPTION_LEN or "?" not in opt:
            continue
        key = opt.lower()
        if key not in seen:
            seen.add(key)
            out.append(opt)
        if len(out) >= count:
            break
    return out


def heuristic_clarify(query: str, spans: list[RetrievedSpan], count: int) -> list[str]:
    """Model-free disambiguation: one grounded "Did you mean …?" option per DISTINCT retrieved
    doc cluster, phrased from that cluster's salient terms. Fewer than two clusters → no options
    (never fabricate a topic). Highest-scoring cluster first, capped at ``count``."""
    clusters = cluster_spans(spans)
    if len(clusters) < _MIN_CLUSTERS:
        return []
    asked = _content_terms(query)
    out: list[str] = []
    seen: set[str] = set()
    for cluster in clusters:
        topic = _cluster_topic(cluster, asked)
        key = topic.lower()
        if not topic or key in seen:
            continue
        seen.add(key)
        out.append(f"Did you mean: {topic}?")
        if len(out) >= count:
            break
    return out


class HeuristicClarifyGenerator:
    """Deterministic doc-cluster disambiguation (default; no model, CI-safe)."""

    async def suggest(self, query: str, spans: list[RetrievedSpan], count: int) -> list[str]:
        return heuristic_clarify(query, spans, count)
