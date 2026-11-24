"""Agentic query planning — pure logic, no LLM/client imports, no IO.

The STATIC intent classifier (:mod:`rag_core.intent`) keys on tokens and a singular/plural
rule to pick ONE label. It cannot decide *scope* ("read the whole document" vs "find the passage")
because that is not a property of the words — "summarise the key obligations in this document" and
"what is the notice period in this document" share every catalog token, yet one wants exhaustive
coverage of one file and the other wants a single fact from it.

A query PLAN is the richer decision the pipeline routes on: a :class:`QueryPlan` names the answer
STRATEGY, whether coverage must be ``exhaustive`` or ``targeted``, the target document (when the
question is about one), the answer SHAPE, a one-line rationale, and a confidence. The pipeline then
selects its answer mechanism FROM the plan, rather than the plan being a single hard-coded label.

Two backends behind one seam (mirroring ``routing.py`` / ``intent.py``):

* :class:`HeuristicQueryPlanner` — the deterministic FALLBACK and the default backend. It wraps the
  EXISTING :func:`~rag_core.intent.classify_intent_heuristic` and maps its label onto a plan,
  so with the planner flag OFF the pipeline's routing is UNCHANGED (byte-identical).
* The real semantic path — an LLM prompted by :func:`build_query_plan_prompt` and parsed by
  :func:`parse_query_plan` (the adapter lives in ``rag_clients``). :func:`parse_query_plan` is
  robust by construction: ANY malformed field returns the caller's ``fallback`` plan, so a model
  error can never mis-route or crash a query — worst case is today's heuristic behaviour.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Protocol, get_args, runtime_checkable

from rag_core.intent import Intent, classify_intent_heuristic
from rag_core.json_reply import extract_json_object

Strategy = Literal["whole_document", "passages", "whole_kb", "catalog", "count", "filter", "compare"]
Coverage = Literal["exhaustive", "targeted"]
AnswerShape = Literal["summary", "direct", "list", "comparison"]

_STRATEGIES: frozenset[str] = frozenset(get_args(Strategy))
_COVERAGES: frozenset[str] = frozenset(get_args(Coverage))
_ANSWER_SHAPES: frozenset[str] = frozenset(get_args(AnswerShape))


@dataclass(frozen=True)
class QueryPlan:
    """The structured routing decision the pipeline selects an answer mechanism from.

    ``strategy`` names the mechanism, ``coverage`` says whether the answer must reflect the whole
    unit (``exhaustive``) or a targeted slice (``targeted``), ``target_doc`` is the document the
    question is about (when it is about one; else None), ``answer_shape`` the presentation, and
    ``rationale``/``confidence`` are non-authoritative explainability carried onto the audit trail.
    """

    strategy: Strategy
    coverage: Coverage
    target_doc: str | None
    answer_shape: AnswerShape
    rationale: str
    confidence: float


@dataclass(frozen=True)
class PlanContext:
    """The light KB context the planner reads alongside the query — never the full corpus.

    ``doc_titles`` is a BOUNDED sample of caller-visible document titles (or the referenced ones),
    so the planner can resolve "this document" / a named title without loading every chunk.
    """

    doc_count: int
    doc_titles: tuple[str, ...]
    has_history: bool


@runtime_checkable
class QueryPlanner(Protocol):
    async def plan(self, query: str, ctx: PlanContext) -> QueryPlan: ...


# Deterministic label -> plan mapping. The heuristic classifier answers "which meta-shape", and the
# plan adds the scope the classifier cannot express: doc_summary (summarise EVERY doc) is a whole-KB
# exhaustive request; a plain rag/summarize question is a targeted passage lookup. This is the safe
# fallback, so it never chooses whole_document (a single-target exhaustive read) — the LLM path,
# which actually understands "this document", is what elects that.
_INTENT_TO_PLAN: dict[Intent, tuple[Strategy, Coverage, AnswerShape]] = {
    "catalog": ("catalog", "targeted", "list"),
    "count": ("count", "targeted", "direct"),
    "filter": ("filter", "targeted", "list"),
    "compare": ("compare", "targeted", "comparison"),
    "doc_summary": ("whole_kb", "exhaustive", "summary"),
    "summarize": ("passages", "targeted", "direct"),
    "rag": ("passages", "targeted", "direct"),
}


def plan_from_intent(intent: Intent, *, confidence: float = 0.5) -> QueryPlan:
    """Map a heuristic :data:`~rag_core.intent.Intent` label onto a :class:`QueryPlan`."""
    strategy, coverage, shape = _INTENT_TO_PLAN.get(intent, ("passages", "targeted", "direct"))
    return QueryPlan(
        strategy=strategy,
        coverage=coverage,
        target_doc=None,
        answer_shape=shape,
        rationale=f"heuristic intent classifier → {intent}",
        confidence=confidence,
    )


class HeuristicQueryPlanner:
    """Deterministic, model-free planner (the default + fallback). Wraps the keyword classifier."""

    async def plan(self, query: str, ctx: PlanContext) -> QueryPlan:
        # ctx is unused by the heuristic (it keys on the query alone) but is part of the Protocol.
        del ctx
        return plan_from_intent(classify_intent_heuristic(query))


def build_query_plan_prompt(query: str, ctx: PlanContext) -> str:
    """Prompt the model to read the WHOLE query and emit a strict-JSON execution plan.

    The model decides scope/coverage/target/shape with a one-line rationale — the decision the
    keyword classifier cannot make. Emphasises the three scope calls the pipeline needs to get
    right: comprehensive coverage of ONE named/"this" document → ``whole_document`` + ``exhaustive``;
    a pointed fact → ``passages`` + ``targeted``; a broad "across everything" → ``whole_kb``.
    """
    titles = "\n".join(f"- {t}" for t in ctx.doc_titles) if ctx.doc_titles else "(none listed)"
    history_note = (
        "There IS a prior conversation, so a bare 'summarise/recap' may mean the conversation.\n"
        if ctx.has_history
        else ""
    )
    return (
        "You are the QUERY PLANNER for a retrieval-augmented knowledge base. Read the ENTIRE "
        "question and decide how it should be answered. Reply with STRICT JSON and nothing else.\n\n"
        "Return an object with these fields:\n"
        '- "strategy": one of "whole_document", "passages", "whole_kb", "catalog", "count", '
        '"filter", "compare".\n'
        '- "coverage": "exhaustive" (the answer must reflect the WHOLE unit) or "targeted" (a '
        "specific slice is enough).\n"
        '- "target_doc": the exact title of the ONE document the question is about (from the list '
        "below), or null when it is not about a single document.\n"
        '- "answer_shape": "summary", "direct", "list", or "comparison".\n'
        '- "rationale": one short sentence explaining the choice.\n'
        '- "confidence": a number from 0 to 1.\n\n'
        "How to choose the strategy:\n"
        '- "whole_document" + "exhaustive": the reader wants COMPREHENSIVE coverage of ONE document '
        '— every obligation/point/section of a named document or of "this document". '
        'Example: "Summarise the key obligations in this document", "list all requirements in '
        'the policy".\n'
        '- "passages" + "targeted": a POINTED fact, definition, or short answer found in a passage '
        'or two. Example: "what is the notice period?", "who approved the budget?".\n'
        '- "whole_kb": the question ranges ACROSS EVERYTHING in the knowledge base '
        '(all documents / KB-wide themes). Example: "what are the main themes across all documents?".\n'
        '- "catalog": the LIST/inventory of documents (their names). "count": HOW MANY documents. '
        '"filter": the documents matching a criterion (year/type/recency). "compare": two NAMED '
        "documents contrasted.\n\n"
        f"{history_note}"
        f"The knowledge base holds {ctx.doc_count} document(s). Sample of visible titles:\n{titles}\n\n"
        f"QUESTION: {query.strip()}\n\nJSON:"
    )


def _coerce_float(value: object, default: float) -> float:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, f))


def parse_query_plan(reply: str, *, fallback: QueryPlan) -> QueryPlan:
    """Parse a model reply into a :class:`QueryPlan`, returning ``fallback`` on ANY malformed field.

    Robust by construction: strips code fences, isolates the first JSON object, tolerates junk, and
    validates every enum. A missing/invalid ``strategy``/``coverage``/``answer_shape`` — or any
    parse failure — degrades to ``fallback`` so a model error never mis-routes or crashes a query.
    """
    blob = extract_json_object(reply)
    if blob is None:
        return fallback
    try:
        data = json.loads(blob)
    except (ValueError, TypeError):
        return fallback
    if not isinstance(data, dict):
        return fallback

    strategy = data.get("strategy")
    coverage = data.get("coverage")
    shape = data.get("answer_shape")
    if strategy not in _STRATEGIES or coverage not in _COVERAGES or shape not in _ANSWER_SHAPES:
        return fallback

    raw_target = data.get("target_doc")
    target = raw_target.strip() if isinstance(raw_target, str) and raw_target.strip() else None

    rationale = data.get("rationale")
    rationale_str = rationale.strip() if isinstance(rationale, str) and rationale.strip() else "query planner"

    return QueryPlan(
        strategy=strategy,
        coverage=coverage,
        target_doc=target,
        answer_shape=shape,
        rationale=rationale_str[:280],
        confidence=_coerce_float(data.get("confidence"), fallback.confidence),
    )
