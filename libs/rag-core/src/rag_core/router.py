"""The query router — RAG v2 Phase 7 (pure logic; the gateway call is in rag_clients).

One decision per question, read off the PROFILE CARDS of the in-scope documents rather than off
the query's words alone: which of the plan's six intents this is, which documents it is about,
which of those documents' OWN facets it touches, the facet-specific sub-queries an aggregate
needs, the query and answer languages, and whether the answer must take the amendment view.

WHY THE CARDS ARE THE INPUT, and why this could not be built before Phase 3. The existing routers
read the query and, at most, document TITLES (``query_plan.PlanContext``). A title cannot say
that a document is a corrigendum, or that "financial qualification" is one of its content areas —
so "what are the financial requirements" could be matched to a document only by its filename
happening to contain the word. The Phase-3 card carries ``doc_type``, ``purpose`` and a facet
taxonomy drawn from the document's own content, which is what makes "map query words onto each
document's own facets" (the plan's §7 wording) a thing a router can actually do.

THIS EXTENDS THE EXISTING TAXONOMY, it does not replace it — discovery §8.4, decided before any
of it was built. :func:`plan_for_intent` maps the plan's six intents onto the
:class:`~rag_core.query_plan.QueryPlan` the pipeline already routes on, so a v2 decision
selects an EXISTING answer mechanism and the fall-through-to-RAG fail-safe is preserved. Nothing
here introduces a second routing vocabulary for the pipeline to disagree with itself over.

TWO BACKENDS BEHIND ONE SEAM, heuristic default, mirroring every other such seam in this codebase
(``routing.py``, ``intent.py``, ``query_plan.py``, graph extraction, contextualisation):

* :func:`heuristic_decision` — deterministic, model-free, and therefore MEASURABLE. It is the
  default, the fallback, and the only path whose intent accuracy can be asserted in a unit test
  against a labelled gold set, which matters because the plan's acceptance is an accuracy figure.
* :func:`build_router_prompt` + :func:`parse_router_decision` — the plan's own prompt and a strict
  parser. The adapter lives in ``rag_clients.query_router_v2``.

VALIDATION IS AGAINST THE SCOPE, not against a schema alone. A ``doc_id`` the scope does not
contain and a ``facet_id`` that document's card does not define are DROPPED and reported, for the
same reason Phase 3 drops a relation whose quote is not in the document: a router decision is an
instruction to later phases, and an instruction naming a document this question cannot see would
scope retrieval to nothing, while an invented facet would group an aggregate answer by a label
that joins to no ``facets`` row.

NEVER MIS-ROUTES. Any unusable field falls back to ``lookup`` with everything else empty — which
is the plan's own stated fallback ("if router output fails validation, default to lookup and log")
and is also, not coincidentally, the behaviour of the pipeline with the router switched off.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal, get_args

from rag_core.doc_profile import snake_id
from rag_core.json_reply import extract_json_object
from rag_core.prompts import render_prompt
from rag_core.query_plan import AnswerShape, Coverage, QueryPlan, Strategy

__all__ = [
    "ROUTER_INTENTS",
    "ROUTER_VERSION",
    "DocCard",
    "RouterDecision",
    "RouterIntent",
    "RouterParse",
    "RouterScope",
    "build_router_prompt",
    "classify_router_intent",
    "heuristic_decision",
    "parse_router_decision",
    "plan_for_intent",
]

RouterIntent = Literal["lookup", "aggregate", "compare", "summarize", "procedural", "relational"]

# The plan's §7 intent contract. THE CANONICAL DEFINITION — ``svc_guardrails_eval.suite`` imports
# it so a Phase-11 gold set cannot label a query with an intent the router does not emit, and so
# adding a seventh intent cannot leave a gold file validating against the old six.
ROUTER_INTENTS: frozenset[str] = frozenset(get_args(RouterIntent))

# The version of the ROUTING CONTRACT — this prompt, this parser, this intent set — stored beside
# each decision in ``query_log.router_json``. Same role as ``doc_profile.PROFILE_VERSION``: it
# answers "which stored decisions did a build this one would no longer produce".
ROUTER_VERSION = 1

# Bounds. Guards against a model reply, not tuning knobs: everything here lands in a Postgres
# JSONB column and decides what retrieval does next.
_MAX_DOC_IDS = 20
_MAX_FACETS = 40
_MAX_SUB_QUERIES = 8
_MIN_SUB_QUERIES = 2  # the plan's "2-8"; one sub-query is the query again
_MAX_SUB_QUERY_CHARS = 300
_MAX_RATIONALE_CHARS = 300
_MAX_LANG_CHARS = 8


@dataclass(frozen=True, slots=True)
class DocCard:
    """One in-scope document, as the router sees it: its Phase-3 card, flattened.

    Deliberately NOT the whole ``DocProfile``. The router is shown what it needs to choose
    between documents — what kind of document this is, why it exists, and what it is about — and
    not the relations or the evidence quotes, which are Phase 8's and Phase 9's inputs and would
    crowd the prompt with text no routing decision reads."""

    doc_id: str
    title: str
    doc_type: str = ""
    purpose: str = ""
    facet_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RouterScope:
    """The documents this question may be about (the plan's "profile cards of in-scope documents").

    ``explicit`` records HOW the scope was chosen: True when the caller named documents, False
    when it fell back to a sample of the KB. It is carried because the two are different evidence
    — a decision made over a named document is about that document, while one made over a sample
    is as good as the sample — and a later phase reading a stored decision needs to know which."""

    cards: tuple[DocCard, ...] = ()
    explicit: bool = False

    def card(self, doc_id: str) -> DocCard | None:
        return next((c for c in self.cards if c.doc_id == doc_id), None)


@dataclass(frozen=True, slots=True)
class RouterDecision:
    """The validated routing decision (plan §7).

    Every collection is empty rather than absent when the router has nothing to say, so a consumer
    never has to distinguish "the router did not run" from "the router found nothing" — that
    distinction lives in whether a decision exists at all."""

    intent: RouterIntent = "lookup"
    doc_ids: tuple[str, ...] = ()
    facets: tuple[tuple[str, str], ...] = ()  # (doc_id, facet_id)
    sub_queries: tuple[str, ...] = ()
    query_lang: str = ""
    answer_lang: str = ""
    needs_amendment_view: bool = False
    rationale: str = ""
    confidence: float = 0.5
    router_version: int = ROUTER_VERSION

    def as_json(self) -> str:
        """The shape stored in ``query_log.router_json`` (plan §1.1).

        Phase 12 gave that column its writer and left it NULL with a ``TODO(Phase-7)``; this is
        what fills it."""
        return json.dumps(
            {
                "router_version": self.router_version,
                "intent": self.intent,
                "doc_ids": list(self.doc_ids),
                "facets": [{"doc_id": d, "facet_id": f} for d, f in self.facets],
                "sub_queries": list(self.sub_queries),
                "query_lang": self.query_lang,
                "answer_lang": self.answer_lang,
                "needs_amendment_view": self.needs_amendment_view,
                "rationale": self.rationale,
                "confidence": round(self.confidence, 3),
            },
            ensure_ascii=False,
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class RouterParse:
    """What :func:`parse_router_decision` made of a reply, INCLUDING what it threw away.

    Same discipline as :class:`~rag_core.doc_profile.ProfileParse`: a dropped field is
    reported, never silently discarded. ``fell_back`` is the one that matters most — a router that
    quietly degrades to ``lookup`` on every question looks exactly like a router that is working
    and happens to see only lookups."""

    decision: RouterDecision
    fell_back: bool = False
    error: str = ""
    unknown_docs: tuple[str, ...] = ()
    unknown_facets: tuple[str, ...] = ()
    dropped_sub_queries: int = 0


# --- mapping onto the existing taxonomy (discovery §8.4) -----------------------------------------

# The plan's six intents onto the strategy the pipeline ALREADY routes on. This is the whole of
# §8.4's "map the plan's six intents onto the existing strategy taxonomy and keep the
# fall-through-to-RAG fail-safe": a v2 decision selects an existing mechanism, so the router can
# ship before Phases 8-9 exist without the pipeline needing a second vocabulary.
#
# `aggregate` maps to `whole_document`/exhaustive, which is the nearest existing mechanism and NOT
# the plan's aggregate strategy — that one enumerates section-level units and reranks them against
# sub-queries, and has no implementation (Phase 8). The mapping is honest about being the closest
# available answer rather than the right one; see discovery §8.24.
_INTENT_TO_PLAN: dict[str, tuple[Strategy, Coverage, AnswerShape]] = {
    "lookup": ("passages", "targeted", "direct"),
    "aggregate": ("whole_document", "exhaustive", "list"),
    "compare": ("compare", "targeted", "comparison"),
    "summarize": ("whole_document", "exhaustive", "summary"),
    "procedural": ("passages", "targeted", "list"),
    "relational": ("passages", "targeted", "direct"),
}


def plan_for_intent(intent: str, decision: RouterDecision | None = None) -> QueryPlan:
    """Map a router intent onto the :class:`QueryPlan` the pipeline selects a mechanism from.

    ``target_doc`` comes from the decision's single ``doc_ids`` entry when there is exactly one: a
    ``whole_document`` strategy needs to know WHICH document, and two candidates mean the router
    did not narrow it to one, in which case the existing passage path is the honest mechanism."""
    strategy, coverage, shape = _INTENT_TO_PLAN.get(intent, ("passages", "targeted", "direct"))
    target = decision.doc_ids[0] if decision and len(decision.doc_ids) == 1 else None
    if strategy == "whole_document" and target is None:
        strategy, coverage = "passages", "targeted"
    return QueryPlan(
        strategy=strategy,
        coverage=coverage,
        target_doc=target,
        answer_shape=shape,
        rationale=f"rag_v2 router → {intent}",
        confidence=decision.confidence if decision else 0.5,
    )


# --- the heuristic backend -----------------------------------------------------------------------
#
# Cue tables, narrow on purpose and ORDERED on purpose — the order is the classifier. Each table
# holds the phrases the plan's own definition names, not a general vocabulary for the concept:
# "relational = links between documents/entities", so the relational cues are link words, not
# every word that can relate two things.

# "which documents amend this", "what supersedes it", "related to", "cross-referenced in".
#
# A REGEX, not a substring table, and the two misses that forced it are worth keeping: "amend" as
# a substring matched "the amendment", turning "tl;dr of the amendment" — a summarize question
# whose TOPIC happens to be an amendment — into a relational one; and "refers to" missed "which
# clauses does this corrigendum refer to", because the verb was not inflected the way the cue
# was. Word boundaries fix the first and an inflection group the second. The other tables stay
# substring matches because their cues are phrases ("difference between", "walk me through")
# where a bare word cannot collide the same way.
_RELATIONAL_RE = re.compile(
    r"\b(?:amends?|amended|amending|supersedes?|superseded|supersession)\b"
    r"|\brefers?\s+to\b|\breferred\s+to\s+in\b"
    r"|\brelated\s+to\b|\brelates?\s+to\b|\blinked\s+to\b"
    r"|\bcross[- ]reference|\battach(?:ed|ment)\s+to\b"
    r"|\brelationships?\s+between\b|\bwhich\s+documents?\b|\bwhat\s+other\s+document"
)
# "compare", "versus", "difference between". Checked AFTER relational so "the relationship between
# A and B" is not read as a comparison of them.
_COMPARE = (
    "compare", "comparison", "compared to", "compared with", " vs ", " vs. ", "versus",
    "difference between", "differences between", "differ from", "differs from", "better than",
    "which is better", "which of the two",
)  # fmt: skip
# "how do I", "the steps", "the procedure". Checked BEFORE aggregate so "what are the steps to X"
# is procedural rather than a list request — the plan defines procedural as steps, and a question
# about steps wants them in order, which is a different answer shape from an exhaustive list.
_PROCEDURAL = (
    "how do i", "how do we", "how to ", "how is ", "how are ", "step by step", "the steps",
    "what steps", "which steps", "procedure for", "process for", "process of", "the procedure",
    "the process", "walk me through", "in what order", "sequence of",
)  # fmt: skip
# "all", "every", "list", "details of", "requirements". The plan: aggregate = "all/list/details/
# requirements of X" across the document.
_AGGREGATE = (
    "all the", "all of the", "every ", "each of the", "list the", "list of", "list all",
    "give me the", "full list", "complete list", "enumerate", "details of", "the details",
    "requirements", "obligations", "specifications", "what are the", "which are the",
    "everything ", "comprehensive",
)  # fmt: skip
# A PLURAL SUBJECT expects more than one answer, which is an aggregate however the question is
# phrased. "Which designations are responsible for preparing, approving and authorising an SOP?"
# carries no all/list/every cue and is unmistakably a three-item question — a disagreement the
# Phase-11 gold set surfaced, having been labelled before this classifier existed. Narrow on
# purpose: a plural noun AND a plural verb, and procedural is still checked first so "what steps
# are required" keeps its order-sensitive answer.
_AGGREGATE_RE = re.compile(r"\b(?:which|what)\s+\w{3,}s\s+(?:are|were|must|shall|should|can|do)\b")
# "summarise", "overview", "gist". Checked LAST before the default so "summarise all the
# obligations" is aggregate (it wants exhaustive coverage) rather than an overview.
_SUMMARIZE = (
    "summar", "overview", "in brief", "briefly", "the gist", "tl;dr", "high level", "high-level",
    "what is this document about", "what does this document cover",
)  # fmt: skip
# Words the amendment view is about. A question that says "current"/"in force"/"latest" is asking
# what applies NOW, which is the one case where an answer quoting a superseded clause is wrong.
# Substring matching is RIGHT here, unlike for the relational cues: "the amendment" as a topic is
# exactly a question where the amendment view applies, so the looseness that was wrong for intent
# is correct for this flag.
_AMENDMENT = (
    "amend", "supersede", "corrigendum", "revised", "revision", "current", "currently",
    "in force", "latest", "as amended", "up to date", "up-to-date", "still applicable",
)  # fmt: skip

_WORD_RE = re.compile(r"[a-z0-9]+")
# Below this a facet's token is too generic to match a query on ("the", "of", "and" never reach
# here, but "data" and "year" would).
_MIN_FACET_TOKEN = 4
# The share of a facet's own tokens a query must use for the facet to count as touched. AT LEAST
# HALF, which for the common two-token facet means EITHER of its words — "financial requirements"
# matches `financial_qualification`. That is deliberately loose and the looseness is the point: a
# real query almost never restates a full snake_case id, so a strict-majority rule would mean the
# matcher effectively never fires. The asymmetry of costs supports it — a loose match adds one
# facet-specific sub-query, which biases retrieval toward a plausibly relevant area, while a
# strict one silently removes the only deterministic facet signal the router has.
_FACET_MIN_RATIO = 0.5


def classify_router_intent(query: str) -> RouterIntent:
    """Deterministic intent, model-free — the default backend and the fallback.

    ORDER IS THE CLASSIFIER, and each step of it is a reading of the plan's own definitions:
    relational before compare (a "relationship between A and B" is a link question, not a
    comparison); procedural before aggregate ("what are the steps" wants an order, not a list);
    aggregate before summarize ("summarise all the obligations" wants exhaustive coverage, which
    is what aggregate means and what an overview is not); anything unmatched is ``lookup``, which
    never mis-routes because it is the behaviour of the pipeline with the router off.
    """
    text = f" {query.lower().strip()} "
    if _RELATIONAL_RE.search(text):
        return "relational"
    if any(cue in text for cue in _COMPARE):
        return "compare"
    if any(cue in text for cue in _PROCEDURAL):
        return "procedural"
    if any(cue in text for cue in _AGGREGATE) or _AGGREGATE_RE.search(text):
        return "aggregate"
    if any(cue in text for cue in _SUMMARIZE):
        return "summarize"
    return "lookup"


def _query_tokens(query: str) -> set[str]:
    return set(_WORD_RE.findall(query.lower()))


def match_facets(query: str, scope: RouterScope) -> tuple[tuple[str, str], ...]:
    """Facets of the in-scope documents whose own words the query uses.

    The deterministic half of the plan's "map query words onto each document's own facets". A
    facet id is snake_case drawn from the document's content (Phase 3), so its tokens ARE the
    document's vocabulary for that content area, and a query using them is a query about it.

    A RATIO over the facet's own tokens, not a whole-id match — see :data:`_FACET_MIN_RATIO` for
    why it is deliberately loose. Tokens shorter than :data:`_MIN_FACET_TOKEN` are ignored
    entirely: they are the words every document shares, and a facet matching on "the" or "data"
    would fire on every question."""
    words = _query_tokens(query)
    out: list[tuple[str, str]] = []
    for card in scope.cards:
        for facet in card.facet_ids:
            tokens = [t for t in facet.split("_") if len(t) >= _MIN_FACET_TOKEN]
            if not tokens:
                continue
            hits = sum(1 for t in tokens if t in words)
            if hits / len(tokens) >= _FACET_MIN_RATIO:
                out.append((card.doc_id, facet))
    return tuple(out[:_MAX_FACETS])


def heuristic_decision(query: str, scope: RouterScope, *, query_lang: str = "") -> RouterDecision:
    """A full decision with no model call. The default backend and the LLM path's fallback.

    ``doc_ids`` is the scope's own ids ONLY when the caller named them. A sample of the KB is not a
    routing decision about those documents, and echoing it back as ``doc_ids`` would scope
    retrieval to an arbitrary sample — the one way this path could be worse than no router at all.

    ``sub_queries`` are facet-specific and therefore only produced where facets matched and the
    intent is one the plan says they are for. A single match yields none: one sub-query is the
    query again.

    EACH SUB-QUERY IS A QUESTION ABOUT THE FACET, not the original query with the facet's words
    appended — which is what this built until Phase 8 measured it. Appending made the string
    LESS discriminating than the bare query: scored against the real AEGCL corrigendum's section
    cards on this deployment's BGE cross-encoder, "give me the compliance details from this
    document certifications" separated the three certification-bearing sections from the
    letterhead by 4.6x, while "What are the certifications requirements?" separated them by
    infinity (the noise scored exactly 0.0) and scored 740x higher in absolute terms. The reason
    is visible in the strings: appending keeps every vague word of the original and adds one, so
    the model still reads a question that is not about anything in particular.
    """
    intent = classify_router_intent(query)
    facets = match_facets(query, scope)
    sub_queries: tuple[str, ...] = ()
    if intent in {"aggregate", "compare"} and len(facets) >= _MIN_SUB_QUERIES:
        seen: dict[str, None] = {}
        for _doc_id, facet in facets:
            seen.setdefault(f"What are the {facet.replace('_', ' ')} requirements?", None)
        sub_queries = tuple(list(seen)[:_MAX_SUB_QUERIES])
    return RouterDecision(
        intent=intent,
        doc_ids=tuple(c.doc_id for c in scope.cards) if scope.explicit else (),
        facets=facets,
        sub_queries=sub_queries,
        query_lang=query_lang,
        answer_lang=query_lang,
        needs_amendment_view=any(cue in query.lower() for cue in _AMENDMENT),
        rationale=f"heuristic cue tables → {intent}",
        confidence=0.5,
    )


# --- the LLM backend -----------------------------------------------------------------------------


def build_router_prompt(query: str, scope: RouterScope) -> str:
    """Render ``prompts/router.md`` with the query and the in-scope documents' cards (plan §7).

    EVERY STRING MUST ALREADY BE MASKED. The cards come from ``documents.profile_json``, which
    Phase 3 built from masked text and masked the title of; the query is the user's own. What
    comes back is stored in ``query_log.router_json``.

    A scope with no cards still renders, and says so rather than rendering an empty section: a KB
    whose documents have no profile card yet is the ORDINARY state (the card is written only on
    the v2 ingest path), and a model shown a blank heading reads it as a missing input."""
    listing = "\n\n".join(
        "\n".join(
            part
            for part in (
                f"- doc_id: {card.doc_id}",
                f"  title: {card.title}" if card.title else "",
                f"  doc_type: {card.doc_type}" if card.doc_type else "",
                f"  purpose: {card.purpose}" if card.purpose else "",
                f"  facets: {', '.join(card.facet_ids)}" if card.facet_ids else "  facets: (none)",
            )
            if part
        )
        for card in scope.cards
    )
    return render_prompt(
        "router",
        profiles=listing or "(no document in scope has a profile card yet)",
        query=query.strip(),
    )


def parse_router_decision(
    reply: str, *, scope: RouterScope, query_lang: str = "", fallback: RouterDecision | None = None
) -> RouterParse:
    """Validate a model reply into a :class:`RouterDecision`, or fall back to ``lookup``.

    ``fallback`` is what an unusable reply yields — the caller passes its heuristic decision, so a
    model failure degrades to the default backend rather than to nothing. With no fallback given,
    a bare ``lookup`` decision, which is the pipeline's behaviour with the router off.

    ``query_lang`` is the CALLER's detection and it WINS over the model's, for the same reason
    ``doc_profile.parse_doc_profile`` overrides the reply's ``lang``: the platform answers this
    question deterministically in ISO-639-1 and a measured real-model run returned the prose word
    "english" instead. ``answer_lang`` is taken from the reply only when it is a plausible code,
    because it is the one language field a user can legitimately ask to differ from the query's.

    Never raises.
    """
    safe = fallback or RouterDecision(query_lang=query_lang, answer_lang=query_lang)
    blob = extract_json_object(reply)
    if blob is None:
        return RouterParse(decision=safe, fell_back=True, error="no JSON object in the reply")
    try:
        data = json.loads(blob)
    except (ValueError, TypeError) as exc:
        return RouterParse(decision=safe, fell_back=True, error=f"malformed JSON: {str(exc)[:120]}")
    if not isinstance(data, dict):
        return RouterParse(decision=safe, fell_back=True, error="the JSON value is not an object")

    intent = str(data.get("intent") or "").strip().lower()
    if intent not in ROUTER_INTENTS:
        # Reported rather than coerced to the nearest label. A guessed intent selects a different
        # answer mechanism, and the plan's own fallback for a failed validation is `lookup`.
        return RouterParse(decision=safe, fell_back=True, error=f"intent {intent or '(blank)'!r} is not one of the six")

    doc_ids, unknown_docs = _doc_ids(data.get("doc_ids"), scope)
    facets, unknown_facets = _facets(data.get("facets"), scope)
    sub_queries, dropped = _sub_queries(data.get("sub_queries"), intent)
    answer_lang = _lang(data.get("answer_lang")) or query_lang
    return RouterParse(
        decision=RouterDecision(
            intent=intent,  # type: ignore[arg-type]  # membership checked against ROUTER_INTENTS
            doc_ids=doc_ids,
            facets=facets,
            sub_queries=sub_queries,
            query_lang=query_lang,
            answer_lang=answer_lang,
            needs_amendment_view=bool(data.get("needs_amendment_view", False)),
            rationale=_text(data.get("rationale"), _MAX_RATIONALE_CHARS),
            confidence=_confidence(data.get("confidence")),
        ),
        unknown_docs=unknown_docs,
        unknown_facets=unknown_facets,
        dropped_sub_queries=dropped,
    )


def _doc_ids(value: object, scope: RouterScope) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """In-scope ids, plus the ones the scope does not contain.

    An out-of-scope id is dropped, not kept: ``doc_ids`` scopes retrieval, and scoping it to a
    document this question cannot see retrieves nothing — a confident, cited, empty answer."""
    if not isinstance(value, list):
        return (), ()
    known = {c.doc_id for c in scope.cards}
    kept: list[str] = []
    unknown: list[str] = []
    for raw in value[:_MAX_DOC_IDS]:
        doc_id = str(raw).strip()
        if not doc_id or doc_id in kept:
            continue
        (kept if doc_id in known else unknown).append(doc_id)
    return tuple(kept), tuple(unknown)


def _facets(value: object, scope: RouterScope) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    """(doc_id, facet_id) pairs the cards actually define, plus the ones they do not.

    An invented facet would land in a decision that Phase 8 groups an aggregate answer by, with
    nothing in ``facets`` to name it — the same reason Phase 4 drops a label outside the
    document's taxonomy."""
    if not isinstance(value, list):
        return (), ()
    kept: list[tuple[str, str]] = []
    unknown: list[str] = []
    for raw in value[:_MAX_FACETS]:
        if not isinstance(raw, dict):
            continue
        doc_id = str(raw.get("doc_id") or "").strip()
        facet_id = snake_id(str(raw.get("facet_id") or ""))
        if not doc_id or not facet_id:
            continue
        card = scope.card(doc_id)
        if card is not None and facet_id in card.facet_ids:
            if (doc_id, facet_id) not in kept:
                kept.append((doc_id, facet_id))
        else:
            unknown.append(f"{doc_id}:{facet_id}")
    return tuple(kept), tuple(unknown)


def _sub_queries(value: object, intent: str) -> tuple[tuple[str, ...], int]:
    """2-8 rewrites, and how many were dropped.

    DROPPED ENTIRELY for an intent the plan does not give them to. They drive decomposition, and
    decomposing a single-fact lookup into sub-queries multiplies the retrieval cost of the
    commonest question shape for no gain. Fewer than two after validation are dropped for the
    plan's own reason: one sub-query is the query again."""
    if not isinstance(value, list):
        return (), 0
    offered = len(value)
    if intent not in {"aggregate", "compare"}:
        return (), offered
    seen: dict[str, None] = {}
    for raw in value:
        text = _text(raw, _MAX_SUB_QUERY_CHARS)
        if text:
            seen.setdefault(text, None)
    kept = list(seen)[:_MAX_SUB_QUERIES]
    if len(kept) < _MIN_SUB_QUERIES:
        return (), offered
    return tuple(kept), offered - len(kept)


_LANG_RE = re.compile(r"^[a-z]{2,3}(?:[-_][a-z]{2,4})?$")


def _lang(value: object) -> str:
    """An ISO-639-1/2 code, or "". A model that answers "english" is answering a different
    question — see :func:`parse_router_decision`."""
    code = _text(value, _MAX_LANG_CHARS).lower().replace("_", "-")
    return code if _LANG_RE.match(code) else ""


def _text(value: object, limit: int) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def _confidence(value: object) -> float:
    try:
        return min(1.0, max(0.0, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.5
