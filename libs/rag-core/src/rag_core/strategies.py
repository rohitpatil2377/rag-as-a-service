"""Retrieval strategies — RAG v2 Phase 8 (pure logic; the stores and the reranker stay outside).

Phase 7 decides WHAT the question is. This decides WHAT TO FETCH for it, and the plan gives each
of the six intents its own recipe. The two halves of that live here:

* :func:`levels_for_intent` and :func:`plan_strategy` — the declarative part. Which retrieval
  GRAINS a strategy asks the hybrid primitive for (``RetrievalQuery.levels``, new in this phase),
  whether it runs a section pass, whether hits expand to their parent section, whether the
  expansion is made contiguous.
* :func:`select_sections` — the plan's §8 `aggregate` recipe, which is the one strategy that does
  not retrieve passages at all. It reads EVERY section of the in-scope documents out of the
  catalog and decides which of them the answer must be built from.

WHY AGGREGATE CANNOT BE A PASSAGE SEARCH, measured rather than assumed. The acceptance query is
"give me the compliance details from this document". Scored against this deployment's BGE
cross-encoder, that query returns ~1.2e-5 against an irrelevant section AND ~3.5e-5 against a
directly relevant one — the two are indistinguishable, because a vague whole-document question
has no passage it is "about". The SAME reranker, given the facet-specific sub-query "what are the
switch certifications?", scores the switches section 0.65 against 1.2e-5 for the canteen policy.

That measurement is the reason the plan's step 2 reranks each section against the main query AND
every sub-query and takes the MAX, and the reason step 4 unions in the router's facet matches
rather than trusting the scores alone. An aggregate strategy built on the main query's scores
would select nothing above any usable threshold. See :func:`max_scores`.

NOTHING HERE CALLS A MODEL OR A STORE. The scores arrive as a mapping, the cards arrive as a
sequence, and the caller (``svc_control_plane.pipeline``) owns the reranker and the catalog. That
is what makes the selection — the part the plan states an acceptance criterion against —
assertable in a unit test with no infrastructure.

SECTION TEXT NEVER APPEARS HERE. A :class:`SectionCard` carries the breadcrumb, the heading and
the Phase-4 SUMMARY, all of which were produced from masked text (``pipeline._section_enrichment``
masks every string that reaches the enrichment prompt). ``blocks.text`` and the text behind
``sections.text_hash`` are unmasked PII and are deliberately not on this path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from rag_schemas.common import ChunkLevel

__all__ = [
    "STRATEGY_VERSION",
    "SectionCard",
    "SectionSelection",
    "SelectedSection",
    "SelectionReason",
    "StrategyPlan",
    "contiguous_sections",
    "levels_for_intent",
    "max_scores",
    "parent_sections",
    "plan_strategy",
    "select_per_target",
    "select_sections",
    "taxonomy_of",
]

# The version of the SELECTION CONTRACT — these recipes, these levels, this cap rule. Stored
# beside a selection in ``query_log`` the same way ``ROUTER_VERSION`` is stored beside a decision:
# it answers "would this build still select these sections".
STRATEGY_VERSION = 1

#: Why a section is in the answer's scope. The three are not interchangeable and the distinction
#: survives into the cap (see :func:`select_sections`): ``score`` was DISCOVERED by the reranker,
#: while ``facet`` and ``amendment`` were DECIDED — by the router's facet matches and by the
#: amendment view respectively — and a decided inclusion is not a guess the cap may silently undo.
SelectionReason = Literal["score", "facet", "amendment"]

# Bounds. Guards, not tuning knobs — the tunable numbers (`section_threshold`, `max_sections`)
# are config and arrive as arguments.
_MAX_RERANK_CHARS = 1200  # a cross-encoder truncates its input; truncate the CANDIDATE, not the query
_MAX_CONTIGUOUS_GAP = 3  # procedural gap-filling: a run, not the whole document


@dataclass(frozen=True, slots=True)
class SectionCard:
    """One section as the selector sees it: the Phase-2 skeleton plus the Phase-4 enrichment.

    This is to a section what :class:`~rag_core.router.DocCard` is to a document — the
    queryable projection, not the row. ``summary`` is the Phase-4 section summary and is the only
    body text here; a section whose enrichment never ran (the flag was off, or the call failed —
    an ordinary outcome by design) has an empty one and is scored on its breadcrumb and heading
    alone, which is degraded but not broken: a heading like "Video Wall Technical Requirements"
    carries real signal for a facet sub-query.

    ``amends`` is a BOOL, not the stored object: the plan's step 5 asks only whether the section
    amends anything ("union all sections with ``amends`` not null"), and carrying the payload
    would put the amended CLAUSE TEXT — which is document text — on a path that does not need it.
    """

    section_id: str
    doc_id: str
    breadcrumb: str = ""
    heading: str = ""
    summary: str = ""
    facet_ids: tuple[str, ...] = ()
    amends: bool = False
    ordinal: int = 0
    depth: int = 0
    token_count: int = 0

    def rerank_text(self, *, max_chars: int = _MAX_RERANK_CHARS) -> str:
        """What the cross-encoder is given for this section.

        Breadcrumb first, because it is the most compressed statement of what the section IS and
        a truncated candidate loses its tail; then the heading when it adds something the
        breadcrumb does not already end with; then the summary.

        Truncation is on the CANDIDATE and never on the query — the plan says so explicitly, and
        the reason is that a truncated query changes what is being asked, while a truncated
        candidate only limits what the model gets to read of an answer that is already identified
        by its heading."""
        head = self.breadcrumb or self.heading
        parts = [head]
        if self.heading and not head.endswith(self.heading):
            parts.append(self.heading)
        if self.summary:
            parts.append(self.summary)
        return " — ".join(p for p in parts if p)[:max_chars]


@dataclass(frozen=True, slots=True)
class SelectedSection:
    """One section in the answer's scope, with the score and the reason(s) that put it there."""

    section_id: str
    doc_id: str
    score: float
    reasons: tuple[SelectionReason, ...]

    @property
    def decided(self) -> bool:
        """True when the router or the amendment view asked for this section by name.

        The cap keeps decided sections ahead of discovered ones — see :func:`select_sections`."""
        return "facet" in self.reasons or "amendment" in self.reasons


@dataclass(frozen=True, slots=True)
class SectionSelection:
    """What an aggregate-class strategy decided to build the answer from.

    ``dropped`` is the plan's step 6 — "record the rest for the coverage statement". It is NOT an
    error and NOT a failure: it is the set of sections that were in scope, passed selection, and
    were cut by the cap, and Phase 9's reduce prompt states it to the reader rather than letting
    the answer silently describe part of a document as if it were all of it."""

    selected: tuple[SelectedSection, ...] = ()
    dropped: tuple[str, ...] = ()
    considered: int = 0
    threshold: float = 0.0
    strategy_version: int = STRATEGY_VERSION

    @property
    def section_ids(self) -> tuple[str, ...]:
        """The ids, in selection order — what ``query_log.selected_sections`` stores."""
        return tuple(s.section_id for s in self.selected)

    def as_json(self) -> str:
        """The diagnostic shape. Mirrors ``RouterDecision.as_json`` so a stored selection can be
        read back without the code that produced it."""
        return json.dumps(
            {
                "strategy_version": self.strategy_version,
                "considered": self.considered,
                "threshold": round(self.threshold, 4),
                "selected": [
                    {
                        "section_id": s.section_id,
                        "doc_id": s.doc_id,
                        "score": round(s.score, 4),
                        "reasons": list(s.reasons),
                    }
                    for s in self.selected
                ],
                "dropped": list(self.dropped),
            },
            ensure_ascii=False,
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class StrategyPlan:
    """The plan's §8 recipe for one intent, as flags a caller can act on.

    Deliberately declarative. The pipeline already owns a hybrid retriever, a reranker, a parent
    expander and a decomposition block; what Phase 8 adds is not a second execution engine but
    the decision of WHICH of those to run at WHICH grain — and keeping that decision as data is
    what lets a unit test assert the recipe without standing up any of them."""

    intent: str
    levels: tuple[ChunkLevel, ...] = ()
    section_pass: bool = False
    expand_parents: bool = False
    contiguous: bool = False
    per_target: bool = False
    graph: bool = False
    exhaustive: bool = False

    @property
    def needs_sections(self) -> bool:
        """True when this strategy reads the catalog's ``sections`` rows at all."""
        return self.section_pass or self.expand_parents


# The plan's §8 strategy table, one row per intent.
#
# `exhaustive` marks the intents whose scope is the document's WHOLE facet taxonomy rather than
# the facets the router matched in the query's words. It is a statement about what the question
# MEANS, not a tuning choice: "give me the compliance details from this document" deliberately
# names no facet, so matching facets by word overlap — which is right for a lookup — is exactly
# backwards for an aggregate, and the plan's own QueryPlan mapping already marks these intents
# `coverage="exhaustive"`. Measured on the real AEGCL corrigendum: probing the matched facet
# alone reached 6 of the acceptance's 14 section groups, probing the whole taxonomy reached 11.
# See :func:`taxonomy_of`.
#
# `lookup` is the plan's own level list verbatim. `aggregate` and `summarize` ask for NO levels
# because they do not run a passage search at all — they read every section out of the catalog —
# and an empty tuple is "unscoped", the pre-Phase-8 behaviour, for any caller that retrieves
# anyway. `compare` runs the aggregate recipe once per target. `procedural` is lookup plus the
# contiguity rule. `relational` is the GraphRAG traversal the pipeline already implements, whose
# cited chunks then expand to their sections for grounding.
_STRATEGIES: dict[str, StrategyPlan] = {
    "lookup": StrategyPlan(
        intent="lookup",
        levels=(ChunkLevel.CHUNK, ChunkLevel.ROW, ChunkLevel.PROPOSITION),
        expand_parents=True,
    ),
    "aggregate": StrategyPlan(intent="aggregate", section_pass=True, exhaustive=True),
    "compare": StrategyPlan(intent="compare", section_pass=True, per_target=True, exhaustive=True),
    "summarize": StrategyPlan(intent="summarize", section_pass=True, exhaustive=True),
    "procedural": StrategyPlan(
        intent="procedural",
        levels=(ChunkLevel.CHUNK, ChunkLevel.ROW, ChunkLevel.PROPOSITION),
        expand_parents=True,
        contiguous=True,
    ),
    "relational": StrategyPlan(intent="relational", expand_parents=True, graph=True),
}

_FALLBACK = _STRATEGIES["lookup"]


def plan_strategy(intent: str) -> StrategyPlan:
    """The recipe for ``intent``, falling back to ``lookup``.

    Same fallback the router itself takes on an unusable decision, and for the same reason: the
    question still has to be answered, and lookup is the mechanism that was already running
    before any of this existed."""
    return _STRATEGIES.get(intent, _FALLBACK)


def levels_for_intent(intent: str) -> tuple[ChunkLevel, ...]:
    """The retrieval grains ``intent``'s hybrid call asks for; empty means unscoped.

    Empty is NOT "all levels" as a filter would express it — it is no filter at all, which also
    returns v1 chunks that carry no level. That matters: every document ingested before the v2
    flag has level-less chunks, and a strategy that filtered to the three v2 levels would make
    them invisible."""
    return plan_strategy(intent).levels


def taxonomy_of(cards: Sequence[SectionCard]) -> tuple[tuple[str, str], ...]:
    """Every ``(doc_id, facet_id)`` the in-scope sections actually carry, sorted.

    THE APPLIED taxonomy, read off the sections, not the profile card's declared one. The two
    differ for a real reason: Phase 3 proposes a facet vocabulary and Phase 4 decides which
    sections each label actually describes, so a facet the card declares and no section was given
    names nothing to select. Reading it here also needs no extra plumbing — the cards are already
    in hand — and it cannot drift from what ``select_sections``'s step 4 joins against.

    This is what an EXHAUSTIVE intent's scope is (see :class:`StrategyPlan`)."""
    return tuple(sorted({(card.doc_id, facet) for card in cards for facet in card.facet_ids}))


def max_scores(per_query: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Plan step 2: "section score = max" across the main query and every sub-query.

    MAX, not mean, and the measurement in this module's header is why. A section scores ~1e-5
    against the vague main query and 0.65 against the one sub-query that is actually about it;
    a mean over six sub-queries would divide that 0.65 by six and put the section's only real
    evidence below any threshold worth setting. The max asks "did ANY facet of this question
    find this section", which is the question an aggregate answer needs answered."""
    best: dict[str, float] = {}
    for scores in per_query:
        for section_id, score in scores.items():
            if score > best.get(section_id, float("-inf")):
                best[section_id] = score
    return best


def select_sections(
    cards: Sequence[SectionCard],
    scores: Mapping[str, float],
    *,
    facets: Sequence[tuple[str, str]] = (),
    amendment_view: bool = False,
    threshold: float = 0.15,
    relative: float = 0.1,
    max_sections: int = 40,
) -> SectionSelection:
    """The plan's §8 `aggregate` selection, steps 3 to 6.

    * **Step 3** — keep sections scoring at or above ``threshold``, OR within ``relative`` of the
      best-scoring section. The plan sets the absolute floor low (~0.15) and says why: recall
      over precision. An aggregate answer that omits a requirement is wrong in a way that an
      aggregate answer carrying an extra section is not, because Phase 9's map step returns
      ``[]`` for a section with nothing relevant in it — an over-selected section costs a model
      call, while an under-selected one costs a missing clause.
    * **Step 4** — union in every section whose ``facet_ids`` intersect the router's facets.
    * **Step 5** — with ``amendment_view``, union in every section that amends something.
    * **Step 6** — cap at ``max_sections``, recording the rest.

    THE RELATIVE FLOOR IS NOT IN THE PLAN. It is here because the plan's absolute threshold was
    measured against the real AEGCL corrigendum and selected NOTHING — 0 of 78 sections, at
    0.15 and at every value down to 0.001. A cross-encoder's score is absolute for a fixed
    CANDIDATE SHAPE, and the shape the plan calibrated 0.15 against is a retrieved PASSAGE. A
    section card is a breadcrumb plus a one-line summary, and against that shape the same model's
    scores land two to four orders of magnitude lower while still separating cleanly: on the
    real document, a facet probe scored the three relevant sections at 1.0 / 0.657 / 0.24 of the
    best and every irrelevant one at 0.000. The RATIO carries the signal that the magnitude does
    not, so a scale-free floor is what makes the plan's own "recall over precision" intent
    achievable, and it is the same hazard ``CrossEncoderReranker._verify_model`` already names:
    a threshold is a claim about one model's score distribution.

    BOTH FLOORS APPLY, by OR, so the plan's knob is kept rather than replaced: an operator who
    calibrates an absolute threshold for their own reranker gets exactly what they asked for,
    and ``relative=0.0`` turns the addition off entirely.

    THE DEGENERATE CASE IS DELIBERATE AND IS STATED HERE because it is the cost. When NOTHING in
    the document is relevant, the best score is itself noise and the ratios among noise are
    flat — so the relative floor selects broadly. For a scoped aggregate that is the honest
    answer rather than a bug: the caller named this document and asked for everything in it, the
    fan-out is bounded by ``max_sections``, and Phase 9's reduce states the coverage. It is
    ALSO why ``relative`` must never be applied to a lookup, which is a question that genuinely
    can have no answer — no strategy with ``section_pass`` is one of those.

    THE CAP IS REASON-TIERED, which is a deliberate reading of step 6 rather than its literal
    wording. The plan says "keep highest-scoring"; taken literally, a cap could drop a section
    that steps 4 and 5 had just explicitly UNIONED IN, because a facet match carries whatever
    score the reranker gave it and an amendment section matched by step 5 alone may score near
    zero. That would make the two union steps quietly conditional on a number they never see —
    and on the plan's own acceptance document, the amendment table is the single most important
    thing in it. So decided inclusions (``facet``, ``amendment``) are kept ahead of discovered
    ones (``score``), each tier ordered by score. Within a tier, ties break on (doc_id, ordinal)
    so the selection is deterministic and reads in document order.

    ``scores`` may be missing a section entirely — a section that no query scored is simply
    absent, treated as 0.0, and still selectable by step 4 or step 5."""
    wanted = {(doc_id, facet_id) for doc_id, facet_id in facets}
    by_id = {card.section_id: card for card in cards}
    # The best score ANY section reached, which is what the relative floor is a fraction of. A
    # card absent from `scores` counts as 0.0 here exactly as it does below.
    best = max((scores.get(card.section_id, 0.0) for card in cards), default=0.0)
    floor = min(threshold, best * relative) if (relative > 0.0 and best > 0.0) else threshold
    picked: list[SelectedSection] = []
    for card in cards:
        reasons: list[SelectionReason] = []
        score = scores.get(card.section_id, 0.0)
        if score >= floor:
            reasons.append("score")
        if wanted and any((card.doc_id, facet_id) in wanted for facet_id in card.facet_ids):
            reasons.append("facet")
        if amendment_view and card.amends:
            reasons.append("amendment")
        if reasons:
            picked.append(
                SelectedSection(section_id=card.section_id, doc_id=card.doc_id, score=score, reasons=tuple(reasons))
            )

    def rank(item: SelectedSection) -> tuple[int, float, str, int]:
        card = by_id[item.section_id]
        return (0 if item.decided else 1, -item.score, item.doc_id, card.ordinal)

    picked.sort(key=rank)
    keep, cut = picked[: max(0, max_sections)], picked[max(0, max_sections) :]
    return SectionSelection(
        selected=tuple(keep),
        dropped=tuple(s.section_id for s in cut),
        considered=len(cards),
        # The floor that was actually APPLIED, not the one that was asked for — on a real
        # document those differ, and a stored selection has to say which number produced it.
        threshold=floor,
    )


def select_per_target(
    cards: Sequence[SectionCard],
    scores: Mapping[str, float],
    *,
    facets: Sequence[tuple[str, str]] = (),
    amendment_view: bool = False,
    threshold: float = 0.15,
    relative: float = 0.1,
    max_sections: int = 40,
) -> SectionSelection:
    """The plan's `compare` strategy: "run aggregate per entity/doc with shared facets".

    PER DOCUMENT, because a shared cap is not a comparison. With one global cap of 40 and two
    documents in scope, a 200-section tender takes every slot and the 10-section corrigendum it
    is being compared against gets none — and the answer then compares a document with itself
    while reporting that it compared two. Each target gets its own share of the budget (at least
    one section), so the cap can thin a side but never erase it.

    THE SCORES AND THE FLOORS ARE SHARED, deliberately. ``scores`` is computed once over all the
    candidates, which is what makes the two sides comparable at all — a relative floor computed
    per document would mean "the best section IN THIS DOCUMENT", so a document with nothing
    relevant would contribute its best irrelevant section as though it were a match. The floor
    is a property of the question, not of the document.

    The result reads in document order and carries the combined ``considered`` and ``dropped``,
    so a caller cannot tell from the shape that the budget was split — only from the fact that
    both documents are represented."""
    by_doc: dict[str, list[SectionCard]] = {}
    for card in cards:
        by_doc.setdefault(card.doc_id, []).append(card)
    if len(by_doc) <= 1:
        return select_sections(
            cards,
            scores,
            facets=facets,
            amendment_view=amendment_view,
            threshold=threshold,
            relative=relative,
            max_sections=max_sections,
        )
    # The floors are computed against the WHOLE candidate set and then handed down as absolutes,
    # so each per-document pass applies the same bar. `relative=0.0` below is not "off" — it is
    # "already applied", and recomputing it per document is the bug this avoids.
    whole = select_sections(
        cards,
        scores,
        facets=facets,
        amendment_view=amendment_view,
        threshold=threshold,
        relative=relative,
        max_sections=len(cards),
    )
    share = max(1, max_sections // len(by_doc))
    selected: list[SelectedSection] = []
    dropped: list[str] = []
    for _doc_id, doc_cards in sorted(by_doc.items()):
        ids = {c.section_id for c in doc_cards}
        mine = [s for s in whole.selected if s.section_id in ids]
        selected.extend(mine[:share])
        dropped.extend(s.section_id for s in mine[share:])
    return SectionSelection(
        selected=tuple(selected),
        dropped=tuple(dropped),
        considered=len(cards),
        threshold=whole.threshold,
    )


def parent_sections(section_ids: Sequence[str | None]) -> tuple[str, ...]:
    """Plan step 2 of the `lookup` strategy: "expand each hit to its parent, dedupe by section".

    Order-preserving, because the order is the RANKING — the first hit's section is the strongest
    evidence the retriever found and must stay first. ``None`` is the ordinary case for a v1 chunk
    (no ``section_id``) and is skipped rather than grouped under a null section."""
    seen: dict[str, None] = {}
    for section_id in section_ids:
        if section_id:
            seen.setdefault(section_id, None)
    return tuple(seen)


def contiguous_sections(
    seeds: Sequence[str], cards: Sequence[SectionCard], *, max_gap: int = _MAX_CONTIGUOUS_GAP
) -> tuple[str, ...]:
    """Plan `procedural`: "expand to full contiguous sections in document order".

    A procedure retrieved as three matching sections out of a run of five reads as three steps,
    with no sign that two are missing — which is the specific failure the plan's procedural
    strategy exists to fix. So the gaps BETWEEN seeds are filled, per document, in ordinal order.

    ``max_gap`` bounds it: a document whose first and last section both matched would otherwise
    "fill the gap" with the entire document. A gap wider than ``max_gap`` is left open, which
    says the two seeds are not parts of one run.

    The result is in document order, not ranked order — unlike :func:`parent_sections`, because
    the whole point is to present the steps in the sequence they are performed in."""
    by_doc: dict[str, list[SectionCard]] = {}
    for card in sorted(cards, key=lambda c: (c.doc_id, c.ordinal, c.section_id)):
        by_doc.setdefault(card.doc_id, []).append(card)
    wanted = set(seeds)
    out: list[str] = []
    for doc_cards in by_doc.values():
        hits = [i for i, card in enumerate(doc_cards) if card.section_id in wanted]
        if not hits:
            continue
        keep: set[int] = set(hits)
        for left, right in pairwise(hits):
            if right - left - 1 <= max_gap:
                keep.update(range(left + 1, right))
        out.extend(doc_cards[i].section_id for i in sorted(keep))
    return tuple(out)
