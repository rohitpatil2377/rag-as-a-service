"""Query transformation (FR-D6) — interfaces, deterministic decomposition, clean merge.

Plugs into the D5 router's **heavy** path: a complex query is transformed into several
sub-queries (decomposition) or a hypothetical passage (HyDE) / rewrite, each retrieved
independently, then the results are **merged, de-duplicated by chunk, and kept by best
score** so the union lifts recall WITHOUT adding duplicates or noise.

Pure-additive: no schema change, gated by ``query_transform_enabled`` (default OFF).
The deterministic ``DecompositionTransformer`` is the CI default; HyDE/rewrite are LLM
opt-ins. See ADR-0009.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from rag_schemas import RetrievalQuery, RetrievedSpan

_MAX_SUBQUERIES = 4
_MIN_SEGMENT_LEN = 4
_SPLIT_SEPS = (" vs ", " versus ", " and also ", " as well as ", " and ")
# Numbering/bullets a model puts in front of a list item, stripped before the line is judged.
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")
# How many reformulations to ask for. Each one is a full retrieval, so this is a cost knob as
# much as a recall one; the union is merged and de-duplicated by chunk afterwards.
MAX_REFORMULATIONS = 3
# A reasoning model spends its budget thinking BEFORE it answers, so a budget sized for the
# answer alone truncates it mid-thought and no answer is ever emitted. Sized for the working
# plus the answer; a non-reasoning model simply finishes early.
REFORMULATION_MAX_TOKENS = 900
# How much of each grounding passage to show. Enough to carry the vocabulary and the rule
# names, short enough that several passages plus the model's own reasoning fit its context.
_EXCERPT_CHARS = 400
# How many passages to ground on. The failing retrieval's best few are the ones most likely to
# be in the right neighbourhood even when none of them answers the question.
MAX_GROUNDING_EXCERPTS = 6
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


@runtime_checkable
class QueryTransformer(Protocol):
    """Expands a query into 1+ retrieval strings (always includes the original) (FR-D6)."""

    async def transform(self, query: RetrievalQuery, context: list[str] | None = None) -> list[str]:
        """Alternative phrasings to retrieve on. ``context`` is passage text from a previous,
        unsuccessful retrieval — the corpus's own vocabulary, which a transformer may use to
        stop guessing synonyms. Implementations that cannot use it ignore it."""
        ...


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        key = s.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(s.strip())
    return out


class DecompositionTransformer:
    """Deterministic, no-LLM decomposition of multi-part / comparative queries."""

    async def transform(self, query: RetrievalQuery, context: list[str] | None = None) -> list[str]:
        """Deterministic and corpus-blind: splitting a question needs no knowledge of the
        documents, so ``context`` is deliberately unused here."""
        text = query.text.strip()
        subs: list[str] = []

        # Multi-question: split on '?'.
        parts = [p.strip() for p in text.split("?") if p.strip()]
        if len(parts) > 1:
            subs.extend(f"{p}?" for p in parts)

        # Comparative / conjunctive: split on the first matching separator.
        low = text.lower()
        for sep in _SPLIT_SEPS:
            if sep in low:
                segs = re.split(re.escape(sep), text, flags=re.IGNORECASE)
                subs.extend(s for s in (seg.strip() for seg in segs) if len(s) >= _MIN_SEGMENT_LEN)
                break

        subs.append(text)  # always keep the holistic query
        return _dedupe(subs)[:_MAX_SUBQUERIES]


def build_hyde_prompt(query: RetrievalQuery) -> str:
    """HyDE: generate a hypothetical answer passage to embed for retrieval."""
    return (
        "Write a short, factual passage (2-3 sentences) that would plausibly answer the "
        "QUESTION, as if excerpted from a source document. Answer with ONLY the passage.\n\n"
        f"QUESTION: {query.text}\n\nPASSAGE:"
    )


def build_reformulation_prompt(query: RetrievalQuery, limit: int, excerpts: list[str] | None = None) -> str:
    """Ask for alternative phrasings that would find the RULE a situation turns on.

    This exists because of a measured failure, not a theory. Asked as a scenario — "The CMD
    decides at 9 PM to add an unrehearsed keynote segment for tomorrow's flagship event. Is
    this permitted?" — retrieval scored its best passage 0.0084 and the platform abstained. The
    answering sentence was in the corpus all along ("If the rehearsal has not been completed
    and signed off, the speaker does not go on stage. No exceptions, regardless of seniority"),
    and asking "Can an unrehearsed speaker go on stage?" scored it 0.9886.

    A scenario and a rule share almost no vocabulary: the question is about CMD, 9 PM, keynote,
    tomorrow; the rule is about rehearsal, sign-off, seniority. The deterministic transformer
    splits on "?" and conjunctions, so it cannot bridge that — it returned the query unchanged.

    Asks for the QUESTION the situation turns on, not a bag of keywords: a keyword soup scores
    no better against a cross-encoder than the original did.

    ``excerpts`` are passages the failing search DID return. They matter more than the wording
    of this prompt. Measured on the same question:

        no excerpts   0.0084 -> 0.0084   the model invents plausible-sounding vocabulary
                                         ("unscheduled segments", "event programs") because it
                                         cannot see what the corpus calls things
        excerpts      0.0084 -> 0.9976   it used "Content Curator", "event sign-off",
                                         "No-Change Rule" — terms lifted from real passages —
                                         and the answering chunk came back first

    Even a failed retrieval returns text from the right corpus in the corpus's own words. That
    is the grounding: the model stops guessing synonyms and starts using the vocabulary the
    documents actually use."""
    grounding = ""
    if excerpts:
        passages = "\n\n".join(f"[{i}] {e.strip()[:_EXCERPT_CHARS]}" for i, e in enumerate(excerpts, 1))
        grounding = (
            "Below are the passages the search DID return. They are from the same corpus and "
            "show the words, roles and rule names these documents actually use.\n"
            "Use ONLY vocabulary that appears in them.\n\n"
            f"PASSAGES:\n{passages}\n\n"
        )
    return (
        "A search over a policy and procedure corpus found nothing useful for the QUESTION "
        "below. It may describe a SITUATION rather than ask about the rule that governs it, or "
        "use different words from the documents.\n"
        f"{grounding}"
        f"Write up to {limit} alternative questions that would find the governing rule. Ask "
        "about the underlying policy, not about the people or times in the scenario. One per "
        "line, no numbering, no explanation. Invent no facts.\n\n"
        f"QUESTION: {query.text}\n\nALTERNATIVES:"
    )


def parse_reformulations(raw: str, limit: int) -> list[str]:
    """Lines of a reformulation response, cleaned of numbering and bullets.

    Deliberately forgiving about the model's formatting and strict about what counts: a line
    has to look like a question, or it is the model explaining itself rather than asking."""
    # A reasoning model writes its working first, and that working is FULL of question-shaped
    # lines ("Maybe the policies talk about last-minute changes?"). Scraping those would feed
    # the model's guesses back into retrieval as if they were reformulations. An unclosed block
    # means the budget ran out before any answer was written — there is nothing to parse.
    if _THINK_CLOSE in raw:
        raw = raw.rsplit(_THINK_CLOSE, 1)[-1]
    elif _THINK_OPEN in raw:
        return []
    out: list[str] = []
    for line in raw.splitlines():
        cleaned = _LIST_PREFIX_RE.sub("", line.strip()).strip(' -*"\u2022').strip()
        if cleaned.endswith("?") and len(cleaned) >= _MIN_SEGMENT_LEN:
            out.append(cleaned)
    return _dedupe(out)[:limit]


def build_rewrite_prompt(query: RetrievalQuery) -> str:
    """Rewrite the query into a clearer, keyword-rich search query."""
    return (
        "Rewrite the QUESTION into a single clear search query with the key entities and "
        "terms, no filler. Answer with ONLY the rewritten query.\n\n"
        f"QUESTION: {query.text}\n\nSEARCH QUERY:"
    )


def merge_retrieved(
    results: list[list[RetrievedSpan]], top_k: int, *, diversify: bool = True
) -> tuple[list[RetrievedSpan], float]:
    """Union sub-query results, de-dupe by ``chunk_id`` keeping the BEST score, take top_k.

    This is the quality-critical step (FR-D6): dedupe prevents the same chunk counting
    twice, and keeping the max score means a chunk is ranked by the strongest evidence any
    sub-query found for it. Returns ``(spans, max_score)`` — ``max_score`` feeds the
    grounded-or-abstain gate.

    ``diversify`` (default) interleaves the sub-queries round-robin so EACH contributes
    evidence — a "compare A and B" question then surfaces sources for both A and B, instead
    of one high-scoring topic crowding the other out of the global top-k. With a single
    sub-query (or ``diversify=False``) this is a plain global sort by score.
    """
    best: dict[str, RetrievedSpan] = {}
    for spans in results:
        for s in spans:
            cur = best.get(s.chunk_id)
            if cur is None or s.score > cur.score:
                best[s.chunk_id] = s

    if not diversify or len(results) <= 1:
        merged = sorted(best.values(), key=lambda s: s.score, reverse=True)[:top_k]
        return merged, (merged[0].score if merged else 0.0)

    # Round-robin: each sub-query takes a turn contributing its best not-yet-picked chunk.
    # Visit lists STRONGEST-top-first so the single best span always leads (evidence order
    # is preserved), then the weaker lists get their fair share behind it.
    ranked = [sorted(spans, key=lambda s: s.score, reverse=True) for spans in results]
    order = sorted(range(len(ranked)), key=lambda i: ranked[i][0].score if ranked[i] else -1.0, reverse=True)
    pos = [0] * len(ranked)
    picked: set[str] = set()
    merged = []
    while len(merged) < top_k:
        advanced = False
        for li in order:
            lst = ranked[li]
            while pos[li] < len(lst) and lst[pos[li]].chunk_id in picked:
                pos[li] += 1
            if pos[li] < len(lst):
                s = lst[pos[li]]
                pos[li] += 1
                picked.add(s.chunk_id)
                merged.append(best[s.chunk_id])
                advanced = True
                if len(merged) >= top_k:
                    break
        if not advanced:
            break
    return merged, max((s.score for s in merged), default=0.0)
