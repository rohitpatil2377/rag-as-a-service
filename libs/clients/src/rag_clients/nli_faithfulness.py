"""NLI-based faithfulness checker (implements :class:`FaithfulnessChecker`) — FR-E4.

The lexical checker scores TOKEN OVERLAP and the LLM-judge asks the generator's own
model family to grade itself (circular, and weak on a small model — ADR-0005). Neither
tests *entailment*: whether each answer sentence logically FOLLOWS FROM the cited spans.

A Natural-Language-Inference cross-encoder (e.g. ``cross-encoder/nli-deberta-v3-base``)
does exactly that — it reads (premise = sources, hypothesis = answer sentence) and
returns P(entailment). This checker splits the answer into sentences, scores each
against the union of cited spans, and reports the fraction that are entailed. A
sentence that contradicts or is merely unrelated to the sources is unsupported.

The scoring LOGIC is pure and takes an injected async ``entail`` callable, so it is
unit-tested with a deterministic fake — no model, no network. The real scorer
(:class:`TEINLIScorer`) posts to a self-hosted TEI ``/predict`` endpoint; it is
sovereign (egress-allowlisted) exactly like the reranker and embedder.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import replace
from os.path import commonprefix

import httpx

from rag_common import ConfigError, RetrievalError, assert_sovereign, get_logger, get_settings, model_tls_verify
from rag_core import FaithfulnessVerdict
from rag_core.anchoring import anchor_score
from rag_core.faithfulness import FaithfulnessChecker
from rag_schemas import RetrievedSpan

_log = get_logger(__name__)

# Split on Latin AND Devanagari danda (।) / double-danda (॥) so Indic answers segment too. Also
# split on NEWLINES so a Markdown bullet list (each "- ..." on its own line, no danda) becomes one
# claim per line instead of one giant glued "sentence" — a glued blob confuses the cross-encoder.
_SENT_RE = re.compile(r"(?<=[.!?।॥])\s+|\n+")
# A "sentence" that is only a citation marker / punctuation ("[1]", "।") carries no claim to
# verify — it must not drag the score down. Require at least one letter (any script).
_HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
# Markdown the generator adds for readability but which is NOT part of the CLAIM: leading list
# markers / headings and inline bold/italic/code markers + citation brackets. Stripped before NLI
# so the cross-encoder scores the sentence's meaning, not its formatting.
_MD_LEAD = re.compile(r"^\s*(?:[-*+•]|\d+[.)]|#{1,6})\s+")
_MD_INLINE = re.compile(r"(\*\*|__|\*|_|`|\[\d+\])")

# --- FR-E4 adversarial-test fix: relevance-gate a COMPOUND query's off-topic half ----------
# "not contradicted" (see _support_prob below) only tests whether a sentence conflicts with the
# sources — it says nothing about whether the sources are ABOUT what was asked. That distinction
# is invisible for an ordinary single-topic query: the upstream retrieval abstain gate already
# screens the whole KB on `max_score`, so an off-topic KB never reaches generation at all. But a
# COMPOUND question ("What is X, and what is Y?") can have a high overall max_score because X is
# well covered, while Y has NO real coverage in this KB — and a chunk that merely shares surface
# vocabulary with Y (a year, "budget", a bare number) still gets retrieved, cited, and then
# rubber-stamped, because nothing in the sources CONTRADICTS the wrong figure it hands back for Y.
#
# Confirmed live against the deployed dev cross-encoder: for the query "What is the annual leave
# policy, and what is the FY2027 Platform Engineering budget?" against a KB that has leave data
# but an unrelated "Project Ganga" budget doc, the reranker scores the Ganga chunk 0.53 vs 0.54
# for the genuinely relevant leave chunk — i.e. on THIS small dev model, per-query relevance score
# alone does not reliably separate "answers half the question" from "coincidentally similar
# vocabulary, wrong topic" (ADR-0005's own documented "weak judge on a small model" limitation).
# Lexical coverage of the query AS A WHOLE has the identical problem (both chunks cover ~29% of
# the compound query's terms). What DOES separate them is coverage PER SUB-QUESTION: the leave
# chunk covers a majority of the leave clause's terms; the Ganga chunk covers only "budget" and
# "fy2027" out of the budget clause's terms, missing the clause's own distinguishing noun phrase
# ("Platform Engineering") entirely. So: split a compound query into its sub-questions, and ask
# which of them the RETRIEVED SET covers — a sub-question no passage covers is one this KB cannot
# answer, and a sentence answering it is unsupported however uncontradicted it reads.
#
# Asked of the whole retrieved set, not of the one passage a sentence best matches. The first cut
# did the latter and it was wrong twice over, both measured live on a real HR corpus: coverage was
# counted on exact surface forms, so a policy writing "approved"/"expenses" against a clause
# saying "approves"/"expense" scored 0.40 and a correct answer ABSTAINED (hence _same_term); and a
# document states its subject in a heading and its rule in the paragraph below, which chunking
# puts in different passages, so the paragraph that WAS the answer covered 3 of 6 clause terms and
# was thrown away while its own heading chunk sat in the same result set. Neither loosens the
# adversarial case: no passage covers the budget clause, so it stays uncovered and gated.
#
# An ordinary (non-compound) query never triggers this: the split only fires on an explicit
# "<clause>, and <wh-question>" pattern, so a normal single-topic answer is scored exactly as
# before.
_CLAUSE_SPLIT = re.compile(
    r"\s*(?:,\s*and\s+|;\s*and\s+|\s+and\s+)"
    r"(?=(?:what|how|when|where|who|whom|why|which|does|do|did|is|are|can|will|should|could)\b)",
    re.IGNORECASE,
)
_STOPWORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "of",
        "to",
        "in",
        "on",
        "for",
        "that",
        "this",
        "these",
        "those",
        "it",
        "do",
        "does",
        "did",
        "be",
        "been",
        "being",
        "can",
        "will",
        "would",
        "should",
        "could",
    ]
)
# The verb a QUESTION uses to ask, which the answering document never contains. "What does
# Supersedes SAY on first issue" puts "say" in the clause's term set, and no policy writes "say" —
# so the clause could not reach the strict majority the coverage test requires, and the passage
# that answers it verbatim scored 2/4. Measured live: that alone gated a correct, fully-anchored
# answer and abstained a question whose answer is written in the source.
#
# Deliberately narrow. Only verbs that are asking-words in this position are here: "state" is a
# place in an Indian corpus, "require" and "report" are what policies DO, and removing those would
# eat the topic instead of the question. Applied to clause TERMS only — retrieval is untouched.
_ASKING_VERBS = frozenset(
    [
        "say",
        "says",
        "said",
        "mention",
        "mentions",
        "mentioned",
        "tell",
        "tells",
        "told",
        "describe",
        "describes",
        "described",
        "explain",
        "explains",
        "explained",
        "specify",
        "specifies",
        "specified",
        "indicate",
        "indicates",
        "indicated",
    ]
)
_TERM_WORD = re.compile(r"[A-Za-z0-9]+")
_RELEVANCE_MIN_COVERAGE = 0.5  # a supporting span must cover a STRICT MAJORITY of a clause's terms
# Characters of shared prefix before two surface forms count as the same word. Clause coverage is
# measured on the words a writer happened to use, and a policy writes "approved"/"expenses" where
# the question says "approves"/"expense" -- measured live, that inflection gap alone scored a chunk
# that plainly answers the sub-question at 0.40 and abstained a correct answer. Deliberately crude:
# a real stemmer is a dependency, and this only has to tell an inflection from a different word.
_MIN_SHARED_STEM = 5


def _terms(text: str) -> set[str]:
    return {w.lower() for w in _TERM_WORD.findall(text) if len(w) > 2 and w.lower() not in _STOPWORDS}  # noqa: PLR2004


def _clause_terms(text: str) -> set[str]:
    """Terms of a QUESTION's sub-clause — its asking verb removed (see ``_ASKING_VERBS``).

    Scoped to the clause alone, NOT to :func:`_terms`. Measured the hard way: filtering these
    globally also stripped them from premise ranking and window selection, where they are real
    signal — the source writes "shall be MENTIONED in this column", and losing that word sent the
    focus window to the wrong clause and re-broke the very question this was fixing."""
    return _terms(text) - _ASKING_VERBS


# A shorter term that IS the whole stem of the longer one is the same word even below
# _MIN_SHARED_STEM, which a four-letter noun can never reach. Matches rag_core.anchoring.
_MIN_WHOLE_STEM = 4


def _same_term(left: str, right: str) -> bool:
    """Whether two terms are the same word — equal, or sharing a long enough stem that only an
    ending separates them ("approves"/"approved"/"approval", "expense"/"expenses", "atom"/"atoms")."""
    if left == right:
        return True
    shared = len(commonprefix([left, right]))
    if shared >= _MIN_SHARED_STEM:
        return True
    return shared >= _MIN_WHOLE_STEM and shared == min(len(left), len(right))


def _clause_coverage(clause: set[str], span_terms: set[str]) -> float:
    """The share of a sub-question's terms this span actually uses, counting inflections."""
    if not clause:
        return 1.0
    return sum(1 for term in clause if any(_same_term(term, t) for t in span_terms)) / len(clause)


def _query_clauses(query: str) -> list[set[str]]:
    """Term sets for a compound query's sub-questions. Empty for an ordinary single-topic
    query (or an empty/unset query) — callers must treat that as 'nothing to gate'."""
    if not query or not query.strip():
        return []
    parts = [p.strip() for p in _CLAUSE_SPLIT.split(query) if p.strip()]
    if len(parts) < 2:  # noqa: PLR2004
        return []
    return [t for t in (_clause_terms(p) for p in parts) if t]


def _uncovered_clauses(spans: list[RetrievedSpan], clauses: list[set[str]]) -> list[set[str]]:
    """The sub-questions NO retrieved passage actually answers.

    This is the failure the gate exists for: a compound question whose second half the knowledge
    base has no coverage of, answered anyway from a passage that merely shares surface vocabulary.
    Asked of the whole retrieved set — not of one passage — because a document states its subject
    in its heading and its rule in the paragraph below, and chunking puts those in different
    passages. Measured live, judging a passage alone rejected the paragraph that WAS the answer."""
    span_terms = [_terms(sp.text) for sp in spans]
    return [
        clause
        for clause in clauses
        if clause and not any(_clause_coverage(clause, bt) > _RELEVANCE_MIN_COVERAGE for bt in span_terms)
    ]


def _is_clause_relevant(sentence: str, clauses: list[set[str]], uncovered: list[set[str]]) -> bool:
    """False iff `sentence` answers a sub-question the retrieved passages do not cover — i.e. it
    is only "supported" because nothing in the sources CONTRADICTS it. A sentence is taken to
    answer the sub-question it shares the most vocabulary with; one that shares none with any is
    left alone (not this gate's job — entailment and the lexical anchor already own that case)."""
    if not uncovered:
        return True
    st = _terms(sentence)
    if not st:
        return True
    shared = [(sum(1 for term in c if any(_same_term(term, t) for t in st)), c) for c in clauses if c]
    best_shared, best_clause = max(shared, key=lambda pair: pair[0], default=(0, set()))
    if best_shared == 0:
        return True
    return not any(best_clause == c for c in uncovered)


def _candidate_premises(
    sentence: str, spans: list[RetrievedSpan], keep: int = 3, max_chars: int | None = None
) -> list[str]:
    """The few spans most likely to support this sentence, EACH on its own, each inside the budget.

    Joining them was the other half of the premise problem. The budget bounds the WHOLE premise,
    so once chunks grow the best-ranked span alone fills it, is truncated, and every other span is
    dropped — including the one that actually says it. A claim is supported if ANY passage
    supports it, so the right question to ask the model is per passage, not against a
    concatenation that may not fit. Each candidate then fits the window on its own, and the
    verdict stops depending on how large the chunker happened to make the chunks.

    The per-candidate budget is what keeps each inside mDeBERTa's 512-token window. Measured
    on the HR corpus with a sentence quoted VERBATIM from its source: 400ch -> 0.969, 700ch ->
    0.950, 1400ch -> 0.981, 1983ch -> 0.279 — past the window the model returns CONTRADICTION
    for a sentence it can no longer see. A span longer than the budget is truncated rather
    than dropped: a sentence supported only by that chunk must still be judged against it.

    Composes with :func:`_focus`, which narrows WITHIN a span: separate candidates fix the join,
    the window fixes a single multi-topic chunk. Both were measured; neither subsumes the other."""
    budget = max_chars if max_chars is not None else get_settings().nli_premise_max_chars
    want = _terms(sentence)
    ranked = sorted(spans, key=lambda sp: len(want & _terms(sp.text)), reverse=True) if want else list(spans)
    out: list[str] = []
    for sp in ranked[:keep]:
        text = _focus(sp.text.strip(), want)[:budget]
        if text:
            out.append(text)
    return out


# A chunk is a retrieval unit, not a premise. Below this it is already one idea and is passed
# through whole; above it, the part that could support the sentence is selected out of it.
_FOCUS_MIN_CHARS = 420
# How much text a window carries. Wide enough to keep a clause with the label that names it
# ("5.8.8 Supersedes: ..."), which is what the sentence is usually about; narrow enough that
# neighbouring clauses about other fields do not come with it.
_FOCUS_WINDOW_CHARS = 300
_FOCUS_STEP_CHARS = 150

# A row of a serialized table, e.g. "| Quality Control | Q | G | O | C | QG/QO/QC |".
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
# A separator row ("| --- | --- |") marks the line above it as the HEADER.
_TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
# Rows to keep beside the header. More than one because a fact often needs its neighbour (a
# sub-heading row like "| Non Beta Lactam | | | |" governs the rows beneath it).
_TABLE_KEEP_ROWS = 4
# A header, its separator, and at least one row -- fewer is not a table worth reshaping.
_MIN_TABLE_LINES = 3
_HEADER_LINES = 2


def _focus(text: str, want: set[str]) -> str:
    """The window of ``text`` that shares most vocabulary with the sentence being judged.

    A sentence-pair NLI model is trained on single-idea premises. Give it a chunk that packs
    several unrelated clauses together and it loses the one that matters: measured on Acme's SOP
    corpus, a sentence written almost verbatim inside a 685-character chunk scored CONTRADICTION
    0.80 against the whole chunk and ENTAILMENT 0.983 against a 300-character window around its
    own clause. The answer was correct, cited, and abstained on.

    Distinct from the length bound in :func:`_premise_for`: 685 characters is well inside
    mDeBERTa's window, so this is granularity, not overflow. Both matter, and both are bounded
    here — the window is chosen first, then the budget applies.

    Falls back to the whole text when nothing overlaps, because a premise chosen by no evidence
    is worse than the chunk retrieval actually picked."""
    if len(text) <= _FOCUS_MIN_CHARS or not want:
        return text
    table = _table_focus(text, want)
    if table is not None:
        return table
    best, best_hits = text[:_FOCUS_WINDOW_CHARS], -1
    for start in range(0, max(1, len(text) - _FOCUS_WINDOW_CHARS // 2), _FOCUS_STEP_CHARS):
        window = text[start : start + _FOCUS_WINDOW_CHARS]
        hits = len(want & _terms(window))
        if hits > best_hits:
            best, best_hits = window, hits
    return best if best_hits > 0 else text


def _table_focus(text: str, want: set[str]) -> str | None:
    """Header + the rows that could answer, for a serialized table. ``None`` when not a table.

    A character window is the wrong instrument here, and it fails in a way that looks like the
    answer is wrong. A table's HEADING and HEADER row carry every word of the question --
    "department", "code", "Quality Control" -- while the row holding the ANSWER carries almost
    none of it, so the highest-overlap window is reliably the one WITHOUT the answer in it.

    Measured on the SOP corpus: a correct, correctly-cited answer ("the department code for
    Quality Control is Q") scored 0.000 against its own 698-character chunk, because the chosen
    window ended four rows above the row that says Q. The same table trimmed under the focus
    threshold scored 1.000. Neither the chunker nor retrieval was at fault -- the premise was.

    So the premise is assembled instead of cut: whatever precedes the table (the sentence that
    introduces it), the header, and the rows with the most claim vocabulary. A row without its
    header reads "| Q |"; a header without its row does not contain the answer at all.
    """
    lines = text.splitlines()
    rows = [i for i, ln in enumerate(lines) if _TABLE_ROW.match(ln)]
    if len(rows) < _MIN_TABLE_LINES:
        return None
    first, last = rows[0], rows[-1]
    lead = [ln for ln in lines[:first] if ln.strip()]
    header = [lines[i] for i in rows[:_HEADER_LINES] if i <= last]
    body = [lines[i] for i in rows[_HEADER_LINES:] if i <= last]
    if not body:
        return None
    # Rank rows by how much of the claim they carry; keep document order among those chosen so
    # a sub-heading row still reads above the row it governs.
    scored = sorted(range(len(body)), key=lambda i: len(want & _terms(body[i])), reverse=True)
    keep = sorted(scored[:_TABLE_KEEP_ROWS])
    if not len(want & _terms(body[scored[0]])):
        return None  # no row relates to the claim — a window is no worse and stays the default
    return "\n".join([*lead[-1:], *header, *[body[i] for i in keep]])


def _content_weight(sentence: str) -> int:
    """How much CLAIM a sentence carries, in content words (min 1).

    The score used to be a plain fraction of sentences, which made it sensitive to FORMATTING
    rather than truth: the same three facts scored 1.0 written as one prose sentence and 0.33
    written as three bullets, because each bullet became an independent pass/fail with a tiny
    denominator. Weighting by content words makes prose and bullets carrying identical content
    score identically — the metric measures the answer, not its layout."""
    return max(1, len(_terms(sentence)))


def _lexical_anchor(sentence: str, spans: list[RetrievedSpan], *, length_norm: float = 0.0) -> float:
    """How strongly the sentence is anchored in its best ONE span (``rag_core.anchoring``).

    This is the test 'not contradicted' cannot make. NLI scores CONTRADICTION sharply, so an
    unrelated-but-true sentence ("Narendra Modi is the prime minister of India") is merely
    NEUTRAL against a passage about something else — and therefore passed as supported at 1.0,
    with a citation attached to a source that never mentions it. Verified live.

    A sentence genuinely drawn from the sources shares their vocabulary even when it paraphrases
    heavily; a sentence recalled from model memory does not. Compared per-span, not against the
    concatenation, so a sentence must be anchored in ONE passage rather than scavenging a word
    from each.

    Words are IDF-weighted across the retrieved spans, matched through inflections, and compared
    only within scripts the evidence contains. Measured against the unweighted ASCII version at its
    old threshold: on 138 sentences of real grounded answers it wrongly fails 13 instead of 14, and
    on a labelled set it catches 6 invented sentences instead of 4. The ASCII version also scored
    Hindi sentences that contain a number at 0.00 (it saw only the digits), failing correct Hindi
    answers; those now anchor correctly."""
    return anchor_score(sentence, [s.text for s in spans], stopwords=_STOPWORDS, length_norm=length_norm)


def _to_claims(answer: str) -> list[str]:
    """Split an answer into clean claim sentences for entailment: segment on sentence
    terminators + newlines, strip Markdown list/emphasis markers, drop fragments with no letters."""
    claims: list[str] = []
    for raw in _SENT_RE.split(answer.strip()):
        s = _MD_INLINE.sub("", _MD_LEAD.sub("", raw)).strip()
        if s and _HAS_LETTER.search(s):
            claims.append(s)
    return claims


# (premise, hypothesis) -> P(supported) in [0,1] — the probability the hypothesis is NOT
# contradicted by the premise. The seam that makes the aggregation logic testable without a model.
EntailFn = Callable[[str, str], Awaitable[float]]


class NLIFaithfulnessChecker:
    """Faithfulness by sentence-level natural-language inference.

    ``support_threshold`` is the per-sentence P(supported) = 1 - P(contradiction) cutoff above
    which a sentence counts as supported (i.e. it must not be CONTRADICTED by the sources — see
    :func:`_support_prob` for why 'not contradicted' beats 'entailed' on abstractive answers).
    The verdict ``score`` is the fraction of supported sentences; ``unsupported_claims`` are the
    sentences the sources contradict.
    """

    def __init__(
        self,
        entail: EntailFn,
        *,
        support_threshold: float = 0.5,
        anchor_threshold: float | None = None,
        fallback: FaithfulnessChecker | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        self._entail = entail
        self._support = support_threshold
        self._max_concurrency = get_settings().nli_max_concurrency if max_concurrency is None else max_concurrency
        # Minimum share of a sentence's content words that must appear in ONE retrieved span.
        # 0 disables the check (restoring the old not-contradicted-only behaviour).
        self._anchor = get_settings().faithfulness_anchor_threshold if anchor_threshold is None else anchor_threshold
        self._length_norm = get_settings().faithfulness_anchor_length_norm
        # NLI is a self-hosted CPU cross-encoder — it can time out or be briefly unreachable. That
        # is an INFRA condition, not an unfaithful answer, so it must never fail the request or force
        # a good answer to abstain. On any scorer error we fall back to the deterministic lexical
        # checker (same philosophy as the LLM judge's fallback), so answers keep flowing.
        self._fallback = fallback

    async def _support_for(self, sentence: str, spans: list[RetrievedSpan]) -> float:
        """P(supported) for one claim: the best score over its candidate spans, judged one at a time.

        Stops at the first span that clears the support threshold — the common case still costs
        ONE model call, exactly as the joined premise did. Only a claim that no single span
        vouches for pays for the rest, and those are the claims worth the extra look."""
        best = 0.0
        for premise in _candidate_premises(sentence, spans):
            prob = await self._entail(premise, sentence)
            if prob >= self._support:
                return prob
            best = max(best, prob)
        return best

    async def check(self, answer: str, spans: list[RetrievedSpan], query: str = "") -> FaithfulnessVerdict:
        sentences = _to_claims(answer)
        usable = [sp for sp in spans if sp.text.strip()]
        if not sentences or not usable:
            # Nothing to check (empty answer, or an abstain with no spans) — treat as vacuously
            # faithful; the grounded-or-abstain gate upstream already handles the empty case.
            return FaithfulnessVerdict(score=1.0, faithful=True, unsupported_claims=[])
        # BOUNDED concurrency. Unbounded `gather` was right while answers were single-pass and a
        # dozen sentences long, and became wrong when RAG v2's map-reduce path started producing
        # aggregate answers of several hundred claims: those opened more requests than httpx's
        # default 100-connection pool, the overflow waited for a slot, the wait counted against
        # the per-call timeout, and ONE PoolTimeout sent the whole answer to the lexical fallback.
        # Measured on the AEGCL corrigendum: a 377-claim answer failed this way every time, so
        # FR-E4 on the v2 aggregate path was never actually running NLI — and said so only in a
        # warning log. The bound costs nothing on short answers (fewer sentences than the limit
        # run exactly as before) and is what makes long ones work at all.
        gate = asyncio.Semaphore(self._max_concurrency)

        async def bounded(sentence: str) -> float:
            async with gate:
                return await self._support_for(sentence, usable)

        try:
            # Score each sentence against the passages that could actually support it, NOT against
            # every retrieved span concatenated. Measured live: the identical answer scored 1.0
            # against 8 spans and 0.0 against the same 8 plus 10 graph-augmented neighbours — the
            # evidence was present and in the same position both times, but a cross-encoder given
            # a wall of mostly-irrelevant text loses the signal. That made recall and faithfulness
            # fight each other: improving retrieval lowered the quality score.
            probs = await asyncio.gather(*(bounded(sent) for sent in sentences))
        except (RetrievalError, httpx.HTTPError) as exc:
            # A VERIFICATION FAILURE IS AN OUTCOME, NOT AN EXCEPTION. Raising here made the
            # caller decide what a failure meant, which is how it became an answer-shaped result
            # somewhere far from this module -- and taking the fallback's verdict verbatim made a
            # LEXICALLY scored batch indistinguishable from an NLI-verified one. Measured in a
            # real run: one of 13 batch verifications was scored by the fallback after a
            # ReadError, and the answer reported it as verified.
            #
            # The fallback's SCORE still stands when it runs; only the claim about what produced
            # it changes. When nothing can score the answer the verdict carries no score at all,
            # because a number here would be an assertion about claims nobody checked.
            detail = f"{type(exc).__name__}: {exc}"[:300]
            if self._fallback is None:
                _log.warning("nli_scorer_unavailable_no_fallback", error=detail[:200])
                return FaithfulnessVerdict(
                    score=None,
                    faithful=None,
                    method="unavailable",
                    degraded=True,
                    degraded_reason="nli_failed_fallback_unconfigured",
                    degraded_detail=detail,
                )
            try:
                fallback = await self._fallback.check(answer, spans, query)
            except Exception as fallback_exc:
                # TWO distinct unavailable causes, kept apart because they call for different
                # actions: this one means two services are down and someone is paged, where an
                # unconfigured fallback is a deployment choice and a config review.
                _log.warning(
                    "nli_scorer_and_fallback_both_failed",
                    error=detail[:200],
                    fallback_error=f"{type(fallback_exc).__name__}: {fallback_exc}"[:200],
                )
                return FaithfulnessVerdict(
                    score=None,
                    faithful=None,
                    method="unavailable",
                    degraded=True,
                    degraded_reason="nli_failed_fallback_failed",
                    degraded_detail=f"{detail} | fallback: {type(fallback_exc).__name__}: {fallback_exc}"[:300],
                )
            _log.warning("nli_scorer_failed_fallback_lexical", error=detail[:200])
            return replace(
                fallback,
                method="lexical_fallback",
                degraded=True,
                degraded_reason="nli_failed_fallback_used",
                degraded_detail=detail,
            )
        # Relevance gate (see module comment above _CLAUSE_SPLIT): only engages for a compound
        # query, and only ever moves a sentence NLI already accepted into "unsupported" — it
        # never overrides an NLI-flagged contradiction.
        clauses = _query_clauses(query)
        uncovered = _uncovered_clauses(usable, clauses)
        supported_weight = 0
        total_weight = 0
        unsupported: list[str] = []
        for sent, prob in zip(sentences, probs, strict=True):
            weight = _content_weight(sent)
            total_weight += weight
            entailed = prob >= self._support
            # NLI says only "does this conflict?". A sentence recalled from the model's own
            # memory conflicts with nothing, so it passes — that is how a true-but-unrelated
            # answer earned 1.0 while citing a source that never mentioned it. Require the
            # sentence to also be ANCHORED in some retrieved span's vocabulary.
            if entailed and _lexical_anchor(sent, spans, length_norm=self._length_norm) < self._anchor:
                entailed = False
            if entailed and uncovered and not _is_clause_relevant(sent, clauses, uncovered):
                entailed = False
            if entailed:
                supported_weight += weight
            else:
                unsupported.append(sent)
        score = supported_weight / total_weight if total_weight else 1.0
        return FaithfulnessVerdict(
            score=round(score, 4),
            faithful=not unsupported,
            unsupported_claims=unsupported,
            method="nli",
        )


class TEINLIScorer:
    """Real entailment scorer: a self-hosted TEI ``/predict`` NLI cross-encoder.

    TEI serves a sequence-classification model and returns per-label scores for each
    (text, text_pair). We read the ``entailment`` label's probability. Sovereign: the
    endpoint host must be egress-allowlisted, same as the reranker/embedder.
    """

    def __init__(self, url: str | None = None, *, model: str | None = None, timeout: float | None = None) -> None:
        settings = get_settings()
        self._url = (url or settings.nli_url).rstrip("/")
        assert_sovereign(self._url)
        self._model = model or settings.nli_model
        # THE BOUND LIVES HERE, not only in the checker, because this object owns the connection
        # pool and is an `lru_cache`d singleton — so it is the only place that sees ALL the NLI
        # traffic in the process. Bounding per `check()` call was not enough: measured against a
        # live run, four faithfulness checks were in flight at once (the aggregate answer plus the
        # numeric-grounding wrapper), each honouring its own 16 and together holding 65 sockets,
        # which exhausted the pool exactly as the unbounded version had. A caller-side bound can
        # always be multiplied by another caller; a pool-side bound cannot.
        #
        # The pool is sized TO the semaphore so the two can never disagree: a request that holds a
        # slot is guaranteed a connection, so `PoolTimeout` is unreachable and the per-call
        # timeout measures the model rather than the queue in front of it. Waiting on the
        # semaphore costs nothing against that timeout, because the request is not made until the
        # slot is held.
        self._gate = asyncio.Semaphore(settings.nli_max_concurrency)
        self._client = httpx.AsyncClient(
            base_url=self._url,
            timeout=timeout or settings.nli_timeout_seconds,
            verify=model_tls_verify(),
            limits=httpx.Limits(max_connections=settings.nli_max_concurrency),
        )

    async def __call__(self, premise: str, hypothesis: str) -> float:
        async with self._gate:
            return await self._predict(premise, hypothesis)

    async def _predict(self, premise: str, hypothesis: str) -> float:
        try:
            # TEI NLI pair classification: the premise+hypothesis go as a two-element `inputs`
            # array (the `text_pair` form is mis-paired by TEI → spurious 'neutral'). mDeBERTa
            # scores cross-lingual pairs correctly (Hindi premise, English/Hindi hypothesis).
            resp = await self._client.post("/predict", json={"inputs": [premise, hypothesis]})
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise RetrievalError(
                f"NLI checker at {self._url} did not answer /predict ({type(exc).__name__}: {exc})",
                details={"url": self._url},
            ) from exc
        return _support_prob(resp.json())


def _support_prob(payload: object) -> float:
    """Pull P(supported) = 1 - P(contradiction) from a TEI /predict response.

    An abstractive RAG answer synthesises the sources, so a FAITHFUL sentence is usually scored
    'neutral' (consistent but not verbatim-entailed), not 'entailment' — using P(entailment)
    makes good answers abstain. The signal NLI gives SHARPLY is CONTRADICTION: a hallucination
    that conflicts with the sources scores high there. So faithfulness = 'not contradicted'. This
    pairs with the upstream retrieval gate (which handles off-topic relevance) and the numeric-
    grounding wrapper (which catches invented figures). Falls back to P(entailment) if the model
    exposes no contradiction label.

    TEI returns a list of ``{"label","score"}`` objects (optionally nested one level for a
    batch of one). Labels vary by model — matched case-insensitively.
    """
    rows = payload
    if isinstance(rows, list) and rows and isinstance(rows[0], list):
        rows = rows[0]  # batch-of-one nesting
    if not isinstance(rows, list):
        raise ConfigError("unexpected NLI /predict response shape", details={"payload": str(payload)[:160]})
    contradiction: float | None = None
    entailment: float | None = None
    for item in rows:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).lower()
        score = max(0.0, min(1.0, float(item.get("score", 0.0))))
        if label in {"contradiction", "contradict", "label_2"}:
            contradiction = score
        elif label in {"entailment", "entail", "label_0"}:
            entailment = score
    if contradiction is not None:
        return 1.0 - contradiction
    if entailment is not None:
        return entailment
    # Neither label present → cannot judge; treat as supported (the retrieval gate + numeric
    # check still guard grounding) rather than force a good answer to abstain.
    return 1.0


def build_nli_faithfulness_checker() -> NLIFaithfulnessChecker:
    """Factory for the NLI checker wired to the real TEI scorer (reads config for url/model).

    The lexical checker is injected as the fail-soft fallback: if the CPU NLI endpoint times out
    or is unreachable, faithfulness degrades to deterministic overlap rather than 502-ing the answer.
    """
    from rag_clients.faithfulness import LexicalFaithfulnessChecker  # noqa: PLC0415 — avoid import cycle

    settings = get_settings()
    return NLIFaithfulnessChecker(
        TEINLIScorer(),
        support_threshold=settings.nli_support_threshold,
        fallback=LexicalFaithfulnessChecker(),
    )
