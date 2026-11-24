"""Strict, ADAPTIVE completeness guard — pure logic (no I/O).

A grounded answer can be partial in two ways, and neither may ship silently:

* **Truncation-partial** — the completion was cut off by the token budget. Authoritative signal:
  the gateway's ``finish_reason == "length"``; cheap structural fallback: the answer ends
  mid-sentence or on a dangling enumeration (it was numbering ``1. … 2. …`` and stopped without a
  terminal). :func:`is_truncated` / :func:`looks_truncated`.
* **Coverage-partial** — grounded but a distinct, clearly-relevant retrieved point was omitted
  (:func:`~rag_core.coverage.uncovered_spans`, reused by the caller's bounded loop).

Everything here is ADAPTIVE — derived per query / per KB from evidence and signals, never a bare
literal in the hot path. :func:`continuation_bound` derives HOW MANY continuations to allow from the
query's own demand (capped by a per-KB Settings ceiling); :func:`escalated_budget` grows the token
budget per attempt (capped at the heavy budget); :func:`expected_coverage` reads the "expected" point
count off the retrieved spans actually above the relevance bar for THIS query (reusing
``coverage_min_score`` semantics), not a fixed target. :func:`build_completeness` assembles the
non-authoritative :class:`~rag_schemas.Completeness` signal the caller attaches to the answer.

This module makes ZERO I/O and ZERO accept/reject decisions — the caller owns the bounded loops and
the faithfulness gate; here we only detect, size the bounds, and describe the result.
"""

from __future__ import annotations

import re

from rag_core.coverage import uncovered_spans
from rag_core.grounding import _content_terms
from rag_schemas import Completeness, RetrievedSpan

_FINISH_LENGTH = "length"

# Characters a COMPLETE sentence/enumeration item legitimately ends on: sentence terminals (Latin +
# Indic danda/double danda), a closing citation/quote/bracket, a bold marker, an inline-code tick,
# a table row's trailing pipe, an ellipsis.
_TERMINATORS = ".!?।॥\"')]}`*|…"  # Indic danda + markdown/quote/table terminators are intentional

# A query that DEMANDS exhaustiveness ("list every …", "all", "each") earns the full continuation
# ceiling; an ordinary question is unlikely to need many continuations, so it gets half. Derived
# from the query, not a fixed N — and each attempt early-exits the moment truncation clears.
_LIST_DEMAND_RE = re.compile(
    r"\b(all|every|each|entire|complete|comprehensive|list|enumerate|itemi[sz]e|full|exhaustive)\b",
    re.IGNORECASE,
)

# A leading enumeration / bullet marker on a line: "1." / "2)" / "-" / "*" / "•". Used to detect a
# DANGLING list item — a marker whose item was never finished — which is the one structural cutoff
# shape safe to flag. Longest algorithmic overlap we bother stitching (below this it is likely a
# coincidental short match, not a re-emitted phrase).
_ENUM_MARKER_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•·])(?:\s|$)")
_MIN_STITCH_OVERLAP = 4
# A cut is treated as MID-TOKEN (join with no space) only when the prior ends on a SHORT trailing
# ALPHABETIC fragment ("Recovery ti"): a longer run is almost always a complete word, so a missing
# space there would wrongly fuse two words ("time" + "objective" → "timeobjective"), and a digit
# tail ("... 24" + "days") is a complete number, not a split token.
_MID_TOKEN_MAX = 3


def looks_truncated(text: str) -> bool:
    """Cheap structural heuristic: does ``text`` read as cut off mid-enumeration?

    CONSERVATIVE by design — it is the fallback used on the streaming path where no authoritative
    ``finish_reason`` is available, so it must NOT flag a perfectly good concise answer merely for
    ending without terminal punctuation ("See section 3"). It fires ONLY when the last non-empty
    line is an enumeration / bullet marker whose item is DANGLING: a bare marker with no content, or
    a started list item that ends without a terminator. Empty text is NOT truncated (that is the
    empty-generation case, handled elsewhere)."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    last = lines[-1].rstrip()
    marker = _ENUM_MARKER_RE.match(last)
    if marker is None:
        return False  # a non-list last line — do not flag a concise answer just for missing "."
    content = last[marker.end() :].strip()
    return not content or content[-1] not in _TERMINATORS


def stitch_continuation(prior: str, cont: str) -> str:
    """Join a truncated answer to its continuation, de-duplicating a re-emitted overlap and never
    seaming mid-token.

    Models re-emit despite "do not repeat", so strip the longest suffix of ``prior`` that ``cont``
    repeats as its head (``_MIN_STITCH_OVERLAP``+ chars, to avoid a coincidental short match). When
    the cut fell MID-WORD ("Recovery ti" + "me…") both sides are alphanumeric at the seam, so join
    with no space; otherwise a single space."""
    prior = prior.rstrip()
    cont = cont.strip()
    if not cont:
        return prior
    if not prior:
        return cont
    overlap = 0
    for k in range(min(len(prior), len(cont)), _MIN_STITCH_OVERLAP - 1, -1):
        if prior[-k:].lower() == cont[:k].lower():
            overlap = k
            break
    body = cont[overlap:].lstrip()
    if not body:
        return prior
    trailing_alpha = 0
    for ch in reversed(prior):
        if ch.isalpha():
            trailing_alpha += 1
        else:
            break
    mid_token = prior[-1].isalpha() and body[0].isalpha() and 0 < trailing_alpha <= _MID_TOKEN_MAX
    sep = "" if mid_token else " "
    return f"{prior}{sep}{body}"


class GuardBudget:
    """Shared, mutable cap on the TOTAL extra guard-driven LLM calls across ONE request.

    A single instance is threaded through every ``GenerationService.answer`` call of a request —
    including the per-doc fan-out of a doc-summary — so continuation + coverage passes summed across
    docs can never exceed the ceiling. :meth:`take` reserves one call and returns whether it was
    granted; when exhausted the guard stops looping and marks the answer incomplete."""

    def __init__(self, remaining: int) -> None:
        self.remaining = max(0, remaining)

    def take(self) -> bool:
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True


def is_truncated(text: str, finish_reason: str | None) -> bool:
    """True if the completion was cut off by the token budget.

    ``finish_reason == "length"`` is authoritative (the server says it stopped for length);
    otherwise fall back to the structural heuristic so a gateway that does not report a reason
    (or a stream) still catches an obvious mid-sentence cutoff."""
    if finish_reason is not None and finish_reason.strip().lower() == _FINISH_LENGTH:
        return True
    return looks_truncated(text)


def continuation_bound(query_text: str, *, cap: int) -> int:
    """How many bounded continuation attempts to allow for THIS query, in ``[1, cap]``.

    ``cap`` is the per-KB Settings ceiling (never a bare literal here). The actual bound is DERIVED
    from the query's demand: an exhaustive "list every point" question is the one that genuinely
    needs several continuations, so it gets the full ceiling; an ordinary question gets half
    (rounded up, never below 1). Every continuation also early-exits as soon as the answer is no
    longer truncated, so this only ever bounds a genuinely still-cut-off answer."""
    cap = max(1, cap)
    if _LIST_DEMAND_RE.search(query_text):
        return cap
    return max(1, (cap + 1) // 2)


def escalated_budget(base_budget: int, attempt: int, *, factor: float, cap: int) -> int:
    """The token budget for continuation ``attempt`` (0-based), escalated adaptively.

    Grows the base budget by ``factor`` compounded per attempt so a still-truncating answer gets
    progressively more room, bounded by ``cap`` (the heavy-path budget) so it can never run away.
    ``base_budget`` is the budget the first generation used (the caller resolves it — e.g. the
    request's ``max_tokens`` or the configured default), so the escalation is relative to what was
    actually produced, not an absolute magic number."""
    base = base_budget if base_budget > 0 else max(1, cap)
    grown = int(base * (factor ** (attempt + 1)))
    bounded = min(grown, cap) if cap > 0 else grown
    return max(base, bounded)


def expected_coverage(spans: list[RetrievedSpan], *, min_score: float) -> int:
    """The ADAPTIVE count of relevant points a complete answer is expected to reflect for THIS query.

    Exactly the retrieved spans the reranker scored ``>= min_score`` (clearly relevant, reusing the
    ``coverage_min_score`` bar) AND that carry salient content terms — i.e. the same population
    :func:`~rag_core.coverage.uncovered_spans` measures omission against. Not a fixed target
    count: it rises and falls with the evidence actually retrieved above the bar."""
    return sum(1 for s in spans if s.score >= min_score and _content_terms(s.text))


def build_completeness(
    answer_text: str,
    spans: list[RetrievedSpan],
    *,
    min_score: float,
    coverage_ratio: float,
    max_spans: int,
    truncated: bool,
    measure_coverage: bool = True,
    exhausted: bool = False,
) -> Completeness:
    """Assemble the non-authoritative :class:`~rag_schemas.Completeness` signal for ``answer_text``.

    ``expected`` is the per-query relevant-point count (:func:`expected_coverage`); the number STILL
    omitted is measured by :func:`~rag_core.coverage.uncovered_spans` and ``covered =
    expected - uncovered``. The uncovered count is measured UNCAPPED (``max_spans`` bounds only how
    many the re-prompt targets, not the true count), so ``covered``/``expected`` never over-report
    when the real omission exceeds the re-prompt cap. The answer is ``complete`` iff it is neither
    still truncated nor still missing a relevant point; otherwise ``reason`` explains which, so the
    console can surface "answer may be incomplete" rather than shipping a partial answer.

    ``measure_coverage=False`` (the doc-summary path, where omission is INTENTIONAL) reports a
    truncation-only signal — coverage is meaningless for a deliberately-lossy summary. ``exhausted``
    marks that the per-request guard budget ran out before the answer could be finished."""
    if measure_coverage:
        expected = expected_coverage(spans, min_score=min_score)
        # Count the TRUE uncovered total, not the re-prompt-capped list: cap the counting at the
        # span count so `max_spans` (a re-prompt batch size) can never make covered/expected lie.
        uncovered = len(
            uncovered_spans(
                answer_text, spans, min_score=min_score, coverage_ratio=coverage_ratio, max_spans=len(spans) or 1
            )
        )
    else:
        expected, uncovered = 0, 0
    covered = max(0, expected - uncovered)
    # Exhausting the per-request guard budget means the guard bailed with work possibly remaining, so
    # never claim completeness in that case (defensive — it coincides with truncated/uncovered anyway).
    complete = (not truncated) and uncovered == 0 and not exhausted
    reason: str | None = None
    if not complete:
        parts: list[str] = []
        if truncated:
            parts.append("answer was cut off by the token budget after the bounded continuation attempts")
        if uncovered:
            parts.append(f"{uncovered} clearly-relevant retrieved point(s) still not covered")
        if exhausted:
            parts.append("the completeness guard budget was exhausted before the answer could be completed")
        reason = "; ".join(parts) or "guard_budget_exhausted"
    return Completeness(complete=complete, reason=reason, covered=covered, expected=expected, truncated=truncated)


# --- Tier 3: substantive-passage awareness (whole-document coverage verification) -----------------
#
# The whole-document read (``pipeline._answer_document_full_summary``) used to measure completeness
# against EVERY chunk of a file. On a real SOP corpus a 181-chunk document that WAS covered correctly
# (all ten SOPs cited) still reported "107 of 181 passages not reflected", because the count included
# BOILERPLATE: page footers, "Page 1 of 1" markers, fax/phone letterhead lines, the SOP master-listing
# table rows, revision-history scaffolding and duplicate headers. None of those carry an obligation a
# summary must reflect, so counting them as "uncovered" makes an accurate summary look badly
# incomplete AND sends the coverage-refine loop chasing content that does not exist.
# :func:`passage_is_substantive` filters them out so completeness measures only passages that carry
# real document content, and the refine loop chases only genuinely-missed obligations.
#
# CONSERVATIVE by design (the same principle as :func:`looks_truncated`): a passage is treated as
# non-substantive ONLY when it clearly matches a known boilerplate shape. Anything ambiguous stays
# SUBSTANTIVE — the cost of hiding a real obligation (silently dropping it from the coverage measure)
# is far worse than the cost of counting one boilerplate line, so every rule below errs toward "keep".

# The minimum content-word count a passage needs to POSSIBLY be substantive. Below it a passage is a
# bare marker / heading / label ("Annexure A", a lone section number, a page number) — never an
# obligation. Kept small so a genuinely short sentence is not hidden; only tiny fragments fall under it.
_MIN_SUBSTANTIVE_WORDS = 3

# A pure page marker: "Page 1", "Page 1 of 12", "1 of 1", "- 3 -", "[Page 2]". Structural navigation
# that repeats on every page — it tells a reader WHERE they are, never WHAT the document requires.
_PAGE_MARKER_RE = re.compile(r"^[\[\(\-\s]*(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?[\]\)\-.\s]*$", re.IGNORECASE)

# Contact / letterhead cues. A footer that is ONLY an address, phone, fax, email or URL is the
# organisation's letterhead reprinted on every page, not a requirement — so a line dominated by these
# cues, with little real prose once they and the digits are removed, is boilerplate (rule 5 below).
_CONTACT_CUE_RE = re.compile(r"^(tel|telephone|phone|fax|e-?mail|mob(?:ile)?|www|http|contact)$", re.IGNORECASE)
_HAS_CONTACT_CUE_RE = re.compile(r"\b(tel|telephone|phone|fax|e-?mail|mob(?:ile)?|www|http|contact)\b", re.IGNORECASE)

# A table-of-contents leader: "Introduction .......... 3" — a dotted leader ending in a page number.
_TOC_LEADER_RE = re.compile(r"\.{3,}\s*\d+\s*$")

# An SOP / document identifier code: "PBSOP001", "SOP-0034", "QA-SOP-12". The SOP MASTER LIST
# reproduces every SOP's id + title + revision as index rows; each such row read as a separate
# obligation is exactly what made a fully-covered summary look 60% incomplete on the live corpus.
_DOC_CODE_RE = re.compile(r"\b[A-Z]{2,}[A-Z/ -]*\d{2,}\b")

# Revision-history / document-control scaffolding: the header of the change-history table, or a row
# within it. "Rev 3 | 2024-01-01 | Updated section 4" is metadata ABOUT the document, not its content.
_REVISION_HEADER_RE = re.compile(
    r"^\s*(?:revision|change|version|document)\s+(?:history|control|log|no\.?|number)\b", re.IGNORECASE
)
_REV_ROW_RE = re.compile(r"\brev(?:ision)?\.?\s*(?:no\.?)?\s*\d+\b", re.IGNORECASE)

# Email addresses and URLs — stripped before counting a contact line's "real prose" so a letterhead
# footer's domain parts ("info", "example", "com") are not mistaken for content words.
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|org|net|in|co|gov|edu|io)\b", re.IGNORECASE)

# Words = runs of >=2 letters, Latin + common Indic ranges (so a Hindi/Tamil passage is not scored as
# "too short" merely because a Latin-only \w+ found nothing salient in it).
_SUBSTANTIVE_WORD_RE = re.compile(r"[A-Za-zÀ-ɏऀ-෿]{2,}")

# A delimited row needs at least this many pipe separators to read as a table/index row.
_MIN_ROW_PIPES = 2

# Obligation / directive cues. A passage carrying ANY of these states a requirement — it is REAL
# content that a summary must reflect, so it short-circuits to substantive BEFORE any boilerplate
# rule can drop it. This is the strongest "never hide a real obligation" guard: an obligations- or
# requirements-register ROW ("REQ-101 | Data must be encrypted at rest | Rev 2") looks exactly like
# a master-listing index row (doc code + revision token) and would otherwise be dropped by rule 4,
# silently excluding a real obligation from the coverage measure. A master-list INDEX row carries a
# title but no such verb, so it is still correctly dropped. (Also keeps a short cued obligation like
# "PPE must be worn." above the too-short floor.) A false positive here only KEEPS more as
# substantive — the safe direction.
_OBLIGATION_CUE_RE = re.compile(
    r"\b(?:"
    r"must(?:\s+not)?|shall(?:\s+not)?|may\s+not|may\s+only|"
    r"required|requires?|responsible|prohibited|mandatory|obligation|"
    r"(?:is|are)\s+to"
    r")\b",
    re.IGNORECASE,
)

# The most title-like a verb-less delimited row's residual prose (after removing its doc-code and
# revision cells) may be while still counting as a pure INDEX row. A LONGER descriptive row is not an
# index — belt-and-suspenders alongside the obligation-verb short-circuit so a real, verb-less
# descriptive table row is never mistaken for a master-listing entry.
_INDEX_ROW_MAX_TITLE_WORDS = 8


def passage_is_substantive(text: str) -> bool:  # noqa: PLR0911 - one guarded return per boilerplate shape
    """True when a passage carries real document content; False for structural boilerplate.

    Used by the whole-document coverage measure to exclude the page markers, letterhead footers,
    table-of-contents / SOP master-listing rows, revision-history scaffolding and tiny label lines
    that a summary is not expected to reflect. An OBLIGATION/directive cue (must/shall/required/…)
    short-circuits to substantive FIRST so a real requirement is never dropped by a boilerplate rule;
    beyond that every rule fires only on a CLEAR boilerplate shape, and an ambiguous passage is
    reported SUBSTANTIVE so real content is never hidden from the measure.
    """
    stripped = text.strip()
    if not stripped:
        return False  # empty / whitespace-only — nothing to cover
    single_line = " ".join(stripped.split())

    # 0. NEVER-MISS short-circuit: a passage stating an obligation is real content, full stop — it is
    #    kept before any boilerplate rule can drop it (guards obligations-register rows that share the
    #    doc-code + revision shape of a master-listing index row).
    if _OBLIGATION_CUE_RE.search(single_line):
        return True

    # 1. A pure page marker — navigation, never an obligation.
    if _PAGE_MARKER_RE.match(single_line):
        return False

    words = _SUBSTANTIVE_WORD_RE.findall(single_line)
    # 2. Too short to be an obligation: a bare heading / label / section number. Conservative floor,
    #    so a genuine short sentence is never hidden — only tiny fragments fall under it.
    if len(words) < _MIN_SUBSTANTIVE_WORDS:
        return False

    # 3. Revision-history / document-control scaffolding header ("Revision History", "Document Control").
    if _REVISION_HEADER_RE.match(single_line):
        return False

    # 4. A table-of-contents / master-listing row. A dotted-leader TOC entry is unambiguous. A
    #    delimited row (pipes, tabs, or wide column gaps) is flagged ONLY when it ALSO carries BOTH a
    #    document code AND a revision token — the SOP master-listing shape ("PBSOP001 | Title | Rev 3")
    #    — so a genuine data table of requirements (no doc-id/rev columns) is NOT mistaken for an index.
    if _TOC_LEADER_RE.search(single_line):
        return False
    looks_like_row = (
        single_line.count("|") >= _MIN_ROW_PIPES or "\t" in stripped or re.search(r"\s{3,}", stripped) is not None
    )
    if looks_like_row and _DOC_CODE_RE.search(single_line) and _REV_ROW_RE.search(single_line):
        # A genuine index row is code + revision + a SHORT title; a long descriptive (verb-less) row
        # is not an index, so keep it. (An obligation-carrying row already returned True at rule 0.)
        title_words = _SUBSTANTIVE_WORD_RE.findall(_REV_ROW_RE.sub(" ", _DOC_CODE_RE.sub(" ", single_line)))
        if len(title_words) <= _INDEX_ROW_MAX_TITLE_WORDS:
            return False

    # 5. A footer that is ONLY contact/letterhead detail: few real words once the emails/URLs, the
    #    contact cue tokens and the digits (already excluded from `words`) are removed.
    if _HAS_CONTACT_CUE_RE.search(single_line):
        residual = _URL_RE.sub(" ", _EMAIL_RE.sub(" ", single_line))
        prose = [w for w in _SUBSTANTIVE_WORD_RE.findall(residual) if not _CONTACT_CUE_RE.match(w)]
        if len(prose) < _MIN_SUBSTANTIVE_WORDS:
            return False

    return True


def substantive_coverage(
    passages: list[tuple[str, str]],
    cited_ids: set[str],
) -> tuple[int, int, list[str]]:
    """Coverage over SUBSTANTIVE passages only: ``(covered, expected, uncovered_substantive_ids)``.

    ``passages`` is ``(chunk_id, text)`` in document order (the whole document's chunks). A passage
    counts toward ``expected`` only if it is substantive (:func:`passage_is_substantive`) AND is not a
    verbatim duplicate of its immediate predecessor (a header/footer reprinted on consecutive pages —
    counting each copy would inflate the obligation count). ``covered`` is the substantive passages
    whose ``chunk_id`` is in ``cited_ids``; ``uncovered_substantive_ids`` names the rest, in document
    order, so the caller can refine coverage over EXACTLY the genuinely-missed obligations. Boilerplate
    passages appear in NEITHER count, so a fully-covered document reports ``covered == expected`` even
    when most of its chunks are markers/footers, and the refine loop never chases a boilerplate line.
    """
    covered = 0
    expected = 0
    uncovered: list[str] = []
    prev_norm: str | None = None
    for chunk_id, text in passages:
        norm = " ".join(text.lower().split())
        if not passage_is_substantive(text) or norm == prev_norm:
            prev_norm = norm
            continue
        prev_norm = norm
        expected += 1
        if chunk_id in cited_ids:
            covered += 1
        else:
            uncovered.append(chunk_id)
    return covered, expected, uncovered
