"""Semantic query-intent classification — pure logic, no framework, no IO.

A question's INTENT decides which answer shape the pipeline produces, BEFORE retrieval:

* ``catalog``   — "what documents / files / artifacts / records are in this KB?" → answer
  from the KB inventory (NAMES only), not an out-of-scope abstain.
* ``doc_summary`` — "summarize / explain / describe EACH or ALL documents" → a per-document
  grounded summary of every doc's CONTENTS (not just their names). Distinct from ``catalog``:
  a CONTENT verb (summarize/explain/describe/"in detail"/…) about the documents is NOT a
  bare inventory request.
* ``summarize`` — "summarize the chat / recap this conversation / tl;dr" → summarize the
  conversation (only meaningful WITH history).
* ``count``     — "how many documents / number of files" → the KB size (catalog-with-a-number).
* ``filter``    — "which documents from 2026 / policies of type agreement / latest docs" → the
  inventory narrowed by a criterion the handler extracts heuristically (year/type/recency).
* ``compare``   — "difference between A and B / A vs B" → a grounded contrast of two named docs.
* ``rag``       — everything else → normal grounded retrieval (the safe default).

The LLM classifier returns exactly ONE label; any PARAMETERS (a filter's criteria, a compare's
two target documents) are extracted heuristically by the handlers from the pure helpers here
(:func:`extract_filter_criteria`, :func:`match_compare_targets`) — the classifier never returns
structured JSON.

Two backends behind one seam (mirroring ``routing.py``): the deterministic
:func:`classify_intent_heuristic` keyword FALLBACK, and the real semantic path — an LLM
prompted by :func:`build_intent_prompt` and parsed by :func:`parse_intent` (the adapter
lives in ``rag_clients``). Both are flag-gated and default OFF, so with intent routing
disabled the pipeline is byte-identical to today.
"""

from __future__ import annotations

import re
from typing import Literal

Intent = Literal["catalog", "doc_summary", "summarize", "count", "filter", "compare", "rag"]

_ALLOWED: tuple[Intent, ...] = (
    "catalog",
    "doc_summary",
    "summarize",
    "count",
    "filter",
    "compare",
    "rag",
)

# --- heuristic fallback signals (deliberately simple; the LLM path is the real semantic one) ---
# Single-word markers matched on word boundaries (so "profile" never matches "file").
_CATALOG_WORDS = frozenset(
    {
        "document",
        "documents",
        "file",
        "files",
        "artifact",
        "artifacts",
        "record",
        "records",
        "upload",
        "uploads",
        "doc",
        "docs",
        "inventory",
        "catalog",
        "catalogue",
    }
)
# Phrase markers matched as substrings.
_CATALOG_PHRASES = (
    "what do we have",
    "what's in",
    "whats in",
    "what is in",
    "list the",
    "show me the",
    "what documents",
    "how many documents",
)
# doc_summary — a CONTENT/SYNTHESIS verb about the documents' CONTENTS (not their names). These
# distinguish "summarize each doc" (doc_summary) from "what documents do we have" (catalog): a
# content verb means the query is NOT a bare inventory request. Matched as substrings ("summariz"
# also catches "summarize/summarizing"; "summaris" the British spelling).
_CONTENT_VERBS = (
    "summariz",
    "summaris",
    # The NOUN forms. Only the verb stems were here, so "give me a summary of every document" and
    # "detailed summary analysis of the 10 SOPs" never reached this guard at all and fell through
    # to catalog — the first was answered with a list of filenames, the second by trying to
    # summarise the knowledge base's three files. A request for a summary is a request for
    # CONTENTS however it is phrased.
    "summary",
    "analysis",
    "analyse",
    "analyz",
    "explain",
    "describe",
    "description",
    "in detail",
    "tell me about",
    "walk me through",
    "walk through",
    "go through",
    "overview of",
    "what does each",
    "what do each",
    "what does every",
    "what do the documents say",
    "what's in each",
    "whats in each",
)
# A doc_summary needs a PLURAL/collective document reference — the content verb alone (about a
# single topic) is plain rag. Plural doc nouns, or each/every/all APPLIED TO a doc word.
_PLURAL_DOC_WORDS = frozenset({"documents", "files", "artifacts", "records", "uploads", "docs"})
_PLURAL_DOC_PHRASES = ("the documents", "the docs", "every document", "all documents")
# "each/every/all" must actually GOVERN the document word, not merely appear in the same sentence.
# Reported live: "summary analysis for ALL the 10 SOPs mentioned in THE DOCUMENT from PBSOP001..."
# was read as "all documents" and answered by trying to summarise the three files in the knowledge
# base — "all" belonged to the SOPs and "the document" was singular, one file the SOPs live inside.
# One intervening word is allowed so "all the uploaded files" and "each of the documents" match.
_EACH_ALL_DOC = re.compile(
    r"\b(?:each|every|all|entire)\s+(?:of\s+)?(?:the\s+)?(?:\w+\s+)?"
    r"(?:documents?|files?|docs?|artifacts?|records?|uploads?)\b",
    re.IGNORECASE,
)

# "recap"/"tl;dr" alone read as summarize; a summarize VERB needs a conversation anchor.
_SUMMARIZE_STRONG = ("recap", "tl;dr", "tldr")
_SUMMARIZE_VERB = ("summariz", "summarise", "summary")
_CONVERSATION_WORDS = frozenset({"chat", "conversation", "thread", "discussion"})

# count — "how many documents / number of files": catalog-with-a-number. Matched as substrings
# so "how many documents are uploaded" and "the number of files" both fire.
_COUNT_PHRASES = (
    "how many document",
    "how many file",
    "how many doc",
    "how many record",
    "how many artifact",
    "how many upload",
    "number of document",
    "number of file",
    "number of doc",
    "number of record",
    "number of artifact",
    "count of document",
    "count of file",
    "count of doc",
    "count of record",
)

# filter — an explicit "which documents … / what changed …" lead OR a catalog word narrowed by a
# criterion (see extract_filter_criteria). Kept conservative: a bare doc_type word in an ordinary
# question ("what is the leave policy?") never fires, because filter needs a doc word + a criterion.
_FILTER_LEAD = (
    "which document",
    "which file",
    "which doc",
    "which record",
    "which artifact",
    "what changed",
    "what has changed",
    "whats changed",
    "what's changed",
    "of type",
)

# compare — an explicit comparison structure. Liberal on purpose: the handler is STRICT (it only
# proceeds when it confidently identifies EXACTLY two target documents, else falls back to rag), so
# a broad classifier signal here can never fabricate a comparison.
_COMPARE_RE = re.compile(r"\b(compare|compared|comparison|contrast|contrasted|versus|vs\.?|difference|differences)\b")

# Coarse doc-type tokens → the classify_doc_type classes (tagging.py). Kept as a local, dependency-
# free copy so intent.py stays pure. Plurals and near-synonyms map to the same class.
_DOC_TYPE_TOKENS: dict[str, str] = {
    "agreement": "agreement",
    "agreements": "agreement",
    "contract": "agreement",
    "contracts": "agreement",
    "policy": "policy",
    "policies": "policy",
    "report": "report",
    "reports": "report",
    "invoice": "invoice",
    "invoices": "invoice",
    "manual": "manual",
    "manuals": "manual",
    "guide": "manual",
    "guides": "manual",
    "resume": "resume",
    "resumes": "resume",
    "cv": "resume",
    "minutes": "minutes",
}

_MIN_TOKEN_LEN = 2  # tokens shorter than this are dropped from compare-target matching
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_LAST_N_DAYS_RE = re.compile(r"last\s+(\d+)\s+day")
# Default window for a bare "recent/latest" with no explicit number.
_DEFAULT_RECENCY_DAYS = 30

# Content-token stopwords for compare target matching — dropped so a document's distinctive label
# tokens drive the overlap, not connective words present in every query.
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "from",
        "with",
        "that",
        "this",
        "what",
        "which",
        "are",
        "was",
        "does",
        "between",
        "difference",
        "differences",
        "compare",
        "compared",
        "comparison",
        "contrast",
        "vs",
        "versus",
        "of",
        "to",
        "is",
        "in",
        "on",
        "an",
        "or",
        "how",
        "many",
        "do",
        "we",
        "have",
        "list",
        "show",
        "me",
        "us",
        "a",
        "as",
        "at",
        "by",
        "it",
        "be",
        "vs.",
    }
)
# A document is a confident compare target when at least this fraction of its distinctive label
# tokens appear in the query (and at least one does).
_COMPARE_MATCH_MIN = 0.5


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


def _content_tokens(text: str) -> set[str]:
    """Distinctive tokens of ``text`` for fuzzy label matching: lowercased alphanumerics of length
    >= 2, with connective stopwords dropped (so 'HR Handbook' → {hr, handbook})."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= _MIN_TOKEN_LEN and w not in _STOPWORDS}


def _match_doc_type(text: str) -> str | None:
    """The coarse doc_type a query names (first match on a word boundary), or None."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    for token, cls in _DOC_TYPE_TOKENS.items():
        if token in words:
            return cls
    return None


def _recency_days(text: str) -> int | None:
    """A recency window in days a query names, or None. 'last N days' → N; week/month phrases →
    7/30; a bare 'recent/latest/recently' → the default window."""
    m = _LAST_N_DAYS_RE.search(text)
    if m:
        return int(m.group(1))
    if any(p in text for p in ("this week", "last week", "past week")):
        return 7
    if any(p in text for p in ("this month", "last month", "past month")):
        return 30
    if any(w in text for w in ("recent", "recently", "latest")):
        return _DEFAULT_RECENCY_DAYS
    return None


def extract_filter_criteria(query: str) -> dict[str, str]:
    """Heuristically parse a ``filter`` query's criteria — pure, no IO. Returns any of:

    * ``year``         — a 4-digit ``19|20\\d\\d`` (matched against a doc's ``year``/``doc_date``);
    * ``doc_type``     — a coarse class the query names (agreement/policy/report/…);
    * ``recency_days`` — a time window in days ("recent/latest", "last N days", "this week/month").

    Returns ``{}`` when nothing parses — the handler then falls back to ``catalog`` (list all)
    rather than abstaining, since an unparseable filter is not a no-match."""
    text = query.lower()
    criteria: dict[str, str] = {}
    year = _YEAR_RE.search(query)
    if year:
        criteria["year"] = year.group(0)
    doc_type = _match_doc_type(text)
    if doc_type:
        criteria["doc_type"] = doc_type
    days = _recency_days(text)
    if days is not None:
        criteria["recency_days"] = str(days)
    return criteria


def score_compare_targets(query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float, int]]:
    """Score the documents a query references — pure, no IO. ``candidates`` is ``(doc_id, label)``
    for the caller's VISIBLE documents; returns ``(doc_id, overlap, matched_count)`` for those whose
    label is confidently referenced (case-insensitive token overlap against label + doc_id),
    sorted best-first. Callers that need to detect a TIE vs a strict winner read the ``overlap``.
    """
    q_tokens = _content_tokens(query)
    if not q_tokens:
        return []
    scored: list[tuple[str, float, int]] = []
    for doc_id, label in candidates:
        label_tokens = _content_tokens(label) | _content_tokens(doc_id)
        if not label_tokens:
            continue
        matched = label_tokens & q_tokens
        overlap = len(matched) / len(label_tokens)
        if matched and overlap >= _COMPARE_MATCH_MIN:
            scored.append((doc_id, overlap, len(matched)))
    scored.sort(key=lambda s: (s[1], s[2]), reverse=True)
    return scored


def match_compare_targets(query: str, candidates: list[tuple[str, str]]) -> list[str]:
    """Identify the documents a ``compare`` query names — pure, no IO. ``candidates`` is a list of
    ``(doc_id, label)`` for the caller's VISIBLE documents; returns the doc_ids whose label is
    confidently referenced (case-insensitive token overlap against label + doc_id).

    Conservative by design: the caller proceeds only when this returns EXACTLY two ids, so an
    ambiguous or unmatched query falls back to normal retrieval and never fabricates a comparison.
    """
    return [doc_id for doc_id, _, _ in score_compare_targets(query, candidates)]


def classify_intent_heuristic(query: str) -> Intent:  # noqa: PLR0911 - one guarded return per intent label
    """Deterministic, model-free intent — the FALLBACK when the LLM backend is off/unavailable.

    Order matters: summarize is checked before the doc-word intents so "summarize the chat" is not
    swallowed by an unrelated catalog word; compare/count are checked before catalog/filter so a
    comparison or a "how many" is not read as a plain listing; anything unmatched is ``rag``
    (never mis-route)."""
    text = query.lower()
    words = _words(text)
    if any(tok in text for tok in _SUMMARIZE_STRONG):
        return "summarize"
    if any(v in text for v in _SUMMARIZE_VERB) and (bool(words & _CONVERSATION_WORDS) or "so far" in text):
        return "summarize"
    # compare BEFORE the doc-word intents so "compare the two policies" is not swallowed by "policy".
    if _COMPARE_RE.search(text):
        return "compare"
    # count BEFORE catalog/filter: "how many documents" is catalog-with-a-number, not a plain list.
    if any(p in text for p in _COUNT_PHRASES):
        return "count"
    # doc_summary / content-verb guard BEFORE catalog+filter: a CONTENT verb (summarize/explain/
    # describe/"in detail"/…) means the query wants the documents' CONTENTS, so it is NEVER catalog.
    # With a PLURAL/each-all document reference it is doc_summary (summarize every doc); a content
    # verb about a single topic falls through to plain rag (not catalog, not filter).
    if any(v in text for v in _CONTENT_VERBS):
        plural_docs = (
            bool(words & _PLURAL_DOC_WORDS)
            or bool(_EACH_ALL_DOC.search(text))
            or any(p in text for p in _PLURAL_DOC_PHRASES)
        )
        return "doc_summary" if plural_docs else "rag"
    # filter: an explicit "which documents …/what changed …" lead, or a catalog word narrowed by a
    # parseable criterion (year/type/recency). A bare doc_type word alone never triggers it.
    if any(p in text for p in _FILTER_LEAD) or (bool(words & _CATALOG_WORDS) and bool(extract_filter_criteria(query))):
        return "filter"
    if any(p in text for p in _CATALOG_PHRASES) or bool(words & _CATALOG_WORDS):
        return "catalog"
    return "rag"


def build_intent_prompt(query: str, *, has_history: bool) -> str:
    """Prompt the LLM to classify the question's INTENT semantically (near-synonyms count) and
    reply with EXACTLY one label from the allowed set. ``summarize`` is offered only when there
    is a conversation to summarize (``has_history``)."""
    labels = (
        "catalog, summarize, count, filter, compare, doc_summary, rag"
        if has_history
        else "catalog, count, filter, compare, doc_summary, rag"
    )
    summarize_line = (
        '- "summarize" — the user wants a summary/recap/tl;dr of THIS conversation so far.\n'
        '  Examples: "summarize the chat", "recap this conversation", "tl;dr".\n'
        if has_history
        else ""
    )
    return (
        "Classify the INTENT of the QUESTION. Understand near-semantic meaning, not just literal "
        "words (artifacts/files/records/uploads ≈ documents; recap/tl;dr ≈ summarize). Reply with "
        f"EXACTLY ONE word from this set and nothing else: {labels}.\n\n"
        '- "catalog" — the user wants the LIST/INVENTORY of documents — their NAMES only, NOT '
        "their contents.\n"
        '  Examples: "what documents do we have?", "list the files".\n'
        '- "doc_summary" — the user wants to SUMMARIZE/EXPLAIN/DESCRIBE EACH or ALL documents '
        "(their CONTENTS, not just names). The UNIT must be the documents THEMSELVES — one "
        "summary per file in the knowledge base.\n"
        '  Examples: "summarize each doc in detail", "explain every document we have".\n'
        "  NOT doc_summary: a summary of THINGS INSIDE a document — sections, clauses, procedures, "
        'items in a list. "Summarise all ten SOPs in the document" asks about ten things inside '
        "ONE file, so it is 'rag'. The word 'all' belongs to the SOPs, not to the documents.\n"
        f"{summarize_line}"
        '- "count" — the user wants HOW MANY documents/files/records the knowledge base holds.\n'
        '  Examples: "how many documents are there?", "number of files", "how many records do we have?".\n'
        '- "filter" — the user wants the documents that MATCH a criterion (a year, a document type, '
        "or recency).\n"
        '  Examples: "which documents are from 2026?", "policies of type agreement", "what changed recently?".\n'
        '- "compare" — the user wants two NAMED documents contrasted against each other.\n'
        '  Examples: "difference between the Code of Conduct and the HR Handbook", "compare policy A and policy B".\n'
        '- "rag" — anything else: a normal factual question to answer from the documents, '
        "INCLUDING a summary or analysis of specific things described inside them.\n"
        '  Examples: "what is the leave policy?", "who approved the budget?", "explain the SLA", '
        '"give a detailed summary of SOPs PBSOP001 to PBSOP005".\n\n'
        f"QUESTION: {query.strip()}\n\nINTENT:"
    )


def parse_intent(raw: str, *, has_history: bool) -> Intent:
    """Robustly parse the model's reply into one :data:`Intent`.

    Safe by construction: ANYTHING ambiguous or unrecognised → ``rag`` (never mis-route a real
    question); a ``summarize`` label with no conversation history also degrades to ``rag``."""
    tokens = _words(raw)
    # doc_summary is two tokens under ``_words`` ("doc"+"summary"); detect the contiguous label on
    # the compacted reply FIRST so a genuine doc_summary is not double-counted as summarize.
    compact = re.sub(r"[ _\-]+", "", raw.lower())
    is_doc_summary = "docsummary" in compact or "docsummaries" in compact
    found: set[Intent] = set()
    if is_doc_summary:
        found.add("doc_summary")
    if tokens & {"catalog", "catalogue"}:
        found.add("catalog")
    if not is_doc_summary and tokens & {"summarize", "summarise", "summary"}:
        found.add("summarize")
    if "count" in tokens:
        found.add("count")
    if "filter" in tokens:
        found.add("filter")
    if tokens & {"compare", "comparison", "compared"}:
        found.add("compare")
    if "rag" in tokens:
        found.add("rag")
    if len(found) != 1:
        return "rag"
    intent = found.pop()
    if intent == "summarize" and not has_history:
        return "rag"
    return intent
