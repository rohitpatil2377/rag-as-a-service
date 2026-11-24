"""Document scoping — when a question NAMES a document, answer from that document.

Retrieval ranks passages by similarity to the question, which quietly fails a whole class of
request: *"summarise SOP Writing Guidance Pharma"*. Measured on a real two-document corpus,
that question retrieved 5 of 6 passages from ``SOPs - Leslie Dan Faculty of Pharmacy`` and one
from the document actually named. Both are about SOPs, so on content alone they are near
neighbours — and the answer came back confident, cited, and faithfulness 1.0, because the
faithfulness gate asks "is this answer supported by the passages?" and never "are these the
passages the user asked for?".

Contextual retrieval (prepending ``Document: <title>``) lifts the right document to the top but
does not stop the wrong one filling the rest of the pool. The reliable fix is a FILTER, not a
better score: if the question names a document, retrieve only from that document.

The matching rule is deliberately conservative, because a false positive is worse than no
scoping at all — it would silently hide the rest of the corpus:

* match on the document's DISTINCTIVE tokens, after dropping generic ones ("document", "final",
  "v2", "pdf") that every filename shares;
* require most of those tokens to be present in the question, not merely one;
* require a clear WINNER — if two documents match comparably the question is ambiguous, and
  guessing between them is exactly the failure this exists to prevent.

Pure logic, no I/O: the caller supplies the candidate names and applies the returned filter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Tokens that carry no identifying power: they appear in filenames across a corpus, so matching
# on them would scope "the final document" to whichever file happened to be called that.
_GENERIC = frozenset(
    {
        "document",
        "documents",
        "doc",
        "file",
        "final",
        "draft",
        "copy",
        "version",
        "v1",
        "v2",
        "v3",
        "new",
        "old",
        "updated",
        "revised",
        "single",
        "full",
        "the",
        "a",
        "an",
        "of",
        "for",
        "and",
        "in",
        "on",
        "to",
    }
)
# File extensions are never part of what a person says out loud.
_EXT = re.compile(r"\.(pdf|docx?|pptx?|xlsx?|txt|md|html?|png|jpe?g|tiff?|csv)$", re.IGNORECASE)
_TOKEN = re.compile(r"[a-z0-9]+")

# Fraction of a document's distinctive tokens the question must contain. 0.6 accepts
# "writing guidance pharma" for "SOP Writing Guidance Pharma.pdf" (3 of 4 distinctive tokens,
# 0.75) while rejecting a question that merely shares one word with a long filename.
MIN_COVERAGE = 0.6
# How far ahead of the runner-up the winner must be. Two documents scoring alike means the
# question did not identify one, and picking either would be a guess presented as a filter.
MIN_MARGIN = 0.2
# A one- or two-character token identifies nothing and matches everywhere.
_MIN_TOKEN_LEN = 3
# Prefix length for inflection matching ("pharma"/"pharmacy", "sop"/"sops"). Four is the
# shortest prefix that does not let unrelated words collide.
_PREFIX_LEN = 4


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(_EXT.sub("", text).lower())


def distinctive(name: str) -> set[str]:
    """A document name reduced to the tokens that could identify it."""
    return {t for t in _tokens(name) if t not in _GENERIC and len(t) >= _MIN_TOKEN_LEN}


def _coverage(doc_tokens: set[str], query_tokens: set[str]) -> float:
    """Fraction of the document's distinctive tokens the question mentions.

    A token also counts when the question holds an obvious inflection of it ("pharma" for
    "pharmacy", "sop" for "sops"), because people do not retype a filename exactly — they
    paraphrase it. Prefix matching at ``_PREFIX_LEN`` characters is the cheapest rule that catches this
    without letting "pharmacy" match "pharaoh".
    """
    if not doc_tokens:
        return 0.0
    hit = 0
    for d in doc_tokens:
        if d in query_tokens:
            hit += 1
            continue
        if len(d) >= _PREFIX_LEN and any(
            q.startswith(d[:_PREFIX_LEN]) or d.startswith(q[:_PREFIX_LEN])
            for q in query_tokens
            if len(q) >= _PREFIX_LEN
        ):
            hit += 1
    return hit / len(doc_tokens)


@dataclass(frozen=True, slots=True)
class DocScope:
    """The document a question named, and how strongly."""

    doc_id: str
    coverage: float
    margin: float


def scope_for_query(query: str, doc_ids: list[str]) -> DocScope | None:
    """The document this question names, or None when it names none unambiguously.

    ``None`` is the safe answer and the common one: an ordinary question mentions no document
    and must keep searching the whole knowledge base.
    """
    q = set(_tokens(query))
    if not q or not doc_ids:
        return None
    scored = sorted(((_coverage(distinctive(d), q), d) for d in doc_ids), reverse=True)
    best, best_doc = scored[0]
    if best < MIN_COVERAGE:
        return None
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    margin = best - runner_up
    if margin < MIN_MARGIN:
        return None  # two documents match alike — the question did not choose between them
    return DocScope(doc_id=best_doc, coverage=best, margin=margin)
