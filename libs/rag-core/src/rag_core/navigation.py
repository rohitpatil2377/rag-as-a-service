"""Navigation text — a table of contents is a map of the document, not evidence from it.

A contents page names every policy in the book and states none of them: "Salary Advance Policy
.... 50" is the strongest lexical match a salary-advance question can find and supports no answer
to it. Measured on Acme's HR Handbook, 14 of 554 chunks are contents pages, and one has been seen
winning a citation over the policy text it points at.

Applied at RETRIEVAL, not at ingest, for two reasons: the text is worth keeping (a question about
which chapter covers something is answered by exactly this), and a rule about what counts as
evidence should be changeable without re-indexing every document that was ingested under the
old one.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TypeVar

# The dot leader that joins a title to its page number. Four or more, so the three of an ellipsis
# ("shall... be reimbursed") is never mistaken for one.
_LEADER = re.compile(r"\.{4,}")
# One leader could be an artefact of bad extraction; a run of them is a contents page.
_MIN_LEADERS = 2

_S = TypeVar("_S")


def is_navigation(text: str) -> bool:
    """Is this text a pointer to content rather than content?

    Deliberately narrow: dot leaders only. A contents page WITHOUT leaders reads as ordinary
    short lines ("CHAPTER 2: EMPLOYEE CLASSIFICATION 10") and separating those from a genuine
    numbered list of clauses needs evidence this rule does not have — and mistaking a clause list
    for navigation would drop the very text that answers the question."""
    return len(_LEADER.findall(text)) >= _MIN_LEADERS


def drop_navigation(spans: Sequence[_S]) -> list[_S]:
    """Remove navigation spans, unless they are the only spans there are.

    Order and scores are untouched: this decides what counts as evidence, not how good the
    evidence is — that number belongs to the reranker alone (ADR-0015)."""
    kept = [s for s in spans if not is_navigation(getattr(s, "text", ""))]
    return kept if kept else list(spans)
