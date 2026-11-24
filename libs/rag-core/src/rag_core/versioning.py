"""Version conflict resolution — which of two documents is current. Pure logic, no I/O.

A real corpus holds more than one version of the same policy: last year's handbook, this year's,
a departmental copy. Measured live, two Acme handbooks listed DIFFERENT members of the same
committee. The platform reported both and said the sources differ — honest, but a reader cannot
tell which list is in force, and at a thousand documents that becomes the normal answer.

This module decides only one thing: **given the documents a question retrieved, which are
superseded?** Reading their dates, fetching them and writing the answer belong elsewhere.

Supersession is a claim about ONE document having replaced ANOTHER, so it may only be made
between versions of the same document. Measured live, the first cut ignored that and told a
reader the Night Shift Allowance Policy "has been superseded by" the HR Handbook — two unrelated
policies that merely happened to be retrieved together, one older than the other. Documents are
therefore grouped into families by title first, and only compared inside a family.

Within a family, two rules, in order:

1. **An operator pin wins.** A knowledge-base manager who marks a document authoritative has
   information the platform does not — a correct date the file never stated, or a decision that
   the older wording still governs. Never overrule them.
2. **Otherwise the newer date wins**, and only when BOTH documents state one. An undated document
   is UNKNOWN, not old: guessing would let a document that merely omits its date be silently
   buried by one that happens to print theirs.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass

# A trailing ".pdf"/".docx" is a container format, not part of the document's identity. Bounded to
# short alphabetic runs so a title that simply ends in a decimal ("Policy v1.2") keeps its number.
_EXTENSION = re.compile(r"\.[a-z]{1,5}$", re.IGNORECASE)
# Anything but letters and digits separates words: "HR_Handbook 4.0_2026" -> hr, handbook, 4, 0, 2026.
_SEPARATOR = re.compile(r"[^a-z0-9]+")
# A token that is only digits (with optional "v"/dots consumed by the split) is a version or a
# year — exactly what DIFFERS between two versions of one document, so it cannot be evidence of
# identity. Dropping it is what lets "Handbook 4.0_2026" and "Handbook 3.0_2025" find each other.
_VERSION_TOKEN = re.compile(r"^v?\d+$")

# How alike two titles must be, sharing of their combined vocabulary, to count as the same
# document. The operator overrides it from config; this is the default the tuning starts from.
DEFAULT_FAMILY_SIMILARITY = 0.6


@dataclass(frozen=True, slots=True)
class DocumentVersion:
    """What is known about one retrieved document's currency."""

    doc_id: str
    date: str | None = None  # YYYY-MM-DD from its own release block, or None if it does not say
    pinned: bool = False  # a KB manager marked this the authoritative version
    title: str | None = None  # the human title, when the catalog holds one; else the doc_id reads as one


def title_tokens(title: str, *, ignore: Collection[str] = ()) -> frozenset[str]:
    """The identity-bearing words of a document title, lowercased.

    Drops the file extension, version/year numbers, and any token the caller says carries no
    identity in this corpus (the organisation's own name, which prefixes every one of its files)."""
    stem = _EXTENSION.sub("", title)
    ignored = {t.lower() for t in ignore}
    return frozenset(
        tok for tok in _SEPARATOR.split(stem.lower()) if tok and not _VERSION_TOKEN.match(tok) and tok not in ignored
    )


def title_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    """Jaccard overlap of two token sets, in [0, 1]. Two titles with no words in common score 0."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def resolve_supersession(
    docs: list[DocumentVersion],
    *,
    ignore_tokens: Collection[str] = (),
    min_similarity: float = DEFAULT_FAMILY_SIMILARITY,
) -> dict[str, str]:
    """``{superseded_doc_id: superseding_doc_id}`` for the documents that are not current.

    Only documents that can be COMPARED are resolved — same family, and both stating a date. One
    dated document alongside three undated ones supersedes none of them: being the only one to
    print a date is not evidence of being the newest, and acting as though it were would quietly
    hide three documents on no evidence."""
    superseded: dict[str, str] = {}
    for family in _families(docs, ignore_tokens=ignore_tokens, min_similarity=min_similarity):
        superseded.update(_resolve_family(family))
    return superseded


def _families(
    docs: list[DocumentVersion], *, ignore_tokens: Collection[str], min_similarity: float
) -> list[list[DocumentVersion]]:
    """Group documents whose titles name the same document. A document joins a family when it is
    similar enough to ANY member, so "Handbook 3.0" and "Handbook 4.0" pull each other in through
    the wording they share, not through a canonical name nobody wrote down."""
    grouped: list[tuple[list[frozenset[str]], list[DocumentVersion]]] = []
    for doc in docs:
        tokens = title_tokens(doc.title or doc.doc_id, ignore=ignore_tokens)
        for known, members in grouped:
            if any(title_similarity(tokens, other) >= min_similarity for other in known):
                known.append(tokens)
                members.append(doc)
                break
        else:
            grouped.append(([tokens], [doc]))
    return [members for _, members in grouped]


def _resolve_family(docs: list[DocumentVersion]) -> dict[str, str]:
    """Pin-then-date, applied to versions of ONE document."""
    if len(docs) < 2:  # noqa: PLR2004 - a lone version supersedes nothing
        return {}
    # An operator decision is absolute, and a SINGLE winner is the point of pinning. With more
    # than one pin there is no decision to honour, so fall through to dates rather than pick one
    # arbitrarily and present it as though someone had chosen it.
    pinned = [d for d in docs if d.pinned]
    if len(pinned) == 1:
        winner = pinned[0].doc_id
        return {d.doc_id: winner for d in docs if d.doc_id != winner}

    dated = [d for d in docs if d.date]
    if len(dated) < 2:  # noqa: PLR2004 - nothing to compare against
        return {}
    newest = max(dated, key=lambda d: d.date or "")
    return {d.doc_id: newest.doc_id for d in dated if d.doc_id != newest.doc_id and d.date != newest.date}
