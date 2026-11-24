"""Near-duplicate detection between chunks (FR-C9 extension) — pure logic, no I/O.

Ingest already rejects a re-upload of byte-identical CONTENT (``content_hash``). This is the
different question: two chunks whose TEXT is the same or nearly so, in different documents.

Measured on the real corpus, this is not hypothetical. `acme__hr_policies` holds 15 chunk
pairs at similarity 1.0 — the Code of Conduct is reproduced verbatim inside the HR Policy
Handbook, so `Code_Of_Conduct.pdf::11` and `HR_POLICY_HANDBOOK.pdf::32` are the same text under
two doc_ids.

That is the mechanism behind an answer citing "[1], [3], [5]" and saying "the same requirement
is reiterated in..." — one rule reproduced three times reads as three independent
confirmations, and a reader weights it accordingly. Detecting it is what lets a later step
present them as one source rather than several.

Shingled Jaccard rather than a hash: an exact hash misses the near-misses (a reprinted clause
with a different heading or page number), and those are the ones that read as corroboration.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Words per shingle. Long enough that ordinary shared phrasing ("the Content Curator shall")
# does not make two unrelated chunks look alike; short enough that a reprint with small edits
# still matches. Not a deployment knob: changing it changes what "duplicate" means, and the
# stored band keys would no longer be comparable with the ones already indexed.
SHINGLE_WORDS = 5
# Length below which a chunk is not compared at all. Short chunks share shingles by accident —
# a heading, a page number, a single sentence of boilerplate — and flagging those as duplicates
# would bury the real reprints.
MIN_COMPARABLE_CHARS = 80
# The band key is a cheap bucket, NOT a decision: chunks sharing one are candidates that then
# get a real similarity check. It exists so a new document is compared against a handful of
# plausible neighbours instead of every chunk in the knowledge base.
_BAND_CHARS = 6


def shingles(text: str) -> set[str]:
    """Overlapping word n-grams — the unit similarity is measured in."""
    tokens = _TOKEN_RE.findall(text.lower())
    if len(tokens) < SHINGLE_WORDS:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + SHINGLE_WORDS]) for i in range(len(tokens) - SHINGLE_WORDS + 1)}


def similarity(left: str, right: str) -> float:
    """Jaccard overlap of the two texts' shingles, in [0, 1]. 1.0 means the same text."""
    a, b = shingles(left), shingles(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def band_key(text: str) -> str:
    """A bucket label for candidate lookup. Two chunks that share a lot of text are very likely
    to share this; two that do not, are not — so it narrows the search without deciding it.

    Empty for text too short to compare, which keeps headings and page furniture out of the
    candidate pool entirely."""
    grams = shingles(text)
    if not grams or len(text) < MIN_COMPARABLE_CHARS:
        return ""
    return min(hashlib.blake2b(g.encode(), digest_size=8).hexdigest()[:_BAND_CHARS] for g in grams)


def near_duplicates(
    candidates: Sequence[tuple[str, str]], existing: Iterable[tuple[str, str]], threshold: float
) -> dict[str, tuple[str, float]]:
    """Which candidates duplicate something in ``existing``.

    Both are ``(id, text)``. Returns ``{candidate id: (matched id, similarity)}`` for those at
    or above ``threshold``, keeping the STRONGEST match — a clause reprinted in three documents
    should point at the one it resembles most, not at whichever was scanned first.

    A candidate is never reported against itself, so this is safe to run with a document's own
    chunks in both arguments (a document that repeats itself is a real finding)."""
    by_band: dict[str, list[tuple[str, str]]] = {}
    for cid, text in existing:
        key = band_key(text)
        if key:
            by_band.setdefault(key, []).append((cid, text))

    out: dict[str, tuple[str, float]] = {}
    for cid, text in candidates:
        key = band_key(text)
        if not key:
            continue
        for other_id, other_text in by_band.get(key, ()):
            if other_id == cid:
                continue
            score = similarity(text, other_text)
            if score >= threshold and score > out.get(cid, ("", 0.0))[1]:
                out[cid] = (other_id, round(score, 4))
    return out
