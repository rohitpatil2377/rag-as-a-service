"""Enumerated items in a question — the things a reader listed and expects each answered.

"Summarise PBSOP001, PBSOP002 ... PBSOP010" names ten documents. Retrieval ranks passages by
similarity to the whole question, so a few of the ten take the entire passage budget and the
rest are never fetched; the answer then truthfully reports them as absent. Coverage has to be
asked for per item. This module only decides WHAT the items are; the pipeline applies the quota.

Recognition is by SHAPE, not by a list of known prefixes: an identifier is a token with letters
and at least two digits (``PBSOP007``, ``XQA-001``, ``SOP-2026-11``, ``ISO9001``), and a question
counts as enumerated when it carries at least ``MIN_ITEMS`` distinct ones. Three is the floor
because two identifiers are a comparison the ordinary path already handles well, and one is
just a subject.
"""

from __future__ import annotations

import re

MIN_ITEMS = 3
# Letters+digits with an optional hyphen/underscore run; case-insensitive so "pbsop007" typed
# in a hurry still counts. Must contain at least one letter AND two digits.
_IDENT = re.compile(r"\b(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*\d{2})[A-Za-z][A-Za-z0-9_-]{2,}\b")
# Ordinary words with digits that are not identifiers of anything: years, times, versions.
_NOISE = re.compile(r"^(?:\d{4}|v?\d+(?:\.\d+)+|[0-9]+(?:st|nd|rd|th)|covid19|h1n1)$", re.IGNORECASE)


def enumerated_items(question: str) -> list[str]:
    """Distinct identifiers the question lists, in order of first mention; [] if fewer than
    ``MIN_ITEMS`` -- the safe answer, and the common one."""
    seen: dict[str, str] = {}
    for m in _IDENT.finditer(question):
        tok = m.group(0)
        if _NOISE.match(tok):
            continue
        key = tok.upper()
        if key not in seen:
            seen[key] = tok
    items = list(seen.values())
    return items if len(items) >= MIN_ITEMS else []
