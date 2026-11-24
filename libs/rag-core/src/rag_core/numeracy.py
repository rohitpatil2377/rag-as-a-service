"""Numeric grounding (FR-E4 support) — pure logic, no I/O.

Faithfulness verification scores token overlap, and deliberately drops short tokens so that
scoring "keys on content". That makes a FIGURE invisible to it: a number the model computed
rather than read has no effect on the score, while the words around it — "hours", "days",
"working" — are all present in the sources and carry the sentence to "supported".

For a governance corpus this is backwards. The numbers are the content: the notice period, the
review deadline, the allowance, the threshold. Prose can be paraphrased and still be true; a
figure cannot. If it is not in the retrieved text, it was invented.

The check is deliberately narrow — "does this figure appear in what we retrieved" — because
deciding whether a figure is used *correctly* requires understanding the claim, which is the
judge's job. This one is structural and deterministic, so it holds even when the judge is a
small model that cannot be trusted with arithmetic.
"""

from __future__ import annotations

import re

from rag_schemas import RetrievedSpan

# A figure, with optional thousands separators and decimal part. Currency symbols and units sit
# outside the match, so "Rs. 50,000" and "50000" reduce to the same value.
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
# Inserted by the citation assembler, not asserted by the model — never a claim.
_CITATION_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
# Ordered-list markers. A numbered list is layout; flagging "1." would fire on almost every
# structured answer the platform produces and the guard would have to be switched off.
_LIST_MARKER_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)


def _canonical(raw: str) -> str:
    """One spelling per value, so a true figure written differently is not called fabricated.

    "50,000" and "50000" are the same amount; "24.0" and "24" are the same count. A guard that
    flagged those would fire on correct answers, and a guard that fires on correct answers gets
    turned off."""
    value = raw.replace(",", "")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value or "0"


def numbers_in(text: str) -> set[str]:
    """Every figure ASSERTED in ``text``, canonicalised.

    Citation markers and list numbering are removed first: both are produced by the platform's
    own formatting rather than claimed by the model, so counting them would measure us."""
    stripped = _LIST_MARKER_RE.sub("", _CITATION_RE.sub(" ", text))
    return {_canonical(m.group()) for m in _NUMBER_RE.finditer(stripped)}


def unsupported_numbers(answer: str, spans: list[RetrievedSpan]) -> list[str]:
    """Figures the answer states that appear nowhere in the retrieved spans, in reading order.

    Sources are read WITHOUT the list-marker and citation stripping applied to the answer: a
    "1." that is layout in an answer may be a real value in a source, and being generous about
    what counts as supported keeps the guard aimed at fabrication rather than at formatting."""
    supported = {_canonical(m.group()) for s in spans for m in _NUMBER_RE.finditer(s.text)}
    seen: set[str] = set()
    out: list[str] = []
    stripped = _LIST_MARKER_RE.sub("", _CITATION_RE.sub(" ", answer))
    for match in _NUMBER_RE.finditer(stripped):
        value = _canonical(match.group())
        if value not in supported and value not in seen:
            seen.add(value)
            out.append(value)
    return out
