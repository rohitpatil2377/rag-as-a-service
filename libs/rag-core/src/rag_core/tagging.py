"""Auto-metadata & taxonomy tagging (FR-B7) — pure logic, no framework.

At ingest, derive filterable metadata from a document's text with zero tenant effort:
* **entities** — reuses the GraphRAG proper-noun extractor;
* **year / dates** — the most recent 4-digit year mentioned (useful for recency filters, D9);
* **doc_type** — a coarse, domain-neutral class from surface cues (agreement / policy / report
  / invoice / manual / …), default ``document``;
* **taxonomy categories** — per-KB configurable ``{label: [keywords]}``; a label is applied
  when any of its keywords appears.

Everything lands in ``DocumentEnvelope.metadata`` (and, stamped onto chunks, becomes a D8
retrieval filter). Deterministic + CPU-free; the LLM path can refine classes later.
"""

from __future__ import annotations

import re

from rag_core.graph import extract_entities

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_MAX_ENTITIES = 8

# Coarse, domain-neutral document classes from surface cues (checked in order; first wins).
_DOC_TYPE_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("agreement", ("agreement", "hereby agree", "party of the", "terms and conditions", "this contract")),
    ("invoice", ("invoice", "amount due", "total due", "bill to", "purchase order")),
    ("policy", ("policy", "shall be entitled", "eligibility", "is applicable to all")),
    ("report", ("executive summary", "findings", "fiscal year", "quarterly", "annual report")),
    ("manual", ("step 1", "how to", "instructions", "user guide", "installation")),
    ("resume", ("curriculum vitae", "work experience", "professional summary")),
    ("minutes", ("minutes of the meeting", "attendees", "action items")),
)


# A document's own release/version block, which is what actually dates it. Acme documents carry a
# "Document Release History" table -- Version | Prepared Date | Review Date | Release Date -- and
# the RELEASE date of the LAST row is the date the document took effect.
_RELEASE_CUES = (
    "document release history",
    "release date",
    "effective date",
    "date of issue",
    "version history",
    "revision history",
    "last revised",
    "last updated",
)
# "1st January 2024", "1 Jan 2024", "01-01-2024", "2024-01-01".
_DATE_RE = re.compile(
    r"(?:(\d{1,2})\s*(?:st|nd|rd|th)?[\s./-]*"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s./-]*(\d{4}))"
    r"|(?:(\d{4})[-/](\d{1,2})[-/](\d{1,2}))"
    r"|(?:(\d{1,2})[-/](\d{1,2})[-/](\d{4}))",
    re.IGNORECASE,
)
_MONTHS = {
    m: i
    for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), start=1)
}
# How far past the release cue to look for its dates. A release block is a small table; scanning
# further pulls in body prose and reintroduces exactly the noise this replaces.
_RELEASE_WINDOW = 900
_MAX_MONTH = 12


def _dates_in(text: str) -> list[tuple[int, int, int]]:
    """Every parseable (year, month, day) in ``text``, in document order."""
    out: list[tuple[int, int, int]] = []
    for m in _DATE_RE.finditer(text):
        if m.group(3):
            out.append((int(m.group(3)), _MONTHS[m.group(2)[:3].lower()], int(m.group(1))))
        elif m.group(4):  # unambiguous ISO: YYYY-MM-DD
            out.append((int(m.group(4)), int(m.group(5)), int(m.group(6))))
        elif m.group(7):  # all-numeric with the year last
            a, b, year = int(m.group(7)), int(m.group(8)), int(m.group(9))
            # 15-03-2025 can only be DD-MM. 03-15-2025 can only be MM-DD. 03-04-2025 is genuinely
            # ambiguous, and this platform is built for India, so DD-MM is the reading.
            day, month = (a, b) if b <= _MAX_MONTH else (b, a)
            if month <= _MAX_MONTH:
                out.append((year, month, day))
    return out


def document_date(text: str) -> str | None:
    """The date the document took effect, as ``YYYY-MM-DD`` — or None if it does not say.

    Read from the document's OWN release/version block, not from the body. The previous rule was
    "the most recent 4-digit year mentioned anywhere", which is wrong in a way that matters once
    anything ranks on it: measured on a real corpus it dated a 2026 handbook as **2029** (a future
    year quoted in a table) and a travel policy as **2000**. A signal that confidently wrong is
    worse than none, because recency ranking would then prefer the wrong document.

    Returning None is a real answer and the common one. A caller must treat "undated" as
    "unknown", never as "old"."""
    low = text.lower()
    for cue in _RELEASE_CUES:
        at = low.find(cue)
        if at == -1:
            continue
        dates = _dates_in(text[at : at + _RELEASE_WINDOW])
        if dates:
            # The LAST row of a release-history table is the current version, so take the latest
            # date in the block rather than the first.
            y, m, d = max(dates)
            return f"{y:04d}-{m:02d}-{d:02d}"
    return None


def latest_year(text: str) -> str | None:
    """The document's effective YEAR, from its release block. None when it does not say.

    Deliberately NOT "the most recent year mentioned" — see :func:`document_date`."""
    dated = document_date(text)
    return dated[:4] if dated else None


def classify_doc_type(text: str) -> str:
    """A coarse document class from surface cues; ``document`` when nothing matches."""
    low = text.lower()
    for label, cues in _DOC_TYPE_CUES:
        if any(cue in low for cue in cues):
            return label
    return "document"


def apply_taxonomy(text: str, taxonomy: dict[str, list[str]]) -> list[str]:
    """Per-KB taxonomy labels whose keywords appear in the text (case-insensitive)."""
    low = text.lower()
    hits: list[str] = []
    for label, keywords in taxonomy.items():
        if any(kw.lower() in low for kw in keywords if kw.strip()):
            hits.append(label)
    return hits


# Every key auto_tags may write. A re-index must CLEAR these before re-deriving: a plain merge
# only overwrites keys the new derivation still produces, so a key it stops producing survives
# from the previous index. Measured after the dating fix: documents with no release block kept
# their old wrong `year` (Code of Conduct 2027, a travel policy 2000) even though re-indexing
# reported success -- the exact stale value the fix existed to remove.
DERIVED_KEYS = frozenset({"doc_type", "doc_date", "year", "entities", "category", "categories"})


def auto_tags(text: str, taxonomy: dict[str, list[str]] | None = None) -> dict[str, str]:
    """Derive the filterable metadata dict for a document (FR-B7). Values are strings so they
    persist in the ``metadata`` JSONB and match D8 exact-value filters (single-valued keys
    like ``doc_type``/``year``/``category`` filter cleanly; multi-valued are informational)."""
    tags: dict[str, str] = {"doc_type": classify_doc_type(text)}
    # The document's OWN effective date, when it states one. Absent rather than guessed: a wrong
    # date is worse than none once anything ranks or resolves conflicts on it.
    dated = document_date(text)
    if dated:
        tags["doc_date"] = dated
        tags["year"] = dated[:4]
    entities = extract_entities(text)[:_MAX_ENTITIES]
    if entities:
        tags["entities"] = ", ".join(entities)
    categories = apply_taxonomy(text, taxonomy or {})
    if categories:
        tags["category"] = categories[0]  # primary, exact-filterable
        tags["categories"] = ", ".join(categories)
    return tags
