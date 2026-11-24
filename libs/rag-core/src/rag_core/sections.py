"""Document section structure (FR-C2 support) — pure logic, no I/O.

A chunk embedded on its own has lost the heading it was written under. "Written sign-off is
required before publication" means something different beneath "Mandatory Rehearsal Policy"
than beneath "Central Content Repository", and nothing in the chunk says which. Prepending the
section path at embed time restores that, while the stored chunk — and therefore every
citation — stays exactly as the document wrote it.

The detector is narrow ON PURPOSE. The rule below came from measuring two real corpora, not
from a general idea of what a heading looks like:

    Acme Content Governance SOP        HR Policy Handbook
    5.1   Process                      1. Does the employee acquire new skills quickly?
    5.1.1 Entry Criteria               1. Is the employee a critical/key resource?
    6.1.1 Mandatory Rehearsal Policy   1 Q1 - April to June 01st June 15th June
    6.3   Event and Run-of-Show Mgmt   1 H a p p in e s s L e a v e -

Multi-level numbering is a heading. Single-level numbering, on real documents, is far more
often an appraisal question, a list item or a table row — so it is rejected. A WRONG section
path is worse than none: it would tell the embedder a chunk belongs somewhere it does not, and
unlike a missing path that error is invisible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# "6.1.1 Mandatory Rehearsal Policy" — at least one dot, so a bare "1." never matches, then a
# title starting with a letter. A trailing dot on the number is allowed ("6.1. Title").
_HEADING_RE = re.compile(r"^(?P<number>\d+(?:\.\d+)+)\.?\s+(?P<title>[A-Za-z][^\n]*)$")
# The same heading, but merged onto the tail of the preceding paragraph by PDF extraction. It
# must FOLLOW a sentence terminator and RUN TO the end of the line: that is what separates a
# heading the layout ran together from a mid-sentence cross-reference.
_TRAILING_HEADING_RE = re.compile(r"(?<=[.!?])\s+(?P<number>\d+(?:\.\d+)+)\.?\s+(?P<title>[A-Za-z][^\n]*?)\s*$")
# A heading is a title, not a sentence. Long enough to be a name, short enough not to be a
# clause that happens to be numbered.
_MAX_HEADING_CHARS = 80
_MIN_TITLE_CHARS = 3
# A title made mostly of digits is a table row ("1 Q1 - April to June 01st June 15th June").
_MAX_DIGIT_SHARE = 0.3
PATH_SEP = " > "
"""The v1 section-path separator. PUBLIC because it is a STORAGE/QUERY contract, not a
formatting choice: it is what `metadata["section"]` holds on every chunk ever indexed, and
`filters` matches that value EXACTLY. The layout path's own breadcrumbs use
`layout.BREADCRUMB_SEP` (" \u203a "), so anything writing a breadcrumb into the shared
`section` key must convert to THIS separator or the filter silently matches nothing."""
# How far below a heading its scope still reaches. "Until the next heading" is unbounded when
# the next one was never detected — a document that numbers its paragraphs ("8016. Resignation
# Policy") or writes an unnumbered one ("Content Classification:") has no heading a numbering
# detector can see, so text pages below inherits a scope nothing supports. Real and measured:
# the HR handbook's notice-period rule sat 10 193 characters below "1.4 Role of Investigating
# Officer" and was labelled with it.
#
# Derived from the two corpora whose structure the detector actually resolves — the SOP (23
# gaps, p75 876) and the CAG RFP (28 gaps, p75 2 154). NOT from the pooled distribution: that
# one is contaminated by this very defect, since a 44 000-character "section" is a region where
# headings were missed rather than a long section, and calibrating against it would justify the
# behaviour being fixed. 2 000 keeps the great majority of genuine spans and cuts 9 of the 12
# wrong attributions measured on the golden set. Beyond it, no path: a wrong scope is worse
# than none.
_MAX_SECTION_SPAN = 2_000

# A heading a numbering-based detector cannot see, in the two styles the real corpora use.
#
# UNNUMBERED, ending in a colon and running into its own body on one line: the SOP writes
# "Content Classification: All Acme content is classified into four categories." Nothing
# numbered precedes it, so every Category A-D rule inherited "1.3 Objective" from pages above.
# Two or more words is the guard that keeps ordinary prose ("Note: ...", "For example: ...")
# from becoming a section.
# NOT DETECTED, deliberately: the UNNUMBERED heading style ("Content Classification: All Acme
# content is classified into four categories."). Measured 2026-07-31 on the three real corpora,
# a colon rule found 50 "headings" in the SOP against 32 genuine numbered ones — "Prepared by",
# "Version Number", "SANJAY SHARDUL", "Confidential Information", i.e. page furniture, table
# cells and glossary terms. Tightening it (no pipe on the line, a real sentence after the colon)
# cut those to 6 but they were still wrong — "Marketing Head", "All Acme employees" — AND it no longer
# matched the case it was written for, because that line carries a pipe further along.
#
# So the two Category A-D items in the golden set stay UNLABELLED rather than mislabelled, which
# is the safe answer and the rule this module is built on. Detecting these needs layout signal
# (font size, weight, position) that the text layer does not carry.

# SINGLE-LEVEL numbering ("1. Company Introduction"). Refused outright until now, because on the
# HR handbook single-level numbers are list items: its candidates run 2, 5, 1, 4, 1, 1, 3. But
# the Code of Conduct numbers its sections 1..14 with no sub-levels, so a blanket refusal left
# it with no structure at all. An ASCENDING RUN is what distinguishes a document that numbers
# its sections from one that numbers its bullets — measured, not assumed.
_SINGLE_LEVEL_RE = re.compile(r"^(?P<number>\d{1,3})\.\s+(?P<title>[A-Z][^\n]*)$")
_MIN_SINGLE_LEVEL_RUN = 4
# Measured: Code of Conduct 22 of 25 candidates ascend (genuine sections, with the contents page
# repeating 12-17); the HR handbook, Mediclaim and Abstract sit far below this because their
# single-level numbers are list items and table values that restart constantly.
_MIN_ASCENDING_SHARE = 0.7


@dataclass(frozen=True, slots=True)
class Section:
    """One heading and where it starts in the document text."""

    offset: int
    number: str  # "6.1.1"
    title: str  # "Mandatory Rehearsal Policy"
    # Is this heading reliable enough to decide document STRUCTURE — i.e. to force a chunk
    # boundary — or only good enough to label a chunk that happens to sit under it?
    #
    # Multi-level numbering ("6.2.3 Social Media Content") is unambiguous, so it is structural.
    # A single-level number is not: this module's own rule is that "1." is far more often a list
    # item than a section, and the ascending-run escape hatch below re-admits it for the whole
    # DOCUMENT on the evidence of a few genuine headings elsewhere in it. On the Acme HR handbook
    # that promoted every numbered list item in 174 pages to a heading, and since a heading is a
    # hard chunk boundary it shattered each enumerated list into one chunk per item — "what are
    # the core values" could then only ever be answered 2 of 6, because no chunk held the list.
    #
    # Measured on that handbook: treating single-level guesses as non-structural took chunks from
    # 686 to 552, sub-350-character chunks from 239 to 76, and the median chunk from 478 to 609
    # characters, while section labelling only moved 184 -> 164. The boundary was costing a third
    # of the corpus's chunk quality and buying almost no scope.
    structural: bool = True

    @property
    def label(self) -> str:
        return f"{self.number} {self.title}".strip()

    @property
    def key(self) -> str:
        """What identifies this heading among its levels — its number, or its title when it has
        no number. ``levels`` is keyed the same way, so ancestry works for both kinds."""
        return self.number or self.title

    @property
    def levels(self) -> tuple[str, ...]:
        """("6", "6.1", "6.1.1") — this heading and every ancestor it implies.

        An UNNUMBERED heading has no ancestry to imply: it is its own only level, keyed by its
        title so it cannot collide with a numbered one."""
        if not self.number:
            return (self.title,)
        parts = self.number.split(".")
        return tuple(".".join(parts[: i + 1]) for i in range(len(parts)))


def detect_sections(text: str) -> list[Section]:
    """Every heading in ``text``, in document order, with real offsets."""
    sections: list[Section] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        match = _HEADING_RE.match(stripped)
        if match and _is_title(match.group("title"), len(stripped)):
            sections.append(Section(offset=offset, number=match.group("number"), title=match.group("title").strip()))
        else:
            trailing = _trailing_heading(line)
            if trailing is not None:
                start, number, title = trailing
                sections.append(Section(offset=offset + start, number=number, title=title))
        offset += len(line)
    return sorted([*sections, *_single_level_headings(text)], key=lambda s: s.offset)


def _single_level_headings(text: str) -> list[Section]:
    """Single-level numbered headings, but only if this document's numbers ASCEND.

    All candidates are collected first because the decision is about the DOCUMENT, not any one
    line: a run of 1..14 is a table of sections, while 2, 5, 1, 4, 1 is a set of lists that each
    restart. Neither line tells you which on its own."""
    found: list[tuple[int, int, str]] = []  # (offset, number, title)
    offset = 0
    for line in text.splitlines(keepends=True):
        match = _SINGLE_LEVEL_RE.match(line.strip())
        if match and _is_title(match.group("title"), len(line.strip())):
            found.append((offset, int(match.group("number")), match.group("title").strip()))
        offset += len(line)
    ascending = [f for i, f in enumerate(found) if i == 0 or f[1] > found[i - 1][1]]
    if len(ascending) < _MIN_SINGLE_LEVEL_RUN or len(ascending) / len(found) < _MIN_ASCENDING_SHARE:
        return []  # numbers that restart are lists, not sections
    return [Section(offset=o, number=str(n), title=t, structural=False) for o, n, t in found]


def _trailing_heading(line: str) -> tuple[int, str, str] | None:
    """A heading PDF extraction ran onto the end of the previous paragraph.

    Real and measured: in the Acme SOP the text reads
    ``"... f) PR agency distributes only after written sign-off is on record. 6.2.3 Social Media
    Publishing"`` on ONE line. Anchoring headings to line starts missed it, so every social-media
    rule in the document inherited "6.2.1 Website Content Updates" — the exact scope transfer
    (QA Q10) this module exists to prevent, produced by the module itself.

    The preceding sentence terminator is what keeps this narrow. A cross-reference reads "as set
    out in section 6.2.3 Social Media Publishing" with no full stop before the number, so it is
    not mistaken for a heading."""
    match = _TRAILING_HEADING_RE.search(line)
    if match is None:
        return None
    title = match.group("title").strip()
    # Measured against the heading itself, not the whole line: the paragraph in front of it says
    # nothing about whether what follows is a title.
    if not _is_title(title, len(match.group("number")) + 1 + len(title)):
        return None
    return match.start("number"), match.group("number"), title


def _is_title(title: str, line_length: int) -> bool:
    """A heading names a section; a numbered clause states a rule and a table row lists values."""
    title = title.strip()
    if len(title) < _MIN_TITLE_CHARS or line_length > _MAX_HEADING_CHARS:
        return False
    digits = sum(1 for c in title if c.isdigit())
    return digits / len(title) <= _MAX_DIGIT_SHARE


def section_path_at(sections: list[Section], offset: int, *, max_span: int = _MAX_SECTION_SPAN) -> str:
    """The heading path covering ``offset`` — "5.1 Process > 5.1.1 Entry Criteria".

    Ancestors are taken from headings the document ACTUALLY contains. A document that jumps
    from "6" straight to "6.1.1" gets a two-part path, not an invented "6.1": showing a level
    that was never written would be a claim about the document's structure that is not true.

    ``max_span`` bounds how far a heading reaches, because "until the next heading" is unbounded
    when the next one was never detected — see _MAX_SECTION_SPAN."""
    current = None
    for section in sections:
        if section.offset > offset:
            break
        current = section
    if current is None:
        return ""  # before the first heading: a title block is not a section
    if offset - current.offset > max_span:
        return ""  # too far below its heading for that heading to be evidence of anything

    # Walk back for the nearest preceding heading at each ancestor level.
    by_level: dict[str, str] = {}
    for section in sections:
        if section.offset > offset:
            break
        for level in section.levels:
            if section.key == level:
                by_level[level] = section.label
    return PATH_SEP.join(by_level[level] for level in current.levels if level in by_level)


def section_path_for(sections: list[Section], start: int, end: int, *, max_span: int = _MAX_SECTION_SPAN) -> str:
    """The heading path governing the whole range ``[start, end)`` — or "" if none does.

    A range that crosses into a later section is governed by NO single heading, and stamping it
    with the one it happened to begin under is the P39 defect a layer up: the chunk would carry
    a scope that is wrong for part of its own text, which is exactly the misreading section
    scope exists to prevent (QA Q10). A heading AT ``start`` opens the range rather than
    interrupting it — that is the common, correct case of a chunk led by its own heading."""
    if any(start < section.offset < end for section in sections):
        return ""
    return section_path_at(sections, start, max_span=max_span)
