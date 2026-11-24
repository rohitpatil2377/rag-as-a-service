"""D.3: a wrapped sentence promoted to a heading, and what that cost downstream.

THE DEFECT, TRACED END TO END ON A REAL DOCUMENT. The AEGCL SOC/NOC corrigendum says:

    8. The requirement of Individual Confidentiality Declaration for Site Personnel per
       Sl no: D.3 of Appendix D - Vendor Undertaking Formats, Vol-II Part-A, is withdrawn.

The sentence wraps, and its continuation is set in heading type. `_classify_headings` promoted it,
which split sentence 8 across a section boundary. Consequences, every one measured:

  * the stored chunk ends mid-clause at "per Sl no: D.3 of"
  * the disposition ("... is withdrawn.") became its own SECTION, which the Phase-8 section pass
    never selected for the aggregate question
  * the platform therefore never held the fact, so no prompt could recover it
  * the orphaned word "withdrawn" sat in an unrelated section and produced a FALSE POSITIVE in
    the supersession metric, which credited the incumbent with reporting an amendment it never
    mentioned

The guard that should have stopped it is `_nameable`, whose own docstring says "a sentence is too
much (it is a paragraph that happens to be set large)" -- and which tests only LENGTH. The
continuation is 69 characters, under the 80-char bound, so it passes.

WHY THE FIXTURE IS GEOMETRY, NOT TEXT. Asserting on the already-flattened chunk string would let
a future parser change pass because somebody adjusted the flattening. These build the overlay the
extractor actually persists -- page height, line bboxes, font sizes, bold flags, and a table's own
`cells` grid -- so the assertions run through layout classification and chunk construction from
the same input the real document produced.

STATUS: FIXED by `layout._reads_as_sentence`, and these assertions now pass. The discriminator
rejects a terminator-ending candidate unless it is numbered-heading-shaped -- calibrated over 18
documents in three families, 190 prose rejections, 190/190 adjudicated by hand, which is the
PRECISION of the rejected set and not recall over every prose fragment. The numbered exception is
a narrow compatibility carve-out for one adjudicated heading whose typography could not be
independently verified (its PDF is not available; the evidence is text-level). The residual risk it
carries is pinned separately at the bottom of this file.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from rag_core.layout import (
    BLOCK_HEADING,
    BLOCK_PARAGRAPH,
    BLOCK_TABLE,
    BLOCK_TABLE_ROW,
    LayoutBlock,
    LayoutTree,
    parse_layout,
)

_DOC = "corrigendum-5.pdf"
_PAGE_H = 792.0

# The document's own type scale, measured from the real PDF: body at 11.04 pt, the heading band
# above it. The continuation line is set in the heading band, which is exactly why typography
# alone cannot reject it and a textual guard has to.
_BODY = 11.04
_HEAD = 13.0

_SENTENCE_8_HEAD = "8. The requirement of Individual Confidentiality Declaration for Site Personnel per Sl no: D.3 of"
_SENTENCE_8_TAIL = "Appendix D - Vendor Undertaking Formats, Vol-II Part-A, is withdrawn."
#: A real heading from the same document, ending in a period. It is the one case that stops the
#: fix being "reject anything ending in a full stop" -- see `test_a_genuine_numbered_heading...`.
_REAL_HEADING_WITH_PERIOD = "4. NG-SIEM / SOAR / UEBA Solution — License Details."


def _line(text: str, *, size: float = _BODY, bold: bool = False, top: float, page: int = 1) -> dict[str, Any]:
    return {
        "block_type": "line",
        "text": text,
        "page": page,
        "bbox": [72.0, top, 540.0, top + size],
        "font_size": size,
        "bold": bold,
        "cells": None,
    }


def _table(cells: list[list[str | None]], *, text: str, top: float, page: int = 1) -> dict[str, Any]:
    return {
        "block_type": "table",
        "text": text,
        "page": page,
        "bbox": [72.0, top, 540.0, top + 20.0 * len(cells)],
        "font_size": None,
        "bold": None,
        "cells": cells,
    }


def _overlay(*pages: tuple[int, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "extractor": "pdfplumber",
        "content_hash": "",
        "pages": [
            {
                "page_no": no,
                "markup": json.dumps(
                    {
                        "format": "layout-json",
                        "version": 1,
                        "page_no": no,
                        "page_width": 612.0,
                        "page_height": _PAGE_H,
                        "blocks": blocks,
                    }
                ),
                "markup_format": "layout-json",
                "degradations": [],
            }
            for no, blocks in pages
        ],
    }


#: The amendment page as the extractor persists it: a real heading, body text whose last sentence
#: wraps, the wrapped tail set in heading type, and then the amendment table.
_AMENDMENT_PAGE = [
    _line("7. Vendor Undertaking Formats", size=_HEAD, bold=True, top=60.0),
    _line("For vendor undertaking as per Sl no: D.1 of Appendix D, the bidder shall refer", top=100.0),
    _line("to the CERT-In guideline or latest version.", top=114.0),
    _line(_SENTENCE_8_HEAD, top=140.0),
    _line(_SENTENCE_8_TAIL, size=_HEAD, bold=True, top=154.0),
    _table(
        [["Clause", "Erstwhile", "Amended to"], ["ITB 1.3.9.1", "Rs. 35 Crores", "Rs. 19.44 Crores"]],
        text="Clause | Erstwhile | Amended to\nITB 1.3.9.1 | Rs. 35 Crores | Rs. 19.44 Crores",
        top=200.0,
    ),
]


def _tree() -> LayoutTree:
    return parse_layout(_overlay((1, _AMENDMENT_PAGE)), _DOC)


def _blocks_of(tree: LayoutTree) -> list[LayoutBlock]:
    return [b for section in tree.sections for b in section.blocks]


def test_a_wrapped_sentence_is_not_a_heading() -> None:
    """The classification itself. Typography says heading; the text says continuation."""
    headings = [b.text for b in _blocks_of(_tree()) if b.block_type == BLOCK_HEADING]

    assert _SENTENCE_8_TAIL not in headings


def test_sentence_8_survives_chunk_construction_intact() -> None:
    """What the map prompt is given. The two halves must end up in ONE section's text, in order
    and adjacent -- the real failure was a chunk ending mid-clause at 'per Sl no: D.3 of', with
    the disposition in a different section the section pass never selected."""
    tree = _tree()
    joined = {
        section.section_id: " ".join(b.text for b in section.blocks if b.block_type == BLOCK_PARAGRAPH)
        for section in tree.sections
    }
    reunited = [s for s, body in joined.items() if _SENTENCE_8_HEAD in body and _SENTENCE_8_TAIL in body]

    assert reunited, "the two halves of sentence 8 are in different sections"


def test_the_disposition_reaches_the_map_with_its_clause_identity() -> None:
    """What a map call over this section would be given. `D.3` and `withdrawn` have to be in the
    same section text for any extraction to tie the clause to its disposition -- the map prompt is
    built per section, so a fact split across two of them cannot be related by the model."""
    tree = _tree()
    for section in tree.sections:
        body = " ".join(b.text for b in section.blocks)
        if "D.3" in body and "withdrawn" in body:
            return
    pytest.fail("no single section carries both the clause reference and its disposition")


# --- the safeguards: these pass TODAY and must keep passing ---------------------------------------


def test_a_genuine_numbered_heading_ending_in_a_period_is_still_a_heading() -> None:
    """THE COUNTEREXAMPLE THAT KILLED THE SIMPLE RULE. Found by running the candidate rule
    counterfactually over this document: '4. NG-SIEM / SOAR / UEBA Solution — License Details.' is
    a real section heading that ends in a full stop. Rejecting every terminator-ending candidate
    would have discarded it, which is why the discriminator needs a numbering exemption -- and why
    that exemption needs its own adversarial test below."""
    page = [
        _line(_REAL_HEADING_WITH_PERIOD, size=_HEAD, bold=True, top=60.0),
        _line("The licence counts below apply to the solution as offered.", top=100.0),
    ]
    headings = [b.text for b in _blocks_of(parse_layout(_overlay((1, page)), _DOC)) if b.block_type == BLOCK_HEADING]

    assert _REAL_HEADING_WITH_PERIOD in headings


def test_a_table_still_belongs_to_the_paragraph_above_it() -> None:
    """THE SAFEGUARD AGAINST OVERCORRECTING. A fix for the split sentence must not detach every
    table from the text that introduces it: the amendment table here belongs under the same
    section as sentence 8, and a correction that pushed tables into their own sections would
    break every "the table below shows..." relationship in the corpus."""
    tree = _tree()
    owning = [s for s in tree.sections if any(b.block_type == BLOCK_TABLE for b in s.blocks)]

    assert len(owning) == 1, "the table should sit in exactly one section"
    body = " ".join(b.text for b in owning[0].blocks)
    assert "ITB 1.3.9.1" in body, "and the table's content must survive with it"


def test_the_table_keeps_its_rows_separately_addressable() -> None:
    """ROW/COLUMN BOUNDARIES SURVIVE AS STRUCTURE, which here means one `table_row` block per data
    row alongside the whole-table block -- not one flattened string. That is what lets a later
    stage tell a header from a value; a single `=`-joined blob could not, and the linearisation
    would be inventing relationships rather than reporting them."""
    blocks = _blocks_of(_tree())
    whole = [b for b in blocks if b.block_type == BLOCK_TABLE]
    rows = [b for b in blocks if b.block_type == BLOCK_TABLE_ROW]

    assert len(whole) == 1, "the table is one retrievable unit"
    assert len(rows) == 1, "and its one DATA row is separately addressable (the header is not a row)"
    assert "ITB 1.3.9.1" in rows[0].text
    assert "Rs. 19.44 Crores" in rows[0].text
    assert "Clause" in rows[0].text, "the row carries its header labels, so it stands alone"


# --- the adversarial case for the numbering exemption ---------------------------------------------
#
# THE EXEMPTION IS THE WEAK PART OF THE CORRECTION and these are its missing evidence. Calibration
# over 18 documents in three families found 190 terminator-ending prose candidates and rejected all
# 190 correctly -- but the exemption that preserves a NUMBERED terminator-ending heading fired
# exactly once in that whole population, on `4. NG-SIEM / SOAR / UEBA Solution — License Details.`
# One real positive and no real negatives is not evidence of safety; it is an absence of evidence of
# harm, and those are different claims. So the negatives are constructed here.
#
# WRITING THEM CORRECTED TWO OF MY OWN MISTAKES, both worth keeping visible.
#
# First, the existing 80-character bound already rejects most numbered prose: "8. The requirement of
# Individual Confidentiality Declaration for Site Personnel is withdrawn." is 93 characters and
# never becomes a heading. The exemption's real exposure is SHORT numbered prose, under the bound --
# a much narrower surface than I assumed, and the tests say which side of the line each case is on.
#
# Second, a page whose every line is set in heading type yields NO candidates at all: with no size
# variation and no non-bold body line, `_classify_headings` declines to invent a hierarchy (its
# documented degenerate case). A fixture without real body text therefore proves nothing, because
# nothing is promoted for any reason. Every page below carries body lines at body size.

#: Numbered prose the EXISTING length bound already stops. Keeping them as passing tests records
#: where the current guard does work, so a future change cannot quietly give that up.
_LONG_NUMBERED_PROSE = (
    "8. The requirement of Individual Confidentiality Declaration for Site Personnel is withdrawn.",
    "7. For vendor undertaking as per Sl no: D.1 the bidder shall refer to the CERT-In guideline.",
    "2. The bidder shall submit the GTP/Data Sheet offered by the respective OEM against each item.",
    "1.2 The successful bidder shall deploy the solution within ninety days of the award.",
)

#: Numbered prose SHORT enough to pass the length bound. This is the exemption's actual exposure.
_SHORT_NUMBERED_PROSE = (
    "10.1.1 Payment shall be released against certified milestones only.",
    "8. The requirement per Sl no: D.3 is withdrawn.",
    "3. The bidder shall provide the GTP/Data Sheet.",
)


def _page_with_candidate(candidate: str) -> list[dict[str, Any]]:
    """A page where `candidate` is a genuine typographic heading candidate.

    The body lines are load-bearing: without a non-bold line at body size there is no size or
    bold contrast, `_classify_headings` yields no candidates at all, and a test over such a page
    passes for the wrong reason."""
    return [
        _line("7. Vendor Undertaking Formats", size=_HEAD, bold=True, top=60.0),
        _line("Preamble text introducing the list, set in ordinary body type.", top=100.0),
        _line("A second body line, so bold and size both have something to contrast with.", top=114.0),
        _line(candidate, size=_HEAD, bold=True, top=150.0),
    ]


def _headings(page: list[dict[str, Any]]) -> list[str]:
    tree = parse_layout(_overlay((1, page)), _DOC)
    return [b.text for b in _blocks_of(tree) if b.block_type == BLOCK_HEADING]


@pytest.mark.parametrize("prose", _LONG_NUMBERED_PROSE)
def test_long_numbered_prose_is_already_rejected_by_the_length_bound(prose: str) -> None:
    """Passes TODAY, pinned so the correction cannot regress it. `_MAX_HEADING_CHARS` is 80 and
    each of these is longer, so the existing guard stops them without any new rule -- which also
    means they are the wrong cases to judge a new rule by."""
    assert prose not in _headings(_page_with_candidate(prose))


@pytest.mark.parametrize("prose", _SHORT_NUMBERED_PROSE)
@pytest.mark.xfail(
    strict=True,
    reason="ACCEPTED RESIDUAL RISK of the numbering exception, not an unadopted fix. The exception "
    "preserves one adjudicated genuine heading and fires zero times across the other 18 documents, "
    "and no document in that population contains short numbered prose, so this shape could be "
    "neither validated nor falsified against real text. These are SYNTHETIC adversarial cases, not "
    "corpus evidence: they will flip the day the exception is tightened, which is what they are for.",
)
def test_short_numbered_prose_is_not_rescued_by_the_numbering_exemption(prose: str) -> None:
    """THE EXEMPTION'S REAL EXPOSURE. Each is under the length bound and reads as a sentence, and
    a numeric prefix is not a heading signal on its own."""
    assert prose not in _headings(_page_with_candidate(prose))


def test_nested_numbering_that_is_a_heading_stays_one() -> None:
    """The other side of the same question, and it passes today. Deep numbering is normal in these
    documents: a discriminator must not start rejecting "17.3 HCI-2 Availability and Sizing" while
    it learns to reject numbered sentences."""
    for heading in ("17.3 HCI-2 Availability and Sizing", "5.5 HCI-1 Licensing and Storage", "1.2.3 Scope"):
        assert heading in _headings(_page_with_candidate(heading)), f"genuine nested heading rejected: {heading!r}"
