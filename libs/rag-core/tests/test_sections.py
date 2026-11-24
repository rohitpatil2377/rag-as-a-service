"""Where a chunk sits in the document's structure (P14).

A chunk embedded on its own loses the heading it was written under. "Written sign-off is
required before publication" retrieves very differently depending on whether it sat under
"6.1.1 Mandatory Rehearsal Policy" or "6.4.1 Central Content Repository", and the chunk itself
carries no trace of which.

The detector is deliberately narrow, and the rule came from measuring two real corpora rather
than from a guess about what headings look like:

    Acme Content Governance SOP        HR Policy Handbook
    5.1   Process                      1. Does the employee acquire new skills quickly?
    5.1.1 Entry Criteria               1. Is the employee a critical/key resource?
    6.1.1 Mandatory Rehearsal Policy   1 Q1 - April to June 01st June 15th June
    6.3   Event and Run-of-Show Mgmt   1 H a p p in e s s L e a v e -

MULTI-LEVEL numbering ("6.1.1") is a heading. SINGLE-LEVEL ("1.") is mostly appraisal
questions, list items and table rows — on a real corpus it is far more often not a heading
than it is. A wrong section path is worse than none: it would tell the embedder a chunk
belongs to a section it does not.
"""

from __future__ import annotations

from rag_core.sections import detect_sections, section_path_at, section_path_for

_SOP = """Acme Content Governance Master SOP

5.1 Process
Content requests are received through the intake form.

5.1.1 Entry Criteria
A request is accepted only when the brief is complete.

6.1.1 Mandatory Rehearsal Policy
If the rehearsal has not been completed and signed off, the speaker does not go on stage.
No exceptions, regardless of seniority.

6.3 Event and Run-of-Show Management
The Content Curator owns the run-of-show document.
"""


def _path_for(text: str, needle: str) -> str:
    return section_path_at(detect_sections(text), text.index(needle))


def test_a_chunk_carries_the_heading_it_was_written_under() -> None:
    assert _path_for(_SOP, "does not go on stage") == "6.1.1 Mandatory Rehearsal Policy"


def test_the_path_includes_the_ancestors_that_exist() -> None:
    """ "5.1.1" under "5.1" is the whole point — the parent narrows what the child means."""
    assert _path_for(_SOP, "brief is complete") == "5.1 Process > 5.1.1 Entry Criteria"


def test_text_before_any_heading_has_no_path() -> None:
    """A title block is not a section, and inventing one would mislabel every chunk in it."""
    assert _path_for(_SOP, "Master SOP") == ""


def test_a_missing_ancestor_is_not_invented() -> None:
    """The SOP jumps straight to 6.1.1 with no "6.1" heading anywhere. The path shows what the
    document actually has, rather than fabricating the level in between."""
    assert _path_for(_SOP, "does not go on stage") == "6.1.1 Mandatory Rehearsal Policy"


# --- the refusals, measured on a real corpus ---------------------------------------------------


def test_a_numbered_list_item_is_not_a_heading() -> None:
    """Straight from the HR handbook's appraisal form. Treating these as headings would tell
    the embedder that everything after them belongs to a section called "Does the employee..."."""
    text = (
        "Appraisal questions follow.\n"
        "1. Does the employee acquire new skills quickly & effectively?\n"
        "2. Does the employee possess any leadership qualities?\n"
        "Ratings are recorded by the reporting manager.\n"
    )
    assert detect_sections(text) == []


def test_a_table_row_starting_with_a_number_is_not_a_heading() -> None:
    """Also real: "1 Q1 - April to June 01st June 15th June June" is a row of a quarter table."""
    text = "1 Q1 - April to June 01st June 15th June June\n1 LTA 1. Domestic Travel - within India\n"
    assert detect_sections(text) == []


def test_a_very_long_line_is_prose_not_a_heading() -> None:
    """A numbered clause that runs on is a clause, not a title."""
    long_clause = (
        "6.1 The Content Curator shall ensure that every speaker completes a full dress "
        "rehearsal and obtains written sign-off before the event, and shall record the "
        "outcome in the central repository together with the date and the approving party."
    )
    assert detect_sections(long_clause) == []


def test_a_document_with_no_numbering_yields_no_paths() -> None:
    """Better no context than wrong context — a chunk with no path is embedded exactly as it
    is today, which is the current behaviour and safe."""
    text = "Annual leave is 24 days.\nNotice period is 60 days.\n"
    assert detect_sections(text) == []
    assert section_path_at(detect_sections(text), 5) == ""


# --- a chunk covering two sections claims neither (P15) -----------------------------------------

_TWO = (
    "6.2.3 Social Media Content\n"
    "All content must be approved by the Communications team before release.\n"
    "6.5.1 Customer Collateral\n"
    "Sales decks may be shared with a customer directly by the account owner.\n"
)


def test_a_range_inside_one_section_takes_that_section() -> None:
    sections = detect_sections(_TWO)
    start = _TWO.index("All content")
    assert section_path_for(sections, start, start + 20) == "6.2.3 Social Media Content"


def test_a_range_spanning_two_sections_claims_neither() -> None:
    """Attributing by the START offset alone is the P39 defect in another layer: a chunk that
    runs from 6.2.3 into 6.5.1 would stamp every sentence in it "6.2.3 Social Media Content",
    and a WRONG scope is worse than no scope — it authorises exactly the misreading P15 exists
    to stop."""
    sections = detect_sections(_TWO)
    assert section_path_for(sections, _TWO.index("All content"), len(_TWO)) == ""


def test_a_range_that_starts_at_its_own_heading_is_not_spanning() -> None:
    """The common good case: the chunk opens with the heading that governs it. That heading
    begins AT the range, not inside it."""
    sections = detect_sections(_TWO)
    start = _TWO.index("6.2.3")
    assert section_path_for(sections, start, _TWO.index("6.5.1")) == "6.2.3 Social Media Content"


def test_a_range_before_any_heading_has_no_section() -> None:
    sections = detect_sections("A title block\n\n" + _TWO)
    assert section_path_for(sections, 0, 5) == ""


# --- two defects the 45-item corpus golden set exposed (2026-07-31) ------------------------------
#
# Both put a WRONG section path on a chunk, which since P15 is shown to the generator as the
# scope governing a rule. That inverts the property P14/P15 were built for: a wrong scope is
# worse than none. Measured on the real SOP and HR corpora, 8 of 38 answerable golden items get
# a section that is not theirs.


def test_a_heading_run_together_with_the_previous_line_is_still_a_heading() -> None:
    """PDF extraction routinely emits a heading on the same line as the paragraph before it. In
    the real SOP, "6.2.3 Social Media Publishing" follows "...written sign-off is on record. " on
    ONE line, so an anchored pattern never sees it — and every social-media rule in the document
    inherits "6.2.1 Website Content Updates" instead. That is the QA Q10 scope transfer, produced
    by the very code meant to prevent it."""
    text = (
        "6.2.1 Website Content Updates\nCopy is refreshed quarterly by the web team.\n"
        "f) PR agency distributes only after written sign-off is on record. 6.2.3 Social Media Publishing\n"
        "All social media content posted on Acme official accounts requires prior approval.\n"
    )
    numbers = [s.number for s in detect_sections(text)]
    assert "6.2.3" in numbers, f"mid-line heading missed; detected {numbers}"

    at = text.index("All social media content")
    assert section_path_at(detect_sections(text), at).startswith("6.2.3")


def test_a_heading_does_not_govern_arbitrarily_far_down_the_document() -> None:
    """The HR handbook numbers paragraphs sequentially ("796.", "8016. Resignation Policy"), which
    are not hierarchical headings and are correctly refused. But the last ACCEPTED heading then
    governs everything after it however distant, so the resignation notice period — tens of pages
    later — is labelled "2.2 Grievance Committee". Inheriting a heading across that distance is a
    claim about the document's structure that nothing supports."""
    # Sized to the REAL distance measured on the HR handbook (10 193 characters), not to a
    # number chosen to make the assertion pass — an over-long fixture would let the bound
    # regress to a value that no longer cuts the case it was built for.
    body = " ".join(f"Paragraph {i} of unrelated handbook text about various matters." for i in range(160))
    assert len(body) > 10_000
    text = (
        f"2.2 Grievance Committee\nThe committee meets monthly.\n{body}\n"
        "8016. Resignation Policy\nAll levels 90 days notice.\n"
    )
    at = text.index("All levels 90 days")
    assert section_path_at(detect_sections(text), at) == "", "a heading 400 paragraphs above is not this text's scope"


# --- headings a numbering-based detector cannot see ---------------------------------------------


def test_an_unnumbered_heading_is_not_detected_and_that_is_deliberate() -> None:
    """The SOP writes "Content Classification: All Acme content is classified into four
    categories." — unnumbered, and running into its own body on one line.

    A colon rule was tried and REFUSED on measurement: it found 50 "headings" in the SOP against
    32 genuine numbered ones (Prepared by, Version Number, SANJAY SHARDUL, Confidential
    Information — page furniture, table cells, glossary terms). Tightened to exclude table rows
    and short values it still produced "Marketing Head" and "All Acme employees", and by then it no
    longer matched the case it was written for. Detecting these needs layout signal (font size,
    weight, position) the text layer does not carry.

    Pinned so the attempt is not silently repeated."""
    text = (
        "1.3 Objective\nThe objective is to make governance auditable.\n"
        "Content Classification: All Acme content is classified into four categories.\n"
    )
    assert [s.label for s in detect_sections(text)] == ["1.3 Objective"]


def test_a_sentence_that_merely_contains_a_colon_is_not_a_heading() -> None:
    """ "Note: ..." and "For example: ..." are prose. A heading names a section; requiring more
    than one word in the label is what keeps this from swallowing ordinary sentences."""
    text = "2.1 Rules\nSome rule text.\nNote: this applies to everyone and is not a heading.\n"
    assert [s.number for s in detect_sections(text)] == ["2.1"]


def test_single_level_headings_are_accepted_when_the_document_numbers_its_sections() -> None:
    """The Code of Conduct numbers its sections 1..14 with no sub-levels, so refusing all
    single-level numbering left it with no structure at all."""
    text = "".join(f"{i}. Section Title {i}\nBody text for section {i} goes here.\n" for i in range(1, 6))
    numbers = [s.number for s in detect_sections(text)]
    assert numbers == ["1", "2", "3", "4", "5"]


def test_repeated_and_unordered_single_level_numbers_are_still_refused() -> None:
    """Measured on the HR handbook: its single-level candidates run 2, 5, 1, 4, 1, 1, 3 — list
    items inside sections, not sections. An ascending run is what distinguishes a document that
    numbers its sections from one that numbers its bullets."""
    text = (
        "2. Dress Code Policy:\nWear formals.\n"
        "5. Attendance Policy:\nBe punctual.\n"
        "1. Team Get Together Policy:\nBe social.\n"
        "4. LGBTQIA Policy:\nBe kind.\n"
    )
    assert [s.number for s in detect_sections(text) if "." not in s.number] == []


def test_an_appraisal_question_is_never_a_heading() -> None:
    """The HR handbook's appraisal form is a numbered list of questions."""
    text = "1. Does the employee acquire new skills quickly?\n2. Is the employee a key resource?\n"
    assert detect_sections(text) == []


# --- a heuristic heading must not be allowed to shred a list (measured on the HR handbook) ----


def _handbook_style() -> str:
    """A document whose single-level numbers genuinely ascend (so the escape hatch opens) AND
    which also contains an ordinary enumerated list. Real shape: the Acme HR handbook numbers its
    policies 1..26 and also writes "Core Values" as six numbered items."""
    policies = "".join(
        f"{n}. Policy Number {n}\nBody text for policy {n}. " + "Filler. " * 12 + "\n" for n in range(1, 9)
    )
    return (
        policies + "Core Values\n"
        "The six simple ingredients to a successful workplace are more than buzzwords.\n"
        "1. Exuberant Customer Support:\nWe support our employees to the fullest.\n"
        "2. Proactive Approach:\nWe bestow tasks that are challenging.\n"
        "3. Continuous Knowledge Enhancement:\nWe encourage an environment of learning.\n"
        "4. Organization above self:\nOur people go beyond self and team.\n"
        "5. Respect and Care:\nIndividual opinion and expression are valued.\n"
        "6. Love for Environment:\nIt is our duty to give back to society.\n"
    )


def test_a_single_level_number_is_detected_but_not_structural() -> None:
    """The module's own docstring: single-level numbering "is far more often an appraisal
    question, a list item or a table row". The ascending-run escape hatch re-admits it document
    WIDE, so on a handbook that numbers its policies 1..26 every numbered LIST item anywhere in
    the document was promoted to a heading too. Keep them for labelling — many are real — but
    mark them unreliable so nothing structural is decided on them."""
    secs = detect_sections(_handbook_style())
    single = [s for s in secs if "." not in s.number]
    assert single, "the ascending run should still open the escape hatch"
    assert not any(s.structural for s in single), "a single-level guess must not be structural"


def test_a_multi_level_number_stays_structural() -> None:
    """The reliable case the detector was built for must be unaffected: "6.2.3 Social Media
    Content" is what scopes a rule, and it still has to define a boundary."""
    secs = detect_sections("Intro text.\n6.2.3 Social Media Content\nAll content must be approved.\n")
    assert secs and all(s.structural for s in secs)
