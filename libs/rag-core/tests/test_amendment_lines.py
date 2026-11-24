"""An amendment's relationship is PRINTED, not asked of the model.

THE DEFECT THIS CLOSES, isolated boundary by boundary on the real AEGCL corrigendum. All three of
its amendments survive ingestion and map extraction with `is_amendment`, `old_value` and
`new_value` populated, all three reach the same reduce prompt with `AMENDMENT: <old> -> <new>`
rendered verbatim in the items block, and `reduce.md` rule 3 tells the model to show the change.

Two of them come back correct, and they are the two whose sides are SHORT:

    MAAT  "amended from ₹35 Crores → ₹19.44 Crores"      a value pair
    D.3   "... Vol-II Part-A is withdrawn"                a disposition

The third has two sides that are each ~200-character competing requirements, and in BOTH captured
attempts the model reported the replacement and silently dropped what it replaced. The current
requirement was right; the fact that it superseded anything was gone -- which for a bidder reading
a tender is the part that matters, because an answer that states only the current rule gives them
no way to tell their old assumption is stale.

WHY CODE RATHER THAN MORE PROMPT. The relationship is already structured data on the item. This
module already makes the argument for the coverage line -- "making the model restate them is a
weaker guarantee than printing them" -- and a 200-character pair carried through a summarisation
step is exactly that weaker guarantee, now with two measurements behind it. The prompt is
unchanged: rule 3 still works for the short shapes, and nothing was added to compensate for the
long one.
"""

from __future__ import annotations

from rag_core.mapreduce import Coverage, MappedItem, ReduceInput, amendment_lines, assemble_answer


def _item(item_id: str, *, old: str | None, new: str | None, amend: bool = True) -> MappedItem:
    return MappedItem(
        item_id=item_id,
        section_id="sec-amend",
        doc_id="corrigendum-5.pdf",
        facet_id="technical_compliance_documentation",
        statement="statement",
        quote="quote",
        is_amendment=amend,
        old_value=old,
        new_value=new,
    )


# The en dashes below are VERBATIM from the corrigendum. These constants are the document's own
# wording, which is the point of them, so RUF001 is silenced rather than the text normalised.
_BDS_OLD = (
    "The Bidder's compliance to the Data Requirement Sheet (DRS – Volume-II, Part B, Appendix D) "  # noqa: RUF001
    "shall be submitted on the letterhead of the respective OEM, duly signed by an authorized "
    "signatory of that OEM"
)
_BDS_NEW = (
    "The bidder shall offer their solution against the requirements of DRS Data Requirement Sheet "
    "(DRS – Volume-II, Part B, Appendix D) and shall also provide the GTP/Data Sheet offered by "  # noqa: RUF001
    "the respective OEM(s) against each requirement"
)
_D3_OLD = "Individual Confidentiality Declaration for Site Personnel per Sl no: D.3 of Appendix D"


def test_1_a_long_form_amendment_keeps_both_sides_and_their_relationship() -> None:
    """THE FAILURE THIS EXISTS FOR. Two ~200-character requirements: the model kept the
    replacement and dropped the superseded side in both captured attempts."""
    lines = amendment_lines([_item("m4", old=_BDS_OLD, new=_BDS_NEW)])

    assert len(lines) == 1
    assert "letterhead of the respective OEM" in lines[0], "the superseded requirement"
    assert "GTP/Data Sheet" in lines[0], "the requirement that controls now"
    assert "→" in lines[0], "and that one replaced the other"


def test_2_a_numerical_amendment_is_unchanged() -> None:
    """MAAT already worked through the model. It must keep working through here."""
    lines = amendment_lines([_item("m3", old="Rs. 35 Crores", new="Rs. 19.44 Crores")])

    assert lines == ["- **Rs. 35 Crores** → **Rs. 19.44 Crores** [m3]"]


def test_3_a_withdrawal_stays_connected_to_what_was_withdrawn() -> None:
    """D.3. "withdrawn" on its own is the finding that produced a FALSE POSITIVE in the
    supersession metric -- the bare word appeared elsewhere in the document, about a different
    clause. The disposition has to name its requirement."""
    lines = amendment_lines([_item("m2", old=_D3_OLD, new="withdrawn")])

    assert len(lines) == 1
    assert "D.3" in lines[0]
    assert "withdrawn" in lines[0]


def test_4_one_reported_amendment_cannot_conceal_another() -> None:
    """All three in one batch, which is how they actually arrive. A successful MAAT line must not
    be allowed to stand in for a missing BDS line -- that is precisely how the single aggregate
    `supersession_ok` hid two different failures behind one number."""
    lines = amendment_lines(
        [
            _item("m2", old=_D3_OLD, new="withdrawn"),
            _item("m3", old="Rs. 35 Crores", new="Rs. 19.44 Crores"),
            _item("m4", old=_BDS_OLD, new=_BDS_NEW),
        ]
    )

    assert len(lines) == 3
    joined = "\n".join(lines)
    assert "D.3" in joined and "Rs. 19.44 Crores" in joined and "GTP/Data Sheet" in joined


def test_5_a_missing_side_is_said_rather_than_invented() -> None:
    """The map records a missing old side when it did not capture what the new value changed.
    Supplying one would be the platform asserting a requirement no document contains -- the same
    error as crediting an amendment because the word "withdrawn" appeared somewhere."""
    lines = amendment_lines([_item("m9", old=None, new="Rs. 50 Crores")])

    assert len(lines) == 1
    assert "not in the extracted evidence" in lines[0]
    assert "→" not in lines[0], "no arrow, because there is no left-hand side"


def test_a_missing_side_is_described_as_absent_evidence_not_an_absent_requirement() -> None:
    """WHAT THE RENDERER MAY CLAIM. An empty `old_value` is a fact about this extraction, not about
    the source: it means the previous wording was not captured, which is true both when the
    document never stated one and when the map missed it. "Previously unstated" asserts the first,
    which nothing here establishes -- so the deterministic path would be inventing history while
    preventing the model from inventing it. Holds for a withdrawal with no named requirement too,
    where the temptation is the same."""
    replaced = amendment_lines([_item("m9", old=None, new="Rs. 50 Crores")])[0]
    withdrawn = amendment_lines([_item("m9", old=None, new="withdrawn")])[0]

    for line in (replaced, withdrawn):
        assert "not in the extracted evidence" in line
        assert "unstated" not in line, "a claim about the document, from the absence of a field"
        assert "never" not in line and "no previous" not in line


def test_an_amendment_with_no_replacement_at_all_is_not_reported() -> None:
    """Neither a replacement nor a disposition is nothing to say. Printing "amended" alone would
    tell a reader a change exists while withholding what it was."""
    assert amendment_lines([_item("m9", old="Rs. 35 Crores", new=None)]) == []


def test_several_items_describing_one_change_are_one_line() -> None:
    """The map returns one item per distinct requirement, so a long amendment arrives as two or
    three items carrying the SAME pair -- m4 and m5 did. They are one change to a reader."""
    lines = amendment_lines([_item("m4", old=_BDS_OLD, new=_BDS_NEW), _item("m5", old=_BDS_OLD, new=_BDS_NEW)])

    assert len(lines) == 1
    assert "[m4,m5]" in lines[0], "and both items are cited, so either resolves the citation"


def test_the_assembled_answer_carries_the_change_log_as_its_own_block() -> None:
    """An amendment belongs to the DOCUMENT's history, not to one topic: the same change can bear
    on several facets, and a reader checking whether their assumption is stale should not have to
    find the right heading first."""
    data = ReduceInput(
        query="list every requirement",
        profiles="- corrigendum-5.pdf",
        items=[_item("m4", old=_BDS_OLD, new=_BDS_NEW)],
        coverage=Coverage(used=1, selected=1, total=1),
        answer_lang="English",
    )
    answer = assemble_answer(data, [("technical_compliance_documentation", "- A requirement [m4]")])

    assert "### amendments in this document" in answer
    assert "letterhead of the respective OEM" in answer
    assert answer.index("### technical compliance documentation") < answer.index("### amendments")
    assert answer.index("### amendments") < answer.index("Coverage:"), "before the coverage line"


def test_an_answer_with_no_amendments_gains_no_empty_block() -> None:
    data = ReduceInput(
        query="q",
        profiles="- doc.pdf",
        items=[_item("m1", old=None, new=None, amend=False)],
        coverage=Coverage(used=1, selected=1, total=1),
        answer_lang="English",
    )
    answer = assemble_answer(data, [("facet", "- A requirement [m1]")])

    assert "amendments" not in answer
