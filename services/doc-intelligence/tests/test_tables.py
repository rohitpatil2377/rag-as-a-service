"""Table serialization: self-describing rows that survive a chunk boundary (FR-B2).

Positional rows lose their meaning the moment a chunk starts mid-table — "Officer | 24" is a
number with no idea what it measures, which is the shape of our table-retrieval failures.
Rows therefore carry their column label when a header can be identified CONFIDENTLY, and fall
back to positional when it cannot: a mislabelled cell is worse than an unlabelled one.

This module is extractor-agnostic on purpose — a spreadsheet row is the same problem.
"""

from __future__ import annotations

import pytest
from svc_doc_intelligence.tables import (
    CELL_SEP,
    detect_header_rows,
    parse_table_row,
    serialize_table,
)

_WITH_HEADER = [["Grade", "Days"], ["Officer", "24"], ["Staff", "18"], ["Trainee", "12"]]


# --- header identification: high precision, safe refusal ------------------------------------


def test_a_real_header_is_identified() -> None:
    """Row 0 is all text, the column below it is all numeric — a type discontinuity."""
    assert detect_header_rows(_WITH_HEADER) == 1
    assert serialize_table(_WITH_HEADER).splitlines()[0] == "Grade = Officer | Days = 24"


def test_a_row_header_table_is_not_mistaken_for_a_column_header() -> None:
    """Labels in the FIRST COLUMN, no header row. Treating row 0 as a header would emit
    "Basic Salary = HRA | 50000 = 20000" — every cell mislabelled."""
    rows = [["Basic Salary", "50000"], ["HRA", "20000"], ["PF", "6000"]]
    assert detect_header_rows(rows) == 0
    assert serialize_table(rows).splitlines()[0] == "Basic Salary | 50000"


def test_a_table_whose_first_row_is_data_gets_no_labels() -> None:
    rows = [["2024", "100"], ["2025", "120"], ["2026", "140"]]
    assert detect_header_rows(rows) == 0
    assert ":" not in serialize_table(rows)


def test_an_all_text_table_gets_no_invented_labels() -> None:
    """Nothing distinguishes a header from a first data row here, so we decline. Unlabelled
    text rows are still readable; wrong labels would not be."""
    rows = [["Name", "Department"], ["Anita", "Finance"], ["Ravi", "Operations"]]
    assert detect_header_rows(rows) == 0
    assert serialize_table(rows).splitlines() == ["Name | Department", "Anita | Finance", "Ravi | Operations"]


def test_a_multi_row_header_is_declined_rather_than_half_read() -> None:
    """Taking only row 0 of a two-row header would label the Days column "Annual"."""
    rows = [["Employee", "Annual"], ["Name", "Days"], ["Anita", "24"], ["Ravi", "18"]]
    assert detect_header_rows(rows) == 0
    assert ":" not in serialize_table(rows)


def test_a_caller_with_better_evidence_can_assert_the_header() -> None:
    """The seam for a format that KNOWS (spreadsheet frozen panes, bold PDF header rows):
    an explicit header_rows overrides the heuristic, including multi-row headers."""
    rows = [["Employee", "Annual"], ["Name", "Days"], ["Anita", "24"]]
    assert serialize_table(rows, header_rows=2).splitlines() == ["Employee Name = Anita | Annual Days = 24"]


def test_a_header_with_a_blank_cell_is_declined() -> None:
    """A gap in row 0 usually means a column spanned from a second header row, so the
    heuristic will not guess which. Declined -> positional."""
    assert detect_header_rows([["Grade", ""], ["Officer", "24"], ["Staff", "18"]]) == 0


def test_an_asserted_header_with_a_blank_cell_labels_only_what_it_can() -> None:
    """When a caller asserts the header anyway, an unnamed column degrades to positional for
    that field alone rather than emitting an empty label."""
    rows = [["Grade", ""], ["Officer", "24"], ["Staff", "18"]]
    assert serialize_table(rows, header_rows=1).splitlines()[0] == "Grade = Officer | 24"


def test_two_row_tables_are_left_positional() -> None:
    """One body row is not enough evidence of a column type — and a two-row table is far too
    short to be split by any chunker, so labels would buy nothing."""
    assert detect_header_rows([["Grade", "Days"], ["Officer", "24"]]) == 0


# --- the separator inside cell content -------------------------------------------------------


def test_a_cell_containing_the_separator_stays_unambiguous() -> None:
    rows = [["Grade", "Days", "Range"], ["Officer", "24", "20 | 24"], ["Staff", "18", "16 | 18"]]
    line = serialize_table(rows).splitlines()[0]
    assert line.count(CELL_SEP) == 2  # two real field boundaries, not three
    assert parse_table_row(line) == [("Grade", "Officer"), ("Days", "24"), ("Range", "20 | 24")]


def test_a_cell_containing_a_backslash_round_trips() -> None:
    rows = [["Grade", "Days", "Path"], ["Officer", "24", r"a\b | c"], ["Staff", "18", r"d\e"]]
    assert parse_table_row(serialize_table(rows).splitlines()[0]) == [
        ("Grade", "Officer"),
        ("Days", "24"),
        ("Path", r"a\b | c"),
    ]


def test_a_separator_in_a_positional_row_is_escaped_too() -> None:
    """Escaping is a property of the format, not of the labelled path."""
    rows = [["Basic Salary", "20 | 24"], ["HRA", "10 | 12"], ["PF", "5 | 6"]]
    line = serialize_table(rows).splitlines()[0]
    assert parse_table_row(line) == [(None, "Basic Salary"), (None, "20 | 24")]


def test_a_label_containing_the_label_separator_stays_unambiguous() -> None:
    rows = [["Note: grade", "Days"], ["Officer", "24"], ["Staff", "18"]]
    assert parse_table_row(serialize_table(rows).splitlines()[0]) == [("Note: grade", "Officer"), ("Days", "24")]


def test_positional_rows_parse_too() -> None:
    rows = [["Basic Salary", "50000"], ["HRA", "20000"], ["PF", "6000"]]
    assert parse_table_row(serialize_table(rows).splitlines()[0]) == [(None, "Basic Salary"), (None, "50000")]


# --- shape guards -----------------------------------------------------------------------------


def test_a_table_too_short_to_be_a_table_renders_as_nothing() -> None:
    assert serialize_table([["Grade", "Days"]]) == ""
    assert serialize_table([]) == ""


def test_blank_rows_are_dropped_before_anything_else() -> None:
    rows: list[list[str | None]] = [["Grade", "Days"], [None, None], ["Officer", "24"], ["", ""], ["Staff", "18"]]
    assert len(serialize_table(rows).splitlines()) == 2


def test_empty_cells_are_omitted_from_a_labelled_row() -> None:
    rows = [["Grade", "Days", "Notes"], ["Officer", "24", ""], ["Staff", "18", "part time"]]
    assert serialize_table(rows).splitlines()[0] == "Grade = Officer | Days = 24"


@pytest.mark.parametrize("header_rows", [None, 1])
def test_the_header_row_is_not_emitted_separately_when_labels_carry_it(header_rows: int | None) -> None:
    """Its content is on every row already; emitting it again would be the duplication we
    just removed from the page text."""
    out = serialize_table(_WITH_HEADER, header_rows=header_rows)
    assert out.splitlines()[0] != "Grade | Days"
    assert len(out.splitlines()) == 3  # three body rows


# --- the label separator must not be a chunk boundary (P52) ------------------------------------


def test_a_labelled_row_is_not_split_between_a_label_and_its_value() -> None:
    """The serialized row is what reaches the chunker, so its separators have to survive
    `_CLAUSE_RE` — the same emergent coupling that makes PAGE_SEPARATOR load-bearing.

    `": "` was a colon followed by whitespace, which is exactly what the clause chunker splits
    on, so "Grade: Officer | Days: 24" became ['Grade:', 'Officer | Days:', '24']. It
    reconstructs exactly, so nothing was corrupted — but a chunk boundary could land between a
    label and its value, leaving "Days:" in one chunk and "24" in the next. That is a smaller
    version of the bare-"24" failure self-describing rows were built to prevent.

    This test imports the real regex rather than restating it, so the two cannot drift."""
    from rag_core.chunking import _CLAUSE_RE

    row = serialize_table(_WITH_HEADER).splitlines()[0]
    assert _CLAUSE_RE.split(row) == [row], f"row is splittable: {_CLAUSE_RE.split(row)}"


def test_a_serialized_table_is_one_atomic_clause_unit() -> None:
    """Measured consequence of the separator change, recorded rather than assumed.

        before: ['Grade:', 'Officer | Days:', '24\\nGrade:', 'Staff | Days:', '18']
        after:  ['Grade = Officer | Days = 24\\nGrade = Staff | Days = 18']

    `_CLAUSE_RE` needs whitespace AFTER its lookbehind character, and rows are joined by a
    single "\\n" followed immediately by a letter — so a row boundary was never a clause
    boundary. The only split points inside a table came from ": ", and they fell in the WRONG
    place: between a label and its value rather than between rows.

    So the table now stays whole, which is right — but it means a long table is a single
    indivisible unit for the clause chunker, and can only be broken by row-aligned chunking.
    That is the oversized-chunk problem, not this one."""
    from rag_core.chunking import _CLAUSE_RE

    rendered = serialize_table(_WITH_HEADER)
    assert _CLAUSE_RE.split(rendered) == [rendered]


def test_the_format_still_round_trips() -> None:
    """`parse_table_row` is the canonical inverse (I9); changing a separator must not break it."""
    row = serialize_table(_WITH_HEADER).splitlines()[0]
    assert parse_table_row(row) == [("Grade", "Officer"), ("Days", "24")]
