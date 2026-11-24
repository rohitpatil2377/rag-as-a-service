"""A layout grid is not a table (P45 follow-on).

Word writes page layout as a table: one paragraph per row, ruled rectangles around it, and
mostly empty cells. pdfplumber's `find_tables()` reports it exactly as it reports a real one,
and since a detected table's characters are now REMOVED from the narrative and replaced by
rendered rows, accepting it does not add noise beside the prose — it shreds the prose.

Measured on the real corpus, `Acme_Business_Travel_Policy` came back 99.7% table by character
across all 8 pages, with page 1 detected as 46 rows x 3 columns and 72% of its cells empty.
A genuine table on the same corpus (HR handbook, a grade/entitlement grid) is 4 rows x 7
columns with 0% empty.

The rule that separates them needs no threshold and is not a classifier. Serializing a table
buys exactly one thing: the ASSOCIATION between the values on a row ("Grade = Officer |
Days = 24"). A grid whose rows do not, on average, carry more than one value has no
associations to express, so the serialized form conveys nothing the plain text did not — and
costs reading order to produce.
"""

from __future__ import annotations

from svc_doc_intelligence.tables import serialize_table

# Page 1 of the travel policy in miniature: three columns, one populated per row.
_LAYOUT_GRID = [
    ["Business Travel Policy Acme Software Solution Limited", "", ""],
    ["Purpose of the Policy:", "", ""],
    ["It defines the requirements which apply to employees", "", "Third"],
    ["Parties who expense funds on behalf of the company.", "", ""],
]
_ONE_COLUMN = [["Respect and Care for Fellow Acme employees"], ["Love for Environment"], ["Alignment with Values"]]
_REAL_TABLE = [["Grade", "Days"], ["Officer", "24"], ["Staff", "18"]]


def test_a_layout_grid_is_not_serialized() -> None:
    """Its rows would render as prose fragments split mid-sentence with empty cells between
    them — which is what the corpus actually contained."""
    assert serialize_table(_LAYOUT_GRID) == ""


def test_a_single_column_grid_is_not_a_table() -> None:
    """With one column there is nothing to associate a value WITH: the serialized form is the
    original text plus separators. Word uses these constantly as page containers."""
    assert serialize_table(_ONE_COLUMN) == ""


def test_a_real_table_is_unaffected() -> None:
    assert "Grade = Officer | Days = 24" in serialize_table(_REAL_TABLE)


def test_a_sparse_but_genuine_table_survives() -> None:
    """The rule is about association, not about density: a table with gaps still relates the
    values that ARE present, so it must not be caught."""
    rows = [["Grade", "Days", "Notes"], ["Officer", "24", ""], ["Staff", "18", "pro rata"]]
    assert "Grade = Officer | Days = 24" in serialize_table(rows)


def test_a_declared_grid_is_still_serialized() -> None:
    """A spreadsheet column and a w:tbl are tables by declaration. A one-column sheet is one
    column of real data, and this rule is about what DETECTION guessed."""
    assert serialize_table(_ONE_COLUMN, declared=True) != ""
