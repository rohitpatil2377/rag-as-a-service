"""Tables whose cells are merged or wrapped, as a real PDF extractor hands them over.

Fixture is the verbatim pdfplumber extraction of the release-history table in Acme's HR Handbook
4.0. It shows the two shapes that made that table unreadable once indexed:

* MERGED cells become phantom all-empty columns — 10 of the 15 here — so every serialized row
  carried "| | |" where a value should be, and the labels no longer lined up with the values.
* A WRAPPED row is split across several physical rows: version 3.0's dates arrive as "1st",
  then "December", then "2024" on three separate rows, with the columns left None.

Measured consequence before the fix: "Who approved version 4.0 of the HR handbook?" abstained,
with the answer sitting in the document.
"""

from __future__ import annotations

from svc_doc_intelligence.tables import parse_table_row, serialize_table

# The raw rows, exactly as pdfplumber returns them.
RELEASE_HISTORY: list[list[str | None]] = [
    [
        "Version",
        "Prepared\nDate",
        None,
        None,
        "Review\nDate",
        None,
        None,
        "Release\nDate",
        None,
        None,
        "Prepared\nBy",
        "Reviewed\nBy",
        None,
        None,
        "Approved\nBy",
    ],
    [
        "1.0",
        "1st\nDecember\n2022",
        None,
        None,
        "15th\nDecember\n2022",
        None,
        None,
        "1st\nJanuary\n2023",
        None,
        None,
        "HR\nOperations",
        "HR\nOperations\n- Head",
        None,
        None,
        "CHRO",
    ],
    ["3.0", "", "1st", "", "", "15th", "", "", "1st", "", "HR\nOperations", "", "HR", "", "CHRO"],
    [
        None,
        None,
        "December",
        None,
        None,
        "December",
        None,
        None,
        "January",
        None,
        None,
        None,
        "Operations-",
        None,
        None,
    ],
    [None, None, "2024", None, None, "2024", None, None, "2025", None, None, None, "Head", None, None],
    [
        "4.0",
        "1st\nDecember\n2025",
        None,
        None,
        "15th\nDecember\n2025",
        None,
        None,
        "1st\nJanuary\n2026",
        None,
        None,
        "HR\nOperations",
        "HR\nOperations-\nHead",
        None,
        None,
        "VP-\nHR/CHRO",
    ],
]


def _row_for(serialized: str, version: str) -> dict[str, str]:
    """The serialized line for one version, as {label: value}."""
    for line in serialized.splitlines():
        fields = parse_table_row(line)
        if fields and fields[0][1] == version:
            return {label or "": value for label, value in fields}
    raise AssertionError(f"version {version} has no row of its own in:\n{serialized}")


def test_a_value_is_labelled_with_its_own_column() -> None:
    """The question that failed live: who approved version 4.0. It is answerable only if the
    approver is attached to the Approved By column and not left to positional alignment."""
    row = _row_for(serialize_table(RELEASE_HISTORY), "4.0")
    assert row["Approved By"] == "VP- HR/CHRO"
    assert row["Release Date"] == "1st January 2026"
    assert row["Reviewed By"] == "HR Operations- Head"


def test_a_wrapped_row_keeps_its_labels_even_though_it_stays_split() -> None:
    """KNOWN LIMIT, pinned deliberately. Version 3.0's date wraps onto three physical rows, and
    nothing in a list of strings distinguishes that from a table whose first column is merged
    across several real records:

        North | Delhi | 30          3.0 | 1st   | 15th
              | Agra  | 15              | Dec   | Dec

    Both are "a row with an empty first column". Joining them would repair the first and corrupt
    the second into "Delhi Agra | 30 15", so the fragments stay separate — but each one is now
    LABELLED and they stay adjacent, which is what lets a reader (and the model) see that
    December belongs to the prepared date. Before the span fold they were bare "| | 1st | |"."""
    serialized = serialize_table(RELEASE_HISTORY)
    row = _row_for(serialized, "3.0")
    assert row["Prepared Date"] == "1st"
    assert row["Approved By"] == "CHRO"
    assert "Prepared Date = December" in serialized
    assert "Release Date = 2025" in serialized


def test_phantom_columns_from_merged_cells_are_gone() -> None:
    serialized = serialize_table(RELEASE_HISTORY)
    assert "|  |" not in serialized, "an empty column between two values is a merged-cell artefact"
    assert ": :" not in serialized


def test_every_version_that_arrived_intact_gets_a_row_of_its_own() -> None:
    serialized = serialize_table(RELEASE_HISTORY)
    starts = [parse_table_row(line)[0] for line in serialized.splitlines() if parse_table_row(line)]
    assert [value for label, value in starts if label == "Version"] == ["1.0", "3.0", "4.0"]


def test_a_table_whose_header_names_every_column_is_untouched() -> None:
    """The guard on the fix: the fold only ever acts on GAPS in a header. An ordinary table has
    none, so it must come out exactly as it did before — values under their own labels."""
    rows: list[list[str | None]] = [
        ["Name", "Joined", "Leave Days"],
        ["Asha", "2024-01-05", "12"],
        ["Dana", "2024-03-11", "4"],
    ]
    out = serialize_table(rows)
    assert _row_for(out, "Asha") == {"Name": "Asha", "Joined": "2024-01-05", "Leave Days": "12"}
    assert _row_for(out, "Dana")["Leave Days"] == "4"


def test_a_column_no_row_fills_is_dropped_without_shifting_the_others() -> None:
    """An extractor's phantom column carries nothing. Serialized it read as a missing value —
    "Asha |  | 12" — which says the leave days are unknown rather than that the column is not
    real. Dropping it must not slide the next value under its neighbour's label."""
    rows: list[list[str | None]] = [
        ["Name", "Joined", "Leave Days"],
        ["Asha", "2024-01-05", "12"],
        ["Dana", "2024-03-11", "4"],
    ]
    with_phantom: list[list[str | None]] = [[r[0], None, r[1], None, r[2]] for r in rows]
    assert serialize_table(with_phantom) == serialize_table(rows)


def test_a_table_row_boundary_is_not_a_hyphenated_line_break() -> None:
    """Found in the re-indexed handbook: "Reviewed By = Operations-" and the row after it were
    welded into "OperationsPrepared Date = 2024". De-hyphenation joins a word split across a line
    break ("alloca-\\ntion"), and a serialized table row can legitimately END in a hyphen — the
    org name is "HR Operations- Head". The newline after it separates two records, not two
    halves of a word."""
    from svc_doc_intelligence.ocr import _dehyphenate

    rows = "Version = 3.0 | Reviewed By = Operations-\nPrepared Date = 2024 | Review Date = 2024"
    assert _dehyphenate(rows) == rows

    # ...and the case de-hyphenation exists for still works.
    assert _dehyphenate("The alloca-\ntion increased.") == "The allocation increased."
