"""Table serialization — rows in, text out (FR-B2).

Deliberately extractor-agnostic: a PDF table, a spreadsheet sheet and a .docx ``w:tbl`` are
the same problem, so this takes ``list[list[str | None]]`` and knows nothing about where the
rows came from. Anything needing page geometry stays in the extractor.

**Why rows carry labels.** A positional row loses its meaning the moment a chunk begins
mid-table: ``Officer | 24`` is a number with no idea what it measures, which is the shape of
our table-retrieval failures. Measured at the shipped chunk sizes, a table's body is
separated from its header row in 32-50% of realistic cases, so this is not an edge case. A
labelled row survives being cut out of its table entirely.

**Why the header is not simply repeated every N rows.** It would be cheaper, but it
re-introduces the duplication we just removed from page text (inflating BM25 term frequency
for header terms), it still requires positional column alignment inside the fragment, and the
interval N is only meaningful relative to a chunk size the extractor cannot see.

**Why labelling can refuse.** Nothing in a row of strings marks a header — row 0 is not
reliably one. Labels are emitted only when a header is identified with high confidence; every
other shape falls back to positional rows. A mislabelled cell is worse than an unlabelled one.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

CELL_SEP = " | "  # between fields of one row
# Between a field's column label and its value. NOT a free choice, and not ": ".
#
# A serialized row is text that goes on to be chunked, so its separators have to survive
# `rag_core.chunking._CLAUSE_RE` — ``(?<=[.;:\n])\s+``, which splits on whitespace
# preceded by a colon. ": " is precisely that, so "Grade: Officer | Days: 24" split into
# ['Grade:', 'Officer | Days:', '24'] and a chunk boundary could land between a label and its
# value, leaving "Days:" in one chunk and "24" in the next — a smaller version of the bare-"24"
# failure that self-describing rows exist to prevent.
#
# "=" is not in that lookbehind class, reads unambiguously as label-to-value, and is ASCII.
# The row-joining newline IS in the class, which is what we want: rows stay separable from each
# other. Same kind of emergent coupling as PAGE_SEPARATOR — a code change with a test, not a
# knob. Pinned by `test_a_labelled_row_is_not_split_between_a_label_and_its_value`.
LABEL_SEP = " = "

# A one-row "table" is layout noise, not data — when the table was DETECTED. pdfplumber infers
# tables from ruling lines and whitespace, and a single row it found in running prose is almost
# always a false positive; rendering it would restructure narrative text.
#
# This is a rule about detection confidence, so it does not apply to a table the FORMAT states
# is a table: a spreadsheet sheet and a w:tbl element are grids by declaration, and a one-row
# grid there is one row of real data. Callers say which they have via `declared=`. Kept here
# rather than in each extractor so the policy — and therefore what `tables_found` means — is
# the same everywhere.
MIN_TABLE_ROWS = 2

# Minimum rows before a header can be identified: one header plus TWO body rows, because a
# single body cell cannot establish a column's type. A two-row table is also far too short for
# any chunker to split, so declining costs nothing there.
_MIN_ROWS_FOR_HEADER = 3
_MIN_TYPED_BODY_CELLS = 2  # a column needs this many typed cells to count as evidence
_LABELLED_FIELD_PARTS = 2  # a field splits into exactly label + value when it carries a label

# A cell that reads as a quantity: optional currency word ("Rs."), sign, symbol, digits with
# separators, optional percent. Deliberately requires a digit, so "Yes"/"N/A" are text.
_NUMERIC_RE = re.compile(r"^\(?\s*(?:[A-Za-z]{1,3}\.?\s*)?[-+]?\s*[₹$€£]?\s*\d[\d,\s]*(?:\.\d+)?\s*%?\.?\s*\)?$")
_DATE_RE = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\s*$")


def _cell(value: str | None) -> str:
    return (value or "").strip().replace("\n", " ")


def _is_typed(text: str) -> bool:
    """Does this cell read as a quantity or a date rather than prose? The signal a header
    sits above — a text label over a column of numbers."""
    return bool(_NUMERIC_RE.match(text) or _DATE_RE.match(text))


def collapse_spans(rows: Sequence[Sequence[str | None]]) -> list[list[str | None]]:
    """Fold a MERGED-CELL grid back into one column per header cell.

    An extractor reading a table whose header cells span several sub-columns emits the grid, not
    the table: the header names column 1 and leaves 2 and 3 empty, and each body value lands in
    whichever sub-column it happened to be drawn in. Measured on Acme's HR Handbook release
    history, that produced a 15-column grid for a 7-column table, where one version's date sat in
    sub-column 2 and another's in sub-column 1 of the same field.

    Two consequences, both fatal to reading the table: ``detect_header_rows`` requires a header
    that names every column and so declined outright (the whole table fell back to positional
    rows), and even positionally the values no longer lined up with each other row to row.

    A gap in the header is not an unnamed column — it is the column to its left continuing, which
    is what a spanned cell IS. So each run of header gaps folds into the header cell that opens
    it, and every body row folds the same way, joining whatever non-empty pieces it holds in that
    span. Columns beyond the last named one keep their own identity: they are not part of any
    span, and merging them into the final header would put a value under a label it never had.

    Left alone when the header has no gaps — the ordinary table is untouched."""
    table = [list(r) for r in rows]
    if not table:
        return table
    width = max(len(r) for r in table)
    head = [_cell(table[0][i]) if i < len(table[0]) else "" for i in range(width)]
    if not head or not head[0] or all(head[:width]):
        return table  # no spans to fold (or nothing to anchor them to)

    # Each group is the run of sub-columns one header cell owns.
    groups: list[list[int]] = []
    for index, label in enumerate(head):
        if label or not groups:
            groups.append([index])
        else:
            groups[-1].append(index)

    # A gap is only a SPAN if the columns under it are alternative placements of ONE value: in a
    # spanned field every row fills at most one of them (which sub-column depends on how the row
    # was drawn). A row that fills two at once is telling us they are different fields and the
    # header simply failed to name the second — an unnamed column, not a span. Folding there
    # would concatenate two real values ("Officer 24"), so that group is left alone.
    groups = [g for group in groups for g in ([group] if _is_span(group, table) else [[i] for i in group])]

    folded: list[list[str | None]] = []
    for row in table:
        merged: list[str | None] = []
        for group in groups:
            pieces = [_cell(row[i]) for i in group if i < len(row) and _cell(row[i])]
            merged.append(" ".join(pieces) if pieces else None)
        folded.append(merged)
    return folded


def detect_header_rows(rows: Sequence[Sequence[str | None]]) -> int:
    """How many leading rows form a column header — ``0`` when none can be identified.

    High precision by design; recall is deliberately sacrificed. The evidence required is a
    TYPE DISCONTINUITY: row 0 is entirely non-empty prose, and some column beneath it holds
    nothing but quantities or dates. That single rule is what separates the shapes that would
    otherwise be confused:

    * **row-header tables** (labels down the first column, ``Basic Salary | 50000``) — row 0
      contains a quantity, so it is not prose, so no header;
    * **first row is data** (``2024 | 100``) — same;
    * **multi-row headers** (``Employee/Name``, ``Annual/Days``) — the second header row puts
      prose in the numeric column, so no column is cleanly typed, so no header. Declined
      rather than half-read, which would label the values with only the first row's words;
    * **all-text tables** — no column is typed, so nothing distinguishes a header from a first
      data row. Declined; unlabelled text rows still read sensibly.

    A caller that genuinely knows better (spreadsheet frozen panes, bold header rows in a PDF)
    passes ``header_rows`` to :func:`serialize_table` instead of relying on this.
    """
    body_start = 1
    if len(rows) < _MIN_ROWS_FOR_HEADER:
        return 0
    head = [_cell(c) for c in rows[0]]
    if not head or any(not c for c in head):
        return 0  # a header names every one of its columns
    if len(set(head)) != len(head):
        return 0  # repeated labels -> almost always a spanned multi-row header
    if any(_is_typed(c) for c in head):
        return 0  # a quantity in row 0 means row 0 is data

    body = [[_cell(c) for c in row] for row in rows[body_start:]]
    for column, label in enumerate(head):
        values = [row[column] for row in body if column < len(row) and row[column]]
        if len(values) < _MIN_TYPED_BODY_CELLS or not all(_is_typed(v) for v in values):
            continue
        if any(label == row[column] for row in body if column < len(row)):
            continue  # the "label" recurs as data in its own column
        return 1
    return 0


def _is_span(group: list[int], table: list[list[str | None]]) -> bool:
    """Do these columns hold one value between them on every row (a spanned header cell)?"""
    if len(group) == 1:
        return True
    return all(sum(1 for i in group if i < len(row) and _cell(row[i])) <= 1 for row in table[1:])


def _drop_empty_columns(table: list[list[str]]) -> list[list[str]]:
    """Drop columns no row fills. A merged-cell grid is full of them, and each one serialized as
    an empty field — "1.0 |  |  | 15th December 2022" — which reads as a missing value rather
    than as a column that never existed."""
    if not table:
        return table
    width = max(len(row) for row in table)
    keep = [i for i in range(width) if any(i < len(row) and row[i] for row in table)]
    return [[row[i] if i < len(row) else "" for i in keep] for row in table]


def _escape(text: str, *, label: bool = False) -> str:
    """Make a cell unambiguous: the separators can then only appear as separators."""
    out = text.replace("\\", "\\\\").replace("|", "\\|")
    return out.replace(LABEL_SEP, "\\" + LABEL_SEP) if label else out


def _unescape(text: str) -> str:
    out: list[str] = []
    index = 0
    while index < len(text):
        if text[index] == "\\" and index + 1 < len(text):
            out.append(text[index + 1])
            index += 2
        else:
            out.append(text[index])
            index += 1
    return "".join(out)


def _split_unescaped(text: str, separator: str, *, limit: int | None = None) -> list[str]:
    """Split on `separator`, ignoring occurrences that were escaped when written."""
    parts: list[str] = []
    buffer: list[str] = []
    index = 0
    while index < len(text):
        if text[index] == "\\" and index + 1 < len(text):
            buffer.extend(text[index : index + 2])
            index += 2
            continue
        if text.startswith(separator, index) and (limit is None or len(parts) < limit):
            parts.append("".join(buffer))
            buffer = []
            index += len(separator)
            continue
        buffer.append(text[index])
        index += 1
    parts.append("".join(buffer))
    return parts


def _is_table(table: list[list[str]]) -> bool:
    """Is this DETECTED grid actually a table, or a layout container?

    Word writes page layout as a table — one paragraph per row, ruled rectangles around it,
    mostly empty cells — and ``find_tables()`` reports it identically to a real one. Since a
    detected table's characters are removed from the narrative and replaced by rendered rows,
    accepting a layout grid shreds the prose it claimed. Measured on the real corpus: a travel
    policy came back 46 rows x 3 columns with 72% empty cells on page 1, versus 4 rows x 7
    columns and 0% empty for a genuine entitlement grid.

    Two conditions, and neither is a tuned threshold:

    * More than one row. A single row has nothing to be a column OF; pdfplumber reports these
      constantly for a boxed heading in running prose.
    * More rows carrying SEVERAL values than rows carrying one. Serializing a table buys
      exactly one thing — the ASSOCIATION between values on a row ("Grade: Officer |
      Days: 24"). A grid whose rows mostly hold a single value has no associations to express,
      so the serialized form says nothing the plain text did not, while costing reading order
      to produce. This subsumes the single-column case, which can never have a multi-value row.

    A strict majority between two counted populations, not a tuned fraction — and about
    ASSOCIATION rather than density, so a genuine table with gaps still passes: those rows
    relate the values they do carry."""
    if len(table) < MIN_TABLE_ROWS:
        return False
    filled = [sum(1 for cell in row if cell) for row in table]
    return sum(1 for n in filled if n > 1) > sum(1 for n in filled if n == 1)


def serialize_table(
    rows: Sequence[Sequence[str | None]], *, header_rows: int | None = None, declared: bool = False
) -> str:
    """Render a table as one line per row, or ``""`` if it is not a table worth keeping.

    ``header_rows`` overrides :func:`detect_header_rows` for a caller that knows the shape;
    ``0`` forces positional rows. When a header is used its own row is NOT emitted separately
    — its words are on every body row already, and emitting it again would be the duplication
    this pipeline just removed from page text.

    ``declared`` says the SOURCE FORMAT calls this a table (a spreadsheet sheet, a ``w:tbl``)
    rather than an extractor having inferred one from ruling lines. The minimum-row rule is
    about detection confidence, so it does not apply — dropping a one-row sheet would lose
    real data to a guard aimed at false positives in PDF prose."""
    table = [[_cell(c) for c in row] for row in collapse_spans(rows)]
    table = [row for row in table if any(row)]
    table = _drop_empty_columns(table)
    if not table or (not declared and not _is_table(table)):
        return ""

    count = detect_header_rows(rows=[list(r) for r in table]) if header_rows is None else header_rows
    body = table[count:] if count > 0 else table
    if count <= 0 or not body:
        return "\n".join(CELL_SEP.join(_escape(c) for c in row) for row in table)

    width = max(len(row) for row in table)
    labels = [" ".join(filter(None, (row[i] if i < len(row) else "" for row in table[:count]))) for i in range(width)]
    lines: list[str] = []
    for row in body:
        fields = []
        for index, value in enumerate(row):
            if not value:
                continue  # an absent value needs no label; alignment is what labels replace
            label = labels[index] if index < len(labels) else ""
            fields.append(f"{_escape(label, label=True)}{LABEL_SEP}{_escape(value)}" if label else _escape(value))
        if fields:
            lines.append(CELL_SEP.join(fields))
    return "\n".join(lines)


def parse_table_row(line: str) -> list[tuple[str | None, str]]:
    """Inverse of one serialized row: ``[(label, value), ...]``, label ``None`` if unlabelled.

    The canonical proof that the format is unambiguous — and the seam row-level chunk
    serialization will read back through rather than re-deriving the escaping rules."""
    fields: list[tuple[str | None, str]] = []
    for field in _split_unescaped(line, CELL_SEP):
        parts = _split_unescaped(field, LABEL_SEP, limit=1)
        if len(parts) == _LABELLED_FIELD_PARTS:
            fields.append((_unescape(parts[0]), _unescape(parts[1])))
        else:
            fields.append((None, _unescape(field)))
    return fields
