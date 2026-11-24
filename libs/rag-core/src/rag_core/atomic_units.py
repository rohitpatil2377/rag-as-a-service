"""Atomic units — spans of text that must NEVER be split across chunks (FR-C2 support).

Pure logic, no I/O. A chunker that cuts a table between its header and a row, or a code
fence in half, or a formula between the base and its exponent, produces fragments that are
worse than useless: a bare "| Q |" supports no statement, "H₂SO₄" flattened to "H2SO4" is a
different substance, and half a code block will not run. This module finds the ranges that a
structure-aware chunker must keep whole.

The signals are read from the EXTRACTED TEXT ONLY — no doc-intelligence change, no layout
metadata — so they survive whatever produced the text:

- Tables: a maximal run of >=2 consecutive lines each containing the cell separator " | "
  (``svc_doc_intelligence.tables.CELL_SEP``). Highest confidence — it covers the dosage,
  parameter and reaction tables that narrow lookups depend on.
- Fenced code blocks: the text between a matched pair of ``` fences.
- Formula / equation / reaction lines: a short line whose characters are dominated by
  math/notation symbols (super/subscripts, operators, arrows, Greek, LaTeX ``$…$``). LOWEST
  confidence — deliberately narrow (symbol density AND short length) so ordinary prose is
  never captured; a miss simply degrades to clause splitting (the status quo).

``atomic_spans`` returns sorted, non-overlapping ``[start, end)`` offsets into the text.
"""

from __future__ import annotations

# The field separator a serialized table row carries in the extracted text. Mirrors
# ``svc_doc_intelligence.tables.CELL_SEP`` — duplicated as a literal rather than imported so
# rag-core keeps its clean dependency direction (a library never imports a service package).
CELL_SEP = " | "

# A fence line is a line whose stripped text STARTS with three backticks (an optional info
# string may follow the opening fence, e.g. "```python"). Matched in pairs.
_FENCE = "```"

# A table run needs at least this many consecutive separator lines to be a table (one line with
# a stray " | " is prose); a formula line needs at least this many non-space characters (so a bare
# "x=" is not one).
_MIN_TABLE_RUN = 2
_MIN_FORMULA_NONSPACE = 3
# A formula line is SHORT — long enough for "a² - b² = (a+b)(a-b)" but far below a prose
# sentence, so a symbol-sprinkled paragraph can never be mistaken for one.
_MAX_FORMULA_CHARS = 80
# ...and its non-space characters are DOMINATED by math notation. Below this density a line is
# prose that happens to contain a symbol ("it costs $5 today"), not an equation.
_MIN_FORMULA_DENSITY = 0.30

# STRONG signals — at least one must be present, or a line of digits and dots ("Section 3.2",
# "1,234.56") would qualify on density alone. These are characters that essentially only occur
# in mathematical/chemical notation.
_SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ"
_SUBSCRIPTS = "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓ"
_OPERATORS = "=±×÷√∑∫∂∇∞≈≠≤≥·⋅→⇌↔⇒⇔≡∈∉⊂⊆∪∩°∝∀∃"  # noqa: RUF001 — math symbols, not their Latin look-alikes
_STRONG = frozenset(_SUPERSCRIPTS + _SUBSCRIPTS + _OPERATORS + "$")


def _is_greek(ch: str) -> bool:
    """Greek letters are a strong formula signal (alpha, beta, Sigma, Omega), not English prose."""
    return "Ͱ" <= ch <= "Ͽ" or "ἀ" <= ch <= "῿"


# Characters that count toward math DENSITY but are too common in prose to stand alone as the
# strong signal (a bare hyphen, brackets and digits appear everywhere).
_MATH_ISH = frozenset("+-*/^_()[]{}<>|.,0123456789")


def _line_records(text: str) -> list[tuple[int, str]]:
    """``(start_offset, line_including_newline)`` for each line, offsets real into ``text``."""
    records: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        records.append((offset, line))
        offset += len(line)
    return records


def _content_end(start: int, line: str) -> int:
    """The offset just past the line's last non-newline character."""
    return start + len(line.rstrip("\r\n"))


def _table_spans(records: list[tuple[int, str]]) -> list[tuple[int, int]]:
    """Maximal runs of >=2 consecutive lines each containing the cell separator."""
    spans: list[tuple[int, int]] = []
    run_start_idx: int | None = None
    for i, (_start, line) in enumerate(records):
        if CELL_SEP in line:
            if run_start_idx is None:
                run_start_idx = i
        else:
            if run_start_idx is not None and i - run_start_idx >= _MIN_TABLE_RUN:
                spans.append((records[run_start_idx][0], _content_end(*records[i - 1])))
            run_start_idx = None
    if run_start_idx is not None and len(records) - run_start_idx >= _MIN_TABLE_RUN:
        last = records[-1]
        spans.append((records[run_start_idx][0], _content_end(*last)))
    return spans


def _code_spans(records: list[tuple[int, str]]) -> list[tuple[int, int]]:
    """Text between MATCHED pairs of ``` fence lines. An unmatched trailing fence is ignored."""
    fence_idx = [i for i, (_s, line) in enumerate(records) if line.strip().startswith(_FENCE)]
    spans: list[tuple[int, int]] = []
    for open_i, close_i in zip(fence_idx[::2], fence_idx[1::2], strict=False):
        spans.append((records[open_i][0], _content_end(*records[close_i])))
    return spans


def _is_formula_line(line: str) -> bool:
    """A short line dominated by math/notation symbols, with at least one strong signal."""
    stripped = line.strip()
    non_space = [c for c in stripped if not c.isspace()]
    if not (0 < len(non_space) <= _MAX_FORMULA_CHARS) or len(non_space) < _MIN_FORMULA_NONSPACE:
        return False
    strong = sum(1 for c in non_space if c in _STRONG or _is_greek(c))
    if strong == 0:
        return False
    mathish = sum(1 for c in non_space if c in _STRONG or _is_greek(c) or c in _MATH_ISH)
    return mathish / len(non_space) >= _MIN_FORMULA_DENSITY


def _formula_spans(records: list[tuple[int, str]]) -> list[tuple[int, int]]:
    """Runs of consecutive formula lines (a multi-line derivation stays one atomic unit)."""
    spans: list[tuple[int, int]] = []
    run_start_idx: int | None = None
    for i, (_start, line) in enumerate(records):
        if _is_formula_line(line):
            if run_start_idx is None:
                run_start_idx = i
        else:
            if run_start_idx is not None:
                spans.append((records[run_start_idx][0], _content_end(*records[i - 1])))
            run_start_idx = None
    if run_start_idx is not None:
        spans.append((records[run_start_idx][0], _content_end(*records[-1])))
    return spans


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort and union overlapping/touching ranges into a minimal non-overlapping set.

    A code fence that also contains ``|`` lines, or a table abutting a formula line, would
    otherwise yield overlapping ranges; the chunker needs one clean set of edges to snap to."""
    if not spans:
        return []
    ordered = sorted(spans)
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def atomic_spans(text: str) -> list[tuple[int, int]]:
    """Sorted, non-overlapping ``[start, end)`` ranges that must never be split across chunks.

    Empty when the text has no table run, code fence or formula line — in which case a
    structure-aware chunker's output is identical to ordinary clause chunking ("never worse
    than clause")."""
    records = _line_records(text)
    return _merge(_table_spans(records) + _code_spans(records) + _formula_spans(records))
