"""Dependency-free born-digital PDF builder for extraction tests.

Hand-rolled rather than via a PDF library so the doc-intelligence unit tests stay free of
the demo app's (LGPL) fpdf2 dependency. Object ids and byte offsets are computed in ONE
pass over the object list, so the xref table can never drift from the body it indexes.
"""

from __future__ import annotations

_LEADING = 14  # pt between baselines; only needs to be > font size so lines don't overlap
_FONT_SIZE = 12

_TABLE_ROW_HEIGHT = 20
_TABLE_LEFT = 72
_TABLE_COL = 200  # x of the column divider
_TABLE_RIGHT = 330
_TABLE_GAP = 12  # blank space between surrounding text and the table's border


def _table_ops(rows: list[list[str]], top: float) -> tuple[str, float]:
    """Draw a 2-column ruled table with its top border at ``top``; return the ops and the y the
    page may continue writing from. pdfplumber's default table detection keys off ruling LINES,
    so the grid has to actually be stroked — text alignment alone would not be found."""
    height = len(rows) * _TABLE_ROW_HEIGHT
    ops = ["0.5 w"]
    for i in range(len(rows) + 1):
        y = top - i * _TABLE_ROW_HEIGHT
        ops.append(f"{_TABLE_LEFT} {y} m {_TABLE_RIGHT} {y} l S")
    for x in (_TABLE_LEFT, _TABLE_COL, _TABLE_RIGHT):
        ops.append(f"{x} {top} m {x} {top - height} l S")
    ops.append("BT /F1 10 Tf")
    for i, row in enumerate(rows):
        y = top - i * _TABLE_ROW_HEIGHT - 14
        for x, cell in zip((_TABLE_LEFT + 4, _TABLE_COL + 4), row, strict=False):
            ops.append(f"1 0 0 1 {x} {y} Tm ({cell}) Tj")
    ops.append("ET")
    return " ".join(ops), top - height - _TABLE_GAP


_FOOTER_Y = 40  # baseline, in PDF points up from the page bottom
_FOOTER_SIZE = 9


def _footer_ops(text: str) -> str:
    """One line drawn in the page's BOTTOM MARGIN, well below the type area.

    A line laid out by ``emit`` cannot reach here — it starts at y=720 and walks down a line at a
    time — so a fixture that needs a running footer at the real foot of the page has to place it
    directly. At y=40 on a 792 pt page pdfplumber reports it inside the bottom ~5% of the page,
    which is what makes it a FOOTER by geometry rather than merely by being emitted last.

    Drawing a line here does not make it furniture: whether it is a page marker or a clause that
    happens to sit low is the parser's judgement, and a fixture can use this for either."""
    return f"BT /F1 {_FOOTER_SIZE} Tf 1 0 0 1 72 {_FOOTER_Y} Tm ({text}) Tj ET"


def _page_ops(
    lines: list[str],
    rows: list[list[str]] | None,
    after_line: int | None,
    styles: dict[int, tuple[str, float]] | None = None,
    footer: str | None = None,
) -> str:
    """Lay a page out top-down: text lines, optionally interrupted by a table after
    ``after_line`` of them, then the remaining lines BELOW the table.

    ``styles`` maps a 0-based line index to ``(font, size)`` — ``"F1"`` is Helvetica and
    ``"F2"`` Helvetica-Bold — so a fixture can draw a page whose heading is genuinely bigger and
    bolder than its body, which is what the born-digital typography capture reads."""
    y = 720.0
    split = len(lines) if after_line is None else after_line
    ops: list[str] = []

    def emit(block: list[str], start: int) -> None:
        nonlocal y
        for offset, line in enumerate(block):
            font, size = (styles or {}).get(start + offset, ("F1", float(_FONT_SIZE)))
            ops.append(f"BT /{font} {size:g} Tf 1 0 0 1 72 {y:.1f} Tm ({line}) Tj ET")
            # Leading has to clear the tallest glyph on the line, or a big heading would overlap
            # the line under it and pdfplumber would read them as one.
            y -= max(_LEADING, size + 2)

    emit(lines[:split], 0)
    if rows:
        table_ops, y = _table_ops(rows, y - _TABLE_GAP)
        ops.append(table_ops)
    emit(lines[split:], split)
    if footer:
        ops.append(_footer_ops(footer))
    return " ".join(ops)


def make_pdf(
    pages: list[list[str]],
    *,
    image_pages: tuple[int, ...] = (),
    tables: dict[int, list[list[str]]] | None = None,
    table_after_line: dict[int, int] | None = None,
    image_px: int = 2,
    styles: dict[int, dict[int, tuple[str, float]]] | None = None,
    footers: dict[int, str] | None = None,
) -> bytes:
    """Build a PDF with one text line per string, ``pages[i]`` being page ``i + 1``.

    ``image_pages`` lists 1-based page numbers that additionally carry a drawn image XObject —
    a page with an image and no text is what a scanned/diagram page looks like to the
    classifier. ``tables`` maps a 1-based page number to the rows of a ruled table drawn on it;
    ``table_after_line`` says how many of that page's lines come BEFORE the table (the rest are
    laid out below it), so a table can sit mid-page between two paragraphs. ``image_px`` is the
    drawn image's intrinsic size in pixels (square): the default 2x2 swatch is decoration to the
    OCR check, a few hundred pixels is a pasted scan. ``styles`` maps a 1-based page number to a
    0-based line index to ``(font, size)``, ``"F1"`` being Helvetica and ``"F2"``
    Helvetica-Bold. ``footers`` maps a 1-based page number to one line drawn in that page's
    BOTTOM MARGIN (see :func:`_footer_ops`) — the only way to put a line at the real foot of the
    page, which is what a geometric header/footer rule reads. Text must be plain
    ASCII without parentheses or backslashes (unescaped PDF string literals) — these are test
    fixtures, not a general-purpose writer."""
    n = len(pages)
    page_ids = [3 + 2 * i for i in range(n)]  # 1=Catalog, 2=Pages, then Page/Contents pairs
    font_id = 3 + 2 * n
    bold_font_id = font_id + 1
    image_id = font_id + 2

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Count {} /Kids [{}] >>".format(n, " ".join(f"{p} 0 R" for p in page_ids)).encode(),
    ]
    for i, lines in enumerate(pages):
        page_no = i + 1
        content = _page_ops(
            lines,
            (tables or {}).get(page_no),
            (table_after_line or {}).get(page_no),
            (styles or {}).get(page_no),
            (footers or {}).get(page_no),
        )
        resources = f"/Font << /F1 {font_id} 0 R /F2 {bold_font_id} 0 R >>"
        if page_no in image_pages:
            content += " q 400 0 0 300 72 300 cm /Im1 Do Q"
            resources += f" /XObject << /Im1 {image_id} 0 R >>"
        stream = content.encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << {resources} >> /Contents {page_ids[i] + 1} 0 R >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    # An 8-bit greyscale image — by default a 2x2 swatch, enough for the page to genuinely carry
    # image content without looking like anything worth reading.
    pixels = b"\x00\x40\x80\xff" if image_px == 2 else bytes((i * 7) % 256 for i in range(image_px * image_px))
    objects.append(
        b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceGray "
        b"/BitsPerComponent 8 /Length %d >>\nstream\n" % (image_px, image_px, len(pixels)) + pixels + b"\nendstream"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % idx + obj + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref_at)
    return bytes(out)
