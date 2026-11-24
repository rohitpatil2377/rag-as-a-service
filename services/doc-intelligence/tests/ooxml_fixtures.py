"""Minimal, VALID OOXML archives built with the stdlib.

Real .xlsx/.pptx files, not stubs: the parts, namespaces and relationship indirection are the
ones Excel and PowerPoint actually emit, because the thing under test is precisely whether we
follow that indirection instead of guessing part names from a filename convention.
"""

from __future__ import annotations

import io
import zipfile

_SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_PRES_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
_DRAW_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _zip(parts: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, body in parts.items():
            z.writestr(name, body)
    return buf.getvalue()


def _rels(entries: list[tuple[str, str]]) -> str:
    body = "".join(f'<Relationship Id="{rid}" Type="{_DOC_REL_NS}/x" Target="{target}"/>' for rid, target in entries)
    return f'<?xml version="1.0"?><Relationships xmlns="{_PKG_REL_NS}">{body}</Relationships>'


def make_xlsx(sheets: dict[str, list[list[str]]]) -> bytes:
    """A workbook of ``{sheet name: rows of cell text}``.

    Strings go through ``sharedStrings.xml`` and cells reference them by index, exactly as
    Excel writes them — a reader that ignores the shared-string table gets integers back
    instead of text. Sheet ORDER is expressed through r:id -> rels -> part, deliberately NOT
    matching the alphabetical order of the part names, so a reader that sorts filenames
    instead of following the relationships reads the workbook in the wrong order.
    """
    strings: list[str] = []

    def sid(value: str) -> int:
        if value not in strings:
            strings.append(value)
        return strings.index(value)

    parts: dict[str, str] = {}
    sheet_decls: list[str] = []
    rel_entries: list[tuple[str, str]] = []
    # Reverse the part numbering against declaration order: sheet 1 lives in the highest-
    # numbered file, so filename sort and true order disagree.
    total = len(sheets)
    for position, (name, rows) in enumerate(sheets.items(), start=1):
        part_no = total - position + 1
        rid = f"rId{position}"
        sheet_decls.append(f'<sheet name="{name}" sheetId="{position}" r:id="{rid}"/>')
        rel_entries.append((rid, f"worksheets/sheet{part_no}.xml"))
        body = ""
        for r, row in enumerate(rows, start=1):
            cells = "".join(
                f'<c r="{chr(ord("A") + c)}{r}" t="s"><v>{sid(value)}</v></c>' for c, value in enumerate(row)
            )
            body += f'<row r="{r}">{cells}</row>'
        parts[f"xl/worksheets/sheet{part_no}.xml"] = (
            f'<?xml version="1.0"?><worksheet xmlns="{_SHEET_NS}"><sheetData>{body}</sheetData></worksheet>'
        )

    parts["xl/workbook.xml"] = (
        f'<?xml version="1.0"?><workbook xmlns="{_SHEET_NS}" xmlns:r="{_DOC_REL_NS}">'
        f"<sheets>{''.join(sheet_decls)}</sheets></workbook>"
    )
    parts["xl/_rels/workbook.xml.rels"] = _rels(rel_entries)
    si = "".join(f"<si><t>{s}</t></si>" for s in strings)
    parts["xl/sharedStrings.xml"] = f'<?xml version="1.0"?><sst xmlns="{_SHEET_NS}">{si}</sst>'
    return _zip(parts)


def make_pptx(slides: list[list[str]]) -> bytes:
    """A deck of ``[[paragraph, ...], ...]``. Slide order likewise runs through the
    relationship table, with the part numbering reversed against it."""
    parts: dict[str, str] = {}
    ids: list[str] = []
    rel_entries: list[tuple[str, str]] = []
    total = len(slides)
    for position, paragraphs in enumerate(slides, start=1):
        part_no = total - position + 1
        rid = f"rId{position}"
        ids.append(f'<p:sldId id="{255 + position}" r:id="{rid}"/>')
        rel_entries.append((rid, f"slides/slide{part_no}.xml"))
        body = "".join(f"<a:p><a:r><a:t>{text}</a:t></a:r></a:p>" for text in paragraphs)
        parts[f"ppt/slides/slide{part_no}.xml"] = (
            f'<?xml version="1.0"?><p:sld xmlns:p="{_PRES_NS}" xmlns:a="{_DRAW_NS}">'
            f"<p:cSld><p:spTree>{body}</p:spTree></p:cSld></p:sld>"
        )
    parts["ppt/presentation.xml"] = (
        f'<?xml version="1.0"?><p:presentation xmlns:p="{_PRES_NS}" xmlns:r="{_DOC_REL_NS}">'
        f"<p:sldIdLst>{''.join(ids)}</p:sldIdLst></p:presentation>"
    )
    parts["ppt/_rels/presentation.xml.rels"] = _rels(rel_entries)
    return _zip(parts)


def make_docx(paragraphs: list[str]) -> bytes:
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    return _zip(
        {
            "word/document.xml": (
                f'<?xml version="1.0"?><w:document xmlns:w="{_WORD_NS}"><w:body>{body}</w:body></w:document>'
            )
        }
    )


def make_epub() -> bytes:
    """A zip container that is neither Word, Excel nor PowerPoint — the case that used to be
    handed to the .docx extractor purely because it starts with PK."""
    return _zip({"mimetype": "application/epub+zip", "META-INF/container.xml": "<container/>"})
