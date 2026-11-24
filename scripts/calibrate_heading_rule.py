"""Counterfactual heading-rule calibration across document FAMILIES.

Nothing here changes production parsing. Each PDF is extracted with its real page geometry and
typography, the layout parser's own candidate set is reproduced, and three rules are applied to
the SAME candidates:

    current       length + letter count only (what ships)
    terminator    reject anything ending . ! ?
    combined      reject that, UNLESS it is numbered-heading-shaped

The point is the rejections, each of which is printed for hand adjudication. A rule is not
validated by a count.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from rag_core.layout import _MAX_HEADING_CHARS, _MIN_HEADING_CHARS, _MIN_HEADING_LETTERS, _body_size

_NUMBERED = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S")


def current(t: str) -> bool:
    s = t.strip()
    return _MIN_HEADING_CHARS <= len(s) <= _MAX_HEADING_CHARS and sum(c.isalpha() for c in s) >= _MIN_HEADING_LETTERS


def terminator(t: str) -> bool:
    return current(t) and not t.strip().endswith((".", "!", "?"))


def combined(t: str) -> bool:
    s = t.strip()
    return current(t) and not (s.endswith((".", "!", "?")) and not _NUMBERED.match(s))


def candidates(pdf: Path) -> list[tuple[str, float, bool]]:
    """(line text, size, bold) for every line the heading classifier would consider."""
    import pdfplumber
    from svc_doc_intelligence.ocr import _line_typography

    out: list[tuple[str, float, bool]] = []
    with pdfplumber.open(pdf) as doc:
        for page in doc.pages:
            for line in page.extract_text_lines():
                text = str(line.get("text") or "").strip()
                if not text:
                    continue
                size, bold = _line_typography(list(line.get("chars") or []))
                if size is not None:
                    out.append((text, size, bool(bold)))
    return out


def report(pdf: Path) -> None:
    lines = candidates(pdf)
    if not lines:
        print(f"\n{pdf.name}: no measured lines\n")
        return
    body = _body_size([(size, len(t)) for t, size, _ in lines])
    bold_speaks = any(size == body and not bold for _, size, bold in lines)
    # The typographic half of `_classify_headings`, unchanged.
    typo = [(t, size, bold) for t, size, bold in lines if size > body or (size == body and bold and bold_speaks)]
    print(
        f"\n{'=' * 78}\n{pdf.name}\n  {len(lines):,} measured lines, body size {body}, "
        f"{len(typo)} pass the typographic test"
    )
    sets = {
        name: {t for t, _, _ in typo if fn(t)}
        for name, fn in (("current", current), ("terminator", terminator), ("combined", combined))
    }
    for name in ("current", "terminator", "combined"):
        print(f"  {name:<11} promotes {len(sets[name]):>4}")
    for name in ("terminator", "combined"):
        lost = sorted(sets["current"] - sets[name])
        print(f"\n  {name} rejects {len(lost)} of current's {len(sets['current'])} — every one, for adjudication:")
        for t in lost:
            numbered = "numbered" if _NUMBERED.match(t) else "unnumbered"
            print(f"      [{numbered:<10}] {t[:88]!r}")


def aggregate(paths: list[Path]) -> None:
    """The population view, and the ONE question the numbering exemption has to answer: what does
    `combined` KEEP that `terminator` rejects, and is each of those a genuine heading?"""
    tot = {"docs": 0, "cand": 0, "cur": 0, "term": 0, "comb": 0}
    exempted: list[tuple[str, str]] = []
    for pdf in paths:
        try:
            lines = candidates(pdf)
        except Exception as exc:
            print(f"  {pdf.name}: unreadable ({type(exc).__name__})")
            continue
        if not lines:
            continue
        body = _body_size([(size, len(t)) for t, size, _ in lines])
        bold_speaks = any(size == body and not bold for _, size, bold in lines)
        typo = [t for t, size, bold in lines if size > body or (size == body and bold and bold_speaks)]
        cur = {t for t in typo if current(t)}
        term = {t for t in typo if terminator(t)}
        comb = {t for t in typo if combined(t)}
        tot["docs"] += 1
        tot["cand"] += len(typo)
        tot["cur"] += len(cur)
        tot["term"] += len(term)
        tot["comb"] += len(comb)
        for t in sorted(comb - term):
            exempted.append((pdf.name, t))
    print(f"\n{'=' * 78}\nPOPULATION: {tot['docs']} documents, {tot['cand']:,} typographic candidates")
    print(f"  current promotes  {tot['cur']:>5}")
    print(f"  terminator        {tot['term']:>5}   ({tot['cur'] - tot['term']} rejected)")
    print(f"  combined          {tot['comb']:>5}   ({tot['cur'] - tot['comb']} rejected)")
    print(f"\nWHAT THE NUMBERING EXEMPTION PRESERVES — {len(exempted)} case(s), each adjudicated by hand:")
    for name, t in exempted:
        print(f"  {name[:38]:<40} {t[:70]!r}")
    if not exempted:
        print("  none in this population: the exemption is UNTESTED outside AEGCL")


def exemption_population(paths: list[Path]) -> None:
    """THE NARROW POPULATION THE EXEMPTION ACTUALLY GOVERNS, and nothing else.

    A candidate only reaches the exemption when all four hold at once:

        * it passes the typographic test (size above body, or body-size bold where bold speaks)
        * it is within `_MAX_HEADING_CHARS` -- the 80-char bound already rejects longer prose
        * it ends in a sentence terminator -- otherwise the terminator rule never fires
        * it is numbered-heading-shaped -- otherwise the terminator rule rejects it outright

    Anything failing one of those is decided elsewhere and tells us nothing about the exemption.
    Every member is printed for hand adjudication, because the whole question is whether these are
    headings or sentences and no count answers that.
    """
    members: list[tuple[str, str, float, float, bool]] = []
    for pdf in paths:
        try:
            lines = candidates(pdf)
        except Exception as exc:
            print(f"  {pdf.name}: unreadable ({type(exc).__name__})")
            continue
        if not lines:
            continue
        body = _body_size([(size, len(t)) for t, size, _ in lines])
        bold_speaks = any(size == body and not bold for _, size, bold in lines)
        for t, size, bold in lines:
            contrast = size > body or (size == body and bold and bold_speaks)
            if not (contrast and current(t)):
                continue
            s = t.strip()
            if s.endswith((".", "!", "?")) and _NUMBERED.match(s):
                members.append((pdf.name, s, size, body, bold))

    print(f"\n{'=' * 78}\nTHE EXEMPTION'S POPULATION across {len(paths)} documents: {len(members)} candidate(s)")
    print("  (typographic contrast AND <= 80 chars AND ends in a terminator AND numbered)\n")
    if not members:
        print("  EMPTY. Outside the one AEGCL heading, no document in this population reaches the")
        print("  exemption at all -- so it can neither be validated nor falsified here, and the")
        print("  terminator rule alone would behave identically on all 18.")
        return
    for name, text, size, body, bold in members:
        print(f"  {name[:34]:<36} {size:>5.2f}pt (body {body:.2f}){' bold' if bold else '     '}  {text[:66]!r}")


if __name__ == "__main__":
    paths = [Path(a) for a in sys.argv[1:]]
    aggregate(paths)
    exemption_population(paths)
