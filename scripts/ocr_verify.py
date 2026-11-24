"""Live proof that preprocessing lifts Tesseract accuracy on hard, low-quality scans.

Renders a faint, low-contrast, slightly noisy receipt-like image (the worst case that raw
Tesseract fails on), then OCRs it (a) raw and (b) through preprocess_for_ocr, and reports
how many of the expected tokens each path recovered + the real confidence. Run inside the
api container: docker exec ragaas-dev-api-1 uv run python /app/scripts/ocr_verify.py
"""

from __future__ import annotations

import asyncio
import io

from PIL import Image, ImageDraw
from svc_doc_intelligence.ocr import TesseractOCR

_LINES = [
    "THE CITY OF NEW YORK",
    "COMMISSARY PURCHASE",
    "ORANGE SLICE 1.40",
    "CHEEZ-IT 2.72",
    "RAMEN SOUP 5.25",
    "KOOL AID GRAPE 1.38",
    "SNICKERS 2.64",
    "TOTAL 50.16",
]
_EXPECTED = {
    "NEW",
    "YORK",
    "COMMISSARY",
    "ORANGE",
    "SLICE",
    "CHEEZ",
    "RAMEN",
    "SOUP",
    "KOOL",
    "SNICKERS",
    "TOTAL",
    "1.40",
    "2.72",
    "5.25",
    "50.16",
}


def _faint_receipt() -> bytes:
    """Low-contrast grey text on off-white — mimics a faded thermal receipt / halftone."""
    img = Image.new("RGB", (620, 340), (236, 236, 234))
    draw = ImageDraw.Draw(img)
    y = 20
    for line in _LINES:
        draw.text((24, y), line, fill=(158, 158, 158))  # faint grey ink, hard for raw OCR
        y += 38
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _recovered(text: str) -> set[str]:
    up = text.upper().replace(":", " ").replace("|", " ")
    return {t for t in _EXPECTED if t in up}


def _receipt(size: tuple[int, int], bg: int, ink: int, step: int) -> bytes:
    img = Image.new("RGB", size, (bg, bg, bg))
    draw = ImageDraw.Draw(img)
    y = 12
    for line in _LINES:
        draw.text((16, y), line, fill=(ink, ink, ink))
        y += step
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def _score(label: str, data: bytes) -> None:
    import pytesseract

    raw = _recovered(pytesseract.image_to_string(Image.open(io.BytesIO(data))) or "")
    gentle = await TesseractOCR(preprocess=True, binarize=False).ocr(data, "image/png")
    binz = await TesseractOCR(preprocess=True, binarize=True).ocr(data, "image/png")
    print(
        f"{label:24s} RAW {len(raw):2d}/{len(_EXPECTED)}  |  "
        f"GENTLE {len(_recovered(gentle.text)):2d}/{len(_EXPECTED)} (conf {gentle.min_confidence:.2f})  |  "
        f"BINARIZE {len(_recovered(binz.text)):2d}/{len(_EXPECTED)} (conf {binz.min_confidence:.2f})"
    )


async def main() -> None:
    print(f"Expected tokens per image: {len(_EXPECTED)}  (higher = better)\n")
    # Moderate: readable grey — preprocessing must NOT hurt this.
    await _score("moderate grey", _receipt((620, 340), 236, 150, 38))
    # Faint + small: the hard case where raw Tesseract struggles and preprocessing should help.
    await _score("faint + small", _receipt((300, 168), 232, 176, 19))
    # Very faint: low-contrast thermal-receipt-like.
    await _score("very faint", _receipt((620, 340), 240, 196, 38))


if __name__ == "__main__":
    asyncio.run(main())
