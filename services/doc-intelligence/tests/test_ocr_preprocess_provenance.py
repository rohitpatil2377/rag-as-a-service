"""Image preprocessing must degrade VISIBLY (P36/P37).

`preprocess_for_ocr` and `_maybe_deskew` are best-effort by design — a preprocessing
failure must never stop OCR. But "never stops OCR" was implemented as "returns the original
image and says nothing", so a page read from an unprocessed (or un-deskewed) image is
indistinguishable at every later layer from one that was cleaned properly. On a skewed or
faint scan that is a large, invisible accuracy loss.

The fix is not to make them raise. It is to make them REPORT, in the same open reason-code
vocabulary as every other extraction degradation, so the flag reaches the PageRecord and
therefore the document metadata and the corpus census.
"""

from __future__ import annotations

from typing import Any

import structlog
from svc_doc_intelligence.ocr import (
    REASON_OCR_DESKEW_FAILED,
    REASON_OCR_PREPROCESS_FAILED,
    _maybe_deskew,
    preprocess_for_ocr,
)


class _NotAnImage:
    """Anything the Pillow pipeline will choke on partway through."""

    def convert(self, mode: str) -> Any:
        raise ValueError("cannot convert")


def test_a_preprocessing_failure_is_reported_not_just_logged() -> None:
    original = _NotAnImage()
    with structlog.testing.capture_logs() as logs:
        prepared = preprocess_for_ocr(original)

    # Still best-effort: OCR gets an image and runs.
    assert prepared.image is original
    # ...but the page can now say it was read from an uncleaned image.
    assert REASON_OCR_PREPROCESS_FAILED in prepared.degradations
    # The old log said only "ocr_preprocess_skipped" with a bare message — not diagnosable.
    warned = [e for e in logs if e.get("event") == "ocr_preprocess_failed"]
    assert warned and warned[0]["error_type"] and warned[0]["error"] and warned[0]["exc_info"]


def test_a_deskew_failure_is_distinguished_from_opencv_being_absent() -> None:
    """The bare handler swallowed both identically. They are not the same event: OpenCV
    missing is a deployment choice visible in the image manifest, while cv2 raising on a real
    page is a per-page accuracy loss that nothing else will ever surface."""
    import sys

    class _Exploding:
        """A stand-in for cv2 that imports fine and then fails on the page."""

        def __getattr__(self, name: str) -> Any:
            # Let the import machinery read cv2 dunders: Python 3.13 probes __spec__ on the
            # cached sys.modules entry during \`import cv2\`, so raising here would blow up at
            # IMPORT time. Only a real cv2 API call on the page should fail (what we test).
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            raise RuntimeError(f"cv2.{name} blew up")

    saved = sys.modules.get("cv2")
    sys.modules["cv2"] = _Exploding()  # type: ignore[assignment]
    try:
        with structlog.testing.capture_logs() as logs:
            result = _maybe_deskew(_NotAnImage())
    finally:
        if saved is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = saved

    assert REASON_OCR_DESKEW_FAILED in result.degradations
    assert [e for e in logs if e.get("event") == "ocr_deskew_failed"]


def test_the_reason_reaches_the_page_record(monkeypatch: Any) -> None:
    """Reporting a degradation changes nothing unless it lands on the page. This drives
    TesseractOCR.ocr with a stub engine (the tesseract binary is not in this image) and a
    preprocessing step that reports a failure, and asserts the page carries it — which is what
    makes it flow on to the document metadata and the corpus census."""
    import asyncio
    import io
    import sys

    from PIL import Image
    from svc_doc_intelligence import ocr as ocr_mod

    class _StubTesseract:
        @staticmethod
        def image_to_data(image: Any, **kw: Any) -> dict[str, list[Any]]:
            return {"text": ["hello"], "conf": ["90"], "block_num": [0], "par_num": [0], "line_num": [0]}

        class Output:
            DICT = "dict"

    monkeypatch.setitem(sys.modules, "pytesseract", _StubTesseract())
    monkeypatch.setattr(
        ocr_mod,
        "preprocess_for_ocr",
        lambda img, **kw: ocr_mod.PreparedImage(image=img, degradations=(REASON_OCR_PREPROCESS_FAILED,)),
    )

    buf = io.BytesIO()
    Image.new("L", (40, 20), color=255).save(buf, format="PNG")
    extraction = asyncio.run(ocr_mod.TesseractOCR().ocr(buf.getvalue(), "image/png"))

    assert REASON_OCR_PREPROCESS_FAILED in extraction.pages[0].degradations
    assert extraction.degradations == (REASON_OCR_PREPROCESS_FAILED,)  # and up to the document


def test_opencv_being_absent_is_not_a_degradation() -> None:
    """Stamping every page of every document in an image without OpenCV would make the flag
    useless for finding the pages that actually lost something."""
    import builtins

    real_import = builtins.__import__

    def no_cv2(name: str, *args: Any, **kw: Any) -> Any:
        if name == "cv2":
            raise ImportError("no cv2 in this image")
        return real_import(name, *args, **kw)

    builtins.__import__ = no_cv2
    try:
        result = _maybe_deskew(_NotAnImage())
    finally:
        builtins.__import__ = real_import

    assert result.degradations == ()
