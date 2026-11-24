"""Unit test: ChandraOCREngine turns a Chandra transcription into a doc-intelligence Extraction."""

from __future__ import annotations

import httpx
from svc_doc_intelligence.ocr import CAP_TABLES, CAP_TEXT, OCR_EXTRACTOR_CHANDRA, ChandraOCREngine

from rag_schemas.common import Modality

_IMG = b"\x89PNG\r\n fake image bytes of a table"


def _engine_returning(text: str) -> ChandraOCREngine:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})

    # localhost is egress-allowlisted, so the engine's default client construction passes the
    # sovereignty guard; we then swap in a mock transport (no real Chandra).
    eng = ChandraOCREngine(url="http://localhost/v1", model="chandra-test")
    eng._client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://localhost/v1")
    return eng


async def test_image_ocr_builds_a_single_visual_page(inmemory: None) -> None:
    eng = _engine_returning("# Report\n\n| A | B |\n|---|---|\n| 1 | 2 |")
    extraction = await eng.ocr(_IMG, "image/png")

    assert extraction.modality is Modality.OCR
    assert len(extraction.pages) == 1
    page = extraction.pages[0]
    assert page.page_no == 1
    assert page.extractor == OCR_EXTRACTOR_CHANDRA
    assert "| A | B |" in page.text  # table preserved as markdown
    # Chandra declares both text and table capabilities.
    assert CAP_TEXT in page.capabilities and CAP_TABLES in page.capabilities
    # render() joins pages + derives the page map from the same pass.
    rendered = extraction.render()
    assert "Report" in rendered.text
