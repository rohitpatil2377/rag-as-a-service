"""E2E: multimodal ingest + vision answer (in-memory, echo vision twin).

Proves the additive multimodal path end-to-end: an image document is captioned at ingest
and becomes retrievable by a text query through the UNCHANGED text pipeline, and the vision
gateway can answer a question by reading the stored image bytes. All default-OFF: with the
flag unset, an image takes the existing OCR path and nothing here runs.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator

import httpx
import pytest

pytestmark = pytest.mark.e2e

# Minimal fake image bytes — the echo vision twin derives a deterministic caption from them.
_IMG = b"\x89PNG\r\n\x1a\n Acme quarterly revenue chart: Q3 revenue 42 crore"


@pytest.fixture
def multimodal_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_MULTIMODAL_ENABLED", "true")
    monkeypatch.setenv("RAG_ABSTAIN_THRESHOLD_LEXICAL", "0.2")  # let the caption ground
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["reader", "ingestor"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def _ingest_image(c: httpx.AsyncClient, kb: str, doc_id: str, img: bytes) -> None:
    r = await c.post(
        f"/v1/kb/{kb}/ingest",
        json={"doc_id": doc_id, "mime": "image/png", "content_b64": base64.b64encode(img).decode()},
    )
    assert r.status_code == 200, r.text


async def test_image_is_captioned_and_retrievable(inmemory: None, multimodal_on: None) -> None:
    c = _client("cag")
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "Charts"})
        await _ingest_image(c, "cag__charts", "revenue_chart", _IMG)

        # The image was captioned at ingest, so a text query retrieves it (via the caption).
        r = await c.post("/v1/kb/cag__charts/retrieve", json={"query": "image description", "top_k": 5})
        assert r.status_code == 200, r.text
        spans = r.json()["spans"]
        assert spans, "the captioned image should be retrievable by text"
        assert any("image" in s["text"].lower() for s in spans)


async def test_vision_describe_reads_the_stored_image(inmemory: None, multimodal_on: None) -> None:
    c = _client("cag")
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "Charts"})
        await _ingest_image(c, "cag__charts", "revenue_chart", _IMG)

    # The vision gateway can answer by reading the stored image bytes (echo twin here).
    from svc_control_plane.multimodal import describe_image

    answer = await describe_image("cag__charts", "revenue_chart", "What does this chart show?")
    assert answer  # non-empty; the echo twin echoes the prompt + image digest
    assert "chart" in answer.lower()


async def test_multimodal_off_leaves_images_on_the_ocr_path(inmemory: None) -> None:
    """With the flag unset, an image with no text layer takes the existing OCR path — which,
    on a deployment without an OCR engine, is a 422 asking for OCR. This proves the vision
    caption path is NOT invoked when multimodal is off (and that multimodal is what makes an
    image ingestable here at all)."""
    c = _client("cag")
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "Charts"})
        r = await c.post(
            "/v1/kb/cag__charts/ingest",
            json={"doc_id": "plain", "mime": "image/png", "content_b64": base64.b64encode(_IMG).decode()},
        )
        assert r.status_code == 422, r.text
        assert "OCR" in r.json()["message"]


async def test_multimodal_on_still_sends_images_through_ocr(
    inmemory: None, multimodal_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live failure this pins: with multimodal ON, the pipeline branched images away from
    OCR and indexed only the vision caption. A JPEG e-ticket went in as 32 characters — "Flight
    Number: EK 602 and EK 605" — while OCR, which read 4,544 characters from it, was never
    called. Both must reach the index: the OCR text (what the image SAYS) and the caption."""
    from svc_control_plane import wiring
    from svc_doc_intelligence.ocr import CAP_TEXT, Extraction, PageRecord
    from svc_doc_intelligence.service import DocIntelligenceService

    from rag_schemas.common import Modality

    ocr_calls: list[str] = []

    class _TicketOCR:
        async def ocr(self, data: bytes, mime: str) -> Extraction:
            ocr_calls.append(mime)
            page = PageRecord(
                page_no=1,
                text="BOOKING REFERENCE HC76IW BAGGAGE ALLOWANCE 30KGS",
                extractor="fake-ocr",
                capabilities=(CAP_TEXT,),
            )
            return Extraction(pages=[page], modality=Modality.OCR)

    service = DocIntelligenceService(ocr_engine=_TicketOCR(), image_describer=wiring.describe_image_for_ingest)
    monkeypatch.setattr(wiring, "doc_intel_service", lambda: service)

    c = _client("cag")
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "Tickets"})
        await _ingest_image(c, "cag__tickets", "ticket", _IMG)

        assert ocr_calls == ["image/png"], "an image must be OCR'd even when multimodal is on"
        r = await c.get("/v1/kb/cag__tickets/documents/ticket/content")
        assert r.status_code == 200, r.text
        stored = " ".join(ch["text"] for ch in r.json()["chunks"])
        assert "BOOKING REFERENCE HC76IW" in stored, "the OCR text must be indexed"
        assert "[Image description]" in stored, "the caption is added alongside, not dropped"
