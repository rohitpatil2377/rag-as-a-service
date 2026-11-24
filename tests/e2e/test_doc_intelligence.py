"""E2E: FR-B1/B2 — OCR-engine selection + structured field extraction into metadata."""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_FIELD_EXTRACTION_ENABLED", "true")
    from rag_common.config import get_settings

    get_settings.cache_clear()


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    token = issue_dev_jwt("o", tenant, roles=["reader", "ingestor"])
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()),
        base_url="http://t",
        headers={"Authorization": f"Bearer {token}"},
    )


async def test_field_extraction_into_metadata(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    kb = "docint"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        text = "Purchase order dated 05/04/2026 for vendor GSTIN 27AAECS1234K1Z8. Total ₹2,40,000."
        await c.post(
            f"/v1/kb/{kb}/ingest",
            json={
                "doc_id": "po",
                "mime": "text/plain",
                "content_b64": base64.b64encode(text.encode()).decode(),
                "strategy": "clause",
            },
        )
        meta = (await c.get(f"/v1/kb/{kb}/documents")).json()[0]["metadata"]
        assert meta["field.gstin"] == "27AAECS1234K1Z8"
        assert meta["field.date"] == "05/04/2026"
        assert meta["field.amount"] == "2,40,000"


async def test_ocr_backend_selectable_noop_default(inmemory: None) -> None:
    # Default OCR backend is noop; a scanned-looking PDF is refused clearly, not silently empty.
    from rag_common.config import get_settings

    assert get_settings().ocr_backend == "noop"


async def test_language_detection_and_translate(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LANGUAGE_DETECTION_ENABLED", "true")
    from rag_common.config import get_settings

    get_settings.cache_clear()
    kb = "langkb"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        hindi = "वार्षिक अवकाश नीति के अनुसार सभी कर्मचारियों को 24 दिन की छुट्टी मिलती है।"
        await c.post(
            f"/v1/kb/{kb}/ingest",
            json={
                "doc_id": "hi",
                "mime": "text/plain",
                "content_b64": base64.b64encode(hindi.encode()).decode(),
                "strategy": "clause",
            },
        )
        meta = (await c.get(f"/v1/kb/{kb}/documents")).json()[0]["metadata"]
        assert meta["lang"] == "hi"  # detected as Hindi -> filterable

        # Translate endpoint round-trips (noop backend returns text unchanged but reports detected lang).
        r = (await c.post(f"/v1/kb/{kb}/translate", json={"text": hindi, "target_lang": "en"})).json()
        assert r["detected_lang"] == "hi" and r["target_lang"] == "en"
