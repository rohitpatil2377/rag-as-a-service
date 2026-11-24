"""Doc-pipeline hardening: DOCX extraction, table capture, normalization, scanned-PDF guard."""

from __future__ import annotations

import base64
import io
import pathlib
import zipfile

import httpx
import pytest

pytestmark = pytest.mark.e2e

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
# The OOXML fixture builder lives with the doc-intelligence tests that own the format.
_DOC_INTEL_TESTS = pathlib.Path(__file__).resolve().parents[2] / "services" / "doc-intelligence" / "tests"
_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _make_docx(paragraphs: list[str], table: list[list[str]] | None = None) -> bytes:
    """Build a minimal .docx in memory (a zip holding word/document.xml)."""

    def p(text: str) -> str:
        return f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"

    body = "".join(p(t) for t in paragraphs)
    if table:
        rows = "".join("<w:tr>" + "".join(f"<w:tc>{p(c)}</w:tc>" for c in row) + "</w:tr>" for row in table)
        body += f"<w:tbl>{rows}</w:tbl>"
    xml = f'<?xml version="1.0"?><w:document xmlns:w="{_W}"><w:body>{body}</w:body></w:document>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", xml)
    return buf.getvalue()


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["reader", "ingestor"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


def test_docx_extracts_paragraphs_and_tables() -> None:
    from svc_doc_intelligence.ocr import extract_docx_text

    data = _make_docx(
        ["Annual Leave Policy", "Employees receive paid leave as below."],
        table=[["Grade", "Days"], ["Officer", "24"], ["Staff", "18"]],
    )
    text = extract_docx_text(data, doc_id="fixture.docx").render().text
    assert "Annual Leave Policy" in text
    # The .docx table goes through the SAME serializer as the PDF path, so its rows are
    # self-describing too — one format across every extractor, not a per-extractor rendering.
    assert "Grade = Officer | Days = 24" in text


def test_normalization_dehyphenates_and_strips_running_headers() -> None:
    from svc_doc_intelligence.ocr import _normalize_pages

    pages = [
        "MINISTRY OF FINANCE\nThe alloca-\ntion increased.\nPage 1",
        "MINISTRY OF FINANCE\nSpending was stable.\nPage 2",
        "MINISTRY OF FINANCE\nDebt fell sharply.\nPage 3",
    ]
    out = [p.text for p in _normalize_pages(pages)]
    assert all("MINISTRY OF FINANCE" not in p for p in out)  # running header removed
    assert "allocation increased" in out[0]  # de-hyphenated across the line break
    # The gap this used to pin is CLOSED. A worded footer whose number changes ("Page 1") is
    # furniture now, because a line whose only words are page words cannot be a heading — it was
    # sitting inside 324 of 554 chunks of a real handbook. The rule stayed narrow exactly so the
    # case it was protecting still holds; that case is the next test.
    assert all("Page" not in p for p in out), out


def test_a_numbered_clause_heading_survives_footer_stripping() -> None:
    """What the footer rule was always protecting: "Article 5." repeats at the top of every page
    with a different number, exactly like a footer, and is content. Losing a footer is a nuisance;
    losing a clause number is a wrong answer."""
    from svc_doc_intelligence.ocr import _normalize_pages

    pages = [
        "Article 5.\nThe lessee shall maintain the premises.\nPage 1",
        "Article 6.\nThe lessor shall insure the building.\nPage 2",
        "Article 7.\nNotice shall be served in writing.\nPage 3",
    ]
    out = [p.text for p in _normalize_pages(pages)]
    assert [p.splitlines()[0] for p in out] == ["Article 5.", "Article 6.", "Article 7."]
    assert all("Page" not in p for p in out)


async def test_docx_ingest_end_to_end(inmemory: None, lexical_gate: None) -> None:
    tenant = "acme"
    data = _make_docx(["The office relocation completes on 15 August 2026."])
    client = await _client(tenant)
    async with client:
        await client.post("/v1/knowledge-bases", json={"display_name": "KB"})
        r = await client.post(
            f"/v1/kb/{tenant}/ingest",
            json={"doc_id": "memo.docx", "mime": _DOCX_MIME, "content_b64": base64.b64encode(data).decode()},
        )
        assert r.status_code == 200, r.text
        assert r.json()["chunk_count"] >= 1

        r2 = await client.post(f"/v1/kb/{tenant}/query", json={"query": "When does the office relocation complete?"})
        ans = r2.json()
        assert "15 August" in ans["answer"]


async def test_textless_pdf_fails_clearly_not_silently(inmemory: None) -> None:
    """A PDF with no extractable text must 422 with a human explanation — never be silently
    indexed as an empty document.

    This fixture is a genuinely BLANK page: no text and no image. Since page routing became
    per-page it is no longer classified as a scan (OCR would find nothing on it), so it is
    refused for its own reason. The image-only-page variant, which does route to OCR and 422s
    when no engine is configured, is covered in test_page_routing.py."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()  # no text, no image — an empty page
    data = bytes(pdf.output())

    tenant = "acme"
    client = await _client(tenant)
    async with client:
        await client.post("/v1/knowledge-bases", json={"display_name": "KB"})
        r = await client.post(
            f"/v1/kb/{tenant}/ingest",
            json={"doc_id": "scan.pdf", "mime": "application/pdf", "content_b64": base64.b64encode(data).decode()},
        )
        assert r.status_code == 422
        assert "no extractable text" in r.json()["message"].lower()
        # And nothing was registered.
        assert (await client.get(f"/v1/kb/{tenant}/documents")).json() == []


async def test_spreadsheets_and_decks_ingest_instead_of_500ing(inmemory: None) -> None:
    """Every OOXML format starts PK\\x03\\x04, so the magic-byte table called them all OFFICE
    and the service handed each to the Word extractor. A workbook or a deck died on
    `KeyError: no item named 'word/document.xml'` and surfaced as a 500."""
    import sys

    sys.path.insert(0, str(_DOC_INTEL_TESTS))
    from ooxml_fixtures import make_pptx, make_xlsx

    tenant = "acme"
    client = await _client(tenant)
    async with client:
        await client.post("/v1/knowledge-bases", json={"display_name": "KB"})
        for doc_id, raw, needle in (
            ("leave.xlsx", make_xlsx({"Leave": [["Grade", "Days"], ["Officer", "24"]]}), "Officer"),
            ("deck.pptx", make_pptx([["Q3 Budget Review"]]), "Budget"),
        ):
            r = await client.post(
                f"/v1/kb/{tenant}/ingest",
                # Deliberately generic: the routing must come from the bytes, not the label.
                json={
                    "doc_id": doc_id,
                    "mime": "application/octet-stream",
                    "content_b64": base64.b64encode(raw).decode(),
                },
            )
            assert r.status_code == 200, f"{doc_id} -> {r.status_code} {r.text[:200]}"
            content = (await client.get(f"/v1/kb/{tenant}/documents/{doc_id}/content")).json()
            assert needle in " ".join(c["text"] for c in content["chunks"])


async def test_a_corrupt_upload_is_a_422_not_a_500(inmemory: None) -> None:
    """Every extractor bottoms out in a third-party library, and each raises its own type on
    an unparseable file — pypdf a PdfStreamError, zipfile a BadZipFile, a missing archive part
    a bare KeyError. None subclass PlatformError, so the middleware had nothing to map and the
    caller got a 500: a server fault reported for a broken input."""
    tenant = "acme"
    client = await _client(tenant)
    async with client:
        await client.post("/v1/knowledge-bases", json={"display_name": "KB"})
        for doc_id, mime, raw in (
            ("broken.pdf", "application/pdf", b"%PDF-1.4\n" + b"\x00garbage" * 50),
            ("broken.docx", _DOCX_MIME, b"PK\x03\x04" + b"\x00garbage" * 50),
        ):
            r = await client.post(
                f"/v1/kb/{tenant}/ingest",
                json={"doc_id": doc_id, "mime": mime, "content_b64": base64.b64encode(raw).decode()},
            )
            assert r.status_code == 422, f"{doc_id} -> {r.status_code}"
            assert doc_id in r.json()["message"]
        # Nothing half-ingested survives either.
        assert (await client.get(f"/v1/kb/{tenant}/documents")).json() == []


def _text_pdf(lines: list[str]) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    for line in lines:
        pdf.cell(0, 10, line, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


async def test_extraction_provenance_is_stored_and_refreshed_on_reindex(inmemory: None) -> None:
    """Which extractor produced the text — and whether anything degraded — must be stored on
    the document, on BOTH the ingest and the reindex path, so the corpus stays queryable."""
    from svc_doc_intelligence.ocr import CAP_TABLES, META_CAPABILITIES, META_DEGRADATION, META_DEGRADED, META_EXTRACTOR

    tenant = "acme"
    data = _text_pdf(["Leave entitlement is 24 days per year."])
    client = await _client(tenant)
    async with client:
        await client.post("/v1/knowledge-bases", json={"display_name": "KB"})
        r = await client.post(
            f"/v1/kb/{tenant}/ingest",
            json={"doc_id": "policy.pdf", "mime": "application/pdf", "content_b64": base64.b64encode(data).decode()},
        )
        assert r.status_code == 200, r.text

        meta = (await client.get(f"/v1/kb/{tenant}/documents")).json()[0]["metadata"]
        assert meta[META_EXTRACTOR] == "pdfplumber"
        assert CAP_TABLES in meta[META_CAPABILITIES]  # the table-capable path was used
        assert meta[META_DEGRADED] == "false"
        assert meta[META_DEGRADATION] == ""  # always present, so a stale reason can't survive

        # A replay re-runs extraction, so the stamp must be re-derived, not left behind.
        r2 = await client.post(f"/v1/kb/{tenant}/documents/policy.pdf/reindex", json={})
        assert r2.status_code == 200, r2.text
        after = (await client.get(f"/v1/kb/{tenant}/documents")).json()[0]["metadata"]
        assert after[META_EXTRACTOR] == "pdfplumber"
        assert after[META_DEGRADED] == "false"
