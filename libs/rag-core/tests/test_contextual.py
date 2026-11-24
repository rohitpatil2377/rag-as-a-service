"""Unit tests for the deterministic metadata contextualizer (FR-C2)."""

from __future__ import annotations

from datetime import UTC, datetime

from rag_core import MetadataContextualizer, contextualize_text
from rag_schemas import Chunk, DocumentEnvelope
from rag_schemas.common import SourceKind


def _doc() -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id="t",
        doc_id="brief",
        source_id="s",
        source_kind=SourceKind.UPLOAD,
        mime="text/plain",
        content_b64="",
        metadata={"title": "Platform Brief", "dept": "cloud"},
        ingested_at=datetime(2026, 7, 2, tzinfo=UTC),
    )


def _chunk() -> Chunk:
    return Chunk(chunk_id="brief::0", doc_id="brief", tenant_id="t", text="Uptime target is 99.5 percent.", page=1)


async def test_metadata_context_includes_title_and_page() -> None:
    ctx = await MetadataContextualizer().context_for(_doc(), _chunk(), "full document text")
    assert "Platform Brief" in ctx
    assert "page 1" in ctx
    assert "dept=cloud" in ctx
    assert ctx.startswith("[context]")


def test_contextualize_text_prepends_and_preserves_original() -> None:
    out = contextualize_text("[context] Document Platform Brief", "Uptime target is 99.5 percent.")
    assert out.endswith("Uptime target is 99.5 percent.")
    assert out.startswith("[context]")
    # Empty context is a no-op (original returned).
    assert contextualize_text("", "abc") == "abc"


# --- the section a chunk was written under (P14) ------------------------------------------------


def _sop_text() -> str:
    return (
        "Acme Content Governance Master SOP\n\n"
        "6.1.1 Mandatory Rehearsal Policy\n"
        "Written sign-off is required before publication.\n\n"
        "6.4.1 Central Content Repository\n"
        "Written sign-off is required before publication.\n"
    )


def _chunk_at(text: str, needle: str, nth: int = 0) -> Chunk:
    """Through the real chunker, because the section is RECORDED there and read here — the
    point of the change is that the two cannot compute it differently."""
    from rag_core.chunking import ClauseChunker
    from rag_core.sections import detect_sections

    doc = DocumentEnvelope(
        tenant_id="t",
        doc_id="sop",
        source_id="s",
        source_kind=SourceKind.UPLOAD,
        mime="application/pdf",
        ingested_at=datetime(2026, 7, 31, tzinfo=UTC),
    )
    chunks = ClauseChunker(max_chars=90, overlap=0).chunk(doc, text, sections=detect_sections(text))
    carrying = [c for c in chunks if needle in c.text]
    return carrying[nth]


def test_identical_text_under_different_headings_gets_different_context() -> None:
    """The case section paths exist for. Both chunks read the same; only the heading they sat
    under distinguishes them, and that heading lives in some OTHER chunk."""
    import asyncio

    text = _sop_text()
    doc = DocumentEnvelope(
        tenant_id="t",
        doc_id="sop",
        source_id="s",
        source_kind=SourceKind.UPLOAD,
        mime="application/pdf",
        ingested_at=datetime(2026, 7, 31, tzinfo=UTC),
    )
    ctx = MetadataContextualizer()
    first = asyncio.run(ctx.context_for(doc, _chunk_at(text, "Written sign-off is required", 0), text))
    second = asyncio.run(ctx.context_for(doc, _chunk_at(text, "Written sign-off is required", 1), text))

    assert "6.1.1 Mandatory Rehearsal Policy" in first
    assert "6.4.1 Central Content Repository" in second
    assert first != second


def test_a_document_without_numbering_is_contextualized_as_before() -> None:
    """No path is the safe outcome: the chunk is embedded exactly as it is today."""
    import asyncio

    text = "Annual leave is 24 days.\n"
    doc = DocumentEnvelope(
        tenant_id="t",
        doc_id="hr",
        source_id="s",
        source_kind=SourceKind.UPLOAD,
        mime="text/plain",
        ingested_at=datetime(2026, 7, 31, tzinfo=UTC),
    )
    out = asyncio.run(MetadataContextualizer().context_for(doc, _chunk_at(text, "Annual leave"), text))
    assert "section" not in out
