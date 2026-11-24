"""Document Intelligence service (TDD §4.2 · FR-B1/B2/B3/B6/G2).

Turns raw document bytes into masked, page-mapped extracted text ready for indexing.
Every document passes the ingestion OCR check (``ocr_check``): whatever its type, anything whose
words exist only as pixels is read by the OCR engine — an image file, a scanned PDF page, a large
image pasted onto a texty PDF page, or media embedded in a Word/PowerPoint/Excel file. Text layers
are read directly and never replaced by OCR.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace

from rag_common import ValidationError, get_logger
from rag_schemas import DocumentEnvelope
from rag_schemas.common import Modality
from svc_doc_intelligence.classify import DEFAULT_SCANNED_CHARS_PER_PAGE, DocKind, classify, page_needs_ocr
from svc_doc_intelligence.fields import extract_fields
from svc_doc_intelligence.ocr import (
    CAP_CAPTION,
    PAGE_SEPARATOR,
    REASON_ISOLATED_NORMALIZATION,
    REASON_OCR_FAILED,
    REASON_OCR_IMAGE_CAP,
    REASON_OCR_UNAVAILABLE,
    Extraction,
    LayoutOverlay,
    NoOpOCR,
    OCREngine,
    PageRecord,
    RenderedText,
    extract_html_text,
    extract_office_text,
    extract_pdf_text,
    extract_plain_text,
    normalize_extracted_text,
    pptx_slide_parts,
    provenance_tags,
    split_pdf_page,
)
from svc_doc_intelligence.ocr_check import (
    DEFAULT_MAX_IMAGES_PER_DOC,
    DEFAULT_MIN_IMAGE_EDGE_PX,
    DEFAULT_MIN_PAGE_SHARE,
    OCR_REASON_EMBEDDED_IMAGES,
    OCR_REASON_IMAGE_FILE,
    OCR_REASON_PAGE_IMAGES,
    OCR_REASON_SCANNED_PAGE,
    image_mime,
    image_worth_ocr,
    office_embedded_images,
    pdf_page_image_crops,
)
from svc_doc_intelligence.pii import PIIMasker, RegexPIIMasker
from svc_doc_intelligence.sanitize import CleanResult, clean

_log = get_logger(__name__)


# Every zip container, whatever it turns out to be. The kinds are distinct because they need
# distinct extractors, but they share ONE entry point so the namelist markers live in one place.
_ARCHIVE_KINDS = frozenset({DocKind.OFFICE_DOC, DocKind.OFFICE_SHEET, DocKind.OFFICE_SLIDES, DocKind.ARCHIVE})

# (image bytes, mime, tenant_id) -> what a vision model says the image shows. Injected by the
# control plane when multimodal is on; the service never imports a model client itself.
ImageDescriber = Callable[[bytes, str, str], Awaitable[str]]

_CAPTION_LABEL = "[Image description]"
_EMBEDDED_LABEL = "[Text read from an embedded image]"


def _stamp_reason(extraction: Extraction, reason: str) -> Extraction:
    """Mark every page of an engine's result with why OCR ran on it."""
    return replace(extraction, pages=[replace(p, ocr_reason=reason) for p in extraction.pages])


def _degrade(page: PageRecord, reason: str) -> PageRecord:
    """Stamp a reason on a page we could not improve, keeping whatever text it already had."""
    return replace(page, degradations=tuple(sorted({*page.degradations, reason})))


@dataclass(slots=True)
class ExtractedDoc:
    text: str
    page_map: list[int]
    modality: Modality
    min_confidence: float
    pii_masked: int
    # Per-entity redaction counts ({"AADHAAR": 3, "EMAIL": 1, …}) — drives the PII classification
    # floor at ingest (which TYPE was found, not just how many). Empty when nothing was masked.
    pii_counts: dict[str, int] = field(default_factory=dict)
    fields: dict[str, str] = field(default_factory=dict)  # structured fields (FR-B2)
    # Which extractor produced `text`, what it could recover, and whether anything degraded
    # on the way. Stamped onto the document's stored metadata so the corpus stays auditable.
    provenance: dict[str, str] = field(default_factory=dict)
    # The engine's raw markup for the pages that have any (see `ocr.LayoutOverlay`). An OVERLAY
    # on `text`, carried through so the pipeline can PERSIST it; `text`, `page_map` and
    # everything derived from them are untouched by it. `extract` always sets it, so an overlay
    # with no pages means "this document's engine produced no markup" and None means only that
    # the caller constructed an ExtractedDoc without going through `extract`.
    layout: LayoutOverlay | None = None


def _sanitization_tags(verdict: CleanResult) -> dict[str, str]:
    """Sanitisation verdict as metadata tags (stamped onto the document, auditable)."""
    return {
        "sanitization_risk": verdict.risk.value,
        "sanitization_hits": str(len(verdict.matches)),
        "hidden_chars_removed": str(verdict.hidden_removed),
    }


def _decloak_rendered(rendered: RenderedText) -> RenderedText:
    """De-cloak + NFKC-normalise the text that gets INDEXED, keeping the page map consistent.

    ``sanitize.clean()`` is what the pipeline's contract promises indexes: zero-width / soft-hyphen
    / NFKC-compatibility / control characters removed so a cloaked payload can never reach the index
    or the citation viewer. But ``clean()`` runs over the WHOLE document — its ``_BLANK_RUN`` collapse
    and outer ``.strip()`` rewrite the ``PAGE_SEPARATOR`` boundaries, so its output no longer lines up
    with ``rendered.page_map`` (offsets would drift and citations land on the wrong page).

    Trade-off (documented): we clean each page's content INDIVIDUALLY and re-render with the separator,
    exactly as :meth:`Extraction.render` does, so ``sum(page_map) == len(text)`` still holds and every
    offset maps to the right page. The RISK VERDICT stays a whole-document ``clean()`` in ``extract`` so
    a payload split across a page boundary is still caught — only the INDEXED text is rebuilt per-page.
    Per-page normalisation cannot collapse whitespace that straddles a page break, which is correct:
    such a run is never one word, so nothing indexable is lost.
    """
    page_texts: list[str] = []
    cursor = 0
    last = len(rendered.page_map) - 1
    for i, length in enumerate(rendered.page_map):
        segment = rendered.text[cursor : cursor + length]
        cursor += length
        # Non-final pages carry the trailing separator inside their span; clean the CONTENT only,
        # then the re-render below re-emits the separator (which clean()'s .strip() would drop).
        if i != last and segment.endswith(PAGE_SEPARATOR):
            segment = segment[: -len(PAGE_SEPARATOR)]
        page_texts.append(clean(segment).text)

    parts: list[str] = []
    page_map: list[int] = []
    last = len(page_texts) - 1
    for i, page_text in enumerate(page_texts):
        emitted = page_text if i == last else page_text + PAGE_SEPARATOR
        parts.append(emitted)
        page_map.append(len(emitted))
    return RenderedText(text="".join(parts), page_map=page_map)


class DocIntelligenceService:
    """Extract + mask + (optional) field-extract. Wired with a PII masker and OCR engine."""

    def __init__(
        self,
        masker: PIIMasker | None = None,
        ocr_engine: OCREngine | None = None,
        *,
        extract_fields_enabled: bool = False,
        scanned_chars_per_page: int = DEFAULT_SCANNED_CHARS_PER_PAGE,
        pdf_image_pages: str = "text_only",
        image_describer: ImageDescriber | None = None,
        embedded_image_ocr: bool = True,
        min_image_edge_px: int = DEFAULT_MIN_IMAGE_EDGE_PX,
        min_image_page_share: float = DEFAULT_MIN_PAGE_SHARE,
        max_ocr_images: int = DEFAULT_MAX_IMAGES_PER_DOC,
        ocr_concurrency: int = 4,
        pdf_layout_overlay: bool = False,
    ) -> None:
        self._masker = masker or RegexPIIMasker()
        self._ocr = ocr_engine or NoOpOCR()
        self._extract_fields = extract_fields_enabled
        self._scanned_chars_per_page = scanned_chars_per_page
        # "text_only" (default): a page with a usable text layer is read directly, so an image
        # EMBEDDED on a texty page (chart/figure/photo) is not read. "ocr": ANY page that draws an
        # image is also sent through the OCR engine (Chandra reads text+tables+figures together)
        # and merged back by page — captures embedded-image content at the cost of more OCR calls.
        self._pdf_image_pages = pdf_image_pages
        # A caption is ADDED to an image's OCR text, never a substitute for it: OCR reads the
        # words, a vision model says what the picture shows. See _route_image.
        self._describe = image_describer
        self._embedded_ocr = embedded_image_ocr
        self._min_edge_px = min_image_edge_px
        self._min_page_share = min_image_page_share
        self._max_ocr_images = max_ocr_images
        self._ocr_concurrency = max(1, ocr_concurrency)
        # Capture the born-digital PDF layout overlay (`ocr.MARKUP_FORMAT_LAYOUT_JSON`)? Wired
        # from `v2_ingest`, and gating the CAPTURE rather than only the persist is a
        # DELIBERATE asymmetry with the Chandra path: Chandra's markup already arrived in the
        # response and was being discarded, so keeping it cost nothing and is unconditional,
        # while a pdfplumber layout pass is real work on every page of every PDF. Off ⇒ the pass
        # does not run at all and `text` is byte-identical.
        self._pdf_layout_overlay = pdf_layout_overlay

    @property
    def masker(self) -> PIIMasker:
        """The masker THIS service de-identified ``ExtractedDoc.text`` with (FR-B6/G2).

        Exposed for one caller: RAG v2 Phase 5 chunks a document from its layout tree, and the
        layout overlay is captured from the UNMASKED engine output (see :meth:`extract`), so the
        chunker has to run the same masker over the text it derives from the tree before it becomes
        a ``Chunk``. "The same one" is the point — a second masker built from config beside this
        service could drift in strategy, salt or entity set, and then the structure-derived chunks
        of a document would be de-identified differently from its own indexed text."""
        return self._masker

    def prepare_for_index(self, value: str) -> str:
        """Make an arbitrary string from THIS extraction safe to index: normalise, mask, de-cloak.

        The three steps `extract` applies to the page text, in the same order, for a caller that
        derived its own text from this extraction — today the layout-aware chunker, whose units come
        from the layout overlay rather than from `ExtractedDoc.text`. Handing that caller the bare
        masker was two defects:

        * the overlay is captured BEFORE `normalize_extracted_text`, so a structured identifier
          broken across a line ("Card 4111 1111" / "1111 1111 on file.", or "PAN ABCDE-" /
          "1234F") was two fragments that matched no pattern. Measured: the text layer masked both,
          the block stream masked neither, and `extracted.pii_counts` — computed from the text
          layer — still reported the document de-identified, so the INDEX audit event, the PII
          classification floor and the signed DPDP attestation all agreed with the wrong answer.
        * `sanitize.clean` never ran, though its own contract is that ONLY cleaned text reaches the
          index. A zero-width/soft-hyphen/NFKC-foldable payload under the document-level risk
          threshold went verbatim into `Chunk.text`, the embedding, BM25, the citation viewer and
          the generation prompt.

        Normalise BEFORE masking (the masker must see healed identifiers) and clean AFTER (so a
        redaction token cannot be cloaked). Returning a plain `str` is deliberate: the caller has no
        page map to keep in step, which is the one thing `extract` needs the richer types for.
        """
        return clean(self._masker.mask(normalize_extracted_text(value)).text).text

    async def extract(self, doc: DocumentEnvelope, raw: bytes) -> ExtractedDoc:
        extraction = await self._route(doc, raw)
        # The joined text and its page map are produced together and cannot be obtained apart.
        rendered = extraction.render()
        # FR-B2: pull structured fields from the RAW text (before PII masking removes IDs).
        fields = extract_fields(rendered.text) if self._extract_fields else {}
        masked = self._masker.mask(rendered.text)
        # Redaction changes lengths ("officer@acme.example" -> "[REDACTED:EMAIL]"), so the page
        # map has to move with the text. The masker reports WHERE it redacted and the type that
        # owns the invariant rebuilds both together, which is why this can now be a RenderedText
        # again: `__post_init__` would reject any pair that had drifted apart.
        rendered = rendered.apply_replacements(masked.spans)
        # Adversarial-content scan (indirect prompt injection). Scans the NORMALISED text so a
        # zero-width/homoglyph-cloaked payload is revealed first; the verdict rides provenance
        # (metadata), so the pipeline can QUARANTINE a SUSPECT doc instead of indexing it.
        verdict = clean(rendered.text)
        # The de-cloaked text is what gets INDEXED + shown in the citation viewer: sanitize.py's
        # contract is that ONLY cleaned text reaches the index, so zero-width / soft-hyphen /
        # NFKC-compat / control chars must not survive here. Rebuilt per-page so page_map/offsets
        # stay consistent (see _decloak_rendered); the whole-document verdict above is unchanged, so
        # a SUSPECT doc still quarantines exactly as before.
        indexed = _decloak_rendered(rendered)
        _log.info(
            "doc_extracted",
            doc_id=doc.doc_id,
            pii=masked.total,
            mode=extraction.modality,
            fields=len(fields),
            extractor=extraction.extractor,
            sanitization=verdict.risk.value,
            degradations=list(extraction.degradations),
        )
        return ExtractedDoc(
            text=indexed.text,
            page_map=indexed.page_map,
            modality=extraction.modality,
            min_confidence=extraction.min_confidence,
            pii_masked=masked.total,
            pii_counts=dict(masked.counts),
            fields=fields,
            provenance={**provenance_tags(extraction), **_sanitization_tags(verdict)},
            # Captured from the UNMASKED engine output, like the raw bytes and for the same
            # reason: it is an extraction artifact, not indexed content. See
            # `svc_control_plane.pipeline._persist_layout_overlay` for the compliance position.
            layout=extraction.layout_overlay(),
        )

    async def _route(self, doc: DocumentEnvelope, raw: bytes) -> Extraction:
        """Auto-pick the best extractor for this file (FR-B4). The classifier identifies the
        kind from MIME + magic bytes; PDFs are further split native-vs-scanned PER PAGE.
        Digital formats extract directly (perfect); scans/images/unknown go to OCR."""
        kind = classify(doc.mime, raw)
        _log.info("doc_classified", doc_id=doc.doc_id, mime=doc.mime, kind=kind.value)
        if kind is DocKind.NATIVE_PDF:
            return await self._route_pdf(doc, raw)
        if kind in _ARCHIVE_KINDS:
            # One entry point for every zip container: it re-reads the namelist and picks the
            # right extractor, or names the archive it cannot read. Routing each OOXML family
            # from here would put the marker table in two places.
            extraction = extract_office_text(raw, doc_id=doc.doc_id)
            return await self._ocr_office_images(doc, raw, kind, extraction)
        if kind is DocKind.HTML:
            return extract_html_text(raw)
        if kind is DocKind.TEXT:
            return extract_plain_text(raw)
        if kind is DocKind.IMAGE:
            return await self._route_image(doc, raw)
        # SCANNED_PDF / UNKNOWN → OCR (the only way to get text out of pixels).
        return _stamp_reason(await self._ocr_or_explain(doc, raw, kind=kind, pages=1), OCR_REASON_SCANNED_PAGE)

    async def _route_pdf(self, doc: DocumentEnvelope, raw: bytes) -> Extraction:
        """Read the PDF's text layer, then decide page by page which pages still need OCR."""
        extraction = extract_pdf_text(raw, doc_id=doc.doc_id, capture_layout=self._pdf_layout_overlay)
        # In "ocr" mode, ANY page carrying an image is OCR'd — even one with a full text layer —
        # so an embedded figure/chart/table on a texty page is read (Chandra returns text+figures
        # together). In the default "text_only" mode only a textless image page needs OCR.
        ocr_all_image_pages = self._pdf_image_pages == "ocr"
        scanned = [
            p.page_no
            for p in extraction.pages
            if (ocr_all_image_pages and p.has_image_content)
            # Two DIFFERENT questions, so two different signals:
            #   * the image-page policy above: is a PICTURE carrying this page's content?
            #   * the blank-page test: does this page have no usable text, and is there ink on it?
            # Vector ink answers only the second. A page whose glyphs were traced into outlines
            # has no characters and no images, but it is emphatically not blank.
            or page_needs_ocr(
                text_layer_chars=len(p.text),
                has_image_content=p.has_image_content or p.has_vector_ink,
                min_chars_per_page=self._scanned_chars_per_page,
            )
        ]
        if not scanned:
            # Every page has a usable text layer — no page-level OCR. Unless there is no text
            # ANYWHERE: a PDF of entirely blank pages is not a scan (nothing to OCR) but it is
            # also not indexable, and indexing it silently is the failure this guard exists to
            # prevent. Distinct from the needs-OCR refusal below: OCR would not help.
            self._refuse_if_textless(doc, extraction)
            # A texty page can still carry a pasted scan or a screenshot of a table.
            return await self._ocr_pdf_page_images(doc, raw, extraction)

        if len(scanned) == len(extraction.pages):
            # EVERY page takes the same path, so hand the whole document to the engine in one
            # call: same result, one poppler spawn instead of N, and cross-page normalization
            # inside the engine still sees the full page set.
            _log.warning("pdf_looks_scanned", doc_id=doc.doc_id, pages=len(extraction.pages))
            whole = await self._ocr_or_explain(doc, raw, kind=DocKind.SCANNED_PDF, pages=len(extraction.pages))
            return _stamp_reason(whole, OCR_REASON_SCANNED_PAGE)

        _log.info(
            "pdf_pages_need_ocr",
            doc_id=doc.doc_id,
            pages_total=len(extraction.pages),
            pages_needing_ocr=len(scanned),
            ocr_pages=scanned,
            min_chars_per_page=self._scanned_chars_per_page,
        )
        reasons = {
            p.page_no: (
                OCR_REASON_SCANNED_PAGE
                if page_needs_ocr(
                    text_layer_chars=len(p.text),
                    has_image_content=p.has_image_content or p.has_vector_ink,
                    min_chars_per_page=self._scanned_chars_per_page,
                )
                else OCR_REASON_PAGE_IMAGES
            )
            for p in extraction.pages
            if p.page_no in scanned
        }
        merged = await self._ocr_selected_pages(doc, raw, extraction, scanned, reasons)
        # The pages that kept their text layer can still carry a pasted scan.
        return await self._ocr_pdf_page_images(doc, raw, merged)

    async def _ocr_selected_pages(
        self,
        doc: DocumentEnvelope,
        raw: bytes,
        extraction: Extraction,
        page_nos: list[int],
        reasons: dict[int, str] | None = None,
    ) -> Extraction:
        """OCR just the pages that need it and merge the results back by page number.

        Each page is split out as a standalone one-page PDF (a structural copy, no rasterising)
        and sent through the unchanged OCREngine seam, so the engine renders one page instead
        of the whole document. The pages that had a text layer are never touched."""
        replacements: dict[int, PageRecord] = {}
        engine_missing: str | None = None
        for page_no in page_nos:
            original = extraction.pages[page_no - 1]
            if engine_missing is not None:
                # The engine is absent for the whole run — do not retry it once per page.
                replacements[page_no] = _degrade(original, REASON_OCR_UNAVAILABLE)
                continue
            try:
                page_bytes = split_pdf_page(raw, page_no)
                result = await self._ocr.ocr(page_bytes, doc.mime)
                ocr_page = result.pages[0]
            except NotImplementedError as exc:
                engine_missing = str(exc)
                # NOT fatal here: unlike a wholly-scanned document there IS text worth keeping
                # on the other pages, so the document is indexed with the unreadable pages
                # stamped rather than rejected outright.
                _log.warning(
                    "pdf_page_ocr_unavailable",
                    doc_id=doc.doc_id,
                    pages=page_nos,
                    detail=engine_missing,
                )
                replacements[page_no] = _degrade(original, REASON_OCR_UNAVAILABLE)
            except Exception as exc:
                _log.warning(
                    "pdf_page_ocr_failed",
                    doc_id=doc.doc_id,
                    page=page_no,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    exc_info=True,
                )
                replacements[page_no] = _degrade(original, REASON_OCR_FAILED)
            else:
                replacements[page_no] = replace(
                    ocr_page,
                    page_no=page_no,  # the engine saw a one-page PDF; restore the document's number
                    degradations=tuple(sorted({*ocr_page.degradations, REASON_ISOLATED_NORMALIZATION})),
                    ocr_reason=(reasons or {}).get(page_no, OCR_REASON_SCANNED_PAGE),
                )
        pages = [replacements.get(p.page_no, p) for p in extraction.pages]
        return Extraction(pages=pages, modality=extraction.modality)

    async def _route_image(self, doc: DocumentEnvelope, raw: bytes) -> Extraction:
        """An image file: OCR reads its words, and a vision caption (when enabled) is ADDED.

        The two answer different questions. OCR recovers the text — measured live on an e-ticket,
        4,544 characters against the caption's 32. A caption says what the picture shows, which
        matters for a photo or a diagram with no words at all. Replacing OCR with the caption is
        what this method exists to stop: every image in two OCR knowledge bases was indexed as a
        one-line description while the OCR engine sat idle.

        If OCR fails but a caption exists, the document is indexed on the caption and stamped
        degraded, so the gap is visible rather than silent. With neither, ingestion fails loudly."""
        ocr_result: Extraction | None = None
        ocr_problem = ""
        ocr_error: Exception | None = None
        try:
            ocr_result = _stamp_reason(await self._ocr.ocr(raw, doc.mime), OCR_REASON_IMAGE_FILE)
        except NotImplementedError as exc:
            ocr_problem, ocr_error = REASON_OCR_UNAVAILABLE, exc
        except Exception as exc:
            _log.warning(
                "image_ocr_failed",
                doc_id=doc.doc_id,
                error_type=type(exc).__name__,
                error=str(exc)[:300],
                exc_info=True,
            )
            ocr_problem, ocr_error = REASON_OCR_FAILED, exc
        caption = await self._describe_image(doc, raw)

        if ocr_result is None:
            if not caption:
                if isinstance(ocr_error, NotImplementedError):
                    return await self._ocr_or_explain(doc, raw, kind=DocKind.IMAGE, pages=1)  # raises the 422
                assert ocr_error is not None
                raise ocr_error
            page = PageRecord(
                page_no=1,
                text=f"{_CAPTION_LABEL} {caption}",
                extractor="vision_caption",
                capabilities=(CAP_CAPTION,),
                degradations=(ocr_problem,),
            )
            _log.warning("image_indexed_on_caption_only", doc_id=doc.doc_id, reason=ocr_problem)
            return Extraction(pages=[page], modality=Modality.VISUAL)

        pages = list(ocr_result.pages)
        if caption:
            first = pages[0]
            body = first.text.strip()
            pages[0] = replace(
                first,
                text=f"{body}\n\n{_CAPTION_LABEL} {caption}" if body else f"{_CAPTION_LABEL} {caption}",
                capabilities=tuple(sorted({*first.capabilities, CAP_CAPTION})),
            )
        if not any(p.text.strip() for p in pages):
            raise ValidationError(
                f'"{doc.doc_id}" is an image, and OCR found no readable text in it'
                + ("" if self._describe else " (image description is not enabled on this deployment)")
                + ". Upload a clearer scan, or a document with a text layer.",
                details={"doc_id": doc.doc_id, "kind": DocKind.IMAGE.value, "pages": len(pages)},
            )
        _log.info(
            "image_ocr_done",
            doc_id=doc.doc_id,
            ocr_chars=sum(len(p.text) for p in ocr_result.pages),
            caption_chars=len(caption),
        )
        return replace(ocr_result, pages=pages)

    async def _describe_image(self, doc: DocumentEnvelope, raw: bytes) -> str:
        """The vision caption for an image, or "" when captioning is off or failed.

        A failed caption is logged and skipped, never fatal: OCR carries the image's text, and
        a missing description loses only the words about the picture, not the words in it."""
        if self._describe is None:
            return ""
        try:
            return (await self._describe(raw, doc.mime, doc.tenant_id)).strip()
        except Exception as exc:
            _log.warning("image_caption_failed", doc_id=doc.doc_id, error_type=type(exc).__name__, error=str(exc)[:300])
            return ""

    async def _ocr_pdf_page_images(self, doc: DocumentEnvelope, raw: bytes, extraction: Extraction) -> Extraction:
        """Read large images pasted onto PDF pages that KEPT their text layer.

        Only the image regions are OCR'd and their text is appended to the page; the text layer
        itself is left exactly as extracted (see ``ocr_check.pdf_page_image_crops``)."""
        if not self._embedded_ocr:
            return extraction
        candidates = [
            p.page_no for p in extraction.pages if p.has_image_content and not p.ocr_reason and p.text.strip()
        ]
        if not candidates:
            return extraction
        try:
            crops = await asyncio.to_thread(
                pdf_page_image_crops,
                raw,
                candidates,
                min_page_share=self._min_page_share,
                min_edge_px=self._min_edge_px,
            )
        except Exception as exc:
            # Rendering is best-effort: the text layer is already safe, so a crop failure stamps
            # the candidate pages instead of losing the document.
            _log.warning(
                "pdf_image_crop_failed", doc_id=doc.doc_id, error_type=type(exc).__name__, error=str(exc)[:300]
            )
            stamped = {n: _degrade(extraction.pages[n - 1], REASON_OCR_FAILED) for n in candidates}
            return replace(extraction, pages=[stamped.get(p.page_no, p) for p in extraction.pages])
        return await self._append_image_ocr(doc, extraction, crops)

    async def _ocr_office_images(
        self, doc: DocumentEnvelope, raw: bytes, kind: DocKind, extraction: Extraction
    ) -> Extraction:
        """Read images embedded in a Word, PowerPoint or Excel file.

        A slide's images are attached to that slide's page. Word and Excel give no page an image
        can honestly be attributed to, so their images are appended after the body."""
        if not self._embedded_ocr or not extraction.pages:
            return extraction
        slide_parts = pptx_slide_parts(raw) if kind is DocKind.OFFICE_SLIDES else None
        by_index = office_embedded_images(raw, slide_parts=slide_parts)
        images_by_page: dict[int, list[bytes]] = {}
        last_page = len(extraction.pages)
        for index, images in by_index.items():
            page_no = index + 1 if slide_parts is not None else last_page
            if page_no > last_page:
                continue
            worth = [b for b in images if image_worth_ocr(b, min_edge_px=self._min_edge_px)]
            if worth:
                images_by_page.setdefault(page_no, []).extend(worth)
        return await self._append_image_ocr(doc, extraction, images_by_page)

    async def _append_image_ocr(
        self, doc: DocumentEnvelope, extraction: Extraction, images_by_page: dict[int, list[bytes]]
    ) -> Extraction:
        """OCR embedded images and append what they say to the page they belong to.

        Bounded twice: at most ``max_ocr_images`` per document, ``ocr_concurrency`` at a time.
        Images past the cap are not read and their pages are stamped, so a 300-screenshot deck
        cannot stall the ingest queue and cannot pass as fully read either."""
        queue = [(page_no, img) for page_no in sorted(images_by_page) for img in images_by_page[page_no]]
        if not queue:
            return extraction
        kept, skipped = queue[: self._max_ocr_images], queue[self._max_ocr_images :]
        gate = asyncio.Semaphore(self._ocr_concurrency)

        async def read(img: bytes) -> str | Exception:
            async with gate:
                try:
                    result = await self._ocr.ocr(img, image_mime(img))
                except Exception as exc:  # recorded per image; one bad image must not sink the rest
                    return exc
                return "\n\n".join(p.text.strip() for p in result.pages if p.text.strip())

        outcomes = await asyncio.gather(*(read(img) for _, img in kept))
        texts: dict[int, list[str]] = {}
        problems: dict[int, set[str]] = {}
        for (page_no, _), outcome in zip(kept, outcomes, strict=True):
            if isinstance(outcome, NotImplementedError):
                problems.setdefault(page_no, set()).add(REASON_OCR_UNAVAILABLE)
            elif isinstance(outcome, Exception):
                _log.warning(
                    "embedded_image_ocr_failed",
                    doc_id=doc.doc_id,
                    page=page_no,
                    error_type=type(outcome).__name__,
                    error=str(outcome)[:300],
                )
                problems.setdefault(page_no, set()).add(REASON_OCR_FAILED)
            elif outcome:
                texts.setdefault(page_no, []).append(outcome)
        for page_no, _ in skipped:
            problems.setdefault(page_no, set()).add(REASON_OCR_IMAGE_CAP)
        if skipped:
            _log.warning(
                "embedded_image_ocr_capped",
                doc_id=doc.doc_id,
                images_found=len(queue),
                images_read=len(kept),
                cap=self._max_ocr_images,
            )

        pages: list[PageRecord] = []
        for original in extraction.pages:
            record = original
            added = texts.get(original.page_no, [])
            if added:
                body = original.text.rstrip()
                appended = "".join(f"\n\n{_EMBEDDED_LABEL}\n{t}" for t in added)
                record = replace(
                    original,
                    text=(body + appended) if body else appended.lstrip("\n"),
                    ocr_reason=original.ocr_reason or OCR_REASON_EMBEDDED_IMAGES,
                    ocr_images=original.ocr_images + len(added),
                )
            for reason in sorted(problems.get(original.page_no, ())):
                record = _degrade(record, reason)
            pages.append(record)
        _log.info(
            "embedded_image_ocr_done",
            doc_id=doc.doc_id,
            images_found=len(queue),
            images_with_text=sum(len(v) for v in texts.values()),
            pages_enriched=len(texts),
        )
        return replace(extraction, pages=pages)

    def _refuse_if_textless(self, doc: DocumentEnvelope, extraction: Extraction) -> None:
        """Refuse a PDF that yielded no text at all, with 422 rather than an empty index entry."""
        if any(p.text.strip() for p in extraction.pages):
            return
        pages = len(extraction.pages)
        _log.warning("pdf_has_no_text", doc_id=doc.doc_id, pages=pages)
        raise ValidationError(
            f'"{doc.doc_id}" contains no extractable text on any of its {pages} page(s), and no '
            "page carries an image for OCR to read — it appears to be blank. Upload a document "
            "with a text layer or a readable scan.",
            details={"doc_id": doc.doc_id, "kind": DocKind.NATIVE_PDF.value, "pages": pages},
        )

    async def _ocr_or_explain(self, doc: DocumentEnvelope, raw: bytes, *, kind: DocKind, pages: int) -> Extraction:
        """Run OCR, or raise an actionable 422 when no OCR engine is enabled on this deployment."""
        try:
            return await self._ocr.ocr(raw, doc.mime)
        except NotImplementedError as exc:
            raise ValidationError(
                f'"{doc.doc_id}" needs OCR ({kind.value}: no machine-readable text layer), but OCR '
                "is not enabled on this deployment — upload a text-based document, or ask your "
                "operator to enable an OCR engine (RAG_OCR_BACKEND=tesseract).",
                details={"doc_id": doc.doc_id, "kind": kind.value, "pages": pages},
            ) from exc
