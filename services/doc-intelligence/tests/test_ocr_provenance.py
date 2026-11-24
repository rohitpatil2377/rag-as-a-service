"""Extraction provenance + loud fallback (observability for FR-B2 table capture).

`extract_pdf_text` downgrades from pdfplumber to pypdf on ANY failure, which silently
drops table capture for the WHOLE document. These tests pin the two things that make
that downgrade auditable: a WARN log carrying enough detail to diagnose it, and
provenance tags stamped on the record so the corpus can be counted after the fact.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import pytest
import structlog
from pdf_fixtures import make_pdf
from svc_doc_intelligence import ocr
from svc_doc_intelligence.ocr import (
    CAP_TABLES,
    CAP_TEXT,
    META_CAPABILITIES,
    META_DEGRADATION,
    META_DEGRADED,
    META_EXTRACTOR,
    REASON_EXTRACTOR_FALLBACK,
    Extraction,
    PageRecord,
    extract_pdf_text,
    provenance_tags,
)

from rag_schemas.common import Modality


def _make_pdf(pages_text: list[str]) -> bytes:
    """One line per page — see :mod:`pdf_fixtures` for the builder."""
    return make_pdf([[line] for line in pages_text])


def _fallback_events(logs: list[MutableMapping[str, Any]]) -> list[MutableMapping[str, Any]]:
    return [e for e in logs if e.get("event") == "pdf_extraction_degraded"]


# --- happy path: pdfplumber owns the text and declares table support ----------------------


def test_pdf_primary_path_records_extractor_and_table_capability() -> None:
    pdf = _make_pdf(["Annual Leave Policy", "Officer 24 days"])
    with structlog.testing.capture_logs() as logs:
        ex = extract_pdf_text(pdf, doc_id="policy.pdf")

    assert "Annual Leave Policy" in ex.render().text
    assert ex.extractor == "pdfplumber"
    assert CAP_TABLES in ex.capabilities  # pdfplumber is the table-capable extractor
    assert ex.degradations == ()
    assert _fallback_events(logs) == []  # nothing degraded -> nothing to warn about

    tags = provenance_tags(ex)
    assert tags[META_EXTRACTOR] == "pdfplumber"
    assert tags[META_DEGRADED] == "false"
    assert tags[META_DEGRADATION] == ""
    assert CAP_TABLES in tags[META_CAPABILITIES]


# --- the downgrade: loud, attributed, and stamped -----------------------------------------


def fail_pdfplumber_on(monkeypatch: pytest.MonkeyPatch, pages: set[int], message: str = "bad table bbox") -> None:
    """Make pdfplumber raise while reading the given 1-based pages, and only those.

    Hooks ``_page_tables``, which the extraction loop calls exactly once per page in order, so
    the Nth call is page N — including pages that have no table at all."""
    seen = {"n": 0}
    real = ocr._page_tables

    def maybe_boom(page: Any) -> Any:
        seen["n"] += 1
        if seen["n"] in pages:
            raise ValueError(message)
        return real(page)

    monkeypatch.setattr(ocr, "_page_tables", maybe_boom)


def test_one_bad_page_falls_back_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """The page that failed falls back; the pages that succeeded KEEP their tables."""
    pdf = _make_pdf(["page one alpha", "page two bravo", "page three charlie"])
    fail_pdfplumber_on(monkeypatch, {2}, message="cannot parse table on this page")

    with structlog.testing.capture_logs() as logs:
        ex = extract_pdf_text(pdf, doc_id="handbook.pdf")

    by_page = {p.page_no: p for p in ex.pages}
    for good in (1, 3):
        assert by_page[good].extractor == "pdfplumber"
        assert CAP_TABLES in by_page[good].capabilities  # NOT collateral damage
        assert by_page[good].degradations == ()
    assert by_page[2].extractor == "pypdf"
    assert by_page[2].capabilities == (CAP_TEXT,)
    assert by_page[2].degradations == (REASON_EXTRACTOR_FALLBACK,)

    # The document view stays derived: one page without tables => the DOCUMENT has no tables.
    assert ex.extractor == "pdfplumber,pypdf"
    assert ex.capabilities == (CAP_TEXT,)
    assert ex.degradations == (REASON_EXTRACTOR_FALLBACK,)
    tags = provenance_tags(ex)
    assert tags[META_DEGRADED] == "true"
    assert tags[META_DEGRADATION] == REASON_EXTRACTOR_FALLBACK
    assert CAP_TABLES not in tags[META_CAPABILITIES]

    # Per-page WARN: which page, why, and what it cost.
    page_events = [e for e in logs if e.get("event") == "pdf_page_extraction_failed"]
    assert len(page_events) == 1
    ev = page_events[0]
    assert ev["log_level"] == "warning"
    assert ev["doc_id"] == "handbook.pdf"
    assert ev["page"] == 2
    assert ev["error_type"] == "ValueError"
    assert "cannot parse table on this page" in ev["error"]
    assert ev["from_extractor"] == "pdfplumber"
    assert ev["to_extractor"] == "pypdf"
    assert ev["lost_capabilities"] == [CAP_TABLES]

    # Document WARN: the blast radius, which is no longer "everything".
    doc_events = _fallback_events(logs)
    assert len(doc_events) == 1
    summary = doc_events[0]
    assert summary["log_level"] == "warning"
    assert summary["pages_total"] == 3
    assert summary["pages_degraded"] == 1
    assert summary["degraded_pages"] == [2]


def test_several_bad_pages_are_each_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = _make_pdf([f"page {i}" for i in range(1, 7)])
    fail_pdfplumber_on(monkeypatch, {2, 5})

    with structlog.testing.capture_logs() as logs:
        ex = extract_pdf_text(pdf, doc_id="h.pdf")

    assert [p.page_no for p in ex.pages if p.degradations] == [2, 5]
    assert [p.page_no for p in ex.pages if not p.degradations] == [1, 3, 4, 6]
    assert len([e for e in logs if e.get("event") == "pdf_page_extraction_failed"]) == 2
    summary = _fallback_events(logs)[0]
    assert summary["pages_degraded"] == 2
    assert summary["degraded_pages"] == [2, 5]
    assert summary["pages_total"] == 6


def test_unopenable_pdf_falls_back_for_every_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """pdfplumber missing / the file unopenable is not a PAGE failure — no page is blamed,
    and every page is degraded because none of them could be read by the primary."""
    import pdfplumber

    def unopenable(*args: Any, **kwargs: Any) -> Any:
        raise OSError("not a pdf")

    monkeypatch.setattr(pdfplumber, "open", unopenable)
    pdf = _make_pdf(["only page", "second page"])

    with structlog.testing.capture_logs() as logs:
        ex = extract_pdf_text(pdf, doc_id="broken.pdf")

    assert [p.extractor for p in ex.pages] == ["pypdf", "pypdf"]
    assert all(p.degradations == (REASON_EXTRACTOR_FALLBACK,) for p in ex.pages)
    assert not [e for e in logs if e.get("event") == "pdf_page_extraction_failed"]  # no page to blame
    summary = _fallback_events(logs)[0]
    assert summary["error_type"] == "OSError"
    assert summary["pages_degraded"] == summary["pages_total"] == 2
    assert summary["degraded_pages"] == [1, 2]


# --- the vocabulary is an OPEN set --------------------------------------------------------


def test_provenance_tags_are_extractor_agnostic() -> None:
    """A future extractor with new capabilities and a new degradation reason needs no
    change to the tagging code — it just declares them."""
    ex = Extraction(
        pages=[
            PageRecord(
                page_no=1,
                text="x",
                extractor="some-future-extractor",
                capabilities=(CAP_TEXT, CAP_TABLES, "figures"),
                degradations=("model_unavailable", "page_limit_reached"),
            )
        ],
        modality=Modality.TEXT,
    )
    tags = provenance_tags(ex)
    assert tags[META_EXTRACTOR] == "some-future-extractor"
    assert tags[META_CAPABILITIES] == "figures,tables,text"  # sorted -> stable for querying
    assert tags[META_DEGRADED] == "true"
    assert tags[META_DEGRADATION] == "model_unavailable,page_limit_reached"


def test_every_extractor_declares_its_own_provenance() -> None:
    """No extractor may return text without saying who produced it and what it can do."""
    for ex in (
        ocr.extract_plain_text(b"hello"),
        ocr.extract_html_text(b"<p>hello</p>"),
    ):
        assert ex.extractor, "extractor id must not be empty"
        assert CAP_TEXT in ex.capabilities
        assert provenance_tags(ex)[META_DEGRADED] == "false"
