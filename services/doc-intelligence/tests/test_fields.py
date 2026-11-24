"""Unit tests for FR-B2 structured field extraction."""

from __future__ import annotations

from svc_doc_intelligence.fields import _SWIFT, extract_fields


def test_swift_extracted_with_context() -> None:
    f = extract_fields("Wire to SWIFT: HDFCINBBXXX at the branch.")
    assert f["field.swift"] == "HDFCINBBXXX"


def test_swift_not_matched_on_plain_words() -> None:
    # "COMPUTER" fits the raw SWIFT shape (6 letters + 2 alnum) but has no SWIFT/BIC keyword.
    assert _SWIFT.search("The COMPUTER SYSTEMS policy") is None


def test_ifsc_prefers_keyword_anchored_over_cif_lookalike() -> None:
    # A CIF like CUST0098765 fits the bare IFSC shape; the labelled IFSC must win.
    f = extract_fields("CIF No: CUST0098765 with IFSC HDFC0000340 for the branch.")
    assert f["field.ifsc"] == "HDFC0000340"


def test_extracts_india_business_ids_dates_amounts() -> None:
    text = (
        "Invoice dated 14/03/2026. Vendor GSTIN 29ABCDE1234F1Z5, PAN ABCDE1234F, "
        "bank IFSC HDFC0001234. Total ₹1,25,000.50 due; advance Rs 5,000."
    )
    f = extract_fields(text)
    assert f["field.gstin"] == "29ABCDE1234F1Z5"
    assert f["field.pan"] == "ABCDE1234F"
    assert f["field.ifsc"] == "HDFC0001234"
    assert f["field.date"] == "14/03/2026"
    assert f["field.amount"] == "1,25,000.50"
    assert "5,000" in f["field.amounts"]


def test_empty_when_nothing_present() -> None:
    assert extract_fields("just some ordinary prose with no identifiers") == {}
