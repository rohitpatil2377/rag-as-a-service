"""Phone masking must cover the formats people actually write, not only Indian mobiles.

Found live on a property form: "What is the phone number on the property form?" answered
"714-814-5829" in cleartext, grounded and cited. The detector matched a 10-digit Indian mobile
(`[6-9]\\d{4}[-\\s]?\\d{5}`) and a NANP number is grouped 3-3-4, so it could never match -- the
number was stored unmasked at ingest and every layer after that faithfully reported it.

The hard part is not matching more; it is not matching things that merely look like numbers. This
codebase already refuses to mask a bare long number without a checksum or a context keyword (an
invoice number is not an Aadhaar), and that rule holds here: a 10-digit run with no separators and
no "phone" nearby stays untouched.
"""

from __future__ import annotations

import pytest
from svc_doc_intelligence.pii import build_pii_masker


def _mask(text: str) -> str:
    r = build_pii_masker().mask(text)
    return str(r.text if hasattr(r, "text") else r)


@pytest.mark.parametrize(
    "text",
    [
        "PHONE: 714-814-5829",
        "Call 714-814-5829 for details",
        "Contact (714) 814-5829 today",
        "Tel +1 714-814-5829",
        "phone 714.814.5829",
    ],
)
def test_a_north_american_number_is_masked(text: str) -> None:
    assert "5829" not in _mask(text) or "REDACTED" in _mask(text), f"leaked: {_mask(text)!r}"
    assert "714-814-5829" not in _mask(text)
    assert "714.814.5829" not in _mask(text)


def test_an_indian_mobile_is_still_masked() -> None:
    """The format this detector was built for must not regress."""
    for text in ("+91-98200-10005", "9820010005", "98200 10005"):
        assert "10005" not in _mask(text), f"leaked: {_mask(text)!r}"


@pytest.mark.parametrize(
    "text",
    [
        "Invoice number 4180592371 is outstanding",
        "Order 1234567890 shipped on Tuesday",
        "Section 5.7.3 of the procedure applies",
        "The revision date is 2024-09-18",
        "Page 1234 of 5678",
    ],
)
def test_a_number_that_is_not_a_phone_is_left_alone(text: str) -> None:
    """The false-positive rule this codebase already holds to: a long number is not sensitive
    just for being long. Masking an invoice or a clause reference corrupts the document."""
    assert _mask(text) == text, f"over-masked: {_mask(text)!r}"


def test_a_grouped_international_number_is_masked() -> None:
    """4-3-4 grouping (UK and others), still separator-required."""
    out = _mask("Phone No.: 0161 496 0134")
    assert "0134" not in out, f"leaked: {out!r}"


def test_the_detector_set_has_exactly_one_phone_entry() -> None:
    """_DETECTORS_BY_LABEL is keyed by label, so a second _Detector("PHONE", ...) would silently
    REPLACE the first and quietly stop masking whatever the first one covered. That is how the
    Indian format regressed while the new one was being added."""
    from svc_doc_intelligence.pii import _DETECTORS

    assert sum(1 for d in _DETECTORS if d.label == "PHONE") == 1
