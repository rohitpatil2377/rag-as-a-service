"""Structured field extraction (FR-B2) — the "intelligence" beyond raw text.

Pulls typed, filterable fields out of a document's extracted text with zero tenant effort:
business identifiers (GSTIN / PAN / IFSC), monetary amounts, and dates. Deterministic +
CPU-free (regex), India-aware; lands in the document's ``metadata`` so it becomes a D8
retrieval filter (e.g. find the invoice with a given GSTIN). Personal identifiers
(Aadhaar, personal contact) are deliberately left to the PII masker, not surfaced here.
"""

from __future__ import annotations

import re

# India business identifiers (public, not sensitive PII).
_GSTIN = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]\b")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_IFSC = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
# A CIF/customer id (e.g. CUST0098765) can coincidentally fit the bare IFSC shape, so prefer
# a keyword-anchored IFSC when the doc labels it, and fall back to the bare pattern otherwise.
_IFSC_CTX = re.compile(r"\bifsc\s*(?:code)?\s*[:\-]?\s*([A-Z]{4}0[A-Z0-9]{6})\b", re.IGNORECASE)
# SWIFT/BIC bank code — context-anchored (6 letters + 2 alnum [+ 3] would otherwise match
# ordinary 8-letter words like "COMPUTER", so we require the SWIFT/BIC keyword).
_SWIFT = re.compile(r"\b(?:swift|bic)\s*(?:code)?\s*[:\-]?\s*([A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)\b", re.IGNORECASE)
# Monetary amounts: ₹/Rs/INR followed by a number (with optional thousands separators).
_AMOUNT = re.compile(r"(?:₹|Rs\.?|INR)\s?([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE)
# Dates: dd/mm/yyyy, dd-mm-yyyy, dd Mon yyyy, yyyy-mm-dd.
_DATE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)


def _first(pattern: re.Pattern[str], text: str, group: int = 0) -> str | None:
    m = pattern.search(text)
    return m.group(group).strip() if m else None


def extract_fields(text: str) -> dict[str, str]:
    """Typed structured fields for a document, as string metadata (empty when none found).

    Single-valued keys (first match) so they filter cleanly under D8; ``field.amounts`` keeps
    all amounts for display. Keys are namespaced ``field.*`` so they never collide with B7 tags."""
    fields: dict[str, str] = {}
    if gstin := _first(_GSTIN, text):
        fields["field.gstin"] = gstin
    if pan := _first(_PAN, text):
        fields["field.pan"] = pan
    if ifsc := (_first(_IFSC_CTX, text, group=1) or _first(_IFSC, text)):
        fields["field.ifsc"] = ifsc.upper()
    if swift := _first(_SWIFT, text, group=1):
        fields["field.swift"] = swift.upper()
    if date := _first(_DATE, text):
        fields["field.date"] = date
    amounts = [m.group(1) for m in _AMOUNT.finditer(text)]
    if amounts:
        fields["field.amount"] = amounts[0]
        fields["field.amounts"] = ", ".join(dict.fromkeys(amounts))[:200]
    return fields
