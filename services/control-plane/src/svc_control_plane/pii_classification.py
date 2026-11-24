"""PII → minimum data-classification (banking pack).

A fail-closed net for "everyone can ingest": the ingest masker already detects structured,
checksum-validated identifiers (Aadhaar via Verhoeff, card via Luhn, context-anchored bank IDs).
This maps the set it found to the LOWEST classification the document may safely carry, so a KYC
form full of Aadhaar numbers can't land as Public even if the uploader marked it so.

Pure + testable; the pipeline decides what to DO with the floor (raise vs block) —
see ``pipeline`` ingest path. Detection is high-precision but structured-ID only (regex+checksum),
so this is a floor for identifiers, NOT a full DLP — manual classification + the clearance cap
remain the backstop for unstructured sensitivity.
"""

from __future__ import annotations

from collections.abc import Mapping

# Classification tiers: 0 Public · 1 Internal · 2 Confidential · 3 Restricted.
# Each detected PII type implies a MINIMUM tier. Only GENUINELY-SENSITIVE structured identifiers
# raise the floor: financial secrets sit highest (raw card / CVV / account numbers), government
# identity documents Confidential. A type not listed implies nothing (0).
#
# DELIBERATELY EXCLUDED — contact PII (EMAIL / PHONE / IP): it appears in almost every business
# document, so raising the floor on it would block/lock ordinary uploads for any ordinary-clearance
# user — while adding little protection, since the masker already REDACTS these values from the
# stored text regardless of classification. So contact info stays masked but never gates ingestion.
# Kept in code (not per-tenant config) so the floor is a consistent platform guarantee; an operator
# who needs a different mapping changes it here deliberately.
_PII_FLOOR: dict[str, int] = {
    "CREDIT_CARD": 3,
    "CVV": 3,
    "BANK_ACCOUNT": 3,
    "AADHAAR": 2,
    "PAN": 2,
    "PASSPORT": 2,
    "VOTER_ID": 2,
    "MICR": 2,
    "CUSTOMER_ID": 2,
}


def pii_classification_floor(counts: Mapping[str, int]) -> int:
    """The minimum data classification the detected PII implies (0 if none) — the max floor over
    every entity type that was actually found (count > 0)."""
    return max((_PII_FLOOR.get(label, 0) for label, n in counts.items() if n > 0), default=0)


def pii_triggers(counts: Mapping[str, int], floor: int) -> list[str]:
    """The detected PII types that hold the floor at ``floor`` — for the audit event and the
    user-facing block message (e.g. ['AADHAAR', 'PAN'])."""
    return sorted(label for label, n in counts.items() if n > 0 and _PII_FLOOR.get(label, 0) >= floor)
