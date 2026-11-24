"""Presidio-backed PII masker (opt-in, ``RAG_PII_MASKING_BACKEND=presidio``).

Adds Microsoft Presidio's ML/NER recognisers — PERSON, LOCATION, ORGANIZATION, and other
free-text entities a regex can't catch — ON TOP OF the checksum-aware structured-ID layer
(:class:`RegexPIIMasker`), then applies the SAME operator-chosen redaction strategy so the
output is consistent whichever backend is active.

Presidio (``presidio-analyzer``/``presidio-anonymizer`` + a spaCy model) is a heavy,
optional dependency and is import-guarded — the platform runs fully on the regex backend
with zero extra install. Install with the ``pii`` extra to enable it. This is the
production upgrade path for the FR-B6 seam, with no call-site change.
"""

from __future__ import annotations

from svc_doc_intelligence.pii import MaskResult, RegexPIIMasker, apply_spans

# Presidio's default entity labels we surface (free-text entities regex can't find).
_NER_ENTITIES = ("PERSON", "LOCATION", "NRP", "ORGANIZATION")


class PresidioPIIMasker:
    """Regex structured-ID layer + Presidio NER, unified under one redaction strategy."""

    def __init__(
        self,
        *,
        entities: tuple[str, ...] | None = None,
        strategy: str = "redact",
        hash_salt: str = "",
    ) -> None:
        from presidio_analyzer import AnalyzerEngine  # noqa: PLC0415 — optional heavy dep

        self._regex = RegexPIIMasker(entities=entities, strategy=strategy, hash_salt=hash_salt)
        self._analyzer = AnalyzerEngine()

    def mask(self, text: str) -> MaskResult:
        # 1) structured IDs (checksum-validated) via the shared regex layer.
        base = self._regex.mask(text)
        # 2) free-text named entities via Presidio. A person/location name has no stable
        #    canonical form to partial-mask or tokenise, so it is always fully redacted.
        #
        # Analyze the ORIGINAL text, not the regex-masked one: both span sets must be offsets
        # into the same string or they cannot be merged, and a caller holding a page map needs
        # them against the text that map describes.
        counts = dict(base.counts)
        spans = list(base.spans)
        for res in self._analyzer.analyze(text=text, entities=list(_NER_ENTITIES), language="en"):
            # The regex layer claimed first — an Aadhaar inside a detected address stays an
            # Aadhaar. Overlaps would break apply_replacements' ascending, disjoint contract.
            if any(s < res.end and res.start < e for s, e, _ in spans):
                continue
            spans.append((res.start, res.end, f"[REDACTED:{res.entity_type}]"))
            counts[res.entity_type] = counts.get(res.entity_type, 0) + 1
        spans.sort(key=lambda s: s[0])
        return MaskResult(text=apply_spans(text, spans), counts=counts, spans=spans)
