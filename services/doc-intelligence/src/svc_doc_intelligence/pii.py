"""PII detection & masking / redaction (FR-B6/G2).

Masking runs at extraction so ONLY de-identified text is ever chunked, embedded, or
indexed — the raw bytes stay in the restricted object store. Detection is checksum-aware
where the identifier carries one (Aadhaar Verhoeff, credit-card Luhn), which cuts the
false positives a naive regex makes on invoice/order numbers.

Three redaction strategies (operator-configurable, ``RAG_PII_STRATEGY``):
* ``redact`` — replace with ``[REDACTED:<TYPE>]`` (default; maximal privacy);
* ``mask``   — keep the last few characters, e.g. ``XXXXXXXX1234`` (human-recognisable);
* ``hash``   — deterministic salted token ``[<TYPE>:ab12cd]`` — the SAME value always maps
  to the SAME token, so an analyst can see "the same person recurs" WITHOUT seeing who.

The :class:`PIIMasker` Protocol is the stable seam: ``regex`` (dependency-free, default)
or ``presidio`` (NER + Indic context, opt-in) plug in via :func:`build_pii_masker` with no
call-site change — the same default-safe / production-plugs-in pattern used elsewhere.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# --- Verhoeff checksum (Aadhaar) ---------------------------------------------
# Dihedral-group tables; validates the 12th digit so a bare 12-digit number (an invoice
# total, an order id) is NOT mistaken for an Aadhaar. Standard, deterministic.
_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 9, 7, 4, 2, 1, 6, 3),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


_AADHAAR_LEN = 12
_CARD_MIN_LEN, _CARD_MAX_LEN = 13, 19


def _verhoeff_ok(number: str) -> bool:
    digits = [int(d) for d in reversed(re.sub(r"\D", "", number))]
    if len(digits) != _AADHAAR_LEN:
        return False
    check = 0
    for i, d in enumerate(digits):
        check = _VERHOEFF_D[check][_VERHOEFF_P[i % 8][d]]
    return check == 0


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if not _CARD_MIN_LEN <= len(digits) <= _CARD_MAX_LEN:
        return False
    total, parity = 0, len(digits) % 2
    for i, d in enumerate(digits):
        doubled = d * 2
        # Luhn: a doubled digit > 9 has its digits summed, i.e. subtract 9.
        total += (doubled - 9 if doubled > 9 else doubled) if i % 2 == parity else d  # noqa: PLR2004
    return total % 10 == 0


# --- detectors ---------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _Detector:
    label: str
    pattern: re.Pattern[str]
    validator: Callable[[str], bool] | None = None  # reject structural matches that fail a checksum
    keep_last: int = 4  # how many trailing chars the 'mask' strategy preserves
    group: int = 0  # which capture group is the sensitive value (0 = whole match). Context-
    # anchored detectors put the keyword ("A/c No:") outside the group so only the number masks.
    # Rejects a match from what SURROUNDS it, where the value alone cannot decide. Takes the full
    # text and the match span; returns True to mask. Distinct from `validator`, which sees only
    # the matched string and answers a question about the value itself (a checksum).
    context_validator: Callable[[str, int, int], bool] | None = None


#: Words that introduce a CONTRACTUAL IDENTIFIER rather than a network address. A four-part
#: clause number is syntactically a valid IPv4 address and no amount of looking at the digits can
#: tell them apart -- `1.3.9.1` is both a legal address and a real clause in the corpus that
#: exposed this. Only the surrounding words can decide.
#:
#: FOUND IN PRODUCTION DATA, not constructed. The AEGCL corrigendum's "BID DATA SHEET (BDS), ITB
#: 1.3.9.1 item 5" was stored as "ITB [REDACTED:IP] item 5", so the clause identity was destroyed
#: at ingest and no downstream stage could recover it: retrieval could not match it, the map could
#: not extract it and the answer could not cite it. Four-part numbering is ordinary in tender and
#: standards documents, so this silently damages exactly the corpora the platform is for.
_CLAUSE_CUES = frozenset(
    {
        "itb",
        "bds",
        "gcc",
        "scc",
        "clause",
        "sub-clause",
        "subclause",
        "section",
        "para",
        "paragraph",
        "article",
        "appendix",
        "annexure",
        "annex",
        "schedule",
        "table",
        "sl",
        "s.no",
        "rule",
        "regulation",
    }
)

#: A SEPARATE CATEGORY, named separately because its justification is different. A version string
#: is not a contractual clause and does not belong in the list above -- an earlier draft of this
#: fix put `version` there unannounced, which is how an allowlist stops meaning what its name says.
#:
#: It is here because `2.0.1.3` is the same ambiguity with the same cost: a redacted version
#: destroys which release a requirement refers to, and the corpus already cites
#: "CERT-In Technical Guidelines ... Version 2.0". Deliberately NOT included: a bare `v`, which is
#: one letter and would suppress far more than it should for no case anyone has shown.
_VERSION_CUES = frozenset({"version", "ver", "rev", "revision", "release"})

#: Both groups, dots stripped so "Sl." and "sl" are one entry. Kept as one frozenset for the
#: lookup while the two above stay separately readable and separately tested.
_DOTTED_QUAD_CUES = frozenset(c.strip(".") for c in _CLAUSE_CUES | _VERSION_CUES)

#: How far back to look. One word, because the cue that governs such a number sits immediately
#: before it ("ITB 1.3.9.1", "clause 1.3.9.1", "Version 2.0.1.3"). A wider window would start
#: suppressing genuine addresses that merely share a sentence with the word "section".
_CLAUSE_CUE_LOOKBEHIND = 24


def _ip_not_a_clause(text: str, start: int, end: int) -> bool:
    """True when a dotted quad should be treated as an IP ADDRESS and masked.

    Suppressed for two separately-justified cue groups: :data:`_CLAUSE_CUES` (contractual
    identifiers) and :data:`_VERSION_CUES` (release identifiers). They are kept apart because
    "this is a clause number" and "this is a version" are different claims, and a single list
    named for one of them quietly accumulating the other is how an allowlist stops being
    reviewable.

    REDACTION REMAINS THE DEFAULT. A bare `1.3.9.1` with nothing around it is still masked,
    because failing to redact a real address is the worse error and the digits cannot settle it.
    Masking is suppressed only when an explicit contractual cue immediately precedes the number,
    which is a shape no network address has: nobody writes "clause 192.168.1.1".
    """
    before = text[max(0, start - _CLAUSE_CUE_LOOKBEHIND) : start]
    # The last word before the number, ignoring punctuation and the separators a clause
    # reference uses ("ITB 1.3.9.1", "Sl no: D.3", "section-1.3.9.1").
    words = re.findall(r"[A-Za-z.]+", before.lower())
    return not (words and words[-1].strip(".") in _DOTTED_QUAD_CUES)


# India-context identifiers + universal contact PII. Ordered: higher-signal first so a value
# is labelled by its most specific type. Business identifiers (GSTIN/IFSC/SWIFT) are handled by
# field extraction, not masked here (see fields.py).
_DETECTORS: tuple[_Detector, ...] = (
    _Detector("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), keep_last=0),
    _Detector("AADHAAR", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), _verhoeff_ok),
    _Detector("PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")),
    _Detector("PASSPORT", re.compile(r"\b[A-PR-WYa-pr-wy][0-9]{7}\b")),
    _Detector("VOTER_ID", re.compile(r"\b[A-Z]{3}[0-9]{7}\b")),
    _Detector("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b"), _luhn_ok),
    # 10-digit Indian mobile, tolerating one separator inside the number (e.g. 98200-10005,
    # 98200 10005, +91-98200-10005) as well as the contiguous form — grouped numbers leaked
    # in cleartext when only the contiguous pattern was matched (QA ISS-012).
    # Phone numbers, as people actually write them. ONE detector per label by construction --
    # _DETECTORS_BY_LABEL is keyed by label, so a second _Detector("PHONE", ...) silently replaces
    # the first and the formats it covered stop being masked. Hence an alternation.
    #
    # Found live on a property form: "PHONE: 714-814-5829" was stored in cleartext and answered
    # verbatim, grounded and cited, because only the Indian-mobile shape was ever matched and a
    # NANP number is grouped 3-3-4.
    #
    # A SEPARATOR is required in the international branch, deliberately. A bare ten-digit run is
    # an invoice or an order number far more often than a phone, and this module's standing rule
    # is that a long number is not sensitive merely for being long -- the same reason Aadhaar
    # needs Verhoeff and a bank account needs its keyword.
    _Detector(
        "PHONE",
        re.compile(
            # India: 10 digits starting 6-9, optional +91, one optional separator inside.
            r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{4}[-\s]?\d{5}(?!\d)"
            # Separated international grouping: 714-814-5829, (714) 814-5829, +1 714.814.5829,
            # 0161 496 0134. Three groups, separators mandatory, optional country code.
            r"|(?<![\d-])(?:\+\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]\d{3,4}[-.\s]\d{4}(?![\d-])"
        ),
    ),
    _Detector("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), keep_last=0, context_validator=_ip_not_a_clause),
    # --- Banking identifiers (context-anchored: the account/CVV/MICR/CIF keyword must be
    # present, so a bare long number — a policy amount, a section ref — is NOT masked). ---
    _Detector(
        "BANK_ACCOUNT",
        re.compile(r"\b(?:a/c|acct|account)\.?\s*(?:no\.?|number|num|#)?\s*[:\-]?\s*(\d{9,18})\b", re.IGNORECASE),
        group=1,
    ),
    _Detector(
        "MICR",
        re.compile(r"\bmicr\b\s*(?:code|no\.?)?\s*[:\-]?\s*(\d{9})\b", re.IGNORECASE),
        keep_last=0,
        group=1,
    ),
    _Detector(
        "CVV",
        re.compile(r"\b(?:cvv2?|cvc)\b\.?\s*(?:no\.?)?\s*[:\-]?\s*(\d{3,4})\b", re.IGNORECASE),
        keep_last=0,
        group=1,
    ),
    _Detector(
        "CUSTOMER_ID",
        re.compile(
            r"\b(?:cif(?:\s*(?:no\.?|id))?|customer\s*id|cust\.?\s*id)\s*[:\-]?\s*([A-Za-z0-9]{6,20})\b", re.IGNORECASE
        ),
        group=1,
    ),
)
_DETECTORS_BY_LABEL = {d.label: d for d in _DETECTORS}
DEFAULT_ENTITIES: tuple[str, ...] = tuple(d.label for d in _DETECTORS)


@dataclass(slots=True)
class MaskResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)
    # WHERE each redaction landed in the ORIGINAL text, ascending and non-overlapping. A
    # caller that also holds offsets into that text (a page map, a citation span) needs these:
    # a replacement is almost never the same length as the value it replaces, so applying the
    # rewritten string alone leaves every later offset wrong. `text` is built from these, so it
    # cannot describe a different set of redactions.
    spans: list[tuple[int, int, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def apply_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    """Rewrite ``text`` at the given ascending, non-overlapping spans."""
    parts: list[str] = []
    at = 0
    for start, end, replacement in spans:
        parts.append(text[at:start])
        parts.append(replacement)
        at = end
    parts.append(text[at:])
    return "".join(parts)


@runtime_checkable
class PIIMasker(Protocol):
    def mask(self, text: str) -> MaskResult: ...


def _partial(raw: str, keep_last: int) -> str:
    """Mask everything but the last ``keep_last`` alphanumerics, preserving separators."""
    if keep_last <= 0:
        return "".join("X" if ch.isalnum() else ch for ch in raw)
    alnum = [i for i, ch in enumerate(raw) if ch.isalnum()]
    reveal = set(alnum[-keep_last:])
    return "".join(ch if i in reveal or not ch.isalnum() else "X" for i, ch in enumerate(raw))


class RegexPIIMasker:
    """Dependency-free checksum-aware masker. Configurable entity set + redaction strategy."""

    def __init__(
        self,
        *,
        entities: tuple[str, ...] | None = None,
        strategy: str = "redact",
        hash_salt: str = "",
    ) -> None:
        chosen = entities or DEFAULT_ENTITIES
        self._detectors = tuple(_DETECTORS_BY_LABEL[e] for e in chosen if e in _DETECTORS_BY_LABEL)
        self._strategy = strategy
        self._salt = hash_salt.encode() or b"rag-pii"

    def _replacement(self, det: _Detector, raw: str) -> str:
        if self._strategy == "mask":
            return _partial(raw, det.keep_last)
        if self._strategy == "hash":
            token = hmac.new(self._salt, raw.encode(), hashlib.sha256).hexdigest()[:6]
            return f"[{det.label}:{token}]"
        return f"[REDACTED:{det.label}]"

    def mask(self, text: str) -> MaskResult:
        """Find every sensitive value and report WHERE it is, not just the rewritten string.

        Spans are what let a caller holding a page map redact without invalidating it — see
        ``RenderedText.apply_replacements``. The masked text is then built FROM the spans, so
        the two can never describe different redactions."""
        counts: dict[str, int] = {}
        spans: list[tuple[int, int, str]] = []
        for det in self._detectors:
            hits = 0
            for m in det.pattern.finditer(text):
                raw = m.group(det.group)  # the sensitive value (group 0 = whole match)
                if det.validator and not det.validator(raw):
                    continue  # structural match failed its checksum → leave untouched
                start, end = m.span(det.group)
                if det.context_validator and not det.context_validator(text, start, end):
                    continue  # the surroundings say this is not the thing the pattern looks like
                # A later detector may match INSIDE a region an earlier one already claimed
                # (a phone number inside an address line). First claim wins, so the spans stay
                # non-overlapping — which is what apply_replacements requires.
                if any(s < end and start < e for s, e, _ in spans):
                    continue
                hits += 1
                spans.append((start, end, self._replacement(det, raw)))
            if hits:
                counts[det.label] = hits
        spans.sort(key=lambda s: s[0])
        return MaskResult(text=apply_spans(text, spans), counts=counts, spans=spans)


def build_pii_masker(
    backend: str = "regex",
    *,
    entities: tuple[str, ...] | None = None,
    strategy: str = "redact",
    hash_salt: str = "",
) -> PIIMasker:
    """Select a PII masker (``regex`` default, ``presidio`` opt-in) — the additive seam."""
    if backend == "presidio":
        from svc_doc_intelligence.pii_presidio import PresidioPIIMasker  # noqa: PLC0415 — optional dep

        return PresidioPIIMasker(entities=entities, strategy=strategy, hash_salt=hash_salt)
    return RegexPIIMasker(entities=entities, strategy=strategy, hash_salt=hash_salt)


def parse_entities(csv: str) -> tuple[str, ...] | None:
    """Parse ``RAG_PII_ENTITIES`` (comma list) → validated tuple, or None for the default set."""
    picked = tuple(e.strip().upper() for e in csv.split(",") if e.strip())
    valid = tuple(e for e in picked if e in _DETECTORS_BY_LABEL)
    return valid or None
