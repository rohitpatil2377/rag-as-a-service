"""Glossary / synonym query-expansion (pre-retrieval) — pure logic, no I/O.

A casual question ("who signs off when a product test fails?") and the SOP that answers it
("...the OOS result shall be reviewed and approved by Quality Assurance...") share almost no
surface vocabulary, so a passage search — dense embedder AND cross-encoder reranker alike —
can rank a confident-WRONG chunk above the right one. Bridging the plain wording to the
canonical SOP/GMP terminology BEFORE retrieval closes that gap: it was validated empirically
that appending the canonical terms to the query text flips the reranker's top chunk to the
correct document.

The expansion therefore has to reach ``query.text`` — the field BOTH the dense embedder and
the cross-encoder read — not a sparse-only side field (that is the chem-terms pattern, which
serves a different purpose: an InChIKey hash is noise to the reranker, whereas a canonical SOP
TERM is exactly what the reranker needs to see).

This is the deterministic twin of the chem canonicaliser: one small, readable, extensible
table (mirroring the style of :mod:`rag_core.notation`), pure (no I/O, no logging), and
FAIL-SAFE — ``expand_glossary`` returns the text unchanged (identity) whenever nothing matches,
and is idempotent (expanding an already-expanded query appends nothing new). Matching is a
case-insensitive substring test on the query, so a casual phrase anywhere in the question
triggers its canonical terms; acronyms expand bidirectionally (the long form appends the short,
the short appends the long).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache

# A term rule: the casual trigger phrases (matched case-insensitively as substrings) and the
# canonical SOP/GMP terms to append when any of them is present. Kept as plain tuples so the
# table reads as data and a JSON overlay (same shape) merges without a translation layer.
TermRule = tuple[tuple[str, ...], str]


@dataclass(frozen=True)
class Glossary:
    """A deterministic expansion table.

    ``acronyms`` maps a short form to its long form and is expanded BIDIRECTIONALLY (the short
    appends the long and vice-versa). ``terms`` maps casual trigger phrases to the canonical
    terms to append. Both are read-only; a file overlay merges OVER this (acronyms override by
    key, term rules append) — see :func:`load_glossary`.
    """

    acronyms: Mapping[str, str] = field(default_factory=dict)
    terms: Sequence[TermRule] = field(default_factory=tuple)


# --- the embedded default glossary -------------------------------------------------------------
# Short<->long acronym pairs, expanded in BOTH directions so a query naming either form retrieves
# a document written with the other.
_DEFAULT_ACRONYMS: dict[str, str] = {
    "OOS": "out-of-specification",
    "OOT": "out-of-trend",
    "CAPA": "corrective and preventive action",
    "QA": "quality assurance",
    "QC": "quality control",
    "GMP": "good manufacturing practice",
    "SOP": "standard operating procedure",
    "BMR": "batch manufacturing record",
    "CoA": "certificate of analysis",
}

# Casual trigger phrases -> canonical SOP/GMP terms. Triggers are lowercase (matching is
# case-insensitive) and matched as SUBSTRINGS, so "who signs off" fires the sign-off rule.
_DEFAULT_TERMS: tuple[TermRule, ...] = (
    (
        (
            "test fails",
            "test failed",
            "failing test",
            "test comes back failing",
            "failing result",
            "failed result",
            "out of spec",
            "out of limits",
            "below spec",
            "above spec",
            "didn't meet spec",
        ),
        "out-of-specification OOS result",
    ),
    (("out of trend", "trending out"), "out-of-trend OOT result"),
    (
        (
            "signs off",
            "sign off",
            "sign-off",
            "signed off",
            "who approves",
            "who approved",
            "gives approval",
            "gives the ok",
            "authorize",
            "authorise",
            "who authorizes",
            "who signs",
        ),
        "approve approval authorized reviewed by",
    ),
    (("who checks", "who reviews", "who verifies"), "reviewed approved by quality assurance QA"),
    (
        (
            "deviation",
            "deviated",
            "didn't follow the procedure",
            "not as per procedure",
            "departure from procedure",
        ),
        "deviation incident",
    ),
    (
        ("fix it", "corrective action", "stop it happening again", "prevent recurrence", "root cause fix"),
        "corrective and preventive action CAPA",
    ),
    (
        ("release the batch", "batch released", "ship the batch", "can we release", "disposition", "release for sale"),
        "batch disposition release",
    ),
    (("reject the batch", "scrap the batch", "fail the batch"), "batch rejection disposition"),
    (("complaint", "customer complained", "market complaint"), "market complaint handling"),
    (("recall", "pull the product", "withdraw the product"), "product recall withdrawal"),
    (("make a change", "change the procedure", "modify the process"), "change control"),
    (("clean the equipment", "cleaning", "residue"), "cleaning validation"),
    (
        ("expiry", "shelf life", "retest date", "how long is it good for"),
        "expiry date retest date shelf life",
    ),
    (("stability", "does it degrade"), "stability study"),
    (
        ("contaminated", "contamination", "cross contamination", "mix up", "mixup"),
        "contamination cross-contamination control",
    ),
    (
        (
            "fridge broke",
            "temperature went up",
            "cold chain",
            "temperature excursion",
            "left out of the fridge",
        ),
        "temperature excursion cold chain storage condition",
    ),
    (("training", "not trained", "who trains"), "training qualification record"),
    (("audit", "inspection", "inspector asked"), "audit inspection"),
    (("lab test", "quality check", "qc test", "testing the sample"), "quality control testing analysis"),
    (("equipment qualification", "qualify the equipment"), "equipment qualification IQ OQ PQ"),
)

_DEFAULT_GLOSSARY = Glossary(acronyms=_DEFAULT_ACRONYMS, terms=_DEFAULT_TERMS)


def _match(haystack_low: str, needle_low: str) -> int:
    """Index of a WORD-BOUNDARY match of ``needle_low`` in ``haystack_low`` (both lowercased), or
    -1. Word boundaries stop a trigger from matching inside a larger word — e.g. the trigger
    ``authorize`` must NOT match its own canonical output ``authorized`` (which would break replace
    idempotency), and ``QA`` must not match mid-word. Both inputs must already be lowercase.
    """
    m = re.search(rf"\b{re.escape(needle_low)}\b", haystack_low)
    return m.start() if m else -1


def expand_glossary(text: str, glossary: Glossary, mode: str = "append") -> str:
    """Rewrite ``text`` with canonical SOP/GMP terms for every glossary entry it matches.

    Matching is a case-insensitive SUBSTRING test on ``text``. Acronyms are ALWAYS additive and
    bidirectional (a matched short form appends its long form and vice-versa), in both modes.
    ``mode`` governs how matched casual TERM triggers are handled:

    * ``"append"`` (default, safe): the original text is preserved and returned FIRST, with the
      canonical terms space-joined after it — best recall, never loses the user's words.
    * ``"replace"``: each matched term rule's FIRST trigger occurrence is swapped IN PLACE for its
      canonical terms — disambiguates a lexical lookalike that would otherwise out-rank the right
      doc, at some recall cost when the casual word was itself the right signal.

    De-duplication (append mode) and the replace-then-not-re-match property keep the function
    idempotent — expanding an already-expanded query adds nothing. Returns ``text`` UNCHANGED when
    nothing matches. Pure: no I/O, no logging.
    """
    low = text.lower()
    # Tokens already present (lowercased) — the dedup/idempotency baseline for appended tokens.
    present = {tok.lower() for tok in text.split()}
    appended: list[str] = []

    def _append(canonical: str) -> None:
        for token in canonical.split():
            if token.lower() not in present:
                present.add(token.lower())
                appended.append(token)

    # Acronyms: additive + bidirectional, regardless of mode (word-boundary matched).
    for short, long in glossary.acronyms.items():
        if _match(low, short.lower()) != -1:
            _append(long)
        if _match(low, long.lower()) != -1:
            _append(short)

    if mode == "replace":
        # Swap each matched rule's first trigger occurrence for its canonical terms, in place.
        result = text
        for triggers, canonical in glossary.terms:
            for trigger in triggers:
                idx = _match(result.lower(), trigger.lower())
                if idx != -1:
                    result = result[:idx] + canonical + result[idx + len(trigger) :]
                    break  # one replacement per rule
        if appended:
            result = result + " " + " ".join(appended)
        return result if result != text else text

    # append mode (default)
    for triggers, canonical in glossary.terms:
        if any(_match(low, trigger.lower()) != -1 for trigger in triggers):
            _append(canonical)
    if not appended:
        return text
    return text + " " + " ".join(appended)


@cache
def load_glossary(path: str = "") -> Glossary:
    """Return the embedded default glossary (``path`` empty) or the default merged with a JSON
    overlay at ``path``. The overlay has the same shape — ``{"acronyms": {short: long},
    "terms": [{"triggers": [...], "canonical": "..."}]}`` — and merges OVER the default: acronyms
    override by key, term rules append. Stdlib ``json`` only (no new deps). Cached per path.
    """
    if not path:
        return _DEFAULT_GLOSSARY
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    acronyms = {**_DEFAULT_ACRONYMS, **raw.get("acronyms", {})}
    overlay_terms = tuple((tuple(rule["triggers"]), rule["canonical"]) for rule in raw.get("terms", []))
    return Glossary(acronyms=acronyms, terms=(*_DEFAULT_TERMS, *overlay_terms))
