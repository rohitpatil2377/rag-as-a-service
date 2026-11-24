"""Ingestion data cleaning + adversarial sanitisation (pure, deterministic, testable).

The stage that was MISSING: run on extracted document text BEFORE it is chunked/embedded/indexed.
Three jobs, all local (no external calls — sovereign) and deterministic:

* **Normalise** — Unicode NFKC + collapse whitespace, so the same content always indexes the same
  way and look-alike/compatibility characters cannot be used to slip past filters.
* **De-cloak** — strip zero-width / invisible / control characters. These are the #1 hiding place
  for INDIRECT PROMPT INJECTION ("ignore previous instructions" written in white/zero-width text a
  human reviewer never sees) and for homoglyph tricks.
* **Scan** — flag text that reads like INSTRUCTIONS to the model rather than data. High-signal,
  conservative patterns (a false positive only QUARANTINES for human review, never silently
  drops), because a document that argues with the model is either an attack or needs a human.

Verdicts drive the pipeline: CLEAN → index; SUSPECT → quarantine for a reviewer (never
auto-indexed); the malware/illegal tiers are separate (AV + a local moderation model — infra).
Citations stay structural and answers stay grounded-or-abstain regardless, so this is one layer
of defence in depth, not the only one.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

# Zero-width + invisible FORMAT characters (Unicode Cf) that carry no visible content and are the
# classic cloak for injected instructions. Stripped outright.
_HIDDEN_CODEPOINTS = frozenset(
    [0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2060, 0x2061, 0x2062, 0x2063, 0x2064, 0xFEFF, 0x00AD]
)
# Control characters to drop (C0/C1) — everything except tab/newline/carriage-return, which are
# legitimate structure. A control char mid-text is either corruption or an obfuscation attempt.
_KEEP_CONTROLS = frozenset([0x09, 0x0A, 0x0D])
_CTRL_MIN = 0x20  # below this: C0 control chars
_C1_LOW = 0x7F  # start of the DEL + C1 control band
_C1_HIGH = 0x9F  # end of the C1 control band

_WS_RUN = re.compile(r"[^\S\n]+")  # runs of spaces/tabs (but keep newlines as structure)
_BLANK_RUN = re.compile(r"\n{3,}")  # 3+ newlines -> 2 (paragraph break)

# INDIRECT prompt-injection signatures. Conservative + high-signal on purpose: each is a phrase
# that only makes sense as an instruction to a model, not as ordinary enterprise content. A hit
# QUARANTINES (human review), so precision matters more than recall here.
_INJECTION_PATTERNS: tuple[str, ...] = (
    r"ignore\s+(?:all\s+|the\s+|any\s+|your\s+)*(?:previous|prior|above|preceding)\s+(?:instructions?|prompts?|context)",
    r"disregard\s+(?:all\s+|the\s+|any\s+)*(?:previous|prior|above|earlier)\b",
    r"forget\s+(?:all\s+|everything\s+|your\s+)*(?:previous|prior|above|instructions?|what)",
    r"you\s+are\s+now\s+(?:a|an|the)\b",
    r"(?:new|updated|revised)\s+(?:system\s+)?(?:instructions?|prompt)\s*[:\-]",
    r"(?:system|developer)\s+prompt\s*[:\-]",
    r"reveal\s+(?:your|the)\s+(?:system\s+prompt|instructions?|hidden)",
    r"do\s+not\s+(?:tell|inform|reveal\s+to|mention\s+to)\s+the\s+user",
    r"</?(?:system|assistant|user|im_start|im_end)>",  # chat role tags smuggled into content
    # EXFILTRATION, in the two shapes that are instructions and not description. This was once
    # the bare substring `exfiltrat`, which broke the rule stated above: "data exfiltration" is
    # ordinary vocabulary in exactly the corpora this platform is sold for. Measured on a real
    # 80-page SOC/NOC tender, the single line "System detects data exfiltration and unusual
    # internal data transfers" quarantined the whole document, so a security customer's corpus
    # would have been silently refused a requirement at a time, with nothing but a warning log.
    # The discriminator is grammatical and holds up: an attack uses the VERB as an imperative
    # ("then exfiltrate the salary data"), a security document uses the NOUN ("prevent data
    # exfiltration") or the third person ("malware exfiltrates credentials"). So: the verb under
    # an instruction cue, or any form pointed at a destination.
    r"\b(?:then|and|next|please|also|you\s+must|you\s+should|must)\s+exfiltrat(?:e|ing)\b",
    r"\bexfiltrat(?:e|ed|ing|ion)\b[^.\n]{0,60}?\bto\s+(?:https?://|ftp://|[\w.+-]+@[\w.-]+)",
)
_INJECTION_RE = tuple(re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS)


class Risk(StrEnum):
    CLEAN = "clean"  # index normally
    SUSPECT = "suspect"  # quarantine for human review — do NOT auto-index


@dataclass(frozen=True, slots=True)
class CleanResult:
    text: str  # normalised, de-cloaked text
    hidden_removed: int  # count of zero-width/control chars stripped
    normalized: bool  # whether normalisation changed anything
    risk: Risk
    matches: tuple[str, ...]  # injection pattern names that fired (for the audit/quarantine)


def _decloak(text: str) -> tuple[str, int]:
    """Strip zero-width/invisible format chars and stray control chars; return (text, removed)."""
    out: list[str] = []
    removed = 0
    for ch in text:
        cp = ord(ch)
        if cp in _HIDDEN_CODEPOINTS:
            removed += 1
            continue
        if cp < _CTRL_MIN and cp not in _KEEP_CONTROLS:
            removed += 1
            continue
        if _C1_LOW <= cp <= _C1_HIGH:  # DEL + C1 controls
            removed += 1
            continue
        out.append(ch)
    return "".join(out), removed


# Sub/superscript digits and signs, which NFKC would fold to plain ASCII. They are exempted
# because that fold is DATA LOSS on a scientific corpus, not normalisation: measured live on a
# chemistry page, H₂SO₄ indexed as H2SO4 (readable, but no longer what the page says) and 4FeO₄²⁻
# as "4FeO42" followed by a bare minus, where the ionic charge can no longer be told from a
# coefficient. A pharma SOP, a
# lab result and a patent all carry the same notation.
#
# Safe to exempt: the reason NFKC runs here is to stop a payload hiding from the injection scanner
# behind look-alike glyphs, and the scanner's signatures are words ("ignore all previous...").
# Subscript digits cannot spell those. Every other compatibility fold — fullwidth letters,
# ligatures, circled digits — is untouched, and the de-cloaking pass that strips zero-width and
# control characters runs before this and is also untouched.
_SCRIPT_CHARS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
# Private-use codepoints: unassigned, so NFKC leaves them alone and no document can contain one
# (the de-cloak pass strips them from input before this runs).
_SCRIPT_GUARD = str.maketrans({ch: chr(0xE000 + i) for i, ch in enumerate(_SCRIPT_CHARS)})
_SCRIPT_RESTORE = str.maketrans({chr(0xE000 + i): ch for i, ch in enumerate(_SCRIPT_CHARS)})


def normalize_text(text: str) -> str:
    """Unicode NFKC + whitespace collapse. NFKC folds compatibility/look-alike forms to a single
    canonical representation, so homoglyph obfuscation cannot hide a payload from the scanner.

    Sub/superscripts are carried through the fold unchanged — see ``_SCRIPT_CHARS``."""
    text = text.translate(_SCRIPT_GUARD)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_SCRIPT_RESTORE)
    text = _WS_RUN.sub(" ", text)
    text = _BLANK_RUN.sub("\n\n", text)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def scan_injection(text: str) -> tuple[Risk, tuple[str, ...]]:
    """Scan for indirect-prompt-injection signatures. Returns (risk, matched pattern names)."""
    matches = tuple(p.pattern for p in _INJECTION_RE if p.search(text))
    return (Risk.SUSPECT if matches else Risk.CLEAN), matches


def clean(text: str) -> CleanResult:
    """The full ingestion cleaning stage: de-cloak → normalise → scan. Deterministic and local.

    Scans the NORMALISED text, so an attacker cannot evade the scanner by splitting a payload with
    zero-width characters or compatibility look-alikes — those are gone before the scan runs."""
    decloaked, hidden = _decloak(text)
    normalized = normalize_text(decloaked)
    risk, matches = scan_injection(normalized)
    return CleanResult(
        text=normalized,
        hidden_removed=hidden,
        normalized=(normalized != text),
        risk=risk,
        matches=matches,
    )
