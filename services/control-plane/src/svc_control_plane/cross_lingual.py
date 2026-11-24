"""Indic cross-lingual QUERY expansion (FR-B* Indic) — additive, pure core.

Today a query retrieves as-is: a Hindi/Marathi question only matches Hindi/Marathi
chunks, so it misses the English documents that answer it (and vice-versa). This module
closes that gap on the QUERY side without touching the corpus.

The approach is "retrieve on the original AND on a translation, then union the evidence":
detect the query's script; if it differs from the corpus language, translate the query into
the corpus language and retrieve on BOTH variants. The two hit-sets are merged downstream,
so a cross-lingual answer surfaces whether the phrasing matched the Hindi or the English side.

Translation is an INJECTED async callback (``translate_fn``) — the LLM gateway — not an import.
That keeps this core PURE: script detection is Unicode-range arithmetic, and every function is
model-, DB-, and network-free, so the whole seam is unit-testable with a fake ``translate_fn``.
Retrieval must never fail because translation did: a ``translate_fn`` error logs and falls back
to the original query alone.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from rag_common import get_logger

_log = get_logger(__name__)

# Callback the parent must supply: (text, target_lang) -> translated text.
TranslateFn = Callable[[str, str], Awaitable[str]]

# Coarse script → inclusive Unicode codepoint ranges of its alphabetic block.
_SCRIPT_RANGES: dict[str, tuple[int, int]] = {
    "devanagari": (0x0900, 0x097F),  # Hindi, Marathi, …
    "bengali": (0x0980, 0x09FF),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
}


def _classify_char(ch: str) -> str | None:
    """Script label for one character, or None if it is not an alphabetic letter we count."""
    cp = ord(ch)
    for script, (lo, hi) in _SCRIPT_RANGES.items():
        if lo <= cp <= hi:
            return script
    if ch.isalpha():
        # Everything alphabetic outside the named Indic blocks is bucketed as "latin"
        # (ASCII/Latin) vs "other" (e.g. CJK, Arabic) by ASCII-ness.
        return "latin" if ch.isascii() else "other"
    return None


def detect_script(text: str) -> str:
    """Return a coarse script label for ``text`` from Unicode ranges — PURE.

    Labels: "devanagari", "tamil", "telugu", "bengali", "latin", else "other". The winner is
    the script of the MAJORITY of alphabetic characters (non-letters — digits, spaces, punctuation
    — are ignored). Empty/whitespace-only (no letters) => "latin".
    """
    counts: dict[str, int] = {}
    for ch in text:
        label = _classify_char(ch)
        if label is not None:
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "latin"
    # Majority wins; ties break deterministically by label for stable behaviour.
    return max(counts, key=lambda label: (counts[label], label))


def needs_translation(text: str, *, target_script: str = "latin") -> bool:
    """True when ``text``'s dominant script differs from ``target_script`` — PURE.

    A difference means cross-lingual retrieval would help: the query is written in one script
    while the corpus is indexed in another, so retrieving on the original alone under-recalls.
    """
    return detect_script(text) != target_script


async def expand_query_cross_lingual(
    text: str,
    *,
    target_lang: str,
    translate_fn: TranslateFn,
    enabled: bool,
    target_script: str = "latin",
) -> list[str]:
    """Query variants to retrieve on: the original plus, when cross-lingual, a translation.

    Returns ``[text]`` when disabled or when no translation is needed (query already in the
    corpus script). Otherwise calls ``translate_fn(text, target_lang)`` and returns
    ``[text, translated]``, DEDUPED — a blank or identical translation collapses to ``[text]``.

    Retrieval must never fail because translation did: any ``translate_fn`` exception is logged
    and swallowed, falling back to ``[text]``. ``translate_fn`` is injected (the LLM gateway),
    so this stays pure and testable with no model.
    """
    if not enabled or not needs_translation(text, target_script=target_script):
        return [text]

    try:
        translated = await translate_fn(text, target_lang)
    except Exception:
        # Log-and-fallback: a translation failure degrades recall, it does not break retrieval.
        _log.warning("cross_lingual.translate_failed", exc_info=True)
        return [text]

    variants = [text]
    candidate = (translated or "").strip()
    if candidate and candidate != text:
        variants.append(translated)
    return variants
