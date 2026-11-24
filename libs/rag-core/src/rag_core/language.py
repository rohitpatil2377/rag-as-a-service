"""Language detection + translation seam (FR — Indic multilingual, pure logic).

Detection is **script-based** (Unicode blocks) — deterministic, dependency-free, and exactly
right for Indic scripts where the writing system identifies the language family. Detected
language lands in document metadata (a D8 filter: "only Odia documents") and lets the query
path translate/route cross-lingually. Multilingual *embeddings/generation* are a model swap
(e.g. multilingual-e5 / AI4Bharat via the TEI/vLLM config) — this module makes the platform
language-aware so those models slot in behind the existing seams.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# Unicode block start (inclusive) -> ISO-639-1/3 language code for the dominant Indic scripts.
_SCRIPTS: tuple[tuple[int, int, str], ...] = (
    (0x0900, 0x097F, "hi"),  # Devanagari (Hindi/Marathi/…)
    (0x0980, 0x09FF, "bn"),  # Bengali/Assamese
    (0x0A00, 0x0A7F, "pa"),  # Gurmukhi (Punjabi)
    (0x0A80, 0x0AFF, "gu"),  # Gujarati
    (0x0B00, 0x0B7F, "or"),  # Odia
    (0x0B80, 0x0BFF, "ta"),  # Tamil
    (0x0C00, 0x0C7F, "te"),  # Telugu
    (0x0C80, 0x0CFF, "kn"),  # Kannada
    (0x0D00, 0x0D7F, "ml"),  # Malayalam
)

LANG_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "pa": "Punjabi",
    "gu": "Gujarati",
    "or": "Odia",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "und": "Unknown",
}


def _script_lang(ch: str) -> str | None:
    cp = ord(ch)
    for lo, hi, code in _SCRIPTS:
        if lo <= cp <= hi:
            return code
    if ch.isascii() and ch.isalpha():
        return "en"
    return None


def detect_language(text: str, *, sample: int = 4000) -> str:
    """The dominant language of ``text`` by script (``en`` for Latin, ``und`` if no letters).
    Samples the first ``sample`` chars — enough to classify, cheap on large documents."""
    counts: dict[str, int] = {}
    for ch in text[:sample]:
        lang = _script_lang(ch)
        if lang:
            counts[lang] = counts.get(lang, 0) + 1
    if not counts:
        return "und"
    # Prefer any Indic script over Latin when both appear (mixed docs are usually Indic content
    # with English loan-words / headers).
    non_en = {k: v for k, v in counts.items() if k != "en"}
    pool = non_en or counts
    return max(pool, key=lambda k: pool[k])


@runtime_checkable
class Translator(Protocol):
    async def translate(self, text: str, target_lang: str) -> str: ...


class NoopTranslator:
    """Default: returns text unchanged (translation is opt-in via the KB model)."""

    async def translate(self, text: str, target_lang: str) -> str:
        return text


def build_translate_prompt(text: str, target_lang: str) -> str:
    """Prompt the LLM backend to translate into ``target_lang`` (name resolved from the code)."""
    target = LANG_NAMES.get(target_lang, target_lang)
    return (
        f"Translate the following text into {target}. Preserve meaning, names, numbers and "
        f"formatting. Return ONLY the translation, no preamble.\n\nTEXT:\n{text.strip()[:4000]}\n\n{target}:"
    )
