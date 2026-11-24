"""Unit tests for Indic language detection (FR — multilingual)."""

from __future__ import annotations

from rag_core.language import build_translate_prompt, detect_language


def test_detects_scripts() -> None:
    assert detect_language("The uptime target is 99.9 percent.") == "en"
    assert detect_language("वार्षिक अवकाश 24 दिन है।") == "hi"  # Devanagari
    assert detect_language("ବାର୍ଷିକ ଛୁଟି 24 ଦିନ।") == "or"  # Odia
    assert detect_language("વાર્ષિક રજા 24 દિવસ છે.") == "gu"  # Gujarati
    assert detect_language("ஆண்டு விடுப்பு 24 நாட்கள்.") == "ta"  # Tamil
    assert detect_language("12345 ...") == "und"  # no letters


def test_mixed_prefers_indic() -> None:
    # HR handbook line with English loan-words but Hindi content.
    assert detect_language("Acme की छुट्टी नीति के अनुसार annual leave 24 दिन है।") == "hi"


def test_translate_prompt_names_target() -> None:
    p = build_translate_prompt("hello", "or")
    assert "Odia" in p and "hello" in p
