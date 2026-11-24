"""Unit tests for the pure Indic cross-lingual query-expansion core."""

from __future__ import annotations

from svc_control_plane import cross_lingual
from svc_control_plane.cross_lingual import (
    detect_script,
    expand_query_cross_lingual,
    needs_translation,
)


def test_detect_script_devanagari() -> None:
    assert detect_script("नमस्ते") == "devanagari"


def test_detect_script_tamil() -> None:
    assert detect_script("வணக்கம்") == "tamil"


def test_detect_script_latin() -> None:
    assert detect_script("hello") == "latin"


def test_detect_script_mixed_majority_wins() -> None:
    # More Devanagari letters than Latin → Devanagari.
    assert detect_script("नमस्ते hi") == "devanagari"
    # More Latin letters than Devanagari → Latin.
    assert detect_script("hello नम") == "latin"


def test_detect_script_empty_is_latin() -> None:
    assert detect_script("") == "latin"
    assert detect_script("   \t\n") == "latin"
    assert detect_script("123 !?") == "latin"  # no letters → latin


def test_needs_translation_hindi_vs_latin_true() -> None:
    assert needs_translation("नमस्ते", target_script="latin") is True


def test_needs_translation_latin_vs_latin_false() -> None:
    assert needs_translation("hello", target_script="latin") is False


class _FakeTranslator:
    """Records calls; returns a canned translation."""

    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, text: str, target_lang: str) -> str:
        self.calls.append((text, target_lang))
        return self.result


async def test_expand_disabled_returns_original_only() -> None:
    fake = _FakeTranslator("translated")
    out = await expand_query_cross_lingual("नमस्ते", target_lang="en", translate_fn=fake, enabled=False)
    assert out == ["नमस्ते"]
    assert fake.calls == []  # disabled → no translate call


async def test_expand_latin_query_no_translation_needed() -> None:
    fake = _FakeTranslator("bonjour")
    out = await expand_query_cross_lingual("hello", target_lang="en", translate_fn=fake, enabled=True)
    assert out == ["hello"]
    assert fake.calls == []  # already in target script → no translate call


async def test_expand_hindi_query_translates_once() -> None:
    fake = _FakeTranslator("what is the weather")
    out = await expand_query_cross_lingual("मौसम कैसा है", target_lang="en", translate_fn=fake, enabled=True)
    assert out == ["मौसम कैसा है", "what is the weather"]
    assert fake.calls == [("मौसम कैसा है", "en")]  # exactly one call


async def test_expand_identical_translation_deduped() -> None:
    # A translation that echoes the input (or is blank) collapses to a single variant.
    same = _FakeTranslator("मौसम कैसा है")
    out_same = await expand_query_cross_lingual("मौसम कैसा है", target_lang="en", translate_fn=same, enabled=True)
    assert out_same == ["मौसम कैसा है"]

    blank = _FakeTranslator("   ")
    out_blank = await expand_query_cross_lingual("मौसम कैसा है", target_lang="en", translate_fn=blank, enabled=True)
    assert out_blank == ["मौसम कैसा है"]


async def test_expand_translate_error_falls_back_to_original() -> None:
    async def _boom(text: str, target_lang: str) -> str:
        raise RuntimeError("gateway down")

    # Must NOT raise — retrieval never fails because translation did.
    out = await expand_query_cross_lingual("नमस्ते", target_lang="en", translate_fn=_boom, enabled=True)
    assert out == ["नमस्ते"]


def test_module_exports() -> None:
    assert hasattr(cross_lingual, "TranslateFn")
