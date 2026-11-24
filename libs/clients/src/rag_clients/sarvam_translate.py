"""Sarvam Translate adapter (implements :class:`Translator`) — dedicated Indic↔English MT.

A REASONING chat model (Sarvam-105b) is unreliable at translation — it echoes the source back
or returns ``content: null`` — so translate-then-extract got no entities from Indic text. Sarvam's
DEDICATED translation model (Sarvam-Translate / Mayura) is purpose-built, deterministic, and fast.

This calls the hosted ``/translate`` REST endpoint (``api.sarvam.ai`` — same subscription key as
the LLM). The SAME wire is served by the open GPL-3.0 weights (``sarvamai/sarvam-translate``), so
pointing ``translate_sarvam_url`` at a self-hosted GPU serving keeps the deployment fully sovereign.

Fails soft: any error returns the original text (translation is a quality boost, never a gate).
"""

from __future__ import annotations

import httpx

from rag_common import assert_sovereign, get_logger, get_settings, model_tls_verify
from rag_core.language import detect_language

_log = get_logger(__name__)

# Our internal script codes (rag_core.language) → Sarvam BCP-47-ish codes. Sarvam wants a
# region suffix; Devanagari maps to Hindi (Marathi shares the script — "auto" source covers it).
_LANG = {
    "en": "en-IN",
    "hi": "hi-IN",
    "bn": "bn-IN",
    "pa": "pa-IN",
    "gu": "gu-IN",
    "or": "od-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
}


def _sarvam_lang(code: str) -> str:
    """Map an internal 2-letter code to Sarvam's code; default to English target."""
    return _LANG.get(code, "en-IN")


def _chunks(text: str, limit: int) -> list[str]:
    """Split text to fit the API's per-request character cap, preferring paragraph/sentence
    boundaries so a translated chunk is still coherent. Greedy: pack lines until the next would
    overflow. A single over-long line is hard-split as a last resort."""
    out: list[str] = []
    buf = ""
    for line in text.splitlines(keepends=True):
        cleaned = line
        while len(cleaned) > limit:  # a single monster line — hard split
            if buf:
                out.append(buf)
                buf = ""
            out.append(cleaned[:limit])
            cleaned = cleaned[limit:]
        if len(buf) + len(cleaned) > limit:
            out.append(buf)
            buf = cleaned
        else:
            buf += cleaned
    if buf:
        out.append(buf)
    return out or [text]


class SarvamTranslator:
    """Translation via Sarvam's dedicated ``/translate`` endpoint."""

    def __init__(
        self,
        url: str | None = None,
        *,
        model: str | None = None,
        api_key: str | None = None,
        api_key_header: str | None = None,
        max_chars: int | None = None,
        timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        self._url = (url or settings.translate_sarvam_url).rstrip("/")
        assert_sovereign(self._url)  # host must be egress-allowlisted, same as every model call
        self._model = model or settings.translate_sarvam_model
        self._key = api_key if api_key is not None else (settings.translate_sarvam_api_key or settings.llm_api_key)
        self._key_header = api_key_header or settings.translate_sarvam_api_key_header
        self._max_chars = max_chars or settings.translate_max_chars
        self._client = httpx.AsyncClient(
            base_url=self._url,
            timeout=timeout or settings.rerank_timeout_seconds,
            headers={self._key_header: self._key} if self._key else {},
            verify=model_tls_verify(),
        )

    async def translate(self, text: str, target_lang: str) -> str:
        text = text.strip()
        if not text:
            return text
        target = _sarvam_lang(target_lang)
        # Sarvam /translate needs an explicit source code (it rejects "auto"), so detect it by
        # script. Devanagari → hi-IN also covers Marathi (shared script) well enough for MT.
        source = _sarvam_lang(detect_language(text))
        if source == target:  # already in the target language — nothing to do
            return text
        try:
            parts = [await self._one(part, source, target) for part in _chunks(text, self._max_chars)]
        except httpx.HTTPError as exc:  # never fail hard — translation is a quality boost
            _log.warning("sarvam_translate_failed", url=self._url, error=str(exc)[:160])
            return text
        joined = "".join(parts).strip()
        return joined or text

    async def _one(self, part: str, source: str, target: str) -> str:
        resp = await self._client.post(
            "/translate",
            json={
                "input": part,
                "source_language_code": source,
                "target_language_code": target,
                "model": self._model,
            },
        )
        resp.raise_for_status()
        return str(resp.json().get("translated_text") or "")

    async def aclose(self) -> None:
        await self._client.aclose()


def build_translator() -> SarvamTranslator:
    """Factory for the Sarvam translator wired from config (url/model/key)."""
    return SarvamTranslator()
