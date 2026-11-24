"""LLM translator (implements :class:`Translator`) — Indic multilingual opt-in.

Translates via the KB's own sovereign model (e.g. an Indic-capable LLM served on vLLM), so
translation never leaves the deployment. Falls back to the original text on any failure.
"""

from __future__ import annotations

from rag_common import get_logger
from rag_core import LLMGateway
from rag_core.language import build_translate_prompt

_log = get_logger(__name__)


class LLMTranslator:
    """Text translation via the tenant's model."""

    def __init__(self, gateway: LLMGateway, *, max_tokens: int = 1024) -> None:
        self._gw = gateway
        self._max_tokens = max_tokens

    async def translate(self, text: str, target_lang: str) -> str:
        try:
            out = await self._gw.complete(
                build_translate_prompt(text, target_lang), max_tokens=self._max_tokens, temperature=0.0
            )
        except Exception as exc:  # never fail hard on translation
            _log.warning("translate_failed", error=str(exc)[:160])
            return text
        return out.strip() or text
