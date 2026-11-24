"""LLM community summariser (implements :class:`CommunitySummarizer`) — GraphRAG slice 2 opt-in.

Prompts the KB's own sovereign model to name each thematic cluster in one sentence, from its
top entities + representative passages. Falls back to the deterministic heuristic label on any
empty/failed response, so a flaky model never leaves a community unlabelled.
"""

from __future__ import annotations

from rag_common import get_logger
from rag_core import LLMGateway
from rag_core.graph import build_community_summary_prompt, summarize_community_heuristic

_log = get_logger(__name__)


class LLMCommunitySummarizer:
    """Prose theme summary via the tenant's model (richer than the entity-list heuristic)."""

    def __init__(self, gateway: LLMGateway, *, max_tokens: int = 256) -> None:
        self._gw = gateway
        self._max_tokens = max_tokens

    async def summarize(self, top_entities: list[str], sample_texts: list[str], size: int) -> str:
        prompt = build_community_summary_prompt(top_entities, sample_texts)
        try:
            raw = (await self._gw.complete(prompt, max_tokens=self._max_tokens, temperature=0.0)).strip()
        except Exception as exc:  # never let summarisation fail a community build
            _log.warning("community_summary_failed_fallback_heuristic", error=str(exc)[:160])
            return summarize_community_heuristic(top_entities, size)
        # A reasoning model may return an empty string after a <think> block, or an over-long
        # paragraph — fall back / trim to keep the theme a single crisp line.
        line = raw.splitlines()[0].strip() if raw else ""
        return line[:280] if line else summarize_community_heuristic(top_entities, size)
