"""LLM clarify generator (implements :class:`ClarifyGenerator`) — Layer-2 item 4 opt-in.

Prompts the KB's own model for grounded "Did you mean …?" options over the SAME retrieved
spans the heuristic clusters. Fully fail-safe: a gateway error yields NO options (never
raises, never fabricates), so a flaky model can neither break an answer nor nag on one.
"""

from __future__ import annotations

from rag_common import get_logger
from rag_core import LLMGateway
from rag_core.clarify import build_clarify_prompt, parse_clarify
from rag_schemas import RetrievedSpan

_log = get_logger(__name__)


class LLMClarifyGenerator:
    """Grounded disambiguation via the tenant's model (richer phrasing than the heuristic)."""

    def __init__(self, gateway: LLMGateway, *, max_tokens: int = 1024) -> None:
        self._gw = gateway
        self._max_tokens = max_tokens

    async def suggest(self, query: str, spans: list[RetrievedSpan], count: int) -> list[str]:
        if not spans:
            return []
        try:
            raw = await self._gw.complete(
                build_clarify_prompt(query, spans, count), max_tokens=self._max_tokens, temperature=0.3
            )
        except Exception as exc:  # disambiguation must never fail — or nag on — an answer
            _log.warning("clarify_model_failed", error=str(exc)[:160])
            return []
        return parse_clarify(raw, count)
