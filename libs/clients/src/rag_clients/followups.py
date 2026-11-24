"""LLM follow-up generator (implements :class:`FollowupGenerator`) — FR-I8 opt-in.

Prompts the KB's own model for follow-up questions grounded strictly in the retrieved
context. Falls back to the deterministic heuristic on an empty/failed reply, so a flaky
model never leaves the user without suggestions.
"""

from __future__ import annotations

from rag_common import get_logger
from rag_core import LLMGateway
from rag_core.followups import build_followups_prompt, heuristic_followups, parse_followups

_log = get_logger(__name__)


class LLMFollowupGenerator:
    """Context-grounded follow-ups via the tenant's model (richer than the heuristic)."""

    def __init__(self, gateway: LLMGateway, *, max_tokens: int = 256) -> None:
        self._gw = gateway
        self._max_tokens = max_tokens

    async def suggest(self, query: str, context: str, count: int) -> list[str]:
        try:
            raw = await self._gw.complete(
                build_followups_prompt(query, context, count), max_tokens=self._max_tokens, temperature=0.3
            )
        except Exception as exc:  # never let follow-up generation fail an answer
            _log.warning("followups_model_failed_fallback", error=str(exc)[:160])
            return heuristic_followups(query, context, count)
        parsed = parse_followups(raw, count)
        return parsed or heuristic_followups(query, context, count)
