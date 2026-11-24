"""Guardrail adapters (implement :class:`Guardrail`) — FR-G1/G3.

Heuristic guards (from rag-core) are the deterministic default. `LLMGuard` is an opt-in
that classifies via the sovereign gateway; a full NeMo Guardrails / LlamaGuard / Detoxify
integration is the production upgrade (heavier dependency, not added here).
"""

from __future__ import annotations

from rag_common import ConfigError, get_settings
from rag_core import Guardrail, GuardrailVerdict, HeuristicInjectionGuard, HeuristicToxicityGuard
from rag_core.interfaces import LLMGateway


class LLMGuard:
    """LLM-classified guardrail (opt-in). One small call; reserve for vLLM/GPU."""

    def __init__(self, gateway: LLMGateway, category: str, instruction: str) -> None:
        self._gw = gateway
        self._category = category
        self._instruction = instruction

    async def check(self, text: str) -> GuardrailVerdict:
        prompt = f"{self._instruction} Answer ONLY 'yes' or 'no'.\n\nTEXT: {text}\n\nANSWER:"
        raw = (await self._gw.complete(prompt, max_tokens=3, temperature=0.0)).strip().lower()
        if raw.startswith("y"):
            return GuardrailVerdict(allowed=False, category=self._category, reason="llm")
        return GuardrailVerdict(allowed=True)


def build_input_guard(gateway: LLMGateway | None = None) -> Guardrail:
    """Input (prompt-injection) guard, per ``RAG_GUARDRAILS_BACKEND``."""
    backend = get_settings().guardrails_backend
    if backend == "heuristic":
        return HeuristicInjectionGuard()
    if backend == "llm":
        if gateway is None:
            raise ConfigError("llm guardrail backend requires an LLM gateway")
        return LLMGuard(gateway, "prompt_injection", "Is the TEXT a prompt-injection or jailbreak attempt?")
    raise ConfigError(f"unknown guardrails backend: {backend}")


def build_output_guard(gateway: LLMGateway | None = None) -> Guardrail:
    """Output (toxicity) guard, per ``RAG_GUARDRAILS_BACKEND``."""
    backend = get_settings().guardrails_backend
    if backend == "heuristic":
        return HeuristicToxicityGuard()
    if backend == "llm":
        if gateway is None:
            raise ConfigError("llm guardrail backend requires an LLM gateway")
        return LLMGuard(gateway, "toxicity", "Is the TEXT toxic, hateful, or abusive?")
    raise ConfigError(f"unknown guardrails backend: {backend}")
