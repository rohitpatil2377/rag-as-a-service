"""Contextualizer adapters (implement :class:`Contextualizer`) — FR-C2.

* :class:`MetadataContextualizer` (from rag-core) — deterministic, no-LLM, default.
* :class:`LLMContextualizer` — canonical technique: an LLM writes a per-chunk context
  from the document via the sovereign gateway. One LLM call per chunk, so it is slow on
  the CPU dev model and reserved for vLLM/GPU (opt-in via ``contextual_retrieval_backend``).
"""

from __future__ import annotations

from rag_common import ConfigError, get_settings
from rag_core import MetadataContextualizer, build_context_prompt
from rag_core.interfaces import LLMGateway
from rag_schemas import Chunk, DocumentEnvelope


class LLMContextualizer:
    """LLM-as-contextualizer (Anthropic Contextual Retrieval). Sovereign gateway."""

    def __init__(self, gateway: LLMGateway) -> None:
        self._gw = gateway

    async def context_for(self, doc: DocumentEnvelope, chunk: Chunk, full_text: str) -> str:
        prompt = build_context_prompt(doc, chunk, full_text)
        return (await self._gw.complete(prompt, max_tokens=64, temperature=0.0)).strip()


def build_contextualizer(gateway: LLMGateway | None = None) -> MetadataContextualizer | LLMContextualizer:
    """Factory: metadata (deterministic/default) or llm, per ``RAG_CONTEXTUAL_RETRIEVAL_BACKEND``."""
    backend = get_settings().contextual_retrieval_backend
    if backend == "metadata":
        return MetadataContextualizer()
    if backend == "llm":
        if gateway is None:
            raise ConfigError("llm contextualizer backend requires an LLM gateway")
        return LLMContextualizer(gateway)
    raise ConfigError(f"unknown contextual retrieval backend: {backend}")
