"""Translate-then-extract GraphRAG entity extraction — robust cross-lingual entities.

Direct entity extraction on Indic text is unreliable: the heuristic extractor keys on
English capitalization (Devanagari has none), reasoning LLMs over-think structured output,
and multilingual NER models (GLiNER) are inconsistent on Hindi. This wrapper sidesteps all
of that by extracting where NER IS reliable — English:

    non-English chunk
      → translate to English (the tenant's model; a simple task it does cleanly)
      → extract entities from the ENGLISH text (reliable inner extractor)
      → MAP the entities back to the source language for the graph nodes

The nodes end up in the source language (so a Hindi query matches Hindi nodes), while the
extraction quality comes from the mature English pipeline. English chunks skip translation
entirely. Every step fails soft — a translation hiccup yields no graph, never a failed ingest.
"""

from __future__ import annotations

from rag_common import get_logger
from rag_core.graph import GraphExtractor, GraphTriple, cooccurrence_edges
from rag_core.language import Translator, detect_language

_log = get_logger(__name__)


class TranslateThenExtract:
    """Extract graph entities via English, then map them back to the source language."""

    def __init__(self, translator: Translator, inner: GraphExtractor, *, max_entities: int = 16) -> None:
        self._tr = translator
        self._inner = inner
        self._max = max_entities

    async def extract(self, text: str) -> GraphTriple:
        if not text.strip():
            return GraphTriple(entities=[], edges=[])
        lang = detect_language(text)
        if lang == "en":
            return await self._inner.extract(text)

        # 1) translate the chunk to English (a simpler task than structured extraction).
        english = await self._tr.translate(text, "en")
        # 2) extract entities where NER is reliable.
        triple = await self._inner.extract(english)
        entities_en = triple.entities[: self._max]
        if not entities_en:
            return GraphTriple(entities=[], edges=[])
        # 3) map the entities back to the source language for the nodes.
        mapped = await self._map_back(entities_en, lang)
        return GraphTriple(entities=mapped, edges=cooccurrence_edges(mapped))

    async def _map_back(self, entities_en: list[str], lang: str) -> list[str]:
        """Translate the entity list back to ``lang``, one per line, preserving order.

        A line-count mismatch means the translation didn't keep the list structure — fall back
        to the English entities (still a valid graph) rather than risk mis-aligned labels."""
        back = await self._tr.translate("\n".join(entities_en), lang)
        lines = [ln.strip(" \t-•*.") for ln in back.splitlines() if ln.strip()]
        if len(lines) == len(entities_en):
            return lines
        _log.warning("graph_mapback_misaligned", got=len(lines), want=len(entities_en), fallback="english")
        return entities_en
