"""LLM knowledge-graph extractor (implements :class:`GraphExtractor`) — GraphRAG opt-in.

Prompts the KB's own sovereign model for a typed entity/relationship graph. Falls back to
the deterministic heuristic extractor on any unparsable output, so a flaky judge never
silently drops a chunk from the graph.
"""

from __future__ import annotations

import json
import re

from rag_common import get_logger
from rag_core import LLMGateway
from rag_core.graph import (
    GraphTriple,
    HeuristicDomainRelationExtractor,
    HeuristicGraphExtractor,
    build_domain_relation_prompt,
    build_graph_extract_prompt,
    normalize_entity,
)

_log = get_logger(__name__)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_MAX_ENTITIES = 16


class LLMGraphExtractor:
    """Entity/relationship extraction via the tenant's model (richer, typed graph)."""

    def __init__(self, gateway: LLMGateway, *, max_tokens: int = 512) -> None:
        self._gw = gateway
        self._max_tokens = max_tokens
        self._fallback = HeuristicGraphExtractor()

    async def extract(self, text: str) -> GraphTriple:
        raw = await self._gw.complete(build_graph_extract_prompt(text), max_tokens=self._max_tokens, temperature=0.0)
        match = _JSON_RE.search(raw)
        if not match:
            return await self._fallback.extract(text)
        try:
            data = json.loads(match.group(0))
            entities = [normalize_entity(str(e)) for e in data.get("entities", []) if str(e).strip()][:_MAX_ENTITIES]
            names = {e.lower() for e in entities}
            edges: list[tuple[str, str]] = []
            for pair in data.get("edges", []):
                if isinstance(pair, list) and len(pair) >= 2:  # noqa: PLR2004 - [source, target]
                    a, b = normalize_entity(str(pair[0])), normalize_entity(str(pair[1]))
                    if a.lower() in names and b.lower() in names and a.lower() != b.lower():
                        edges.append((a, b))
        except (ValueError, TypeError):
            _log.warning("graph_extract_bad_json_fallback_heuristic", raw=raw[:160])
            return await self._fallback.extract(text)
        return GraphTriple(entities=entities, edges=edges)


class LLMDomainRelationExtractor:
    """Typed domain-relation extraction via the tenant's model, CONSTRAINED to the KB profile's
    relation vocabulary (Slice 6, sub-slice B). Falls back to the deterministic heuristic
    extractor on any unparsable output — the identical safety pattern as :class:`LLMGraphExtractor`
    — so a flaky judge never silently drops a chunk's typed edges."""

    def __init__(self, gateway: LLMGateway, vocab: tuple[str, ...], *, max_tokens: int = 512) -> None:
        self._gw = gateway
        self._vocab = vocab
        self._allowed = {v.lower() for v in vocab}
        self._max_tokens = max_tokens
        self._fallback = HeuristicDomainRelationExtractor(vocab)

    async def extract(self, text: str) -> GraphTriple:
        raw = await self._gw.complete(
            build_domain_relation_prompt(text, self._vocab), max_tokens=self._max_tokens, temperature=0.0
        )
        match = _JSON_RE.search(raw)
        if not match:
            return await self._fallback.extract(text)
        try:
            data = json.loads(match.group(0))
            entities: dict[str, str] = {}
            edges: list[tuple[str, str, str]] = []
            seen: set[tuple[str, str, str]] = set()
            for triple in data.get("edges", []):
                if not (isinstance(triple, list) and len(triple) >= 3):  # noqa: PLR2004 - [subject, relation, object]
                    continue
                a, rel, b = normalize_entity(str(triple[0])), str(triple[1]).strip(), normalize_entity(str(triple[2]))
                # Keep the graph TYPED: only vocabulary relations survive (a free-text label the
                # model invented is dropped rather than stored).
                if rel.lower() not in self._allowed or not a or not b or a.lower() == b.lower():
                    continue
                key = (a.lower(), b.lower(), rel.lower())
                if key in seen:
                    continue
                seen.add(key)
                entities.setdefault(a.lower(), a)
                entities.setdefault(b.lower(), b)
                edges.append((a, b, rel))
        except (ValueError, TypeError):
            _log.warning("domain_relation_bad_json_fallback_heuristic", raw=raw[:160])
            return await self._fallback.extract(text)
        return GraphTriple(
            entities=list(entities.values()),
            edges=[(a, b) for a, b, _ in edges],
            diagram_edges=edges,
        )
