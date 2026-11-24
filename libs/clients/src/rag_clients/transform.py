"""Query transformer adapters (implement :class:`QueryTransformer`) — FR-D6.

* :class:`DecompositionTransformer` (from rag-core) — deterministic default.
* :class:`HyDETransformer` — generate a hypothetical answer passage and retrieve on it.
* :class:`RewriteTransformer` — rewrite the query into a clearer search string.

HyDE/rewrite use the sovereign gateway (one call per query) and always keep the original
query alongside the transform, so the union can only add recall.
"""

from __future__ import annotations

from rag_common import ConfigError, get_logger, get_settings
from rag_core import DecompositionTransformer, build_hyde_prompt, build_rewrite_prompt
from rag_core.interfaces import LLMGateway
from rag_core.transform import (
    MAX_GROUNDING_EXCERPTS,
    MAX_REFORMULATIONS,
    REFORMULATION_MAX_TOKENS,
    build_reformulation_prompt,
    parse_reformulations,
)
from rag_core.transform import QueryTransformer as QueryTransformerLike
from rag_schemas import RetrievalQuery

_log = get_logger(__name__)


class HyDETransformer:
    """Hypothetical Document Embeddings: retrieve on a generated answer passage."""

    def __init__(self, gateway: LLMGateway) -> None:
        self._gw = gateway

    async def transform(self, query: RetrievalQuery, context: list[str] | None = None) -> list[str]:
        passage = (await self._gw.complete(build_hyde_prompt(query), max_tokens=160, temperature=0.0)).strip()
        return [passage, query.text] if passage else [query.text]


class RewriteTransformer:
    """LLM query rewrite for clearer, keyword-rich retrieval."""

    def __init__(self, gateway: LLMGateway) -> None:
        self._gw = gateway

    async def transform(self, query: RetrievalQuery, context: list[str] | None = None) -> list[str]:
        rewrite = (await self._gw.complete(build_rewrite_prompt(query), max_tokens=48, temperature=0.0)).strip()
        return [rewrite, query.text] if rewrite else [query.text]


class ReformulationTransformer:
    """Ask the model for the RULE-shaped questions a situation turns on.

    Distinct from :class:`RewriteTransformer`, which compresses a question into keywords. A
    keyword bag does not help here: the failure is that a scenario and the rule governing it
    share almost no vocabulary, and a cross-encoder scores keyword soup against a rule no
    better than it scored the original. See ``build_reformulation_prompt`` for the measurement.

    Never replaces the original — the union can only add recall, and `merge_retrieved` keeps
    the best score per chunk."""

    def __init__(self, gateway: LLMGateway, *, limit: int = MAX_REFORMULATIONS) -> None:
        self._gw = gateway
        self._limit = limit

    async def transform(self, query: RetrievalQuery, context: list[str] | None = None) -> list[str]:
        try:
            raw = await self._gw.complete(
                build_reformulation_prompt(query, self._limit, (context or [])[:MAX_GROUNDING_EXCERPTS]),
                max_tokens=REFORMULATION_MAX_TOKENS,
                temperature=0.0,
            )
        except Exception as exc:
            # A reformulation is an ENHANCEMENT. Losing it must not lose the query it was meant
            # to help, so this degrades to the literal question rather than failing the ask.
            _log.warning("query_reformulation_failed", error_type=type(exc).__name__, error=str(exc)[:160])
            return [query.text]
        alternatives = parse_reformulations(raw, self._limit)
        if not alternatives:
            _log.info("query_reformulation_empty", detail="model returned no usable questions")
        return [query.text, *alternatives]


class CompositeQueryTransformer:
    """Run complementary strategies and union their outputs.

    The strategies answer DIFFERENT failures and were mutually exclusive by construction, so
    picking one meant giving up the others: decomposition splits a compound question but cannot
    rephrase; reformulation rephrases but will not split. The measured abstention needed the
    second while the platform was configured for the first, and the agentic retry — which
    exists precisely to recover a weak retrieval — was handed the query unchanged and reported
    "no improvement".

    The union is safe: every strategy keeps the original query, and retrieval merges by chunk
    keeping the best score, so an unhelpful reformulation can only cost latency."""

    def __init__(self, *transformers: QueryTransformerLike) -> None:
        self._transformers = transformers

    async def transform(self, query: RetrievalQuery, context: list[str] | None = None) -> list[str]:
        out: list[str] = [query.text]
        for transformer in self._transformers:
            out.extend(await transformer.transform(query, context))
        seen: set[str] = set()
        unique: list[str] = []
        for text in out:
            key = text.strip().lower()
            if text.strip() and key not in seen:
                seen.add(key)
                unique.append(text.strip())
        return unique


def build_query_transformer(gateway: LLMGateway | None = None) -> QueryTransformerLike:
    """Factory: auto (default) | decompose | hyde | rewrite.

    ``auto`` composes the deterministic decomposition with model reformulation, because they
    fix different failures and choosing one used to mean forfeiting the other. Without a
    gateway it degrades to decomposition alone rather than refusing to start."""
    backend = get_settings().query_transform_backend
    if backend == "auto":
        if gateway is None:
            _log.warning(
                "query_transform_deterministic_only",
                detail="no LLM gateway, so a question phrased differently from the documents "
                "cannot be reformulated; compound questions are still split",
            )
            return DecompositionTransformer()
        return CompositeQueryTransformer(DecompositionTransformer(), ReformulationTransformer(gateway))
    if backend == "decompose":
        return DecompositionTransformer()
    if backend in {"hyde", "rewrite"}:
        if gateway is None:
            raise ConfigError(f"{backend} transformer requires an LLM gateway")
        return HyDETransformer(gateway) if backend == "hyde" else RewriteTransformer(gateway)
    raise ConfigError(f"unknown query transform backend: {backend}")
