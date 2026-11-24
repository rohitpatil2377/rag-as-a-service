"""Query-router adapter (RAG v2 Phase 7): one gateway call per question, strictly validated.

Mirrors ``rag_clients.doc_profiler`` and ``rag_clients.section_enricher`` — the prompt and the
parser are pure logic in ``rag_core.router``; this holds the gateway, the retry and the
fail-safe. Named ``query_router_v2`` to sit beside the EXISTING ``rag_clients.routing``
(``HeuristicRouter``/``LLMRouter``), which answers a different question entirely: that one picks a
retrieval DEPTH (``standard``/``heavy``, FR-D5), this one picks the plan's six intents plus the
documents, facets and sub-queries a v2 answer path needs.

FAIL-SAFE INTO THE HEURISTIC, not into nothing. Every other "one small JSON call" in this package
returns None on trouble and lets the caller carry on without it. A router cannot do that — the
pipeline has to route the question somewhere — so the failure mode here is the DETERMINISTIC
BACKEND's decision, which is the plan's own stated fallback ("if router output fails validation,
default to lookup and log") and is also exactly the behaviour of the pipeline with the router
switched off. A model outage therefore costs routing quality and nothing else.

ONE RETRY, on the same terms and for the same reason as the profile call: nothing behind
``LLMGateway.complete`` speaks a constrained-decoding API — there is no ``response_format`` or
grammar parameter anywhere in this codebase — so strict validation plus one retry carrying a
CORRECTIVE LINE is what is available, and an unmodified retry at ``temperature=0.0`` would re-read
the same prompt and write the same bytes.

A QUESTION IS ON THE LATENCY PATH, which is the one way this differs in posture from Phases 3 and
4. Those run at ingest, where a retry costs a document a second or two. This runs before every
answer, so the budget is small (``max_tokens``), the retry is one, and the heuristic is right
there — a slow router is a worse router than a less accurate one.
"""

from __future__ import annotations

from rag_common import get_logger
from rag_core.interfaces import LLMGateway
from rag_core.router import (
    RouterDecision,
    RouterParse,
    RouterScope,
    build_router_prompt,
    heuristic_decision,
    parse_router_decision,
)

_log = get_logger(__name__)

_RETRY_SUFFIX = (
    "\n\nYour previous reply could not be used: {reason}. Reply with ONLY the JSON object "
    "described above — no prose before or after it, no code fence."
)


class LLMQueryRouter:
    """Routes one question through the KB's own sovereign gateway, over the Phase-3 cards."""

    def __init__(self, gateway: LLMGateway, *, max_tokens: int = 400) -> None:
        self._gw = gateway
        self._max_tokens = max_tokens

    async def route(self, query: str, scope: RouterScope, *, query_lang: str = "") -> RouterDecision:
        """The decision. NEVER raises, and never returns nothing — see the module docstring."""
        # Computed FIRST, not as a late fallback: it is the value a failed call degrades to, and
        # building it up front means there is no path on which a model error leaves the pipeline
        # without a decision.
        safe = heuristic_decision(query, scope, query_lang=query_lang)
        prompt = build_router_prompt(query, scope)
        parse = await self._attempt(prompt, scope=scope, query_lang=query_lang, safe=safe, attempt=1)
        if parse is not None and parse.fell_back:
            parse = await self._attempt(
                prompt + _RETRY_SUFFIX.format(reason=parse.error),
                scope=scope,
                query_lang=query_lang,
                safe=safe,
                attempt=2,
            )
        if parse is None or parse.fell_back:
            _log.info("router_used_heuristic", intent=safe.intent, reason="gateway or reply unusable")
            return safe
        if parse.unknown_docs or parse.unknown_facets:
            # WARNING, not info: a router naming documents or facets that do not exist is a router
            # whose decision would have scoped retrieval to nothing or grouped an answer by a label
            # that joins to no row. The decision is still usable — those entries are dropped — but
            # a model doing it consistently is a prompt problem somebody should see.
            _log.warning(
                "router_named_unknown_targets",
                unknown_docs=list(parse.unknown_docs)[:10],
                unknown_facets=list(parse.unknown_facets)[:10],
            )
        _log.info(
            "router_decided",
            intent=parse.decision.intent,
            docs=len(parse.decision.doc_ids),
            facets=len(parse.decision.facets),
            sub_queries=len(parse.decision.sub_queries),
            amendment_view=parse.decision.needs_amendment_view,
            dropped_sub_queries=parse.dropped_sub_queries,
        )
        return parse.decision

    async def _attempt(
        self, prompt: str, *, scope: RouterScope, query_lang: str, safe: RouterDecision, attempt: int
    ) -> RouterParse | None:
        """One call. None when the GATEWAY failed (there is nothing to retry a parse on); a parse
        with ``fell_back`` set when the model answered and the answer was unusable."""
        try:
            raw = await self._gw.complete(prompt, max_tokens=self._max_tokens, temperature=0.0)
        except Exception as exc:  # fail-safe: a model error must never cost a question its answer
            _log.warning("router_call_failed", attempt=attempt, error=str(exc)[:160])
            return None
        parse = parse_router_decision(raw, scope=scope, query_lang=query_lang, fallback=safe)
        if parse.fell_back:
            _log.info("router_unusable_reply", attempt=attempt, reason=parse.error)
        return parse
