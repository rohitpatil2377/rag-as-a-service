"""Retrieval service (TDD §4.4 · FR-D1/D2/D7/D8).

Thin orchestration over the hand-rolled :class:`HybridRetriever` (dense + BM25 + RRF
+ rerank). Returns a :class:`RetrievalResult` carrying the ranked spans and the
``max_score`` confidence signal that the generation layer's abstain gate consumes.

Tenant isolation is enforced at the store boundary (collection/index per tenant);
``query.tenant_id`` is the sole authority and is passed explicitly downstream.
"""

from __future__ import annotations

from rag_common import get_logger
from rag_core import HybridRetriever
from rag_schemas import RetrievalQuery, RetrievalResult

_log = get_logger(__name__)


class RetrievalService:
    def __init__(self, retriever: HybridRetriever) -> None:
        self._retriever = retriever

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        spans, max_conf = await self._retriever.retrieve(query)
        _log.info("retrieval_done", tenant_id=query.tenant_id, returned=len(spans), max_conf=round(max_conf, 4))
        # What the retriever performed, not what the caller asked for (P26).
        return RetrievalResult(query=query, spans=spans, max_score=max_conf, mode_used=self._retriever.mode)
