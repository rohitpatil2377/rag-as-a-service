"""Offline unit tests for adapters that need no live stores.

Adapters that require Qdrant/OpenSearch/MinIO/Postgres are exercised in
``tests/integration`` against the docker-compose stack.
"""

from __future__ import annotations

import pytest

from rag_clients.llm_gateway import ExtractiveDevGateway, _split_sources
from rag_clients.reranker import LexicalReranker
from rag_core.grounding import build_prompt
from rag_schemas import RetrievedSpan


def _span(chunk_id: str, text: str) -> RetrievedSpan:
    return RetrievedSpan(chunk_id=chunk_id, doc_id="d1", text=text, score=0.0)


async def test_lexical_reranker_orders_by_overlap() -> None:
    spans = [
        _span("c1", "the sky is blue and clear"),
        _span("c2", "platform uptime target is 99.5 percent baseline"),
    ]
    out = await LexicalReranker().rerank("what is the uptime target", spans, top_k=2)
    assert out[0].chunk_id == "c2"
    assert 0.0 <= out[0].score <= 1.0


def test_extractive_gateway_parses_sources_and_cites() -> None:
    spans = [_span("c1", "Uptime target is 99.5% baseline, 99.9% peak."), _span("c2", "Other text.")]
    prompt = build_prompt("uptime target?", spans)
    parsed = _split_sources(prompt)
    assert parsed and parsed[0][0] == 1


async def test_extractive_gateway_emits_citation_marker() -> None:
    spans = [_span("c1", "Uptime target is 99.5% baseline.")]
    prompt = build_prompt("uptime?", spans)
    answer = await ExtractiveDevGateway().complete(prompt)
    assert "[1]" in answer


async def test_extractive_gateway_handles_no_sources() -> None:
    answer = await ExtractiveDevGateway().complete("no sources here")
    assert "don't know" in answer.lower()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
