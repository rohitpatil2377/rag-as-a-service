"""Slice 2a L1 — a per-KB exact_boost override is clamped to non-negative before it reaches the
sparse leg. `model_copy` skips pydantic validation, so RetrievalQuery's ge=0 guard does not fire
on this server-resolved path; a misconfigured `exact_boost: -1` in tenant_config would otherwise
reach OpenSearch as a negative `boost` and error every query for that KB while the flag is on.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from svc_control_plane import pipeline

from rag_common import get_settings
from rag_schemas import RetrievalQuery


@pytest.fixture
def _exact_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("RAG_EXACT_TERM_MATCH_ENABLED", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _query() -> RetrievalQuery:
    return RetrievalQuery(tenant_id="kb1", text="SOP-BULK-001", top_k=5)


async def test_negative_per_kb_override_is_clamped_to_zero(_exact_on: None, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _overrides(_kb: str) -> dict[str, float]:
        return {"exact_boost": -1.0}

    monkeypatch.setattr(pipeline, "kb_threshold_overrides", _overrides)
    result = await pipeline._with_exact_boost(_query())
    assert result.exact_boost == 0.0


async def test_positive_per_kb_override_passes_through(_exact_on: None, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _overrides(_kb: str) -> dict[str, float]:
        return {"exact_boost": 4.0}

    monkeypatch.setattr(pipeline, "kb_threshold_overrides", _overrides)
    result = await pipeline._with_exact_boost(_query())
    assert result.exact_boost == 4.0


async def test_flag_off_leaves_the_query_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_EXACT_TERM_MATCH_ENABLED", "false")
    get_settings.cache_clear()
    try:
        result = await pipeline._with_exact_boost(_query())
        assert result.exact_boost is None
    finally:
        get_settings.cache_clear()
