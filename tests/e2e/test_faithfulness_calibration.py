"""E2E: a connected tenant model judges its own faithfulness (calibration, FR-E4/E1).

The platform's default judge is lexical (deterministic, for the CPU dev tier). But when a
KB connects a real generative model, keyword-overlap scoring over-flags good answers — so a
connected model must be judged by an LLM, not the lexical fallback, regardless of the
platform's default backend.
"""

from __future__ import annotations

import pytest

from rag_clients.faithfulness import LexicalFaithfulnessChecker, LLMFaithfulnessChecker

pytestmark = pytest.mark.e2e


async def _connect(tenant: str) -> None:
    from svc_control_plane import wiring
    from svc_control_plane.model_connection import StoredModelConnection

    from rag_clients.factory import get_llm_gateway  # noqa: F401 - ensure module import order

    await wiring.model_connection_store().put(
        StoredModelConnection(
            tenant_id=tenant,
            conn_id="c1",
            name="Tenant LLM",
            endpoint="http://localhost/v1",
            model="tenant-model",
            is_default=True,
        )
    )


async def test_connected_model_is_judged_by_llm(inmemory: None) -> None:
    from svc_control_plane import wiring

    tenant = "acme"
    # No connection → platform default (lexical) is used (resolver returns None).
    assert await wiring.resolve_tenant_faithfulness(tenant) is None

    await _connect(tenant)
    checker = await wiring.resolve_tenant_faithfulness(tenant)
    assert isinstance(checker, LLMFaithfulnessChecker)  # NOT the lexical fallback


async def test_calibration_can_be_disabled(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_FAITHFULNESS_TENANT_MODEL_JUDGE", "false")
    get_settings.cache_clear()
    from svc_control_plane import wiring

    tenant = "acme"
    await _connect(tenant)
    # Flag off → fall back to the platform default checker (resolver returns None).
    assert await wiring.resolve_tenant_faithfulness(tenant) is None
    # And the platform default is indeed lexical here.
    from rag_clients.factory import get_faithfulness_checker

    # Every backend is wrapped for the numeric check; what this asserts is the SELECTION.
    assert isinstance(get_faithfulness_checker().inner, LexicalFaithfulnessChecker)
