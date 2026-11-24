"""Shared test fixtures.

The ``inmemory`` fixture activates a store profile for the test and rebuilds all
cached singletons. Two modes, selected by the ``RAG_TEST_REAL_STORES`` env var:

* **default (unset)** — the in-memory profile (no external stores/models). Fast,
  deterministic, zero infra — this is what CI and everyday `make test` run.
* **``RAG_TEST_REAL_STORES=true``** — real-store mode. The fixture does NOT force
  ``RAG_INMEMORY``; it only clears caches so the process re-reads whatever real
  backend env vars (``RAG_QDRANT_URL``, ``RAG_OPENSEARCH_URL``, etc.) the caller
  already exported, pointing at the stack started by `make up`. This lets the exact
  same test bodies validate the real adapters (Qdrant/OpenSearch/MinIO/Postgres/TEI)
  without a second copy of the tests.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator

import pytest

_REAL_STORES_ENV = "RAG_TEST_REAL_STORES"


def _clear_caches() -> None:
    from svc_control_plane import wiring

    from rag_clients import factory
    from rag_common.config import get_settings

    get_settings.cache_clear()
    cached = (
        factory.get_embedder,
        factory.get_vector_store,
        factory.get_sparse_index,
        factory.get_reranker,
        factory.get_object_store,
        factory.get_external_object_fetcher,
        factory.get_catalog,
        factory.get_graph_store,
        factory.get_audit_sink,
        factory.get_llm_gateway,
        factory.get_faithfulness_checker,
        factory.get_contextualizer,
        factory.get_query_router,
        factory.get_query_transformer,
        factory.get_intent_classifier,
        factory.get_molecule_normalizer,
        factory.get_input_guard,
        factory.get_output_guard,
        factory.get_retriever,
        wiring.audit_service,
        wiring.ingestion_service,
        wiring.doc_intel_service,
        wiring.indexing_service,
        wiring.retrieval_service,
        wiring.generation_service,
        wiring.query_router,
        wiring.query_transformer,
        wiring.input_guard,
        wiring.hitl_store,
        wiring.hitl_service,
        wiring.identity_store,
        wiring.kb_store,
        wiring.model_connection_store,
        wiring.sql_connection_store,
        wiring.conversation_store,
        wiring.api_key_store,
        wiring.grant_store,
        wiring.org_unit_store,
        wiring.tenant_registry,
    )
    for fn in cached:
        # Tolerate tests that monkeypatch a cached wiring fn with a plain callable.
        clear = getattr(fn, "cache_clear", None)
        if clear is not None:
            clear()

    from svc_control_plane.quota import reset_limiter
    from svc_control_plane.semantic_cache import cache as semantic_cache

    reset_limiter()
    semantic_cache.clear()  # module singleton — cached answers must not leak across tests


async def _reset() -> None:
    """Close any real adapters cached by the previous test, then clear all caches.

    Real adapters (Qdrant/OpenSearch/TEI/Postgres) hold connection pools that leak if
    never closed — invisible with the in-memory profile, which holds no such resources.
    See :func:`rag_clients.factory.close_all`.
    """
    from rag_clients.factory import close_all

    await close_all()
    _clear_caches()
    if os.environ.get(_REAL_STORES_ENV, "").lower() in {"1", "true", "yes"}:
        await _truncate_global_registries()


async def _truncate_global_registries() -> None:
    """Real-store mode shares ONE database for the whole test session, so the GLOBAL
    ``rag_identity`` registries (federation grants, eval schedules) leak between tests —
    unlike the in-memory twin, which is fresh per test. Truncate them so a real-store run gets
    the same per-test isolation. Best-effort: swallow errors (e.g. migrations not yet applied)
    so a store hiccup never masks the actual test failure."""
    import contextlib

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from rag_common import get_settings

    engine = create_async_engine(get_settings().postgres_dsn)
    try:
        with contextlib.suppress(Exception):
            async with engine.begin() as c:
                await c.execute(
                    text(
                        'TRUNCATE "rag_identity".federation_grants, '
                        '"rag_identity".eval_schedules, '
                        '"rag_identity".org_units, '
                        '"rag_identity".tenants CASCADE'
                    )
                )
    finally:
        await engine.dispose()


@pytest.fixture
async def inmemory(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    """Activate the configured store profile for a test; rebuild all cached singletons.

    See module docstring: real-store mode is opt-in via ``RAG_TEST_REAL_STORES=true``
    with the real backend URLs already exported by the caller; default is in-memory.
    """
    use_real_stores = os.environ.get(_REAL_STORES_ENV, "").lower() in {"1", "true", "yes"}
    monkeypatch.setenv("RAG_ENV", "dev")
    monkeypatch.setenv("RAG_OTEL_ENABLED", "false")
    if not use_real_stores:
        monkeypatch.setenv("RAG_INMEMORY", "true")
        monkeypatch.setenv("RAG_LOG_JSON", "false")
    else:
        monkeypatch.setenv("RAG_INMEMORY", "false")
    await _reset()
    yield
    await _reset()


# The abstain gate, on the scale of the scorer the in-memory profile actually runs.
#
# RAG_ABSTAIN_THRESHOLD is calibrated against the CROSS-ENCODER (see docs/calibration.md):
# genuinely unanswerable questions score at most 0.0575 there, so the shipped default sits low.
# The in-memory profile uses LexicalReranker, whose score is |query terms ∩ chunk terms| /
# |query terms| — a different scale with a noise floor made of function words. Measured:
#
#     "Who won the 1998 world cup?"            vs an SLA document -> 0.1667   (only "the")
#     "What colour is the moon made of cheese?" vs a leave policy  -> 0.1250
#     "What is the uptime target?"              vs an SLA document -> 0.8000   (genuine)
#
# So an unrelated question scores ~0.17 here purely from stopwords, and the cross-encoder
# default falls inside that noise band. A test asserting "an unanswerable question abstains"
# must therefore pin the LEXICAL gate rather than inherit a number that means something else —
# otherwise it is not testing the gate, it is testing a coincidence between two scales.
#
# The platform now carries its own lexical-scale gate (`abstain_threshold_lexical`, default
# 0.60 — the measured best operating point for a scorer whose populations OVERLAP). This
# fixture RELAXES it, for the many e2e tests that need the twin to ground an answer at all so
# they can exercise the wiring downstream of the gate. 0.35 sits clear of the measured stopword
# noise floor and under a genuine match.
#
# A test that needs the twin to produce a grounded answer must say so by taking this fixture.
# That dependency used to be invisible — the tests passed because one cross-encoder number
# happened to fall inside the twin's noise band.
LEXICAL_ABSTAIN_THRESHOLD = 0.35


@pytest.fixture
def lexical_gate(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Relax the lexical-scale abstain gate so the twin can ground. See LEXICAL_ABSTAIN_THRESHOLD.

    The cache clear is load-bearing: `inmemory` runs first and its reset calls back into
    `get_settings()`, so the settings are already warm by the time this fixture sets the
    variable. Without clearing, the env change is read by nothing."""
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_ABSTAIN_THRESHOLD_LEXICAL", str(LEXICAL_ABSTAIN_THRESHOLD))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
