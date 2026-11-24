"""E2E: glossary / synonym query expansion (pre-retrieval) — additive, flag-gated, fail-safe.

A casual question ("who signs off when a product test fails?") shares almost no surface vocabulary
with the SOP that answers it ("...the OOS result shall be reviewed and approved by QA..."). When the
glossary flag is ON, matched casual triggers append the canonical SOP/GMP terms to ``query.text``
BEFORE retrieval — the field the dense embedder AND the cross-encoder reranker both read — so the
right chunk is reachable. The user's ORIGINAL question is preserved on ``original_text`` and is what
the user-facing answer's ``query`` field shows. Flag OFF ⇒ the query is byte-identical.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator

import httpx
import pytest

from rag_schemas import RetrievalQuery

pytestmark = pytest.mark.e2e

_QUESTION = "who signs off when a product test fails?"
# A doc written in canonical SOP terminology — the surface wording the casual question lacks.
_DOC = (
    "Standard operating procedure for out-of-specification OOS results: every OOS result must be "
    "reviewed and approved by Quality Assurance QA before the batch disposition release decision."
)


@pytest.fixture
def _glossary_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Master flag on; bundled default glossary (empty path)."""
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_GLOSSARY_EXPANSION_ENABLED", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["admin", "reader", "ingestor"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def _ingest(c: httpx.AsyncClient, tenant: str, doc_id: str, text: str) -> None:
    body = base64.b64encode(text.encode()).decode()
    await c.post(f"/v1/kb/{tenant}/ingest", json={"doc_id": doc_id, "mime": "text/plain", "content_b64": body})


async def _expand(tenant: str) -> RetrievalQuery:
    """Run the pipeline's glossary step on the casual question for this KB."""
    from svc_control_plane.pipeline import _with_glossary_expansion

    return await _with_glossary_expansion(RetrievalQuery(tenant_id=tenant, text=_QUESTION, top_k=10))


async def test_flag_on_expands_query_and_preserves_original(
    inmemory: None, lexical_gate: None, _glossary_on: None
) -> None:
    """With the flag ON: the query text reaching retrieval is expanded with the canonical SOP
    terms, the original is preserved on ``original_text``, and the user-facing answer's ``query``
    field still shows the ORIGINAL question."""
    tenant = "pharma"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "SOP KB"})
        await _ingest(c, tenant, "sop", _DOC)

        expanded = await _expand(tenant)
        assert expanded.text != _QUESTION, "query.text was not expanded with the flag on"
        assert "out-of-specification" in expanded.text and "OOS" in expanded.text
        assert "approve" in expanded.text and "approval" in expanded.text
        assert expanded.original_text == _QUESTION, "the original question was not preserved"

        from svc_control_plane.pipeline import answer_query

        answer = await answer_query(RetrievalQuery(tenant_id=tenant, text=_QUESTION, top_k=10), actor="officer")
        assert answer.query == _QUESTION, "the user-facing answer must show the ORIGINAL question"


async def test_flag_off_is_byte_identical(inmemory: None) -> None:
    """CONTROL — with the flag OFF the query is not rewritten: same text, no original_text."""
    tenant = "pharma_off"
    expanded = await _expand(tenant)
    assert expanded.text == _QUESTION
    assert expanded.original_text is None
