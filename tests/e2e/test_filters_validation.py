"""E2E: a chunk RECORD field cannot be smuggled in as a metadata filter.

`filters` is the FR-D8 metadata/tag channel. Naming a record field there used to be accepted and
then SILENTLY under-match: the sparse index resolves every filter key to `meta.<key>`
(`opensearch_index.meta_path`), so `filters={"doc_id": X}` matched nothing on the BM25 leg while
the dense leg's flat payload matched the record. A document-scoped query therefore ran dense-only,
with no lexical leg and no error — measured live at 181 BM25 hits for the record field vs 0 for
`meta.doc_id`. A query whose distinguishing tokens are exact identifiers (standards codes, clause
numbers) is exactly the case that loses most when BM25 drops out.

The dedicated predicate is `RetrievalQuery.doc_ids`. These tests pin the refusal at the HTTP
boundary, so an external caller is told that before any retrieval runs rather than receiving a
quietly degraded answer.
"""

from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.e2e

_RECORD_FIELDS = ["chunk_id", "doc_id", "tenant_id", "page", "acl"]


async def _client(org: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("owner", org, roles=["reader", "ingestor", "admin"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


@pytest.mark.parametrize("field", _RECORD_FIELDS)
@pytest.mark.parametrize("path", ["query", "query/stream", "retrieve"])
async def test_a_record_field_filter_is_refused_on_every_query_endpoint(inmemory: None, field: str, path: str) -> None:
    """All three endpoints share one QueryRequest, so all three must refuse it."""
    c = await _client("acme")
    async with c:
        kb = (await c.post("/v1/knowledge-bases", json={"display_name": "Filters"})).json()
        res = await c.post(f"/v1/kb/{kb['kb_id']}/{path}", json={"query": "anything?", "filters": {field: "x"}})
        assert res.status_code == 422, f"{path} accepted a {field} filter"


async def test_the_refusal_names_the_field_to_use_instead(inmemory: None) -> None:
    """A 422 that does not say what to do instead just moves the confusion."""
    c = await _client("acme")
    async with c:
        kb = (await c.post("/v1/knowledge-bases", json={"display_name": "Filters"})).json()
        res = await c.post(f"/v1/kb/{kb['kb_id']}/query", json={"query": "anything?", "filters": {"doc_id": "a.pdf"}})
        assert res.status_code == 422
        assert "doc_ids" in res.text


async def test_ordinary_tag_filters_are_unaffected(inmemory: None) -> None:
    """Only the five record fields are refused — the B7/D8 tag channel keeps working."""
    c = await _client("acme")
    async with c:
        kb = (await c.post("/v1/knowledge-bases", json={"display_name": "Filters"})).json()
        res = await c.post(
            f"/v1/kb/{kb['kb_id']}/query",
            json={"query": "anything?", "filters": {"doc_type": "policy", "lang": "en"}},
        )
        assert res.status_code == 200, res.text
