"""E2E (FR-C2, Slice 3): a table document answered with parent CONTEXT but cited at the precise
child, through the real API (in-process ASGI, in-memory profile).

Proves the parent/child path end-to-end with the flag ON: ingest under strategy=parent_child (small
children + catalog-only parents) -> narrow retrieval finds the table child -> grounded answer cites
that child (never a parent) -> the in-document viewer shows children only (parents never leak).
"""

from __future__ import annotations

import base64

import httpx
import pytest

from rag_common import get_settings

pytestmark = pytest.mark.e2e

_HEADER = "| Department | Department Code |"
_SEP = "| --- | --- |"
_ROWS = [
    "| Manufacturing | B |",
    "| Capsules | C |",
    "| Quality Control | Q |",
    "| Tablets | T |",
    "| Human Resources | H |",
]
_DOC = (
    "5.7 Department codes. The following table lists each department and its one-letter code, "
    "used throughout the site for document numbering and traceability.\n"
    + "\n".join([_HEADER, _SEP, *_ROWS])
    + "\n6.0 References. None applicable to this procedure."
)


@pytest.fixture
def _pc_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_PARENT_CHILD_ENABLED", "true")
    get_settings.cache_clear()


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["admin", "reader", "ingestor"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def test_table_doc_answers_with_parent_context_and_cites_the_child(
    inmemory: None, lexical_gate: None, _pc_flag: None
) -> None:
    tenant = "acme"
    client = await _client(tenant)
    async with client:
        assert (await client.post("/v1/knowledge-bases", json={"display_name": "KB"})).status_code == 201

        r = await client.post(
            f"/v1/kb/{tenant}/ingest",
            json={
                "doc_id": "sop",
                "mime": "text/plain",
                "content_b64": base64.b64encode(_DOC.encode()).decode(),
                "strategy": "parent_child",
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["chunk_count"] >= 1

        # Narrow lookup: the answer must come from the table row, cited at that child.
        r = await client.post(
            f"/v1/kb/{tenant}/query",
            json={"query": "What is the department code for Quality Control?", "top_k": 5},
        )
        assert r.status_code == 200, r.text
        ans = r.json()
        assert ans["status"] == "grounded", ans
        assert ans["citations"], "grounded answer must carry a citation"
        cited_ids = {c["chunk_id"] for c in ans["citations"]}

        # A parent context record exists for this doc, in the CATALOG only.
        from rag_clients.factory import get_catalog

        all_chunks = await get_catalog().document_chunks(tenant, "sop")
        parent_ids = {c.chunk_id for c in all_chunks if c.metadata.get("chunk_role") == "parent"}
        assert parent_ids, "no parent context record was created for the table section"

        # The in-document viewer returns CHILDREN only — a parent is a non-citable context bank and
        # must never leak into it, nor be cited: every citation resolves to a real, retrievable child.
        r = await client.get(f"/v1/kb/{tenant}/documents/sop/content")
        assert r.status_code == 200, r.text
        viewer_ids = {c["chunk_id"] for c in r.json()["chunks"]}
        assert not (parent_ids & viewer_ids), "a parent leaked into the citation viewer"
        assert not (parent_ids & cited_ids), "a parent was cited (parents are non-citable context)"
        assert cited_ids <= viewer_ids, f"citation outside the child set: cited={cited_ids} viewer={viewer_ids}"
