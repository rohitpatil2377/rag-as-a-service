"""Chunking behaviour is declared once and recorded (P12/P50).

Two faults, one root cause: nothing declared how a document should be chunked, and nothing
recorded how it actually WAS.

* **P50.** Four API entry points each carried their own literal default — ``/ingest`` said
  FIXED while reindex, sync and bulk said CLAUSE, and pipeline/crawl/MCP added three more.
  A document uploaded through the single-document API was chunked FIXED and then silently
  re-chunked CLAUSE the first time anything re-indexed it. Different chunk count, different
  boundaries, different chunk ids — and no signal anywhere. Any before/after measurement
  spanning a re-index was comparing two chunkers.
* **P12.** Size and overlap were constructor defaults in rag-core, so tuning them meant
  editing library code.

The rule these tests pin: a re-index reproduces what the document already had, unless the
caller explicitly asks for something else. That is what makes the Wave-3 re-index a
measurement rather than a confound.
"""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    token = issue_dev_jwt("o", tenant, roles=["reader", "ingestor", "admin"])
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()),
        base_url="http://t",
        headers={"Authorization": f"Bearer {token}"},
    )


# Long enough that FIXED (character windows) and CLAUSE (sentence accumulation) cannot
# possibly agree on the chunk count.
_TEXT = " ".join(
    f"Clause {i} states that the retention period for record type {i:02d} is {i} calendar years." for i in range(40)
)


async def _ingest(c: httpx.AsyncClient, kb: str, doc_id: str, **extra: object) -> dict:
    body = {"doc_id": doc_id, "mime": "text/plain", "content_b64": base64.b64encode(_TEXT.encode()).decode()}
    r = await c.post(f"/v1/kb/{kb}/ingest", json={**body, **extra})
    assert r.status_code == 200, r.text
    return dict(r.json())


async def test_a_reindex_reproduces_the_documents_own_chunking(inmemory: None) -> None:
    """The P50 failure. Reindex used to hardcode CLAUSE regardless of how the document was
    ingested, so a FIXED document was silently converted the first time anyone re-indexed."""
    kb = "chunkco"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        ingested = await _ingest(c, kb, "a", strategy="fixed")

        r = await c.post(f"/v1/kb/{kb}/documents/a/reindex")
        assert r.status_code == 200, r.text
        assert r.json()["chunk_count"] == ingested["chunk_count"], "reindex re-chunked it differently"


async def test_an_explicit_strategy_still_wins_on_reindex(inmemory: None) -> None:
    """Adopting a new chunking strategy is the whole point of reindex — it just must not
    happen by accident."""
    kb = "chunkco2"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await _ingest(c, kb, "a", strategy="fixed")
        before = [ch["text"] for ch in (await c.get(f"/v1/kb/{kb}/documents/a/content")).json()["chunks"]]

        r = await c.post(f"/v1/kb/{kb}/documents/a/reindex", json={"strategy": "clause"})
        assert r.status_code == 200, r.text

        # Compare BOUNDARIES, not counts: the two chunkers can coincidentally agree on how
        # many chunks a given text yields while cutting it in completely different places.
        after = [ch["text"] for ch in (await c.get(f"/v1/kb/{kb}/documents/a/content")).json()["chunks"]]
        assert after != before, "explicit strategy did not re-chunk the document"

        # ...and the document now records what it was actually re-chunked as, so a LATER
        # reindex reproduces the new strategy rather than reverting to the old one.
        docs = (await c.get(f"/v1/kb/{kb}/documents")).json()
        assert docs[0]["metadata"]["index_chunk_strategy"] == "clause"


async def test_the_strategy_used_is_recorded_on_the_document(inmemory: None) -> None:
    """A re-index can only reproduce what it can read back. It is also the only way to answer
    "which chunker is this corpus actually using?" without re-deriving it from chunk sizes."""
    kb = "chunkco3"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await _ingest(c, kb, "a", strategy="fixed")

        docs = (await c.get(f"/v1/kb/{kb}/documents")).json()
        assert docs[0]["metadata"]["index_chunk_strategy"] == "fixed"


async def test_every_entry_point_shares_one_configured_default(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Four endpoints carried four literals. There is now one setting, and changing it moves
    all of them — which is what makes the default auditable instead of accidental."""
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_CHUNK_STRATEGY", "fixed")
    get_settings.cache_clear()
    kb = "chunkco4"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await _ingest(c, kb, "a")  # no strategy given -> the configured one
        docs = (await c.get(f"/v1/kb/{kb}/documents")).json()
        assert docs[0]["metadata"]["index_chunk_strategy"] == "fixed"
    get_settings.cache_clear()


async def test_chunk_size_is_operator_tunable(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """P12: tuning size/overlap used to mean editing rag-core."""
    from rag_common.config import get_settings

    counts = []
    for size in ("400", "1200"):
        monkeypatch.setenv("RAG_CHUNK_STRATEGY", "fixed")
        monkeypatch.setenv("RAG_CHUNK_SIZE", size)
        get_settings.cache_clear()
        kb = f"chunksize{size}"
        c = await _client(kb)
        async with c:
            await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
            counts.append((await _ingest(c, kb, "a"))["chunk_count"])
    get_settings.cache_clear()
    assert counts[0] > counts[1], f"smaller windows must produce more chunks: {counts}"
