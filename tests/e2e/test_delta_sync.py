"""E2E: FR-A8 content-hash delta + FR-C9 near-duplicate detection at ingest."""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    token = issue_dev_jwt("o", tenant, roles=["admin", "reader", "ingestor"])
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()),
        base_url="http://t",
        headers={"Authorization": f"Bearer {token}"},
    )


async def _ingest(c, kb, doc_id, text):
    body = base64.b64encode(text.encode()).decode()
    r = await c.post(
        f"/v1/kb/{kb}/ingest", json={"doc_id": doc_id, "mime": "text/plain", "content_b64": body, "strategy": "clause"}
    )
    return r.json()


async def test_delta_skips_unchanged_and_flags_duplicates(inmemory: None) -> None:
    kb = "deltaco"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        # First ingest -> created.
        r1 = await _ingest(c, kb, "a", "Annual leave is 24 days per year.")
        assert r1["outcome"] == "created" and r1["duplicate_of"] is None

        # Re-ingest identical content, same doc -> unchanged (delta skip).
        r2 = await _ingest(c, kb, "a", "Annual leave is 24 days per year.")
        assert r2["outcome"] == "unchanged"

        # Changed content, same doc -> updated.
        r3 = await _ingest(c, kb, "a", "Annual leave is now 30 days per year.")
        assert r3["outcome"] == "updated"

        # Identical bytes under a DIFFERENT doc -> near-duplicate flag.
        r4 = await _ingest(c, kb, "b", "Annual leave is now 30 days per year.")
        assert r4["duplicate_of"] == "a"


# A long v1 that clause-chunks into several chunks, and a v2 that fits in one. The unique
# ZULU token per sentence is what makes "is the old text still reachable?" answerable.
_V1 = " ".join(
    f"Clause {i} states that the retention period for record type ZULU{i:02d} is {i} calendar years." for i in range(30)
)
_V2 = "This retention policy has been withdrawn in its entirety."


async def test_reingesting_shorter_content_leaves_no_chunk_of_the_previous_version(inmemory: None) -> None:
    """Chunk ids are positional (``{doc_id}::{idx}``) and every store upserts BY chunk_id, so
    overwriting a document with SHORTER content only rewrites ``::0..N`` — the old tail survives
    in the catalog, the vector store and BM25, still retrievable and still citable. That is a
    DPDP right-to-erasure gap: a tenant who redacts a document by re-uploading it does not
    actually remove anything."""
    kb = "purgeco"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})

        r1 = await _ingest(c, kb, "a", _V1)
        assert r1["outcome"] == "created"
        assert r1["chunk_count"] > 1, "fixture must produce several chunks or it proves nothing"

        r2 = await _ingest(c, kb, "a", _V2)
        assert r2["outcome"] == "updated"
        assert r2["chunk_count"] == 1

        # The catalog holds ONLY the new version.
        content = (await c.get(f"/v1/kb/{kb}/documents/a/content")).json()
        assert len(content["chunks"]) == 1
        assert "ZULU" not in content["chunks"][0]["text"]

        # ...and neither index can still serve the withdrawn text.
        hits = await c.post(f"/v1/kb/{kb}/retrieve", json={"query": "retention period for ZULU17", "top_k": 10})
        assert not any("ZULU" in s["text"] for s in hits.json()["spans"])


async def test_the_same_clause_in_two_documents_is_reported(inmemory: None) -> None:
    """Byte-identical re-upload is already refused (FR-A8/C9). This is what that cannot see:
    the SAME TEXT under a different doc_id.

    Real and measured: acme__hr_policies holds 15 chunk pairs at similarity 1.0, because the
    Code of Conduct is reproduced verbatim inside the HR Policy Handbook. Retrieve both, cite
    both, and one rule reads as two independent confirmations — which is exactly what the QA
    answer citing "[1], [3], [5] ... the same requirement is reiterated in..." was doing.

    Reported, not dropped: a reprint is real content and may be the passage a reader wants."""
    import structlog

    clause = (
        "All employees shall conduct themselves with integrity and shall not disclose "
        "confidential information relating to the Company, its customers or its partners "
        "to any third party without the prior written approval of the Company."
    )
    kb = "dupeco"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await _ingest(c, kb, "code_of_conduct", clause)
        with structlog.testing.capture_logs() as logs:
            await _ingest(c, kb, "hr_handbook", f"Chapter 4: Conduct.\n{clause}")

    reported = [e for e in logs if e.get("event") == "chunks_duplicate_existing_content"]
    assert reported, "the reprint was indexed with nothing recording that it is a reprint"
    assert reported[0]["doc_id"] == "hr_handbook"


async def test_a_document_that_shares_no_text_is_not_reported(inmemory: None) -> None:
    """Two policies from one organisation share vocabulary and register. Flagging those would
    make the signal useless and would cast doubt on genuine second sources."""
    import structlog

    kb = "dupeco2"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await _ingest(c, kb, "leave", "Annual leave entitlement is 24 days for confirmed staff each year.")
        with structlog.testing.capture_logs() as logs:
            await _ingest(c, kb, "shift", "The night shift allowance is paid to operations engineers monthly.")

    assert not [e for e in logs if e.get("event") == "chunks_duplicate_existing_content"]


async def test_a_reindex_that_fails_leaves_the_document_intact(inmemory: None) -> None:
    """Observed for real on 2026-07-31: the single re-index purged each document BEFORE
    re-running extract → chunk → embed → index, and the embedder timed out on the three largest
    documents. Two of them were left with ZERO chunks — present in the catalog, retrievable by
    nothing — until they were re-indexed by hand. On a customer install that is silent data
    loss, and the tenant's own raw bytes are the only reason it was recoverable.

    Write-then-swap: the new chunks are indexed FIRST and only the ids that did not survive are
    purged, so a failure anywhere leaves the previous version serving."""
    kb = "swapco"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await _ingest(c, kb, "a", _V1)
        before = (await c.get(f"/v1/kb/{kb}/documents/a/content")).json()["chunks"]
        assert len(before) > 1

        # Make the embed step fail the way the real re-index failed.
        import httpx as _httpx

        from rag_clients.factory import get_embedder

        embedder = get_embedder()
        original = embedder.embed

        async def _boom(texts: list[str]) -> list[list[float]]:
            raise _httpx.ReadTimeout("embedder timed out")

        embedder.embed = _boom  # type: ignore[method-assign]
        try:
            # In-process ASGI re-raises rather than returning 500; either way the re-index failed.
            with pytest.raises(_httpx.ReadTimeout):
                await c.post(f"/v1/kb/{kb}/documents/a/reindex", json={"strategy": "clause"})
        finally:
            embedder.embed = original  # type: ignore[method-assign]

        after = (await c.get(f"/v1/kb/{kb}/documents/a/content")).json()["chunks"]
        assert len(after) == len(before), "a failed re-index emptied the document"
        hits = await c.post(f"/v1/kb/{kb}/retrieve", json={"query": "retention period for ZULU17", "top_k": 5})
        assert any("ZULU" in s["text"] for s in hits.json()["spans"]), "the old version stopped being retrievable"


async def test_a_failed_re_ingest_leaves_the_previous_version_serving(inmemory: None) -> None:
    """A failed UPDATE must not destroy a previously-good document (HIGH data-loss bug).

    ingest_document used to PURGE the old version's chunks BEFORE extraction/indexing and, on any
    extract/index exception, DELETE the catalog row AND the raw object — so re-uploading a good,
    indexed doc that then failed to embed wiped it entirely. Write-then-swap (mirroring
    reindex_document): index the new version first, purge only genuinely-stale ids AFTER success,
    and on failure of an UPDATE leave the prior version COMPLETELY intact (still retrievable)."""
    kb = "reingestswap"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await _ingest(c, kb, "a", _V1)
        before = (await c.get(f"/v1/kb/{kb}/documents/a/content")).json()["chunks"]
        assert len(before) > 1

        # A completed create is a billable INGEST; a FAILED update must NOT add another one.
        ingests_before = len(
            [e for e in (await c.get(f"/v1/kb/{kb}/audit?limit=200")).json() if e["action"] == "ingest"]
        )

        # Force the embed step of the UPDATE's indexing to fail, the way the real re-index failed.
        import httpx as _httpx

        from rag_clients.factory import get_embedder

        embedder = get_embedder()
        original = embedder.embed

        async def _boom(texts: list[str]) -> list[list[float]]:
            raise _httpx.ReadTimeout("embedder timed out")

        embedder.embed = _boom  # type: ignore[method-assign]
        try:
            with pytest.raises(_httpx.ReadTimeout):
                await _ingest(c, kb, "a", _V2)  # a shorter new version that will fail to index
        finally:
            embedder.embed = original  # type: ignore[method-assign]

        # The prior version is untouched: same chunks in the catalog...
        after = (await c.get(f"/v1/kb/{kb}/documents/a/content")).json()["chunks"]
        assert len(after) == len(before), "a failed re-ingest emptied the document"
        assert any("ZULU" in ch["text"] for ch in after), "the old text vanished from the catalog"
        # ...and still retrievable from the indexes.
        hits = await c.post(f"/v1/kb/{kb}/retrieve", json={"query": "retention period for ZULU17", "top_k": 5})
        assert any("ZULU" in s["text"] for s in hits.json()["spans"]), "the old version stopped being retrievable"

        # The failed update was NOT metered as an INGEST (billing counts INGEST events).
        ingests_after = len(
            [e for e in (await c.get(f"/v1/kb/{kb}/audit?limit=200")).json() if e["action"] == "ingest"]
        )
        assert ingests_after == ingests_before, "a failed ingest was still recorded as a billable event"


async def test_reindex_leaves_exactly_the_new_chunks_and_no_leftovers(inmemory: None) -> None:
    """The write-then-swap has to compare against the ids the re-index ACTUALLY produced. Reading
    them back from the catalog does not work: nothing was purged first, so the catalog holds old
    union new and `old - fresh` is empty by construction — the swap silently deletes nothing.

    Observed live on 2026-07-31: the re-index reported 120 chunks for the Code of Conduct while
    the catalog held 132, and 552 vs 627 for the Abstract. Stale chunks stay retrievable and
    citable with text the document no longer contains."""
    kb = "swapcount"
    c = await _client(kb)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        await _ingest(c, kb, "a", _V1)

        # Re-chunk with a different strategy so the boundaries — and therefore the ids — move.
        r = await c.post(f"/v1/kb/{kb}/documents/a/reindex", json={"strategy": "fixed"})
        assert r.status_code == 200, r.text
        reported = r.json()["chunk_count"]

        live = (await c.get(f"/v1/kb/{kb}/documents/a/content")).json()["chunks"]
        assert len(live) == reported, f"re-index reported {reported} chunks, catalog holds {len(live)}"
