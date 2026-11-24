"""A folder sync must be able to FINISH a bucket bigger than one run (FR-A2b).

Before this, every sync listed from the beginning and stopped at the cap, so a bucket with more
objects than the cap ingested its first N and reported found=N -- indistinguishable from success.
Measured live at limit=5: two consecutive syncs of the same prefix both returned SOP-BULK-001..005.

The fix is a persisted CURSOR, not a bigger cap. One run ingests a bounded batch and records the
last key it reached; the next run -- manual or scheduled -- continues from there. A bucket drains
over successive runs instead of one request trying to hold an hour of extraction open, and a cap
still bounds what any single run can cost.
"""

from __future__ import annotations

import pytest
from svc_control_plane import pipeline, sync_jobs
from svc_control_plane.sync_jobs import SyncSource

from rag_schemas.envelope import ExternalObjectSource

_KEYS = [f"sops/SOP-{i:03d}.txt" for i in range(1, 13)]


class _Fetcher:
    """Lists lexicographically and honours start_after, like every real backend."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def list_keys(self, *, prefix: str, limit: int = 200, start_after: str = "", **kw: object) -> list[str]:
        self.calls.append((start_after, limit))
        return [k for k in _KEYS if k.startswith(prefix) and k > start_after][:limit]

    async def fetch(self, **kw: object) -> bytes:
        return b"x"


def _source(limit: int) -> SyncSource:
    return SyncSource(
        external_source=ExternalObjectSource(
            endpoint="obj.local", bucket="b", secure=False, provider="s3", credentials_ref="c"
        ),
        prefix="sops/",
        limit=limit,
    )


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> _Fetcher:
    f = _Fetcher()
    monkeypatch.setattr(sync_jobs, "external_fetcher_for", lambda provider: f)
    monkeypatch.setattr(sync_jobs, "external_credentials", lambda src, secrets: ("a", "s"))
    monkeypatch.setattr(sync_jobs, "get_secrets_provider", lambda: None)
    monkeypatch.setattr(sync_jobs, "assert_sovereign", lambda url: None)

    async def _ingest(doc: object, **kw: object) -> object:
        class _Out:
            outcome = "created"
            chunk_count = 1

        return _Out()

    monkeypatch.setattr(pipeline, "ingest_document", _ingest)
    monkeypatch.setattr(sync_jobs, "check_document_cap", lambda kb: None)
    return f


@pytest.mark.asyncio
async def test_successive_runs_advance_through_the_bucket(inmemory: None, wired: _Fetcher) -> None:
    src = _source(limit=5)
    seen: list[str] = []
    for _ in range(10):  # bounded so a cursor that never advances fails instead of hanging
        r = await sync_jobs.run_sync("kb-cursor", src, actor="t")
        seen.extend(r.keys)
        if not r.more:
            break
    assert seen == _KEYS, "the sync could not reach past the first page"


@pytest.mark.asyncio
async def test_a_run_is_bounded_by_the_cap(inmemory: None, wired: _Fetcher) -> None:
    """The cap still limits what ONE run costs -- the point is resumability, not unbounded work."""
    r = await sync_jobs.run_sync("kb-bound", _source(limit=5), actor="t")
    assert r.found == 5


@pytest.mark.asyncio
async def test_the_cursor_reports_whether_more_remains(inmemory: None, wired: _Fetcher) -> None:
    """Silence about the remainder is how "found=200" read as success on a 2000-object bucket."""
    first = await sync_jobs.run_sync("kb-more", _source(limit=5), actor="t")
    assert first.more is True, "a full page must say more remains"
    last = first
    for _ in range(10):
        last = await sync_jobs.run_sync("kb-more", _source(limit=5), actor="t")
        if not last.more:
            break
    assert last.more is False, "the end of the bucket was never reported"


@pytest.mark.asyncio
async def test_a_finished_bucket_restarts_so_new_objects_are_picked_up(inmemory: None, wired: _Fetcher) -> None:
    """Once the end is reached the cursor resets, or a completed sync would never see an object
    added later -- the delta check makes re-listing cheap."""
    for _ in range(10):  # drain it
        if not (await sync_jobs.run_sync("kb-wrap", _source(limit=5), actor="t")).more:
            break
    again = await sync_jobs.run_sync("kb-wrap", _source(limit=5), actor="t")
    assert again.keys[0] == _KEYS[0], "a drained bucket never re-listed, so new objects would be missed"
