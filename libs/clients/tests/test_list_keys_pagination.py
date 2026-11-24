"""Listing must be able to CONTINUE past the first page (FR-A2b folder sync).

Proven live before this existed: syncing the same prefix twice with limit=5 returned the same
five keys both times --

    sync #1 limit=5 -> SOP-BULK-001 ... 005
    sync #2 limit=5 -> SOP-BULK-001 ... 005   (identical)

Every lister stops at `limit` counting from the START of the listing, so objects past the cap
were unreachable no matter how many times a sync ran. With the API's cap at 200, a bucket of 500
or 2000 SOPs ingested its first 200 and reported found=200 -- which looks exactly like success.

`start_after` is the continuation: keys strictly greater than the marker, in the same
lexicographic order every backend already returns. The marker is the last key of the previous
page, so no page is skipped and none is fetched twice.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from rag_clients.memory import InMemoryExternalFetcher
from rag_common import get_settings

_EP = "obj.local"
_KEYS = [f"sops/SOP-{i:03d}.txt" for i in range(1, 13)]


@pytest.fixture(autouse=True)
def _allow_endpoint(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Allowlist the fake endpoint per test, settings cache cleared on both sides — the twin
    still runs assert_sovereign, which is the point of the twin."""
    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", f"localhost,{_EP}")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fetcher() -> InMemoryExternalFetcher:
    f = InMemoryExternalFetcher()
    for k in _KEYS:
        f.seed(_EP, "b", k, b"x")
    return f


async def _page(f: InMemoryExternalFetcher, after: str, limit: int) -> list[str]:
    return await f.list_keys(
        endpoint=_EP,
        bucket="b",
        prefix="sops/",
        access_key="a",
        secret_key="s",
        secure=False,
        limit=limit,
        start_after=after,
    )


@pytest.mark.asyncio
async def test_paging_walks_the_whole_prefix_without_gaps_or_repeats(fetcher: InMemoryExternalFetcher) -> None:
    seen: list[str] = []
    after = ""
    while True:
        page = await _page(fetcher, after, 5)
        if not page:
            break
        seen.extend(page)
        after = page[-1]
    assert seen == _KEYS, "paging lost, duplicated or reordered keys"


@pytest.mark.asyncio
async def test_a_page_starts_strictly_after_the_marker(fetcher: InMemoryExternalFetcher) -> None:
    first = await _page(fetcher, "", 5)
    second = await _page(fetcher, first[-1], 5)
    assert first[-1] not in second, "the marker key was returned twice"
    assert second[0] > first[-1]


@pytest.mark.asyncio
async def test_no_marker_starts_at_the_beginning(fetcher: InMemoryExternalFetcher) -> None:
    assert (await _page(fetcher, "", 3)) == _KEYS[:3]


@pytest.mark.asyncio
async def test_a_marker_past_the_end_returns_nothing(fetcher: InMemoryExternalFetcher) -> None:
    """How a caller knows it is finished, rather than looping forever on the last page."""
    assert (await _page(fetcher, "sops/zzz", 5)) == []
