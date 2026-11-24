"""Neighbour expansion (contiguity recall) — pure logic, no I/O.

Retrieval ranks each passage on its own merits, which silently fails a whole class of question:
**anything whose answer is a list longer than one passage.**

Measured on a real HR handbook. The six core values occupy ~1,900 characters, so they span three
passages. The FIRST holds the heading "Core Values" and the phrase "six simple ingredients", and
ranks well because it shares vocabulary with the question. The SECOND begins "…assets to ensure
there is Luxury in Hard work. 3. Continuous Knowledge Enhancement…" — it holds half the answer and
shares almost no vocabulary with the question, so it ranks below passages that merely mention
values in passing. Verified: one value appeared in NO retrieved passage at top_k=5 or top_k=10.
Raising top_k does not help, because the problem is relevance ranking, not depth.

The fix is structural rather than statistical: a list is CONTIGUOUS in the document, so retrieving
any part of it should bring its neighbours. This module answers one question — *given the passages
retrieval chose, which adjacent passages should come with them?* — and nothing else. Fetching them,
checking the caller may see them, and merging them is the caller's job.
"""

from __future__ import annotations

from collections.abc import Iterable


def neighbour_ids(ordered_ids: list[str], anchors: Iterable[str], *, window: int) -> list[str]:
    """Ids within ``window`` positions of any anchor, in document order, anchors excluded.

    ``ordered_ids`` is one document's passages in READING order — the order matters entirely, and
    is why the caller must read them from the catalog rather than from a search result.

    Returns document order, not anchor order: the neighbours of two nearby anchors overlap, and a
    reader (or a model) should receive the passage before the anchor before the one after it."""
    if window <= 0:
        return []
    positions = {cid: i for i, cid in enumerate(ordered_ids)}
    wanted: set[int] = set()
    for anchor in anchors:
        at = positions.get(anchor)
        if at is None:
            continue  # an anchor from another document, or one deleted since it was indexed
        lo, hi = max(0, at - window), min(len(ordered_ids) - 1, at + window)
        wanted.update(range(lo, hi + 1))
    anchored = {positions[a] for a in anchors if a in positions}
    return [ordered_ids[i] for i in sorted(wanted - anchored)]


def anchors_by_document(spans: Iterable[tuple[str, str]], *, limit: int) -> dict[str, list[str]]:
    """Group ``(doc_id, chunk_id)`` into ``{doc_id: [chunk_id, …]}`` for the top ``limit`` spans.

    Bounded on purpose. Expanding every span of a 20-span result would fetch 20 documents' passage
    lists and could double the prompt; expanding the few the reranker ranked highest gets the
    contiguity benefit at a fraction of the cost."""
    grouped: dict[str, list[str]] = {}
    for doc_id, chunk_id in list(spans)[: max(0, limit)]:
        grouped.setdefault(doc_id, []).append(chunk_id)
    return grouped
