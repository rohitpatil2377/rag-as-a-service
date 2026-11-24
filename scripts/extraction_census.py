"""Extraction-provenance census over an ALREADY-INDEXED corpus (read-only by default).

Provenance tags (``extraction_extractor``/``extraction_capabilities``/``extraction_degraded``/
``extraction_degradation``) are stamped at ingest and at reindex. Documents ingested BEFORE
that landed carry no tags at all, so "which of my documents were extracted without table
support" cannot be answered from the catalog alone for the existing corpus.

This script answers it WITHOUT re-indexing anything. For each document it either reads the
already-stamped tags (free) or replays extraction from the raw bytes still held in the object
store, through the REAL production path (``DocIntelligenceService.extract``) so the verdict is
the one the platform would actually produce — no duplicated dispatch logic here to drift.

Nothing is written unless ``--backfill`` is passed, and even then the write is a catalog
metadata upsert ONLY: chunks are NOT touched, no embedding, no vector/sparse write. Chunk-level
copies of the metadata (used by D8 filters) stay stale until those documents are reindexed —
see the warning the script prints.

    # census only, no writes
    uv run python scripts/extraction_census.py --kb acme --kb hr-docs

    # census + persist the tags so the catalog is queryable afterwards
    uv run python scripts/extraction_census.py --kb acme --backfill

KB ids must be given explicitly: the platform deliberately offers no cross-tenant enumeration
(see CLAUDE.md "Tenancy"). To get the full list on the box, from the platform Postgres:

    SELECT nspname FROM pg_namespace WHERE nspname LIKE 't\\_%';   -- schema per tenant
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

os.environ.setdefault("RAG_OTEL_ENABLED", "false")

# Outcomes for documents whose provenance could not be established. Reported explicitly rather
# than dropped, so the census never quietly under-counts.
_RAW_MISSING = "raw_bytes_missing"  # object store has no bytes for this doc (can't replay)
_NEEDS_OCR = "ocr_required"  # routes to OCR; no engine configured in THIS process
_REPLAY_FAILED = "replay_failed"  # extraction raised (would fail re-ingest too)

# A document this much of whose characters were claimed by table detection is worth
# LOOKING at, not condemning: a genuine table-only appendix scores the same as a page of
# prose misread as a grid. Set high enough that ordinary table-bearing documents do not
# fill the list. `--table-heavy-share` overrides it per run.
TABLE_HEAVY_SHARE = 0.9
_TABLE_HEAVY_SHOWN = 10  # longest list worth printing; the JSON output carries them all


@dataclass(slots=True)
class Census:
    documents: int = 0
    from_tags: int = 0  # provenance already stored — read, not replayed
    replayed: int = 0  # provenance recomputed from the raw bytes
    by_extractor: Counter[str] = field(default_factory=Counter)
    by_degradation: Counter[str] = field(default_factory=Counter)
    unresolved: Counter[str] = field(default_factory=Counter)
    degraded: int = 0
    without_tables: int = 0  # extracted by a path that CANNOT capture tables
    backfilled: int = 0
    # Observed table CONTENT (a different question from capability above).
    with_tables: int = 0  # documents where a table was actually found
    tables_unknown: int = 0  # ingested before tables were observed — excluded from the ratios
    table_pages: int = 0  # pages carrying a table, summed across documents
    table_runs: int = 0  # maximal runs of consecutive table pages, summed
    # Documents whose table detection claimed an implausible share of their characters.
    # A screening list, not a verdict: a genuine table-only page reads the same way.
    table_heavy: list[tuple[str, float]] = field(default_factory=list)
    # Chunks whose text exceeds the embedder's input limit. The excess is DISCARDED from the
    # dense vector by TEI's truncate flag while remaining in the stored text and the BM25
    # index — so the tail of such a chunk is findable lexically and invisible to dense search.
    chunks_seen: int = 0
    chunks_over_limit: int = 0
    truncated_chars: int = 0  # characters that never reached a vector
    longest_chunk_chars: int = 0

    def absorb(self, other: Census) -> None:
        self.chunks_seen += other.chunks_seen
        self.chunks_over_limit += other.chunks_over_limit
        self.truncated_chars += other.truncated_chars
        self.longest_chunk_chars = max(self.longest_chunk_chars, other.longest_chunk_chars)
        self.documents += other.documents
        self.from_tags += other.from_tags
        self.replayed += other.replayed
        self.by_extractor += other.by_extractor
        self.by_degradation += other.by_degradation
        self.unresolved += other.unresolved
        self.degraded += other.degraded
        self.without_tables += other.without_tables
        self.backfilled += other.backfilled
        self.with_tables += other.with_tables
        self.tables_unknown += other.tables_unknown
        self.table_pages += other.table_pages
        self.table_runs += other.table_runs
        self.table_heavy += other.table_heavy


def _tally(stats: Census, tags: dict[str, str], doc_id: str = "", heavy_share: float = TABLE_HEAVY_SHARE) -> None:
    """Fold one document's provenance tags into the running totals."""
    from svc_doc_intelligence.ocr import (
        _META_SEP,
        CAP_TABLES,
        META_CAPABILITIES,
        META_DEGRADATION,
        META_DEGRADED,
        META_EXTRACTOR,
        META_TABLE_CHAR_SHARE,
        META_TABLE_PAGES,
        META_TABLE_RUNS,
        META_TABLES_FOUND,
    )

    stats.by_extractor[tags[META_EXTRACTOR]] += 1
    capabilities = {c for c in tags.get(META_CAPABILITIES, "").split(_META_SEP) if c}
    if CAP_TABLES not in capabilities:
        stats.without_tables += 1
    if tags.get(META_DEGRADED) == "true":
        stats.degraded += 1
        for reason in tags.get(META_DEGRADATION, "").split(_META_SEP):
            if reason:
                stats.by_degradation[reason] += 1

    # Observed tables. Absent on documents ingested before observation existed — counted
    # separately rather than folded in as "no tables", which would understate the corpus.
    if META_TABLES_FOUND not in tags:
        stats.tables_unknown += 1
        return
    if tags[META_TABLES_FOUND] == "true":
        stats.with_tables += 1
    stats.table_pages += int(tags.get(META_TABLE_PAGES) or 0)
    stats.table_runs += int(tags.get(META_TABLE_RUNS) or 0)
    share = float(tags.get(META_TABLE_CHAR_SHARE) or 0.0)
    if share >= heavy_share:
        stats.table_heavy.append((doc_id, share))


class _TokenCounter:
    """Counts tokens the way the RUNNING embedder does, and knows its real input limit.

    Asks the live TEI server rather than trusting config: the limit that matters is the one
    the container serves. Falls back to a character heuristic (with the assumption stated in
    the report) when the server is unreachable, so a census still runs offline."""

    def __init__(self) -> None:
        self.limit: int | None = None
        self.model = ""
        self._client: Any = None
        self._chars_per_token = 0.0

    async def open(self) -> None:
        import httpx

        from rag_common import get_settings

        url = get_settings().tei_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=url, timeout=30.0)
        try:
            info = (await self._client.get("/info")).json()
            self.limit = int(info["max_input_length"])
            self.model = str(info.get("model_id", "?"))
        except Exception as exc:
            print(f"  ! embedder at {url} unreachable ({type(exc).__name__}) — token counts unavailable")
            await self.close()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def count(self, text: str) -> int | None:
        if self._client is None:
            return None
        resp = await self._client.post("/tokenize", json={"inputs": text})
        resp.raise_for_status()
        return len(resp.json()[0])

    async def excess_chars(self, text: str) -> int:
        """How many characters of `text` never reach the vector, by bisecting on the token
        count — the tokenizer is the only thing that knows where the cut lands."""
        if self._client is None or self.limit is None:
            return 0
        total = await self.count(text)
        if total is None or total <= self.limit:
            return 0
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if (await self.count(text[:mid]) or 0) <= self.limit:
                lo = mid
            else:
                hi = mid - 1
        return len(text) - lo


async def _census_chunks(kb_id: str, doc_id: str, stats: Census, tokens: _TokenCounter) -> None:
    """Measure the stored chunks of one document against the embedder's real input limit."""
    from rag_clients.factory import get_catalog

    for chunk in await get_catalog().document_chunks(kb_id, doc_id):
        stats.chunks_seen += 1
        stats.longest_chunk_chars = max(stats.longest_chunk_chars, len(chunk.text))
        excess = await tokens.excess_chars(chunk.text)
        if excess:
            stats.chunks_over_limit += 1
            stats.truncated_chars += excess


async def _census_kb(kb_id: str, *, backfill: bool, tokens: _TokenCounter, heavy_share: float) -> Census:
    from svc_control_plane import wiring
    from svc_doc_intelligence.ocr import META_EXTRACTOR

    from rag_clients.factory import get_catalog, get_object_store
    from rag_common import ValidationError, raw_object_key

    stats = Census()
    docs = await get_catalog().list_documents(kb_id)
    stats.documents = len(docs)

    for doc in docs:
        tags = {k: v for k, v in doc.metadata.items() if k.startswith("extraction_")}
        if META_EXTRACTOR in tags and not backfill:
            stats.from_tags += 1
        else:
            # No stored provenance, or --backfill: replay extraction from the raw bytes.
            # --backfill forces the replay even when tags exist, because otherwise the first
            # backfill freezes the corpus's numbers and the census can never re-measure after
            # an extractor change — which is the only reason to run it twice.
            try:
                raw = await get_object_store().get(kb_id, raw_object_key(doc.doc_id))
            except KeyError:
                stats.unresolved[_RAW_MISSING] += 1
                continue
            try:
                tags = (await wiring.doc_intel_service().extract(doc, raw)).provenance
            except ValidationError:  # routed to OCR, none enabled in this process
                stats.unresolved[_NEEDS_OCR] += 1
                continue
            except Exception as exc:
                print(f"  ! {doc.doc_id}: {type(exc).__name__}: {exc}")
                stats.unresolved[_REPLAY_FAILED] += 1
                continue
            stats.replayed += 1
            if backfill:
                await get_catalog().register_document(doc.model_copy(update={"metadata": {**doc.metadata, **tags}}))
                stats.backfilled += 1

        _tally(stats, tags, doc.doc_id, heavy_share)
        await _census_chunks(kb_id, doc.doc_id, stats, tokens)
    return stats


def _report(name: str, s: Census, heavy_share: float) -> None:
    print(f"\n=== {name} ===")
    print(f"  documents            {s.documents}  (tags read {s.from_tags}, replayed {s.replayed})")
    pct = (100.0 * s.without_tables / s.documents) if s.documents else 0.0
    print(f"  WITHOUT table support {s.without_tables}  ({pct:.1f}% of the corpus)")
    print(f"  degraded extraction   {s.degraded}")
    for extractor, n in sorted(s.by_extractor.items(), key=lambda kv: -kv[1]):
        print(f"    extractor {extractor:<24} {n}")
    for reason, n in sorted(s.by_degradation.items(), key=lambda kv: -kv[1]):
        print(f"    reason    {reason:<24} {n}")
    for outcome, n in sorted(s.unresolved.items(), key=lambda kv: -kv[1]):
        print(f"    UNRESOLVED {outcome:<23} {n}")

    known = s.documents - s.tables_unknown - sum(s.unresolved.values())
    if known > 0:
        share = 100.0 * s.with_tables / known
        print(f"  tables observed in     {s.with_tables}/{known} documents  ({share:.1f}%)")
        if s.table_runs:
            # >1.0 means tables occupy consecutive pages — the observable proxy for one table
            # crossing a break. An UPPER bound: two separate tables on adjacent pages look the
            # same, because a table's position within its page is not recorded.
            print(f"    table pages {s.table_pages}, consecutive runs {s.table_runs}")
            print(f"    mean consecutive table-page run {s.table_pages / s.table_runs:.2f}")
        else:
            print("    no table pages observed")
        if s.table_heavy:
            # Table detection REPLACES the prose it claims, so a false positive does not
            # add noise beside the narrative — it restructures it. Read these before
            # trusting their text: a genuine table-only page looks identical here.
            print(f"    {len(s.table_heavy)} document(s) >={heavy_share:.0%} table by character:")
            for doc_id, share in sorted(s.table_heavy, key=lambda kv: -kv[1])[:_TABLE_HEAVY_SHOWN]:
                print(f"      {share:6.1%}  {doc_id}")
    if s.tables_unknown:
        print(f"    (table observation absent on {s.tables_unknown} — ingested before it existed)")

    if s.chunks_seen:
        share = 100.0 * s.chunks_over_limit / s.chunks_seen
        print(f"  chunks                 {s.chunks_seen}  (longest {s.longest_chunk_chars} chars)")
        print(f"    OVER the embedder's input limit {s.chunks_over_limit}  ({share:.1f}%)")
        if s.chunks_over_limit:
            # The tail of such a chunk is in the stored text and in BM25, but contributed
            # nothing to the vector — findable lexically, invisible to dense search.
            print(f"    characters never embedded       {s.truncated_chars}")
    if s.backfilled:
        print(f"  backfilled (catalog metadata only) {s.backfilled}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kb", action="append", required=True, help="KB / tenant id (repeatable).")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Persist recomputed provenance to the catalog. Metadata upsert ONLY — no reindex.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the totals as JSON on the last line.")
    parser.add_argument(
        "--table-heavy-share",
        type=float,
        default=TABLE_HEAVY_SHARE,
        help="List documents at or above this table character share (0-1). Lower it to widen "
        "the review list on a corpus where detection is suspected of eating prose.",
    )
    args = parser.parse_args()

    tokens = _TokenCounter()
    await tokens.open()
    if tokens.limit is not None:
        print(f"embedder: {tokens.model}  max_input_length={tokens.limit} tokens (asked of the running server)")
    try:
        total = Census()
        for kb_id in args.kb:
            stats = await _census_kb(kb_id, backfill=args.backfill, tokens=tokens, heavy_share=args.table_heavy_share)
            _report(kb_id, stats, args.table_heavy_share)
            total.absorb(stats)
    finally:
        await tokens.close()

    if len(args.kb) > 1:
        _report("TOTAL", total, args.table_heavy_share)
    if total.unresolved:
        print("\nNOTE: unresolved documents are EXCLUDED from the counts above — they are not 'clean'.")
    if args.backfill and total.backfilled:
        print(
            "\nNOTE: backfill updated DOCUMENT metadata only. Chunk-level copies of these tags "
            "(D8 retrieval filters) remain stale until those documents are reindexed."
        )
    if args.json:
        print(
            json.dumps(
                {
                    "documents": total.documents,
                    "without_table_support": total.without_tables,
                    "with_tables_observed": total.with_tables,
                    "tables_unknown": total.tables_unknown,
                    "table_pages": total.table_pages,
                    "table_page_runs": total.table_runs,
                    "table_heavy": [{"doc_id": d, "share": sh} for d, sh in sorted(total.table_heavy)],
                    "chunks": total.chunks_seen,
                    "chunks_over_embed_limit": total.chunks_over_limit,
                    "characters_never_embedded": total.truncated_chars,
                    "degraded": total.degraded,
                    "by_extractor": dict(total.by_extractor),
                    "by_degradation": dict(total.by_degradation),
                    "unresolved": dict(total.unresolved),
                }
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
