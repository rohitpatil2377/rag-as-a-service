#!/usr/bin/env python3
"""Re-run RAG v2 Phases 3-4 over a KB, without re-chunking or re-indexing it.

The plan's Phase-12 item, verbatim: "when prompts or Gemma version change, bump
``profile_version`` / ``model_version`` and provide a CLI to re-run Phases 3-4 only."

WHY A SEPARATE PATH AND NOT ``reindex``. A better profile prompt, a corrected facet instruction
or a larger model does not change the corpus — the bytes, the extraction and the chunks are all
unchanged. Re-indexing a large KB to pick up a prompt edit means re-embedding every chunk in it,
which on a real tender corpus is hours of GPU for an artifact that lives in two Postgres columns.
This re-derives those columns and nothing else.

WHAT IT REFUSES. If the structure tree re-derives DIFFERENTLY — a different extractor, a changed
layout parser, the Phase-2 heading fix — the stored chunks still carry the old ``section_id`` and
re-enriching would leave Phase 8 selecting sections whose chunks Phase 9 cannot find, with every
count reading plausibly. Such a document is reported ``structure_changed`` and left alone; it
needs ``reindex``, which this tool deliberately does not do on its own.

WHAT STAYS STALE, because it is better said than discovered: a section's summary is also indexed
as a ``section``-level chunk, so after a re-enrichment the catalog holds the new summary and the
INDEX still holds the old. Phase 8 selects on the catalog, so selection is current immediately;
Phase 9 maps over the document's own text and excludes the derived levels, so extraction is
unaffected; what remains stale is the summary as a retrieval target. Reindex when that matters.

Usage:

    # what WOULD be re-enriched, and why
    RAG_V2_INGEST=true uv run python scripts/reenrich.py --kb acme__main --stale --dry-run

    # bring every document with a card older than this build's PROFILE_VERSION forward
    RAG_V2_INGEST=true uv run python scripts/reenrich.py --kb acme__main --stale

    # one document, whatever its version
    RAG_V2_INGEST=true uv run python scripts/reenrich.py --kb acme__main --doc tender.pdf

``RAG_V2_INGEST=true`` IS REQUIRED and the tool says so rather than working anyway: both
phases are gated on it (``_document_profile`` and ``_section_enrichment`` return nothing with it
off), so a run without it would report every document re-enriched and write nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter

from rag_clients.factory import stale_profiles
from rag_common import get_settings, tenant_scope
from rag_core.doc_profile import PROFILE_VERSION


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kb", required=True, help="the KB (tenant) id")
    p.add_argument("--doc", action="append", default=[], help="a document id; repeatable. Default: see --stale/--all")
    p.add_argument(
        "--stale",
        action="store_true",
        help=f"every document whose card predates PROFILE_VERSION={PROFILE_VERSION}, including cardless ones",
    )
    p.add_argument("--all", action="store_true", help="every document in the KB, whatever its version")
    p.add_argument("--dry-run", action="store_true", help="list what would run and exit")
    p.add_argument("--limit", type=int, default=0, help="stop after N documents (0 = no limit)")
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="documents in flight. Each one makes several model calls, which are ALREADY "
        "parallel inside a document (RAG_SECTION_ENRICH_CONCURRENCY) — so raise this only "
        "after checking the endpoint has room, or it is a queue with extra steps.",
    )
    p.add_argument("--actor", default="reenrich-cli", help="recorded on the structured logs")
    return p


async def _targets(args: argparse.Namespace) -> list[str]:
    from rag_clients.factory import get_catalog

    if args.doc:
        return list(dict.fromkeys(args.doc))
    if args.stale:
        return await stale_profiles(args.kb, below=PROFILE_VERSION)
    if args.all:
        return [d.doc_id for d in await get_catalog().list_documents(args.kb)]
    raise SystemExit("choose what to re-enrich: --doc, --stale or --all")


async def main() -> int:
    args = _parser().parse_args()
    if not get_settings().v2_ingest:
        print(
            "RAG_V2_INGEST is off. Phases 3 and 4 are gated on it, so this run would report\n"
            "every document re-enriched and write nothing. Set it and try again.",
            file=sys.stderr,
        )
        return 2

    from svc_control_plane import pipeline

    with tenant_scope(args.kb):
        targets = await _targets(args)
    if args.limit > 0:
        targets = targets[: args.limit]
    if not targets:
        print("nothing to do")
        return 0

    print(f"{len(targets)} document(s) in {args.kb}")
    if args.dry_run:
        for doc_id in targets:
            print(f"  would re-enrich {doc_id}")
        return 0

    gate = asyncio.Semaphore(max(1, args.concurrency))

    async def run(doc_id: str) -> object:
        async with gate, tenant_scope(args.kb):
            return await pipeline.reenrich_document(args.kb, doc_id, actor=args.actor)

    outcomes = await asyncio.gather(*(run(doc_id) for doc_id in targets))

    tally: Counter[str] = Counter()
    for outcome in outcomes:
        tally[outcome.status] += 1  # type: ignore[attr-defined]
        if outcome.status != "reenriched":  # type: ignore[attr-defined]
            print(f"  {outcome.status:18} {outcome.doc_id}")  # type: ignore[attr-defined]
    print("\n" + "  ".join(f"{status}={count}" for status, count in sorted(tally.items())))
    if tally["structure_changed"]:
        print(
            f"\n{tally['structure_changed']} document(s) re-derived a DIFFERENT structure and were left\n"
            "untouched. Their indexed chunks carry the old section ids, so enriching them here would\n"
            "leave the query path selecting sections whose chunks it cannot find. Run a reindex on\n"
            "those documents instead."
        )
    # A non-zero exit for the refusals only: a document with no structure is an ordinary outcome
    # on most of a real corpus, and failing the command for it would make this unusable in CI.
    return 1 if tally["structure_changed"] else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
