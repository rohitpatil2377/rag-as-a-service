#!/usr/bin/env python3
"""Retrieval-stage metrics for a gold suite: Recall@k, MRR, nDCG@10 — and WHERE a miss happened.

WHY THIS EXISTS. Every number this repo currently reports is about the ANSWER, so a wrong answer
has three indistinguishable causes and no way to choose between them:

  * the evidence is not in the corpus at all, and no retriever could have found it;
  * the evidence is in the corpus and retrieval did not surface it;
  * retrieval surfaced it and the generator did not use it.

Those need different fixes and the answer score treats them alike. Measured consequence: the
AEGCL suite's "sustained EPS" lookup fails on BOTH the v1 and v2 paths, the figure is plainly in
the document, and nobody can currently say whether retrieval missed the passage or generation
dropped it. A separate item, `C03`, fails because the clause is in a document that was never
ingested — a corpus gap wearing the same clothes.

GOLD EVIDENCE IS DERIVED, NOT ANNOTATED, and that is a deliberate trade. The suites in
`eval/gold/` carry expected TEXT (`expected_answer`, `must_contain`), not chunk ids, and
hand-labelling evidence ids for 40 items across four document classes is work nobody has done.
So the gold evidence for an item is every chunk whose text contains its cues. That is
imperfect in a knowable direction: a chunk that merely mentions the words counts as relevant,
so Recall is OPTIMISTIC and nDCG is noisy where a cue is common. It is still decisive for the
only question being asked here — was the needed text anywhere in the ranked list — and it costs
no annotation. When real evidence ids exist, pass them and this will prefer them.

RUNS THE REAL RETRIEVER, through `wiring.retrieval_service()`, for the reason several analyses
on this branch had to be retracted: a reconstruction of a stage is not that stage.

    scripts/retrieval_metrics.py --kb zzgoal-abc --suite eval/gold/aegcl_corrigendum.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_NDCG_AT = 10
_DEFAULT_K = 20


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def _has(haystack: str, cues: list[str]) -> bool:
    folded = _norm(haystack)
    return all(_norm(c) in folded for c in cues)


@dataclass(slots=True)
class _Row:
    item: str
    cues: list[str]
    gold: int = 0  # chunks in the corpus carrying the cues
    rank: int | None = None  # 1-based rank of the first gold chunk retrieved
    retrieved_gold: int = 0
    present: bool = False  # every cue appears SOMEWHERE in the corpus
    verdict: str = ""


@dataclass(slots=True)
class _Totals:
    rows: list[_Row] = field(default_factory=list)

    def recall_at(self, k: int) -> float:
        """Share of ANSWERABLE items whose evidence appears in the top k.

        Items with no gold chunk are excluded rather than counted as failures: a corpus gap is
        not a retrieval miss, and averaging them together is what hides both."""
        answerable = [r for r in self.rows if r.gold]
        if not answerable:
            return float("nan")
        hit = sum(1 for r in answerable if r.rank is not None and r.rank <= k)
        return hit / len(answerable)

    def mrr(self) -> float:
        answerable = [r for r in self.rows if r.gold]
        if not answerable:
            return float("nan")
        return sum(1.0 / r.rank if r.rank else 0.0 for r in answerable) / len(answerable)

    def ndcg_at(self, k: int) -> float:
        """Binary-gain nDCG over the derived evidence set.

        Binary because the derived gold has no graded relevance to offer — a chunk either
        carries the cues or it does not — so anything finer would be invented precision."""
        answerable = [r for r in self.rows if r.gold]
        if not answerable:
            return float("nan")
        total = 0.0
        for r in answerable:
            dcg = 1.0 / math.log2(r.rank + 1) if r.rank and r.rank <= k else 0.0
            total += dcg  # ideal DCG with one relevant at rank 1 is 1.0
        return total / len(answerable)


async def _corpus_chunks(kb: str) -> list[tuple[str, str]]:
    """(chunk_id, text) for every chunk in the KB, through the shipped catalog."""
    from rag_clients.factory import get_catalog

    catalog = get_catalog()
    docs = await catalog.list_documents(kb)
    out: list[tuple[str, str]] = []
    for doc in docs:
        for chunk in await catalog.document_chunks(kb, doc.doc_id):
            out.append((chunk.chunk_id, chunk.text))
    return out


def _items_from(suite: dict[str, Any], *, lookups_only: bool) -> list[tuple[str, str, list[str]]]:
    """(item id, question, cues) for the assertions this measurement can speak to.

    LOOKUPS ONLY BY DEFAULT, and the exclusion is the point rather than a convenience. A
    checklist item belongs to ONE aggregate question — "give me the compliance details from
    this document" — so scoring forty of them against that question's top 20 passages asks
    whether twenty chunks cover forty requirements spread over 1,808. They cannot, and the
    resulting Recall says nothing about retrieval: measured, it reads 0.386 and every point of
    it is arithmetic, not quality. The aggregate path does not use passage top-k anyway; it uses
    the Phase-8 section pass, which `scripts/trace_gold_item.py` reports on.

    A lookup is the fair case: its own question, its own answer, and passage retrieval really is
    the mechanism that has to find it."""
    out: list[tuple[str, str, list[str]]] = []
    for q in suite.get("queries") or []:
        question = str(q["query"])
        if q.get("expected_answer"):
            out.append((f"lookup:{question[:38]}", question, [str(q["expected_answer"])]))
        if lookups_only:
            continue
        for item in q.get("checklist") or []:
            out.append((str(item["id"]), question, [str(c) for c in item["must_contain"]]))
    return out


async def run(kb: str, suite: dict[str, Any], suite_name: str, top_k: int, lookups_only: bool) -> int:
    from svc_control_plane import wiring

    from rag_schemas import RetrievalQuery

    items = _items_from(suite, lookups_only=lookups_only)
    print(f"\n  suite {suite_name} on {kb}: {len(items)} assertions, top_k={top_k}")

    chunks = await _corpus_chunks(kb)
    print(f"  corpus: {len(chunks)} chunks\n")

    service = wiring.retrieval_service()
    totals = _Totals()
    by_question: dict[str, list[tuple[str, float]]] = {}

    for item_id, question, cues in items:
        row = _Row(item=item_id, cues=cues)
        # TWO DIFFERENT QUESTIONS, and conflating them invents corpus gaps. "Is the text in the
        # corpus at all" is per-cue across every chunk; "is there a chunk that proves this item"
        # needs them together. A multi-cue item whose words are split over two chunks — "Video
        # Wall" in the heading chunk and "UL 62368" in the row below it — is present and simply
        # has no single-chunk witness, which is a chunking property, not a missing document.
        present = all(any(_norm(c) in _norm(t) for _cid, t in chunks) for c in cues)
        gold_ids = {cid for cid, text in chunks if _has(text, cues)}
        row.gold = len(gold_ids)
        row.present = present

        if question not in by_question:
            result = await service.retrieve(
                RetrievalQuery(tenant_id=kb, text=question, original_text=question, top_k=top_k)
            )
            by_question[question] = [(s.chunk_id, s.score) for s in result.spans]
        ranked = by_question[question]

        if not gold_ids and not present:
            row.verdict = "CORPUS GAP — a cue appears in no chunk at all"
        elif not gold_ids:
            row.verdict = (
                "SPLIT ACROSS CHUNKS — every cue is in the corpus, none in one chunk; "
                "a chunking property, not a retrieval or corpus failure"
            )
        else:
            for position, (cid, _score) in enumerate(ranked, start=1):
                if cid in gold_ids:
                    row.retrieved_gold += 1
                    if row.rank is None:
                        row.rank = position
            row.verdict = (
                f"retrieved at rank {row.rank}"
                if row.rank
                else f"RETRIEVAL MISS — {row.gold} chunk(s) carry it, none in the top {top_k}"
            )
        totals.rows.append(row)

    gaps = [r for r in totals.rows if not r.gold and not r.present]
    split = [r for r in totals.rows if not r.gold and r.present]
    misses = [r for r in totals.rows if r.gold and r.rank is None]
    print(f"  {'item':38} {'gold chunks':>12} {'rank':>6}  verdict")
    for r in totals.rows:
        print(f"  {r.item:38} {r.gold:>12} {r.rank or '-'!s:>6}  {r.verdict}")

    print(f"\n  Recall@{top_k} : {totals.recall_at(top_k):.3f}")
    print(f"  Recall@5  : {totals.recall_at(5):.3f}")
    print(f"  MRR       : {totals.mrr():.3f}")
    print(f"  nDCG@{_NDCG_AT}   : {totals.ndcg_at(_NDCG_AT):.3f}")
    print(
        f"\n  answerable {len(totals.rows) - len(gaps)} of {len(totals.rows)}; "
        f"corpus gaps {len(gaps)}; split across chunks {len(split)}; retrieval misses {len(misses)}"
    )
    if gaps:
        print(f"  corpus gaps: {[r.item for r in gaps]}")
    if split:
        print(f"  split across chunks: {[r.item for r in split]}")
    if misses:
        print(f"  retrieval misses: {[r.item for r in misses]}")
    print(
        "\n  An item that is neither a gap nor a miss and still fails the answer score is a "
        "GENERATION failure — the evidence reached the model.\n"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True, help="a THROWAWAY kb with the corpus ingested")
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=_DEFAULT_K)
    ap.add_argument(
        "--include-checklist",
        action="store_true",
        help="also score checklist items — see `_items_from`: their Recall is arithmetic, "
        "not retrieval quality, because forty of them share one aggregate question",
    )
    args = ap.parse_args()
    sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "services" / "control-plane" / "src")]

    import yaml

    suite = yaml.safe_load(args.suite.read_text())
    return asyncio.run(run(args.kb, suite, args.suite.name, args.top_k, not args.include_checklist))


if __name__ == "__main__":
    sys.exit(main())
