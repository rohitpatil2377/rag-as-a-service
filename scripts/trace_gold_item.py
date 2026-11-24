#!/usr/bin/env python3
"""Follow one gold requirement through the RAG v2 stages and report where it disappears.

WHY THIS EXISTS. When an item from `eval/gold/*.yaml` is absent from an aggregate answer, the
answer says nothing about which stage lost it, and the aggregate counters cannot tell the cases
apart: a section that was never selected, a map window the text fell outside of, a model that
read the text and did not extract it, a quote the validator rejected, an item past the
per-section cap, and a reduce that did not render it all look identical from the outside — one
missing bullet.

Guessing between them is expensive and, on this branch, repeatedly wrong. Three separate
analyses were invalidated because a scratch script reconstructed a pipeline stage slightly
differently from the pipeline:

  * `LayoutSection.text` was read where Phase 4 uses `section_prompt_text` — tables counted
    twice, container nodes empty, and a run that reported "0 enrichable sections";
  * a window sweep was run on that same raw text when the map actually reads the section's
    stored CHUNKS joined, making every size in the conclusion wrong;
  * a profile probe passed `max_tokens=900` against a shipped default of 1400 and produced
    truncated JSON that looked like a prompt regression.

So THIS SCRIPT NEVER REBUILDS A STAGE. It calls the pipeline's own `_route_v2`,
`_select_sections`, `_map_inputs`, `_mapped_items` and `_reduce_batches` against a real KB, and
reads what they actually produce. If a stage changes, this follows it, because it is the same
code path — that property is the point of the tool, not an implementation detail.

IT NEEDS A REAL KB, with the document ingested through the normal path and the v2 flags on. Use
a throwaway tenant: this issues real model calls and writes query-log and audit rows into
whatever KB it is pointed at.

    scripts/trace_gold_item.py --kb zzgoal-abc --suite eval/gold/aegcl_corrigendum.yaml --item C31
    scripts/trace_gold_item.py --kb zzgoal-abc --query "give me the compliance details" --cue EPEAT
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

_EXIT_LOST = 2  # the item was lost somewhere; 0 means it reached the answer


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def _has(haystack: str, cues: list[str]) -> bool:
    folded = _norm(haystack)
    return all(_norm(c) in folded for c in cues)


def _say(stage: str, ok: bool, detail: str) -> None:
    print(f"  {'OK  ' if ok else 'LOST'}  {stage:<22} {detail}")


async def trace(kb: str, question: str, cues: list[str], top_k: int, doc_ids: list[str]) -> int:  # noqa: PLR0911
    """Walk the stages in order and stop at the first one that does not carry the cues."""
    from svc_control_plane import pipeline, wiring

    from rag_common import get_settings
    from rag_core.mapreduce import Coverage, ReduceInput
    from rag_core.section_enrich import section_windows
    from rag_schemas import RetrievalQuery

    settings = get_settings()
    print(f"\n  kb={kb}  cues={cues}\n  question={question!r}\n")
    if not settings.v2_generate:
        print("  RAG_V2_GENERATE is off — this traces the v2 path and there is none.\n")
        return 1

    # NAMING THE DOCUMENT IS NOT OPTIONAL for the v2 section pass, and it is the first thing a
    # trace gets wrong. `_select_sections` computes `scope = decision.doc_ids or query.doc_ids`
    # and returns None when both are empty — deliberately, so a question about "this document"
    # is never answered from documents the asker never mentioned. A query with no doc_ids
    # therefore skips Phases 8 and 9 entirely and is served by passage retrieval instead, which
    # looks from the outside exactly like a section pass that selected nothing.
    query = RetrievalQuery(tenant_id=kb, text=question, original_text=question, top_k=top_k, doc_ids=tuple(doc_ids))
    if not doc_ids:
        print("  note: no --doc-id given, so the v2 section pass will not run (see _select_sections).\n")

    decision = await pipeline._route_v2(kb, query)
    if decision is None:
        print("  RAG_V2_ROUTER is off — no decision, so no section selection.\n")
        return 1
    _say("router", True, f"intent={decision.intent} facets={len(decision.facets)}")

    selection = await pipeline._select_sections(kb, query, decision)
    if selection is None:
        _say(
            "selection",
            False,
            "the section pass did not run at all — `_select_sections` returned None. Either "
            "RAG_V2_RETRIEVAL is off, the intent's strategy has no section pass, or (most "
            "often) no document was named: pass --doc-id",
        )
        return _EXIT_LOST
    if not selection.selected:
        _say(
            "selection",
            False,
            f"the section pass ran over {selection.considered} candidates and chose none — "
            f"look at RAG_V2_SECTION_THRESHOLD and the facets the router matched",
        )
        return _EXIT_LOST
    _say("selection", True, f"{len(selection.selected)} of {selection.considered} sections")

    prepared = await pipeline._map_inputs(kb, selection)
    if prepared is None:
        _say("map inputs", False, "no chunk text for any selected section")
        return _EXIT_LOST
    cards, _texts, jobs = prepared

    # THE MAP'S REAL INPUT. Not the layout text and not the prompt text — the chunks joined, as
    # `_map_inputs` builds them. Finding the cue here is what tells you the model saw it.
    holding = [j for j in jobs if _has(j.text, cues)]
    if not holding:
        anywhere = [s.section_id for s in selection.selected]
        _say(
            "map input",
            False,
            f"cues not in the text of any of the {len(jobs)} map jobs "
            f"(selected sections: {len(anywhere)}) — the section was selected but its CHUNKS "
            f"do not carry this text; look at chunking or at whether it was indexed",
        )
        return _EXIT_LOST
    job = holding[0]
    wins = section_windows(job.text, settings.v2_map_section_chars, settings.v2_map_max_windows)
    in_window = [i for i, w in enumerate(wins) if _has(w, cues)]
    needed = -(-len(job.text) // settings.v2_map_section_chars)
    _say(
        "map input",
        True,
        f"section {job.section_id[-12:]} text={len(job.text)} chars -> {len(wins)} windows kept of ~{needed} needed",
    )
    if not in_window:
        _say(
            "map window",
            False,
            f"the cues sit past window {settings.v2_map_max_windows} — raise "
            f"RAG_V2_MAP_MAX_WINDOWS or lower RAG_V2_MAP_SECTION_CHARS",
        )
        return _EXIT_LOST
    _say("map window", True, f"cues in window {in_window[0]} of {len(wins)}")

    gateway = await wiring.resolve_tenant_gateway(kb)
    mapped = await _trace_map(kb, jobs, cards, decision, question, cues, gateway, settings)
    if mapped is None:
        return _EXIT_LOST
    item, items, report, cached = mapped

    data = ReduceInput(
        query=question,
        profiles="",
        items=tuple(items),
        coverage=Coverage(
            used=report.sections_contributed + cached,
            selected=len(selection.selected),
            total=selection.considered,
            partial=report.sections_truncated,
        ),
        answer_lang="en",
    )
    return await _trace_reduce(data, cards, item, cues, gateway)


async def _trace_map(
    kb: str,
    jobs: list[Any],
    cards: dict[str, Any],
    decision: Any,
    question: str,
    cues: list[str],
    gateway: Any,
    settings: Any,
) -> tuple[Any, list[Any], Any, int] | None:
    """Run the real map over every selected section and find the item carrying the cues.

    All the sections, not just the one holding the cues: the per-section counters that explain a
    miss — quotes rejected, items past the cap — are only meaningful against the whole run, and
    mapping one section in isolation is how a scratch probe starts drifting from the pipeline."""
    from svc_control_plane import pipeline

    facet_ids = sorted({f for _d, f in decision.facets} | {f for c in cards.values() for f in c.facet_ids})
    items, report, cached = await pipeline._mapped_items(
        kb, jobs, question=question, facet_ids=facet_ids, gateway=gateway, settings=settings
    )
    over = getattr(report, "dropped_over_cap", 0)
    _say(
        "map",
        True,
        f"{len(items)} items from {report.sections_mapped} sections "
        f"(cached {cached}, failed {report.sections_failed}, unquoted {report.dropped_unquoted}, "
        f"over cap {over})",
    )
    for q in getattr(report, "dropped_quotes", ())[:4]:
        print(f"        rejected quote: {q[:110]!r}")
    carrying = [i for i in items if _has(f"{i.statement} {i.quote}", cues)]
    if not carrying:
        extra = ""
        if report.dropped_unquoted:
            extra = (
                f" — {report.dropped_unquoted} item(s) were dropped for an unverifiable quote, "
                f"so check whether this was one of them"
            )
        if over:
            extra += f" — {over} item(s) exceeded the per-section cap and were discarded"
        _say("map extraction", False, f"the model was shown the text and returned no item carrying it{extra}")
        return None
    item = carrying[0]
    _say("map extraction", True, f"{item.item_id} facet={item.facet_id!r} :: {item.statement[:70]}")
    return item, items, report, cached


async def _trace_reduce(data: Any, cards: dict[str, Any], item: Any, cues: list[str], gateway: Any) -> int:
    """The last two stages: which batch the item lands in, and whether its wording survives.

    Split out because the two halves fail for unrelated reasons — batching is arithmetic over
    budgets, rendering is the model choosing what to write — and because a trace that reads as
    one long function is the kind nobody extends."""
    from svc_control_plane import pipeline

    from rag_common import get_settings
    from rag_core.mapreduce import build_reduce_prompt

    settings = get_settings()
    crumbs = {sid: c.breadcrumb for sid, c in cards.items()}
    batches = pipeline._reduce_batches(data, crumbs, max_tokens=settings.v2_reduce_max_tokens)
    mine = [(n, f, b) for n, (f, b) in enumerate(batches) if any(i.item_id == item.item_id for i in b)]
    if not mine:
        _say("reduce batching", False, "the item is in no batch — it was grouped out before the call")
        return _EXIT_LOST
    n, facet, batch = mine[0]
    _say("reduce batching", True, f"batch {n + 1} of {len(batches)}, facet {facet!r}, {len(batch)} items")

    out = await gateway.complete(
        build_reduce_prompt(data, facet, batch, crumbs), max_tokens=settings.v2_reduce_max_tokens
    )
    if item.item_id not in out:
        _say("reduce", False, f"the batch was rendered ({len(out)} chars) and {item.item_id} is not cited in it")
        return _EXIT_LOST
    if not _has(out, cues):
        _say(
            "reduce",
            False,
            f"{item.item_id} IS cited but the cues are not in the text — the bullet merged this "
            f"item (rule 4) and paraphrased the distinctive wording away",
        )
        return _EXIT_LOST
    _say("reduce", True, f"cues present and {item.item_id} cited in {len(out)} chars")
    print("\n  the item survives every stage traced here.\n")
    return 0


def _cues_for(suite_path: Path, item_id: str) -> tuple[str, list[str]]:
    import yaml

    data = yaml.safe_load(suite_path.read_text())
    for q in data.get("queries") or []:
        for item in q.get("checklist") or []:
            if str(item.get("id")) == item_id:
                return str(q["query"]), [str(c) for c in item["must_contain"]]
    raise SystemExit(f"{item_id} is not in {suite_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True, help="a THROWAWAY kb with the document ingested")
    ap.add_argument("--suite", type=Path, help="gold suite to take the query and cues from")
    ap.add_argument("--item", help="checklist id within --suite, e.g. C31")
    ap.add_argument("--query", help="the question, when not using --suite")
    ap.add_argument("--cue", action="append", default=[], help="text that must survive; repeatable")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument(
        "--doc-id",
        action="append",
        default=[],
        help="scope the question to these documents; REQUIRED for the v2 section pass",
    )
    args = ap.parse_args()

    if args.suite and args.item:
        question, cues = _cues_for(args.suite, args.item)
    elif args.query and args.cue:
        question, cues = args.query, args.cue
    else:
        raise SystemExit("give either --suite with --item, or --query with at least one --cue")

    sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "services" / "control-plane" / "src")]
    return asyncio.run(trace(args.kb, question, cues, args.top_k, args.doc_id))


if __name__ == "__main__":
    sys.exit(main())
