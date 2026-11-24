#!/usr/bin/env python3
"""What the map's quote rule throws away, and whether throwing it away was right.

WHY THIS EXISTS. `parse_map_items` drops any extracted item whose `quote` is not a substring of
the section text after `quote_key` normalisation, and `MapReport.dropped_unquoted` counts the
drops. A COUNT CANNOT BE ACTED ON, because two opposite things produce the same number:

  * the model PARAPHRASED, so the item asserts something the document does not say in those
    words and the rule is doing exactly its job — an answer built from it would be unciteable;
  * the model quoted real document text and edited its EDGES, or the extractor and the model
    disagree about a dash, a non-breaking space or a curly apostrophe — in which case the rule
    is discarding real evidence, silently, with the quote.

The first needs no fix. The second costs requirements out of the answer and is fixed by widening
the normalisation, not by weakening the rule. `MapParse.dropped_quotes` keeps a bounded sample
precisely so this can be asked; this script asks it.

THE RATE IS MEASURED ON THE REAL MAP PASS, through `pipeline._mapped_items`, because the obvious
cheap substitute is wrong in three ways at once and has already been measured wrong on this
branch: a scratch probe read `section_prompt_text` sliced at `v2_map_section_chars`, where
the pipeline maps the section's CHUNKS joined, split by `section_windows`, with an EMPTY profile
block. It reported 0.5%. Nothing about that number describes production.

HOW THE FULL POPULATION IS RECOVERED. `MapParse.dropped_quotes` is capped at five per parse so a
dataclass on the latency path does not carry a log around, which is right for production and too
few to classify. So this records each real call's `(reply, section_text)` and re-applies the
rule's own predicate — `quote_key(quote) not in quote_key(section_text)`, the shipped function —
over every entry the reply contained. That is a re-derivation of ONE documented predicate, not a
reconstruction of a stage, and it is checked: the re-derived count must equal the shipped
counter, and the script says so either way. If they ever disagree, believe the counter.

    scripts/quote_rejections.py --kb zzgoal-abc --doc-id corrigendum-5.pdf

TURN THE MAP CACHE OFF for a repeat (`RAG_V2_MAP_CACHE_ENABLED=false`). A cached section
returns its items with an empty report, so a second run on the same KB and question measures a
rate of zero over zero.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: A run of this many normalised characters of the quote appearing verbatim in the section is
#: taken as "the model quoted a real span and edited its edges". Long enough that an incidental
#: phrase ("shall be provided by the bidder") cannot reach it.
_RUN_FLOOR = 40

#: How the model writes an elision. Measured as the single largest cause of a rejection on the
#: real corrigendum, which is why it is a class of its own rather than a kind of near-miss.
_ELLIPSIS = re.compile(r"\.{2,}|\u2026")

#: The shortest fragment of an elided quote that is allowed to carry weight. Below this a
#: fragment matches somewhere in any long section by accident ("and", "the RAM"), so counting it
#: as verbatim evidence would turn the check into a formality.
_FRAGMENT_FLOOR = 4

#: Each half of a welded quote must be at least this long to count as evidence. Three
#: characters of a bullet item ('PCI', 'SOX') appear everywhere and prove nothing.
_WELD_FLOOR = 8

#: Characters the extractor and the model routinely disagree about. Folding these is the
#: cheapest candidate fix, so the classes are ordered to show how much it would buy.
_PUNCT = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u00a0": " ",
    "\u00ad": "",
    "\u2026": "...",
}


@dataclass(slots=True)
class _Call:
    """One real map call: what the model replied and what it was replying about."""

    section_id: str
    reply: str
    section_text: str
    facet_ids: tuple[str, ...]
    kept: int = 0
    dropped: int = 0


@dataclass(slots=True)
class _Rejection:
    #: The CALL the quote was judged against, not just the section: a long section is mapped in
    #: several windows and a quote verbatim in window 2 is absent from window 1, so classifying
    #: against the wrong one of its windows would report real evidence as a paraphrase.
    call: _Call
    quote: str
    statement: str
    kind: str = ""


@dataclass(slots=True)
class _Tally:
    calls: list[_Call] = field(default_factory=list)
    rejections: list[_Rejection] = field(default_factory=list)
    #: Rejections the shipped parse counted, for the agreement check.
    shipped_dropped: int = 0
    shipped_kept: int = 0


def _fold_punct(text: str) -> str:
    from rag_core.graph import quote_key

    for raw, plain in _PUNCT.items():
        text = text.replace(raw, plain)
    return quote_key(unicodedata.normalize("NFKC", text))


def _alnum(text: str) -> str:
    from rag_core.graph import quote_key

    return re.sub(r"[^a-z0-9]+", "", quote_key(text))


def _has_long_run(needle: str, haystack: str) -> bool:
    """Whether any `_RUN_FLOOR`-character run of `needle` appears in `haystack`.

    Checked at the floor only, not at every longer length: the question is whether a real span
    is in there at all, and sweeping downwards from the full quote turns a classification into
    a quadratic scan over a 12,000-character section."""
    if len(needle) < _RUN_FLOOR:
        return False
    return any(needle[i : i + _RUN_FLOOR] in haystack for i in range(len(needle) - _RUN_FLOOR + 1))


def _elided_in_order(quote: str, haystack: str) -> bool:
    """Whether an elided quote's fragments are all verbatim in the section, IN ORDER.

    Order matters and is cheap to check: fragments found in document order are a span the model
    shortened, where fragments found out of order are two unrelated clauses welded together,
    which is the thing the quote rule exists to stop. Fragments under `_FRAGMENT_FLOOR`
    characters are skipped rather than failing the quote — they carry no evidence either way."""
    from rag_core.graph import quote_key

    at = 0
    for raw in _ELLIPSIS.split(quote):
        fragment = quote_key(raw).strip()
        if len(fragment) < _FRAGMENT_FLOOR:
            continue
        found = haystack.find(fragment, at)
        if found < 0:
            return False
        at = found + len(fragment)
    return True


def _welded_spans(quote: str, haystack: str) -> bool:
    """Whether the quote is a head and a tail that are each verbatim and together span it.

    This is the list case, and it is NOT a paraphrase: the document carries a stem, a colon
    and a bulleted item, and the model wrote them as one sentence. Both halves really are in
    the section, so the drop costs evidence exactly as an elision does. Measured by binary
    search on the longest verbatim prefix and suffix, which is logarithmic in the quote rather
    than a scan over the section."""

    def longest(fits: Callable[[int], bool]) -> int:
        lo, hi = 0, len(quote)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if fits(mid) else (lo, mid - 1)
        return lo

    prefix = longest(lambda n: quote[:n] in haystack)
    suffix = longest(lambda n: quote[len(quote) - n :] in haystack)
    return prefix >= _WELD_FLOOR and suffix >= _WELD_FLOOR and prefix + suffix >= len(quote)


def _classify(quote: str, section_text: str) -> str:
    from rag_core.graph import quote_key

    if not quote:
        return "no quote at all"
    if _ELLIPSIS.search(quote):
        return (
            "elided quote, fragments verbatim"
            if _elided_in_order(quote, quote_key(section_text))
            else "elided quote, fragments not verbatim"
        )
    needle, haystack = quote_key(quote), quote_key(section_text)
    # Narrowest finding first. Every rung means the section really does hold the quoted text,
    # for a progressively weaker sense of "really"; falling off the end means it does not.
    ladder: tuple[tuple[str, Callable[[], bool]], ...] = (
        ("punctuation near-miss", lambda: _fold_punct(quote) in _fold_punct(section_text)),
        ("spacing near-miss", lambda: _alnum(quote) in _alnum(section_text)),
        ("welded from two verbatim spans", lambda: _welded_spans(needle, haystack)),
        ("real span, edited edges", lambda: _has_long_run(needle, haystack)),
    )
    for label, holds in ladder:
        if holds():
            return label
    return "paraphrase"


#: Which classes mean the document really does contain the quoted text, so the drop cost evidence.
_RECOVERABLE = (
    "elided quote, fragments verbatim",
    "punctuation near-miss",
    "spacing near-miss",
    "real span, edited edges",
    "welded from two verbatim spans",
)


def _rederive(call: _Call, tally: _Tally) -> None:
    """Re-apply the shipped quote predicate over EVERY entry in one reply.

    Mirrors `parse_map_items` step for step — the per-section cap, the strip-and-truncate, and
    the statement check that precedes the quote check — because each of those changes which
    entries the rule ever judges, and a rate over a different denominator is a different number.
    """
    import json

    from rag_core.graph import quote_key
    from rag_core.json_reply import extract_json_array
    from rag_core.mapreduce import _MAX_ITEMS_PER_WINDOW, _MAX_QUOTE_CHARS, _MAX_STATEMENT_CHARS

    body = extract_json_array(call.reply)
    if body is None:
        return
    try:
        raw = json.loads(body)
    except json.JSONDecodeError:
        return
    if not isinstance(raw, list):
        return

    haystack = quote_key(call.section_text)
    for entry in raw[:_MAX_ITEMS_PER_WINDOW]:
        if not isinstance(entry, Mapping):
            continue
        quote = str(entry.get("quote", "")).strip()[:_MAX_QUOTE_CHARS]
        statement = str(entry.get("statement", "")).strip()[:_MAX_STATEMENT_CHARS]
        if not statement:
            continue
        if not quote or quote_key(quote) not in haystack:
            call.dropped += 1
            tally.rejections.append(_Rejection(call=call, quote=quote, statement=statement))
        else:
            call.kept += 1


def _observe(tally: _Tally) -> None:
    """Record every real map call by wrapping the parse the mapper already calls.

    The wrapper CALLS the shipped function and returns its result untouched; the pipeline runs
    exactly as it would without this script. What it adds is the one thing production deliberately
    does not keep: the reply and the section text behind each count."""
    from rag_clients import section_mapper

    real = section_mapper.parse_map_items

    def recording(reply: str, **kwargs: Any) -> Any:
        parse = real(reply, **kwargs)
        tally.calls.append(
            _Call(
                section_id=str(kwargs["section_id"]),
                reply=reply,
                section_text=str(kwargs["section_text"]),
                facet_ids=tuple(kwargs.get("facet_ids") or ()),
            )
        )
        tally.shipped_dropped += parse.dropped_unquoted
        tally.shipped_kept += len(parse.items)
        return parse

    section_mapper.parse_map_items = recording  # type: ignore[assignment]


async def run(kb: str, question: str, doc_ids: list[str], top_k: int, out: Path | None) -> int:
    from svc_control_plane import pipeline, wiring

    from rag_common import get_settings
    from rag_schemas import RetrievalQuery

    settings = get_settings()
    if not settings.v2_generate:
        print("  RAG_V2_GENERATE is off — there is no map pass to measure.\n")
        return 1
    if settings.v2_map_cache_enabled:
        print("  note: the map cache is ON. Fine on a cold KB; a repeat will measure nothing.")

    tally = _Tally()
    _observe(tally)

    query = RetrievalQuery(tenant_id=kb, text=question, original_text=question, top_k=top_k, doc_ids=tuple(doc_ids))
    decision = await pipeline._route_v2(kb, query)
    selection = await pipeline._select_sections(kb, query, decision) if decision else None
    if decision is None or selection is None or not selection.selected:
        print("  no section pass ran — pass --doc-id, and check the v2 router/retrieval flags.\n")
        return 1
    prepared = await pipeline._map_inputs(kb, selection)
    if prepared is None:
        print("  no chunk text for any selected section.\n")
        return 1
    cards, _texts, jobs = prepared

    facet_ids = sorted({f for _d, f in decision.facets} | {f for c in cards.values() for f in c.facet_ids})
    gateway = await wiring.resolve_tenant_gateway(kb)
    print(
        f"  mapping {len(jobs)} sections of {selection.considered} considered, "
        f"{len(facet_ids)} facets, concurrency {settings.v2_map_concurrency} ...",
        flush=True,
    )
    _items, report, _cached = await pipeline._mapped_items(
        kb, jobs, question=question, facet_ids=facet_ids, gateway=gateway, settings=settings
    )
    for call in tally.calls:
        _rederive(call, tally)
    return _report(tally, report, out)


def _report(tally: _Tally, report: Any, out: Path | None) -> int:
    rederived = len(tally.rejections)
    kept = sum(c.kept for c in tally.calls)
    total = kept + rederived
    print(
        f"\n  {len(tally.calls)} map calls over {report.sections_mapped} sections "
        f"({report.sections_failed} failed, {report.windows} windows)"
    )
    print(
        f"  shipped counters : {tally.shipped_kept} items kept, {tally.shipped_dropped} dropped "
        f"unquoted, {report.dropped_over_cap} over cap"
    )
    agree = rederived == tally.shipped_dropped
    print(
        f"  re-derived       : {rederived} dropped — "
        f"{'agrees with the counter' if agree else 'DISAGREES; believe the counter, not this'}"
    )
    if not total:
        print("\n  nothing was extracted, so there is no rate.\n")
        return 1

    print(f"\n  UNQUOTED RATE: {rederived}/{total} = {rederived / total:.1%} of everything the model extracted")
    for rej in tally.rejections:
        rej.kind = _classify(rej.quote, rej.call.section_text)

    kinds = Counter(r.kind for r in tally.rejections)
    recoverable = sum(n for kind, n in kinds.items() if kind in _RECOVERABLE)
    print(f"\n  {'class':26} {'n':>4} {'of rejections':>14}")
    for kind, n in kinds.most_common():
        print(f"  {kind:26} {n:>4} {n / rederived:>13.0%}")
    print(
        f"\n  the rule COSTS EVIDENCE on {recoverable} of {rederived} ({recoverable / rederived:.0%}) "
        f"— the quoted text is in the section and a wider fold would recover it"
    )
    print(
        f"  the rule is PROTECTING THE ANSWER on {rederived - recoverable} "
        f"({(rederived - recoverable) / rederived:.0%}) — the words are not in the section\n"
    )

    for kind in kinds:
        example = next(r for r in tally.rejections if r.kind == kind)
        print(f"  [{kind}] {example.call.section_id[-12:]}")
        print(f"      quote    : {example.quote[:180]!r}")
        print(f"      statement: {example.statement[:120]!r}")

    hot = Counter(r.call.section_id for r in tally.rejections).most_common(3)
    print(f"\n  most-rejecting sections: {', '.join(f'{sid[-12:]}={n}' for sid, n in hot)}")
    if out is not None:
        # EVERY rejection, not the examples. A map pass costs ten minutes and real model calls,
        # so a reader who wants to disagree with a classification should not have to buy another
        # one — and the replies themselves are gone the moment the parse returns.
        import json

        out.write_text(
            json.dumps(
                [
                    {
                        "section_id": r.call.section_id,
                        "kind": r.kind,
                        "quote": r.quote,
                        "statement": r.statement,
                        "window_chars": len(r.call.section_text),
                        # The text the quote was judged against. Without it the dump cannot
                        # settle a disputed classification, which is the only reason it exists.
                        "window_text": r.call.section_text,
                    }
                    for r in tally.rejections
                ],
                indent=1,
            )
        )
        print(f"  every rejection written to {out}")
    print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True, help="a THROWAWAY kb with the document ingested")
    ap.add_argument(
        "--doc-id",
        action="append",
        default=[],
        help="scope the question to these documents; REQUIRED for the section pass",
    )
    ap.add_argument("--query", default="give me the compliance details from this document")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--out", type=Path, help="write every rejection here as JSON, for auditing")
    args = ap.parse_args()
    sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "services" / "control-plane" / "src")]
    return asyncio.run(run(args.kb, args.query, args.doc_id, args.top_k, args.out))


if __name__ == "__main__":
    sys.exit(main())
