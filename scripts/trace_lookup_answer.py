#!/usr/bin/env python3
"""Follow one lookup's expected answer from retrieval into the generated text.

`scripts/retrieval_metrics.py` establishes that both failing AEGCL lookups retrieve their
evidence at rank 1 or 2, so the loss is downstream of retrieval. That narrows the question but
does not answer it: between a ranked span and a printed answer the value can still be lost four
ways, and they need different fixes.

  * the span carrying it is ranked but CUT before generation, by top_k or the span budget;
  * it reaches the prompt and the model does not state it;
  * the model states it in a different FORM — "6,000" for "6000", "six months" for "6 (six)
    months" — and the gold's substring check cannot see it;
  * the answer abstains or is blocked, so there is no text to look in.

The third is not a system failure at all and has already bitten this branch twice, in the
standards designations and the "6 Year"/"6 years" cue. So this prints what each stage actually
contains rather than scoring it, and leaves the judgement to a reader.

Runs the shipped path: the real retriever, the real `build_prompt`, the real generation service.

    scripts/trace_lookup_answer.py --kb zzgoal-abc --suite eval/gold/aegcl_corrigendum.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path
from typing import Any


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def _loose(text: str) -> str:
    """Digits and letters only: "6,000" and "6000" collapse, and so do "₹19.44" and "19.44"."""
    return re.sub(r"[^a-z0-9]+", "", _norm(text))


async def _one(kb: str, question: str, expected: str, top_k: int) -> None:
    from svc_control_plane import wiring

    from rag_core.grounding import build_prompt
    from rag_schemas import RetrievalQuery

    print(f"\n  question : {question}")
    print(f"  expects  : {expected!r}")

    result = await wiring.retrieval_service().retrieve(
        RetrievalQuery(tenant_id=kb, text=question, original_text=question, top_k=top_k)
    )
    spans = list(result.spans)
    hits = [i for i, s in enumerate(spans, start=1) if _norm(expected) in _norm(s.text)]
    loose_hits = [i for i, s in enumerate(spans, start=1) if _loose(expected) in _loose(s.text)]
    print(
        f"  retrieved: {len(spans)} spans; exact-form evidence at ranks {hits or '—'}; any-form at {loose_hits or '—'}"
    )
    if not loose_hits:
        print("  -> the value is in NO retrieved span: a retrieval problem, not a generation one")
        return

    prompt = build_prompt(question, spans)
    in_prompt = _norm(expected) in _norm(prompt)
    loose_prompt = _loose(expected) in _loose(prompt)
    print(f"  prompt   : {len(prompt)} chars; expected value present exact={in_prompt} any-form={loose_prompt}")
    if not loose_prompt:
        print(
            "  -> ranked but NOT in the prompt: the span was cut between retrieval and "
            "generation (top_k or the span budget)"
        )
        return

    answered = await wiring.generation_service().answer(result, actor="trace", trace_id="trace-lookup")
    text = answered.answer or ""
    print(f"  answer   : status={answered.status} {len(text)} chars, {len(answered.citations)} citation(s)")
    exact = _norm(expected) in _norm(text)
    loose = _loose(expected) in _loose(text)
    if exact:
        print("  -> PRESENT in the answer in the gold's own form; if the suite scores this a miss, the suite is wrong")
    elif loose:
        i = _loose(text).find(_loose(expected))
        print(
            f"  -> PRESENT but in a DIFFERENT FORM — the gold's substring check cannot see it. "
            f"This is a gold-cue problem, not a model one (near offset {i} of the folded text)"
        )
    else:
        print(
            "  -> ABSENT from the answer although it was in the prompt: the model had the "
            "value and did not state it. A generation failure."
        )
    snippet = text[:400].replace("\n", " ")
    print(f"  text     : {snippet!r}")


async def run(kb: str, cases: list[tuple[str, str]], top_k: int) -> int:
    for question, expected in cases:
        await _one(kb, question, expected, top_k)
    print()
    return 0


def _lookups(suite: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (str(q["query"]), str(q["expected_answer"])) for q in suite.get("queries") or [] if q.get("expected_answer")
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument(
        "--only", action="append", default=[], help="substring of the question; repeatable. Default: every lookup"
    )
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()
    sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "services" / "control-plane" / "src")]

    import yaml

    cases = _lookups(yaml.safe_load(args.suite.read_text()))
    if args.only:
        cases = [c for c in cases if any(o.lower() in c[0].lower() for o in args.only)]
    return asyncio.run(run(args.kb, cases, args.top_k))


if __name__ == "__main__":
    sys.exit(main())
