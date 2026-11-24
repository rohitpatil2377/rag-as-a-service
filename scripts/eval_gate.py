#!/usr/bin/env python3
"""Run a golden set through the LIVE pipeline and grade it, repeatably.

Why this exists, in one sentence: the 1,800-test suite passes while real answers regress, and a
single live run is not evidence -- twice in one week a conclusion drawn from one run turned out
to be noise in a non-deterministic model.

So every question runs ``--repeat`` times and the report is a RATE, not a verdict. A question
that answers correctly 2 times in 3 is a different thing from one that answers correctly always,
and a tool that hides the difference is worse than no tool.

Scoring, per question:
  * expected fact given  -> the answer must be GROUNDED and contain that fact
  * no expected fact     -> the question is unanswerable from the corpus and the platform must
                            ABSTAIN. Counting only the answerable half is how a gate rewards a
                            model that answers everything.

Turn the SEMANTIC CACHE OFF before measuring (RAG_SEMANTIC_CACHE_ENABLED=false). It is
invalidated on ingest but NOT on a configuration change, so with it on the second and third run
of a question return the first run's answer -- repeats stop measuring anything, and an A/B of
top_k or the chunker reports no difference because the old answer is being replayed.

Comparison is NFKC-folded and case-insensitive, so "NO₃⁻" matches "NO3-" and "A-4" matches "a-4"
-- an extraction that loses a subscript should not read as a wrong answer.

RAG v2 SUITE MODE (Phase 11). With ``--suite`` the questions come from a gold FILE in the plan's
§11 format (``eval/gold/<doc>.yaml``) instead of from the KB's stored golden set, and the report
carries the v2 metrics: lookup accuracy, item recall by must/should tier, value accuracy,
supersession correctness, context precision, faithfulness and per-stage latency. ``--suite
rag_v2`` runs every file in ``eval/gold/``, which is the plan's ``eval run --suite rag_v2``.

``--floor metric=value`` gates the run: the command exits non-zero when an AGREED floor is
breached. No floor is set by default, deliberately — see ``SuiteFloors``.

Usage:
  scripts/eval_gate.py --kb acme__research --repeat 3
  scripts/eval_gate.py --kb acme__research --repeat 3 --out baseline.json
  scripts/eval_gate.py --kb acme__research --compare baseline.json
  scripts/eval_gate.py --kb acme__research --suite rag_v2 --repeat 3
  scripts/eval_gate.py --kb acme__research --suite eval/gold/sop.yaml --floor min_abstain_correctness=1.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx

_DEFAULT_API = "http://localhost:8000"
_HTTP_OK = 200
_RETRY_STATUS = frozenset({429, 502, 503, 504})
_MAX_ATTEMPTS = 4
_BACKOFF_SECONDS = 3.0
_EPSILON = 1e-9


def _fold(text: str) -> str:
    """Case-insensitive, NFKC-normalised, dash-unified, whitespace-collapsed.

    Delegates to ``svc_guardrails_eval.suite.fold_text`` so the v1 gate and the Phase-11 suite
    score a containment check the same way. It also FIXES this script's long-standing claim: NFKC
    alone maps "⁻" to the MINUS SIGN, not to the hyphen, so "NO₃⁻" did NOT match "NO3-" the way
    the docstring above says it does. Folding the dash family makes the claim true. The change can
    only make a match MORE generous, so a question that passed still passes."""
    from svc_guardrails_eval.suite import fold_text

    return fold_text(text)


async def _token(client: httpx.AsyncClient, tenant: str) -> str:
    r = await client.post(
        "/v1/dev/token",
        json={"subject": "eval-gate@acme.example", "tenant_id": tenant, "roles": ["admin", "ingestor", "reader"]},
    )
    r.raise_for_status()
    return str(r.json()["access_token"])


async def _ask(
    client: httpx.AsyncClient, kb: str, token: str, question: str, top_k: int | None = None
) -> dict[str, Any]:
    """One question, with backoff on a rate limit.

    A transport failure is NOT a wrong answer, and scoring it as one is how a gate reports an
    outage as a quality regression. It happened on the first run of this script: a burst of
    questions tripped the per-org rate limiter and four correct abstains were recorded as
    failures, moving the headline score by 17 points for no reason connected to quality.
    """
    last = "unknown"
    for attempt in range(_MAX_ATTEMPTS):
        try:
            r = await client.post(
                f"/v1/kb/{kb}/query",
                headers={"Authorization": f"Bearer {token}"},
                json={"query": question, **({"top_k": top_k} if top_k else {})},
                timeout=180.0,
            )
        except httpx.HTTPError as exc:
            # A dropped connection or a read timeout is a transport failure like any other; it
            # just arrives as an exception rather than a status. Catching only the status is how
            # the first two versions of this script crashed mid-run and lost the whole report.
            last = type(exc).__name__
            if attempt < _MAX_ATTEMPTS - 1:
                await asyncio.sleep(_BACKOFF_SECONDS * (attempt + 1))
                continue
            break
        if r.status_code == _HTTP_OK:
            return dict(r.json())
        last = f"HTTP {r.status_code}"
        if r.status_code in _RETRY_STATUS and attempt < _MAX_ATTEMPTS - 1:
            await asyncio.sleep(_BACKOFF_SECONDS * (attempt + 1))
            continue
        break
    return {"status": "error", "answer": last, "faithfulness_score": None, "error": True}


def _judge(expected: str | None, result: dict[str, Any]) -> bool:
    status = result.get("status")
    if expected is None:
        # Unanswerable: abstaining IS the correct behaviour, and the only correct one.
        return status == "abstained"
    return status == "grounded" and _fold(expected) in _fold(result.get("answer") or "")


async def run(kb: str, repeat: int, api: str, top_k: int | None = None) -> dict[str, Any]:
    tenant = kb.split("__", 1)[0]
    async with httpx.AsyncClient(base_url=api) as client:
        token = await _token(client, tenant)
        gr = await client.get(f"/v1/kb/{kb}/golden-set", headers={"Authorization": f"Bearer {token}"})
        gr.raise_for_status()
        questions = gr.json()["questions"]
        if not questions:
            sys.exit(f"golden set for {kb} is empty — save one first")

        rows: list[dict[str, Any]] = []
        for q in questions:
            passes, faiths, statuses, errors = 0, [], [], 0
            for _ in range(repeat):
                res = await _ask(client, kb, token, q["question"], top_k)
                statuses.append(res.get("status"))
                errors += int(bool(res.get("error")))
                if res.get("faithfulness_score") is not None:
                    faiths.append(float(res["faithfulness_score"]))
                passes += int(_judge(q.get("expected"), res))
            rows.append(
                {
                    "question": q["question"],
                    "expected": q.get("expected"),
                    "pass_rate": passes / repeat,
                    "statuses": statuses,
                    "faithfulness": round(statistics.mean(faiths), 3) if faiths else None,
                    "errors": errors,
                }
            )

    answerable = [r for r in rows if r["expected"] is not None]
    refusals = [r for r in rows if r["expected"] is None]
    return {
        "kb": kb,
        "repeat": repeat,
        "top_k": top_k,
        "questions": len(rows),
        # The headline number: every question, every repeat.
        "score": round(statistics.mean(r["pass_rate"] for r in rows), 4),
        "answerable_score": round(statistics.mean(r["pass_rate"] for r in answerable), 4) if answerable else None,
        "refusal_score": round(statistics.mean(r["pass_rate"] for r in refusals), 4) if refusals else None,
        # A question that is right sometimes is the signal a single run destroys.
        # Transport failures, surfaced separately. A score computed over these is not a score.
        "errors": sum(r["errors"] for r in rows),
        "flaky": [r["question"] for r in rows if 0 < r["pass_rate"] < 1],
        "failing": [r["question"] for r in rows if r["pass_rate"] == 0],
        "rows": rows,
    }


def _report(rep: dict[str, Any]) -> None:
    tk = f", top_k={rep['top_k']}" if rep.get("top_k") else ""
    print(f"\n  golden set : {rep['kb']}  ({rep['questions']} questions x {rep['repeat']} runs{tk})")
    if rep["errors"]:
        print(f"  !! {rep['errors']} request(s) FAILED IN TRANSPORT — this score is not trustworthy")
    print(f"  SCORE      : {rep['score']:.1%}")
    print(f"    answerable : {rep['answerable_score']:.1%}" if rep["answerable_score"] is not None else "")
    print(f"    refusals   : {rep['refusal_score']:.1%}" if rep["refusal_score"] is not None else "")
    if rep["flaky"]:
        print(f"\n  FLAKY ({len(rep['flaky'])}) — right sometimes, which one run would hide:")
        for q in rep["flaky"]:
            row = next(r for r in rep["rows"] if r["question"] == q)
            print(f"    {row['pass_rate']:.0%}  {q}")
    if rep["failing"]:
        print(f"\n  FAILING ({len(rep['failing'])}):")
        for q in rep["failing"]:
            print(f"    0%    {q}")
    print()


def _compare(now: dict[str, Any], before: dict[str, Any]) -> int:
    delta = now["score"] - before["score"]
    print(f"\n  before : {before['score']:.1%}\n  now    : {now['score']:.1%}\n  delta  : {delta:+.1%}")
    prev = {r["question"]: r["pass_rate"] for r in before["rows"]}
    moved = [
        (r["question"], prev.get(r["question"], 0.0), r["pass_rate"])
        for r in now["rows"]
        if r["question"] in prev and abs(r["pass_rate"] - prev[r["question"]]) > _EPSILON
    ]
    if moved:
        print("\n  per-question movement:")
        for q, was, is_ in sorted(moved, key=lambda m: m[2] - m[1]):
            print(f"    {was:.0%} -> {is_:.0%}   {q}")
    print()
    # A regression fails the gate; an improvement or a draw passes.
    return 1 if delta < 0 else 0


# --- RAG v2 suite mode (Phase 11) -----------------------------------------------------------------

_GOLD_DIR = Path(__file__).resolve().parent.parent / "eval" / "gold"
# The per-query metrics that decide a pass for THIS repeat. A metric the gold query did not
# specify is None and cannot fail — the same rule `SuiteReport.failures` applies one level up.
_FULL_RECALL = 1.0


def _suite_paths(arg: str) -> list[Path]:
    """``rag_v2`` -> every gold file in ``eval/gold/``; anything else -> that one path."""
    if arg == "rag_v2":
        files = sorted(p for p in _GOLD_DIR.glob("*.y*ml")) + sorted(_GOLD_DIR.glob("*.json"))
        if not files:
            sys.exit(f"no gold files in {_GOLD_DIR} — the suite set is empty")
        return files
    return [Path(arg)]


def _query_passed(score: Any) -> bool:
    """One repeat of one gold query: did it do everything the gold asked of it?"""
    return bool(
        score.abstain_ok
        and score.lookup_hit is not False
        and (score.item_recall_must is None or score.item_recall_must >= _FULL_RECALL)
        and score.supersession_ok is not False
    )


async def run_suite(kb: str, path: Path, repeat: int, api: str, top_k: int | None = None) -> dict[str, Any]:
    """Run one gold suite ``repeat`` times and report the v2 metrics as RATES.

    Repeats are the whole point and the reason this lives in THIS script rather than in a new one:
    the generation model is non-deterministic, and a question that passes 2 times in 3 looks
    identical to one that always passes if you only run it once. So the suite is scored once per
    repeat and the report is the mean of those runs, never a single verdict."""
    from svc_guardrails_eval.suite import SuiteReport, load_suite, score_query, score_suite

    from rag_schemas import GroundedAnswer

    suite = load_suite(path)
    tenant = kb.split("__", 1)[0]
    # answers[query_index][repeat] — asked question by question so a repeat is a repeat of the
    # same question against a warm index, matching how the v1 path measures.
    answers: list[list[GroundedAnswer | None]] = []
    latencies: list[list[dict[str, int]]] = []
    errors = 0
    async with httpx.AsyncClient(base_url=api) as client:
        token = await _token(client, tenant)
        for gold in suite.queries:
            per_repeat: list[GroundedAnswer | None] = []
            per_latency: list[dict[str, int]] = []
            for _ in range(repeat):
                started = time.perf_counter()
                res = await _ask(client, kb, token, gold.query, top_k)
                elapsed = max(0, round((time.perf_counter() - started) * 1000))
                if res.get("error"):
                    errors += 1
                    per_repeat.append(None)
                    per_latency.append({})
                    continue
                try:
                    per_repeat.append(GroundedAnswer.model_validate(res))
                except Exception:  # a shape the contract does not describe is a transport-class fault
                    errors += 1
                    per_repeat.append(None)
                    per_latency.append({})
                    continue
                # END-TO-END ONLY. The PER-STAGE timeline exists (Phase 12 records it on the
                # `query_stage_timing` log line and in `query_log.latency_ms`) but the query
                # response does not carry it, and putting it there is an API-contract change this
                # harness should not make on its own — see discovery §8.23.
                per_latency.append({"e2e": elapsed})
            answers.append(per_repeat)
            latencies.append(per_latency)

    reports: list[SuiteReport] = []
    per_query_passes: list[list[bool]] = [[] for _ in suite.queries]
    for run_index in range(repeat):
        scores = []
        for q_index, gold in enumerate(suite.queries):
            answer = answers[q_index][run_index]
            if answer is None:
                # A transport fault is not a wrong answer. Excluded from this run's scores so the
                # number stays a measurement of the platform rather than of the network.
                continue
            score = score_query(gold, answer, latency_ms=latencies[q_index][run_index])
            scores.append((gold, score))
            per_query_passes[q_index].append(_query_passed(score))
        if len(scores) == len(suite.queries):
            reports.append(score_suite(suite, [s for _, s in scores]))

    metrics = _mean_metrics(reports)
    rows = [
        {
            "query": gold.query,
            "intent": gold.intent,
            "pass_rate": (sum(p for p in passes) / len(passes)) if passes else None,
        }
        for gold, passes in zip(suite.queries, per_query_passes, strict=True)
    ]
    rated = [r for r in rows if r["pass_rate"] is not None]
    return {
        "suite": suite.name,
        "doc_type": suite.doc_type,
        "kb": kb,
        "repeat": repeat,
        "top_k": top_k,
        "queries": len(suite.queries),
        "complete_runs": len(reports),
        "score": round(statistics.mean(r["pass_rate"] for r in rated), 4) if rated else None,
        "metrics": metrics,
        "intent_counts": reports[0].intent_counts if reports else {},
        "latency_p95": reports[0].latency_p95 if reports else {},
        "errors": errors,
        "flaky": [r["query"] for r in rows if r["pass_rate"] is not None and 0 < r["pass_rate"] < 1],
        "failing": [r["query"] for r in rows if r["pass_rate"] == 0],
        "rows": rows,
    }


_METRIC_KEYS = (
    "abstain_correctness",
    "lookup_accuracy",
    "item_recall_must",
    "item_recall_should",
    "value_accuracy",
    "supersession_correctness",
    "context_precision",
    "faithfulness_avg",
)


def _mean_metrics(reports: list[Any]) -> dict[str, float | None]:
    """Each metric averaged over the complete runs; None where no run could measure it.

    None rather than 0.0, carried all the way from `QueryScore`: a metric this suite does not
    specify — supersession correctness on a document that is not an amendment — must not read as
    a zero score, and must not fail a floor."""
    out: dict[str, float | None] = {}
    for key in _METRIC_KEYS:
        values = [v for r in reports if (v := getattr(r, key)) is not None]
        out[key] = round(statistics.mean(values), 4) if values else None
    return out


def _report_suite(rep: dict[str, Any]) -> None:
    tk = f", top_k={rep['top_k']}" if rep.get("top_k") else ""
    print(f"\n  suite      : {rep['suite']}  [{rep['doc_type']}]  on {rep['kb']}")
    print(f"  run        : {rep['queries']} queries x {rep['repeat']} runs{tk}")
    if rep["errors"]:
        print(f"  !! {rep['errors']} request(s) FAILED IN TRANSPORT — this score is not trustworthy")
    if rep["score"] is not None:
        print(f"  SCORE      : {rep['score']:.1%}   (every gold assertion, every repeat)")
    print("  metrics    :")
    for key, value in rep["metrics"].items():
        shown = "   n/a (not asserted by this suite)" if value is None else f"{value:.1%}"
        print(f"    {key:<26} {shown}")
    if rep["intent_counts"]:
        print(f"  intents    : {rep['intent_counts']}")
    if rep["latency_p95"]:
        print(f"  latency p95: {rep['latency_p95']} ms")
    if rep["flaky"]:
        print(f"\n  FLAKY ({len(rep['flaky'])}) — right sometimes, which one run would hide:")
        for q in rep["flaky"]:
            row = next(r for r in rep["rows"] if r["query"] == q)
            print(f"    {row['pass_rate']:.0%}  {q}")
    if rep["failing"]:
        print(f"\n  FAILING ({len(rep['failing'])}):")
        for q in rep["failing"]:
            print(f"    0%    {q}")
    print()


def _parse_floors(pairs: list[str]) -> Any:
    from svc_guardrails_eval.suite import SuiteFloors

    kwargs: dict[str, float] = {}
    allowed = {f.name for f in __import__("dataclasses").fields(SuiteFloors)}
    for pair in pairs:
        name, _, raw = pair.partition("=")
        name = name.strip()
        if name not in allowed:
            sys.exit(f"unknown floor {name!r}; one of {sorted(allowed)}")
        try:
            kwargs[name] = float(raw)
        except ValueError:
            sys.exit(f"floor {name!r} needs a number, got {raw!r}")
    return SuiteFloors(**kwargs)


def _gate_suite(rep: dict[str, Any], floors: Any) -> int:
    from svc_guardrails_eval.suite import SuiteReport

    report = SuiteReport(
        suite=rep["suite"],
        doc_type=rep["doc_type"],
        n=rep["queries"],
        scored=rep["queries"],
        **dict(rep["metrics"]),
    )
    breaches = report.failures(floors)
    if breaches:
        print("  GATE FAILED:")
        for breach in breaches:
            print(f"    {breach}")
        print()
        return 1
    print("  GATE PASSED (no agreed floor breached)\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kb", required=True)
    ap.add_argument("--repeat", type=int, default=3, help="runs per question (default 3; 1 is not evidence)")
    ap.add_argument("--api", default=_DEFAULT_API)
    ap.add_argument("--top-k", type=int, default=None, help="override retrieval top_k (default: server's)")
    ap.add_argument("--out", type=Path, help="write the report here")
    ap.add_argument("--compare", type=Path, help="compare against a previous report and exit non-zero on regression")
    ap.add_argument(
        "--suite",
        help="RAG v2 gold suite: a path to eval/gold/<doc>.yaml, or 'rag_v2' for every file there",
    )
    ap.add_argument(
        "--floor",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="gate the v2 suite on an agreed floor, e.g. --floor min_abstain_correctness=1.0",
    )
    args = ap.parse_args()

    if args.suite:
        floors = _parse_floors(args.floor)
        reports = [
            asyncio.run(run_suite(args.kb, path, args.repeat, args.api, args.top_k))
            for path in _suite_paths(args.suite)
        ]
        exit_code = 0
        for rep in reports:
            _report_suite(rep)
            exit_code |= _gate_suite(rep, floors)
        if args.out:
            args.out.write_text(json.dumps(reports if len(reports) > 1 else reports[0], indent=2))
            print(f"  written: {args.out}\n")
        return exit_code

    rep = asyncio.run(run(args.kb, args.repeat, args.api, args.top_k))
    _report(rep)
    if args.out:
        args.out.write_text(json.dumps(rep, indent=2))
        print(f"  written: {args.out}\n")
    if args.compare:
        return _compare(rep, json.loads(args.compare.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
