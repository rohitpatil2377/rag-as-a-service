"""Calibration evidence for RAG_ABSTAIN_THRESHOLD (FR-E2) — the grounded-or-abstain gate.

The gate compares ONE number, `RetrievalResult.max_score`, against ONE threshold. Picking that
threshold is only possible if you can see the two populations it has to separate:

  * **answerable** questions the corpus genuinely covers — the gate must let these through;
  * **unanswerable** questions it does not — the gate must stop these.

This script asks both lists against a running stack over HTTP (`/v1/kb/{kb}/retrieve`) and
prints where the two populations actually sit, plus what every candidate threshold would cost.
It reimplements nothing: retrieval, fusion, ACL filtering and reranking all happen in the
platform, and what you read is the number the gate would have read.

`/retrieve` is PLAIN hybrid retrieval — no adaptive router, no query transformation, no agentic
retry, no graph augmentation, no recency. That is deliberate: those paths add spans the gate
also sees, and mixing them in would make the distribution un-attributable. Calibrate against the
bare signal first, then re-check with the features that are enabled for a given KB.

It also reports which scorer produced the numbers. A threshold is a claim about ONE model's
score distribution (see ADR-0015) — calibrating against MiniLM-L-6 on CPU and deploying
bge-reranker-v2-m3 on GPU needs re-checking, and a `lexical` scorer is an overlap fraction on a
different scale entirely.

Usage (stack already up — see README "Quickstart"):

    uv run python scripts/calibrate_abstain.py --kb acme__hr_policies \
        --answerable questions_yes.txt --unanswerable questions_no.txt

    # or edit the two lists below and run with just --kb

No threshold is written anywhere. This produces evidence; a human picks the number.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# EDIT ME (or pass --answerable / --unanswerable files, one question per line).
# ANSWERABLE: the corpus genuinely covers these. UNANSWERABLE: it genuinely does not —
# and they should be plausible-sounding for the domain, not absurd, or the gate looks
# better than it is.
# ---------------------------------------------------------------------------
ANSWERABLE: list[str] = []
UNANSWERABLE: list[str] = []
# ---------------------------------------------------------------------------

_CANDIDATE_THRESHOLDS = (0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)


def _quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}
    return {
        "min": ordered[0],
        "p25": statistics.quantiles(ordered, n=4)[0] if len(ordered) > 1 else ordered[0],
        "median": statistics.median(ordered),
        "p75": statistics.quantiles(ordered, n=4)[2] if len(ordered) > 1 else ordered[0],
        "max": ordered[-1],
    }


def _retrieve(client: httpx.Client, kb: str, question: str, top_k: int) -> dict[str, Any]:
    r = client.post(f"/v1/kb/{kb}/retrieve", json={"query": question, "top_k": top_k})
    r.raise_for_status()
    return r.json()


def _run(client: httpx.Client, kb: str, questions: list[str], top_k: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for q in questions:
        result = _retrieve(client, kb, q, top_k)
        spans = result.get("spans", [])
        out.append(
            {
                "question": q,
                "max_score": float(result.get("max_score", 0.0)),
                "scorers": sorted({s.get("scorer", "?") for s in spans}),
                "top_doc": spans[0]["doc_id"] if spans else None,
                "top_text": (spans[0]["text"][:110].replace("\n", " ") if spans else ""),
                "n_spans": len(spans),
            }
        )
        print(f"  {result.get('max_score', 0.0):.4f}  {q[:88]}", flush=True)
    return out


def _report(label: str, rows: list[dict[str, Any]]) -> None:
    scores = [r["max_score"] for r in rows]
    print(f"\n{label} (n={len(rows)})")
    if not scores:
        print("  no questions")
        return
    q = _quantiles(scores)
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in q.items()))
    print("  per-query: " + " ".join(f"{s:.4f}" for s in sorted(scores, reverse=True)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kb", required=True, help="KB id, e.g. acme__hr_policies")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--token", default=None, help="Bearer token; minted via /v1/dev/token if omitted")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--answerable", default=None, help="file, one question per line")
    p.add_argument("--unanswerable", default=None, help="file, one question per line")
    args = p.parse_args()

    yes = _read(args.answerable) if args.answerable else ANSWERABLE
    no = _read(args.unanswerable) if args.unanswerable else UNANSWERABLE
    if not yes or not no:
        print(
            "Need BOTH an answerable and an unanswerable list — a threshold is a separator, "
            "and one population alone cannot show you where to put it.",
            file=sys.stderr,
        )
        return 2

    token = args.token or _dev_token(args.base_url, args.kb)
    with httpx.Client(base_url=args.base_url, timeout=180.0, headers={"Authorization": f"Bearer {token}"}) as c:
        info = c.get("/v1/system/info").json()
        print(f"reranker model : {info.get('rerank_model', '?')}")
        print(f"threshold now  : {info.get('thresholds', {}).get('abstain', '?')}\n")

        print("ANSWERABLE")
        yes_rows = _run(c, args.kb, yes, args.top_k)
        print("\nUNANSWERABLE")
        no_rows = _run(c, args.kb, no, args.top_k)

    _report("ANSWERABLE", yes_rows)
    _report("UNANSWERABLE", no_rows)

    scorers = sorted({s for r in yes_rows + no_rows for s in r["scorers"]})
    print(f"\nscorer(s) that produced these numbers: {scorers}")
    if scorers != ["cross_encoder"]:
        print("  WARNING: not a pure cross-encoder run — these scores are NOT on the scale the")
        print("  abstain threshold is meant to be calibrated against. See ADR-0015.")

    lo = min((r["max_score"] for r in yes_rows), default=0.0)
    hi = max((r["max_score"] for r in no_rows), default=0.0)
    print(
        f"\nseparation: worst answerable {lo:.4f} vs best unanswerable {hi:.4f} "
        f"-> gap {lo - hi:+.4f}" + ("  (populations overlap)" if lo <= hi else "")
    )

    print("\nthreshold   answerable kept   unanswerable blocked")
    for t in _CANDIDATE_THRESHOLDS:
        kept = sum(1 for r in yes_rows if r["max_score"] >= t)
        blocked = sum(1 for r in no_rows if r["max_score"] < t)
        print(f"  {t:.2f}        {kept:>2}/{len(yes_rows)}              {blocked:>2}/{len(no_rows)}")

    weak = sorted(yes_rows, key=lambda r: r["max_score"])[:3]
    print("\nWeakest answerable retrievals — read these before choosing a floor. A low score here")
    print("means either a genuinely marginal retrieval (the floor is fine) or a good passage the")
    print("reranker underrated (the floor would cost you a correct answer):")
    for r in weak:
        print(f"  {r['max_score']:.4f}  {r['question']}")
        print(f"          top: [{r['top_doc']}] {r['top_text']}")

    print("\nNo threshold has been written. Pick the number from the table above.")
    return 0


def _read(path: str) -> list[str]:
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]


def _dev_token(base_url: str, kb: str) -> str:
    org = kb.split("__", maxsplit=1)[0]
    r = httpx.post(
        f"{base_url}/v1/dev/token",
        json={"subject": "calibrate_abstain", "org_id": org, "tenant_id": kb, "roles": ["reader"]},
        timeout=30.0,
    )
    r.raise_for_status()
    return str(r.json()["access_token"])


if __name__ == "__main__":
    raise SystemExit(main())
