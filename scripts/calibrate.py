"""Unified threshold calibration harness — abstain (FR-E2) AND faithfulness (FR-E4) in one run.

Given a running stack and a labelled golden set, this sweeps both gates' thresholds and prints a
BEFORE/AFTER report: where each gate's two score populations sit, what the CURRENT threshold
costs, and what the swept optimum costs — so the impact of changing a number is explicit. It
writes NOTHING to config: it produces evidence, a human sets the value (ADR-0005/0015).

The sweep math lives in ``rag_core.calibration`` (pure + unit-tested); this script is only
the I/O glue: retrieval + answering happen in the platform over HTTP, so what you read is the
number each gate actually sees.

Stronger judge (the honest faithfulness calibration): the platform default checker is lexical on
CPU CI, which clusters supported answers near 1.0 and gives no borderline samples. Point
``--judge-url`` at a real OpenAI-compatible judge (vLLM / a bigger model) to RE-SCORE each answer
against its cited spans; that is the score distribution worth calibrating the faithful/flag
thresholds against.

Usage (stack up — see README "Quickstart"):

    uv run python scripts/calibrate.py --kb acme__hr_policies \
        --golden sample-data/golden/golden.json \
        --judge-url http://judge:8000/v1 --judge-model Qwen2.5-32B-Instruct --judge-key "$JUDGE_KEY" \
        --report calibration_report.md

    # Abstain-only, platform faithfulness (no external judge):
    uv run python scripts/calibrate.py --kb acme__hr_policies

Requires the stack (stores + the REAL cross-encoder) up; a threshold is a claim about ONE
model's score scale (ADR-0015), so re-run whenever the reranker or judge changes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from rag_core.calibration import candidate_grid, format_report

# Current defaults, used only when /v1/system/info does not surface a live value.
_DEFAULT_ABSTAIN = 0.15
_DEFAULT_FAITHFUL = 0.7
_DEFAULT_FLAG = 0.4


def _dev_token(base_url: str, kb: str) -> str:
    with httpx.Client(base_url=base_url, timeout=30.0) as c:
        r = c.post(
            "/v1/dev/token",
            json={"subject": "calibrate", "tenant_id": kb, "roles": ["reader"], "groups": [], "clearance": "public"},
        )
        r.raise_for_status()
        return str(r.json()["access_token"])


def _load_golden(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (answerable, unanswerable, unfaithful) golden items.

    Schema: each item has a `kind` (answerable|unanswerable|unfaithful). Back-compatible with the
    old shape where `expect_abstain: true` marks the unanswerable and everything else is
    answerable (no unfaithful items). See docs/golden-template.md."""
    data = json.loads(path.read_text())
    items = data.get("items", data if isinstance(data, list) else [])
    buckets: dict[str, list[dict[str, Any]]] = {"answerable": [], "unanswerable": [], "unfaithful": []}
    for raw in items:
        item = {k: v for k, v in raw.items() if not k.startswith("$")}  # drop $doc/$note annotations
        if not item.get("question"):
            continue
        kind = item.get("kind") or ("unanswerable" if item.get("expect_abstain") else "answerable")
        buckets.get(kind, buckets["answerable"]).append(item)
    return buckets["answerable"], buckets["unanswerable"], buckets["unfaithful"]


def _max_score(client: httpx.Client, kb: str, question: str, top_k: int) -> float:
    r = client.post(f"/v1/kb/{kb}/retrieve", json={"query": question, "top_k": top_k})
    r.raise_for_status()
    return float(r.json().get("max_score", 0.0))


def _answer(client: httpx.Client, kb: str, question: str, top_k: int) -> dict[str, Any]:
    r = client.post(f"/v1/kb/{kb}/query", json={"query": question, "top_k": top_k})
    r.raise_for_status()
    return dict(r.json())


def _judge_faithfulness(
    client: httpx.Client, judge_url: str, model: str, key: str, answer: str, sources: list[str]
) -> float | None:
    """Score P(answer is supported by sources) in [0,1] with a stronger OpenAI-compatible judge.

    Returns None if the judge cannot be parsed (so the caller can drop the sample rather than
    fabricate a number). The prompt asks for a bare probability to keep parsing robust."""
    prompt = (
        "You are a strict faithfulness judge. Given SOURCES and an ANSWER, reply with ONLY a "
        "number in [0,1] = the probability that every claim in the ANSWER is supported by the "
        "SOURCES. No words.\n\nSOURCES:\n" + "\n---\n".join(sources) + "\n\nANSWER:\n" + answer
    )
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        r = httpx.post(
            judge_url.rstrip("/") + "/chat/completions",
            headers=headers,
            timeout=120.0,
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0},
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        return max(0.0, min(1.0, float(text.split()[0])))
    except (httpx.HTTPError, KeyError, ValueError, IndexError):
        return None


def _faith_negatives(c: httpx.Client, args: argparse.Namespace, unfaithful: list[dict[str, Any]]) -> list[float]:
    """Judge each labelled unfaithful answer against its question's retrieved sources -> faithfulness
    negatives. Needs an external judge (the platform only scores its own generated answer); returns
    [] without one, so the unfaithful rows are simply skipped."""
    if not (unfaithful and args.judge_url and args.judge_model):
        return []
    neg: list[float] = []
    for item in unfaithful:
        try:
            rr = c.post(f"/v1/kb/{args.kb}/retrieve", json={"query": item["question"], "top_k": args.top_k})
            rr.raise_for_status()
            srcs = [sp.get("text", "") for sp in rr.json().get("spans", [])]
        except httpx.HTTPError:
            continue
        score = _judge_faithfulness(
            c, args.judge_url, args.judge_model, args.judge_key, item.get("answer_to_judge", ""), srcs
        )
        if score is not None:
            neg.append(score)
    return neg


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kb", required=True, help="KB id, e.g. acme__hr_policies")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--token", default=None, help="Bearer token; minted via /v1/dev/token if omitted")
    p.add_argument("--golden", default="sample-data/golden/golden.json", help="labelled golden set")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--step", type=float, default=0.01, help="threshold grid step")
    p.add_argument("--judge-url", default=None, help="OpenAI-compatible judge base URL (stronger faithfulness judge)")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--judge-key", default="")
    p.add_argument("--report", default=None, help="write the markdown report here (default: stdout)")
    args = p.parse_args()

    answerable, unanswerable, unfaithful = _load_golden(Path(args.golden))
    if not answerable:
        print("no answerable golden items — cannot calibrate", file=sys.stderr)
        return 2

    token = args.token or _dev_token(args.base_url, args.kb)
    grid = candidate_grid(0.0, 1.0, args.step)
    out: list[str] = ["# Threshold calibration report", "", f"KB: `{args.kb}` · golden: `{args.golden}`", ""]

    with httpx.Client(base_url=args.base_url, timeout=180.0, headers={"Authorization": f"Bearer {token}"}) as c:
        info: dict[str, Any] = {}
        with contextlib.suppress(httpx.HTTPError):
            info = c.get("/v1/system/info").json()
        # /v1/system/info nests the live gate thresholds under "thresholds".
        th = info.get("thresholds") if isinstance(info.get("thresholds"), dict) else {}
        abstain_cur = float(th.get("abstain", info.get("abstain_threshold", _DEFAULT_ABSTAIN)))
        faithful_cur = float(th.get("faithful", info.get("faithful_threshold", _DEFAULT_FAITHFUL)))

        # --- Abstain gate: answerable should pass, unanswerable should be stopped. ---
        pos_abstain = [_max_score(c, args.kb, i["question"], args.top_k) for i in answerable]
        neg_abstain = [_max_score(c, args.kb, i["question"], args.top_k) for i in unanswerable]
        out.append(
            format_report(
                "Abstain gate (max_score)",
                current=abstain_cur,
                positives=pos_abstain,
                negatives=neg_abstain,
                candidates=grid,
            )
        )
        # Emit the abstain result immediately: it is retrieve-only + robust, while the
        # faithfulness pass below is generation-bound and can time out. A late crash must
        # never discard the abstain report we already have.
        print(out[-1], flush=True)

        # --- Faithfulness gate: supported answers are positives (best-effort, generation-bound). ---
        pos_faith: list[float] = []
        for item in answerable:
            try:
                ans = _answer(c, args.kb, item["question"], args.top_k)
            except httpx.HTTPError:
                continue  # a slow/failed generation call drops that sample, never the run
            sources = [ci.get("quote", "") for ci in ans.get("citations", [])]
            if args.judge_url and args.judge_model:
                score = _judge_faithfulness(
                    c, args.judge_url, args.judge_model, args.judge_key, ans.get("answer", ""), sources
                )
            else:
                score = ans.get("faithfulness_score")
            if score is not None:
                pos_faith.append(float(score))

        # Faithfulness NEGATIVES from labelled `unfaithful` items (judge-scored; see _faith_negatives).
        neg_faith = _faith_negatives(c, args, unfaithful)

        out.append(
            format_report(
                "Faithfulness gate" + (" (external judge)" if args.judge_url else " (platform checker)"),
                current=faithful_cur,
                positives=pos_faith,
                negatives=neg_faith,
                candidates=grid,
            )
        )

    report = "\n".join(out)
    if args.report:
        Path(args.report).write_text(report)
        print(f"wrote {args.report}", file=sys.stderr)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
