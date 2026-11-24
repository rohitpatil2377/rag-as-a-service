"""Federated RAG benchmark harness — measures the numbers the proposal estimates.

Two modes, both over a running stack via HTTP:

  scaling  — federated scatter-gather latency as the peer-department count N scales (e.g.
             10, 50). Drives the federated ask endpoint with N grant-authorised peers and
             times the round-trip; reports p50/p95/p99 per N and, with a --baseline, whether
             the measured reduction hits the proposal's target (e.g. -35%).
  compare  — retrieval latency of two vector backends head-to-head (Qdrant vs Milvus,
             ADR-0003). Times /v1/kb/{kb}/retrieve against two --base-url endpoints an operator
             stood up (one Qdrant stack, one Milvus stack).

The statistics + report formatting live in ``rag_core.benchmark`` (pure, unit-tested);
this script only gathers timings. It writes NOTHING except the report.

Provisioning (scaling mode): the N peer departments — a KB per dept with a document, and an
active grant from each to the caller — is an OPERATOR step (see docs). The script assumes
dept ids ``{prefix}0..{prefix}N-1`` already exist and are granted to --kb; it measures, it does
not provision.

Usage (stack up):
    uv run python scripts/benchmark.py scaling --kb caller --peers 10,50 --repeats 20 \
        --baseline-ms 900 --target-pct 35 --report bench.md
    uv run python scripts/benchmark.py compare --kb acme__hr_policies \
        --base-url-a http://qdrant-stack:8000 --base-url-b http://milvus-stack:8000 --repeats 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from rag_core.benchmark import (
    LatencySummary,
    format_comparison_report,
    format_scaling_report,
    format_target_report,
    summarize,
)


def _dev_token(base_url: str, kb: str) -> str:
    with httpx.Client(base_url=base_url, timeout=30.0) as c:
        r = c.post(
            "/v1/dev/token",
            json={"subject": "bench", "tenant_id": kb, "roles": ["reader"], "groups": [], "clearance": "public"},
        )
        r.raise_for_status()
        return str(r.json()["access_token"])


def _time_call(fn: Any, repeats: int) -> list[float]:
    """Return per-call wall time in ms for `repeats` invocations of fn()."""
    out: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def _bench_scaling(args: argparse.Namespace) -> str:
    token = args.token or _dev_token(args.base_url, args.kb)
    peers = [int(n) for n in args.peers.split(",") if n.strip()]
    rows: list[tuple[int, LatencySummary]] = []
    out = ["# Benchmark - federated scatter-gather scaling", "", f"KB: `{args.kb}` - repeats: {args.repeats}", ""]
    with httpx.Client(base_url=args.base_url, timeout=300.0, headers={"Authorization": f"Bearer {token}"}) as c:
        for n in peers:
            federate_from = [f"{args.dept_prefix}{i}" for i in range(n)]

            def one(fed: list[str] = federate_from) -> None:
                r = c.post(
                    f"/v1/kb/{args.kb}/ask",
                    json={"query": args.query, "top_k": args.top_k, "federate_from": fed},
                )
                r.raise_for_status()

            rows.append((n, summarize(_time_call(one, args.repeats))))
    out.append(format_scaling_report(rows))
    if args.baseline_ms and rows:
        worst = rows[-1][1]  # largest N
        out.append(
            format_target_report(
                f"scatter-gather p95 at N={rows[-1][0]}",
                baseline=args.baseline_ms,
                measured=worst.p95,
                target_pct=args.target_pct,
            )
        )
    return "\n".join(out)


def _bench_compare(args: argparse.Namespace) -> str:
    def measure(base_url: str) -> LatencySummary:
        token = _dev_token(base_url, args.kb)
        with httpx.Client(base_url=base_url, timeout=120.0, headers={"Authorization": f"Bearer {token}"}) as c:

            def one() -> None:
                r = c.post(f"/v1/kb/{args.kb}/retrieve", json={"query": args.query, "top_k": args.top_k})
                r.raise_for_status()

            return summarize(_time_call(one, args.repeats))

    a = measure(args.base_url_a)
    b = measure(args.base_url_b)
    return "\n".join(
        [
            "# Benchmark - vector backend comparison (ADR-0003)",
            "",
            f"KB: `{args.kb}` - repeats: {args.repeats}",
            "",
            format_comparison_report(args.name_a, a, args.name_b, b),
        ]
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--kb", required=True)
    common.add_argument("--query", default="What is the platform's uptime target?")
    common.add_argument("--top-k", type=int, default=5)
    common.add_argument("--repeats", type=int, default=20)
    common.add_argument("--report", default=None, help="write markdown here (default: stdout)")

    s = sub.add_parser("scaling", parents=[common], help="federated scatter-gather latency vs N peers")
    s.add_argument("--base-url", default="http://localhost:8000")
    s.add_argument("--token", default=None)
    s.add_argument("--peers", default="10,50", help="comma-separated peer counts")
    s.add_argument("--dept-prefix", default="dept-", help="peer dept ids are {prefix}0..{prefix}N-1")
    s.add_argument("--baseline-ms", type=float, default=0.0, help="baseline p95 to compare against")
    s.add_argument("--target-pct", type=float, default=35.0, help="required reduction vs baseline")

    c = sub.add_parser("compare", parents=[common], help="two vector backends head-to-head")
    c.add_argument("--base-url-a", required=True)
    c.add_argument("--base-url-b", required=True)
    c.add_argument("--name-a", default="Qdrant")
    c.add_argument("--name-b", default="Milvus")

    args = p.parse_args()
    report = _bench_scaling(args) if args.mode == "scaling" else _bench_compare(args)
    if args.report:
        Path(args.report).write_text(report)
        print(f"wrote {args.report}", file=sys.stderr)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
