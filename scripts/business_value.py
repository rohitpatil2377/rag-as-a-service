#!/usr/bin/env python3
"""Business-value benchmark CLI — turn the -35% / 2x ESTIMATES into MEASURED numbers.

Runs a workload against a KB, prices it from the FinOps meters, compares to a stated baseline,
and writes a shareable markdown report. In-process (reads the same RAG_* config as the app), so
point it at the real stack for real numbers.

    uv run python scripts/business_value.py --kb <kb> --workload workload.json \
        --baseline baseline.json --report business-value.md

workload.json : {"questions": [{"question": "...", "expected": "24 days"}, ...]}  (expected optional)
baseline.json : {"name": "hosted-API centralized RAG", "cost_per_answer": 3.2,
                 "latency_p50_ms": 4200, "latency_p95_ms": 9000, "grounded_rate": 0.71}   (all optional)

A number without a stated baseline is an estimate, not a benchmark — supply one.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


async def _bench(kb: str, questions: list[object], baseline: object, actor: str) -> object:
    from svc_control_plane.business_value import run_business_value

    return await run_business_value(kb, questions, baseline, actor=actor)  # type: ignore[arg-type]


def main() -> int:
    p = argparse.ArgumentParser(description="Business-value benchmark (cost / latency / quality vs a baseline).")
    p.add_argument("--kb", required=True, help="Knowledge base id to benchmark against.")
    p.add_argument("--workload", required=True, help="JSON file: {questions:[{question, expected?}]}.")
    p.add_argument("--baseline", help="JSON file: the 'before' to compare against (optional).")
    p.add_argument("--report", help="Write the markdown report here (optional).")
    p.add_argument("--actor", default="bench", help="Actor recorded on the run (default: bench).")
    args = p.parse_args()

    from svc_control_plane.business_value import BizBaseline, render_markdown
    from svc_control_plane.quality import EvalQuestion

    workload = json.loads(Path(args.workload).read_text())
    items = workload.get("questions", workload if isinstance(workload, list) else [])
    questions = [EvalQuestion(**q) for q in items]
    if not questions:
        print("no questions in the workload file")
        return 2
    baseline = BizBaseline(**json.loads(Path(args.baseline).read_text())) if args.baseline else None

    report = asyncio.run(_bench(args.kb, questions, baseline, args.actor))
    md = render_markdown(report)  # type: ignore[arg-type]
    print(md)
    if args.report:
        Path(args.report).write_text(md)
        print(f"\n[written] {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
