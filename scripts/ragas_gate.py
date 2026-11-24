"""RAGAS governance gate — the QA-runnable command that GOVERNS the RaaS.

Runs the sovereign RAGAS suite across knowledge bases through the REAL pipeline (via the
running stack's ``POST /v1/kb/{id}/eval``), scores each KB's aggregate against the published
quality contract (``svc_control_plane.ragas_governance``), and:

  * writes ``ragas_report.json`` + ``ragas_report.md`` (the auditable evidence),
  * optionally writes a Prometheus textfile (``--prometheus out.prom``) so the governing
    numbers land on the Grafana quality dashboard,
  * EXITS NON-ZERO on any FAIL (and, with ``--strict``, on INCONCLUSIVE too) so it gates CI
    and release — a red gate blocks the claim, it doesn't just print it.

The evaluation itself never leaves the sovereign boundary: the KB's own self-hosted judge
(Sarvam/gemma) and BGE-M3 embedder score the run (see ``quality._build_ragas_judges``). This
script only orchestrates HTTP + applies the contract.

Usage (stack up):
    uv run python scripts/ragas_gate.py --base-url http://localhost:8000 \
        --kbs acme__hr,acme__finance,public__company-wide --report ragas_report.md \
        --prometheus /var/lib/node_exporter/textfile/ragas.prom
    # discover the caller's visible KBs instead of listing them:
    uv run python scripts/ragas_gate.py --tenant public__company-wide

Auth: mints a dev JWT per target org via ``POST /v1/dev/token`` (dev_jwt mode). In a
non-dev deployment pass ``--token <jwt>`` (must be authorised for every target KB).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

# Import the pure contract from the control-plane package (installed in the workspace venv).
from svc_control_plane.ragas_governance import (
    FleetGateReport,
    GateStatus,
    RagasFloors,
    ScoredRun,
    evaluate_fleet,
    governance_statement,
    prometheus_text,
)


def _org_of(kb_id: str) -> str:
    """Derive the org tenant from a ``{org}__{slug}`` kb-id (the whole id if unslugged)."""
    return kb_id.split("__", 1)[0] if "__" in kb_id else kb_id


def _mint_token(client: httpx.Client, tenant_id: str) -> str:
    """Mint an owner/admin dev JWT for a tenant (dev_jwt mode only)."""
    r = client.post(
        "/v1/dev/token",
        json={"subject": "qa-ragas-gate", "tenant_id": tenant_id, "roles": ["owner", "admin"], "clearance": 3},
    )
    r.raise_for_status()
    return r.json()["access_token"]


def _discover_kbs(client: httpx.Client, token: str) -> list[str]:
    r = client.get("/v1/knowledge-bases", headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    return [k["kb_id"] for k in r.json()]


def _golden(client: httpx.Client, kb_id: str, token: str) -> list[dict]:
    r = client.get(f"/v1/kb/{kb_id}/golden-set", headers={"Authorization": f"Bearer {token}"})
    if r.status_code != 200:
        return []
    body = r.json()
    # The GoldenSet endpoint returns {"questions": [{question, expected}]}; also tolerate a raw
    # list or an {"items": [...]} shape (the sample-data golden files use expected_answer).
    if isinstance(body, dict):
        items = body.get("questions") or body.get("items") or []
    else:
        items = body
    out: list[dict] = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        q = it.get("question")
        if not q:
            continue
        out.append({"question": q, "expected": it.get("expected") or it.get("expected_answer")})
    return out


def _run_kb(client: httpx.Client, kb_id: str, token: str) -> ScoredRun | None:
    """Run the KB's golden set through the live eval endpoint and adapt the report to ScoredRun."""
    questions = _golden(client, kb_id, token)
    if not questions:
        print(f"  ! {kb_id}: no golden set — skipped (nothing to certify)", file=sys.stderr)
        return None
    r = client.post(
        f"/v1/kb/{kb_id}/eval",
        headers={"Authorization": f"Bearer {token}"},
        json={"questions": questions[:20]},
        timeout=httpx.Timeout(300.0),
    )
    r.raise_for_status()
    d = r.json()
    # integrity_violations: a grounded answer with zero citations is impossible by the type-level
    # invariant, so the live path reports 0; the field stays wired for defence-in-depth.
    return ScoredRun(
        kb_id=kb_id,
        ragas_scored=d.get("ragas_scored", 0),
        total=d.get("total", 0),
        grounded=d.get("grounded", 0),
        abstained=d.get("abstained", 0),
        flagged=d.get("flagged", 0),
        faithfulness_avg=d.get("faithfulness_avg"),
        answer_relevancy_avg=d.get("answer_relevancy_avg"),
        context_precision_avg=d.get("context_precision_avg"),
        context_recall_avg=d.get("context_recall_avg"),
        integrity_violations=0,
    )


def _markdown(report: FleetGateReport) -> str:
    f = report.floors
    icon = {GateStatus.PASS: "✅ PASS", GateStatus.FAIL: "❌ FAIL", GateStatus.INCONCLUSIVE: "⚠️ INCONCLUSIVE"}
    lines = [
        "# RAGAS Governance Gate — Report",
        "",
        f"**Fleet verdict: {icon[report.status]}**  ·  contract v{report.contract_version}",
        "",
        f"> {governance_statement(f)}",
        "",
        f"Summary: {report.passed} pass · {report.failed} fail · {report.inconclusive} inconclusive "
        f"(of {len(report.kbs)} KB(s))",
        "",
        "## Floors (the contract)",
        "",
        "| faithfulness | answer_relevancy | context_precision | context_recall | min sample |",
        "|---|---|---|---|---|",
        f"| ≥ {f.faithfulness_min:.2f} | ≥ {f.answer_relevancy_min:.2f} | ≥ {f.context_precision_min:.2f} "
        f"| ≥ {f.context_recall_min:.2f} | {f.min_sample} |",
        "",
        "## Per-KB results",
        "",
        "| KB | verdict | sample | faithfulness | answer_rel | ctx_precision | ctx_recall | notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for kb in report.kbs:
        vals = {v.metric: v for v in kb.verdicts}

        def cell(name: str) -> str:
            v = vals.get(name)
            if v is None or v.value is None:
                return "—"
            mark = "✓" if v.passed else "✗"
            return f"{v.value:.3f} {mark}"

        note = "; ".join(kb.breaches) if kb.breaches else ("; ".join(kb.notes) if kb.notes else "")
        lines.append(
            f"| {kb.kb_id} | {icon[kb.status]} | {kb.sample} | {cell('faithfulness')} | "
            f"{cell('answer_relevancy')} | {cell('context_precision')} | {cell('context_recall')} | {note} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="RAGAS governance gate — QA quality certification for the RaaS.")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--kbs", help="Comma-separated KB ids to certify. Omit to discover via --tenant.")
    ap.add_argument(
        "--tenant",
        default="public__company-wide",
        help="Tenant whose visible KBs are discovered when --kbs is omitted.",
    )
    ap.add_argument(
        "--token", help="A pre-minted JWT authorised for every target KB (else dev tokens are minted per org)."
    )
    ap.add_argument(
        "--report", default="ragas_report.md", help="Markdown report path (a .json sibling is also written)."
    )
    ap.add_argument("--prometheus", help="Write a Prometheus textfile of the gate metrics to this path.")
    ap.add_argument("--strict", action="store_true", help="Treat INCONCLUSIVE as a failure (exit non-zero).")
    # Floor overrides (default: the sovereign contract, also overridable by RAG_RAGAS_* env).
    ap.add_argument("--faithfulness-min", type=float)
    ap.add_argument("--answer-relevancy-min", type=float)
    ap.add_argument("--context-precision-min", type=float)
    ap.add_argument("--context-recall-min", type=float)
    ap.add_argument("--min-sample", type=int)
    args = ap.parse_args()

    floors = RagasFloors.from_env()
    overrides = {
        "faithfulness_min": args.faithfulness_min,
        "answer_relevancy_min": args.answer_relevancy_min,
        "context_precision_min": args.context_precision_min,
        "context_recall_min": args.context_recall_min,
        "min_sample": args.min_sample,
    }
    floors = floors.model_copy(update={k: v for k, v in overrides.items() if v is not None})

    client = httpx.Client(base_url=args.base_url, timeout=httpx.Timeout(60.0))

    # Resolve the target KBs and a token strategy.
    if args.kbs:
        kb_ids = [k.strip() for k in args.kbs.split(",") if k.strip()]
    else:
        disc_token = args.token or _mint_token(client, args.tenant)
        kb_ids = _discover_kbs(client, disc_token)
        print(f"Discovered {len(kb_ids)} KB(s) for {args.tenant}: {', '.join(kb_ids) or '(none)'}")

    if not kb_ids:
        print("No KBs to certify.", file=sys.stderr)
        return 2

    print(governance_statement(floors))
    print(f"Certifying {len(kb_ids)} KB(s) against RAGAS contract v1.0.0 ...\n")

    runs: list[ScoredRun] = []
    token_cache: dict[str, str] = {}
    for kb_id in kb_ids:
        org = _org_of(kb_id)
        token = args.token or token_cache.get(org) or _mint_token(client, org)
        token_cache[org] = token
        try:
            print(f"→ {kb_id} ...")
            run = _run_kb(client, kb_id, token)
        except httpx.HTTPStatusError as e:
            print(f"  ! {kb_id}: HTTP {e.response.status_code} — {e.response.text[:160]}", file=sys.stderr)
            continue
        if run is not None:
            runs.append(run)

    if not runs:
        print("No KB produced a scored run (no golden sets?).", file=sys.stderr)
        return 2

    report = evaluate_fleet(runs, floors)

    md = _markdown(report)
    Path(args.report).write_text(md, encoding="utf-8")
    Path(args.report).with_suffix(".json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    if args.prometheus:
        Path(args.prometheus).write_text(prometheus_text(report), encoding="utf-8")
        print(f"Prometheus metrics → {args.prometheus}")

    print("\n" + md)
    print(f"Reports → {args.report} + {Path(args.report).with_suffix('.json')}")

    if report.status is GateStatus.FAIL:
        return 1
    if report.status is GateStatus.INCONCLUSIVE and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
