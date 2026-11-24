"""Federated RAG concurrency load + soak harness — QA-runnable, HTTP-driven.

Complementary to ``scripts/benchmark.py`` (federated scaling + vector-backend compare, one
request at a time, N repeats). THIS script answers a different question: what happens when
MANY callers hit the platform AT ONCE, and does it hold up over TIME? Two load shapes, both
over a running stack via ``httpx.AsyncClient``:

  query load   — fire concurrent grounded queries at a KB (a fixed ``--requests`` count, or a
                 sustained ``--soak SECONDS`` window) from a pool of query strings. Reports
                 p50/p95/p99 latency, throughput (req/s), error rate, AND the grounded /
                 abstained / flagged_for_review mix — quality-under-load, not just speed.
                 Exits non-zero if ``--max-p95-ms`` or ``--max-error-rate`` is breached.
  ingest load  — optional (``--ingest N``): concurrently ingest N small text docs into a
                 TEMPORARY knowledge base (created here, purged after unless ``--keep``),
                 measure ingest latency + success rate, then confirm no data corruption by
                 retrieving each successfully-ingested doc back by a unique marker (uses the
                 no-generation ``/retrieve`` endpoint — gentle on the LLM).

Soak mode (``--soak``) additionally reports latency DRIFT: p95 of the first quartile of the
run (chronological) vs the last quartile, to catch degradation/leaks a short burst would miss.

The pure percentile/summary statistics are reused from ``rag_core.benchmark`` (already
unit-tested); this script only drives HTTP calls and formats the report. Writes nothing except
the optional ``--report`` file.

Usage (stack up):
    uv run python scripts/load_soak.py --kb public__company-wide --concurrency 10 --requests 100
    uv run python scripts/load_soak.py --kb public__company-wide --concurrency 10 --soak 300
    uv run python scripts/load_soak.py --kb public__company-wide --requests 0 --ingest 20
    uv run python scripts/load_soak.py --kb public__company-wide --concurrency 5 --requests 20 \\
        --ingest 10 --report load_soak.md
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import random
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from rag_core.benchmark import LatencySummary, summarize

_HTTP_OK = 200
_MIN_SOAK_SAMPLES = 8  # below this a first/last-quartile split isn't meaningful

# A sensible domain-neutral fallback pool — used when neither --queries-file nor
# --use-golden-set (or the KB's golden set is empty) supplies real questions. Deliberately
# generic so it never 404s/abstains-by-construction on an arbitrary KB.
_BUILTIN_QUERIES = [
    "What is the platform's uptime target?",
    "What is the data retention policy?",
    "How is data classified?",
    "What security controls are in place?",
    "What is the leave policy?",
    "How often should passwords be rotated?",
]


@dataclass
class Sample:
    t_start: float
    latency_ms: float
    ok: bool
    label: str  # ok: answer status (grounded/abstained/flagged_for_review); else "http_NNN" / "error:Type"


@dataclass
class IngestSample:
    doc_id: str
    latency_ms: float
    ok: bool
    error: str | None = None


@dataclass
class IngestPhaseResult:
    kb_id: str
    samples: list[IngestSample] = field(default_factory=list)
    verified: int = 0
    verify_failed: list[str] = field(default_factory=list)
    cleaned_up: bool = False


# --- HTTP helpers -------------------------------------------------------------
async def _mint_token(
    client: httpx.AsyncClient, *, subject: str, tenant_id: str, roles: list[str], clearance: int
) -> str:
    r = await client.post(
        "/v1/dev/token",
        json={"subject": subject, "tenant_id": tenant_id, "roles": roles, "groups": [], "clearance": clearance},
    )
    r.raise_for_status()
    return str(r.json()["access_token"])


def _parse_queries_file(raw: str) -> list[str]:
    try:
        data = json.loads(raw)
        return [str(q) for q in data] if isinstance(data, list) else [str(raw)]
    except json.JSONDecodeError:
        return [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith("#")]


async def _load_query_pool(
    client: httpx.AsyncClient, *, kb: str, queries_file_text: str | None, use_golden_set: bool
) -> list[str]:
    pool = _parse_queries_file(queries_file_text) if queries_file_text is not None else list(_BUILTIN_QUERIES)
    if use_golden_set:
        r = await client.get(f"/v1/kb/{kb}/golden-set")
        if r.status_code == _HTTP_OK:
            questions = [q["question"] for q in r.json().get("questions", [])]
            pool.extend(questions)
    return pool or list(_BUILTIN_QUERIES)


# --- query load ----------------------------------------------------------------
async def run_query_phase(
    client: httpx.AsyncClient,
    *,
    kb: str,
    pool: list[str],
    top_k: int,
    model: str | None,
    concurrency: int,
    total_requests: int,
    soak_seconds: float | None,
    rng: random.Random,
) -> tuple[list[Sample], float]:
    """Fire concurrent grounded queries; return (samples, wall-clock duration seconds)."""
    samples: list[Sample] = []
    sem = asyncio.Semaphore(concurrency)

    async def _one() -> None:
        q = rng.choice(pool)
        body: dict[str, object] = {"query": q, "top_k": top_k}
        if model:
            body["model"] = model
        t0 = time.perf_counter()
        try:
            r = await client.post(f"/v1/kb/{kb}/query", json=body)
            dt = (time.perf_counter() - t0) * 1000.0
            if r.status_code == _HTTP_OK:
                samples.append(Sample(t0, dt, True, r.json().get("status", "unknown")))
            else:
                samples.append(Sample(t0, dt, False, f"http_{r.status_code}"))
        except Exception as exc:  # network error, timeout, etc — counts against the error rate
            dt = (time.perf_counter() - t0) * 1000.0
            samples.append(Sample(t0, dt, False, f"error:{type(exc).__name__}"))

    start = time.perf_counter()
    if soak_seconds is not None:
        end = start + soak_seconds

        async def _soak_worker() -> None:
            while time.perf_counter() < end:
                async with sem:
                    await _one()

        await asyncio.gather(*(_soak_worker() for _ in range(concurrency)))
    else:
        base, rem = divmod(total_requests, concurrency)
        counts = [base + (1 if i < rem else 0) for i in range(concurrency)]

        async def _fixed_worker(n: int) -> None:
            for _ in range(n):
                async with sem:
                    await _one()

        await asyncio.gather(*(_fixed_worker(n) for n in counts if n))
    duration = time.perf_counter() - start
    return samples, duration


def quartile_drift(samples: list[Sample]) -> tuple[LatencySummary, LatencySummary] | None:
    """p95 of the first vs last chronological quartile of a soak run (drift signal)."""
    ordered = sorted(samples, key=lambda s: s.t_start)
    n = len(ordered)
    if n < _MIN_SOAK_SAMPLES:  # too few samples for a meaningful quartile split
        return None
    q = n // 4
    first = summarize([s.latency_ms for s in ordered[:q]])
    last = summarize([s.latency_ms for s in ordered[-q:]])
    return first, last


# --- ingest load -----------------------------------------------------------------
async def run_ingest_phase(
    client: httpx.AsyncClient,
    *,
    ingest_count: int,
    concurrency: int,
    tenant_prefix: str,
    clearance: int,
    keep: bool,
    run_id: str,
) -> IngestPhaseResult:
    tenant_id = f"{tenant_prefix}-{run_id}"
    display_name = f"load-soak-{run_id}"
    token = await _mint_token(
        client,
        subject="load-soak-ingest",
        tenant_id=tenant_id,
        roles=["owner", "admin", "ingestor", "reviewer"],
        clearance=clearance,
    )
    headers = {"Authorization": f"Bearer {token}"}
    r = await client.post("/v1/knowledge-bases", json={"display_name": display_name}, headers=headers)
    r.raise_for_status()
    kb_id = r.json()["kb_id"]
    result = IngestPhaseResult(kb_id=kb_id)

    sem = asyncio.Semaphore(concurrency)

    async def _ingest_one(i: int) -> None:
        doc_id = f"loadsoak-doc-{i}"
        marker = f"LOADSOAK-{run_id}-{i}"
        text = (
            f"Load-soak test document {i}. Unique marker: {marker}. "
            "This is synthetic content generated by scripts/load_soak.py for ingest-path "
            "load testing; it carries no real information."
        )
        content_b64 = base64.b64encode(text.encode()).decode()
        t0 = time.perf_counter()
        async with sem:
            try:
                r2 = await client.post(
                    f"/v1/kb/{kb_id}/ingest",
                    json={"doc_id": doc_id, "mime": "text/plain", "content_b64": content_b64},
                    headers=headers,
                )
                dt = (time.perf_counter() - t0) * 1000.0
                if r2.status_code == _HTTP_OK:
                    result.samples.append(IngestSample(doc_id, dt, True))
                else:
                    result.samples.append(IngestSample(doc_id, dt, False, f"http_{r2.status_code}: {r2.text[:150]}"))
            except Exception as exc:
                dt = (time.perf_counter() - t0) * 1000.0
                result.samples.append(IngestSample(doc_id, dt, False, f"{type(exc).__name__}: {exc}"))

    await asyncio.gather(*(_ingest_one(i) for i in range(ingest_count)))

    # Corruption check: each successfully-ingested doc must be retrievable by its own unique
    # marker. Uses /retrieve (no generation) — cheap, and directly proves the write landed.
    for s in result.samples:
        if not s.ok:
            continue
        i = int(s.doc_id.rsplit("-", 1)[-1])
        marker = f"LOADSOAK-{run_id}-{i}"
        try:
            rr = await client.post(f"/v1/kb/{kb_id}/retrieve", json={"query": marker, "top_k": 3}, headers=headers)
            spans = rr.json().get("spans", []) if rr.status_code == _HTTP_OK else []
            if any(sp.get("doc_id") == s.doc_id for sp in spans):
                result.verified += 1
            else:
                result.verify_failed.append(s.doc_id)
        except Exception:
            result.verify_failed.append(s.doc_id)

    if not keep:
        try:
            await client.post(
                "/v1/knowledge-bases/" + kb_id + "/purge",
                json={"confirm_name": display_name},
                headers=headers,
            )
            result.cleaned_up = True
        except Exception:
            result.cleaned_up = False
    return result


# --- report formatting ---------------------------------------------------------
def _pct_table(rows: list[tuple[str, LatencySummary]]) -> str:
    lines = [
        "| | samples | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) | min (ms) | max (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, s in rows:
        lines.append(
            f"| {label} | {s.count} | {s.p50:.1f} | {s.p95:.1f} | {s.p99:.1f} | "
            f"{s.mean:.1f} | {s.minimum:.1f} | {s.maximum:.1f} |"
        )
    return "\n".join(lines)


def _counts_table(header: str, counts: Counter[str], total: int) -> str:
    if not counts:
        return f"_{header}: none_\n"
    lines = [f"| {header} | count | % |", "|---|---|---|"]
    for label, n in counts.most_common():
        lines.append(f"| {label} | {n} | {(n / total * 100.0):.1f}% |")
    return "\n".join(lines)


def build_report(
    *,
    kb: str,
    mode_label: str,
    samples: list[Sample],
    duration_s: float,
    max_p95_ms: float,
    max_error_rate: float,
    ingest: IngestPhaseResult | None,
    ingest_max_error_rate: float,
) -> tuple[str, bool]:
    out = ["# Load + soak report", "", f"KB: `{kb}` — mode: {mode_label}", ""]
    query_pass = True

    if samples:
        total = len(samples)
        oks = [s for s in samples if s.ok]
        errors = [s for s in samples if not s.ok]
        error_rate = len(errors) / total
        summary = summarize([s.latency_ms for s in samples])
        throughput = total / duration_s if duration_s > 0 else 0.0
        query_pass = summary.p95 <= max_p95_ms and error_rate <= max_error_rate

        out += [
            "## Query load",
            "",
            f"- requests: {total} — duration: {duration_s:.1f}s — throughput: **{throughput:.2f} req/s**",
            f"- error rate: **{error_rate * 100:.2f}%** (threshold {max_error_rate * 100:.1f}%)",
            "",
            _pct_table([("latency", summary)]),
            "",
            "### Answer status mix (quality under load)",
            "",
            _counts_table("status", Counter(s.label for s in oks), total),
            "",
        ]
        if errors:
            out += ["### Errors", "", _counts_table("error", Counter(s.label for s in errors), total), ""]

        drift = quartile_drift(samples)
        if drift is not None:
            first, last = drift
            delta = last.p95 - first.p95
            pct = (delta / first.p95 * 100.0) if first.p95 > 0 else 0.0
            out += [
                "### Soak drift (first quartile vs last quartile, chronological)",
                "",
                _pct_table([("first 25%", first), ("last 25%", last)]),
                "",
                f"- p95 drift: **{delta:+.1f} ms ({pct:+.1f}%)** — positive means it got slower over the run.",
                "",
            ]

        verdict = "PASS" if query_pass else "FAIL"
        out += [
            f"### Query verdict: **{verdict}**",
            "",
            f"- p95 {summary.p95:.1f}ms {'<=' if summary.p95 <= max_p95_ms else '>'} max {max_p95_ms:.1f}ms",
            f"- error rate {error_rate * 100:.2f}% {'<=' if error_rate <= max_error_rate else '>'} "
            f"max {max_error_rate * 100:.1f}%",
            "",
        ]
    else:
        out += ["## Query load", "", "_skipped (--requests 0 and no --soak)_", ""]

    ingest_pass = True
    if ingest is not None:
        total_i = len(ingest.samples)
        ok_i = [s for s in ingest.samples if s.ok]
        err_i = [s for s in ingest.samples if not s.ok]
        error_rate_i = (len(err_i) / total_i) if total_i else 0.0
        corruption = bool(ingest.verify_failed)
        ingest_pass = error_rate_i <= ingest_max_error_rate and not corruption
        i_summary = summarize([s.latency_ms for s in ingest.samples])
        out += [
            "## Ingest load",
            "",
            f"- temp KB: `{ingest.kb_id}` (cleaned up: {ingest.cleaned_up})",
            f"- documents: {total_i} — succeeded: {len(ok_i)} — failed: {len(err_i)} "
            f"({error_rate_i * 100:.2f}% error rate)",
            f"- corruption check: {ingest.verified}/{len(ok_i)} ingested docs retrievable by their own content"
            + (f" — **{len(ingest.verify_failed)} FAILED**: {ingest.verify_failed}" if corruption else " — clean"),
            "",
            _pct_table([("ingest latency", i_summary)]),
            "",
        ]
        if err_i:
            out += ["### Ingest errors (first 10)", ""]
            for s in err_i[:10]:
                out.append(f"- `{s.doc_id}`: {s.error}")
            out.append("")
        out += [f"### Ingest verdict: **{'PASS' if ingest_pass else 'FAIL'}**", ""]

    overall_pass = query_pass and ingest_pass
    out += ["## Overall verdict", "", f"# {'PASS' if overall_pass else 'FAIL'}", ""]
    return "\n".join(out), overall_pass


async def _execute(
    args: argparse.Namespace, *, run_id: str, tenant_id: str, roles: list[str], queries_file_text: str | None
) -> tuple[str, bool]:
    async with httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout) as client:
        samples: list[Sample] = []
        duration = 0.0
        if args.requests > 0 or args.soak is not None:
            token = args.token or await _mint_token(
                client, subject=args.subject, tenant_id=tenant_id, roles=roles, clearance=args.clearance
            )
            client.headers["Authorization"] = f"Bearer {token}"
            pool = await _load_query_pool(
                client, kb=args.kb, queries_file_text=queries_file_text, use_golden_set=args.use_golden_set
            )
            rng = random.Random(args.seed)  # noqa: S311 — load-test query sampling, not cryptographic use
            samples, duration = await run_query_phase(
                client,
                kb=args.kb,
                pool=pool,
                top_k=args.top_k,
                model=args.model,
                concurrency=args.concurrency,
                total_requests=args.requests,
                soak_seconds=args.soak,
                rng=rng,
            )

        ingest_result: IngestPhaseResult | None = None
        if args.ingest > 0:
            ingest_result = await run_ingest_phase(
                client,
                ingest_count=args.ingest,
                concurrency=args.ingest_concurrency or min(args.concurrency, 10),
                tenant_prefix=args.ingest_tenant_prefix,
                clearance=args.clearance,
                keep=args.keep,
                run_id=run_id,
            )

        mode_label = f"soak {args.soak:.0f}s" if args.soak is not None else f"{args.requests} requests"
        mode_label += f", concurrency {args.concurrency}"
        return build_report(
            kb=args.kb,
            mode_label=mode_label,
            samples=samples,
            duration_s=duration,
            max_p95_ms=args.max_p95_ms,
            max_error_rate=args.max_error_rate,
            ingest=ingest_result,
            ingest_max_error_rate=args.ingest_max_error_rate,
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--kb", required=True, help="KB to load-test queries against")
    p.add_argument("--token", default=None, help="existing bearer token for --kb (else a dev token is minted)")
    p.add_argument("--tenant-id", default=None, help="dev-token tenant_id (default: derived from --kb's prefix)")
    p.add_argument("--subject", default="load-soak")
    p.add_argument("--roles", default="reader,ingestor,reviewer")
    p.add_argument("--clearance", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--requests", type=int, default=50, help="fixed request count (ignored if --soak is set)")
    p.add_argument("--soak", type=float, default=None, metavar="SECONDS", help="run sustained load for N seconds")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--model", default=None, help="per-question model override")
    p.add_argument("--queries-file", default=None, help="newline- or JSON-list file of query strings")
    p.add_argument("--use-golden-set", action="store_true", help="mix in the KB's saved golden-set questions")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-p95-ms", type=float, default=8000.0, help="generous default: generation calls an LLM")
    p.add_argument("--max-error-rate", type=float, default=0.02)
    p.add_argument("--ingest", type=int, default=0, metavar="N", help="also ingest-load-test N docs into a temp KB")
    p.add_argument("--ingest-concurrency", type=int, default=None, help="default: min(--concurrency, 10)")
    p.add_argument("--ingest-tenant-prefix", default="loadtest")
    p.add_argument("--ingest-max-error-rate", type=float, default=0.02)
    p.add_argument("--keep", action="store_true", help="keep the temp ingest KB instead of purging it")
    p.add_argument("--report", default=None, help="also write the markdown report here")
    p.add_argument("--timeout", type=float, default=120.0, help="per-request HTTP timeout (seconds)")
    args = p.parse_args()
    if args.requests < 0:
        p.error("--requests must be >= 0")
    if args.soak is not None and args.soak <= 0:
        p.error("--soak must be > 0")
    return args


def main() -> int:
    args = _parse_args()
    run_id = uuid.uuid4().hex[:8]
    tenant_id = args.tenant_id or (args.kb.split("__", 1)[0] if "__" in args.kb else args.kb)
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    queries_file_text = Path(args.queries_file).read_text() if args.queries_file else None

    report, passed = asyncio.run(
        _execute(args, run_id=run_id, tenant_id=tenant_id, roles=roles, queries_file_text=queries_file_text)
    )
    print(report)
    if args.report:
        Path(args.report).write_text(report)
        print(f"wrote {args.report}", file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
