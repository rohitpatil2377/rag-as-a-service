# Load & Soak Testing — QA runbook

**What it is.** A concurrency load + soak harness that answers a question
`scripts/benchmark.py` deliberately does not: not "how does latency scale with N federated
peers" or "which vector backend is faster," but **"what happens when MANY callers hit the
platform AT ONCE, and does it hold up over TIME?"** HTTP-driven (`httpx.AsyncClient`) against a
running stack, same house style as `benchmark.py` (argparse, dev-token mint, pure stats reused
from `rag_core.benchmark`).

- Harness: `scripts/load_soak.py`
- Pure stats (percentile/summarize, unit-tested): `libs/rag-core/src/rag_core/benchmark.py`

## What it measures

**Query load** — fires concurrent grounded queries (`POST /v1/kb/{id}/query`) at a KB from a
pool of query strings, either a fixed `--requests` count or a sustained `--soak SECONDS`
window. Reports:
- p50/p95/p99 latency, throughput (req/s), error rate
- the **grounded / abstained / flagged_for_review mix** — quality-under-load, not just speed;
  a stack that answers fast but abstains on everything under load is not actually healthy
- (soak mode only) **latency drift**: p95 of the first chronological quartile of the run vs
  the last quartile, to catch degradation/leaks a short burst would miss

**Ingest load** (optional, `--ingest N`) — concurrently ingests N small `text/plain` documents
into a **temporary knowledge base** (created for the run, purged afterward unless `--keep`),
measures ingest latency + success rate, then — the corruption check — retrieves each
successfully-ingested document back by a unique marker string via the no-generation
`POST /v1/kb/{id}/retrieve` endpoint (cheap: no LLM call) to confirm the write actually landed
and is queryable, not just "the API returned 200."

**Verdict**: PASS only when the query phase clears `--max-p95-ms` (default 8000ms — generous,
because grounded generation calls an LLM) and `--max-error-rate` (default 2%), AND (if run)
the ingest phase clears its own error-rate threshold with zero corruption-check failures.
Non-zero exit code on any breach — a CI/release gate, not just a printout.

## Run it (stack up)

```bash
# Query load: 100 requests, 10 concurrent
uv run python scripts/load_soak.py --kb public__company-wide --concurrency 10 --requests 100

# Soak: sustained load for 5 minutes, watch for drift
uv run python scripts/load_soak.py --kb public__company-wide --concurrency 10 --soak 300

# Ingest load only (skip query phase): 20 docs into a temp KB, concurrency 5
uv run python scripts/load_soak.py --kb public__company-wide --requests 0 --ingest 20

# Both, plus a saved markdown report:
uv run python scripts/load_soak.py --kb public__company-wide --concurrency 10 --requests 100 \
    --ingest 20 --report load_soak.md

# Sample real questions from the KB's saved golden set instead of the generic built-in pool:
uv run python scripts/load_soak.py --kb acme__hr --use-golden-set --concurrency 10 --requests 50
```

If `uv` isn't on PATH, use the venv the API container itself runs (`/app/.venv/bin/python` —
already has `httpx` + `rag_core`), or any Python with `httpx` installed and the workspace
packages on `sys.path`.

`--token` accepts an existing bearer token in place of minting a dev token (needed outside
`RAG_AUTH_MODE=dev_jwt`). The dev-token tenant defaults to the org prefix of `--kb`
(`acme__hr` → tenant `acme`) so `tenant_guard`'s prefix-ownership check passes without having
to specify it by hand; override with `--tenant-id`.

## Reading the output

- **Latency table** — p50/p95/p99/mean/min/max in ms, over all completed requests (successes
  AND failures both count toward latency, since a slow failure is still latency the caller felt).
- **Answer status mix** — count/% of `grounded` / `abstained` / `flagged_for_review` among
  successful (HTTP 200) responses. A high abstain rate under load with a low abstain rate at
  low concurrency would point at retrieval/generation timing out under contention, not a
  genuine "no answer in the corpus" case — worth a second run at low concurrency to compare.
- **Errors table** — non-200 responses (`http_429`, `http_5xx`, …) and client-side exceptions
  (`error:ConnectTimeout`, …), each as a fraction of total requests (this is the error rate the
  threshold checks against).
- **Soak drift table** (soak mode, ≥8 samples) — p95 of the first vs last chronological
  quartile. A large **positive** delta (later requests are slower) is the signature of a leak
  or resource exhaustion under sustained load; a negative delta (as seen in the live run below)
  usually means warm-up cost (first requests pay a cold cache/connection-pool penalty) rather
  than degradation — read the sign, don't just alarm on any nonzero drift.
- **Ingest section** — per-doc latency, success rate, and the corruption-check line
  (`N/M ingested docs retrievable`); any corruption-check failure is listed by `doc_id` and
  fails the run regardless of the error-rate threshold — data landing wrong is worse than a
  slow request.

## SLO thresholds and rationale

| Flag | Default | Rationale |
|---|---|---|
| `--max-p95-ms` | 8000 | Grounded generation calls a real LLM (this platform's CPU/shared-GPU tiers, not a hosted low-latency API) — a tight web-request SLO would fail on hardware alone, not on a real regression. 8s is generous headroom; tighten it per-deployment once a baseline is established (see the live numbers below). |
| `--max-error-rate` | 2% | Zero-tolerance is unrealistic under concurrent load against a shared LLM/rerank backend with its own timeouts; 2% catches a real regression (crashes, 5xxs, connection resets) without false-alarming on the occasional transient timeout an isolated retry would absorb. |
| `--ingest-max-error-rate` | 2% | Same rationale, ingest side. |

Both thresholds are overridable per run — tune them once you have a real baseline for a given
deployment's hardware tier, and treat a drift from *that* baseline (not the generic defaults)
as the actionable regression signal.

**Per-org rate limits interact with this test.** Quota is enforced per **organisation**, over a
sliding 60s window (`services/control-plane/src/svc_control_plane/quota.py`), default
`quota_queries_per_minute=60` / `quota_ingests_per_minute=30`
(`libs/common/src/rag_common/config.py:672-675`). A single load-test run mints one dev token
per phase, so every request in that phase shares one org's bucket — pushing `--requests` /
`--concurrency` high enough to exceed ~60 queries/min (or ~30 ingests/min) will legitimately
trip `429 rate_limited`, which the harness correctly counts as an error. That is real platform
behavior (noisy-neighbour isolation), not a harness bug — for a higher-scale run either raise
`RAG_QUOTA_QUERIES_PER_MINUTE`/`RAG_QUOTA_INGESTS_PER_MINUTE` on the target deployment first,
or treat the 429s as the actual capacity ceiling being measured.

## Live verification (this deployment, 2026-08-31)

Ran against the live stack at `http://localhost:8000`, `--kb public__company-wide`:

| Run | Result |
|---|---|
| `--concurrency 2 --requests 4` (smoke) | p95 **22798ms**, 0% errors — FAIL vs an intentionally tight 20000ms cap (harness correctly failed) |
| `--concurrency 5 --requests 20` (the QA example above) | p50 **5937ms**, p95 **19171ms**, p99 **23272ms**, throughput 0.54 req/s, **0% error rate**, 40% grounded / 60% abstained — **FAIL** vs the default 8000ms p95 cap |
| `--concurrency 2 --soak 20` | p95 **4034ms**, 0% errors — **PASS** vs a relaxed 30000ms cap; drift −9.4% (no degradation over 20s) |
| `--requests 0 --ingest 5 --ingest-concurrency 3` | 5/5 docs ingested, 5/5 corruption-check clean, temp KB purged — **PASS** |

**Reading the numbers honestly**: this VM serves generation through a shared/CPU-tier LLM
(`sarvam-105b`, see root `CLAUDE.md` "Current state"), so p95 latencies in the 19–23 second
range at concurrency 5 are a real property of this deployment's hardware tier under
concurrent generation load, not a harness defect — zero requests errored or timed out at any
concurrency level tried. The default `--max-p95-ms 8000` is calibrated for a lower-latency
generation backend; **on this specific box a realistic p95 SLO is closer to 20-25 seconds at
concurrency ≥5** — use `--max-p95-ms 25000` (or similar) here, and treat the default 8000ms as
the target for a faster (GPU/vLLM) generation tier. The ingest and corruption-check paths
passed cleanly at every scale tried, and the temporary KB was confirmed fully purged
(`GET /v1/knowledge-bases` and `GET /v1/knowledge-bases/trash` both came back clean) after each
ingest-load run — no residual data left behind.

Every run above used deliberately modest scale (`--concurrency` 2–5, `--requests`/`--ingest`
≤20) specifically to stay gentle on the shared live LLM/embedding endpoint per this box's
operating constraints — the numbers above are a smoke-scale baseline, not a stress-test
ceiling. Push `--concurrency`/`--requests`/`--soak` higher for an actual capacity test, mindful
of the per-org rate limits above.

## Ingest-failure / retry-path robustness (honest findings)

Two distinct ingest paths exist, with **different** robustness properties:

**1. Direct single-document ingest (`POST /v1/kb/{id}/ingest`, api.py:836)** — fully
synchronous: the request blocks through parse → mask → chunk → embed → index and returns
200 or an error in the same HTTP response. **There is no retry, queue, or dead-letter here at
all** — a transient failure (a momentary embedder/vector-store timeout, say) simply returns a
non-2xx response to the caller; retrying is entirely the caller's responsibility. This is what
`load_soak.py`'s `--ingest` mode exercises.

**2. Bulk ingest jobs (`POST /kb/{id}/ingest/jobs` + `/items` + `/start`,
`services/control-plane/src/svc_control_plane/ingest_jobs.py`)** — an async, persisted job
queue drained by an in-process worker, built for "ingest 1000+ documents without blocking the
request" (module docstring, `ingest_jobs.py:1-17`). Its actual robustness properties, read
directly from the code:

- **Per-item failures are captured, not silently dropped.** `run_job`'s `_one()` wraps each
  item's ingest in `try/except`, and any exception marks that item `status='failed'` with the
  error text truncated to 200 chars, bumps the job's `failed` counter, and logs a warning
  (`ingest_jobs.py:469-488`) — one bad document never aborts the batch. Failures are visible
  via `GET /kb/{id}/ingest/jobs/{job_id}` (`JobView.failures`, up to 10 shown,
  `ingest_jobs.py:90-94, 428-432`).
- **No automatic retry or backoff.** A failed item stays `failed` **forever** until a human (or
  script) explicitly calls `POST /kb/{id}/ingest/jobs/{job_id}/retry`
  (`api.py:1996-2007` → `ingest_jobs.retry_failed`, `ingest_jobs.py:187-197` in-memory /
  `349-368` Postgres), which flips failed items back to `pending` and re-queues the job. There
  is **no exponential backoff, no retry-count cap, and no scheduled/automatic re-attempt** —
  if nobody calls the retry endpoint (or clicks it in the console), a failed document simply
  sits `failed` indefinitely. Honest characterization: **manually-retryable, not
  auto-retried** — closer to "visible failure queue" than "resilient auto-retry."
- **No dead-letter queue.** There is no separate DLQ table, no max-retry-then-park behavior,
  and no alerting on a stuck-failed item — `retry_failed` can be called an unbounded number of
  times on the same item. A document that will *always* fail (e.g. corrupt bytes) stays visible
  in the same `failed` state on every retry, distinguishable only by reading its `error` text.
- **Crash recovery does not lose or duplicate work (best-effort).** If the worker process dies
  mid-job, the job is stuck `status='running'`; the next replica to acquire the `bulk_ingest`
  leader lease calls `requeue_running()` once per leadership term
  (`ingest_jobs.py:516-523`), flipping it back to `queued` so pending items (anything not yet
  marked `done`) resume. Staged bytes for a not-yet-successful item are **not deleted** until
  after `pipeline.ingest_document` returns (`ingest_jobs.py:453-457`), so a crash before that
  point leaves the staged copy in the object store for the resumed attempt to read — no data
  loss on that path. A resumed re-ingest of already-`done` items does not re-run (only
  `pending_items()` are drained, `ingest_jobs.py:298-306, 160-161`), and A8 content-hash
  dedup makes even an accidental re-ingest of the same bytes idempotent
  (`outcome="unchanged"`, see root `CLAUDE.md`).
- **A job's overall `status` can mask partial failure.** `run_job` sets the job to `"failed"`
  only when **everything** failed (`final.failed and not final.done`,
  `ingest_jobs.py:491-493`) — a batch where, say, 95/100 documents succeeded and 5 failed is
  reported `status="completed"`, not `"failed"`. The per-item `failed` counter is accurate and
  visible, but a caller checking only the top-level job `status` (not `failed` count) would
  miss a partial-failure batch. This is a real gap worth flagging to whoever consumes job
  status programmatically — check `job.failed`, not just `job.status`.
- **Multi-replica**: exactly one replica drains the queue at a time via a Redis leader lease
  (`_is_leader`, `ingest_jobs.py:45-50`) when `RAG_REDIS_URL` is set — a documented,
  already-implemented safeguard against double-ingesting an item across horizontally-scaled
  API replicas; single-process deployments are trivially "always leader."

**Bottom line**: failed ingests are neither silently dropped (they're persisted, visible, and
manually retryable with no cap) nor automatically retried with backoff (no scheduler drives
`retry_failed` — it is purely operator/console/API-triggered), and there is no dead-letter
queue or alerting on a stuck-failed item. For a QA/ops runbook: **after any bulk ingest job,
check `job.failed > 0` explicitly** (not just `job.status`), and if `failed > 0`, either
inspect `JobView.failures` and fix the root cause before retrying, or call the retry endpoint
directly — nothing does either of those automatically today.
