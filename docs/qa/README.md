# RAG as a Service — QA Test Program

The evidence pack a QA lead (or a regulated buyer's assurance team) runs to certify the
platform against its own two promises — **grounded-or-abstain** and **sovereignty** — plus the
operational qualities a production RaaS must hold: observability, federation integrity,
regulatory compliance, and behaviour under load.

Every item below is a **runnable command with a PASS/FAIL verdict** and a report artifact —
not a claim. The spine is the **RAGAS governance gate**, which turns the grounded-or-abstain
promise into a continuously measured, enforceable contract.

## Prerequisites

- The stack is up (`make up`) and healthy: `curl -s localhost:8000/health` → `200`;
  Grafana `localhost:3000`, Prometheus `localhost:9090`.
- A Python env with the workspace installed (`uv run …` as the Makefile targets use). Scripts
  are HTTP-driven and can also run from inside the `api` container's venv
  (`/app/.venv/bin/python`) if `uv` is not on the host.
- Dev-JWT mode (default here) lets the scripts mint their own tokens. In a hardened
  deployment pass `--token <jwt>` authorised for the target KBs.

## The suite

| # | Area | Command | Proves | Runbook |
|---|---|---|---|---|
| 1 | **RAGAS quality gate** (the spine) | `make ragas ARGS="--tenant public__company-wide"` | Grounded-or-abstain, measured: faithfulness / answer-relevancy / context-precision / context-recall clear the published floors on the platform's **own** models | [RAGAS_GATE.md](RAGAS_GATE.md) |
| 2 | **Federation integrity** | `uv run python scripts/federation_qa.py` | Ed25519 grants: issue → cross-dept ask returns real citations → forged/tampered grant rejected → post-revoke fail-closed | [FEDERATION_QA.md](FEDERATION_QA.md) |
| 3 | **Observability** | `make ragas ARGS="--prometheus infra/observability/textfile/ragas.prom …"` then open Grafana | Quality/SLO metrics are continuously visible; the gate result lands on a dashboard | [OBSERVABILITY.md](OBSERVABILITY.md) |
| 4 | **Compliance evidence** | `uv run python scripts/compliance_evidence.py --kb public__company-wide` | 7 frameworks (NITI, DPDP, ISO 42001, OWASP-LLM, RBI, SEBI, IRDAI) mapped to live controls + signed residency attestation + audit-chain integrity | [COMPLIANCE_EVIDENCE.md](COMPLIANCE_EVIDENCE.md) |
| 5 | **Scale & robustness** | `uv run python scripts/load_soak.py --kb public__company-wide --concurrency 5 --requests 20` | Latency percentiles, error rate, quality-under-load, ingest integrity, soak drift, against SLO thresholds | [LOAD_TEST.md](LOAD_TEST.md) |

### Run the whole program

```bash
# 1. Quality gate (governs the RaaS) — expand each KB's golden set to >=8 first (see RAGAS_GATE.md)
make ragas ARGS="--kbs acme__hr,acme__finance,public__company-wide --strict \
    --prometheus infra/observability/textfile/ragas.prom --report artifacts/ragas.md"

# 2. Federation lifecycle proof
uv run python scripts/federation_qa.py --base-url http://localhost:8000

# 3. Observability — the gate above wrote the metric; confirm it in Prometheus/Grafana
#    (see OBSERVABILITY.md: query rag_ragas_fleet_status, open the "RAGAS Governance" dashboard)

# 4. Compliance evidence pack (per KB)
uv run python scripts/compliance_evidence.py --kb public__company-wide --out artifacts/compliance.md

# 5. Load + soak (be gentle on a CPU-tier LLM — see LOAD_TEST.md for realistic thresholds)
uv run python scripts/load_soak.py --kb public__company-wide --concurrency 5 --requests 20 \
    --max-p95-ms 25000 --report artifacts/load.md
```

Each command exits non-zero on a breach, so the sequence doubles as a CI/release gate.

## What each result means for the product

- **#1 is the headline.** It is the measured, enforceable form of "grounded-or-abstain": a KB
  is certified only when its answers clear every RAGAS floor over a sufficient golden sample,
  with zero grounded-but-uncited answers. The scheduler (FR-G9) runs the same contract
  unattended and alerts on drift — quality is *governed*, not asserted.
- **#2** is the differentiating sovereign-federation claim, reduced to four assertions a
  skeptic can run: a grant works, a forged one is refused, and revocation is immediate and
  fail-closed.
- **#3** turns "we have telemetry" into "you can watch the quality bar in real time."
- **#4** is the regulator-facing evidence: the mapping from live controls to DPDP/RBI/SEBI/
  IRDAI/ISO/NITI/OWASP, backed by a cryptographically verified audit chain and signed
  residency attestation.
- **#5** validates that the promises hold under concurrency and over time, not just on a
  single request.

## Honest known gaps (as of this program)

These are documented, not hidden — each has an owner action:

1. **Golden-set depth.** The strict RAGAS contract needs **≥ 8** golden questions per KB to
   certify (PASS); a thin set correctly returns **INCONCLUSIVE**, never a false PASS. Expanding
   each KB's golden set (Console → Quality → *Generate from documents*, then review) is a QA
   data task. On this box the shared KB scores faithfulness 1.00 / relevancy 0.92 /
   precision 1.00 / recall 1.00 on 3 questions — excellent, but not yet a *certified* sample.
2. **Trace backend.** OTel spans are emitted and received but exported only to the collector's
   debug sink — there is no queryable Tempo/Jaeger yet. Metrics + dashboards work fully;
   OBSERVABILITY.md gives the 3-step path to add a trace store.
3. **Ingest retry/DLQ.** Bulk-ingest failures are *persisted and visible* (and now correctly
   surface as a `partial` job status, not `completed`) and retryable via
   `POST …/ingest/jobs/{id}/retry`, but there is no automatic retry/backoff or dead-letter
   queue — a deliberate operator-driven design, documented in LOAD_TEST.md.
4. **CPU-tier latency.** This deployment serves generation through a shared CPU-tier model, so
   realistic p95 is ~20–25 s; the load runbook sets the SLO threshold accordingly. Production
   GPU serving is a config/infra change, not a code change.
5. **Sovereign cutover.** The eval/judge and generation models here are in-boundary but include
   test endpoints; the documented config-only cutover (model URLs + egress allowlist →
   internal-only) closes the runtime-sovereignty claim when the in-ecosystem models land.
6. **Multi-source citation attribution (platform finding, from #2) — FIXED.** Investigating the
   finding corrected its mechanism: faithfulness is scored against the **union** of retrieved
   spans (the lexical/NLI/LLM checkers all read every span, never the typed `[n]`), so a
   mis-typed marker never lowered the score and never caused an abstain — grounded-or-abstain
   was never at risk. The genuine defect was **display-side**: `_cited_only()` kept the span
   whose index the model *typed*, so a mis-typed `[1]` showed a citation pointing at a source
   that didn't support the sentence while the true supporting span was dropped. Fixed with a
   conservative `_realign_citations()` in `grounding.py` that re-points such a marker to the
   span that actually supports the claim — it runs **after** the grounding gate (so it cannot
   turn an abstain into a grounded answer), never fabricates a citation for an unsupported
   claim, and leaves correct citations byte-identical. Verified (mis-cite re-pointed; unsupported
   claim untouched; no regression across grounding/faithfulness/type-gate tests). See
   FEDERATION_QA.md.
