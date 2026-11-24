# RAGAS Governance Gate — QA runbook

**What it is.** The published quality *contract* that governs the RAG-as-a-Service. The
platform already computes the sovereign RAGAS suite (faithfulness, answer relevancy, context
precision, context recall) on its **own in-boundary models** — no external LLM, no data
leaving the sovereign boundary. The gate turns those numbers into a **PASS / FAIL /
INCONCLUSIVE** verdict against codified floors, so a quality claim is *measured and
enforceable*, not asserted.

- Contract logic (pure, unit-tested): `services/control-plane/src/svc_control_plane/ragas_governance.py`
- QA CLI (drives the live stack): `scripts/ragas_gate.py` · `make ragas`
- Metric definitions (sovereign RAGAS): `services/control-plane/src/svc_control_plane/ragas_metrics.py`

## The contract (v1.0.0)

A knowledge base is **certified (PASS)** only when it clears **every** floor over a
**sufficient sample**, with **zero grounded-but-uncited answers**:

| Metric | Floor | What it guards |
|---|---|---|
| faithfulness | ≥ 0.80 | The grounded-or-abstain core: is the answer supported by its cited evidence? |
| answer_relevancy | ≥ 0.70 | Does the answer actually address the question? |
| context_precision | ≥ 0.70 | Are the useful retrieved passages ranked near the top? |
| context_recall | ≥ 0.70 | Did retrieval bring back the evidence the ground truth needs? |
| sample | ≥ 8 questions | Below this a run is **INCONCLUSIVE**, never PASS — no certifying on a handful of questions. |

Verdict precedence (deterministic):
- **FAIL** — any floor breach, or any grounded-but-uncited answer (a structural must-be-zero). A real defect fails at *any* sample size.
- **INCONCLUSIVE** — clean, but fewer than `min_sample` questions scored. Expand the golden set to reach a PASS.
- **PASS** — clean over a sufficient sample.
- **Fleet** verdict = PASS only when *every* KB passes; any FAIL fails the fleet; otherwise INCONCLUSIVE.

Floors are overridable per governance policy via `RAG_RAGAS_*` env vars or CLI flags
(`--faithfulness-min`, `--min-sample`, …) — an operator can *raise* the bar for a regulated
tenant; a lowered bar always shows in the recorded contract.

## Run it (stack up)

```bash
# Certify every KB visible to a tenant:
make ragas ARGS="--tenant public__company-wide"

# Certify a specific set (mints a dev token per org automatically in dev_jwt mode):
make ragas ARGS="--kbs acme__hr,acme__finance,public__company-wide"

# Gate CI/release — treat INCONCLUSIVE as failure and write the Grafana metric:
uv run python scripts/ragas_gate.py --kbs acme__hr --strict \
    --prometheus /var/lib/node_exporter/textfile/ragas.prom
```

Outputs: `ragas_report.md` + `ragas_report.json` (auditable evidence), and — with
`--prometheus` — a textfile of `rag_ragas_*` metrics for the Grafana quality dashboard (see
`docs/qa/OBSERVABILITY.md`). **Exit code:** `0` PASS (INCONCLUSIVE also `0` unless `--strict`),
`1` FAIL, `2` nothing to certify. So a red gate blocks a release; it does not merely print.

> In a non-dev deployment (no `/dev/token`), pass a real `--token <jwt>` authorised for every
> target KB.

## Prerequisite: a golden set per KB

The gate certifies against each KB's saved **golden set** (`GET/PUT /v1/kb/{id}/golden-set`).
A KB with no golden set is skipped ("nothing to certify"). To reach a certified PASS a KB
needs **≥ 8** golden questions. Draft a baseline from the KB's own documents, review, and save:

```bash
# Console: Quality page → "Generate from documents", review, Save.
# Or API:
curl -sX POST ".../v1/kb/<kb>/golden-set/generate?count=10" -H "Authorization: Bearer $T" \
  | curl -sX PUT ".../v1/kb/<kb>/golden-set" -H "Authorization: Bearer $T" -d @-
```

Add an `expected` fact per question where you can — it enables the hit-rate and context-recall
checks. Re-run the gate after any chunking, embedder, or model change.

## Continuous governance

The same suite runs unattended via **continuous evaluation** (FR-G9): enrol a KB
(`PUT /v1/kb/{id}/eval/schedule`) and every run scores the golden set + a sample of live
traffic, records an `EVAL_RUN` audit event, and raises `EVAL_ALERT` on a floor breach —
in-platform only (audit + logs + console), no external egress. The gate CLI is the
on-demand/CI face of the same contract; the scheduler is the always-on face; the Grafana
dashboard makes both continuously visible.

## Worked example (live, this deployment)

Certifying `public__company-wide` (4 docs, a 3-question generated golden set) produced, on the
platform's own models: **faithfulness 1.000, answer_relevancy 0.915, context_precision 1.000,
context_recall 1.000** — every floor cleared. Verdict at the strict contract: **INCONCLUSIVE**
(3 < 8 sample) — the gate correctly declines to certify until the golden set is expanded. This
is the intended behaviour: excellent measured quality, but not yet a *certified* sample.
