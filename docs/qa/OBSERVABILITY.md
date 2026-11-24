# Observability — QA Runbook

How to see the platform's health, and specifically the **RAGAS governance gate** (the
published quality contract the RaaS governs itself by), through Grafana and Prometheus.

Everything here is served by the local dev stack (`make up`,
`infra/compose/docker-compose.dev.yml`). Nothing calls a foreign/managed service — the
metrics path is `prom/*` + `otel/*` images only (sovereign).

---

## 1. What's wired

```
  API (OTel SDK)                          scripts/ragas_gate.py --prometheus <file>
        │ OTLP/gRPC :4317                            │ writes *.prom
        ▼                                            ▼
  otel-collector ──prometheus exporter :8889──►  infra/observability/textfile/ragas.prom
        │ (traces → debug only, see gap §6)          │ (bind-mounted :ro)
        │                                            ▼
        │                                     node-exporter  :9100/metrics
        │                                     (textfile collector)
        └──────────────┐                            │
                       ▼                            ▼
                  Prometheus  :9090  ◄── scrapes both jobs (otel-collector, ragas-textfile)
                       │
                       ▼
                  Grafana    :3000  (provisioned dashboards)
```

| Service | Port (host) | Purpose |
|---|---|---|
| Grafana | `:3000` | Dashboards (provisioned from `infra/grafana`) |
| Prometheus | `:9090` | Scrapes metrics; PromQL |
| otel-collector | `:4317` | Receives OTLP from the API; re-exposes app metrics on `:8889` |
| node-exporter | `:9100` | Re-exposes the RAGAS gate's Prometheus **textfile** (`rag_ragas_*`) |
| API | `:8000` | The app (emits spans + metrics via OTel) |

---

## 2. Open Grafana

- URL: `http://<host>:3000` (locally `http://localhost:3000`).
- Default dev credentials: **`admin` / `admin`** (set by `GF_SECURITY_ADMIN_PASSWORD=admin`
  in `docker-compose.dev.yml` — dev only; prod uses a real secret). Grafana may prompt to
  change the password on first login; skip is fine for dev.
- Datasource **Prometheus** (`uid: prometheus`) is auto-provisioned
  (`infra/grafana/provisioning/datasources/datasource.yml`) → `http://prometheus:9090`.
- Dashboards are auto-loaded from the **RAG as a Service** folder
  (`infra/grafana/provisioning/dashboards/dashboards.yml`).

### Dashboards

| Dashboard (uid) | Shows |
|---|---|
| **RAG as a Service — Platform Overview** (`ragaas-overview`) | Request rate, P95 latency, abstain rate, AI predicted load. |
| **RAG as a Service — Federation** (`ragaas-federation`) | Cross-department federation: query rate, grants, denials, anomaly alerts. |
| **RAGAS Governance — Quality Contract** (`ragas-governance`) | **NEW.** The quality gate — see below. |

### RAGAS Governance dashboard panels

- **Fleet verdict** (stat, `rag_ragas_fleet_status`) — 1 = every KB clears every floor
  over a sufficient sample with zero grounded-but-uncited answers; 0 = a KB failed or the
  fleet is inconclusive. Renders as **PASS** (emerald) / **NOT PASS** (saffron).
- **KBs passing the contract** (stat, `sum(rag_ragas_gate_pass)`).
- **Questions scored (fleet)** (stat, `sum(rag_ragas_sample)`) — governance certifies a KB
  only above the min-sample floor (default 8), so a low number = golden sets too small to
  certify, not a quality problem.
- **Per-KB verdict** (table, `rag_ragas_gate_pass`) — one row per KB; the `status` label
  carries the exact verdict (`pass` / `fail` / `inconclusive`).
- **Per-KB RAGAS metrics** (bar gauge, `rag_ragas_metric`) — each KB's aggregate per metric
  (faithfulness / answer_relevancy / context_precision / context_recall). Thresholds mark the
  contract floors (faithfulness ≥ 0.80, the rest ≥ 0.70).
- **Faithfulness over time** (timeseries, `rag_ragas_metric{metric="faithfulness"}`) — the
  grounded-or-abstain core metric per KB across gate runs, with the 0.80 floor drawn.

> The dashboard is empty until the gate has run at least once (§3). Until then the
> `ragas-textfile` scrape target is **UP** but exposes no `rag_ragas_*` series.

---

## 3. Run the RAGAS gate so its metrics appear

The gate runs the sovereign RAGAS suite across KBs through the **real** pipeline, applies the
published contract (`svc_control_plane.ragas_governance`), writes `ragas_report.json` +
`.md`, and — with `--prometheus` — drops a Prometheus **textfile** that node-exporter serves.

Write the textfile into the dir node-exporter mounts (`infra/observability/textfile/`, ends
in `.prom`):

```bash
# from the repo root, stack up:
make ragas ARGS="--prometheus infra/observability/textfile/ragas.prom --kbs acme__hr,acme__finance,public__company-wide"

# or discover the caller's visible KBs instead of listing them:
make ragas ARGS="--prometheus infra/observability/textfile/ragas.prom --tenant public__company-wide"
```

`make ragas` is `uv run python scripts/ragas_gate.py $(ARGS)` (see the `Makefile`). The gate
exits **non-zero on any FAIL** (and, with `--strict`, on INCONCLUSIVE too), so the same
command gates CI/release.

Within one Prometheus scrape interval (15s) the metrics appear on the dashboard. The metric
families written are exactly:

| Metric | Labels | Meaning |
|---|---|---|
| `rag_ragas_fleet_status` | `contract` | Fleet verdict (1=pass, 0=not pass). |
| `rag_ragas_gate_pass` | `kb`, `status` | Per-KB verdict (1=pass). `status` = pass/fail/inconclusive. |
| `rag_ragas_metric` | `kb`, `metric` | Per-KB RAGAS aggregate in [0,1]. |
| `rag_ragas_sample` | `kb` | Questions scored on the RAGAS suite for a KB. |

> **CI / no-stack use:** the gate needs the running API. To only *refresh the metrics file*
> from an existing report you can regenerate it from `ragas_governance.prometheus_text(report)`
> in a Python shell; but the normal path is the `make ragas` command above.

---

## 4. Confirm metrics in Prometheus

Open `http://localhost:9090` → **Graph** / **Table**, and query:

```promql
rag_ragas_fleet_status          # → 1 (pass) or 0
rag_ragas_gate_pass             # one series per KB, with a status label
rag_ragas_metric{metric="faithfulness"}
sum(rag_ragas_sample)
```

Check the scrape target is healthy at **Status → Targets** (`http://localhost:9090/targets`):
the job **`ragas-textfile`** (`node-exporter:9100`) should be **UP**. You can also confirm the
textfile collector read cleanly:

```promql
node_textfile_scrape_error       # must be 0
```

Direct check of what node-exporter is serving (bypassing Grafana):

```bash
curl -s http://localhost:9100/metrics | grep '^rag_ragas_'
```

---

## 5. Traces (OTel)

- The API exports spans over OTLP/gRPC to `RAG_OTEL_EXPORTER_OTLP_ENDPOINT` =
  `http://otel-collector:4317` (set in `docker-compose.dev.yml`), and `RAG_OTEL_ENABLED=true`.
- **Spans are really being produced.** Verify from the collector logs:

  ```bash
  docker logs ragaas-dev-otel-collector-1 --tail 40 | grep TracesExporter
  # → info TracesExporter {"name":"debug","resource spans":1,"spans":9}
  ```

- **Metrics** from the API's OTel pipeline are exported to Prometheus by the collector's
  `prometheus` exporter on `:8889` and scraped via the `otel-collector` job — this is where
  `http_server_duration_*`, `rag_abstain_total`, etc. (the Overview dashboard) come from.

---

## 6. Honest remaining gap — no trace backend

The OTel **traces** pipeline currently exports to the collector's **`debug`** exporter only
(`infra/otel/collector.yaml`):

```yaml
service:
  pipelines:
    traces:
      exporters: [debug]        # TODO(prod): tempo/AIOps bridge
```

So spans are received and logged by the collector, then dropped — **there is no queryable
trace backend (Tempo/Jaeger) in the dev stack**, and no traces datasource in Grafana. You can
confirm a span was ingested (collector log line above), but you **cannot open a trace by ID or
search spans** today.

This is deliberate for the dev tier (a trace backend adds a container + memory on a RAM-tight
VM, and the metrics path — which backs every dashboard including the new RAGAS gate — is fully
working without it). To close the gap when wanted, and it is low-risk/additive:

1. Add a `tempo` service (`grafana/tempo`) to `docker-compose.dev.yml` listening on OTLP `:4317`
   (or a distinct port).
2. Add a `tempo` exporter to the collector's **traces** pipeline in `collector.yaml`
   (`exporters: [debug, otlp/tempo]`) and reload the collector.
3. Add a **Tempo** datasource under `infra/grafana/provisioning/datasources/` pointing at it.

Everything else in this runbook (metrics, all three dashboards, the RAGAS gate) works with no
trace backend.

---

## 7. Files involved

- `infra/compose/docker-compose.dev.yml` — `node-exporter` service (textfile bridge) + the
  existing `otel-collector` / `prometheus` / `grafana`.
- `infra/prometheus/prometheus.yml` — scrape job `ragas-textfile`.
- `infra/observability/textfile/` — the bind-mounted drop dir; the gate writes `ragas.prom` here.
- `infra/grafana/dashboards/ragas-governance.json` — the RAGAS dashboard.
- `infra/otel/collector.yaml` — OTel collector pipelines (traces → debug; metrics → prometheus).
- `scripts/ragas_gate.py` + `services/control-plane/src/svc_control_plane/ragas_governance.py`
  — the gate CLI and the pure contract (incl. `prometheus_text`).
