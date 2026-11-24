# Kubernetes: Ingress + in-chart observability toggles

Status: **PREPARED, not deployed.** Both features are fully wired into the `infra/helm`
(`ragaas`) chart and validated with `helm template` / `helm lint`, but they are **off by
default** and **nothing here has been run against a live cluster.** The live Docker-Compose QA
stack is untouched. Both toggles mirror the `global.linkerd.enabled` pattern: with them off the
rendered manifests are **byte-identical** to a chart without them (verified by diffing
`helm template` output against a pre-change baseline).

These are the last two "turnkey" gaps for the one-release-per-department production chart:

1. an optional **Ingress** so a department's console is reachable from outside the cluster, and
2. an optional **self-contained observability sub-stack** (otel-collector + prometheus + grafana)
   for operators who want metrics/dashboards in-cluster without pulling a community Helm chart.

---

## 1. Ingress — external entry point

### What it is

The **console (nginx) is the single external surface.** It serves the SPA and reverse-proxies
`/v1`, `/health`, `/api` (Swagger) and `/mcp` to the api on the **same origin** (see
`apps/console/nginx.conf`) — exactly as the live deployment works. So external access is an Ingress
routing the department host to the **console Service**; the api is reached **through** the console
proxy and is **not** exposed directly. That is the sovereign default and it stays that way unless
you explicitly opt in to `ingress.exposeApi`.

### Enable it

```bash
helm upgrade --install rag-hr infra/helm -f infra/helm/values-hr.example.yaml \
  --set ingress.enabled=true \
  --set ingress.host=rag.hr.example.internal \
  --set ingress.className=nginx \
  --set ingress.tls.enabled=true \
  --set ingress.tls.secretName=rag-hr-tls
```

| Value | Default | Meaning |
|---|---|---|
| `ingress.enabled` | `false` | Master switch. Off ⇒ no Ingress renders. |
| `ingress.className` | `""` | `ingressClassName` (e.g. `nginx`, `traefik`); `""` = cluster default. |
| `ingress.host` | `rag.example.internal` | External hostname routed to the console. |
| `ingress.path` / `ingress.pathType` | `/` / `Prefix` | The console catch-all route. |
| `ingress.annotations` | `{}` | Controller-specific (e.g. `cert-manager.io/cluster-issuer`). |
| `ingress.tls.enabled` | `false` | Terminate TLS using `secretName`. |
| `ingress.tls.secretName` | `""` | A **pre-created** `kubernetes.io/tls` Secret (e.g. from cert-manager). The chart does **not** mint the cert. |
| `ingress.exposeApi` | `false` | Also route `/v1` + `/api` **straight to the api Service**, bypassing the console proxy — for machine callers. When on, those prefixes take precedence over the console catch-all. |

The Ingress is `networking.k8s.io/v1`, named `rag-console`, backing the `rag-console` Service on
its `console.service.port` (80). With `exposeApi=true` two extra `Prefix` paths (`/v1`, `/api`)
back the `rag-api` Service on `api.service.port` (8000), listed first so they win.

### NetworkPolicy note (ingress controller → console)

The chart's NetworkPolicies (`rag-api-egress`, `<dept>-namespace-egress`) are **Egress-only**
(`policyTypes: [Egress]`). There is **no ingress-type default-deny** on the console or api pods, so
the ingress controller can already reach the console pod — **no NetworkPolicy change is needed** to
make the Ingress work. (If you later add an ingress-type default-deny, you would then need an
allow-from-ingress-controller rule; that is not part of this chart today.)

---

## 2. Observability sub-stack — self-contained, optional

### What it is

An in-chart **otel-collector + prometheus + grafana**, mirroring the Compose topology:

```
api --OTLP:4317--> otel-collector --(prometheus exporter :8889)--> prometheus <--datasource-- grafana
```

It is **sovereign**: nothing is pulled from a community chart repo at deploy time. Every manifest is
in-chart, and the configs are the **exact files** the Compose stack already ships, copied into the
chart under `infra/helm/files/observability/`:

- `infra/otel/collector.yaml` → collector config (OTLP in, prometheus exporter out)
- `infra/prometheus/prometheus.yml` → scrape config
- `infra/grafana/provisioning/datasources/datasource.yml` → Prometheus datasource
- `infra/grafana/provisioning/dashboards/dashboards.yml` → dashboard provider
- `infra/grafana/dashboards/*.json` → the overview, federation, and **RAGAS governance** dashboards

Helm reads them via `.Files.Get` / `.Files.Glob` into ConfigMaps mounted into the pods.

### Pinned image tags

Pinned to the **same versions the Compose stack uses** (`infra/compose/docker-compose.dev.yml`),
fully-qualified `docker.io/` so a pre-warmed / air-gapped node's containerd cache hits without a
registry lookup:

| Component | Image |
|---|---|
| otel-collector | `docker.io/otel/opentelemetry-collector-contrib:0.111.0` |
| prometheus | `docker.io/prom/prometheus:v2.54.1` |
| grafana | `docker.io/grafana/grafana:11.2.2` |

### Enable it

```bash
helm upgrade --install rag-hr infra/helm -f infra/helm/values-hr.example.yaml \
  --set observability.enabled=true \
  --set observability.grafana.adminPassword=<a-real-password>
```

Key knobs (see `values.yaml` `observability:` for the full set):

| Value | Default | Meaning |
|---|---|---|
| `observability.enabled` | `false` | Master switch. Off ⇒ none of it renders. |
| `observability.prometheus.retention` | `15d` | `--storage.tsdb.retention.time`. |
| `observability.prometheus.persistence.enabled` / `.storage` | `true` / `10Gi` | Prometheus TSDB PVC (falls back to `emptyDir` if disabled). |
| `observability.grafana.persistence.enabled` / `.storage` | `true` / `2Gi` | Grafana DB PVC. |
| `observability.grafana.adminPassword` | `"admin"` | **Override this.** Stored in the `rag-grafana-admin` Secret and injected via `secretKeyRef` — never inline in the Deployment env. |
| `*.persistence.storageClass` | `""` | `""` → falls back to `stores.storageClass`, else cluster default. |

### OTLP endpoint wiring

The api already carries `RAG_OTEL_ENABLED=true` and
`RAG_OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317` (`api.config`). When
`observability.enabled=true` the in-chart `otel-collector` Service **is** `otel-collector:4317`, so
that endpoint resolves with **no api.config change needed**.

### NetworkPolicy

When `networkPolicy.enabled` (default) **and** `observability.enabled`:

- **api → otel-collector:4317** — an explicit, gated rule is added to `rag-api-egress` keyed on the
  collector pod's `rag.local/observability: "true"` label. (Off ⇒ that rule is not rendered, keeping
  the render byte-identical.)
- **collector ↔ prometheus, grafana → prometheus, prometheus scrape → collector:8889** — all
  intra-namespace and already admitted by the `<dept>-namespace-egress` policy's intra-namespace
  rule. No new rule needed.

Everything else stays default-deny (egress). The observability pods are selected by the
namespace-egress policy (they are not the sovereign-node) and get intra-namespace + DNS egress.

### Reaching Grafana

Grafana is **not** exposed publicly by default. Reach it with:

```bash
kubectl -n rag-hr port-forward svc/grafana 3000:3000   # then http://localhost:3000  (admin / <your password>)
```

To expose it deliberately, add your own Ingress for `svc/grafana:3000` (the chart's `ingress` block
targets the console, not Grafana — keep Grafana behind auth/VPN).

### Managed-observability alternative

If you already run a central collector (e.g. the AIOps platform / a shared observability namespace),
**leave `observability.enabled=false`** and point the api at it:

```bash
--set api.config.RAG_OTEL_EXPORTER_OTLP_ENDPOINT=http://<external-collector-host>:4317
```

Label that collector's namespace `rag.local/plane=observability` so the existing `rag-api-egress`
rule admits the cross-namespace hop. This is the leaner option when observability is centralized.

### Honest notes / known gaps

- The copied `prometheus.yml` keeps the Compose `ragas-textfile` scrape job (`node-exporter:9100`).
  There is **no in-cluster node-exporter**, so that one target stays **DOWN** in K8s (harmless —
  Prometheus tolerates down targets; the `otel-collector:8889` job is the live metrics path). The
  RAGAS textfile bridge is a Compose-specific mechanism; wiring it in K8s (a sidecar writing the
  `.prom` file) is a follow-up if the RAGAS gate is run in-cluster.
- Prometheus/Grafana run as single replicas with `Recreate` strategy (RWO PVCs). This is a per-
  department operator observability plane, not an HA metrics platform.
- Grafana runs with `fsGroup: 472`, Prometheus with `fsGroup: 65534`, so their PVCs are writable on
  a standard StorageClass.

---

## 3. Validation performed (no cluster touched)

```bash
helm dependency build infra/helm
helm lint infra/helm                       # clean, all combos
helm template t infra/helm > baseline.yaml # BEFORE any edits

# both off  -> byte-identical to baseline
helm template t infra/helm | diff baseline.yaml -            # (no diff)

# ingress on  -> exactly one networking.k8s.io/v1 Ingress -> console; tls + host correct
helm template t infra/helm --set ingress.enabled=true --set ingress.host=rag.hr.example.internal \
  --set ingress.tls.enabled=true --set ingress.tls.secretName=rag-hr-tls

# observability on -> +3 Deployments (otel/prom/grafana), +3 Services, +5 ConfigMaps,
#                     +2 PVCs, +1 Secret, + the api-egress 4317 rule
helm template t infra/helm --set observability.enabled=true
```

Each render parses with `python3 -c "import yaml,sys; list(yaml.safe_load_all(open('r.yaml')))"`.
The `infra/helm/charts/` dependency-build artifact is cleaned up after validation; `Chart.lock` is
left as-is.
