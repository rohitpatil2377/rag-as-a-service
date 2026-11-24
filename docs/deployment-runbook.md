# RAG as a Service — Deployment Runbook

How to run RAG as a Service, from the local dev stack to a sovereign, per-department Kubernetes
deployment. The goal of this doc: when a cluster (and GPUs) land, the cut-over is **run these
commands**, not "figure out how to deploy."

> **Status honesty (2026-08).** The **local dev stack** and the **per-department federation
> node** (`infra/helm/sovereign-node`) are complete and exercised. The **control-plane API**
> ships here as validated plain manifests (`infra/k8s/ragaas-api.yaml`); the umbrella Helm
> chart (`infra/helm`) is still a Phase-1 skeleton (its `templates/` are TODO). Nothing below has
> been applied to a live cluster yet — `helm lint` / `kubectl --dry-run=server` on your target
> before prod.

---

## 0. What runs where

| Component | Local (today) | Kubernetes (target) |
|---|---|---|
| Control-plane API (modular monolith) | `infra/compose/docker-compose.dev.yml` | `infra/k8s/ragaas-api.yaml` (Deployment/Service/HPA/PDB/NetworkPolicy) |
| Stores: Qdrant, OpenSearch, MinIO, Postgres | compose services | external/managed, or StatefulSets on dedicated pools |
| Models: TEI, reranker, NLI (CPU) / vLLM, Chandra (GPU) | compose CPU services | GPU node pool (MIG-sliced) |
| Federation node (per dept) | commented in dev compose | `helm install <dept> infra/helm/sovereign-node` |
| Coordinator (leader election) | in-process (single node) | Redis (`RAG_REDIS_URL`) → exactly-one-replica scheduler + bulk-ingest |

---

## 1. Prerequisites

- **Local:** Docker + `uv`. `make up` brings the stack up; `make down` tears it down.
- **Kubernetes:** a managed cluster with a CNI that enforces `NetworkPolicy` (the sovereignty
  control is a NetworkPolicy — a cluster that ignores them does NOT enforce sovereignty),
  a sovereign image registry (`registry.internal.acme/...`), and a secrets manager
  (sealed-secrets / External Secrets / Vault). GPU node pool optional (see §7).

---

## 2. Local dev

```bash
make up                 # stores + CPU models (infra/compose/docker-compose.dev.yml)
make migrate            # alembic upgrade head  (global rag_identity tables + per-tenant on demand)
make demo               # walking skeleton: ingest → grounded, cited answer
curl localhost:8000/health   # {"status":"ok"}
```

Real-store CI (isolated, never touches a shared stack) — see `infra/compose/docker-compose.ci.yml`
+ `infra/compose/ci.env`:

```bash
docker compose -f infra/compose/docker-compose.ci.yml up -d
source infra/compose/ci.env && uv run alembic -c libs/clients/alembic.ini upgrade head
uv run pytest tests/integration tests/e2e -p no:cacheprovider
docker compose -f infra/compose/docker-compose.ci.yml down -v
```

---

## 3. Configuration & secrets (12-factor)

Everything is env-driven (`rag_common.config.Settings`), so config maps 1:1 to a ConfigMap and
secrets to a Secret — no code change between compose and K8s. See `.env.example` for the full
list. The manifest split:

- **ConfigMap** (`rag-api-config`): store hosts, OTel endpoint, `RAG_REDIS_URL`, sovereignty +
  federation flags.
- **Secret** (`rag-api-secrets`): `RAG_POSTGRES_DSN`, MinIO keys, and the **federation grant
  signing key** (`RAG_GRANT_SIGNING_KEY` base64 Ed25519 + `RAG_GRANT_SIGNING_KEY_ID`). Inject
  via sealed-secret / External Secret — never commit real values.

**Grant key rotation** (no downtime, no re-signing): generate a new keypair; move the CURRENT
public key into `RAG_GRANT_RETIRED_VERIFY_KEYS` (JSON list of base64 public keys); set the new
private key as `RAG_GRANT_SIGNING_KEY` with a fresh `..._KEY_ID`. Existing grants keep verifying
against the retired key until they expire; new grants use the new key. Drop the retired key once
no unexpired grant was signed with it (the active `kid` is recorded on every grant-issue audit
event, so you can tell).

---

## 4. Health probes → K8s

The API exposes three endpoints; wire the two K8s cares about:

- **`GET /livez`** → `livenessProbe`. Process-only, dependency-free. A failing liveness RESTARTS
  the pod, so it must never check a store (a store blip would restart-loop every replica).
- **`GET /readyz`** → `readinessProbe`. Checks store reachability (Qdrant/OpenSearch/TEI/Postgres),
  returns `503` + a per-store breakdown when one is down. A failing readiness sheds traffic
  (pod pulled from the Service) WITHOUT a restart, so a store blip self-heals.
- **`GET /health`** — legacy liveness alias, kept for back-compat.

Both probes are already wired in `infra/k8s/ragaas-api.yaml`.

---

## 5. Deploy the control-plane API

One namespace per department (`tenancy.isolation: namespace`). Stores reachable in-namespace
(managed endpoints or StatefulSets), Redis present for the coordinator.

```bash
kubectl create namespace rag-hr
# label store pods so the egress NetworkPolicy can select them:
#   kubectl -n rag-hr label pod -l app=qdrant rag.local/store=true   (…and opensearch/minio/postgres/tei/reranker/nli/redis)
kubectl -n rag-hr apply --dry-run=server -f infra/k8s/ragaas-api.yaml   # validate first
kubectl -n rag-hr apply -f infra/k8s/ragaas-api.yaml
kubectl -n rag-hr rollout status deploy/rag-api
```

Run migrations once per environment (a Job, or `kubectl exec` into a pod):

```bash
kubectl -n rag-hr exec deploy/rag-api -- /app/.venv/bin/alembic -c libs/clients/alembic.ini upgrade head
```

What the manifest gives you: 2 replicas + HPA (CPU 70%, 2→8), PodDisruptionBudget, non-root
read-only container, and the **sovereignty egress NetworkPolicy** (API may reach ONLY in-cluster
stores + DNS + OTel — never a foreign endpoint).

**Multi-replica correctness:** the scheduler (continuous eval / sync) and the bulk-ingest worker
are both **leader-gated** via the Redis coordinator — exactly one replica runs each, so scaling
never double-ingests or double-evaluates. This only holds with `RAG_REDIS_URL` set; without it
every replica believes it is the single leader.

---

## 6. Deploy the federation nodes (per department)

The `sovereign-node` Helm chart is the decentralised mesh carrier — one release per department,
each locked by a NetworkPolicy to (a) its own dept API and (b) explicitly configured peers.

```bash
# values-hr.yaml: department: hr; ragUrl: http://rag-api:8000; bootstrapPeers: [ …finance node… ]
helm install hr infra/helm/sovereign-node -n rag-hr -f values-hr.yaml
```

**Node identity / PKI:** each node keeps a stable Ed25519 libp2p identity injected as a Secret
(`identity.secretName`), so its peer id survives restarts and can be pinned in other depts'
`bootstrapPeers` and in grant records. Seal the key; do not commit it. (SPIFFE/SPIRE-issued
workload identity is the hardening path once the cluster is live — the chart's Secret-injected
identity is the interim.)

**Sovereignty boundary:** keep `networkPolicy.enabled: true` in every real deployment — it is the
enforced per-department isolation. Disable only for a local kind/minikube smoke test with no CNI
policy support.

---

## 7. GPU + full sovereignty (drop hosted egress)

Until models are self-hosted, hosted APIs (`api.sarvam.ai`, `datalab.to`) are a **sovereignty
caveat**, not a sovereign deployment. To close it:

1. Bring up a GPU node pool (MIG-sliced) and deploy vLLM (LLM) + the Chandra vision model on it.
2. Flip config to point at the in-cluster GPU endpoints:
   ```
   RAG_LLM_BACKEND=vllm            RAG_VLLM_URL=http://vllm:8000/v1
   RAG_EMBED_BACKEND=tei           RAG_TEI_URL=http://tei:80
   RAG_VISION_LLM_BACKEND=chandra  RAG_OCR_BACKEND=chandra
   ```
3. Tighten the egress allowlist: remove any hosted host from `RAG_ALLOWED_EGRESS_HOSTS`, and the
   API egress NetworkPolicy already denies everything except in-cluster stores + `vllm` — so once
   the models are in-cluster there is **zero foreign runtime egress**. Prove it: `GET
   /kb/{kb}/attestation` reports `sovereignty_enforced: true` with an empty foreign allowlist.

> Blocked on GPU supply until step 1 — everything downstream is a config flip, no code change.

---

## 8. Verify a deployment

```bash
kubectl -n rag-hr get pods                              # api Ready (readiness green = stores reachable)
curl .../v1/kb/hr__policies/attestation                  # residency + sovereignty + federation posture
# federated smoke: issue a grant, ask across depts, confirm citations name the source dept
```

Dashboards: import `infra/grafana/dashboards/ragaas-federation.json` (federation SLOs +
denied-query alerts) and `ragaas-overview.json`.

---

## 9. Backup / DR & key custody

- **Stores:** per-department Qdrant snapshot + Postgres `pg_dump` (incl. the global
  `rag_identity` grant/eval registries) + MinIO replication (Object-Lock/WORM for the audit
  tier). Restore is per-namespace, so a single department can be recovered independently.
- **Grant table DR:** the `rag_identity.federation_grants` table is the federation source of
  truth — include it in the Postgres backup set; a lost grant table silently closes all
  cross-dept access (fail-closed, not data loss).
- **Key custody:** grant signing key + node identity keys live in the secrets manager; the
  tenant-DEK envelope-encryption seam wires to an external KMS/HSM (Vault Transit) — see the
  KMS config. Rotate per §3.

---

## 10. Air-gapped installs

For a fully disconnected environment (no huggingface.co / registry pulls at runtime), follow
`docs/airgap-install.md` — pre-pulled images, a model-cache volume, and `HF_HUB_OFFLINE`.

---

## Appendix — the "one command" state

- **Federation node:** `helm install <dept> infra/helm/sovereign-node -f values-<dept>.yaml` ✅ ready.
- **Control-plane API:** `kubectl apply -f infra/k8s/ragaas-api.yaml` ✅ manifests ready
  (lint on your cluster first).
- **Umbrella Helm chart** (`infra/helm`): Phase-1 TODO — fold the api manifests above into
  `templates/` when the monolith splits into per-layer services (ADR-0001).
