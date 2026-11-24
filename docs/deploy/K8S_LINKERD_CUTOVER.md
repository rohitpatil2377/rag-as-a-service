# Kubernetes + linkerd mTLS cutover runbook

Status: **PREPARED, not deployed.** The linkerd service-mesh mTLS layer is fully wired into the
Helm chart and validated by `helm template`/`helm lint`, but it is **off by default**
(`global.linkerd.enabled: false`) and **nothing in this document has been run against a live
cluster.** The current QA system runs on Docker Compose and is untouched by any of this. This
runbook is the one-command deploy path for the future Kubernetes production cutover.

> The two other optional, default-off chart toggles — an external **Ingress** for the console and a
> self-contained in-chart **observability** sub-stack (otel-collector + prometheus + grafana) — are
> documented separately in [`K8S_INGRESS_OBSERVABILITY.md`](./K8S_INGRESS_OBSERVABILITY.md).

---

## 1. What this layer is (and is not) — read first

The node-to-node federation mesh **already has transport mTLS today**, on Compose, via
**libp2p-TLS 1.3** (Noise/TLS handshake with pinned Ed25519 peer identities). That hop is already
mutually authenticated and is not waiting on anything in this document.

linkerd adds a **defence-in-depth** layer for the OTHER hops that libp2p does not cover — the
service-to-service calls inside a Kubernetes namespace (`api↔console`, `api↔node` control API) and
a uniform **SPIFFE identity** mesh across the whole release. It is additive. If you never install
linkerd, the platform is still mutually authenticated on the federation hop that matters most.

> Honest summary: this is a belt-and-suspenders layer. The federation mesh is already
> mutually-authenticated via libp2p-TLS. linkerd's value is uniform SPIFFE identity + mTLS on the
> intra-cluster service hops and a single policy/telemetry plane, not "turning on encryption that
> was previously absent."

### What mTLS you get where

| Hop | Transport security | Layer | Status |
|---|---|---|---|
| node ↔ node (cross-department federation, libp2p) | **libp2p-TLS 1.3 + pinned Ed25519 peer IDs** | libp2p (in-app) | **Live today** (Compose + K8s) |
| api ↔ node control API (`:7070`, drives `/federate`) | linkerd sidecar mTLS (opaque TCP) | linkerd (K8s only) | Prepared — needs `global.linkerd.enabled=true` |
| api ↔ console (nginx `/v1` reverse-proxy) | linkerd sidecar mTLS (HTTP) | linkerd (K8s only) | Prepared |
| console ↔ browser | your ingress TLS (out of scope here) | ingress | Operator-provided |
| api ↔ stores (postgres/qdrant/opensearch/minio) | **not meshed** — intra-namespace, NetworkPolicy-fenced | — | Deliberately unmeshed (see §6) |
| grant signing / verification | Ed25519 signatures on canonical claims | application | Live today |

---

## 2. Prerequisites / readiness checklist

Tick every box before flipping the switch.

- [ ] **A Kubernetes cluster** (the sovereign on-prem cluster), `kubectl` context pointed at it,
      cluster-admin for the control-plane install.
- [ ] **A CNI that enforces NetworkPolicy** (Calico/Cilium/etc.). The chart's default-deny egress
      and the linkerd control-plane egress rules are inert without an enforcing CNI.
- [ ] **linkerd CLI** installed locally and a **compatible control plane** installed in the cluster
      (see §3). `linkerd check` must be green.
- [ ] **A private image registry** reachable from the cluster, with the three images pushed:
      `api`, `console`, `sovereign-node` (see §4). Air-gap: images pre-warmed into the nodes'
      containerd (the store images are fully-qualified `docker.io/...` refs for exactly this reason).
- [ ] **StorageClass(es)** for the stores' PVCs (postgres/qdrant/opensearch/minio) and the node
      identity PVC. Set `stores.storageClass` / `sovereign-node.persistence.storageClass` or rely on
      the cluster default.
- [ ] **The K8s Secret populated** for each department release (see §5 for the exact keys). Never
      commit real secret values into the chart.
- [ ] **DNS / ingress** for the console (and API if exposed) — an Ingress or Gateway with TLS.
- [ ] **The `linkerd` namespace exists and is labelled** `kubernetes.io/metadata.name: linkerd`
      (kubelet applies this automatically on modern clusters; the chart's egress rules match on it).
- [ ] **Namespace-per-release / per-department model** understood: one `helm install <dept>` per
      department namespace, each with its own isolated stores + federation node.

---

## 3. Install the linkerd control plane (once per cluster)

```bash
# CLI
curl -sL https://run.linkerd.io/install | sh
export PATH="$HOME/.linkerd2/bin:$PATH"

# Pre-flight the cluster
linkerd check --pre

# CRDs, then the control plane
linkerd install --crds | kubectl apply -f -
linkerd install        | kubectl apply -f -

# Wait until fully healthy
linkerd check

# (recommended) the viz extension — gives `linkerd viz edges` / `tap` for the mTLS proof in §8
linkerd viz install | kubectl apply -f -
linkerd viz check
```

The control plane runs in the `linkerd` namespace and exposes: identity `:8080`, destination
`:8086`, policy `:8090` (the chart opens egress to exactly these), and scrapes each proxy's
admin/tap on `:4190`/`:4191`. Meshed pods carry an inbound proxy on `:4143`.

---

## 4. Build + push the images (per release / air-gap)

```bash
REG=registry.internal.acme/ragaas       # your private registry
TAG=1.0.0

docker build -t $REG/api:$TAG            -f apps/api/Dockerfile .
docker build -t $REG/console:$TAG        -f apps/console/Dockerfile apps/console
docker build -t $REG/sovereign-node:$TAG -f <sovereign-node Dockerfile> .

docker push $REG/api:$TAG
docker push $REG/console:$TAG
docker push $REG/sovereign-node:$TAG
```

The chart resolves images from `global.imageRegistry` + `global.imageTag` unless `api.image` /
`console.image` / `sovereign-node.image` are set explicitly (the per-dept example files pin
`registry.local/...:local`). For a true air-gap install, pre-warm all images (including the
`docker.io/...` store images) into each node's containerd and keep
`global.imagePullPolicy: IfNotPresent`.

---

## 5. The K8s Secret — exact keys

Secret values live ONLY in the K8s Secret (or a sealed-secret / external secret manager), never in
the committed chart. The chart renders `Secret/rag-api-secrets` from `api.secrets`; supply real
values via `--set`, `-f a-secret-values.yaml` (git-ignored), a sealed-secret, or Vault injection.

Required `api.secrets` keys:

| Key | Purpose |
|---|---|
| `RAG_POSTGRES_DSN` | Postgres DSN incl. password (must match `stores.postgres.password`). |
| `RAG_MINIO_ACCESS_KEY` | MinIO access key (must match `stores.minio.rootUser`). |
| `RAG_MINIO_SECRET_KEY` | MinIO secret key (must match `stores.minio.rootPassword`). |
| `RAG_LLM_API_KEY` | Generation model key (empty for keyless in-cluster / air-gap). |
| `RAG_EMBED_API_KEY` | Embedder key (e.g. the AIOps platform bge-m3). |
| `RAG_RERANK_API_KEY` | Primary reranker key (e.g. Jina, `Authorization: Bearer`). |
| `RAG_RERANK_FALLBACK_API_KEY` | Fallback reranker key (e.g. AIOps bge-reranker). |
| `RAG_DATALAB_API_KEY` | Chandra OCR/vision key. |
| `RAG_GRANT_SIGNING_KEY` | Base64 Ed25519 federation grant-signing key. |
| `RAG_GRANT_SIGNING_KEY_ID` | Grant-signing key id (not secret, kept with the key for rotation). |

Also set (as store values / their own secrets): `stores.postgres.password`,
`stores.minio.rootUser`, `stores.minio.rootPassword`, and — for the federation node — the
`sovereign-node.ragToken` secret (bearer the node uses to call its own api) and the
`sovereign-node.identity` key material (stable libp2p PeerID). None of these belong in git.

`RAG_PLATFORM_ADMIN_EMAILS` (super-admin allowlist) is **not** a secret — it is a plain
ConfigMap value; set it per deployment, e.g. `--set 'api.config.RAG_PLATFORM_ADMIN_EMAILS=superadmin@acme.example'`.

---

## 6. The stores meshing decision

**The stores (postgres, qdrant, opensearch, minio) are deliberately left UNMESHED.** This is a
common, valid choice and is documented in `infra/helm/templates/stores.yaml`.

Why:
- They are StatefulSets speaking **non-HTTP** wire protocols. Postgres (`:5432`) is
  *server-speaks-first*, which hangs under linkerd's HTTP protocol detection unless handled with
  care; a sidecar on a stateful pod also adds proxy-lifecycle/shutdown-ordering risk.
- The `api↔store` hop **never leaves the department namespace** and is already fenced by the
  `rag.local/store="true"` selector plus the `rag-api-egress` and `<dept>-namespace-egress`
  NetworkPolicies. It is not an untrusted hop that mTLS would meaningfully harden.
- linkerd's defence-in-depth is applied where it earns its keep: the service-to-service hops.

The meshed api still reaches the unmeshed stores through its own outbound proxy. To avoid a
protocol-detection stall on the postgres wire, `:5432` is marked opaque on the api pod
(`config.linkerd.io/opaque-ports: "7070,5432"`), and the NetworkPolicies keep the api↔store path
open. **To mesh a store later:** add `linkerd.io/inject: enabled` to that StatefulSet's pod
template and list its non-HTTP port in `config.linkerd.io/opaque-ports`, exactly mirroring the
node's pattern in `sovereign-node/templates/deployment.yaml`.

---

## 7. Step-by-step cutover (per department)

```bash
DEPT=hr
NS=rag-$DEPT

# 0) build the subchart dependency once (packages sovereign-node into charts/)
helm dependency build infra/helm

# 1) namespace
kubectl create namespace $NS
kubectl label namespace $NS rag.local/department=$DEPT

# 2) (optional but recommended) install WITHOUT the mesh first, confirm the app is healthy,
#    then flip the switch — so a mesh issue is never conflated with an app issue.
helm install $DEPT infra/helm -n $NS \
  -f infra/helm/values-$DEPT.example.yaml \
  -f my-$DEPT-secrets.yaml            # git-ignored: the K8s Secret values from §5

# 3) THE MESH CUTOVER — one switch meshes the whole release (api + console + node):
helm upgrade $DEPT infra/helm -n $NS \
  -f infra/helm/values-$DEPT.example.yaml \
  -f my-$DEPT-secrets.yaml \
  --set global.linkerd.enabled=true

#    (or install meshed in one shot by adding --set global.linkerd.enabled=true to step 2)

# 4) verify the proxies came up healthy for this namespace
linkerd check --proxy -n $NS
```

`global.linkerd.enabled` lives under Helm `global`, so the single flag propagates from the umbrella
chart into the `sovereign-node` subchart automatically — **api, console, and node are all meshed by
that one value.** With it off, the render is byte-for-byte the non-mesh deploy (no sidecars, no
linkerd NetworkPolicy rules), so this change cannot affect a cluster that never sets it.

---

## 8. Verify mTLS is actually happening

```bash
NS=rag-hr

# every meshed pod should show 2/2 containers (app + linkerd-proxy)
kubectl get pods -n $NS

# proxies healthy
linkerd check --proxy -n $NS

# EDGES: shows the src->dst pairs and whether each is mutually authenticated (SECURED = mTLS)
linkerd viz edges deployment -n $NS
# expect rag-console -> rag-api and rag-api -> <dept>-sovereign-node marked SECURED / tls=true

# TAP: watch live requests; each line reports tls=true when both peers authenticated
linkerd viz tap deploy/rag-api -n $NS
```

Acceptance: `linkerd viz edges` reports the `api↔console` and `api↔node` edges as
mutually-authenticated (`SECURED`), and `linkerd viz tap` shows `tls=true` on live traffic between
them. The node↔node federation hop remains secured independently by libp2p-TLS (it does not show as
a linkerd edge because it is the node's own libp2p transport, not a linkerd-proxied HTTP/TCP hop).

---

## 9. Rollback

```bash
helm upgrade $DEPT infra/helm -n $NS \
  -f infra/helm/values-$DEPT.example.yaml -f my-$DEPT-secrets.yaml \
  --set global.linkerd.enabled=false
```

Sidecars are removed and the linkerd NetworkPolicy rules disappear on the next rollout. The
application keeps running exactly as it did pre-mesh; the federation hop stays mTLS via libp2p-TLS
throughout. The linkerd control plane can be left installed (idle) or removed with
`linkerd viz uninstall | kubectl delete -f -` then `linkerd uninstall | kubectl delete -f -`.

---

## 10. What the chart wires (reference)

- `infra/helm/values.yaml` — `global.linkerd.enabled` (the one switch) + the reconciled non-secret
  config (`RAG_PLATFORM_ADMIN_EMAILS`, model thresholds, the live feature flags).
- `infra/helm/templates/api.yaml` — inject annotation + `opaque-ports: "7070,5432"`; linkerd
  control-plane egress in `rag-api-egress`.
- `infra/helm/templates/console.yaml` — inject annotation (HTTP only, no opaque-ports).
- `infra/helm/templates/networkpolicy-namespace-egress.yaml` — linkerd control-plane egress for
  every meshed namespace pod.
- `infra/helm/templates/_helpers.tpl` — `rag.linkerdEnabled` helper (nil-safe global read).
- `infra/helm/sovereign-node/templates/{deployment,networkpolicy}.yaml` — the reference node
  wiring (inject + `opaque-ports` for libp2p+7070 + proxy/control-plane netpol rules), now gated on
  `or .Values.linkerd.enabled global.linkerd.enabled` so the umbrella switch reaches it while
  standalone installs still honour `sovereign-node.linkerd.enabled`.
- `infra/helm/templates/stores.yaml` — the documented unmeshed-stores decision.
