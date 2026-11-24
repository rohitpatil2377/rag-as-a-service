# infra/helm/sovereign-node — per-department Federated RAG federation node

A standalone Helm chart that deploys **one libp2p federation node per department**. It is
separate from the main `infra/helm` chart on purpose: each department installs its own node,
in its own namespace, with strict network isolation, so cross-department federation is
mediated by **cryptographic grants + NetworkPolicy**, never by ambient trust.

## What it deploys
- **Deployment** — the Go libp2p node (from `sovereign-node/`), one stable replica. It fronts
  the department's local `rag-api` and answers a peer's federated query ONLY when that peer
  holds an active grant (grant checks happen in the app; this chart handles placement + wiring).
- **Service** — ClusterIP exposing the libp2p port (`4001`) to peers on the internal mesh only.
- **NetworkPolicy** — the sovereignty control (see below). Strict ingress + egress.
- **Secret** — stub for the node's stable mTLS/libp2p identity key.

## Per-department deployment
One Helm release per department. The release name is the department id:

```bash
# HR department node
helm install hr ./sovereign-node -n hr --create-namespace -f values-hr.yaml

# Finance department node
helm install finance ./sovereign-node -n finance --create-namespace -f values-finance.yaml
```

A minimal `values-hr.yaml`:

```yaml
department: hr
ragUrl: http://rag-api:8000          # HR's own rag-api (in-namespace)
bootstrapPeers:
  - "/dns4/finance-sovereign-node.finance.svc.cluster.local/tcp/4001/p2p/<finance-peerid>"
```

The libp2p **identity key** should be provisioned as a Secret out-of-band (Sealed Secrets /
External Secrets / Vault) and referenced via `identity.secretName`, or injected at install
time with `--set-file identity.keyPem=./hr-node.key`. Never commit a raw key.

## NetworkPolicy isolation model (the whole point)
`networkPolicy.enabled: true` (default) makes the node **default-deny in both directions** and
then admits only:

- **Ingress** — connections TO the node's libp2p port are allowed ONLY from
  (1) the **federation control plane** (matched by namespace label + pod label — grant
  issuance / peer discovery) and (2) **peer department nodes** (matched by the shared
  `app.kubernetes.io/name: sovereign-node` label).
- **Egress** — the node may connect ONLY to (1) **its own department's `rag-api`**
  (same namespace, port 8000), (2) the configured **bootstrap/peer nodes** (port 4001), and
  (3) **DNS** (kube-system). It CANNOT reach any other department's `rag-api`, arbitrary
  pods, or the internet.

Grants decide *what* a peer may read; the NetworkPolicy decides *who* may open a socket at
all. Both layers are required: a peer with no grant that somehow reached the node still reads
nothing, and a pod with no network path never reaches the node in the first place.

> Requires a CNI that enforces NetworkPolicy (Calico/Cilium/etc.). Disable only for local
> kind/minikube smoke tests without a policy-capable CNI.

## Status
Phase-1 chart. The node container image (`registry.internal.acme/ragaas/sovereign-node`)
is built from `sovereign-node/` — its Dockerfile (multi-stage golang build) is tracked
separately and is a prerequisite for a real install.
