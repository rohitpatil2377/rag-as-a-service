# infra/helm — Kubernetes deployment skeleton (NOT deployed in Phase-0)

Placeholder Helm chart for the sovereign K8s target (TDD §11): managed K8s across 3
zones, per-domain namespaces, GPU node pools (MIG-sliced) for TEI/vLLM, stateful stores
on dedicated pools, GitOps + golden-set eval gate before promotion.

**Phase-0 status:** skeleton only — `make up` uses `infra/compose` for local dev. The
chart is fleshed out in Phase-1 when services split out of the modular monolith
(ADR-0001). Values here mirror the compose topology so the migration is mechanical.

```
helm/
  Chart.yaml            # chart metadata (skeleton)
  values.yaml           # image tags, replicas, per-tenant namespace strategy, GPU pools
  templates/            # Deployment/Service/HPA/NetworkPolicy per layer (TODO Phase-1)
```

Key production concerns to encode here (from TDD §9/§11):
- NetworkPolicies enforcing the sovereignty egress block (no international endpoints).
- mTLS (Istio/SPIFFE), per-tenant resource quotas + MIG slicing (noisy-neighbour).
- Immutable/WORM audit volume; HSM/KMS integration for log signing.
