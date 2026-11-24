# Federated RAG mesh registry operator

Auto-onboards new departments into the libp2p federation mesh. Watches every `sovereign-node`
across all namespaces, seeds an empty registry ConfigMap so a fresh node can boot, reads each
node's PeerID once Running, rebuilds the unified operator registry (dept -> dev_secret + /dns4
peer_addr), writes it into every department's `<dept>-sovereign-node-registry` ConfigMap, and
rolls only the nodes whose registry actually changed.

**Zero-trust preserved:** this maintains REACHABILITY + verify keys only. GRANTS are never
issued here — a department still needs an explicit publisher-signed grant to read a peer.

## Deploy
```
kubectl apply -f operator-manifest.yaml
kubectl create configmap registry-operator-script -n sovereign-system \
  --from-file=registry_operator.py=registry_operator.py --dry-run=client -o yaml | kubectl apply -f -
```

## Onboard a new department (now fully automatic for mesh reach)
```
cp infra/helm/values-hr.yaml infra/helm/values-<dept>.yaml   # change dept name + creds
helm install <dept> infra/helm -n <dept> --create-namespace -f infra/helm/values-<dept>.yaml \
  --set sovereign-node.registry.enabled=true ...
# the operator seeds + wires the registry across all depts automatically; then issue grants.
```

Runs on the already-cached rag-api image (python3 + urllib, no external client lib), so it
works air-gapped. Requires the sovereign-node registry ConfigMap mount to be `optional: true`
(so a new node boots before its registry exists).
