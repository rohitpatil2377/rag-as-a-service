#!/usr/bin/env python3
"""Federated RAG mesh registry operator.

Watches every sovereign-node across all namespaces, rebuilds the unified operator
registry (dept -> dev_secret + /dns4 peer_addr), writes it into each department's
registry ConfigMap, and rolls only the nodes whose registry actually changed.

Zero-trust preserved: this only maintains REACHABILITY + verify keys. GRANTS are never
issued here — a department still needs an explicit signed grant to read a peer.

Talks to the k8s API in-cluster via urllib + the mounted ServiceAccount token, so it
needs no external client library and runs on the already-cached rag-api image.
"""

import hashlib
import json
import os
import ssl
import time
import urllib.error
import urllib.request

API = "https://kubernetes.default.svc"
TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
CA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
NODE_LABEL = "app.kubernetes.io/name=sovereign-node"
DEPT_LABEL = "rag.local/department"
DEV_SECRET = os.environ.get("DEV_SECRET", "dev-only-insecure-secret-change-me-32b+")
INTERVAL = int(os.environ.get("INTERVAL_SECONDS", "30"))
LISTEN_PORT = os.environ.get("LISTEN_PORT", "4001")

_ctx = ssl.create_default_context(cafile=CA_PATH)


def _tok():
    with open(TOKEN_PATH) as f:
        return f.read().strip()


def k8s(method, path, body=None):
    url = API + path
    data = json.dumps(body).encode() if body is not None else None
    ct = "application/strategic-merge-patch+json" if method == "PATCH" else "application/json"
    r = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Authorization": "Bearer " + _tok(), "Accept": "application/json", "Content-Type": ct},
    )
    try:
        return json.loads(urllib.request.urlopen(r, context=_ctx).read())
    except urllib.error.HTTPError as e:
        return {"_error": e.code, "_body": e.read().decode()[:300]}


def pod_log(ns, pod):
    r = urllib.request.Request(
        f"{API}/api/v1/namespaces/{ns}/pods/{pod}/log?container=sovereign-node&tailLines=200",
        headers={"Authorization": "Bearer " + _tok()},
    )
    try:
        return urllib.request.urlopen(r, context=_ctx).read().decode()
    except Exception:
        return ""


def peer_id_of(ns, pod):
    for line in pod_log(ns, pod).splitlines():
        if "peer id:" in line:
            return line.split()[-1].strip()
    return None


def seed_empty_registries():
    """Every namespace running a sovereign-node needs its registry ConfigMap to EXIST before
    the node boots — the Go node fatals on a missing --registry file. Create an empty one where
    absent so a freshly-installed department's node can start; reconcile() fills it once the
    node reports its PeerID. This is what makes onboarding automatic (no manual pre-seed)."""
    res = k8s("GET", f"/apis/apps/v1/deployments?labelSelector={NODE_LABEL}")
    for d in res.get("items", []):
        ns = d["metadata"]["namespace"]
        dept = d["metadata"].get("labels", {}).get(DEPT_LABEL) or ns
        cm = f"{dept}-sovereign-node-registry"
        cur = k8s("GET", f"/api/v1/namespaces/{ns}/configmaps/{cm}")
        if "_error" in cur:
            body = {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {"name": cm, "namespace": ns},
                "data": {"registry.json": json.dumps({"departments": []}, indent=2)},
            }
            k8s("POST", f"/api/v1/namespaces/{ns}/configmaps", body)
            print(f"[operator] seeded empty registry for new dept: {dept}", flush=True)


def discover():
    """Return {dept: {ns, pod, peerid}} for every Running sovereign-node."""
    out = {}
    res = k8s("GET", f"/api/v1/pods?labelSelector={NODE_LABEL}")
    for p in res.get("items", []):
        if p.get("status", {}).get("phase") != "Running":
            continue
        if not all(c.get("ready") for c in p.get("status", {}).get("containerStatuses", [])):
            continue
        ns = p["metadata"]["namespace"]
        dept = p["metadata"].get("labels", {}).get(DEPT_LABEL) or ns
        pod = p["metadata"]["name"]
        pid = peer_id_of(ns, pod)
        if pid:
            out[dept] = {"ns": ns, "pod": pod, "peerid": pid}
    return out


def build_registry(members):
    depts = []
    for dept in sorted(members):
        m = members[dept]
        addr = f"/dns4/{dept}-sovereign-node.{m['ns']}.svc.cluster.local/tcp/{LISTEN_PORT}/p2p/{m['peerid']}"
        depts.append({"dept": dept, "dev_secret": DEV_SECRET, "peer_addr": addr})
    return {"departments": depts}


def reconcile(members):
    registry = build_registry(members)
    reg_json = json.dumps(registry, separators=(",", ":"), sort_keys=True)
    reg_hash = hashlib.sha256(reg_json.encode()).hexdigest()[:12]
    changed = []
    for dept, m in members.items():
        ns, cm = m["ns"], f"{dept}-sovereign-node-registry"
        cur = k8s("GET", f"/api/v1/namespaces/{ns}/configmaps/{cm}")
        cur_json = (cur.get("data") or {}).get("registry.json") if "_error" not in cur else None
        pretty = json.dumps(registry, indent=2)
        if cur_json == pretty:
            continue
        body = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": cm, "namespace": ns},
            "data": {"registry.json": pretty},
        }
        if "_error" in cur:
            k8s("POST", f"/api/v1/namespaces/{ns}/configmaps", body)
        else:
            k8s("PUT", f"/api/v1/namespaces/{ns}/configmaps/{cm}", body)
        # roll the node only when its registry hash changes (idempotent)
        patch = {"spec": {"template": {"metadata": {"annotations": {"sovereign.acme/registry-hash": reg_hash}}}}}
        k8s("PATCH", f"/apis/apps/v1/namespaces/{ns}/deployments/{dept}-sovereign-node", patch)
        changed.append(dept)
    return reg_hash, changed


def main():
    print(f"[operator] sovereign mesh registry operator up; interval={INTERVAL}s", flush=True)
    last = None
    while True:
        try:
            seed_empty_registries()
            members = discover()
            depts = sorted(members)
            if depts != last:
                print(f"[operator] members: {depts}", flush=True)
                last = depts
            reg_hash, changed = reconcile(members)
            if changed:
                print(f"[operator] registry {reg_hash} -> updated + rolled: {changed}", flush=True)
        except Exception as e:  # never die; log + retry
            print(f"[operator] error: {type(e).__name__}: {e}", flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
