#!/usr/bin/env bash
# Decentralized P2P demo — one owner administers three sovereign departments (dept-a/b/c),
# each its own tenant + libp2p node, and asks ONE question answered from ALL THREE at once.
#
# Story:
#   - Each department ingests a DISTINCT fact about "Project Ganga" (budget / lead / deadline),
#     so no single department can answer the whole question alone.
#   - dept-b and dept-c each issue a signed GRANT authorising dept-a to read them.
#   - dept-a asks across the mesh (POST /v1/ask, federate_from=[dept-b,dept-c]) and gets ONE
#     grounded answer citing all three departments — the cross-node hop is grant-checked
#     fail-closed on each responder.
#   - Revoke dept-c's grant -> the same question can no longer draw on dept-c (zero-trust).
#
# Prereq: ./bootstrap.sh has run (nodes + registry up) AND the api has federation enabled
#         (RAG_FEDERATION_ENABLED=true, RAG_FEDERATION_TRANSPORT=mesh,
#          RAG_FEDERATION_NODE_URL=http://<this dept's node>:7070). See mesh/README.md.
set -euo pipefail
API_CTR="${API_CTR:-ragaas-dev-api-1}"
PY="/app/.venv/bin/python"

echo "==> running decentralized A+B+C federated-ask demo inside $API_CTR"
docker exec -i "$API_CTR" $PY - <<'PY'
import httpx, base64, json
B = "http://localhost:8000"
c = httpx.Client(base_url=B, timeout=40, follow_redirects=True)

def tok(dept):
    return c.post("/v1/dev/token", json={"subject": f"owner@{dept}", "tenant_id": dept,
                  "roles": ["owner", "admin", "reader", "ingestor"]}).json()["access_token"]

def H(dept): return {"Authorization": f"Bearer {tok(dept)}"}

FACTS = {
    "dept-a": ("ganga-budget",  "Project Ganga's approved capital budget is 4.2 crore rupees for FY2027."),
    "dept-b": ("ganga-lead",    "Project Ganga's lead engineer is Meera Nair, reporting to the CTO office."),
    "dept-c": ("ganga-deadline","Project Ganga's committed go-live deadline is 15 March 2027."),
}

print("== 1) provision 3 departments, each a REGISTERED KB + one distinct fact ==")
KB = {}
for dept, (doc_id, text) in FACTS.items():
    kb = c.post("/v1/knowledge-bases", headers=H(dept), json={"display_name": "Main"}).json()["kb_id"]
    KB[dept] = kb
    r = c.post(f"/v1/kb/{kb}/ingest", headers=H(dept), json={
        "doc_id": doc_id, "source_kind": "upload", "mime": "text/plain",
        "content_b64": base64.b64encode(text.encode()).decode()})
    print(f"   {dept} (kb={kb}): ingest {doc_id} -> {r.status_code} ({r.json().get('chunk_count','?')} chunks)")

print("== 2) dept-b and dept-c each GRANT dept-a read access to their KB (signed, cross-dept) ==")
grants = {}
for issuer in ("dept-b", "dept-c"):
    r = c.post("/v1/grants", headers=H(issuer), json={
        "subject_dept_id": "dept-a", "kb_ids": [KB[issuer]], "topics": ["project-ganga"], "ttl_days": 7})
    grants[issuer] = r.json()
    print(f"   {issuer} -> dept-a : {r.status_code} grant_id={r.json().get('grant_id')}")

Q = "For Project Ganga, who is the lead engineer, what is the budget, and what is the go-live deadline?"

print("== 3) dept-a asks ONE question across A + B + C (federated over the mesh) ==")
r = c.post("/v1/ask", headers=H("dept-a"),
           json={"query": Q, "kb_ids": [KB["dept-a"]], "federate_from": ["dept-b", "dept-c"]})
a = r.json()
print(f"   status={a.get('status')}")
print(f"   answer: {a.get('answer','')[:400]}")
depts = sorted({cit.get('doc_id','') for cit in a.get('citations', [])})
print(f"   citations from docs: {depts}  (expect facts from all three departments)")

print("== 4) REVOKE dept-c's grant (zero-trust) and re-ask ==")
c.request("DELETE", f"/v1/grants/{grants['dept-c']['grant_id']}", headers=H("dept-c"))
r = c.post("/v1/ask", headers=H("dept-a"),
           json={"query": Q, "kb_ids": [KB["dept-a"]], "federate_from": ["dept-b", "dept-c"]})
a2 = r.json()
print(f"   status={a2.get('status')}")
print(f"   answer: {a2.get('answer','')[:400]}")
print("   -> dept-c's deadline fact should now be ABSENT (grant revoked, responder fail-closed).")
PY

echo
echo "== 5) PROOF the cross-node hop is real libp2p (not in-process) — node-a dials node-b =="
echo "   node-a control /federate -> peer dept-b over /rag/query/1.0:"
docker exec "${PROJECT:-ragaas-dev}-node-a-1" \
  wget -qO- --post-data '{"peer_dept_id":"dept-b","query_text":"lead engineer","top_k":3}' \
  --header 'Content-Type: application/json' http://127.0.0.1:7070/federate 2>/dev/null \
  | head -c 500 || echo "   (run inside the compose network; see README for the grant-carrying form)"
echo
