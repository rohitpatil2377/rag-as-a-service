import base64
import json

import httpx

A = "http://api:8000"  # dept-a's own api entry point
D = "http://api-d:8000"  # dept-d's own api entry point
c = httpx.Client(timeout=40, follow_redirects=True)


def tok(base, dept):
    return c.post(
        f"{base}/v1/dev/token",
        json={"subject": f"owner@{dept}", "tenant_id": dept, "roles": ["owner", "admin", "reader", "ingestor"]},
    ).json()["access_token"]


def H(base, dept):  # noqa: N802
    return {"Authorization": f"Bearer {tok(base, dept)}"}


print("== 1) ingest 2 distinct Operations (dept-d) facts, via dept-d's OWN api-d entry point ==")
FACTS_D = [
    ("ops-oncall", "Operations' current on-call rotation lead is Arjun Verma, reachable via the #ops-oncall pager."),
    (
        "ops-patch-window",
        "Operations' standing monthly patch window for production servers is the second Sunday, 02:00-05:00 IST.",
    ),
]
kb_d = c.post(f"{D}/v1/knowledge-bases", headers=H(D, "dept-d"), json={"display_name": "Operations Main"}).json()[
    "kb_id"
]
print(f"   dept-d kb_id={kb_d}")
for doc_id, text in FACTS_D:
    r = c.post(
        f"{D}/v1/kb/{kb_d}/ingest",
        headers=H(D, "dept-d"),
        json={
            "doc_id": doc_id,
            "source_kind": "upload",
            "mime": "text/plain",
            "content_b64": base64.b64encode(text.encode()).decode(),
        },
    )
    print(f"   ingest {doc_id} -> {r.status_code} ({r.json().get('chunk_count', '?')} chunks)")

print("== 2) get dept-a's existing kb_id (from earlier demo.sh runs) or make a fresh one + fact ==")
FACT_A = ("ganga-budget", "Project Ganga's approved capital budget is 4.2 crore rupees for FY2027.")
kb_a = c.post(f"{A}/v1/knowledge-bases", headers=H(A, "dept-a"), json={"display_name": "Main"}).json()["kb_id"]
print(f"   dept-a kb_id={kb_a}")
r = c.post(
    f"{A}/v1/kb/{kb_a}/ingest",
    headers=H(A, "dept-a"),
    json={
        "doc_id": FACT_A[0],
        "source_kind": "upload",
        "mime": "text/plain",
        "content_b64": base64.b64encode(FACT_A[1].encode()).decode(),
    },
)
print(f"   ingest {FACT_A[0]} -> {r.status_code} ({r.json().get('chunk_count', '?')} chunks)")

print("== 3) GRANT dept-d -> dept-a (dept-d authorizes dept-a to read its KB) ==")
r = c.post(
    f"{D}/v1/grants",
    headers=H(D, "dept-d"),
    json={"subject_dept_id": "dept-a", "kb_ids": [kb_d], "topics": ["operations"], "ttl_days": 7},
)
grant_d_to_a = r.json()
print(f"   dept-d -> dept-a : {r.status_code} grant_id={grant_d_to_a.get('grant_id')}")

print("== 4) GRANT dept-a -> dept-d (dept-a authorizes dept-d to read its KB) ==")
r = c.post(
    f"{A}/v1/grants",
    headers=H(A, "dept-a"),
    json={"subject_dept_id": "dept-d", "kb_ids": [kb_a], "topics": ["project-ganga"], "ttl_days": 7},
)
grant_a_to_d = r.json()
print(f"   dept-a -> dept-d : {r.status_code} grant_id={grant_a_to_d.get('grant_id')}")

print()
print("== 5) dept-a (via its own api on :8000/internal api:8000) asks federate_from=[dept-d] ==")
Qa = "Who is the on-call lead for Operations and what is the patch window?"
r = c.post(f"{A}/v1/ask", headers=H(A, "dept-a"), json={"query": Qa, "kb_ids": [kb_a], "federate_from": ["dept-d"]})
aa = r.json()
print(f"   status={aa.get('status')}")
print(f"   answer: {aa.get('answer', '')[:400]}")
cits_a = [(c_.get("doc_id"), c_.get("source_dept") or c_.get("dept")) for c_ in aa.get("citations", [])]
print(f"   citations: {cits_a}")

print()
print("== 6) dept-d (via its OWN api-d on :8003/internal api-d:8000) asks federate_from=[dept-a] ==")
Qd = "What is the approved capital budget for Project Ganga?"
r = c.post(f"{D}/v1/ask", headers=H(D, "dept-d"), json={"query": Qd, "kb_ids": [kb_d], "federate_from": ["dept-a"]})
ad = r.json()
print(f"   status={ad.get('status')}")
print(f"   answer: {ad.get('answer', '')[:400]}")
cits_d = [(c_.get("doc_id"), c_.get("source_dept") or c_.get("dept")) for c_ in ad.get("citations", [])]
print(f"   citations: {cits_d}")

print()
print("== RAW citation objects (full) ==")
print("dept-a ask citations:", json.dumps(aa.get("citations", []), indent=2)[:2000])
print("dept-d ask citations:", json.dumps(ad.get("citations", []), indent=2)[:2000])
