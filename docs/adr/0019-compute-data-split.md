# ADR-0019 — Compute/data split: centralized compute, decentralized data; federate the query, not the vector

**Status:** accepted · 2026-08-04
**Context:** How should a multi-department sovereign RAG be carved between "shared" and
"per-department"? Two extremes exist, and the original proposal (`federated_rag_proposal.md`)
mixes them in a way that is internally inconsistent:

- It calls for **per-department isolated compute** — "each department runs its own isolated
  embedding engine" — *and*
- It calls for federation to **"broadcast the high-dimensional query vector to permitted
  peers."**

Those two cannot both be true. If each department embeds with its own (possibly different)
model, the resulting vectors live in **different spaces** — a vector produced by department A's
embedder is meaningless to department B's index. A raw vector is not portable across
heterogeneous embedders. So "isolated embedders" and "broadcast the vector" contradict.

Meanwhile the deployed reality is the opposite of "compute per department": GPUs are
expensive and operationally heavy, so the platform already shares one **self-hosted model
fleet** (AIOps: Qwen-family LLM, bge-m3 embedder, bge-reranker; TEI/vLLM/llama.cpp seams),
while **data** is already isolated per tenant (Qdrant collection, OpenSearch index, Postgres
schema, MinIO prefix — all per tenant, ACL-enforced). This ADR makes that split deliberate
and resolves the spec contradiction.

## Decisions

1. **The sovereignty boundary is the ORG, not the department.** "Sovereign" means no
   foreign/managed AI or cloud at runtime (enforced by `assert_sovereign`, the egress
   allowlist) — it does NOT mean each department is its own island. Within one organisation's
   sovereign estate, departments may share infrastructure. `assert_sovereign` continues to
   guard *cross-org / foreign* egress; it does not forbid intra-org cross-department compute.

2. **Compute is centralized (shared), by default.** One self-hosted GPU/model fleet
   (embedder, reranker, LLM) serves every department in the org. Rationale: GPU cost
   amortisation and single-fleet operations (one thing to patch, scale, and secure). The
   model-connection registry (ADR-0016, FR-E1) keeps this model-agnostic — a department may
   still attach its own endpoint, but the *default* is the shared fleet.

3. **Data and permissions are decentralized (per department).** Vectors, sparse indexes,
   objects, catalog rows, and ACLs are namespaced per tenant and enforced at the data owner.
   Cross-department access is opened ONLY by a signed, revocable federation grant (ADR-0017).
   This is where security and independent scalability live.

4. **Federate the QUERY, not the vector** (resolves the spec contradiction). Because a raw
   vector is not portable across heterogeneous embedders, the federation contract carries the
   **query text** (+ the grant), and the answering side re-embeds and searches in ITS OWN
   space. This is strictly more correct than broadcasting a vector, and it is what the
   `sovereign-node` `/rag/query` path already does (the wire `FederatedQueryRequest` carries
   `query_text`; `query_vector` is optional and unused across a heterogeneous boundary).

5. **Compute topology selects the federation transport (ties to ADR-0017).**
   - **Intra-org, shared compute** → departments are co-located behind one embedder, so
     retrieval is in-process (`LocalTransport`). There is no peer to broadcast to and no
     vector to ship — the question does not arise.
   - **Inter-org, isolated compute** → each org runs its own embedder; vectors are not
     comparable, so the `MeshTransport` ships the query and the remote org re-embeds locally.
     Vector-sharing is never used across this boundary.

## Consequences

- **Cost + ops win intra-org:** one GPU fleet instead of one-per-department; data isolation
  and permissions are unchanged and independent of where compute runs.
- **Homogeneous-vector opportunity (intra-org):** because the shared embedder puts every
  department in ONE vector space, future intra-org features MAY compare/merge vectors directly
  (e.g. a single ANN index partitioned by tenant) — an option a per-department-embedder design
  forecloses. Not adopted now (per-tenant collections stay the isolation unit), but recorded
  as available.
- **Privacy trade-off on the mesh path (flag for privacy review):** shipping the query text to
  a peer means the peer's node sees the raw question. This is acceptable under the existing
  controls — the hop is grant-authorised, noise-encrypted, and audited (ADR-0017/0018) — but it
  is a real disclosure of the *question* (not the answer or the corpus). Privacy-sensitive
  deployments should note it; the research-grade alternative (private set intersection / PSI so
  a peer answers without seeing the question — hero innovation H2) is the future mitigation, out
  of scope here.
- **When to run decentralized compute:** separate legal entities, a compliance regime that
  forbids a shared inference tier seeing a department's text, or a department that must keep its
  embedding model private → that department runs its own node and joins over the mesh. The
  architecture already supports this via the transport seam; it is a deployment choice, not a
  rewrite.
- **Revisit when:** the org boundary itself must subdivide (a department becomes its own
  sovereignty domain), or PSI/H2 lands and removes the query-disclosure trade-off.

Related: ADR-0016 (per-KB models), ADR-0017 (federation transport seam), ADR-0018 (transport
security).
