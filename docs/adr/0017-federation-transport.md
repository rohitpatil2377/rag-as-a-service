# ADR-0017 — Federation transport unification (local in-process vs libp2p mesh)

**Status:** accepted · 2026-08-04
**Context:** Federated RAG cross-department federation had grown TWO transports that did not
meet:

1. **Python in-process** — `pipeline.answer_federated` retrieved a foreign department's KBs
   by calling `wiring.retrieval_service_for(kb)` directly. This is the LIVE path, but it
   only works when every department lives in the SAME deployment (the modular monolith,
   multi-tenant by namespace). It cannot reach a department that runs elsewhere.
2. **Go libp2p `sovereign-node/`** — a peer-to-peer node with noise-encrypted
   `/rag/query/1.0` + `/rag/grant/1.0` handlers, whose `grant/` package is byte-for-byte
   interop-compatible with the Python signer (`grants.py`, locked by
   `interop_python_test.go`). It CAN reach a remote department, but nothing in the Python
   answer path ever routed to it — it was built and tested, not wired.

Two transports, one contract, no bridge between them. The federation pillar
("decentralized cryptographic service mesh, not a monolithic server") was therefore only
half-true: the live answer was monolithic; the mesh existed but was dark.

## Decisions

1. **Unify behind a `FederationTransport` seam, not by deleting a transport.** The two are
   not redundant — they cover different *topologies*. `svc_control_plane/federation.py`
   defines `FederationTransport` (`fetch(caller_dept, peer_dept_id, grants, query, top_k)
   -> (per_source, used_grants)`), with two concretes:
   - **`LocalTransport`** — the previous in-process retrieval, byte-for-byte. It is the
     **default** and identity-preserving: with `RAG_FEDERATION_TRANSPORT=local`,
     `answer_federated` behaves exactly as before this ADR (the federated e2e tests pass
     unchanged).
   - **`MeshTransport`** — routes the grant-authorised query to THIS deployment's local
     `sovereign-node` (`POST {federation_node_url}/federate`), which dials the remote peer
     over libp2p and returns spans mapped back to `RetrievedSpan`.

   `answer_federated` now resolves a transport **per peer** via a `FederationRegistry`, so a
   deployment can be monolithic for some departments and meshed for others.

2. **Local is authoritative for co-located departments; the mesh is authoritative across
   deployments.** There is no third "which one wins" question: the peer's *location*
   decides. A single-deployment install (dev, POC, one data-centre tenant) stays entirely
   local and pays nothing for the mesh. A genuinely federated install (departments on
   separate nodes) uses the mesh for the remote hops and local for its own KBs, in the same
   answer.

3. **Grant resolution stays in the caller; the mesh re-validates fail-closed.**
   `answer_federated` calls `grant_store.get_active(issuer, subject)` for BOTH transports —
   a department must hold an active grant to even attempt a peer. For the mesh path that
   same signed grant is the token presented to the remote node, whose `/rag/query` handler
   verifies it fail-closed before retrieving. This is defence-in-depth: local checks in
   Python, remote checks again at the boundary it actually protects.

4. **noise (libp2p) substitutes for mTLS at the transport layer.** The mesh is
   authenticated + encrypted by libp2p's noise handshake keyed to each node's Ed25519
   identity, and every request is additionally gated by a signed, revocable Grant. We do
   NOT run mTLS underneath: the grant is the authorization of record and is already
   end-to-end verifiable offline, so a second PKI (cert issuance/rotation/CRLs) would add
   operational surface without adding a distinct security guarantee. SPIFFE/mTLS remains
   open as a future option if an operator's network policy demands certificate-based peer
   identity independent of the application grant. **This decision is recorded in full, with a
   per-deployment production stance, in ADR-0018.**

5. **The mesh node URL is internal infrastructure, not egress.** `federation_node_url`
   points at the operator's OWN local `sovereign-node` (localhost / intranet), exactly like
   `RAG_QDRANT_URL` or `RAG_REDIS_URL`, so it is not run through `assert_sovereign`. The
   cross-organisation hop happens *inside* the libp2p mesh, where the remote node enforces
   the grant. The runtime egress guard is unchanged.

## Consequences

- **Default-safe + additive.** New settings `federation_transport` (default `local`),
  `federation_node_url`, `federation_peers` (per-peer JSON override). Everything off/local
  by default; no behaviour change to the single-node path; `make ci` stays green.
- **Availability fails open, authorisation fails closed.** An unreachable mesh peer
  contributes zero spans and is logged, never crashing the federated answer; an invalid or
  revoked grant yields zero spans by construction.
- **Score hygiene preserved across the wire.** A remote span's score is clamped to `[0,1]`
  and named (`scorer="cross_encoder"`, or `unscored` at 0), so the grounded-or-abstain gate
  reads it on a known scale — the wire cannot smuggle in an off-scale confidence.
- **Outbound mesh wiring (implemented 2026-08-04):** the Go node's *outbound* control endpoint
  (`POST /federate` → `QueryPeer` dials the peer over libp2p → returns spans) now exists
  (`sovereign-node/node/control.go`, `protocols/query_client.go`), so the mesh path is wired
  end-to-end and `MeshTransport` has a real server to talk to — verified by a two-host libp2p
  round-trip test. Remaining: a live MULTI-DEPLOYMENT interop run (two nodes + two rag-apis)
  and carrying `category` on the wire so topic-enforcement (L1) reaches mesh spans. `local`
  stays the default production path.

## Grant-topic scoping (L1 — addressed, flag-gated)

- **Grant `topics` are not enforced on the in-process retrieval path (pre-existing).** A grant
  scopes access to `topics` AND `kb_ids`, but `RetrievalQuery` carries no topic filter, so
  `LocalTransport` queries the whole granted KB with the caller's arbitrary text — `topics` is
  decorative there. A subject granted a narrow topic on a KB can retrieve any public-ACL content
  in that KB. This is NOT introduced by the seam (the pre-seam loop was identical), and the mesh
  path forwards the full grant so the remote node CAN enforce topics. Closing it means adding a
  topic filter to `RetrievalQuery` and applying it on the local path — tracked as a federation-
  model authorization-scoping follow-up, out of scope for this transport-unification ADR.

**Status: addressed (2026-08-04).** `RetrievedSpan` now carries the chunk's primary
`category` (FR-B7 taxonomy), and `RAG_FEDERATION_TOPIC_ENFORCEMENT` (default OFF) applies a
fail-closed filter in both transports: with it ON, a peer's span is returned only if its
category is one of the grant's `topics` (case-insensitive); a span with no category is
dropped. This assumes an operating model where `grant.topics` name the publisher KB's
taxonomy categories. Default OFF preserves today's behaviour (topics validated by the mesh
node, not content-filtered).
