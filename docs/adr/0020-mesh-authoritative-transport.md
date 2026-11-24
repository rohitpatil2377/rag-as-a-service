# ADR-0020 — One authoritative remote federated transport: libp2p, not HTTP-behind-Istio

**Status:** accepted · 2026-08-04
**Context:** ADR-0017 introduced a `FederationTransport` seam with two concretes —
`LocalTransport` (in-process, co-located departments) and `MeshTransport` (libp2p via the
`sovereign-node`). That resolved "how does `answer_federated` reach a peer" mechanically, but
left a strategic question open, and the conformance review named it the single biggest
architectural debt: **for a genuinely REMOTE department (separate deployment), what is THE
authoritative transport?** Two credible answers exist and only one should be built and
operated:

1. **libp2p `sovereign-node`** — a per-department p2p node; the initiator's node dials the
   peer's node over a noise-encrypted libp2p stream (`/rag/query/1.0`), the peer verifies a
   signed grant fail-closed and retrieves locally. Built, tested, interop-proven this cycle.
2. **HTTP-behind-Istio-mTLS** — the Python control-plane makes a direct HTTPS request to the
   peer's `rag-api`, peer identity carried by service-mesh mTLS (SPIFFE), the signed grant
   still the authorization.

Keeping both "live" is the debt: two code paths, two security models, two failure surfaces,
and ambiguity about which one production actually uses. Additionally, `LocalTransport`'s
in-process scatter-gather was described in the code as "the scatter-gather that proves the
federation model before the Go/libp2p node exists" — i.e. it was a *stand-in* for remote
federation, which risks it being used as one.

## Decision

1. **libp2p `sovereign-node` is THE authoritative remote federated transport.** All
   cross-deployment federation goes over the libp2p mesh. We do not build or operate an
   HTTP-to-remote-`rag-api` federation path.

   Rationale (why libp2p over HTTP-behind-Istio):
   - **One trust root, not two.** A node's libp2p Ed25519 identity IS the identity that signs
     and verifies grants (ADR-0018). HTTP-behind-Istio adds a *second* PKI (SPIFFE/X.509)
     that authenticates the transport but not the authorization — we would still need the
     grant layer underneath, so mTLS adds operational surface (cert issuance/rotation/CRLs, a
     mandatory service mesh) without removing anything.
   - **Sovereignty + offline verifiability.** The grant is offline-verifiable with only the
     issuer's public key; the mesh needs no central CA and no always-on mesh control plane.
     That matches the "sovereign, self-hostable, air-gappable" posture (air-gap bundle,
     ADR-0018 NetworkPolicy). An Istio dependency pulls a heavyweight always-on control plane
     into the sovereignty boundary.
   - **Ships the query, not the vector, natively.** The libp2p wire already carries the query
     text + grant and re-embeds at the peer (ADR-0019). The HTTP path would re-implement the
     same contract with none of the benefits.
   - **Failure surface is the node, which we own.** The mesh's fail-closed-on-authz /
     fail-open-on-availability semantics live in code we control end-to-end.

   HTTP-behind-Istio is **explicitly rejected**, recorded here so it is not re-proposed as an
   "obvious" alternative. It remains available ONLY as the optional mTLS SIDECAR overlay of
   ADR-0018 (mesh mTLS *between pods*, libp2p security *inside* the stream) — that is
   defence-in-depth on the SAME libp2p transport, NOT a competing HTTP transport.

2. **`LocalTransport` is scoped to CO-LOCATED departments only — it is not a remote
   transport.** It exists for the intra-org shared-compute model (ADR-0019) where peer KBs
   live in the same deployment and retrieval is in-process. It MUST NOT be used to reach a
   separately-deployed department; that is what `MeshTransport` is for. The seam is therefore
   **Local (co-located) + Mesh (remote)** — not two competing remote transports. This retires
   the "in-process scatter as a stand-in for remote federation" ambiguity (there is no dead
   HTTP code to delete; the duplicate was conceptual).

3. **Transport is chosen per peer, explicitly for remote peers.** `FederationRegistry` keeps
   `local` as the default (the common single-deployment case), and a genuinely remote peer is
   marked `mesh` (per-peer override or the operator registry file, ADR-0017/registry). A peer
   configured `mesh` with no reachable node fails **closed** (`NullTransport`, from the
   security review), never silently degrading to an in-process lookup.

## Consequences

- **One path to operate, secure, observe, and test.** Federation SLOs, tracing, and SIEM
  wiring (T1 follow-ups) target the libp2p mesh only.
- **No Istio/SPIFFE dependency is required** for federation; it stays an optional hardening
  overlay for shops that already run a mesh (ADR-0018), never a prerequisite.
- **Docstrings/comments updated** so `LocalTransport` reads as co-located-only and the
  pipeline no longer calls the in-process path a stand-in for remote federation.
- **Revisit only if** a customer's network policy forbids libp2p entirely (then the
  HTTP-behind-Istio path would need its own ADR reversing this one) — not expected.

Related: ADR-0017 (transport seam), ADR-0018 (transport security: noise authoritative, mTLS
optional overlay), ADR-0019 (compute/data split, ship-the-query).
