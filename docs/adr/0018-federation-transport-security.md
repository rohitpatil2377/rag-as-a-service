# ADR-0018 — Federation transport security: libp2p noise/TLS vs mTLS (Istio/SPIFFE)

**Status:** accepted · 2026-08-04
**Context:** The Federated RAG mesh (ADR-0017) moves grant-authorised queries between
department nodes over the network. That hop needs two guarantees: the channel is
**encrypted**, and each peer's **identity is cryptographically verifiable** (so a node knows
it is talking to the department it thinks it is, not an impostor). The obvious enterprise
default for "encrypted + mutually-authenticated" is **mTLS** (X.509 certs, usually delivered
by a service mesh — Istio/Linkerd — with SPIFFE/SVID identities). The `sovereign-node` was
built on **libp2p**, which provides its own transport security. So which is the mechanism of
record, and what is the production stance?

What already exists in the code:
- **libp2p transport security** — `node/host.go` builds the host with `libp2p.New(...)`,
  which uses libp2p's default security transports (**noise** + **TLS 1.3**). Both are keyed
  to the node's persisted **Ed25519 identity**; the dialer verifies the responder's PeerID
  (derived from that key) as part of the handshake. Encryption + mutual peer authentication
  are therefore already in place at the transport layer, with no external PKI.
- **A signed, revocable Grant** as the authorization of record — verified fail-closed on
  every `/rag/query` (Go `AllowsTopics`/expiry/signature; interop-identical to Python).
- **A strict default-deny NetworkPolicy** (`infra/helm/sovereign-node/templates/
  networkpolicy.yaml`) — the pod may only be reached by the control plane + peer nodes, and
  may only egress to its OWN department's rag-api + listed bootstrap peers. Network-layer
  isolation independent of the application.
- The helm README already names **"mTLS (Istio/SPIFFE)"** as the intended alternative.

## Decision

1. **libp2p transport security (noise + TLS 1.3, Ed25519 peer identity) is the authoritative
   mechanism.** It gives encryption and mutual peer authentication end-to-end between nodes
   with no certificate authority to run, and the peer identity is the SAME key that signs and
   verifies grants — one identity, one trust root. We do **not** require mTLS underneath it.

2. **The Grant is the authorization of record; transport security only proves *who* the peer
   is, not *what* it may read.** mTLS/noise answer "is this really department X?"; the signed
   grant answers "may X read this, for these topics/KBs, until when?" — the question that
   actually protects the data. A second PKI would re-answer the first question without
   touching the second.

3. **mTLS via a service mesh is a SUPPORTED, OPTIONAL overlay — not required, never a
   replacement.** For deployments already standardised on Istio/Linkerd, or where a security
   authority mandates X.509/SPIFFE peer identity independent of the application, run the
   node's pod inside the mesh with STRICT mTLS at the sidecar. libp2p noise still terminates
   *inside* the node, so this is belt-and-suspenders (mesh mTLS between pods, libp2p security
   inside the stream) with no conflict. The Grant remains authz of record; a SPIFFE SVID
   becomes an *additional* identity signal, not the trust root.

4. **Network isolation is mandatory and orthogonal.** The default-deny NetworkPolicy ships
   enabled and is required in every clustered deployment regardless of the mTLS choice — it
   is what makes "a node can only reach its own rag-api and its peers" true at the network
   layer, closing the blast radius if application logic is ever wrong.

## Production stance (choose deliberately per deployment)

| Deployment | Transport security | Rationale |
|---|---|---|
| **Default / on-prem VM / small cluster** | libp2p noise+TLS + Ed25519 + NetworkPolicy | No CA to operate; peer identity == grant identity; fewer moving parts. |
| **Already on a service mesh, or X.509/SPIFFE mandated** | Above **+ STRICT mTLS sidecar** (Istio/Linkerd, SPIFFE SVID) | Satisfies an org PKI/compliance requirement; additive, no code change in the node. |
| **Regulated multi-org (future)** | Above + SPIFFE as a federation-wide identity root | If peer identity must be attestable by a third party independent of libp2p keys. |

The default is the first row. Moving to mTLS is an operational/helm choice, not a code
change, so the stance can be set per customer without forking the node.

## Consequences

- **No CA to run by default.** Peer identity is the node's Ed25519 key (persisted, the same
  key that signs grants) — no cert issuance/rotation/CRL for the base deployment.
- **mTLS stays available with zero code cost.** Because it is a sidecar/mesh concern, an
  operator adds it via helm values + a mesh, not by changing `sovereign-node`.
- **Defence in depth is explicit:** transport security (who) + Grant (what) + NetworkPolicy
  (reachability). Compromising one does not defeat the others.
- **Revisit when:** a customer requires certificate-based peer identity attestable outside
  libp2p (adopt SPIFFE as an additional root), or a compliance regime mandates mTLS on every
  hop (make the mTLS sidecar the documented default for that customer). This ADR records the
  deviation from a "mTLS everywhere" default and why it is deliberate, per ADR-0017 §4.
