# ADR-0022 — Node identity, revocation, and hardened discovery (zero-trust federation)

**Status:** accepted · 2026-08-04
**Context:** Federation trust rests on two identities: a department's **libp2p peer identity**
(who is on the wire) and its **Ed25519 grant-signing key** (whose grants verify). Both were
*additive-only* — a node learned keys/peers via the operator registry file (ADR-0017/registry)
but could never **withdraw** trust. Three zero-trust gaps followed:

1. **No department-key revocation.** A compromised department's key kept verifying its grants
   and it kept being recognised as a caller — forever.
2. **No grant revocation across the mesh.** A grant carries `revoked_at`, but **revocation is
   NOT part of the signature**. A caller who holds the pre-revocation signed bytes can present
   a grant with `revoked_at` unset; the signature still verifies and `IsActive` still passes.
   Revocation cannot be trusted from the caller's copy.
3. **Discovery/bootstrap trust.** The question of which peers a node will *answer* was implicit.

The heavier question underneath: should department identity move to **SPIFFE/SPIRE** (X.509
SVIDs, automatic cert rotation) or stay on the **libp2p peerstore + Ed25519** identities the
node already uses?

## Decisions

1. **libp2p Ed25519 peer identity + the grant-signing key remain the identity root; SPIFFE/SPIRE
   is an OPTIONAL overlay, not required.** Consistent with ADR-0018/0020 (libp2p is the
   authoritative, CA-free transport): the peer identity IS the grant identity — one key, one
   trust root, offline-verifiable, air-gappable. SPIFFE/SPIRE is supported for shops that
   already run it (SVID becomes an *additional* peer-identity signal at the mTLS sidecar of
   ADR-0018), but it is never the root and never a prerequisite — adopting it would add a CA to
   operate (issuance, rotation, CRLs) without replacing the grant, which is the authorization of
   record.

2. **Revocation is enforced by the RESPONDER from its own lists, never trusted from the caller.**
   The registry gains two revocation sets, checked fail-closed in the query handler:
   - **Revoked departments** — `RevokeDept(dept)` makes `PublicKeyFor` and `DeptForPeer` return
     "unknown", so a compromised department's grants stop verifying AND it can no longer be
     recognised as a caller.
   - **Revoked grants** — `RevokeGrant(id)`; the handler refuses any grant whose id is listed,
     *after* the signature verifies, so a replayed pre-revocation copy is still denied. This is
     the piece that makes grant revocation real across the mesh.

3. **Revocation propagates via the operator registry file** (the channel operators already use):
   a department entry may be marked `"revoked": true`, and a file-level `"revoked_grants": [...]`
   lists grant ids to refuse mesh-wide. Shipping the updated file (a ConfigMap in K8s) is the
   propagation mechanism; a gossip channel over the existing `/rag/grant` protocol is a future
   enhancement, not required for correctness.

4. **Discovery is allowlist-first.** A node only *answers* departments present (and not revoked)
   in its registry — `DeptForPeer` gates every inbound query, and the strict default-deny
   NetworkPolicy (ADR-0018) fences the libp2p port at the network layer. Dialing OUT to bootstrap
   peers is initiator-controlled and unchanged; inbound trust is the registry ∧ NetworkPolicy.

## Consequences

- **Trust is now withdrawable**, closing the zero-trust gap: a compromised department or a
  leaked grant can be cut off by shipping an updated registry file, and the cutoff is enforced
  where it matters — on the answering node, independent of what the caller presents.
- **No CA to operate by default**; SPIFFE/SPIRE stays an opt-in overlay for mTLS-mandated
  environments (ADR-0018), decided per deployment.
- **Revocation latency = registry-file propagation time.** Until a gossip channel lands, an
  operator must distribute the updated file (seconds-to-minutes in a ConfigMap rollout). The
  bound is operational, and documented as the current revocation-propagation SLA.
- **Grant `revoked_at` still matters** for the ISSUER-side store (a revoked grant is dropped
  from `get_active`, so it is never handed out again); the responder revocation list is the
  belt-and-suspenders for a copy already in a caller's hands.

Related: ADR-0017 (transport seam + registry), ADR-0018 (transport security, NetworkPolicy),
ADR-0021 (query replay protection).
