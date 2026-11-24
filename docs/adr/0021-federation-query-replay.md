# ADR-0021 — Replay protection on the federated query (nonce + timestamp), not just the grant

**Status:** accepted · 2026-08-04
**Context:** A federation grant is a LONG-LIVED, reusable credential — signed once, valid for
its `[not_before, expires_at)` window, presented on every query. That is correct for
*authorization* ("may finance read hr's leave-policy KB?"), but it says nothing about a
single *request*: the same signed `FederatedQueryRequest` bytes, if captured, describe a
valid query that a grant check would pass again. The grant's own TTL/revocation does not
prevent a captured in-window request from being re-run. We want per-request replay
protection: a captured query cannot be re-sent to re-execute it.

The transport already provides strong guarantees the design builds on (ADR-0018): every
query rides a **noise-encrypted, peer-authenticated** libp2p stream, and the responder's
step-4 check binds the request to the authenticated caller (the remote peer id must map to
the grant's `subject_dept_id`). So an off-path attacker cannot capture plaintext or inject a
request as another department. What remains is a *same-party / same-path* replay and
freshness bound.

## Decision

1. **Every federated query carries a single-use `nonce` + an `issued_at` (RFC3339 UTC)
   timestamp**, added on the wire (`FederatedQueryRequest`). They are **stamped by the
   initiator's node on the outbound hop** (`QueryPeer`), so the Python `MeshTransport` path
   gets them for free (it POSTs to the local node's `/federate`, which calls `QueryPeer`).

2. **The responder enforces freshness + single-use via a `ReplayGuard`** (`protocols/replay.go`),
   checked fail-closed right after the subject-dept binding:
   - **Freshness:** reject if `|now - issued_at| > skew` (default 120s, `--replay-skew-seconds`).
   - **Single-use:** reject if the `(subject_dept, nonce)` pair has been seen; otherwise record
     it for `2×skew` (long enough to outlast the freshness window) and evict on expiry so the
     cache stays bounded.
   A missing/unparseable nonce or timestamp is rejected. Same nonce from a *different* subject
   is a different key (allowed) — the cache is scoped per calling department.

3. **The nonce/timestamp are NOT part of the grant signature.** They are per-request and added
   after signing; their integrity comes from the noise-encrypted stream (they cannot be
   tampered in flight) and their authenticity from the peer-authenticated caller. This keeps
   the grant canonical form (Python/Go byte-interop) untouched.

## Consequences

- **A captured query cannot be replayed** within the mesh trust model: re-sending it trips the
  seen-nonce cache; sitting on it past the window trips the freshness check.
- **Grant hardening stays orthogonal.** The grant already carries `not_before`/`expires_at`
  and revocation; this ADR adds *request*-level anti-replay. Grant rotation and
  revocation-propagation latency bounds are separate follow-ups.
- **Bounded memory.** The nonce cache is self-evicting (`2×skew` TTL, GC on each check), so a
  flood of distinct nonces cannot grow it without bound beyond the window.
- **Clock dependence.** Freshness needs reasonably synced clocks (NTP); the ±skew window
  tolerates normal drift. A wildly-wrong clock fails closed (rejects), never open.
- **Not yet: transport-independent query signatures.** If a future deployment needs replay
  protection WITHOUT trusting libp2p peer authentication (e.g. an HTTP relay in the path — not
  our design per ADR-0020), the subject would additionally SIGN `(grant_id, query_hash, nonce,
  issued_at)` with its Ed25519 key, verified against the subject's registered public key. The
  wire + registry already carry everything needed; only the signature step would be added.
  Out of scope while libp2p is the one authoritative remote transport (ADR-0020).

Related: ADR-0017 (transport seam), ADR-0018 (transport security), ADR-0020 (libp2p
authoritative).
