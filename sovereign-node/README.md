# SovereignNode — libp2p federation transport

A peer-to-peer federation node for **RAG as a Service / Federated RAG**. Each department runs one
`sovereign-node`. Nodes form a libp2p mesh and exchange **grant-authorised** federated
queries: a department can retrieve from another department's knowledge base *only* if it
holds a valid, signed **Grant** authorising it — verified **fail-closed** on every request.

The `grant/` package is **byte-for-byte interop-compatible** with the Python control-plane
signer (`services/control-plane/src/svc_control_plane/grants.py`): a grant signed on the
Python side verifies here and vice-versa. `grant/interop_python_test.go` locks that in.

## Layout

| Path | Role |
|---|---|
| `grant/` | **Security core** — the Grant struct + Ed25519 sign/verify, canonical claims (Python-interop). Stdlib-only, offline-testable. *Do not reimplement.* |
| `protocols/wire.go` | JSON wire structs (`FederatedQueryRequest`/`Response`, `GrantAnnounce`/`Ack`) + protocol IDs. |
| `protocols/query.go` | `/rag/query/1.0` inbound handler — fail-closed grant validation then local retrieval. |
| `protocols/grant.go` | `/rag/grant/1.0` inbound handler — verify + store an announced grant. |
| `bridge/rag_client.go` | HTTP client to the **local** rag-api (`/v1/kb/{id}/retrieve`). |
| `node/host.go` | libp2p host: identity (load/generate/persist), listen addrs, bootstrap dial. |
| `node/registry.go` | In-memory dept → (Ed25519 pubkey, peer id, multiaddr) directory. |
| `protocols/query_client.go` | **Outbound** `/rag/query/1.0` client — dial a peer, ship the query, read spans. |
| `node/control.go` | **Loopback control API** — `POST /federate` drives an outbound federated query. |
| `main.go` | CLI wiring it all together. |

## Build & run

Requires **Go 1.26+**.

```bash
cd sovereign-node
go mod tidy      # fetches libp2p + multiaddr (network required the first time)
go build ./...
go test ./...
```

> **Network note:** the libp2p/multiaddr modules are fetched from the Go module proxy on
> the first `go mod tidy`/`go build`. Once cached (`$GOMODCACHE`) the build is offline.
> The `grant/` package depends only on the Go standard library, so it builds and tests
> **offline with zero external deps** — `go vet ./grant/` and `go test ./grant/` are always
> green regardless of network.

Run a node:

```bash
sovereign-node \
  --dept finance \
  --listen /ip4/0.0.0.0/tcp/4001 \
  --rag-url http://localhost:8000 \
  --rag-token "$RAG_TOKEN" \
  --key ./identity.key \
  --bootstrap /ip4/10.0.0.5/tcp/4001/p2p/12D3KooW...
```

On start it prints its **PeerID** and dialable **/p2p/ listen addrs** — feed one node's
printed addr into another node's `--bootstrap` (comma-separated for several).

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--dept` | *(required)* | This node's department id. |
| `--listen` | `/ip4/0.0.0.0/tcp/4001` | libp2p listen multiaddr. |
| `--rag-url` | `http://localhost:8000` | Base URL of the local rag-api. |
| `--rag-token` | *(empty)* | Bearer token for the local rag-api. |
| `--bootstrap` | *(empty)* | Comma-separated bootstrap peer multiaddrs (each ending `/p2p/<peerid>`). |
| `--key` | *(empty = ephemeral)* | Identity key file (generated + persisted 0600 if absent). |
| `--dev-secret` | Python dev default | Dev JWT secret used to derive this dept's dev grant-signing key (interop with the Python signer). |
| `--control-listen` | *(empty = disabled)* | Loopback addr for the local control API that drives OUTBOUND federated queries (e.g. `127.0.0.1:7070`). |
| `--registry` | *(empty)* | Path to an operator registry JSON file (dept → pubkey + peer address) so this node can verify and reach peer departments. |
| `--replay-skew-seconds` | `120` | Max clock skew for accepting a federated query; also bounds the replay-nonce cache (ADR-0021). |

### Control API (outbound federation)

With `--control-listen 127.0.0.1:7070` the node also exposes a **loopback** HTTP control
plane so the department's own rag-api / Python `MeshTransport` can drive OUTBOUND queries:

- `POST /federate` `{peer_dept_id, query_text, top_k, grant}` → the node resolves the peer in
  its registry, dials it over libp2p (`/rag/query/1.0`), and returns the peer's
  `FederatedQueryResponse`. It ships the QUERY, never a vector (ADR-0019). Unknown peer or a
  dial failure fail closed with no spans (never a 500).
- `GET /health` → `{status, dept}`.

Bind loopback/intranet only — it acts on behalf of the co-located rag-api and carries no
auth of its own; the pod's NetworkPolicy is what fences it.

### Operator registry file (`--registry`)

Two independently-deployed nodes federate without any in-process seeding by loading the same
kind of **operator registry file** — a JSON directory of the peers each node is allowed to
verify and reach:

```json
{
  "departments": [
    { "dept": "hr",
      "dev_secret": "dev-only-insecure-secret-change-me-32b+",
      "peer_addr": "/ip4/10.0.0.11/tcp/4001/p2p/12D3KooW…hrPeerId" },
    { "dept": "finance",
      "pubkey_b64": "<base64 raw 32-byte Ed25519 public key>",
      "peer_addr": "/ip4/10.0.0.12/tcp/4001/p2p/12D3KooW…financePeerId" }
  ]
}
```

Per entry (all fields but `dept` optional — supply what THIS node needs for that peer):

- `pubkey_b64` — base64 (std) raw 32-byte Ed25519 grant-signing key; needed to VERIFY grants
  that department issued. `dev_secret` derives the dev key instead (dev/POC interop with the
  Python signer). Omit both for an address-only entry (reachability without a verify key).
- `peer_addr` — a full `/p2p/<peerid>` multiaddr: supplies the peer id (subject check /
  reverse lookup) and a dialable address (outbound queries). The node dials it on startup.

A bad line for one peer is logged and skipped — it never stops the node learning the others.
See `registry.example.json`.

**Revocation (zero-trust, ADR-0022).** The same file withdraws trust — enforced by the
RESPONDER, since revocation is not part of a grant's signature:

- `"revoked": true` on a department entry — its key stops verifying grants and it can no
  longer call (compromised-department cutoff).
- a file-level `"revoked_grants": ["grant_id", ...]` — the node refuses those grant ids even
  if a caller presents a still-signature-valid copy.

Propagation = shipping the updated file (a ConfigMap rollout in K8s).

### Live two-node run

```bash
# On the HR host — start the node, print its PeerID + listen addr:
sovereign-node --dept hr --listen /ip4/0.0.0.0/tcp/4001 \
  --rag-url http://localhost:8000 --rag-token "$HR_TOKEN" \
  --key ./hr.key --control-listen 127.0.0.1:7070 --registry ./registry.json

# On the FINANCE host — same, with its own key/rag-api and the shared registry:
sovereign-node --dept finance --listen /ip4/0.0.0.0/tcp/4001 \
  --rag-url http://localhost:8000 --rag-token "$FIN_TOKEN" \
  --key ./finance.key --control-listen 127.0.0.1:7070 --registry ./registry.json

# Then finance's rag-api (or you, for a smoke test) drives an OUTBOUND federated query:
curl -s http://127.0.0.1:7070/federate -H 'content-type: application/json' -d '{
  "peer_dept_id": "hr", "query_text": "annual leave days", "top_k": 5,
  "grant": { … a grant HR signed for finance … }
}'
```

Each `--registry` file lists the PeerIDs printed by the nodes on startup. In Kubernetes the
same file is a ConfigMap and the NetworkPolicy (see `infra/helm/sovereign-node`) fences which
peers are reachable at the network layer.

## The two protocols

### `/rag/query/1.0` — federated retrieval (fail-closed)

Peer sends a `FederatedQueryRequest` (JSON) carrying the **full signed grant**. The handler
validates, **in order**, and on **any** failure replies `{grant_valid:false, spans:[]}` —
**never** a transport error or 500 (silent fail-closed):

1. issuer public key for `grant.issuer_dept_id` is known to this node's registry;
2. `grant.Verify(issuerPubKey)` — Ed25519 signature over the canonical claims;
3. `grant.IsActive()` — not revoked, within `[not_before, expires_at)`;
4. `grant.subject_dept_id` **equals the calling peer's department** (registry reverse-lookup
   of the stream's remote peer id) — blocks replay of a grant issued to someone else;
5. requested `topics ⊆ grant.topics`;
6. requested `kb_id ∈ grant.kb_ids` (defaults to the grant's first kb if unset).

Only if **all** pass does it call the local rag bridge, capped at
`grant.max_results_per_query`, and return the spans (stamped with this node's
`source_dept_id`). If the grant was valid but local retrieval errored, it replies
`grant_valid:true` with `error:"retrieval_failed"` and no spans — distinguishing "not
allowed" from "couldn't fetch".

### `/rag/grant/1.0` — grant announcement

A peer sends a `GrantAnnounce` (a signed `Grant`). The receiver verifies the signature
against the issuer's registered public key and, **only if it is the grant's subject or
issuer**, stores it (`GrantStore`) for later use; otherwise it replies with a rejecting
`GrantAck` (`unknown_issuer` / `bad_signature` / `not_subject`). Verification failure is a
rejection, never a transport error.

## Mapping to the Python grant model

`grant.Grant` mirrors `rag_schemas.federation.Grant`. `CanonicalClaims()` reproduces
Python's `_canonical_claims` exactly — sorted keys, no whitespace (`separators=(",",":")`),
ISO-8601 UTC timestamps with a `+00:00` offset (not `Z`), `topics`/`kb_ids` sorted
ascending — so the detached Ed25519 signature (base64 std) is portable in both directions.
The dev signing key is derived identically: `sha256("grant-dev::" + dev_jwt_secret)` as the
Ed25519 seed. This node verifies grants your Python control-plane issued, with no shared
runtime state — the grant is self-describing and self-authenticating.
