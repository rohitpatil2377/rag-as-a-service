# Decentralized Federated RAG mesh — 3-department demo (dept-a · dept-b · dept-c)

A live, **decentralized peer-to-peer** federation: each department runs its **own
`sovereign-node`** (distinct libp2p identity + Ed25519 grant key). Departments answer each
other's questions **only** with a valid, **signed grant** — verified fail-closed on every
hop. No central authority; a query ships over the libp2p mesh, never a copy of the data.

The A+B+C "one owner asks across all three" flow in `demo.sh` is **just one example** of the
general capability — any department can grant/query/revoke any peer.

## Topology (this deployment)

Per the "shared infra, isolated namespaces" decision:

```
        dept-a                     dept-b                     dept-c
   ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
   │  node-a (go)  │◄────────►│  node-b (go)  │◄────────►│  node-c (go)  │   libp2p /rag/query/1.0
   │  libp2p id A  │  grants  │  libp2p id B  │  grants  │  libp2p id C  │   (grant-checked, fail-closed)
   └──────┬────────┘          └──────┬────────┘          └──────┬────────┘
          │ bridge /v1/kb/{dept}/retrieve  (per-dept bearer token)          │
          └────────────────────────── shared rag-api ─────────────────────┘
                         shared Qdrant / OpenSearch / Postgres / MinIO
                         (per-tenant namespaces: dept-a | dept-b | dept-c)
```

**What is genuinely decentralized:** node identity, the libp2p transport, grant
sign/verify, and zero-trust revocation — three independent nodes dialing each other,
each responder enforcing grants on its own. **What is shared (dev only):** the physical
api + stores (isolated per tenant), and — because the shared api holds one signer — the
dev grant key is derived from a shared secret. **Production hardening:** one api + one
KMS-backed grant key *per department*, registry carries `pubkey_b64` not `dev_secret`.

## Run

```bash
cd infra/compose
./mesh/bootstrap.sh     # build nodes, generate identities, write the peer registry, wire the mesh
./mesh/demo.sh          # provision 3 dept KBs, issue grants, ask across A+B+C, then revoke
```

`bootstrap.sh` is idempotent. Bring the mesh down with:
`docker compose -f docker-compose.dev.yml -f docker-compose.deploy.example.yml -f docker-compose.override.yml -f docker-compose.mesh.example.yml down node-a node-b node-c`

## The api must have federation enabled

The federated **answer** path (`POST /v1/ask` with `federate_from`) needs, on the api:

```
RAG_FEDERATION_ENABLED=true
RAG_FEDERATION_TRANSPORT=mesh
RAG_FEDERATION_NODE_URL=http://<this dept's node>:7070   # the local node that dials peers
RAG_FEDERATION_PEERS={"dept-b":{"transport":"mesh"},"dept-c":{"transport":"mesh"}}
```

Because the single shared api has one `federation_node_url`, routing the *unified ask*
itself over libp2p per-caller wants **one api per department** (each pointing at its own
node). That per-dept-api split is the documented next step; the node-to-node libp2p hop,
grants, and revocation are already fully real via each node's control API (`POST /federate`).

## Onboarding a NEW department (dept-d)

1. Add a `node-d` service to `docker-compose.mesh.example.yml` (copy `node-c`, `--dept=dept-d`).
2. Re-run `./mesh/bootstrap.sh` — it regenerates identities, appends dept-d to the
   registry with its `/dns4/…/p2p/<peerid>` address, and every node reloads the registry.
3. Issue grants to/from dept-d (`POST /v1/grants`) and it's in the mesh.

## Files

| File | Role |
|---|---|
| `../../../sovereign-node/Dockerfile` | builds the Go libp2p node (vendored → offline/sovereign build) |
| `../docker-compose.mesh.example.yml` | node-a/b/c services (join the base stack's network + shared api) |
| `mesh/registry.json` | operator peer directory (dept → grant key + `/p2p` addr); also carries revocation (`revoked`, `revoked_grants`) |
| `mesh/bootstrap.sh` | identities → registry → per-dept tokens → wire the mesh |
| `mesh/demo.sh` | the A+B+C federated-ask + revocation example |
