# Residency attestation

`services/control-plane/src/svc_control_plane/compliance.py` produces two different residency
reports. Both answer "where does the data live and is it protected", but for different
audiences and with different trust models.

## Per-KB attestation (existing, unsigned) — `attestation(kb_id)`

An on-demand report for ONE knowledge base: sovereignty enforcement, the egress allowlist,
encryption posture, audit-chain integrity, retention policy, and the KB's federation posture.
It is generated from LIVE state at request time (it walks the audit chain, reads retention
config, counts documents) and returned unsigned — it's meant to be read directly by whoever
called the API, in the same trust boundary as the control plane itself.

## Per-node residency attestation (new, signed) — `node_residency_attestation(...)`

A different document for a different audience: the report an OPERATOR hands a REGULATOR (or
an auditor who does not otherwise trust the control plane) as proof that a specific NODE's
data — across every KB resident on it — never leaves its region. Because the audience is
external and adversarial-tolerant, the report is cryptographically SIGNED (detached Ed25519,
same scheme as the offline-update bundle signing in `rag_common.update_manifest`) so it can be
handed off, stored, and verified independently — any edit to a single field invalidates the
signature.

### The model — `NodeResidencyAttestation`

| field | meaning |
|---|---|
| `attestation_id` | unique id for this report |
| `node_id` | the node being attested |
| `region` | e.g. `"India"` |
| `kb_ids` | KBs resident on this node (sorted) |
| `data_residency` | human-readable residency claim |
| `no_foreign_egress` | always `True` — the platform enforces egress allow-listing at the network layer |
| `generated_at` | when the report was assembled |
| `signer_kid` | key id of the Ed25519 key that signed it |
| `signed_token` | detached base64 Ed25519 signature over every other field |

### Sign — operator flow

```python
att = await node_residency_attestation(
    node_id="node-blr-01",
    region="India",
    kb_ids=["acme__hr", "acme__finance"],
    private_key_b64=OPERATOR_PRIVATE_KEY_B64,  # caller-supplied, never read from config here
    signer_kid="acme-release-2026",
)
```

`node_residency_attestation` assembles the report and returns it already signed. The private
key is an explicit function argument — signing is deliberately kept pure and free of any config
or settings lookup, so it can be tested with an ephemeral keypair and reasoned about
independently of deployment wiring.

Internally: `sign_node_attestation(att, private_key_b64)` computes the signature and returns a
*copy* of `att` with `signed_token` set — the input attestation is never mutated. The bytes that
get signed come from `_node_canonical_bytes(att)`: `model_dump(mode="json")` with `signed_token`
excluded, serialised as sorted-key JSON with no whitespace, so re-serialising the same
attestation always produces byte-identical input (the same scheme
`rag_common.update_manifest.canonical_bytes` uses for bundle manifests).

### Verify — regulator/auditor flow

```python
ok = verify_node_attestation(att, PUBLISHED_PUBLIC_KEY_B64)
```

`verify_node_attestation` is fail-closed: it returns `False` for a bad key, a malformed or
missing signature, or a tampered field (node id, region, KB list, residency string, timestamp —
anything). It never raises.

### Key differences from the per-KB report

| | `attestation(kb_id)` | `node_residency_attestation(...)` |
|---|---|---|
| scope | one KB | one node (many KBs) |
| signed | no | yes (Ed25519, detached) |
| audience | internal caller | external regulator/auditor |
| source data | live audit chain + catalog + retention config | caller-supplied node/region/KB list |
| tamper-evident | no | yes — signature invalidates on any edit |
