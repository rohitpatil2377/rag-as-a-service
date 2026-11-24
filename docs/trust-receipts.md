# Cryptographic Trust Receipts

A **trust receipt** is a small, signed proof emitted for every grounded answer. It lets anyone
verify — **offline**, without trusting or contacting the Federated RAG platform — that an answer was
authorised, grounded in specific source spans, and produced by an approved model.

## What a receipt proves

A receipt commits (by SHA-256) to the exact facts of one answer, then signs the whole commitment
with the issuer's Ed25519 key:

- **Binding** — `tenant_id`, `answer_id`, `query_sha256`, `answer_sha256`. The receipt is tied to a
  specific tenant and to the exact query and answer bytes (not a paraphrase). Change one character of
  the answer and verification fails.
- **Grounding** — `span_digests`, one `sha256(text|chunk_id)` per cited span. A verifier holding the
  source documents can confirm the answer cited *those* spans and no others — the platform never has
  to disclose the span text to prove grounding.
- **Model integrity** — `model_id` + `model_weight_sha256`. Proof an approved model on approved
  weights produced the answer; a swapped or tampered model breaks the receipt.
- **Federation provenance** — `contributors` (`dept_id`, `grant_id`, `kb_ids`). Which departments,
  under which cross-department grants, supplied the evidence in a federated answer.
- **Ordering / anti-replay** — `prev_audit_hash` links the receipt into the audit hash-chain (below).
- **Authorship** — `signer_kid` names the signing key; `signed_token` is the detached signature.

## Offline verification flow

1. Obtain the receipt JSON and the issuer's public key (`signer_kid` says which key). Both are static
   files — no live service required.
2. Recompute the canonical bytes: sorted-keys JSON of the receipt with no whitespace, **excluding**
   `signed_token` (a signature cannot cover itself).
3. Check the Ed25519 signature over those bytes. Valid → `CLEAN`; anything wrong → `TAMPERED`.

Verification is **fail-closed**: `verify_receipt` returns a `bool` and never raises, so a malformed
receipt, a bad key, or an empty signature all read as untrusted rather than crashing the verifier.

```
python scripts/verify_receipt.py answer.receipt.json issuer.pub.b64
# -> CLEAN   (exit 0)   or   TAMPERED (exit 1)
```

Optionally, a verifier who also holds the sources recomputes each `sha256(text|chunk_id)` and matches
it against `span_digests` to confirm the answer was grounded in exactly those spans.

## Why it is chained to the audit hash-chain

`prev_audit_hash` is the SHA-256 of the immediately-preceding audit entry. Because each receipt embeds
the hash of the prior link *and is itself signed*, receipts inherit the audit log's tamper-evidence:
you cannot back-date, reorder, or drop a receipt without breaking the chain, and you cannot forge one
without the issuer's private key. The receipt is a portable, self-contained slice of that chain — it
travels with the answer instead of living only inside the platform's database.

## Prior-art delta

Centralised RAG systems log provenance *for themselves*: the audit trail is internal, and trusting it
means trusting the operator. A trust receipt inverts that. It is **cross-organisational** (contributors
and grants span federated departments), **offline-verifiable** (a signed, self-contained proof needing
only a public key — no callback to the platform), and **answer-scoped** (one portable receipt per
grounded answer, committing to the query, answer, model weights, and exact cited spans). No centralised
RAG platform emits a per-answer, third-party-verifiable provenance proof of this kind.
