# Federated RAG Grant Token — Open Specification v1

**Status:** Draft v1 · **Layer:** federation authorisation · **Crypto:** Ed25519 (RFC 8032)

A *grant token* is the only thing that opens a hole in Federated RAG's default-deny tenant
isolation: an **issuer** department cryptographically authorises a **subject** department to
read specific `topics` from specific `kb_ids`, bounded by a validity window and revocable.
This document specifies the wire format so any implementation — Python, Go, or a third
party — can issue and verify byte-identical tokens. The reference implementations are
`libs/common`/`svc_control_plane` (Python) and `sovereign-node/grant` (Go); their interop is
proven by `sovereign-node/grant/interop_python_test.go`.

Publishing the format is deliberate: an open, adoptable grant standard is an
industry-leadership play, and it *strengthens* rather than weakens the defensibility of the
surrounding protocol (federated scatter-gather with local search + central rerank).

## 1. Grant claims

A grant is a set of claims plus a detached signature over their canonical form.

| Field | Type | Notes |
|---|---|---|
| `grant_id` | string | Unique id of this grant. |
| `issuer_dept_id` | string | Department that OWNS the KBs and signs the grant. |
| `subject_dept_id` | string | Department authorised to query. |
| `topics` | string[] (≥1) | Taxonomy categories the subject may read. |
| `kb_ids` | string[] (≥1) | Knowledge bases the subject may query. |
| `max_results_per_query` | int (1–20) | Per-query result cap. |
| `not_before` | timestamp | Grant inactive before this instant. |
| `expires_at` | timestamp | Grant inactive at/after this instant. |
| `signed_token` | string | base64 Ed25519 detached signature (see §2). Not itself signed. |
| `revoked_at` | timestamp \| null | Set on revocation (enforced by the responder, not the signature). |
| `parent_grant_id` | string \| null | Present only on a re-delegated grant (§4). |

## 2. Canonical form + signature

The signature covers the **canonical bytes**: a JSON object with **sorted keys**, **no
whitespace** (`separators=(",",":")`), and **ISO-8601 UTC** timestamps. Only these keys are
included, in this exact shape:

```
{"expires_at":"<ISO-UTC>","grant_id":"...","issuer_dept_id":"...","kb_ids":[<sorted>],
 "max_results_per_query":<int>,"not_before":"<ISO-UTC>","subject_dept_id":"...","topics":[<sorted>]}
```

Rules:
- `topics` and `kb_ids` are **sorted ascending** before serialisation.
- Timestamps are converted to UTC and rendered ISO-8601.
- `parent_grant_id` is included **only when non-null** (so a root grant's bytes are unchanged
  and older verifiers stay compatible; sorted keys make insertion order irrelevant).
- `signed_token`, `revoked_at`, `created_at` are **never** part of the canonical bytes.

`signed_token = base64( Ed25519_sign( issuer_private_key, canonical_bytes ) )`.

Verification recomputes the canonical bytes and checks the signature against the issuer's
public key. It is **fail-closed**: any decode/tamper/unknown-key error returns `false`,
never an exception. Signature validity is orthogonal to activity — see §3.

## 3. Activity (freshness + revocation)

A grant is **active** iff `revoked_at is null AND not_before <= now < expires_at`. A verifier
checks activity separately from the signature. Because revocation is not in the signed bytes,
a caller could present a still-valid-looking copy of a revoked grant; therefore the
**responder** (the node being queried) enforces revocation from its own list — never trust the
presenter's `revoked_at`.

## 4. Delegation — bounded trust (A→B→C)

A subject may re-delegate a **narrowed** subset of what it holds. A delegated (child) grant
sets `parent_grant_id` to the parent's `grant_id` and MUST satisfy, else it is rejected:

- `child.issuer_dept_id == parent.subject_dept_id` (you may only re-share what was shared with you);
- `child.topics ⊆ parent.topics`;
- `child.kb_ids ⊆ parent.kb_ids`;
- `child.max_results_per_query <= parent.max_results_per_query`;
- `child.not_before >= parent.not_before` and `child.expires_at <= parent.expires_at`.

A child is usable only while **every** grant on its chain to the root is active (transitive
revocation: revoking the parent disables the child).

## 5. Query envelope + replay protection

When the token is presented on a live federated query over the mesh, the request carries the
full grant plus two freshness fields, stamped by the initiator's node:

| Field | Type | Notes |
|---|---|---|
| `grant_token` | string | Copied into `signed_token` before verification; the two MUST agree. |
| `query_text` | string | Natural-language query (query, not vector — the peer re-embeds locally). |
| `top_k` | int | Requested spans, capped to `max_results_per_query`. |
| `topics` | string[] | Subset of the grant's topics. |
| `nonce` | string | Single-use 128-bit random hex. |
| `issued_at` | string | RFC 3339 (nanosecond) UTC send time. |

The responder rejects a request whose `issued_at` is outside ±skew of its clock, or whose
`(subject_dept_id, nonce)` has been seen inside the freshness window — defence-in-depth on
top of the transport's peer authentication. The grant proves *"may this department read
this?"*; the envelope proves *"is THIS request fresh and first-seen?"*.

## 6. Conformance

An implementation conforms if, for the reference grant vectors, it (a) produces byte-identical
canonical bytes, (b) verifies reference signatures, (c) rejects any tampered field, (d)
enforces §3 activity and §4 attenuation, and (e) enforces §5 freshness + single-use. The
Python↔Go interop test is the executable conformance suite.
