# RAG as a Service Python SDK — Versioning Policy

**Status:** Normative. Applies to the `rag-sdk` Python package (`sdks/python/`) and the
`/v1` REST surface it wraps.

RAG as a Service / Federated RAG is deployed into BFSI and other regulated, sovereign
environments — banks, insurers, and public-sector buyers who integrate once and then run
that integration under change-control for years. Many of these buyers cannot re-certify or
re-deploy a client library on our schedule; a core-banking batch job written against this
SDK in 2025 must keep working, unchanged, for the life of a MAJOR line. This policy exists
so those buyers can reason about compatibility with the same rigour they apply to the rest
of their regulated stack.

## 1. Semantic Versioning

The SDK and the `/v1` API version are expressed as `MAJOR.MINOR.PATCH`, per
[SemVer 2.0.0](https://semver.org/). Given a version `X.Y.Z`:

- **MAJOR (`X`)** — incremented only for a **breaking change** to the stable public surface
  (Section 2): a removed or renamed client method, a removed/renamed parameter, a changed
  default that alters behaviour, a removed REST endpoint or field, or a removed/renamed error
  code. A MAJOR bump is the *only* release in which existing regulated integrations may be
  required to change code.
- **MINOR (`Y`)** — **additive, backward-compatible** changes: a new client method, a new
  **optional** parameter (with a safe default), a new endpoint, a new *optional* response
  field, or a new error code on a new failure path. Code written against `X.Y` keeps working
  on `X.(Y+n)`.
- **PATCH (`Z`)** — backward-compatible bug fixes and internal changes with no surface change.

The single source of truth for the installed SDK version is
`rag_sdk.__version__` (defined in `sdks/python/src/rag_sdk/__init__.py`). The packaging
version in `sdks/python/pyproject.toml` MUST match it on every release.

## 2. The stable public surface

Everything in this section is covered by the compatibility guarantee in Section 3. Anything
**not** listed here (transport internals, private attributes such as `RagClient._http`,
undocumented response fields, wire-level ordering) is **not** part of the contract and may
change in any release.

### 2.1 Client method signatures

The public methods of `rag_sdk.RagClient`, with their current signatures:

- `RagClient(base_url, *, api_key=None, token=None, timeout=30.0, transport=None)`
- `create_knowledge_base(display_name) -> KnowledgeBase`
- `list_knowledge_bases() -> list[KnowledgeBase]`
- `ingest(kb_id, *, doc_id, text=None, content_b64=None, mime="text/plain", strategy="clause") -> dict`
- `list_documents(kb_id) -> list[Document]`
- `delete_document(kb_id, doc_id) -> None`
- `query(kb_id, query, *, top_k=5, model=None) -> GroundedAnswer`

Keyword-only arguments stay keyword-only. New parameters are always added as optional
keyword arguments with a safe default. The typed response models
(`KnowledgeBase`, `Document`, `Citation`, `GroundedAnswer`) and their existing fields are
part of the surface; new fields are always additive and optional.

### 2.2 REST endpoint paths (under `/v1`)

The SDK targets these paths; they are versioned by the `/v1` prefix and are part of the
contract:

- `POST   /v1/knowledge-bases`
- `GET    /v1/knowledge-bases`
- `POST   /v1/kb/{kb_id}/ingest`
- `GET    /v1/kb/{kb_id}/documents`
- `DELETE /v1/kb/{kb_id}/documents/{doc_id}`
- `POST   /v1/kb/{kb_id}/query`

A new REST revision with breaking changes is introduced under a **new prefix** (`/v2`)
served alongside `/v1`; `/v1` is never repurposed to mean something incompatible.

### 2.3 Error codes and types

- `rag_sdk.RagError` is raised for every non-2xx response. Its public attributes
  `status_code: int`, `code: str`, and `message: str` are stable.
- The server error **codes** carried in `RagError.code` (e.g. `forbidden`,
  `http_error`) are part of the contract: existing codes are not removed or repurposed
  within a MAJOR. New codes may appear on new failure paths (a MINOR-level, additive change),
  so clients must treat an unrecognised code defensively rather than assuming a closed set.

## 3. Backward-compatibility guarantee

For legacy banking and other long-lived integrations, within a single **MAJOR** line:

1. **Additive-only.** Changes to the public surface are additive. No public client method,
   parameter, endpoint, response field, or error code is removed or given incompatible
   meaning within a MAJOR.
2. **Endpoints are never removed within a MAJOR.** Every `/v1` path in Section 2.2 remains
   served for the entire `1.x` lifetime. Retirement of a `/v1` path happens only at the
   `/v2` (next-MAJOR) boundary, and `/v1` remains available for the announced support window.
3. **Deprecation window.** Anything slated for removal is first **deprecated for at least two
   MINOR releases**, emitting warnings the whole time (Section 4), and is only removed at the
   **next MAJOR**. A caller who upgrades MINOR/PATCH releases within a MAJOR never has working
   code broken without having first seen deprecation warnings across at least two MINORs.
4. **PATCH is safe.** PATCH releases never change the surface and are always safe to take.

This lets a regulated buyer pin to a MAJOR line, absorb MINOR/PATCH updates under routine
change-control, and schedule the (rare, well-signalled) MAJOR upgrade as a planned project.

## 4. Deprecation process

1. **Announce** the deprecation in the release notes / `CHANGELOG` for the MINOR that
   introduces it, naming the deprecated element, the replacement, and the earliest MAJOR in
   which removal may occur.
2. **Warn at runtime.** Deprecated Python surfaces emit `DeprecationWarning` (with a
   stacklevel pointing at the caller). Deprecated REST behaviour is signalled by the server
   via a `Deprecation` / `Sunset` response header; the deprecated path keeps working.
3. **Sustain** the deprecated element, working, for **at least two MINOR releases**.
4. **Remove** only at the next MAJOR, documented in that MAJOR's migration guide.

Nothing is removed inside a MAJOR line; deprecation only marks a future removal.

## 5. Pinning a version (client side)

Regulated deployments SHOULD pin. Recommended constraints, from strictest to loosest:

```toml
# pyproject.toml — exact pin (maximum reproducibility; typical for change-controlled BFSI)
dependencies = ["rag-sdk==0.1.0"]

# compatible-release: take PATCH fixes within a MINOR
dependencies = ["rag-sdk~=0.1.0"]   # >=0.1.0, <0.2.0

# pin to a MAJOR line: take all additive MINOR/PATCH updates, never a breaking MAJOR
dependencies = ["rag-sdk>=1.0,<2.0"]
```

```bash
pip install "rag-sdk==0.1.0"
```

The installed version is always introspectable at runtime via `rag_sdk.__version__`, so
deployments can log and assert the exact SDK build they are running.

> Pre-1.0 note: the package is currently `0.y.z`. Under SemVer, `0.x` is a stabilisation
> phase where MINOR (`0.y`) may carry breaking changes; the additive/deprecation guarantees
> in Section 3 apply in full from `1.0.0` onward. Pin exactly (`==`) while on `0.x`.

## 6. How the SDK signals its version to the server

On construction, `RagClient` sets two request headers that ride on **every** call
(wired into the shared `httpx.Client` in `sdks/python/src/rag_sdk/client.py`, single-sourced
from `rag_sdk.__version__`):

- `User-Agent: rag-sdk/<version>` — e.g. `rag-sdk/0.1.0`.
- `X-Rag-SDK-Version: <version>` — e.g. `0.1.0`.

This lets the server (control plane) observe, log, and negotiate by client version with **no
extra round-trip and no new endpoint**:

- **Observability / audit** — every request is attributable to an SDK build, which matters
  for BFSI audit trails and incident forensics.
- **Server-side negotiation (policy).** Because the version arrives on each request, the
  server MAY, in future, warn on or refuse client builds below a supported floor by returning
  a standard error (e.g. a `426`-style upgrade-required response carrying an `RagError`
  code) and/or a `Deprecation`/`Sunset` header. The SDK already surfaces any non-2xx response
  as `RagError`, so such negotiation needs no client change. This document defines the
  policy; the SDK's obligation is only to **advertise** its version truthfully on every
  request, which it does today.
