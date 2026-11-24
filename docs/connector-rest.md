# REST-API connector — unified KB (documents + connected API)

A knowledge base can carry documents plus connected structured sources. This connector adds a
REST API as one such source, alongside the existing SQL connector (`sql_query.py`): each is an
**evidence producer** — given a KB id and a question, it returns ONE authoritative
`RetrievedSpan` (or `None`) that the pipeline fuses with document spans into a single, multiply
cited answer. Implementation: `services/control-plane/src/svc_control_plane/rest_source.py`.

## Registering a REST source

```python
from svc_control_plane import rest_source

info = await rest_source.set_rest_connection(
    kb_id="acme__support",
    base_url="https://api.acme.internal",
    path="/v1/search",         # default: "/search"
    query_param="query",       # default: "q"
    token="...",                # optional bearer token; omit to leave an existing one in place
)
```

Config is stored per-KB in `tenant_config` (`rest_base_url`, `rest_path`, `rest_query_param`);
a token, if given, is envelope-encrypted via `tenant_crypto.encrypt` into `rest_token` — never
stored or returned in plaintext.

```python
info = await rest_source.get_rest_connection(kb_id)   # RestConnectionInfo — has_token bool only, never the token
await rest_source.clear_rest_connection(kb_id)         # disconnect
```

## Governance

- **GET-only.** The connector issues a single `HTTP GET`; it has no path for writes.
- **Sovereign.** `assert_sovereign(base_url)` gates the host against the egress allowlist
  twice: once at `set_rest_connection` (so a non-sovereign endpoint is rejected before it's ever
  saved) and again at query time in `rest_evidence_span` (so a later allowlist change is
  honoured immediately, not just at the next registration).
- **Bounded.** One request per question, a hard timeout, and a streamed byte cap
  (`_MAX_RESPONSE_BYTES`, 256 KiB) — an oversized or slow upstream is refused rather than
  buffered or allowed to hang.
- **Fail-safe evidence span.** `rest_evidence_span(kb_id, question, *, actor)` returns `None` —
  never raises — when the KB has no REST connection, the host is blocked, the request
  times out or errors, the response isn't 200, or the payload is empty. It issues **no audit
  call**: like the SQL producer, it's a read-only fan-out that must never be able to deny or
  destabilize the rest of the answer.

## Unified-answer fusion

`rest_evidence_span` follows the same producer shape as `sql_query.sql_evidence_span`:
`(kb_id, question, *, actor) -> RetrievedSpan | None`. Once registered with
`unified.register_producer`, it's called alongside every other structured source by
`unified.gather_evidence`, and its span — `scorer="exact"`, `score=1.0`,
`modality=Modality.API`, `doc_id="api://<host>"` — is unioned with document spans so a single
grounded, cited answer can draw on documents and the connected API together.
