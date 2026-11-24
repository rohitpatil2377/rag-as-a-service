# SharePoint connector — unified KB (documents + connected SharePoint site)

A knowledge base can carry documents plus connected structured sources. This connector adds a
SharePoint site as one such source, alongside the existing SQL and REST connectors
(`sql_query.py`, `rest_source.py`): each is an **evidence producer** — given a KB id and a
question, it returns ONE authoritative `RetrievedSpan` (or `None`) that the pipeline fuses with
document spans into a single, multiply cited answer. Implementation:
`services/control-plane/src/svc_control_plane/sharepoint_source.py`.

## Registering a SharePoint site

```python
from svc_control_plane import sharepoint_source

info = await sharepoint_source.set_sharepoint_connection(
    kb_id="acme__eng",
    site_url="https://acme.sharepoint.com/sites/eng",
    token="...",   # optional bearer token; omit to leave an existing one in place
)
```

Config is stored per-KB in `tenant_config`: the site URL in the clear as `sharepoint_site` (not a
secret), and a token, if given, envelope-encrypted via `tenant_crypto.encrypt` into
`sharepoint_token` — never stored or returned in plaintext.

```python
info = await sharepoint_source.get_sharepoint_connection(kb_id)  # SharePointConnectionInfo — has_token bool only, never the token
await sharepoint_source.clear_sharepoint_connection(kb_id)        # disconnect
```

## Governance

- **Read-only, Search-API only.** The connector issues a single `HTTP GET` against SharePoint's
  Search REST API (`{site_url}/_api/search/query?querytext=...`) — it has no path for writes and
  never touches any other SharePoint endpoint.
- **Sovereign.** `assert_sovereign(site_url)` gates the site's host against the egress allowlist
  twice: once at `set_sharepoint_connection` (so a non-sovereign site is rejected before it's ever
  saved) and again at query time in `sharepoint_evidence_span` (so a later allowlist change is
  honoured immediately, not just at the next registration).
- **Bounded.** One request per question, a hard timeout (`_TIMEOUT_SECONDS`, 10s), and the
  search itself is capped at the query level (`rowlimit=10`) as well as at render time
  (`render_sharepoint_evidence(..., max_results=10)`) — an oversized or slow tenant never blows
  the prompt or hangs the answer.
- **Fail-safe evidence span.** `sharepoint_evidence_span(kb_id, question, *, actor)` returns
  `None` — never raises — when the KB has no SharePoint connection, the site is blocked by the
  sovereignty guard, the request errors or is non-200, the response JSON doesn't parse or has an
  unrecognised shape, or there are no hits. It issues **no audit call**: like the SQL and REST
  producers, it's a read-only fan-out that must never be able to deny or destabilize the rest of
  the answer.
- **Injectable transport.** The outbound `httpx.AsyncClient` is created through a module-level
  seam (`_new_http_client`), so tests substitute an `httpx.MockTransport` and no test ever makes
  a real network call.

## Unified-answer fusion

`sharepoint_evidence_span` follows the same producer shape as `sql_query.sql_evidence_span` and
`rest_source.rest_evidence_span`: `(kb_id, question, *, actor) -> RetrievedSpan | None`. Once
registered with `unified.register_producer`, it's called alongside every other structured source
by `unified.gather_evidence`, and its span — `scorer="exact"`, `score=1.0`,
`modality=Modality.SHAREPOINT`, `doc_id="sharepoint://<host>"` — is unioned with document spans
so a single grounded, cited answer can draw on documents and the connected SharePoint site
together.
