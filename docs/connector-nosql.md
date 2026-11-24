# NoSQL (MongoDB) connector

A knowledge base (KB) can carry documents *plus* connected structured sources. This connector
adds MongoDB as a second such source, alongside the existing SQL connector — same shape, same
governance, same fusion behaviour, just a different backend.

Module: `svc_control_plane.nosql_query`.

## Registering a connection

```python
from svc_control_plane.nosql_query import set_nosql_connection

await set_nosql_connection(
    kb_id="bank__deals",
    uri="mongodb://mongo.internal:27017",
    database="salesdb",
    collections=["deals", "accounts"],
)
```

- `uri` must resolve to a sovereignty-allowlisted host (every host, for a replica-set URI with
  more than one) — registration raises `SovereigntyViolationError` otherwise.
- At least one collection is required; registering with an empty list raises `ValidationError`.
- The URI is envelope-encrypted per-KB (via `tenant_crypto`, the same mechanism the SQL
  connector's DSN uses) before it is stored — it is never written or returned in plaintext.

`get_nosql_connection(kb_id)` returns a non-secret view (`configured`, `database`,
`collections`, `host`) for the console — never the URI or credentials.

`clear_nosql_connection(kb_id)` disconnects: it removes the stored URI, database, and
collection allowlist.

## Governance

- **Read-only.** The connector only ever calls `find`; nothing in this module writes to Mongo.
- **Allowlisted collections only.** Every query runs across exactly the collections registered
  with `set_nosql_connection` — never an arbitrary collection named at query time.
- **Bounded.** Results are capped at a hard limit (20 documents, unioned across all allowlisted
  collections) so a broad match can never return an unbounded result set.
- **Sovereign.** The Mongo host(s) in the URI are checked against the egress allowlist via
  `rag_common.sovereignty.assert_sovereign` before the connection is ever registered.
- **No NL→query translation.** Unlike the SQL connector's NL→SQL path, there is no model step
  that turns the question into Mongo query syntax. The question is only ever used as plain
  keyword terms in a `$text` search — the question text can never become operator syntax, which
  closes off NoSQL query-injection as an attack surface entirely.
- **Fail-safe.** `nosql_evidence_span` catches every exception (driver errors, network errors,
  connection errors, decrypt errors) and returns `None` instead of raising. A misbehaving or
  unreachable Mongo instance can degrade a unified answer to documents-only, but it can never
  fail the query.
- **No audit call in this module.** Audit logging for structured-source access is wired by the
  parent (the way `sql_query.py`'s callers wire `SQL_QUERY` audit events); this connector stays
  focused on the read path.

## How it fuses into a unified answer

`nosql_evidence_span(kb_id, question, *, actor)` is an *evidence producer*: given a KB id and a
question, it returns at most one `RetrievedSpan` — the same contract every unified-KB source
implements (see `svc_control_plane.sql_query.sql_evidence_span` for the SQL twin). It returns
`None` when the KB has no NoSQL connection, the query errors, or nothing matches.

When it does find something, the matched documents are rendered into one compact, readable text
block and returned as a single span:

- `chunk_id="nosql::{kb_id}"`, `doc_id="nosql://{kb_id}"`, `source_uri="nosql://{kb_id}"`
- `score=1.0`, `scorer="exact"` — an authoritative hit, not a similarity-ranked guess
- `modality=Modality.DATABASE`, `category="database"` — so the citation is flagged as a
  database source in the UI, the same way a SQL hit is

The unified pipeline (`svc_control_plane.unified`) collects this span alongside whatever the SQL
producer and the document retriever return, so one question answered partly by documents and
partly by MongoDB still yields a single grounded answer with per-source citations.

## Driver

The Mongo driver (`motor`) is imported lazily, inside the query path, so the rest of the control
plane — including this module's own import — never requires it to be installed. Production
wiring uses the default `_MotorClient`; the `_client_factory` seam lets callers (and tests) swap
in any object implementing the small `NoSqlClient.find(database, collection, terms, limit)`
protocol instead.
