# Automated (retention-driven) erasure

`services/control-plane/src/svc_control_plane/erasure_schedule.py` adds the RETENTION-DRIVEN
half of DPDP erasure. `compliance.py` already covers two on-demand paths: `erase(...)`
(explicit request, e.g. a data principal's right-to-erasure) and `apply_retention(...)` (a
manual, on-demand sweep against a KB's stored retention config). What was missing was a
scheduler: something that periodically decides which documents are now past their retention
window and triggers erasure automatically, without a human calling either function.

## Design: pure core, injected erasure

The module is deliberately split into a decision half and an action half:

* **Decision (pure)** — `due_document_ids(*, now, retention_days, docs)` takes a snapshot of
  documents (`RetainedDoc`: `doc_id` + `created_at`) and returns the sorted ids whose age
  strictly exceeds `retention_days`. No I/O, no imports beyond the standard library and
  pydantic — a plain function you can unit test with hand-built inputs in microseconds.
* **Action (injected)** — `run_scheduled_erasure(*, policy, now, docs, erase_fn)` uses
  `due_document_ids` to compute what's due, then — only if the policy is enabled and something
  is due — calls `erase_fn(kb_id, doc_ids) -> ErasureCertificate`. Nothing in this module talks
  to the catalog, a database, or the network; `erase_fn` is the only seam to the outside world.

This mirrors the on-demand path's semantics (`compliance.apply_retention` treats "age >
retention_days", i.e. exactly-at-the-boundary is not yet due) so scheduled and on-demand
retention agree on the edge case, while staying free of `pipeline`/`wiring`/catalog imports.
The only import from `compliance` is `ErasureCertificate`, used purely as a type — the module
never calls into `compliance` itself.

## Wiring (owned by the parent, not this module)

This module does not start a loop, read config, or call `compliance.erase`. The parent
(background scheduler + wiring layer) is expected to:

1. Load each KB's `RetentionPolicy` (kb_id, retention_days, enabled) from wherever retention
   config lives (today: `compliance.get_retention`/`set_retention`).
2. List that KB's documents and project them into `RetainedDoc` (doc_id, created_at).
3. Supply an `erase_fn` closure that calls `compliance.erase(kb_id, actor="scheduler",
   doc_ids=doc_ids, reason="retention_policy", now=now)` — same audited, certificate-issuing,
   cascade-deleting path as every other erasure.
4. Call `run_scheduled_erasure(...)` on a timer (per KB, e.g. daily) and persist/emit the
   returned `ErasureCertificate` when one comes back.

Because `erase_fn` is just a callback, the parent can also route it through retries, rate
limiting, or a job queue without this module knowing anything changed.

## How this complements on-demand erasure

| | on-demand (`compliance.erase` / `apply_retention`) | scheduled (`erasure_schedule`) |
|---|---|---|
| trigger | explicit call (API request or manual sweep) | timer, per enrolled KB |
| decision logic | inline in `compliance.py`, reads live catalog + config | pure, injected snapshot (`due_document_ids`) |
| erasure mechanics | `compliance.erase` (cascade + certificate + audit chain) | same, via injected `erase_fn` |
| DPDP obligation | right-to-erasure (data principal request) | retention limits enforced without waiting for a request |

Both paths end at the same signed `ErasureCertificate` on the tamper-evident audit chain — the
scheduler just adds a way to reach that certificate automatically once a document's retention
window has passed.

## What erasure reaches, and what it deliberately retains

Both paths above run the same cascade, so this applies to either. Erasure removes a document from
every store *and* from every **derived record that quoted it** — a certificate that claimed
otherwise would be false, because the erased text would still be readable in the derivative.

| Artifact | On erasure | Mechanism |
|---|---|---|
| catalog rows (`documents`, `chunks`) | purged | `pipeline.delete_document` |
| RAG v2 structure rows (`sections`, `blocks`, `facets`, `doc_relations`, `map_cache`) | purged | `PostgresCatalog.delete_document` (per-document, no FK cascade) |
| dense + sparse indexes, raw object bytes, semantic cache | purged | `pipeline.delete_document` |
| conversation history quoting the document | purged **by citation** | `ConversationStore.delete_citing` |
| `query_log` rows quoting the document | purged **by citation** | `PostgresCatalog.delete_query_log_citing` (array overlap on `cited_doc_ids`) |
| the `query_log` row's own `query` text | **retained** — see below | — |

**Why `query_log` is purged by citation rather than by not storing the text.** `query_log.answer`
will hold passages lifted from the corpus and `selected_sections` names the structure they came
from, which makes a log row the same class of artifact as a chat message: derived content quoting a
document. So it gets the same treatment the platform already applies to derived chat history, and
for the same stated reason. Each row carries `cited_doc_ids`, and erasure deletes every row whose
citations **overlap** the erased set — a row is removed if it quoted *any* erased document, not only
if all of its citations were erased, because a partially-erased answer still quotes the erased
document. A row with no citations is never matched, so it is never collaterally purged.

The rejected alternative was to keep the answer text out of the schema altogether. That would have
destroyed the observability the query log exists to provide (per-stage latency and router decisions
are only interpretable next to the answer they produced), while leaving the identical question open
for `selected_sections`, and it would have made `query_log` the only derived artifact on the
platform that answers erasure by not storing data rather than by purging it.

**Why the row's own `query` text is retained, as a reasoned position.** The `query` is the data
principal's *question*. It is not content derived from the erased document, so a document-scoped
erasure has no claim on it — erasing it would be deleting a different person's record of a different
act. The platform already takes exactly this position one layer down: the tamper-evident audit chain
stores query text in its `QUERY` payloads and is **append-only by design**, because a hash chain
that permits selective deletion is not evidence of anything. Erasing a data principal's own query
history is therefore a **separate, subject-scoped question that already has an answer on this
platform** (the audit chain's append-only guarantee, and the subject-match path in
`compliance.erase` for the documents *about* a subject) — not a gap in this table. If a future
obligation requires subject-scoped query-history erasure, it has to be decided for the audit chain
and the query log together, because answering it for only one of them would be incoherent.

**Scope note.** `delete_query_log_citing` runs from `compliance.erase` only, never from a plain
`pipeline.delete_document`. That mirrors conversations exactly: deleting a document purges the
stores and the document's own derived rows, while an *erasure* additionally scrubs derived history
that merely quoted it.
