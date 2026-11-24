# ADR-0025 — the KB `domain_profile` and the per-document profile card stay two concepts

**Status:** accepted · 2026-10-21
**Decides** discovery §9.8, the open question raised while scoping RAG v2 Phase 3 ("Reconcile
`domain_profile` with the Phase 3 doc profile — one profile concept or two?").
**Recorded as** discovery §8.13.

## Context

Two things in this codebase are called a profile, and the discovery pass flagged the collision
rather than letting a second profile notion appear by accident.

**`domain_profile`** (Slice 6-A, `svc_control_plane.domain_profile` over
`rag_core.domain_profile`) samples a KB's already-indexed chunks, classifies the CORPUS
against a fixed vocabulary of domains, and stores three JSON values in `tenant_config`:
the active profile, a pending knob SUGGESTION, and an audit of which knobs an operator confirmed.
Its governing rule is **detect-and-suggest, never auto-apply** — `detect` moves no retrieval knob,
and `apply_suggestion` refuses the reindex-class knobs outright unless an operator opts in, so the
profile can never silently trigger a reindex or an index/config drift.

**The Phase-3 profile card** is a model's reading of ONE document: its `doc_type`, purpose,
issuer, issue date, the relationships it declares about other documents, and a facet taxonomy
drawn from its own content. It is stored in that document's own `documents` columns plus its
`facets` and `doc_relations` rows, and it drives per-doc_type chunk sizing, the contextualize
prefix in front of every v2 chunk, the declared-relation graph edges, and (from Phase 4) the facet
labels sections are grouped by.

## Decision

They remain **two concepts with two scopes**, documented as neighbours. Neither reads the other,
and neither is renamed or folded into the other.

## Why not one

The obvious merge is "a KB profile is just the aggregate of its documents' cards". It fails in
both directions, and each direction breaks something that currently works.

**Collapsing the card into `domain_profile` loses the only distinction the card exists to make.**
A corrigendum and the tender it amends sit in the same KB and would therefore share a corpus
classification — but the whole point of Phase 3 is that one of them AMENDS the other, which is a
statement about one document and has no corpus-level form. A corpus is also classified from a
sample (60 chunks, capped per document), so a single short corrigendum in a large KB contributes
almost nothing to it; the document whose supersession rules the answer would be the document the
profile least describes.

**Collapsing `domain_profile` into the card is worse: it would make a model's per-document reading
move a KB's retrieval knobs.** That is precisely the auto-apply the domain-profile module was
built to refuse, and the knobs it refuses hardest are the reindex-class ones. A per-document
producer feeding them would mean one uploaded file could change how every other document in the
KB is retrieved — with no operator in the loop, and no single place where the decision was made.

**The failure modes are also different in kind, and so are the right responses.** A wrong
`domain_profile` is a suggestion an operator declines. A wrong profile card is stored data on one
document, fixed by re-profiling that document — which is exactly what a reindex does.

## Consequences

- `tenant_config`'s three domain-profile keys and `/profile/detect` are untouched by Phase 3.
- `documents.profile_json` / `profile_version` are the card's; `profile_version` is the CONTRACT
  version (prompt + parser + schema), so `WHERE profile_version < N` selects the corpus to
  re-profile after a prompt change. It is not a per-document re-profiling counter.
- A future feature that genuinely wants both — "this KB is mostly tenders, and this document is a
  corrigendum to one of them" — reads the two sources and joins them at the call site. It does not
  merge the stores.
- Nothing here constrains Phase 7. A router may consult `doc_type` and the KB's domain together;
  consulting two things is not the same as storing them as one.
