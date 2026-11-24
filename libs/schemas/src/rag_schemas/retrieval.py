"""Retrieval contracts — query in, ranked scored spans out (TDD §4.4).

The retrieval output contract (``RetrievedSpan``) carries ``{doc_id, span, page,
score, source_uri, acl}`` so the generation layer can build citations and enforce
permissions without re-fetching (FR-D7).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_schemas.chunk import Span
from rag_schemas.common import ChunkLevel, Modality, RetrievalMode

# A chunk's own RECORD fields. `filters` is the FR-D8 METADATA channel and these are not metadata,
# so naming one as a filter key is always a mistake — and a mistake that USED TO BE SILENT: the
# sparse leg resolves every filter key to `meta.<key>` (`opensearch_index.meta_path`), so
# `filters={"doc_id": X}` matched nothing there while the dense leg's flat payload matched the
# record, leaving document-scoped retrieval dense-only with no BM25 and no error. Rejected rather
# than translated, because a key here is genuinely ambiguous — on the sparse leg it would mean a
# TAG called `doc_id`, which is a different thing and is deliberately tested as such. Use the
# dedicated predicate instead: `doc_ids` for a document scope.
# `level` joined them in Phase 8 for exactly the reason `doc_id` is here: both stores index it
# as a record field, so `filters={"level": ...}` matched the flat dense payload and nothing at
# all on the sparse leg (`meta.level`) — the same silent half-working predicate. Use `levels`.
RESERVED_FILTER_KEYS = frozenset({"chunk_id", "doc_id", "tenant_id", "page", "acl", "level"})


def reject_reserved_filter_keys(filters: dict[str, str]) -> dict[str, str]:
    """Fail-closed on a record field used as a metadata filter key. Shared by the API request
    model (so an external caller gets a clean 422 before the handler runs) and by
    :class:`RetrievalQuery` itself (so internal callers fail loudly at construction)."""
    offending = sorted(RESERVED_FILTER_KEYS & filters.keys())
    if offending:
        hint = " Use `doc_ids` for a document scope." if "doc_id" in offending else ""
        hint += " Use `levels` for a retrieval grain." if "level" in offending else ""
        raise ValueError(
            f"{', '.join(offending)}: a chunk record field cannot be used as a metadata filter key "
            f"(`filters` is the FR-D8 metadata/tag channel).{hint}"
        )
    return filters


class RetrievalQuery(BaseModel):
    """A retrieval request, always tenant-scoped."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    text: str = Field(..., min_length=1)
    # The user's ORIGINAL question before any pre-retrieval rewrite (glossary/synonym expansion).
    # Carried for display/audit so the user-facing answer and the QUERY audit show what was asked,
    # while retrieval runs on the expanded `text`. None ⇒ `text` is the original (no rewrite).
    original_text: str | None = None
    top_k: int = Field(default=5, ge=1, le=100)
    mode: RetrievalMode = RetrievalMode.HYBRID
    filters: dict[str, str] = Field(default_factory=dict, description="Metadata filters (FR-D8).")
    acl: list[str] = Field(
        default_factory=list,
        description="Caller's ACL group memberships (FR-A6). A span is visible if its own acl is "
        "empty (tenant-public) or intersects this set; empty here = only tenant-public spans.",
    )
    clearance: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Caller's classification clearance 0=Public..3=Restricted (banking pack). A span is "
        "visible only if its classification is at or below this; default 0 = only Public spans.",
    )
    # Recency-aware retrieval (FR-D9) — optional, per-query. Filter to a document time window
    # and/or boost fresher documents in ranking. All default-off (behaviour unchanged).
    after: datetime | None = Field(default=None, description="Only documents ingested at/after this instant.")
    before: datetime | None = Field(default=None, description="Only documents ingested at/before this instant.")
    freshness: float = Field(default=0.0, ge=0.0, le=1.0, description="Freshness-boost weight (0=off, 1=strong).")
    # Exact technical-token BM25 boost (Slice 2a) — per-KB adaptive. When set, the sparse leg
    # weights whole-token (case-preserving) matches on chemical formulas/SMILES/identifiers by this
    # amount. None ⇒ the sparse adapter falls back to the global `exact_term_boost` setting. The
    # exact path itself is gated by `exact_term_match_enabled`; when off this is ignored.
    exact_boost: float | None = Field(default=None, ge=0.0)
    # Canonical chemical terms (Slice 6, Phase C) — server-resolved from the query's SMILES via RDKit
    # for a chemistry KB. Carried on a DEDICATED field (NOT appended to `text`) so ONLY the sparse leg
    # sees them: an InChIKey hash is pure noise to the dense embedder, the bge-sparse/ColBERT legs and
    # the cross-encoder reranker, which all consume `text`. The sparse adapter adds a boosted match on
    # the `chem` field from these terms, gated on this being non-empty. Default empty ⇒ byte-identical.
    chem_terms: tuple[str, ...] = ()
    # Multi-vector retrieval (Slice 2b) — per-KB, server-resolved from tenant_config. These say
    # which OPTIONAL bge-m3 legs fuse in when the KB's embedder emits dense+sparse+ColBERT: the
    # learned-sparse leg (SEPARATE from BM25, never replacing it) and the ColBERT MaxSim re-scoring
    # of the dense candidate pool. ColBERT NEVER writes a span's `score` — the cross-encoder
    # reranker remains the sole score writer and the abstain gate is unchanged. All default off ⇒
    # retrieval is byte-identical. `colbert_prefetch` bounds the MaxSim pool (0 ⇒ adapter default).
    multivector_sparse: bool = False
    multivector_colbert: bool = False
    colbert_prefetch: int = Field(default=0, ge=0)
    # Document scope — "answer from THESE documents only" (an OR over the set). A DEDICATED field,
    # NOT a `filters` entry, because the two are different predicates on different data. `filters`
    # is the FR-D8 METADATA channel: the sparse leg namespaces every key under `meta.<key>` (see
    # `opensearch_index.meta_path`), so a filter key names a document/auto TAG — deliberately, and
    # tested (a `doc_id` TAG describes a chunk, it does not redefine it). `doc_ids` is a predicate on
    # the chunk RECORD's own `doc_id`. Routing it through the tag channel worked on the dense leg by
    # accident (its payload is flat) and matched NOTHING on the sparse leg, so document-scoped
    # compare/summary ran dense-only with no BM25 and never said so. Applied IN ADDITION to
    # `filters`, ACL and clearance — never instead of them. Empty ⇒ no document scoping.
    doc_ids: tuple[str, ...] = ()
    # Retrieval GRAIN scope (RAG v2 Phase 8's `hybrid(q, filters, levels, n)` primitive). A
    # predicate on the chunk RECORD's own `level`, exactly like `doc_ids` and for the same reason
    # — both stores already index `level` as an exact-match field (Phase 6 put it there for the
    # per-level reconciliation and said in both adapters that a level-scoped query leg would
    # follow), and routing it through the `filters` tag channel would namespace it under
    # `meta.level` on the sparse leg and match nothing.
    #
    # EMPTY IS NO FILTER, not "every level". A v1 chunk has no level at all, so a strategy that
    # asked for the three v2 levels explicitly would make every pre-v2 document invisible —
    # which is every document in every deployment today. Only a strategy that knows it wants a
    # grain sets this; everything else leaves it empty and sees exactly what it saw before.
    levels: tuple[ChunkLevel, ...] = ()

    @field_validator("filters")
    @classmethod
    def _no_record_fields_as_filters(cls, v: dict[str, str]) -> dict[str, str]:
        return reject_reserved_filter_keys(v)


class RetrievedSpan(BaseModel):
    """One ranked, scored candidate span (the retrieval output contract)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    doc_id: str
    text: str
    # TWO-STAGE, two numbers, and they never share a field. Rank fusion answers "which candidates
    # are worth looking at" and is set-relative (min-max normalised, so the best candidate of ANY
    # result set is 1.0). The cross-encoder answers "how relevant is this passage to this
    # question" and is absolute. Averaging them would produce a number that is neither.
    #
    # ONLY a reranker may write `score`, and it writes `scorer` in the same breath. A span that
    # never reached one keeps score=0.0/unscored and abstains at the gate — so skipping the
    # cross-encoder fails safe and stays queryable, instead of passing a fusion value off as
    # confidence. Do not "helpfully" seed `score` with the fusion value on the way in.
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Relevance in [0,1] from the reranker — ABSOLUTE, comparable across queries. "
        "The abstain gate and Citation.score read this. 0.0 with scorer='unscored' means nothing "
        "has judged this span, NOT that it was judged irrelevant.",
    )
    scorer: Literal["cross_encoder", "lexical", "unscored", "sql", "exact"] = Field(
        default="unscored",
        description="Which scorer produced `score`. A threshold is only meaningful against a "
        "named scorer: 'lexical' is an overlap fraction on a different scale from a "
        "cross-encoder's, and 'unscored' means no judgement was made at all.",
    )
    fusion_score: float = Field(
        default=0.0,
        description="The RRF score this span carried out of fusion, min-max normalised within "
        "its own result set. Diagnostic only: NOT comparable across queries, never a gate input.",
    )
    page: int | None = None
    # The heading this span sits under, when the document numbers its sections. Carried
    # separately from the text because the GENERATOR needs it: a rule under "6.2.3 Social
    # Media Content" does not apply to a sales deck, and nothing in the span itself says so.
    section: str | None = None
    span: Span | None = None
    source_uri: str | None = None
    # The chunk's primary taxonomy category (FR-B7), carried through so federation
    # topic-scoping can filter a peer's spans to the granted topics (ADR-0017 L1) without
    # a second catalog lookup. None when the KB has no taxonomy.
    # The date the SOURCE DOCUMENT took effect (YYYY-MM-DD), read from its own release block --
    # never guessed from body text. None means the document does not state one, which must be
    # read as UNKNOWN and never as old.
    doc_date: str | None = None
    # Set when a NEWER document was also retrieved for this question. The answer must lead with
    # the current version and label this one, rather than silently choosing between them or
    # presenting two contradictory lists as equals.
    superseded_by: str | None = None
    category: str | None = None
    acl: list[str] = Field(default_factory=list)
    classification: int = Field(default=0, ge=0, le=3)
    # Source modality (carried through to the citation so the UI can flag a visual source).
    modality: Modality = Modality.TEXT
    # Parent/child retrieval (FR-C2, Option B) — query-time transport ONLY, never stored. A child
    # span carries its parent's id (stamped at ingest, read from chunk metadata by _to_span) so the
    # query-time _parent_expand step can fetch the parent context; parent_text is that fetched
    # NON-CITABLE background text, rendered in the prompt's CONTEXT block above the numbered SOURCES.
    # Both None on legacy/flag-off spans, so the whole feature no-ops and citations stay child-only.
    parent_id: str | None = None
    parent_text: str | None = None

    @model_validator(mode="after")
    def _score_names_its_scorer(self) -> RetrievedSpan:
        """A score with no scorer is a number on no known scale, and the abstain gate has to
        pick a threshold per scale. Refusing it at construction makes I11 structural: it cannot
        be forgotten at a new construction site, only stated."""
        if self.score > 0.0 and self.scorer == "unscored":
            raise ValueError("score was set without a scorer; only a reranker writes score, and it names itself")
        return self


class RetrievalResult(BaseModel):
    """Ranked spans plus the confidence signal that feeds the abstain gate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: RetrievalQuery
    spans: list[RetrievedSpan]
    max_score: float = Field(default=0.0, description="Top span score; input to grounding gate (FR-E2).")
    mode_used: RetrievalMode = RetrievalMode.HYBRID
