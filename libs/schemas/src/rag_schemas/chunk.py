"""Chunk — the indexed unit of retrieval (TDD §6 "Chunk record").

Carries everything needed for citation (doc/page/span), permission checks (acl),
and audit lineage (lineage_id, version). ``embedding_ref`` points to the vector in
the VectorStore rather than inlining the vector, keeping the metadata record light.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from rag_schemas.common import ChunkLevel, Modality


class Span(BaseModel):
    """Character offsets of the chunk within the extracted document text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)


class Chunk(BaseModel):
    """A retrievable, citable chunk of a document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    doc_id: str
    tenant_id: str
    text: str = Field(..., description="Chunk text (PII already masked, FR-G2).")
    modality: Modality = Modality.TEXT

    page: int | None = Field(default=None, ge=1, description="Source page for citation (FR-D7).")
    span: Span | None = Field(default=None, description="Char span within the doc text.")
    source_uri: str | None = Field(default=None, description="Pointer to source for click-through.")
    acl: list[str] = Field(default_factory=list, description="Inherited ACL principals (FR-D8).")
    classification: int = Field(
        default=0, ge=0, le=3, description="Inherited data classification 0=Public..3=Restricted (banking pack)."
    )

    ocr_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Min per-field OCR confidence (FR-B1)."
    )
    embedding_ref: str | None = Field(default=None, description="Vector id in the VectorStore.")
    lineage_id: str | None = Field(default=None, description="raw->chunk->answer lineage (FR-G7).")
    version: int = Field(default=1, ge=1, description="Index version for replay (FR-C8).")
    metadata: dict[str, str] = Field(default_factory=dict)

    # --- RAG v2 structure (upgrade plan Phase 5 "Chunking and Contextualize") -------------------
    # ADDITIVE on a frozen, extra="forbid" contract, so every field below DEFAULTS TO None and
    # None is exactly what a v1 chunk is. Nothing that produces chunks today sets any of them and
    # nothing that reads chunks today looks at them, so the default path stays byte-identical —
    # which is asserted by a test, not claimed. Only `LayoutAwareChunker` populates them.
    #
    # They are fields rather than `metadata` keys on purpose. `metadata` is `dict[str, str]` and
    # flows wholesale into the Qdrant payload, so structure put there is untyped, stringified and
    # indistinguishable from a tenant's own document tags — and the Phase-1 catalog columns
    # (`chunks.level` / `.section_id` / `.parent_id` / `.embed_text`) are first-class columns that
    # a metadata blob cannot be joined on.
    level: ChunkLevel | None = Field(
        default=None,
        description="RAG v2 retrieval grain (section|chunk|row). None = a v1 chunk, which has no level.",
    )
    section_id: str | None = Field(
        default=None,
        description="The `sections` row this unit sits in (RAG v2) — joins the chunk to its breadcrumb/tree.",
    )
    # NOT the same thing as `metadata["parent_id"]`, which is FR-C2 parent/child's link from a small
    # child to its catalog-only context bank (`rag_core.parent_child.META_PARENT_ID`). This one
    # is the STRUCTURAL parent in the Phase-2 tree: a chunk's/section's own section, a row's table
    # block. The two never coexist on one chunk — a v2 chunk sets this field and no metadata key,
    # a parent/child child sets the metadata key and not this field — but they are kept separate so
    # neither consumer has to ask which meaning a value carries.
    parent_id: str | None = Field(
        default=None,
        description="Structural parent (RAG v2): the section for `chunk`/`section`, the table block for `row`.",
    )
    # The plan's contextualized text. It is NOT a second copy of `text`: `text` is what the catalog
    # stores and what a citation shows (it must keep saying what the document said), and this is
    # what the embedder and BM25 see. The indexing service already separates those two — it embeds a
    # `model_copy` carrying the contextualized text while cataloging the original — so this field
    # supplies the CONTENT for that split and supersedes neither the copy nor the `Contextualizer`
    # seam; see `svc_indexing.service.index_document`.
    embed_text: str | None = Field(
        default=None,
        description="Text to embed/index instead of `text`. None = embed `text` (today's path).",
    )
    # DOCUMENT ORDER for a chunk that has no `span`. A v1 chunk is ordered by `span.start` — its
    # offset into the extracted text — but markup-derived text has no offset into that text, so
    # every v2 chunk has `span=None` and the ordering key collapses. `CatalogAdapter`
    # implementations order on this next, so "read this document in order" stays a real guarantee
    # for the citation viewer (FR-I3), whole-document answering, and neighbour expansion, which
    # computes adjacency from the returned list and would otherwise expand to an unrelated chunk.
    #
    # Assigned by the chunker from its own emission order, so it is DOCUMENT order and not merely a
    # stable one. Not an input to `chunk_id` — re-chunking identical input must still derive
    # identical ids — and None on every v1 chunk, which keeps `span` the key there.
    sequence: int | None = Field(
        default=None,
        ge=0,
        description="Position in document order; the ordering key for chunks with no span.",
    )
    # The embedder's OWN count of the tokens in `text`, asked of the running server at index time
    # (`Embedder.count_tokens`, optional and duck-typed like `token_budget`/`token_offsets`). None
    # means the embedder could not say — NOT zero, and never an estimate. A characters/4 guess is
    # not close on the inputs that matter: measured against this deployment's bge-small, 1 000
    # characters of ToC dot leaders tokenize to 1 000 tokens — twice its 510-token budget, where
    # characters/4 would have recorded 250 (Devanagari runs ~1.6 chars/token, prose ~5.1, so the
    # error has no safe direction), and nothing downstream could tell a guess from a measurement.
    #
    # It counts `text` — the stored, citable body — and not the contextualized string that is
    # actually embedded (`embed_text` plus any ADR-0007 document context). That string is
    # per-KB-config and, with an LLM contextualizer, not even reproducible, so a count of it would
    # describe nothing the row holds. What it therefore answers is "how big is this chunk, in the
    # units the embedder enforces", which is the question `token_budget` is compared against.
    #
    # Set only where the other RAG v2 fields are set (a chunk with a `level`): it is a plan §1.1 v2
    # column, and writing a new key onto a v1 chunk's serialised record would make every record
    # this build writes unreadable by the previous image, whose `Chunk` is `extra="forbid"`.
    token_count: int | None = Field(
        default=None,
        ge=0,
        description="Tokens in `text` per the embedder's own tokenizer. None = the embedder could not say.",
    )
