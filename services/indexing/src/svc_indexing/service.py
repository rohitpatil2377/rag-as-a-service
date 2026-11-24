"""Indexing service (TDD §4.3 · FR-C1/C2/C3/C4/C5/C8).

Chunk -> embed -> write to the vector store (dense), BM25 index (sparse), and the
metadata catalog (chunk records + lineage). All three writes are tenant-scoped via
the explicit ``tenant_id``. Collections/indices/schemas are ensured on first use.

Contextual Retrieval (FR-C2, ADR-0007): when ``contextual_retrieval_enabled`` is on, a
document-level context is prepended to each chunk BEFORE embedding + BM25 indexing. The
contextualized text is embedded/indexed ONLY; the catalog always stores the ORIGINAL
chunk, so citations/answers never show the context prefix. Additive + reversible.

Versioning (FR-C8): chunk ids are deterministic (``doc_id::n``) so re-indexing a
document upserts in place; the ``version`` field carries the index generation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from rag_common import Settings, get_logger, get_settings
from rag_core import Contextualizer, contextualize_text, get_chunker
from rag_core.chem import CHEM_TERMS_META, MoleculeNormalizer, canonical_terms
from rag_core.chunking import text_hash
from rag_core.doc_profile import PROFILE_VERSION, DocProfile, profile_card
from rag_core.interfaces import (
    CatalogAdapter,
    Embedder,
    MultiVectorEmbedder,
    MultiVectorStore,
    SparseIndex,
    VectorStore,
)
from rag_core.layout import LayoutTree
from rag_core.parent_child import is_parent
from rag_core.section_enrich import ENRICH_VERSION, SectionEnrichment
from rag_core.sections import detect_sections
from rag_schemas import Chunk, DocumentEnvelope
from rag_schemas.common import ChunkStrategy, Modality

_log = get_logger(__name__)

# Stamped on the document so a re-index can reproduce this document's own chunking. A STABLE
# QUERY VOCABULARY like the extraction provenance keys, so deliberately not env-tunable:
# renaming it would silently break every saved query and every reindex's ability to read back.
META_CHUNK_STRATEGY = "index_chunk_strategy"
# Fingerprint of every setting that shapes an index. A document whose stamp differs from the
# running fingerprint was indexed by a DIFFERENT pipeline and is STALE -- it still answers, but
# from chunks and vectors this deployment would no longer produce, and nothing else says so.
# Same stable-vocabulary rule as META_CHUNK_STRATEGY: never env-tunable.
META_PIPELINE_VERSION = "index_pipeline_version"
# The strategy this document would use if no structure tree were in hand. Recorded ALONGSIDE
# META_CHUNK_STRATEGY (never instead of it) because the layout path overrides the caller's strategy,
# and META_CHUNK_STRATEGY is the only place a KB's FR-C2 parent/child adoption is stored: without
# this, flag on -> reindex -> flag off -> reindex re-chunked a parent_child KB as the global default
# and nothing recorded that it had ever been parent_child. Same stable-vocabulary rule.
META_BASE_CHUNK_STRATEGY = "index_base_chunk_strategy"


def pipeline_fingerprint(settings: Any) -> str:
    """Short hash of the settings that decide what an index CONTAINS.

    Chunk sizing, the embedder (a different model or dimension is a different vector space),
    contextual retrieval, graph extraction, and the readers that produce the text. Two documents
    with the same stamp were indexed the same way; a stamp that differs from the running value
    means re-indexing would produce something different.

    Deliberately EXCLUDES query-time settings -- thresholds, top_k, the reranker, the generation
    model. Changing those changes the answer without changing the index, so folding them in here
    would mark an entire corpus stale for a change no re-index can act on.
    """
    parts = [
        settings.chunk_strategy,
        str(settings.chunk_size),
        str(settings.chunk_overlap),
        str(settings.chunk_max_chars),
        settings.embed_model,
        str(settings.embed_dim),
        str(settings.contextual_retrieval_enabled),
        settings.contextual_retrieval_backend,
        str(settings.graphrag_enabled),
        settings.graph_extract_backend,
        settings.ocr_backend,
        settings.image_extract_backend,
        settings.vision_llm_backend,
        str(settings.multimodal_enabled),
    ]
    # Slice 2b: append a marker ONLY when multivector retrieval is on, so turning it on marks the
    # corpus stale (its children now need dense+sparse+colbert) while turning it OFF keeps the
    # fingerprint BYTE-IDENTICAL to today — appending an unconditional field would mark every
    # existing document stale on deploy for a change the KB never made.
    if settings.multivector_enabled:
        parts.append(f"multivector:{settings.embed_backend}")
    # RAG v2 Phase 5, same conditional-append rule and for the same reason: with the v2 ingest flag
    # on, a document is chunked from its structure tree at a different size, so the corpus IS stale
    # and should say so. With the flag off the fingerprint stays BYTE-IDENTICAL to before this
    # line existed, so turning v2 on marks the corpus stale and turning it off marks nothing.
    if settings.v2_ingest:
        # Phase 3 extends this component rather than adding a second one: the profile card changes
        # what is EMBEDDED (it prefixes every unit's `embed_text`) and the per-doc_type map changes
        # where chunks are SPLIT, so both decide what the index contains and a corpus indexed under
        # a different value of either is genuinely stale. `PROFILE_VERSION` covers the card's own
        # shape — bumping it after a prompt change marks every profiled document for re-indexing,
        # which is the only way the new card reaches the vectors.
        parts.append(f"rag_v2_chunk:{settings.layout_chunk_max_chars}")
        parts.append(f"rag_v2_profile:{PROFILE_VERSION}:{sorted(settings.doc_type_chunk_chars.items())}")
        # Phase 4, same conditional-append rule: enrichment ADDS `section` and `proposition` units
        # to the index, so a corpus indexed with it off genuinely holds fewer units than this
        # pipeline would now produce and is stale. `ENRICH_VERSION` covers the prompt and parser
        # the same way `PROFILE_VERSION` covers the card's, so bumping it marks every enriched
        # document for re-indexing — the only way a changed enrichment reaches the vectors. The
        # per-section SIZING knobs (`section_enrich_min_chars`, the window settings) are
        # deliberately NOT folded in: they change which sections get a summary, not the shape of
        # the pipeline, and a corpus is not stale because an operator widened a floor.
        parts.append(f"rag_v2_enrich:{ENRICH_VERSION}:{settings.section_enrich_enabled}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]


def _layout_max_chars(settings: Settings, profile: DocProfile | None) -> int:
    """The layout chunk size for THIS document: the per-doc_type override where the operator set
    one, else the platform default (plan §3, "config map: doc_type -> chunk profile").

    The map ships EMPTY (`Settings.layout_chunk_max_chars_by_doc_type`), so this returns
    `layout_chunk_max_chars` for every document until somebody measures a document class and sets
    one — see that setting for why no values are invented here."""
    if profile is None:
        return settings.layout_chunk_max_chars
    return settings.doc_type_chunk_chars.get(profile.doc_type, settings.layout_chunk_max_chars)


@dataclass(slots=True)
class IndexResult:
    doc_id: str
    chunk_count: int
    # How this document was ACTUALLY chunked. Recorded so a re-index can reproduce it instead
    # of applying whichever literal the reindex entry point happened to carry — and so
    # "which chunker is this corpus using?" is answerable without inferring it from chunk sizes.
    strategy: ChunkStrategy = ChunkStrategy.FIXED
    # The strategy that WOULD have been used had no structure tree been supplied — set only when
    # the layout path overrode one, so the override is reversible. None means nothing was overridden
    # and `strategy` is the whole truth.
    base_strategy: ChunkStrategy | None = None
    # The pipeline that produced this index, so the document can be stamped with it and a later
    # deployment can tell whether it is still current. See :func:`pipeline_fingerprint`.
    pipeline_version: str = ""
    # The ids this run actually wrote. A write-then-swap re-index needs them to know what the
    # NEW version contains: reading them back from the catalog returns old AND new together, because
    # nothing was purged first, so "old minus fresh" is empty and the swap deletes nothing.
    # INCLUDES parent ids (FR-C2) so the swap never purges parents.
    chunk_ids: list[str] = field(default_factory=list)
    # The CHILD ids only (FR-C2) — the searchable/citable units. Consumers that must never touch a
    # parent (image-vector indexing, diagram-graph) use this instead of ``chunk_ids``, so a parent
    # cannot receive an image vector and become retrievable/citable. Equals ``chunk_ids`` for every
    # non-parent-child strategy.
    child_chunk_ids: list[str] = field(default_factory=list)
    # How many children were already indexed with identical text and so were NOT re-embedded
    # (Phase 6 `chunk_hash` idempotency). 0 whenever reuse was not asked for, which is every
    # caller that has not opted in.
    reused_chunks: int = 0


class IndexingService:
    """Chunk + embed + fan-out write. Store-agnostic (rag-core Protocols)."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        sparse_index: SparseIndex,
        catalog: CatalogAdapter,
        contextualizer: Contextualizer | None = None,
        *,
        contextual_enabled: bool | None = None,
    ) -> None:
        self._embedder = embedder
        self._vs = vector_store
        self._sparse = sparse_index
        self._catalog = catalog
        self._contextualizer = contextualizer
        self._contextual_enabled = (
            contextual_enabled if contextual_enabled is not None else get_settings().contextual_retrieval_enabled
        )

    def _use_multivector(self) -> bool:
        """True when this KB should store dense+sparse+colbert (Slice 2b): the master flag is on AND
        the resolved embedder + store both carry the multivector capability. Otherwise the dense-only
        path runs, byte-identical to today (a plain embedder or flag-off is a no-op)."""
        return (
            get_settings().multivector_enabled
            and isinstance(self._embedder, MultiVectorEmbedder)
            and isinstance(self._vs, MultiVectorStore)
        )

    async def _token_budget(self, text: str) -> tuple[int | None, list[int] | None]:
        """The embedder's token limit and its tokenisation of ``text``, when it can report them.

        Optional on the Embedder Protocol on purpose: the in-memory twin and any future adapter
        without a tokenizer simply do not implement it, and chunking degrades to the character
        cap rather than failing."""
        budget = getattr(self._embedder, "token_budget", None)
        offsets = getattr(self._embedder, "token_offsets", None)
        if budget is None or offsets is None:
            return None, None
        limit = await budget()
        return (limit, await offsets(text)) if limit else (None, None)

    async def _with_token_counts(self, chunks: list[Chunk]) -> list[Chunk]:
        """Stamp each RAG v2 chunk with the embedder's OWN token count for its stored text.

        The plan's ``chunks.token_count`` had no honest source: the document-level
        ``token_offsets`` cannot supply one, because a v2 chunk is assembled from markup blocks and
        so has no span into the plain-text layer to slice. The embedder's tokenizer is the only
        thing that knows, and it is asked here — once per document, batched — rather than estimated
        from characters, which is wrong by a factor of four on exactly the documents this phase
        exists for (see :attr:`rag_schemas.Chunk.token_count`).

        Deliberately v2-ONLY, on two counts: it is a plan §1.1 v2 column, and adding a key to a v1
        chunk's serialised record would make every record this build writes unreadable by the
        previous image (``Chunk`` is ``extra="forbid"``). With the v2 ingest flag off no chunk has a
        ``level``, so this returns immediately, makes no call, and the default path is
        byte-identical.

        Fail-soft by design: an embedder with no tokenizer endpoint (bge-m3's ``/embed`` facade, the
        in-memory twin) leaves ``token_count`` as None, which reads as "not known" rather than as a
        number. That is logged, because "every chunk in this corpus has a NULL token count" is a
        fact about the deployment's embedder and should not have to be inferred from empty columns.
        """
        targets = [i for i, c in enumerate(chunks) if c.level is not None]
        if not targets:
            return chunks
        count = getattr(self._embedder, "count_tokens", None)
        if count is None:
            _log.info(
                "token_count_unavailable",
                embedder=type(self._embedder).__name__,
                chunks=len(targets),
                detail="this embedder exposes no tokenizer; token_count is left unset, not estimated",
            )
            return chunks
        counts = await count([chunks[i].text for i in targets])
        if counts is None or len(counts) != len(targets):
            return chunks  # the adapter logs why; an unset count beats a wrong one
        stamped = dict(zip(targets, counts, strict=True))
        return [c.model_copy(update={"token_count": stamped[i]}) if i in stamped else c for i, c in enumerate(chunks)]

    async def _reusable(self, tenant_id: str, doc_id: str, children: list[Chunk]) -> set[str]:
        """The ids whose text is byte-identical to what is already indexed, so the embedder can be
        skipped for them (plan Phase 6, "make ingestion idempotent by ``chunk_hash``").

        Safe because of an ordering invariant this method does not create but does depend on: the
        catalog row is written LAST, after the vector upsert and the BM25 index. So a catalog row
        with a matching hash means those two writes completed for that chunk — the reverse (a
        vector with no row) is the failure mode this ordering produces, and it costs a re-embed
        rather than a missing vector. The CALLER is responsible for the other half: the vectors
        must have been produced by the same embedder, which is why the pipeline only passes
        ``reuse_unchanged=True`` when the document's recorded ``pipeline_fingerprint`` — which
        includes the embed model and dimension — still matches the running one.

        Hashes are compared, not ids alone. Ids are already content-derived, so id equality nearly
        implies text equality; but the id is a TRUNCATED digest (12 hex) while the hash is full
        sha256, and for a v2 chunk the id hashes `(section, level, ordinal, text_hash)` — so the
        id can be stable across an edit that moved text between ordinals. The hash is the thing
        that actually answers "is this the same text".
        """
        prior = await self._catalog.chunk_hashes(tenant_id, doc_id)
        if not prior:
            return set()
        return {c.chunk_id for c in children if prior.get(c.chunk_id) == text_hash(c.text)}

    async def ensure_namespaces(self, tenant_id: str) -> None:
        """Provision the tenant's vector collection + BM25 index (KB creation, FR-H1).

        A Slice-2b KB gets the NAMED dense+sparse+colbert collection so the embedder-switch
        recreate (delete_tenant → ensure_namespaces → per-doc reindex) rebuilds it in the 2b shape.
        """
        if self._use_multivector():
            await self._vs.ensure_multivector_collection(tenant_id, self._embedder.dim)  # type: ignore[attr-defined]
        else:
            await self._vs.ensure_collection(tenant_id, self._embedder.dim)
        await self._sparse.ensure_index(tenant_id)

    async def index_document(
        self,
        doc: DocumentEnvelope,
        text: str,
        *,
        page_map: list[int] | None = None,
        strategy: ChunkStrategy | None = None,
        modality: Modality = Modality.TEXT,
        ocr_confidence: float | None = None,
        child_max_chars: int | None = None,
        parent_max_chars: int | None = None,
        chem_normalizer: MoleculeNormalizer | None = None,
        layout: LayoutTree | None = None,
        mask: Callable[[str], str] | None = None,
        base_strategy: ChunkStrategy | None = None,
        reuse_unchanged: bool = False,
        profile: DocProfile | None = None,
        enrichment: Mapping[str, SectionEnrichment] | None = None,
    ) -> IndexResult:
        """``layout`` + ``mask`` are the RAG v2 Phase 5 inputs: the document's parsed structure
        tree and the PII masker its text must be de-identified through (the tree is captured
        pre-masking — see ``rag_core.layout_chunking``). They are only ever supplied by the
        flag-gated v2 ingest path; with the flag off both are None and this method is
        byte-identical to before they existed.

        ``profile`` is the Phase-3 document profile card, and it reaches chunking for two reasons
        the plan names: its ``doc_type`` selects a per-doc_type chunk size (§3) and the card
        itself becomes the contextualize prefix in front of every unit's ``embed_text`` (§5). It
        is None whenever the flag is off, the document has no structure tree, or the profile call
        declined to produce a card, and None leaves this method byte-identical to Phase 5.

        ``enrichment`` is the Phase-4 per-section enrichment keyed by ``section_id``. It reaches
        chunking because two of the plan's five levels are built from it — ``section`` (the
        summary) and ``proposition`` — and both must exist before the embed/index fan-out, not
        after it, or they would be a second write into stores the swap has already published.
        None or empty leaves this method byte-identical to Phase 3."""
        settings = get_settings()
        # One declared default for every entry point (RAG_CHUNK_STRATEGY); sizing injected
        # from config so tuning never means editing rag-core.
        # Kept before the default is applied: a strategy the CALLER named is a decision to
        # preserve, while one that came from `RAG_CHUNK_STRATEGY` is just the default and
        # recording it as a "base" would claim an adoption nobody made.
        requested = strategy
        strategy = strategy if strategy is not None else ChunkStrategy(settings.chunk_strategy)
        # RAG v2 Phase 5: a parsed structure tree is the ONLY input from which the v2 levels
        # (section / chunk / row) and their parent links are derivable, so when one is in hand it
        # decides the strategy — including over a strategy the caller passed, which on the reindex
        # path is the document's previously RECORDED one. That is deliberate and is how an
        # already-ingested corpus adopts v2: reindex with the flag on. `pipeline_fingerprint` gains
        # a v2 component at the same time, so the change is visible as staleness rather than
        # silent. Flag off ⇒ `layout is None` ⇒ this branch is dead.
        if layout is not None and mask is not None:
            # Remember what we are overriding, so the override can be UNDONE. On the reindex path
            # `requested` is the document's own recorded strategy, and for a KB that adopted FR-C2
            # parent/child that record is the only place the adoption exists. An already-v2
            # document re-reindexed keeps the base it was carrying rather than overwriting it with
            # `layout_aware`, and a document with no named strategy records no base at all.
            if requested is not None and requested is not ChunkStrategy.LAYOUT_AWARE:
                base_strategy = requested
            strategy = ChunkStrategy.LAYOUT_AWARE
        elif strategy is ChunkStrategy.LAYOUT_AWARE:
            # The document's RECORDED strategy is layout_aware but no tree came with this call: the
            # v2 flag has been turned off since it was ingested, or this re-extraction produced no
            # markup (Chandra is a stochastic VLM and an OCR engine can be swapped per org).
            # Layout-aware chunking then has no input at all. Re-chunk with the configured default
            # and let `pipeline_fingerprint` report the change as staleness — raising instead would
            # make a reindex fail outright for every document ingested while the flag was on, which
            # is how a corpus becomes un-reindexable by flipping one env var.
            # Restore what the document was chunked as BEFORE the layout path took it over, and
            # only fall back to the configured default when there is no such record (a document
            # first ingested as layout_aware never had another strategy). Falling back
            # unconditionally silently downgraded a parent_child KB to `clause` and purged its
            # parent context records on the swap, with nothing left saying it had been parent_child.
            strategy = base_strategy if base_strategy is not None else ChunkStrategy(settings.chunk_strategy)
            _log.info(
                "layout_aware_without_tree",
                doc_id=doc.doc_id,
                strategy=strategy.value,
                restored_from_record=base_strategy is not None,
            )
        chunker = get_chunker(
            strategy,
            size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            max_chars=settings.chunk_max_chars,
            # Parent/child sizing (FR-C2). Per-KB adaptive: the pipeline resolves the KB's overrides
            # and passes them here; when a caller passes nothing we fall back to the platform
            # default. Only consulted for PARENT_CHILD, so other strategies are unaffected.
            child_max_chars=child_max_chars if child_max_chars is not None else settings.parent_child_child_max_chars,
            parent_max_chars=parent_max_chars
            if parent_max_chars is not None
            else settings.parent_child_parent_max_chars,
            # Only consulted for LAYOUT_AWARE, so every other strategy is unaffected.
            layout=layout,
            mask=mask,
            layout_max_chars=_layout_max_chars(settings, profile),
            layout_card=profile_card(profile),
            layout_enrichment=enrichment,
        )
        # Ask the EMBEDDER how it would tokenize this document, once, before chunking. A
        # character cap cannot bound a token budget (prose ~5.1 chars/token, ToC dot leaders
        # ~1.25, Devanagari ~1.6), and TEI truncates over-long input silently — HTTP 200, a
        # normal-looking vector, no header. Content past the limit stays in the stored text and
        # in BM25 but is invisible to dense search.
        #
        # One call per document, not per chunk: the chunker binary-searches the offsets. An
        # embedder that cannot tokenize returns None and chunking falls back to characters
        # alone, which is logged by the adapter rather than assumed to be fine.
        max_tokens, token_offsets = await self._token_budget(text)
        # Headings are detected ONCE per document and handed to the chunker, so each chunk can
        # record the section that governs it — a rule under "6.2.3 Social Media Content" does
        # not apply to a sales deck, and nothing in the rule's own words says so.
        chunks = chunker.chunk(
            doc,
            text,
            page_map=page_map,
            token_offsets=token_offsets,
            max_tokens=max_tokens,
            sections=detect_sections(text),
        )
        if not chunks:
            _log.warning("no_chunks_produced", doc_id=doc.doc_id)
            return IndexResult(
                doc_id=doc.doc_id,
                chunk_count=0,
                strategy=strategy,
                base_strategy=base_strategy,
                pipeline_version=pipeline_fingerprint(settings),
            )

        # Stamp modality + OCR confidence onto chunks (carried for citations/audit).
        chunks = [c.model_copy(update={"modality": modality, "ocr_confidence": ocr_confidence}) for c in chunks]
        # ...and the embedder's own token count onto the v2 ones (no-op with the flag off).
        chunks = await self._with_token_counts(chunks)

        # FR-C2 parent/child: a PARENT is a context-only record — it must land in the CATALOG but
        # NEVER be embedded or BM25-indexed, or it would be retrieved/ranked/cited as if it were a
        # child (violating "children are the only search unit" and double-hitting dense+sparse).
        # For every other strategy this partition is a no-op (no chunk is a parent), so behaviour is
        # byte-identical. Children are what gets embedded/indexed; both are cataloged and BOTH ids go
        # into IndexResult.chunk_ids so the write-then-swap reindex ("stale = old - fresh") never
        # purges the parents it just wrote.
        parents = [c for c in chunks if is_parent(c)]
        children = [c for c in chunks if not is_parent(c)]

        tenant_id = doc.tenant_id
        # Phase 6 idempotency by `chunk_hash`: text that has not changed does not need embedding
        # again. Decided BEFORE contextualization, because with an LLM contextualizer that is a
        # model call per chunk and skipping the embed while still paying for the context would
        # save the cheaper half. `reuse_unchanged` is off by default, so every existing caller is
        # byte-identical; the pipeline turns it on only when the embedder cannot have changed.
        reusable = await self._reusable(tenant_id, doc.doc_id, children) if reuse_unchanged else set()
        to_index = [c for c in children if c.chunk_id not in reusable]
        multivector = self._use_multivector()
        if multivector:
            await self._vs.ensure_multivector_collection(tenant_id, self._embedder.dim)  # type: ignore[attr-defined]
        else:
            await self._vs.ensure_collection(tenant_id, self._embedder.dim)
        await self._sparse.ensure_index(tenant_id)

        # FR-C2: embed/index the CONTEXTUALIZED text; catalog keeps the ORIGINAL chunk.
        #
        # RAG v2 Phase 5 adds ONE step in front of that, and supersedes none of it. `Chunk.embed_text`
        # is the chunker's own contextualization — the plan's `"[{breadcrumb}] " + text` — which only
        # the chunker can build, because only it holds the structure tree. So it becomes the SOURCE
        # text here, and the existing `Contextualizer` + `model_copy` machinery is unchanged:
        #
        #  * the CONTEXTUALIZER still runs and still composes on top, so a KB with contextual
        #    retrieval on gets both the document-level context and the breadcrumb, and the ADR-0007
        #    seam (including the LLM backend) keeps working untouched;
        #  * the `model_copy` is still the MECHANISM by which "text embedded" differs from "text
        #    stored for citation" — `SparseIndex.index` indexes `Chunk.text`, so the copy is how the
        #    embedded text reaches BM25, and teaching the store adapters to read `embed_text`
        #    instead would be a store-contract change (Phase 6), not this phase;
        #  * the CATALOG keeps `children`, i.e. the ORIGINAL `text`, so a citation still shows what
        #    the document said. That property is the reason the copy exists and it is untouched.
        #
        # The copy is now made only when the text actually differs, which is what keeps the
        # flag-off/contextual-off path byte-identical (it was `index_chunks = children` before).
        embed_sources = [c.embed_text or c.text for c in to_index]
        contextual = self._contextual_enabled and self._contextualizer is not None
        if contextual:
            embed_texts = [
                contextualize_text(await self._contextualizer.context_for(doc, c, text), source)  # type: ignore[union-attr]
                for c, source in zip(to_index, embed_sources, strict=True)
            ]
        else:
            embed_texts = embed_sources
        index_chunks: list[Chunk] = [
            c.model_copy(update={"text": et}) if et != c.text else c
            for c, et in zip(to_index, embed_texts, strict=True)
        ]

        # Chemical normalization (Slice 6, Phase C): when the pipeline passed a normalizer (chem
        # flag on AND the KB's profile is chemistry AND its per-KB gate is set), stamp each child's
        # canonical SMILES/InChIKey terms — computed from its ORIGINAL text — onto the SPARSE-indexed
        # chunk's metadata. The catalog keeps the original chunks (children), so citations show the
        # original text unchanged; only the sparse index gains the extra searchable exact tokens. A
        # child with no detected molecule (or the Null normalizer) is left untouched ⇒ byte-identical.
        if chem_normalizer is not None:
            index_chunks = [
                ic.model_copy(update={"metadata": {**ic.metadata, CHEM_TERMS_META: " ".join(terms)}})
                if (terms := canonical_terms(original.text, chem_normalizer))
                else ic
                for original, ic in zip(to_index, index_chunks, strict=True)
            ]

        # Slice 2b: ONE embed_multi over the CHILDREN (parents are catalog-only, never embedded, so
        # they never get sparse/colbert and can never become retrievable). Dense stays the same
        # space; BM25 + the slice-2a exact path are UNCHANGED (bge-sparse is a separate RRF leg,
        # not a BM25 replacement). Dense-only KBs take the unchanged path.
        if to_index:
            if multivector:
                outputs = await self._embedder.embed_multi(embed_texts)  # type: ignore[attr-defined]
                await self._vs.upsert_multi(tenant_id, to_index, outputs)  # type: ignore[attr-defined]
            else:
                vectors = await self._embedder.embed(embed_texts)
                await self._vs.upsert(tenant_id, to_index, vectors)  # children only; payload has no text
            await self._sparse.index(tenant_id, index_chunks)  # BM25 over children (contextualized text)
        # The CATALOG is written for every child, reused ones included. Unchanged TEXT does not mean
        # an unchanged RECORD: `sequence` shifts when anything earlier in the document changes, and
        # `embed_text`/`token_count` move with it, so a reused chunk still needs its row refreshed.
        # It is also the cheap write — one upsert, no model call.
        await self._catalog.record_chunks(tenant_id, children + parents)  # parents catalog-only

        _log.info(
            "document_indexed",
            doc_id=doc.doc_id,
            chunks=len(children),
            parents=len(parents),
            embedded=len(to_index),
            reused=len(reusable),
            strategy=strategy,
            contextual=contextual,
        )
        return IndexResult(
            doc_id=doc.doc_id,
            chunk_count=len(children),
            strategy=strategy,
            base_strategy=base_strategy,
            pipeline_version=pipeline_fingerprint(settings),
            # Parents INCLUDED so the reindex swap keeps them (see the partition note above).
            chunk_ids=[c.chunk_id for c in children + parents],
            # Children only — for consumers (image vectors, diagram graph) that must never see a parent.
            child_chunk_ids=[c.chunk_id for c in children],
            reused_chunks=len(reusable),
        )
