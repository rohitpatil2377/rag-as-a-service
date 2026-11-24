"""Orchestration pipelines (TDD §5.1 / §5.2) — the two core data flows.

These compose the layer services in-process. Splitting a layer into its own process
later changes only :mod:`svc_control_plane.wiring`, not this orchestration.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from svc_guardrails_eval import AuditService
from svc_indexing.service import (
    META_BASE_CHUNK_STRATEGY,
    META_CHUNK_STRATEGY,
    META_PIPELINE_VERSION,
    pipeline_fingerprint,
)

from rag_clients.doc_profiler import LLMDocProfiler
from rag_clients.factory import (
    document_cards,
    document_relations,
    document_sections,
    get_catalog,
    get_faithfulness_checker,
    get_graph_store,
    get_molecule_normalizer,
    get_object_store,
    get_output_guard,
    get_retriever,
    get_sparse_index,
    get_vector_store,
    persist_document_profile,
    persist_document_structure,
    persist_enrichment_manifest,
    persist_section_enrichment,
    read_map_cache,
    record_layout_key,
    record_query_log,
    write_map_cache,
)
from rag_clients.factory import (
    enrichment_manifests as factory_enrichment_manifests,
)
from rag_clients.factory import (
    facet_vocabulary as factory_facet_vocabulary,
)
from rag_clients.llm_gateway import _estimate_usage
from rag_clients.query_router_v2 import LLMQueryRouter
from rag_clients.section_enricher import LLMSectionEnricher, SectionJob
from rag_clients.section_mapper import LLMSectionMapper, MapJob, MapReport
from rag_common import (
    NotFoundError,
    PlatformError,
    StageTimeline,
    ValidationError,
    get_logger,
    get_settings,
    get_tracer,
    layout_object_key,
    raw_object_key,
)
from rag_core import (
    GuardBudget,
    LLMGateway,
    abstain,
    build_completeness,
    build_condense_prompt,
    build_prompt,
    default_title,
    extract_entities,
    extract_filter_criteria,
    flag_for_review,
    ground,
    heuristic_condense,
    is_refusal_text,
    is_smalltalk,
    looks_truncated,
    match_compare_targets,
    merge_retrieved,
    route_faithfulness,
    should_abstain,
    verification_block,
    with_parent_context,
)
from rag_core.access import is_cleared, is_entitled
from rag_core.budget import batch_to_budget, prompt_budget_chars
from rag_core.chem import canonical_terms
from rag_core.chunking import META_CATEGORY, META_SECTION
from rag_core.clarify import cluster_spans
from rag_core.dedup import near_duplicates
from rag_core.doc_profile import PROFILE_VERSION, DocProfile, ProfileFacet, build_doc_profile_prompt
from rag_core.docscope import DocScope, scope_for_query
from rag_core.enumeration import enumerated_items
from rag_core.expansion import anchors_by_document, neighbour_ids
from rag_core.facet_vocab import CanonicalFacet, document_revision, reconcile
from rag_core.glossary import expand_glossary, load_glossary
from rag_core.graph import (
    GraphCommunity,
    GraphTriple,
    build_diagram_structure_prompt,
    community_label,
    detect_communities,
    filter_image_text_entities,
    is_global_query,
    parse_diagram_structure,
    rank_communities,
    relation_edges,
    relation_vocab,
)
from rag_core.language import detect_language
from rag_core.layout import LayoutTree, UnsupportedLayoutVersionError, parse_layout
from rag_core.mapreduce import (
    Coverage,
    MapCacheEntry,
    MappedItem,
    ReduceInput,
    assemble_answer,
    build_reduce_prompt,
    group_by_facet,
    map_cache_key,
    query_signature,
    resolve_item_chunk,
)
from rag_core.navigation import drop_navigation
from rag_core.parent_child import not_parent
from rag_core.query_plan import PlanContext, QueryPlan
from rag_core.recency import apply_recency
from rag_core.router import RouterDecision, RouterScope, heuristic_decision
from rag_core.section_enrich import (
    EnrichmentManifest,
    SectionEnrichment,
    assess_processing,
    section_prompt_text,
)
from rag_core.strategies import (
    SectionCard,
    SectionSelection,
    max_scores,
    plan_strategy,
    select_per_target,
    select_sections,
    taxonomy_of,
)
from rag_core.tagging import DERIVED_KEYS, auto_tags
from rag_core.verification import unverified_footnote, verify_claims, without_unverified
from rag_core.versioning import DocumentVersion, resolve_supersession
from rag_schemas import (
    AgentStep,
    Chunk,
    Citation,
    Completeness,
    DocumentEnvelope,
    FederationCoverage,
    GroundedAnswer,
    ProcessingReport,
    RetrievalQuery,
    RetrievalResult,
    RetrievedSpan,
)
from rag_schemas.common import AnswerStatus, AuditAction, ChunkLevel, ChunkStrategy, RetrievalMode
from svc_control_plane import (
    architecture,
    cross_lingual,
    graph_bridge,
    guardrails_policy,
    ingest_progress,
    multimodal,
    pii_classification,
    process_flow,
    unified,
    wiring,
)
from svc_control_plane.budgets import enforce_budget
from svc_control_plane.conversation import Conversation, Message
from svc_control_plane.hitl import REASON_FAITHFULNESS
from svc_control_plane.knowledge_base import SHARED_KB_ID
from svc_control_plane.semantic_cache import cache as semantic_cache
from svc_control_plane.semantic_cache import still_passes
from svc_control_plane.v2_cache import v2_cache

if TYPE_CHECKING:
    from svc_control_plane.federation import PeerFetch

_log = get_logger(__name__)

# KB-manager pins: doc_ids an operator marked authoritative, overruling dates.
# Supersession needs something to compare against: one document can never replace itself.
_MIN_COMPARABLE_DOCS = 2
_PINNED_DOCS_KEY = "authoritative_docs"


@dataclass(slots=True)
class IngestOutcome:
    doc_id: str
    chunk_count: int
    pii_masked: int
    trace_id: str
    outcome: str = "created"  # created | updated | unchanged | quarantined | blocked_pii (FR-A8 delta)
    duplicate_of: str | None = None  # another doc with identical content (FR-C9 near-dup)
    classification: int | None = None  # final data-classification (set/raised by the PII floor)
    reason: str | None = None  # human-readable detail when outcome == "blocked_pii"


async def _persist_layout_overlay(tenant_id: str, doc_id: str, extracted: Any, *, content_hash: str) -> None:
    """Store the extraction engine's LAYOUT OVERLAY beside the document's raw bytes, and point
    ``documents.minio_layout_key`` at it. Gated on ``v2_ingest`` — its first reader.

    ``content_hash`` is the SHA-256 of the raw bytes the overlay was extracted from, hashed from
    the bytes in hand rather than read off the catalog row, and stored inside the artifact. It is
    what turns overlay staleness from an argument into a FACT: a consumer re-hashes the stored raw
    object and compares, instead of trusting that every write path refreshed the overlay. It also
    lets a consumer skip re-parsing when the bytes have not changed. Required and keyword-only so
    a future call site cannot quietly omit it.

    No-op, with NOTHING written and the column left NULL, when the flag is off OR the document's
    engine produced no markup. Both Chandra paths produce markup (self-hosted chat → HTML, hosted
    Datalab /convert → Markdown, each declaring its own ``markup_format``); every plain-text path
    — native PDF, pypdf, Tesseract, Paddle, Office — produces none, and an empty artifact would be
    a worse answer than no artifact: a later parsing stage must be able to tell "this document was
    never captured with markup" from "its markup was empty", and a stored empty object says the
    second about a document for which the first is true. Flag off ⇒ this path is byte-identical to
    before it existed.

    WHY IT IS KEPT AT ALL: the markup is the only structural signal this platform receives. It is
    captured here so a future parsing stage (sections/blocks/tree) can be built, re-run and
    corrected WITHOUT re-OCRing the corpus, and so a structure claim can be traced back to the
    bytes the engine actually returned.

    COMPLIANCE POSITION (a reviewer will ask, and the answer is not obvious):

    * The overlay is an EXTRACTION ARTIFACT in the same class as the raw bytes. Both are captured
      BEFORE PII masking and therefore may contain personal data; both are retained deliberately,
      for re-extraction and provenance; both are written under the tenant's own object namespace;
      and both are purged by the SAME paths — ``delete_document`` (which DPDP right-to-erasure
      and KB delete both cascade through) and the failed-ingest rollback. There is no path that
      purges the raw object and leaves this one. A tenant that may keep the one may keep the other.
    * It is NOT chunked, NOT embedded and NOT indexed. Nothing retrieves it, nothing puts it in a
      prompt, and no citation points into it. The platform's invariant that ONLY de-identified
      text is chunked/embedded/indexed is therefore untouched by this slice: masking still runs
      at extraction and still governs everything that reaches the indexes.

    HONEST LIMIT, recorded where it is relevant: the markup carries no bounding boxes (see
    ``PageRecord.markup``), and only the Chandra engines produce it at all — so a parser built on
    it must fall back to the existing numbering-regex heading detection for every document that
    came through a plain-text extractor."""
    if not get_settings().v2_ingest:
        return
    # `extracted` is a doc-intelligence ExtractedDoc (typed Any here, as elsewhere in this
    # module, so the control plane does not import the service package to annotate it).
    overlay = extracted.layout
    if overlay is None or not overlay.pages:
        return
    key = layout_object_key(doc_id)
    body = json.dumps(overlay.for_content(content_hash).as_dict(), ensure_ascii=False).encode("utf-8")
    ceiling = get_settings().layout_overlay_max_bytes
    if len(body) > ceiling:
        # The page cap in `ocr._layout_capture_allowed` bounds the normal case from the page count;
        # this catches what a page count cannot see — a pathological page of thousands of tiny text
        # runs. Refused rather than truncated: a truncated overlay is not a smaller overlay, it is
        # an overlay that lies about where the document ends, and every consumer of it (the parser,
        # the structure rows, a later re-parse) would believe it.
        #
        # The STRUCTURE ROWS still land. They come from the same in-memory parse, not from reading
        # this object back, and they are the queryable form — the object is a convenience for
        # re-parsing without re-OCRing. Its absence is also detectable rather than silent, because
        # `documents.minio_layout_key` stays NULL, which is exactly the "never captured" signal
        # this function already uses for a plain-text document.
        _log.warning(
            "layout_overlay_too_large",
            tenant_id=tenant_id,
            doc_id=doc_id,
            bytes=len(body),
            ceiling=ceiling,
            pages=len(overlay.pages),
        )
        return
    await get_object_store().put(tenant_id, key, body, "application/json")
    # The object exists now, so the catalog MUST learn about it: a stored artifact nothing points
    # at is an orphan. record_layout_key fails closed on an unknown adapter, and on the ingest
    # path this runs inside the try block, so the rollback purges what was just written.
    recorded = await record_layout_key(tenant_id, doc_id, key)
    _log.info(
        "layout_overlay_stored",
        tenant_id=tenant_id,
        doc_id=doc_id,
        key=key,
        pages=len(overlay.pages),
        extractor=overlay.extractor,
        content_hash=content_hash[:12],
        bytes=len(body),
    )
    if not recorded:
        # UNREACHABLE from either call site — both run after `register_document`, so the row is
        # there (ingest registers before extraction; reindex re-registers before this is called).
        # Logged at ERROR rather than raised: by here the object is written, and on the reindex
        # path raising would 500 an otherwise-successful swap WITHOUT removing the orphan, which
        # is strictly worse than naming it. The orphan key is in the message so an operator can
        # purge it by hand.
        _log.error("layout_overlay_orphaned", tenant_id=tenant_id, doc_id=doc_id, key=key)


def _strategy_metadata(indexed: Any) -> dict[str, str]:
    """The two chunking facts recorded on a document, as metadata.

    ``base_strategy`` is recorded ONLY when the layout path overrode a strategy, and is dropped
    from the metadata otherwise — an absent key means "nothing was overridden", which is a
    different statement from a key whose value equals ``strategy``. It is what makes the override
    reversible: a reindex with the flag off restores this instead of falling back to the global
    default, which used to downgrade a parent/child KB to `clause` with nothing left recording
    that it had ever been parent/child.
    """
    out = {
        META_CHUNK_STRATEGY: indexed.strategy.value,
        META_PIPELINE_VERSION: indexed.pipeline_version,
    }
    if indexed.base_strategy is not None:
        out[META_BASE_CHUNK_STRATEGY] = indexed.base_strategy.value
    return out


def _mask_with(doc_intel: Any) -> Callable[[str], str]:
    """``str -> str`` over the extraction service's own "safe to index" pipeline (RAG v2 Phase 5).

    This deliberately does NOT reach for ``doc_intel.masker``. Masking is only one of the three
    things ``extract`` does before text is allowed into the index — it also NORMALISES before
    masking (so an identifier broken across a line is healed and therefore matchable) and
    DE-CLOAKS after (``sanitize.clean``, whose contract is that only cleaned text reaches the index
    or the citation viewer). The layout overlay is captured before normalisation, so a chunker
    handed the bare masker indexed wrapped identifiers in clear and skipped de-cloaking entirely.
    ``DocIntelligenceService.prepare_for_index`` owns all three, so there is ONE definition of what
    is safe to index rather than one per caller.

    The service returns a plain ``str`` here: the ``PIIMasker`` Protocol's ``MaskResult`` exists
    because ``extract`` has to move a page map with the redactions, and asking rag-core to know
    that type would make a pure library depend on a service one. ``doc_intel`` is typed ``Any`` for
    the same reason every other extraction value is in this module: the control plane does not
    import the service package to annotate it."""
    return lambda value: str(doc_intel.prepare_for_index(value))


def _layout_tree(doc_id: str, extracted: Any, *, text: str) -> LayoutTree | None:
    """Parse this document's layout overlay into a section/block tree. Gated on ``v2_ingest``.

    ONE parse, TWO consumers, which is why this is its own function: the tree is what Phase 5
    CHUNKS (``IndexingService.index_document``'s ``layout``) and what
    :func:`_persist_layout_structure` STORES. Parsing twice would let a chunk's ``section_id`` name
    a row that a second, independently-parsed tree never produced — the parse is deterministic, but
    a later edit to one of two call sites would not be.

    None when the flag is off, when the document has no overlay, or when the overlay declares a
    ``schema_version`` this build does not understand (logged, never raised — see
    :func:`_persist_layout_structure` for why an unparseable artifact must not cost a good document
    its ingest).
    """
    if not get_settings().v2_ingest:
        return None
    # `extracted` is a doc-intelligence ExtractedDoc (typed Any here, as elsewhere in this module).
    # `layout_overlay()` always returns an overlay — with no pages when no page carried markup —
    # so the version in the dict is the WRITER's own, never one this module picks.
    overlay = extracted.layout
    if overlay is None:
        return None
    try:
        return parse_layout(overlay.as_dict(), doc_id, text=text)
    except UnsupportedLayoutVersionError as exc:
        _log.warning("layout_structure_unparseable", doc_id=doc_id, reason=str(exc))
        return None


def _chunkable_layout(tree: LayoutTree | None, *, text_pages: frozenset[int]) -> LayoutTree | None:
    """The tree, but only when it carries a BLOCK stream — i.e. only when layout-aware chunking has
    something to chunk.

    A plain-text document's tree (``parse_layout``'s fallback over ``detect_sections``) has sections
    and NO blocks, so there is no block text to build units from. Such a document keeps today's
    clause chunking and keeps RECORDING ``clause`` as its strategy, which is the honest stamp:
    handing the chunker a blockless tree would have it fall back internally and then stamp the
    document ``layout_aware`` for chunks the clause engine produced. Only the Chandra engines emit
    markup, so this is most of a real corpus — a stated Phase-5 limit, not an oversight.

    THE BLOCK STREAM MUST COVER EVERY PAGE THAT PRODUCED TEXT, not merely some page. The chunker
    reads its text from the blocks and ignores the ``text`` argument entirely, so a page absent
    from the overlay is not chunked, not embedded and not indexed — silently absent from the corpus
    while ingest reports success with a plausible chunk count. Pages go missing from an overlay on
    paths this codebase already treats as routine and merely LOGS: pdfplumber raising on one page
    (``_fallback_page``, ``pdf_page_extraction_failed``), the layout capture itself raising
    (``pdf_layout_capture_failed``, deliberately non-fatal), the hosted Chandra ``/convert``
    answering Markdown for a page (recorded in ``skipped_pages``), or the engine answering a page
    with "". On a 200-page tender with three unreadable annexure pages the clauses printed there
    would be permanently unretrievable and uncitable, and a question about them would abstain or
    answer from an adjacent clause at ``faithfulness_score 1.0`` — with nothing correlating the
    extraction warning to the gap.

    So a PARTIAL overlay declines the WHOLE document to clause chunking, which is lossless, rather
    than indexing the covered pages and dropping the rest. Declining is logged at WARNING because
    it costs the document its structure silently, and an operator asking "why did this document not
    get v2 chunks" needs the reason. Chunking the residual text alongside the tree would keep both
    and is the better answer once there is a way to say which spans of ``text`` the blocks already
    cover — there is not one today, so this takes the safe half.
    """
    if tree is None:
        return None
    block_pages = _block_pages(tree.sections)
    if not block_pages:
        return None
    uncovered = sorted(text_pages - block_pages)
    if uncovered:
        _log.warning(
            "layout_chunking_declined_partial_overlay",
            doc_id=tree.doc_id,
            uncovered_pages=uncovered[:20],
            uncovered_count=len(uncovered),
            covered_count=len(block_pages),
        )
        return None
    return tree


def _any_blocks(sections: tuple[Any, ...]) -> bool:
    """Whether any section in this subtree owns blocks."""
    return any(section.blocks or _any_blocks(section.children) for section in sections)


# A page that rendered as nothing but its `ocr.PAGE_SEPARATOR` ("\n\n") carried no text, so the
# block stream owes it no coverage. Compared as a length rather than imported, because the control
# plane does not import the extraction package (see `_mask_with`).
_EMPTY_PAGE_CHARS = 2


def _text_pages(page_map: list[int]) -> frozenset[int]:
    """The 1-based pages that actually produced text.

    ``page_map`` is per-page CHARACTER COUNTS — ``sum(page_map) == len(text)`` is the contract
    `svc_doc_intelligence.service` states — not a list of page numbers.
    """
    return frozenset(i for i, chars in enumerate(page_map, start=1) if chars > _EMPTY_PAGE_CHARS)


def _block_pages(sections: tuple[Any, ...]) -> frozenset[int]:
    """Every page the block stream reaches, including the extra pages of a merged table."""
    pages: set[int] = set()
    for section in sections:
        for block in section.blocks:
            pages.add(block.page)
            pages.update(block.pages)
        pages |= _block_pages(section.children)
    return frozenset(pages)


def _profile_title(doc: DocumentEnvelope) -> str:
    """The platform's own title for a document: what a reader would call it.

    Given TO the profile prompt, never asked FOR — see :class:`rag_core.doc_profile.DocProfile`.
    `documents.title` is the source node's name when `document_relations` seeds the graph, so a
    model-invented title would put a fabricated node in the graph."""
    meta = doc.metadata
    for key in ("title", "filename", "name"):
        if (value := meta.get(key)) and isinstance(value, str) and value.strip():
            return value.strip()
    return (doc.uri or "").rsplit("/", 1)[-1].strip() or doc.doc_id


def _profile_breadcrumbs(tree: LayoutTree, mask: Callable[[str], str], limit: int) -> str:
    """This document's section headings, in document order, as the prompt's outline.

    MASKED here, and this is the one input where that is easy to forget: the breadcrumbs come off
    the structure tree, which is captured PRE-masking (``_persist_layout_structure`` states the
    position), so unlike ``extracted.text`` they carry PII as the extractor read it. They go into a
    prompt whose answer is stored in `documents.purpose`/`profile_json` and embedded into every v2
    vector in the document.

    De-duplicated in order: a breadcrumb repeats once per genuine repeat of the same heading path
    (`LayoutSection.ordinal` is what separates those), and the outline wants the shape of the
    document, not its repetition counts. Truncated with the dropped count SAID, rather than
    silently cut, so a model shown half a 200-page tender's outline is not left to infer that the
    document simply ends there."""
    seen: dict[str, None] = {}
    for section in _walk_sections(tree.sections):
        if crumb := section.breadcrumb.strip():
            seen.setdefault(crumb, None)
    crumbs = [mask(c) for c in list(seen)[:limit]]
    dropped = len(seen) - len(crumbs)
    listing = "\n".join(f"- {c}" for c in crumbs)
    return f"{listing}\n- (... and {dropped} further headings, not listed)" if dropped > 0 else listing


def _walk_sections(sections: tuple[Any, ...]) -> Iterator[Any]:
    """Pre-order, so breadcrumbs come out in document order (mirrors `layout_chunking._walk`)."""
    for section in sections:
        yield section
        yield from _walk_sections(section.children)


def _profile_evidence(extracted: Any, limit: int) -> str:
    """The text the profile call is shown, and the text its quotes are checked against.

    THE PLAN SAYS "first ~2 pages of text". That is the whole of a memo and 2% of an 80-page
    tender, and the measurement says it is the binding constraint on the card's quality. On the
    real AEGCL corrigendum the first two pages are 4,119 of 193,432 characters, and the facet
    taxonomy built from them covers 3-4 of the plan's own 10 themes. Given the full
    `doc_profile_head_chars` budget instead — the same tokens, already provisioned, just not all
    spent on the cover sheet — the same prompt covers 4-6. Measured over five runs each:

        first 2 pages (4,119 chars)    0.30 - 0.40, mean 0.38
        full budget   (12,000 chars)   0.40 - 0.60, mean 0.52

    CONTIGUOUS, AND THAT WAS TESTED TOO. Spending the same budget on the opening plus six evenly
    spaced samples of the rest — reaching the whole document rather than its first 6% — scored
    WORSE, 0.40-0.50 and mean 0.44. Six fragments of mid-document table rows are poorer evidence
    than one coherent passage, so the obvious idea loses to the simple one and the document is
    read from the front.

    A SHORT DOCUMENT IS UNAFFECTED: it is shorter than the budget, so this returns all of it,
    which is what the two-page rule was protecting in the first place.

    ``extracted.text`` is the MASKED text layer — the same string that is chunked and indexed —
    not the overlay's block text, which is unmasked."""
    return str(extracted.text or "")[:limit]


async def _document_profile(
    tenant_id: str, doc: DocumentEnvelope, extracted: Any, tree: LayoutTree | None, *, mask: Callable[[str], str]
) -> DocProfile | None:
    """Build this document's Phase-3 profile card. Gated on ``v2_ingest``; NEVER raises.

    Returns None when the flag is off, when the document has no structure tree, or when the model
    call or its reply was unusable (``LLMDocProfiler`` is fail-safe and logs its own reasons).
    None means the rest of the pipeline behaves exactly as it did in Phase 5.

    IT NEEDS THE TREE, which is why there is no profile for a document that has none. The plan's
    prompt input is "title, first ~2 pages of text, FULL LIST OF BREADCRUMBS, document metadata",
    and the breadcrumbs are the half that makes the facet taxonomy describe the document's own
    structure rather than its opening paragraphs. A document whose extraction produced no overlay
    has no outline to show, and profiling it from two pages alone would produce a card with the
    same columns and much weaker evidence behind them — indistinguishable downstream from one
    built properly. So it is left unprofiled, and a reindex after the extractor improves gives it
    a card.

    CALLED BEFORE INDEXING, PERSISTED AFTER IT. The card decides two things about chunking — the
    per-doc_type size and the `embed_text` prefix — so it has to exist before `index_document`
    runs; it is written to the catalog only after the new chunks are durably indexed, by
    :func:`_persist_document_profile`, for the write-then-swap reason that call states."""
    if not get_settings().v2_ingest or tree is None:
        return None
    settings = get_settings()
    try:
        # Masked ONCE and reused: the prompt is shown this title and the stored card carries it, so
        # computing it twice would let the two drift if the masker ever stopped being pure.
        title = mask(_profile_title(doc))
        head = _profile_evidence(extracted, settings.doc_profile_head_chars)
        prompt = build_doc_profile_prompt(
            title=title,
            metadata=_profile_metadata(doc, extracted),
            breadcrumbs=_profile_breadcrumbs(tree, mask, settings.doc_profile_max_breadcrumbs),
            head_text=head,
        )
        profiler = LLMDocProfiler(
            await wiring.resolve_tenant_gateway(tenant_id), max_tokens=settings.doc_profile_max_tokens
        )
        # The evidence window is the HEAD TEXT, not the whole document — the model was shown this
        # and nothing else, so a quote it produced from elsewhere is a fabrication even if some
        # later page happens to contain it. See `parse_doc_profile`.
        card = await profiler.profile(
            prompt,
            source_text=head,
            title=title,
            # The platform's OWN language detection, which overrides whatever the model says: it
            # answers in ISO-639-1 and the model does not (`parse_doc_profile` records the
            # measurement), and `documents.lang` must hold one vocabulary.
            lang=detect_language(extracted.text or ""),
            doc_id=doc.doc_id,
        )
        return await _pin_facet_vocabulary(tenant_id, doc, card)
    except Exception as exc:  # fail-safe, like every other derived artifact on this path
        _log.warning("doc_profile_skipped", tenant_id=tenant_id, doc_id=doc.doc_id, error=str(exc)[:160])
        return None


async def _pin_facet_vocabulary(tenant_id: str, doc: DocumentEnvelope, profile: DocProfile | None) -> DocProfile | None:
    """Replace a freshly proposed facet taxonomy with this document revision's APPROVED one.

    WHY THE PROPOSAL IS NOT TRUSTED. Phase 3 invents the vocabulary on every call, and a facet id
    is not descriptive metadata: it is the join key of Phase 8's facet union AND, de-underscored,
    the cross-encoder probe text its score stage is driven by. Measured on `corrigendum-5.pdf`
    (2026-11-02): three ingests of identical bytes produced three 10-facet vocabularies sharing
    ONE id, and six of the eight sections that lost score admission had been carried by a single
    probe that two of the three runs never re-invented. So re-ingesting a document re-rolled its
    own selection. See `rag_core.facet_vocab`.

    NEVER RAISES, and the degraded path is the OLD behaviour: no revision (a document with no
    content hash), no stored vocabulary, or a reader that cannot answer all mean "adopt the
    proposal" -- exactly what every ingest did before this existed.

    CHANGED BYTES DO NOT INHERIT. `reconcile` reuses a vocabulary only when it is pinned to THIS
    revision; an amended document takes the taxonomy of the amended text, and the decision is
    logged either way so the change is visible rather than silent."""
    if profile is None:
        return None
    revision = document_revision(doc.content_hash or "", profile.profile_version)
    if not revision:
        return profile
    try:
        stored = await factory_facet_vocabulary(tenant_id, doc.doc_id)
    except Exception as exc:  # a reproducibility improvement must never cost a document its ingest
        _log.warning("facet_vocabulary_read_failed", tenant_id=tenant_id, doc_id=doc.doc_id, error=str(exc)[:160])
        return profile
    decision = reconcile(
        stored,
        [CanonicalFacet(facet_id=f.facet_id, name=f.name, description=f.description) for f in profile.facets],
        revision=revision,
    )
    _log.info(
        "facet_vocabulary_decided",
        tenant_id=tenant_id,
        doc_id=doc.doc_id,
        revision=revision,
        reused=decision.reused,
        # Non-empty with `reused=False` means a vocabulary existed for OTHER bytes and was
        # deliberately not carried forward.
        stored_revision=decision.stored_revision or None,
        facets=len(decision.vocabulary.facets),
        mapped=len(decision.mapped),
        # CANDIDATES for a controlled vocabulary update, never adopted here: a model-proposed
        # label that reached a join key would reintroduce the churn this exists to stop.
        unmapped=list(decision.unmapped)[:10] or None,
        unproposed=list(decision.unproposed)[:10] or None,
    )
    if not decision.reused:
        return profile
    return replace(
        profile,
        facets=tuple(
            ProfileFacet(facet_id=f.facet_id, name=f.name, description=f.description, aliases=f.aliases)
            for f in decision.vocabulary.facets
        ),
    )


def _profile_metadata(doc: DocumentEnvelope, extracted: Any) -> str:
    """The prompt's "document metadata" input: a few platform-observed facts, nothing derived by a
    model and nothing that would lead the answer.

    The v1 heuristic `doc_type` from `tagging.py` is deliberately NOT included even though it is
    sitting in `doc.metadata` by this point: it is a first-match-wins keyword guess over seven
    labels (discovery §5), and handing it to the model as context would anchor a far better
    classifier to a far worse one — the plan's Phase 3 exists to replace it, not to confirm it."""
    pages = len(extracted.page_map or [])
    facts = [f"mime: {doc.mime}", f"source: {doc.source_kind.value}"]
    if pages:
        facts.append(f"pages: {pages}")
    if doc.uri:
        facts.append(f"filename: {doc.uri.rsplit('/', 1)[-1]}")
    return "\n".join(f"- {f}" for f in facts)


async def _persist_document_profile(
    tenant_id: str, doc_id: str, profile: DocProfile | None, content_hash: str = ""
) -> None:
    """Write the profile card: ``documents`` columns + ``facets`` + ``doc_relations`` (Phase 3).

    None is a no-op — flag off, no tree, or no usable card — and leaves whatever is stored exactly
    as it was. That is the same position :func:`_persist_layout_structure` takes with the flag off,
    and it matters more here: the profile call is fail-safe by design, so None is an ORDINARY
    outcome, and purging a good card from the previous ingest because this run's model call timed
    out would turn a transient model fault into data loss.

    Placed after indexing on both paths for the reason `_persist_layout_structure`'s call states:
    on an update, write-then-swap leaves the PREVIOUS version serving if anything above raises, and
    a card written before that point would describe a re-extraction no reader can reach."""
    if profile is None:
        return
    # PINS THE FACET ROWS TO THE BYTES THEY DESCRIBE, which is what lets the next ingest of the
    # same bytes reuse these ids instead of accepting a fresh invention (`_pin_facet_vocabulary`).
    # Empty for a document with no content hash: those rows are written as before and are simply
    # never reused, because `facet_vocabulary` refuses a vocabulary that cannot name its bytes.
    revision = document_revision(content_hash, profile.profile_version)
    if not await persist_document_profile(tenant_id, doc_id, profile, revision=revision):
        # No catalog row to hang the card on. Logged rather than raised: every caller is inside the
        # derived-artifact block that must never abort the swap.
        _log.warning("doc_profile_no_document_row", tenant_id=tenant_id, doc_id=doc_id)
        return
    _log.info(
        "doc_profile_stored",
        tenant_id=tenant_id,
        doc_id=doc_id,
        doc_type=profile.doc_type,
        facets=len(profile.facets),
        relations=len(profile.relations),
        revision=revision or None,
    )


def _enrichable_sections(tree: LayoutTree, settings: Any) -> tuple[list[tuple[Any, str]], int]:
    """The sections worth one model call each, in document order, with the text to send.

    Two filters, both of them about not spending a call on a node that has nothing in it:

    * a section whose OWN prompt text is shorter than `section_enrich_min_chars` is skipped. In a
      deep outline most nodes are container headings whose children hold the content, and a
      summary of "13. Video Wall Technical Requirements" restates the breadcrumb every chunk of
      the section already carries. A LEAF with little text is skipped for the same reason: there
      is nothing to summarise that its own `chunk` unit does not already hold verbatim.
    * the list is truncated at `section_enrich_max_sections`, in document order, so a
      pathological outline cannot turn one ingest into thousands of calls. The caller logs the
      truncation — a document enriched to its cap must not look merely half-interesting.

    The text is :func:`section_prompt_text`, NOT ``LayoutSection.text``: the latter carries every
    table twice (once as the table block, once per row block) because Phase 2 emits both. It is
    still UNMASKED here — the caller masks, at the same point it masks the breadcrumb."""
    out: list[tuple[Any, str]] = []
    for section in _walk_sections(tree.sections):
        body = section_prompt_text(section)
        if len(body.strip()) >= settings.section_enrich_min_chars:
            out.append((section, body))
    # THE PRE-CAP COUNT IS RETURNED TOO, because three things look identical in a `sections` row
    # with no summary and only one of them is a defect: the minimum-length rule DECLINED it, the
    # cap meant it was never offered, or a model call FAILED on it. The manifest has to tell them
    # apart, and after truncation this function can no longer say which happened.
    return out[: settings.section_enrich_max_sections], len(out)


async def _section_enrichment(
    tenant_id: str,
    doc: DocumentEnvelope,
    tree: LayoutTree | None,
    profile: DocProfile | None,
    *,
    mask: Callable[[str], str],
) -> tuple[dict[str, SectionEnrichment], EnrichmentManifest | None]:
    """Enrich this document's sections (Phase 4), and say what it managed to do.

    THE MANIFEST IS RETURNED ALONGSIDE, not logged and forgotten. A section whose enrichment
    failed is written with no summary, no ``facet_ids`` and no ``amends``, which closes every
    Phase-8 selection path into it — so an answer served later cannot see that part of the
    document and, until this was persisted, could not say so either. None when no enrichment ran
    at all (flag off, no tree), which a reader must treat as UNKNOWN rather than as complete.

    Gated on ``v2_ingest``; NEVER raises.

    Returns an empty map when the flag is off, when `section_enrich_enabled` is off, when the
    document has no structure tree, or when every call failed — and an empty map leaves chunking
    exactly as Phase 3 left it. A PARTIAL map is the ordinary outcome and is not an error: the
    sections that were enriched get a ``section`` summary unit and their ``proposition`` units,
    and every other section of the document is untouched.

    IT NEEDS THE TREE, for a harder reason than the profile call does: the unit of work here IS a
    section, and a document with no structure tree has none to enrich.

    IT DOES NOT NEED THE PROFILE. A document whose profile call failed still has sections worth
    summarising, propositions worth extracting and amendments worth recording — what it loses is
    the facet vocabulary, so every facet label the model offers is dropped as unknown and
    `sections.facet_ids` stays empty. Declining to enrich at all would turn one fail-safe model
    call into the loss of a whole phase, which is precisely the coupling Phase 3's fail-safe
    exists to avoid.

    EVERY STRING THAT REACHES THE PROMPT IS MASKED HERE. The structure tree is captured
    PRE-masking, so the breadcrumb and the section body carry PII as the extractor read it — and
    what comes back is written to `sections.summary`/`key_values`/`amends` and indexed as
    ``Chunk.text`` at two levels. `profile.as_json()` is already masked (Phase 3 masked the title
    before building the card and the purpose is derived from masked text).

    CALLED BEFORE INDEXING, PERSISTED AFTER IT, exactly like the profile card and for the same
    reasons: the summaries and propositions become chunks, so they must exist before the
    embed/index fan-out; the `sections` rows describe text that is SERVING, so they are written
    only once the new chunks are durable."""
    settings = get_settings()
    if not settings.v2_ingest or not settings.section_enrich_enabled or tree is None:
        return {}, None
    try:
        candidates, enrichable = _enrichable_sections(tree, settings)
        total = sum(1 for _ in _walk_sections(tree.sections))
        if not candidates:
            # NOT None: enrichment ran and correctly found nothing worth a call. A document of
            # pure furniture is COMPLETE, not unknown, and the counts say why.
            return {}, EnrichmentManifest(
                content_hash=doc.content_hash or "",
                sections_total=total,
                sections_skipped_policy=total,
            )
        jobs = [
            SectionJob(section_id=section.section_id, breadcrumb=mask(section.breadcrumb), text=mask(body))
            for section, body in candidates
        ]
        enricher = LLMSectionEnricher(
            await wiring.resolve_tenant_gateway(tenant_id),
            max_tokens=settings.section_enrich_max_tokens,
            concurrency=settings.section_enrich_concurrency,
            window_chars=settings.section_enrich_window_chars,
            max_windows=settings.section_enrich_max_windows,
            max_attempts=settings.section_enrich_max_attempts,
        )
        enrichment, report = await enricher.enrich(
            jobs,
            profile_json=profile.as_json() if profile is not None else "",
            facet_ids=[f.facet_id for f in profile.facets] if profile is not None else [],
            doc_id=doc.doc_id,
        )
        manifest = EnrichmentManifest(
            content_hash=doc.content_hash or "",
            sections_total=total,
            sections_offered=len(jobs),
            sections_enriched=report.sections_enriched,
            # TOTAL minus what was ENRICHABLE, so the cap is not counted as a policy skip. The
            # cap's own overflow is `capped` and is neither: those sections passed the rule and
            # were never offered to a model.
            sections_skipped_policy=total - enrichable,
            sections_failed=report.sections_failed,
            sections_partial=report.sections_partial,
            failure_classes=dict(report.failure_classes),
            failed_section_ids=report.failed_section_ids,
            capped=enrichable > settings.section_enrich_max_sections,
        )
        _log.info(
            "section_enrich_scoped",
            tenant_id=tenant_id,
            doc_id=doc.doc_id,
            sections_total=total,
            sections_offered=len(jobs),
            sections_enriched=report.sections_enriched,
            sections_failed=report.sections_failed,
            sections_partial=report.sections_partial,
            # What the phase actually SPENT, against the ceiling it was allowed. `max_tokens` is
            # the setting; `completion_tokens` is the only number that says what raising it cost.
            completion_tokens=report.completion_tokens,
            calls_at_cap=report.calls_at_cap,
            max_tokens=settings.section_enrich_max_tokens,
            # A document with more enrichable sections than the cap: the ones past it were never
            # read, and that is worth seeing in the same line as the counts that look complete.
            # `>` and not `>=` now that the pre-cap count is known: a document with exactly
            # `max_sections` enrichable sections lost nothing to the cap, and the old comparison
            # reported it as truncated.
            capped=manifest.capped,
            sections_skipped_policy=manifest.sections_skipped_policy,
            profiled=profile is not None,
        )
        return enrichment, manifest
    except Exception as exc:  # fail-safe, like every other derived artifact on this path
        _log.warning("section_enrich_skipped", tenant_id=tenant_id, doc_id=doc.doc_id, error=str(exc)[:160])
        # None, NOT an empty manifest. Enrichment was attempted and we cannot say what it did, so
        # completeness is UNKNOWN; a zero-failure manifest here would claim a clean run.
        return {}, None


async def _persist_section_enrichment(tenant_id: str, doc_id: str, enrichment: Mapping[str, SectionEnrichment]) -> None:
    """Write the Phase-4 enrichment onto this document's ``sections`` rows.

    An empty map is a no-op — flag off, enrichment off, no tree, or every call failed — and leaves
    whatever is stored exactly as it was, the same position :func:`_persist_document_profile`
    takes with None and for the same reason: the enrichment calls are fail-safe by design, so
    "nothing came back" is an ORDINARY outcome and must not purge a previous run's good rows.

    MUST RUN AFTER :func:`_persist_layout_structure`, not before. That call DELETEs and re-INSERTs
    every one of the document's `sections` rows, so an enrichment written first would be erased by
    it with nothing reporting the loss. The ordering is why both live in the same derived-artifact
    block rather than being written where each is derived.

    This is also what makes the SECOND half of Phase 6's graph seeding live: `document_relations`
    reads `sections.amends`, which has had no producer until now."""
    if not enrichment:
        return
    updated = await persist_section_enrichment(tenant_id, doc_id, enrichment)
    if updated < len(enrichment):
        # Logged rather than raised: every caller is inside the derived-artifact block that must
        # never abort the swap. The adapter warns too; this says it with the tenant in view.
        _log.warning(
            "section_enrichment_partially_stored",
            tenant_id=tenant_id,
            doc_id=doc_id,
            offered=len(enrichment),
            updated=updated,
        )
        return
    _log.info(
        "section_enrichment_stored",
        tenant_id=tenant_id,
        doc_id=doc_id,
        sections=updated,
        amends=sum(1 for e in enrichment.values() if e.amends is not None),
        propositions=sum(len(e.propositions) for e in enrichment.values()),
    )


async def _persist_v2_derived(
    tenant_id: str,
    doc_id: str,
    tree: LayoutTree | None,
    profile: DocProfile | None,
    enrichment: Mapping[str, SectionEnrichment],
    manifest: EnrichmentManifest | None = None,
    content_hash: str = "",
) -> None:
    """The four RAG-v2 derived writes, IN THE ONE ORDER THAT IS CORRECT.

    Here as one function rather than three calls at each site because the order is a real
    constraint and not a style: :func:`_persist_layout_structure` DELETEs and re-INSERTs every one
    of the document's `sections` rows, so :func:`_persist_section_enrichment` — which UPDATEs
    those same rows — must follow it or lose every summary with nothing reporting the loss. Two
    call sites each repeating the order is two places for it to be got wrong.

    Each of the three is individually a no-op for a document that has nothing of its kind, and all
    three are placed AFTER indexing on both paths: they describe the text that is SERVING, and
    write-then-swap leaves the previous version serving if indexing raised."""
    await _persist_layout_structure(tenant_id, doc_id, tree)
    await _persist_document_profile(tenant_id, doc_id, profile, content_hash)
    await _persist_section_enrichment(tenant_id, doc_id, enrichment)
    # LAST, and on `documents` rather than `sections`, so it is outside the delete-and-reinsert
    # hazard the ordering above exists for. It still goes last so a manifest is never stored for
    # a run whose other writes failed: a manifest is a claim about what this document's stored
    # state now is.
    await _persist_enrichment_manifest(tenant_id, doc_id, manifest)


async def _persist_enrichment_manifest(tenant_id: str, doc_id: str, manifest: EnrichmentManifest | None) -> None:
    """Store what Phase 4 managed to do to this document. NEVER raises.

    None IS A NO-OP AND LEAVES ANY PREVIOUS MANIFEST ALONE, which needs stating because it is the
    opposite of what the other three writes do with nothing. They are idempotent re-derivations
    of text that is serving. This is a claim about ONE RUN, and `None` means that run could not
    say what it did — so overwriting a readable manifest with nothing would replace a known
    result with an unknown one. A manifest that no longer matches the document's `content_hash`
    is refused by the READER (:meth:`EnrichmentManifest.matches`), which is where staleness can
    actually be detected.

    FAIL-SAFE like every other derived write here. A completeness record that could fail an
    ingest would make a disclosure dimension more fragile than the thing it discloses."""
    if manifest is None:
        return
    try:
        stored = await persist_enrichment_manifest(tenant_id, doc_id, manifest)
        if not stored:
            _log.warning(
                "enrichment_manifest_not_stored",
                tenant_id=tenant_id,
                doc_id=doc_id,
                detail="no documents row matched, so completeness will read as UNKNOWN",
            )
    except Exception as exc:
        _log.warning(
            "enrichment_manifest_write_failed",
            tenant_id=tenant_id,
            doc_id=doc_id,
            error=str(exc)[:160],
        )


async def _persist_layout_structure(tenant_id: str, doc_id: str, tree: LayoutTree | None) -> None:
    """REPLACE this document's stored structure with ``tree`` (RAG v2 ``sections`` + ``blocks``).

    ``tree`` comes from :func:`_layout_tree`, which owns the flag gate and the parse; None (flag
    off, no overlay, or an overlay this build cannot read) makes this a no-op.

    THE PLAIN-TEXT TREE IS PERSISTED TOO, not only the markup one. A document whose extractor
    emitted no markup still has a tree: ``parse_layout`` falls back to
    ``sections.detect_sections`` over the text. Persisting it is the whole difference between
    structure the later phases can rely on and structure they must special-case, because only the
    Chandra engines emit markup — every native-PDF, pypdf, Tesseract, Paddle and Office document
    on this deployment comes through a plain-text extractor. ``LayoutTree.source`` says which kind
    is in hand, and the ROWS stay honest about it without a column of their own: a text-derived tree
    has sections and NO blocks (the text layer has no block types), so "this document has zero
    blocks" IS the discriminator.

    CORRECTED BY PHASE 5 (2026-10-19): the paragraph above used to argue that restricting these
    tables to markup documents "would leave Phase 5's section-level chunking blind to most of the
    corpus". Phase 5 landed and that is exactly what it is — layout-aware chunking needs BLOCK
    text, and a text-derived tree has none, so a plain-text document keeps clause chunking (see
    :func:`_chunkable_layout`). Persisting the text tree is still right, just not for that reason:
    a section row carries the breadcrumb, depth and parent link that the Phase 7-9 query side needs
    whatever produced it, and it is the only record of a plain-text document's structure.

    AN UNPARSEABLE OVERLAY IS LOGGED AND SKIPPED, never raised (the log is in
    :func:`_layout_tree`, which is where the parse now lives). ``parse_layout`` refuses an
    unrecognised ``schema_version`` so a stored artifact from another build is detectable rather
    than silently misparsed — but that is a statement about the ARTIFACT, and an ingest of a
    perfectly good document must not 500 because the structure tables could not be populated. This
    slice persists structure only; nothing retrieves it yet, so skipping costs the document its
    tree until it is re-indexed by a build that understands the overlay, and raising would cost the
    document its ingest. The previous tree is left EXACTLY as it was in that case (the replace
    never runs), which is the same position the overlay write takes when the flag is off.

    No-op when the flag is off: nothing is parsed, nothing is written, and any tree already stored
    is left alone — the only consumers are gated on the same flag.

    PII, AND THE ONE THING THE NEXT PHASE MUST NOT INHERIT BLINDLY. The markup in the overlay is
    captured from the UNMASKED engine output (``DocIntelligenceService.extract`` says so, and
    ``_persist_layout_overlay`` states the compliance position for it), so for a markup document
    ``blocks.text`` and the text behind ``sections.text_hash`` hold text as the extractor read it —
    PII included. That is the same position the stored overlay and the raw object already take:
    tenant-scoped, never embedded, never indexed, nothing retrieves it, and purged by the SAME
    ``delete_document`` the raw bytes are (verified — it deletes ``blocks``/``sections`` explicitly,
    with ``map_cache`` before them). It cannot be fixed by masking here either: every id in the tree
    is already derived from the unmasked text by the Phase-1 helpers, so re-masking would make the
    stored text disagree with ``text_hash`` and ``block_id``.
    What this does NOT license is Phase 5 chunking a block's text straight into the index. The
    platform's invariant is that only de-identified text is chunked/embedded/indexed, and the text
    in these rows has NOT been through the masker — only ``ExtractedDoc.text`` has. The plain-text
    fallback path is already masked (it parses the indexed text), so this applies to the markup
    path alone.

    PHASE 5 HONOURS THAT, and this is where to check it:
    :class:`rag_core.layout_chunking.LayoutAwareChunker` takes the PII masker as a REQUIRED
    constructor argument — no default, no identity masker — and runs it over every string that
    becomes ``Chunk.text`` / ``Chunk.embed_text`` / ``metadata["section"]``, in the single method
    that builds a chunk. The ids are deliberately NOT masked: a chunk's ``section_id`` /
    ``parent_id`` are this tree's own ids, so they still resolve to the rows written below, while
    every stored or embedded STRING is de-identified.
    ``services/control-plane/tests/test_layout_chunking_pipeline.py`` proves a row chunk derived
    from a block holding an identifier reaches the index masked."""
    if tree is None:
        return
    sections, blocks = await persist_document_structure(tenant_id, tree)
    _log.info(
        "layout_structure_stored",
        tenant_id=tenant_id,
        doc_id=doc_id,
        source=tree.source,
        sections=sections,
        blocks=blocks,
        removed_running_text=len(tree.removed_running_text),
    )


async def _quarantine_suspect(
    doc_final: DocumentEnvelope, extracted: Any, *, actor: str, trace_id: str
) -> IngestOutcome:
    """Quarantine a document whose text reads as INSTRUCTIONS (indirect prompt injection): NOT
    indexed (so it can never be retrieved into a prompt and hijack an answer), the doc + raw bytes
    kept for a reviewer, and a GUARDRAIL event raised for the SOC/SIEM."""
    await wiring.audit_service().record(
        tenant_id=doc_final.tenant_id,
        trace_id=trace_id,
        action=AuditAction.GUARDRAIL,
        actor=actor,
        payload={
            "stage": "ingest",
            "reason": "prompt_injection_suspected",
            "hits": extracted.provenance.get("sanitization_hits", "0"),
            "doc_id": doc_final.doc_id,
        },
    )
    _log.warning("ingest_quarantined_injection", doc_id=doc_final.doc_id, tenant_id=doc_final.tenant_id)
    quarantined = doc_final.model_copy(update={"metadata": {**doc_final.metadata, "quarantined": "true"}})
    await get_catalog().register_document(quarantined)
    return IngestOutcome(doc_final.doc_id, 0, extracted.pii_masked, trace_id, outcome="quarantined")


async def _apply_pii_classification(
    doc_final: DocumentEnvelope,
    extracted: Any,
    *,
    actor: str,
    trace_id: str,
    audit: AuditService,
    object_uri: str,
    clearance: int,
    pii_policy: str,
) -> tuple[DocumentEnvelope, IngestOutcome | None]:
    """Enforce the PII-driven classification floor (fail-closed). If the detected structured PII
    implies a HIGHER classification than requested, either RAISE the document to that floor
    (returns the reclassified doc, ``None``) or BLOCK the ingest (returns the doc unchanged plus a
    ``blocked_pii`` outcome to return). A floor ABOVE the ingestor's ``clearance`` is ALWAYS blocked
    even in ``raise`` mode — the ingestor cannot store content it isn't cleared to read. ``off`` or
    no PII above the requested level -> unchanged, ``None``."""
    if pii_policy == "off":
        return doc_final, None
    # Guardrail toggle: an org/KB may disable the PII classification floor via the Guardrails page.
    # Default ON — a lookup failure keeps the safe default (enforce). doc_final.tenant_id is the kb id.
    try:
        resolved = await guardrails_policy.effective_map(get_catalog(), doc_final.tenant_id)
        if resolved.get("pii_classification_floor") is False:
            return doc_final, None
    except Exception as exc:
        _log.warning("guardrail_pii_floor_lookup_failed", kb_id=doc_final.tenant_id, error=str(exc)[:160])
    floor = pii_classification.pii_classification_floor(extracted.pii_counts)
    if floor <= doc_final.classification:
        return doc_final, None
    triggers = pii_classification.pii_triggers(extracted.pii_counts, floor)
    over_clearance = floor > clearance
    # The company-wide shared KB (Federated RAG Tier C) is readable by EVERY tenant and is Public-only
    # by construction, so a PII floor there can never be satisfied by RAISING (that would make it
    # non-Public and silently broadcast PII company-wide) — it must ALWAYS block.
    shared_public_only = doc_final.tenant_id == SHARED_KB_ID
    blocked = pii_policy == "block" or over_clearance or shared_public_only
    await audit.record(
        tenant_id=doc_final.tenant_id,
        trace_id=trace_id,
        action=AuditAction.CLASSIFICATION,
        actor=actor,
        payload={
            "doc_id": doc_final.doc_id,
            "requested": doc_final.classification,
            "floor": floor,
            "triggers": triggers,
            "result": "blocked" if blocked else "raised",
            "over_clearance": over_clearance,
        },
    )
    if not blocked:
        _log.info(
            "ingest_classification_raised",
            doc_id=doc_final.doc_id,
            requested=doc_final.classification,
            floor=floor,
            triggers=triggers,
        )
        return doc_final.model_copy(update={"classification": floor}), None
    if shared_public_only:
        reason = (
            f"document contains {', '.join(triggers)} (classification {floor}); the company-wide shared KB "
            "is Public-only, so PII-bearing documents can't be shared there"
        )
    else:
        reason = (
            f"document contains {', '.join(triggers)} (classification {floor})"
            + (" — above your clearance" if over_clearance else " — above the requested level")
            + f"; re-ingest at classification >= {floor}"
            + (" via a member cleared for it" if over_clearance else "")
        )
    _log.warning("ingest_blocked_pii", doc_id=doc_final.doc_id, floor=floor, triggers=triggers, over=over_clearance)
    # Billable INGEST (raw bytes retained for a reviewer, like quarantine), then stop — not indexed.
    await audit.record(
        tenant_id=doc_final.tenant_id,
        trace_id=trace_id,
        action=AuditAction.INGEST,
        actor=actor,
        payload={"doc_id": doc_final.doc_id, "object_uri": object_uri, "blocked_pii": True},
    )
    blocked_doc = doc_final.model_copy(update={"metadata": {**doc_final.metadata, "pii_blocked": "true"}})
    await get_catalog().register_document(blocked_doc)
    outcome = IngestOutcome(
        doc_final.doc_id, 0, extracted.pii_masked, trace_id, outcome="blocked_pii", classification=floor, reason=reason
    )
    return doc_final, outcome


async def ingest_document(  # noqa: PLR0915, PLR0912
    doc: DocumentEnvelope,
    *,
    actor: str,
    strategy: ChunkStrategy | None = None,
    clearance: int = 0,
    pii_policy: str = "off",
) -> IngestOutcome:
    """Ingest -> extract/mask -> chunk/embed/index (TDD §5.1). Fully tenant-scoped.

    FR-A8/C9: the raw bytes are content-hashed. Re-ingesting UNCHANGED content for the same
    doc skips the expensive extract/embed/index (delta); identical content under a different
    doc_id is flagged as a near-duplicate."""
    trace_id = str(uuid.uuid4())
    audit: AuditService = wiring.audit_service()

    # Snapshot existing docs BEFORE this ingest overwrites the row (for delta + dup checks).
    existing = {d.doc_id: d for d in await get_catalog().list_documents(doc.tenant_id)}
    prior_hash = existing[doc.doc_id].content_hash if doc.doc_id in existing else None

    ingested = await wiring.ingestion_service().ingest(doc)
    content_hash = hashlib.sha256(ingested.raw).hexdigest()
    await get_catalog().register_document(ingested.doc.model_copy(update={"content_hash": content_hash}))
    ingest_progress.reset(doc.tenant_id, doc.doc_id)
    ingest_progress.mark(doc.tenant_id, doc.doc_id, "ingest", "object")  # live DAG: raw bytes written + registered

    # The INGEST audit event is what billing METERS (FR-H4). It is recorded ONLY once the ingest
    # actually completes — a NEW upload whose extraction/indexing then fails is rolled back and
    # must NOT be left as a billable event, and an UPDATE whose new version fails leaves the prior
    # version serving without a second charge. The delta-skip path (a genuinely completed ingest)
    # records it below before returning; the success path records it after indexing succeeds.

    # FR-A8 delta: same doc, identical bytes -> nothing to re-index.
    if prior_hash is not None and prior_hash == content_hash:
        await audit.record(
            tenant_id=doc.tenant_id,
            trace_id=trace_id,
            action=AuditAction.INGEST,
            actor=actor,
            payload={"doc_id": doc.doc_id, "object_uri": ingested.object_uri, "content_hash": content_hash[:12]},
        )
        existing_chunks = len(await get_catalog().document_chunks(doc.tenant_id, doc.doc_id))
        _log.info("ingest_unchanged_skipped", doc_id=doc.doc_id, tenant_id=doc.tenant_id)
        return IngestOutcome(doc.doc_id, existing_chunks, 0, trace_id, outcome="unchanged")

    # FR-C9 near-duplicate: identical bytes already ingested under a DIFFERENT doc_id.
    duplicate_of = next(
        (did for did, d in existing.items() if did != doc.doc_id and d.content_hash == content_hash), None
    )

    # WRITE, THEN SWAP (mirrors ``reindex_document``). An update REPLACES the document, but the
    # previous version's chunks are NOT purged first: the old version keeps serving until the new
    # one is durably indexed, so a failure anywhere in extract -> chunk -> embed -> index leaves
    # the prior version fully intact. Purging first meant a failed re-ingest wiped a previously
    # good, retrievable document — silent data loss. Capture the old chunk ids WITHOUT deleting;
    # content-derived ids (P44) make the swap clean — surviving text keeps its id and is
    # re-upserted, so only genuinely-gone ids are removed AFTER the new version is written.
    is_update = doc.doc_id in existing
    old_chunk_ids = (
        [c.chunk_id for c in await get_catalog().document_chunks(doc.tenant_id, doc.doc_id)] if is_update else []
    )

    try:
        # EVERY document goes through the ingestion OCR check in doc-intelligence: image files,
        # scanned pages and embedded images are OCR'd, and with multimodal on an image's vision
        # caption is ADDED to its OCR text. This used to branch images away to a caption that
        # REPLACED OCR — measured live, a JPEG e-ticket was indexed as 32 characters of caption
        # while OCR (4,544 characters from the same file) was never called.
        # The service is held rather than called inline because RAG v2 Phase 5 needs its MASKER
        # too: the layout tree it chunks is captured pre-masking, so the chunker must de-identify
        # through the SAME masker that produced `extracted.text` (see `DocIntelligenceService.
        # masker`). A masker rebuilt from config beside it could differ in strategy, salt or
        # entity set, and then one document's structure-derived chunks would be masked differently
        # from its own indexed text.
        doc_intel = await wiring.doc_intel_service_for(doc.tenant_id)
        try:
            extracted = await doc_intel.extract(ingested.doc, ingested.raw)
        except PlatformError:
            raise  # already a clean, typed status (validation / config / connector)
        except Exception as exc:
            # A corrupt, encrypted, truncated or unsupported file makes the extractor/OCR throw a
            # non-domain error — map it to a clean 422 instead of a bare 500. The outer handler's
            # rollback still runs (this propagates into it), so a failed upload never lingers.
            raise ValidationError(
                "could not extract text from this document — it may be corrupt, password-protected, "
                "or an unsupported format"
            ) from exc
        # RAG v2 (flag-gated, no-op by default): KEEP the extraction engine's layout markup as
        # a stored artifact beside the raw bytes. Placed here, on the raw-bytes' own terms: it is
        # captured from the same unmasked engine output and shares their retention and purge
        # paths, so a quarantined or PII-blocked document (both of which keep the raw bytes for a
        # reviewer and return below) keeps its overlay too. Nothing after this line reads it —
        # `text`, `page_map` and every stage downstream are untouched. See
        # _persist_layout_overlay for the compliance position.
        await _persist_layout_overlay(doc.tenant_id, doc.doc_id, extracted, content_hash=content_hash)
        # FR-B7/B2: auto-derive filterable metadata (tags) + structured fields from the
        # extracted text (before chunking, so they flow onto chunks). Merged UNDER
        # source-provided metadata (source wins).
        doc_final = ingested.doc.model_copy(update={"content_hash": content_hash})
        derived = await _derive_tags(doc.tenant_id, extracted)
        # Extraction provenance is stamped LAST, over source-provided tags: it is a
        # platform-observed fact about how this text was produced, so a source must not be
        # able to claim it was extracted cleanly. Always present -> always re-registered.
        doc_final = doc_final.model_copy(
            update={"metadata": {**derived, **ingested.doc.metadata, **extracted.provenance}}
        )
        _log.info("doc_enriched", doc_id=doc.doc_id, keys=list(derived), **extracted.provenance)
        ingest_progress.mark(doc.tenant_id, doc.doc_id, "doc_intel", "sanitize")  # live DAG: extracted + scanned
        # PII-driven classification floor (fail-closed): raise the classification to the floor the
        # detected structured PII implies, or block. Runs BEFORE chunking so a raised level flows
        # onto every chunk. See _apply_pii_classification.
        doc_final, pii_block = await _apply_pii_classification(
            doc_final,
            extracted,
            actor=actor,
            trace_id=trace_id,
            audit=audit,
            object_uri=ingested.object_uri,
            clearance=clearance,
            pii_policy=pii_policy,
        )
        if pii_block is not None:
            return pii_block
        # Malicious-document defense (indirect prompt injection): text that reads as INSTRUCTIONS
        # is NOT indexed, so it can never be retrieved into a prompt and hijack an answer. The doc
        # + raw bytes stay for a reviewer; a GUARDRAIL event alerts the SOC/SIEM. Off = never set.
        if extracted.provenance.get("sanitization_risk") == "suspect":
            # Quarantine is a COMPLETED ingest (doc + raw bytes are kept for a reviewer), so it is
            # a real billable INGEST event — recorded here, on the durable path, before returning.
            await audit.record(
                tenant_id=doc.tenant_id,
                trace_id=trace_id,
                action=AuditAction.INGEST,
                actor=actor,
                payload={"doc_id": doc.doc_id, "object_uri": ingested.object_uri, "content_hash": content_hash[:12]},
            )
            ingest_progress.mark(doc.tenant_id, doc.doc_id, "quarantine")  # live DAG: suspect -> quarantined, stop
            return await _quarantine_suspect(doc_final, extracted, actor=actor, trace_id=trace_id)
        pc_child, pc_parent = await _parent_child_sizing(doc.tenant_id)  # FR-C2: per-KB parent/child sizing
        # Slice 6, Phase C: pass a molecule normalizer ONLY when the triple gate holds (master flag +
        # per-KB gate + chemistry profile), so a chemistry chunk gains canonical SMILES/InChIKey exact
        # tokens. None otherwise ⇒ index_document is byte-identical.
        chem_normalizer = get_molecule_normalizer() if await _chem_normalize_on(doc.tenant_id) else None
        # RAG v2 Phase 5: parse the structure tree ONCE, here, and use it twice — to CHUNK (below)
        # and to PERSIST (further down, where it already was). It is parsed before indexing because
        # the chunker needs it; it is still STORED after indexing, for the reason stated at that
        # call. None when the flag is off, so chunking is byte-identical to today.
        tree = _layout_tree(doc.doc_id, extracted, text=extracted.text)
        # RAG v2 Phase 3: the document's profile card, BEFORE indexing because the card decides
        # the chunk size for this doc_type and rides in front of every unit's `embed_text`. It is
        # STORED after indexing (below), not here. None when the flag is off, so chunking is
        # byte-identical to Phase 5.
        profile = await _document_profile(doc.tenant_id, doc_final, extracted, tree, mask=_mask_with(doc_intel))
        # RAG v2 Phase 4, before indexing for the same reason the card is: the section summaries
        # and the propositions BECOME chunks, so they have to exist before the embed/index
        # fan-out. Stored after indexing (below), and after the structure tree, which replaces
        # the very rows this updates. Empty when the flag is off ⇒ chunking is byte-identical.
        enrichment, manifest = await _section_enrichment(
            doc.tenant_id, doc_final, tree, profile, mask=_mask_with(doc_intel)
        )
        indexed = await (await wiring.indexing_service_for(doc.tenant_id)).index_document(
            doc_final,
            extracted.text,
            page_map=extracted.page_map,
            strategy=strategy,
            modality=extracted.modality,
            ocr_confidence=extracted.min_confidence,
            child_max_chars=pc_child,
            parent_max_chars=pc_parent,
            chem_normalizer=chem_normalizer,
            layout=_chunkable_layout(tree, text_pages=_text_pages(extracted.page_map)),
            mask=_mask_with(doc_intel),
            # An UPDATE to an existing document: A8 already short-circuits identical raw BYTES, so
            # what reaches here is a genuinely changed file — of which, typically, most chunks are
            # still word for word what they were (a corrigendum adds a clause; a policy edits one
            # paragraph). `existing[...]` is the row as it was BEFORE this ingest overwrote it,
            # snapshotted above for exactly this kind of comparison. A first ingest has no previous
            # version, so `_may_reuse_vectors` says no.
            reuse_unchanged=_may_reuse_vectors(existing.get(doc.doc_id)),
            profile=profile,
            enrichment=enrichment,
        )
        # Recorded like extraction provenance: a platform-observed fact about how this text
        # was indexed, so a re-index can reproduce it rather than guess.
        doc_final = doc_final.model_copy(
            update={
                "metadata": {
                    **doc_final.metadata,
                    **_strategy_metadata(indexed),
                }
            }
        )
        await get_catalog().register_document(doc_final)
        # RAG v2 (flag-gated, no-op by default): the STRUCTURE TREE parsed out of the overlay
        # captured above. Placed HERE, after the new chunks are durably indexed and the row
        # re-registered — NOT up beside the overlay write — for the reason `reindex_document`'s
        # call states: `sections`/`blocks` describe the text that is SERVING, and on an update
        # write-then-swap leaves the previous version serving if anything above raises. Written
        # before indexing, an update whose index failed would leave a tree describing a
        # re-extraction no reader can reach, joined by section id to chunks of different text.
        # The overlay is deliberately NOT moved down here with it: that one is an extraction
        # artifact in the raw bytes' class (captured pre-masking, kept even for a quarantined or
        # PII-blocked document, which both return before this line), while this is derived state
        # about the index. A NEW document that fails is rolled back by the handler below, which
        # purges these rows with the rest of the catalog row. The Phase-3 card and the Phase-4
        # section enrichment derived above are written in the same call and for the same reason:
        # `documents.doc_type`/`purpose`, the `facets`/`doc_relations` rows and the `sections`
        # summaries all describe the text that is SERVING. This is also what makes the Phase-6
        # relation seeding live at BOTH grains — `doc_relations` and `sections.amends` have
        # producers from here on.
        # `content_hash`, NOT `doc.content_hash`: the hash is computed from the fetched bytes in
        # this function and set on `doc_final`, while the inbound `doc` envelope carries none. The
        # first version of this passed `doc.content_hash` and stored every facet row with an EMPTY
        # revision -- the pinning would have been silently inert, which the reindex acceptance
        # test is what caught.
        await _persist_v2_derived(doc.tenant_id, doc.doc_id, tree, profile, enrichment, manifest, content_hash)
        # live DAG: chunk -> embed -> the three durable stores all completed inside index_document
        ingest_progress.mark(
            doc.tenant_id, doc.doc_id, "chunk", "contextualize", "embed", "vector", "sparse", "catalog", "index"
        )
        await _record_duplicate_chunks(doc.tenant_id, doc.doc_id)
        # Cross-modal (additive/OFF): also index the image's CLIP vector so a text query can
        # retrieve it by VISUAL similarity, not only its caption. No-op unless multimodal +
        # vision_recall are on AND this is an image; keyed to the same caption chunk ids.
        if get_settings().multimodal_enabled and multimodal.is_image(ingested.doc):
            # FR-C2: children only — a parent must never receive an image vector (would make it citable).
            await multimodal.index_image_vectors(doc.tenant_id, ingested.raw, indexed.child_chunk_ids)
        # Diagram structure (additive/OFF): independent of the multimodal caption/OCR choice
        # above — either way the image's OCR/caption text already indexed unchanged; this ALSO
        # reads the raw bytes for the diagram's arrow structure. See _index_diagram_graph.
        if multimodal.is_image(ingested.doc):
            await _index_diagram_graph(doc.tenant_id, doc.doc_id, ingested.raw, indexed.child_chunk_ids)
    except Exception:
        # Extraction/indexing failed AFTER the raw bytes + catalog row were written. For a NEW
        # document (not previously present) roll those back so a rejected upload never lingers as a
        # half-ingested document. For an UPDATE the prior version is left COMPLETELY intact — its
        # chunks were never purged (write-then-swap), so it keeps serving; deleting the catalog
        # row or raw object here would destroy a previously-good, retrievable document. Either
        # way no INGEST event was recorded, so a failed ingest is never left as a billable event.
        # BEST EFFORT, and deliberately so: this runs inside `except`, and a rollback that raises
        # replaces the real diagnosis (a clean 422 "could not extract text") with a 500 about the
        # rollback. The original is re-raised either way; a rollback that could not finish is logged
        # at ERROR with everything an operator needs to reconcile by hand, which is strictly better
        # than masking why the ingest failed.
        try:
            if not is_update:
                # The chunk ids MATTER. `index_document` may already have written vectors and BM25
                # documents, and the catalog row is the only thing that can enumerate them — so
                # discarding these ids left the chunks in Qdrant and OpenSearch with no row to find
                # them by, permanently unreachable by `delete_document` AND by DPDP erasure. The
                # window is real: `register_document`, `_persist_layout_structure` and
                # `_record_duplicate_chunks` all run after the index write, and the structure write
                # is a Postgres delete+insert under an advisory lock that can time out.
                chunk_ids = await get_catalog().delete_document(doc.tenant_id, doc.doc_id)
                await _purge_chunk_artifacts(doc.tenant_id, chunk_ids)
                await get_object_store().delete(doc.tenant_id, raw_object_key(doc.doc_id))
                # The layout overlay is the raw bytes' sibling artifact (see
                # _persist_layout_overlay) and goes wherever they go. Without this a failed ingest
                # orphans it: the catalog row that pointed at it has just been deleted, so no purge
                # path would ever enumerate it. Unconditional, like the raw delete: object-store
                # delete is a no-op for a key that was never written, and the flag may have been on
                # when the attempt started.
                await get_object_store().delete(doc.tenant_id, layout_object_key(doc.doc_id))
                if chunk_ids:
                    # Communities cite chunk ids that are now gone, and a concurrent query could
                    # have cached an answer from the chunks this attempt briefly had serving.
                    await _refresh_communities(doc.tenant_id)
                    semantic_cache.invalidate(doc.tenant_id)
            else:
                # Re-register the PRIOR version's row (the new version's metadata was just written
                # over it above) so the update leaves the document exactly as it was before this
                # attempt.
                await get_catalog().register_document(existing[doc.doc_id])
        except Exception as rollback_exc:
            _log.error(
                "ingest_rollback_incomplete",
                tenant_id=doc.tenant_id,
                doc_id=doc.doc_id,
                is_update=is_update,
                error=str(rollback_exc)[:200],
            )
        raise
    # THE SWAP: the new version is now durably indexed, so purge only the old chunk ids that did
    # NOT survive (old minus fresh) — mirrors ``reindex_document``. Reached only after success, so
    # there is no window in which the document has neither version. The fresh ids come from the
    # INDEXER, not the catalog (nothing was purged first, so the catalog holds old + new together).
    fresh_ids = set(indexed.chunk_ids)
    stale = [cid for cid in old_chunk_ids if cid not in fresh_ids]
    if stale:
        await get_catalog().delete_chunks(doc.tenant_id, stale)
        await _purge_chunk_artifacts(doc.tenant_id, stale)
    replaced_chunks = len(stale)
    # Persist the real per-stage timeline (offsets in ms from ingest start) into the DURABLE audit
    # payload, so the ingestion DAG can show honest per-stage durations for ANY document later — not
    # only while the in-memory live-progress marks survive their short TTL.
    _marks = ingest_progress.completed_stages(doc.tenant_id, doc.doc_id)
    _base = min(_marks.values()) if _marks else 0.0
    stage_ms = {stage: max(0, round((at - _base) * 1000)) for stage, at in _marks.items()}
    # The ingest completed: record the billable INGEST event now (never for a failed ingest), then
    # the INDEX event.
    await audit.record(
        tenant_id=doc.tenant_id,
        trace_id=trace_id,
        action=AuditAction.INGEST,
        actor=actor,
        payload={"doc_id": doc.doc_id, "object_uri": ingested.object_uri, "content_hash": content_hash[:12]},
    )
    await audit.record(
        tenant_id=doc.tenant_id,
        trace_id=trace_id,
        action=AuditAction.INDEX,
        actor=actor,
        payload={
            "doc_id": doc.doc_id,
            "chunks": indexed.chunk_count,
            "pii_masked": extracted.pii_masked,
            # Replacing a document DESTROYS the previous version's stale chunks — attributable, like
            # every other removal (the reindex path records the same field).
            "replaced_chunks": replaced_chunks,
            "stage_ms": stage_ms,
            # Per-stage signal for the DAG so doc-intelligence and the injection scan show real data
            # instead of an empty cell. Reaching INDEX proves the scan passed (a suspect verdict
            # returns via _quarantine_suspect), so injection is provably clean.
            "modality": getattr(extracted.modality, "value", str(extracted.modality)),
            "classification": getattr(doc_final.classification, "value", str(doc_final.classification)),
            "injection": "clean",
        },
    )
    semantic_cache.invalidate(doc.tenant_id)  # the KB's knowledge changed
    # GraphRAG (no-op unless enabled); is_image tells it to filter OCR/caption junk out of
    # the ordinary co-occurrence path — text/PDF docs pass False and are unaffected.
    graph_entities, graph_edges = await _index_graph(
        doc.tenant_id, doc.doc_id, is_image=multimodal.is_image(ingested.doc)
    )
    ingest_progress.mark(doc.tenant_id, doc.doc_id, "graphrag")  # live DAG: GraphRAG done -> pipeline complete
    if get_settings().graphrag_enabled:
        # Record the graph stage so the ingestion DAG can PROVE it ran (not infer it). Gated on the
        # flag: a graph-disabled KB no-ops _index_graph, so claiming the stage ran would be false.
        _gmarks = ingest_progress.completed_stages(doc.tenant_id, doc.doc_id)
        _gbase = min(_gmarks.values()) if _gmarks else 0.0
        _graph_at = _gmarks.get("graphrag")
        _index_at = _gmarks.get("index")
        await audit.record(
            tenant_id=doc.tenant_id,
            trace_id=trace_id,
            action=AuditAction.GRAPH,
            actor=actor,
            payload={
                "doc_id": doc.doc_id,
                "chunks": indexed.chunk_count,
                "entities": graph_entities,
                "edges": graph_edges,
                "graph_at_ms": max(0, round((_graph_at - _gbase) * 1000)) if _graph_at else 0,
                "graph_ms": max(0, round((_graph_at - _index_at) * 1000)) if _graph_at and _index_at else 0,
            },
        )
    await _refresh_communities(doc.tenant_id)  # re-ingest moves chunk ids too (P48)
    kind = "updated" if prior_hash is not None else "created"
    if duplicate_of:
        _log.info("ingest_near_duplicate", doc_id=doc.doc_id, duplicate_of=duplicate_of, tenant_id=doc.tenant_id)
    _log.info("ingest_pipeline_done", doc_id=doc.doc_id, chunks=indexed.chunk_count, outcome=kind, trace_id=trace_id)
    return IngestOutcome(
        doc.doc_id,
        indexed.chunk_count,
        extracted.pii_masked,
        trace_id,
        outcome=kind,
        duplicate_of=duplicate_of,
        classification=doc_final.classification,
    )


def _may_reuse_vectors(previous: DocumentEnvelope | None) -> bool:
    """Whether this document's already-stored vectors may be kept for text that has not changed
    (`IndexingService._reusable`, plan Phase 6 `chunk_hash` idempotency).

    The whole question is "could the vectors have come from a different pipeline than the one
    running now", and `pipeline_fingerprint` is exactly the answer: it folds in the embed model and
    dimension, the chunk sizing, contextual retrieval and the readers that produce the text, and
    EXCLUDES query-time settings that cannot change an index. So a document whose recorded stamp
    still matches the running fingerprint has vectors this deployment would produce again, and
    re-embedding its unchanged chunks buys nothing.

    Fails CLOSED in every uncertain case — a document with no stamp (ingested before the stamp
    existed), a first-time ingest, a stamp that differs for any reason. The cost of a false no is
    one re-embed; the cost of a false yes is a vector that silently disagrees with its text.

    Used on the INGEST path only, and not on reindex, which is a deliberate narrowing rather than
    an omission. The fingerprint answers "could a different pipeline have produced these vectors",
    but not "are they still there": a collection recreated or wiped out of band keeps the
    fingerprint and loses the vectors. On ingest that does not matter, because the previous
    version's chunks are present by construction — the write-then-swap has not purged anything yet
    and the catalog row proves the vector and BM25 writes completed. On REINDEX it matters a great
    deal, because reindex is the path an operator reaches for to repair exactly that situation, and
    a repair that quietly declined to rebuild would be invisible. Reusing there needs the stores
    asked whether they still hold each id, which is a bigger change than this one and belongs with
    its own measurement.
    """
    if previous is None:
        return False
    return previous.metadata.get(META_PIPELINE_VERSION) == pipeline_fingerprint(get_settings())


@dataclass(frozen=True, slots=True)
class ReenrichOutcome:
    """What a Phase 3-4 re-run did to one document (RAG v2 Phase 12).

    ``status`` is the whole result and the reason this is not a bool:

    * ``reenriched`` — a new card and/or new section enrichment was written.
    * ``no_structure`` — the document has no structure tree to enrich (no overlay, or an
      extractor this build cannot parse). Not an error; most of a real corpus is in this state.
    * ``structure_changed`` — REFUSED, see :func:`reenrich_document`.
    * ``no_bytes`` — the raw object is gone, so nothing can be re-derived from it.
    """

    doc_id: str
    status: str
    sections: int = 0
    enriched: int = 0
    profiled: bool = False


async def reenrich_document(tenant_id: str, doc_id: str, *, actor: str) -> ReenrichOutcome:
    """Re-run Phases 3 and 4 for one document, and NOTHING else (RAG v2 Phase 12).

    The plan's own words: "when prompts or Gemma version change, bump ``profile_version`` /
    ``model_version`` and provide a CLI to re-run Phases 3-4 only." This is what the CLI
    (``scripts/reenrich.py``) calls. It does not chunk, embed or index — which is the point,
    because a prompt change does not change the corpus and a full ``reindex_document`` over a
    large KB to pick up a better profile prompt is an expensive way to get one.

    IT REFUSES WHEN THE STRUCTURE MOVED, and that guard is the reason this is a pipeline
    function rather than three catalog writes in a script. Phases 3 and 4 both take the
    structure tree as input, so re-running them re-derives it — and a tree whose section IDs
    differ from the stored ones cannot be persisted here: the INDEXED chunks still carry the old
    ``section_id``, so Phase 8 would select sections whose chunks it cannot find and Phase 9
    would map over nothing, with every count reading plausibly. A changed tree means the
    extractor or the layout parser changed, and that is a REINDEX, not a re-enrichment. So it is
    detected, refused, and named — the operator runs ``reindex_document`` instead.

    THE DERIVED CHUNKS GO STALE and this does not fix them. A section's Phase-4 summary is also
    indexed as a ``section``-level chunk (and its propositions as ``proposition``-level ones), so
    after a re-enrichment the catalog holds the new summary and the INDEX still holds the old
    one. The consequence is bounded and worth stating plainly: Phase 8 selects on the catalog's
    ``sections`` rows, so selection uses the new summaries immediately; Phase 9 maps over the
    document's own text and deliberately excludes the derived levels
    (:func:`_section_chunk_text`), so extraction is unaffected; what remains stale is the
    summary as a RETRIEVAL target. A reindex clears it when the operator wants it cleared.

    NEVER RAISES for one document's sake. A corpus-wide run must not stop on the one document
    whose model call failed, so a failure is a status, and the CLI reports the tally."""
    docs = {d.doc_id: d for d in await get_catalog().list_documents(tenant_id)}
    doc = docs.get(doc_id)
    if doc is None:
        return ReenrichOutcome(doc_id=doc_id, status="no_bytes")
    try:
        raw = await get_object_store().get(tenant_id, raw_object_key(doc_id))
    except Exception as exc:
        _log.warning("reenrich_no_bytes", tenant_id=tenant_id, doc_id=doc_id, error=str(exc)[:120])
        return ReenrichOutcome(doc_id=doc_id, status="no_bytes")

    doc_intel = wiring.doc_intel_service()
    extracted = await doc_intel.extract(doc, raw)
    tree = _layout_tree(doc_id, extracted, text=extracted.text)
    if tree is None:
        return ReenrichOutcome(doc_id=doc_id, status="no_structure")

    stored = {row.section_id for row in await document_sections(tenant_id, [doc_id])}
    rederived = {section.section_id for section in _walk_sections(tree.sections)}
    if stored and stored != rederived:
        _log.warning(
            "reenrich_structure_changed",
            tenant_id=tenant_id,
            doc_id=doc_id,
            stored=len(stored),
            rederived=len(rederived),
        )
        return ReenrichOutcome(doc_id=doc_id, status="structure_changed", sections=len(rederived))

    mask = _mask_with(doc_intel)
    profile = await _document_profile(tenant_id, doc, extracted, tree, mask=mask)
    enrichment, manifest = await _section_enrichment(tenant_id, doc, tree, profile, mask=mask)
    # The structure is RE-PERSISTED even though its ids are unchanged: `replace_document_structure`
    # rewrites the breadcrumbs, pages and text hashes too, and an enrichment written against a
    # row whose text hash disagrees with the tree it came from is the drift this would introduce.
    # It is also what purges `map_cache` for these sections, which a new summary should.
    await _persist_v2_derived(tenant_id, doc_id, tree, profile, enrichment, manifest, doc.content_hash or "")
    _log.info(
        "reenriched",
        tenant_id=tenant_id,
        doc_id=doc_id,
        actor=actor,
        sections=len(rederived),
        enriched=len(enrichment),
        profiled=profile is not None,
        profile_version=PROFILE_VERSION,
    )
    return ReenrichOutcome(
        doc_id=doc_id,
        status="reenriched",
        sections=len(rederived),
        enriched=len(enrichment),
        profiled=profile is not None,
    )


async def reindex_document(
    tenant_id: str, doc_id: str, *, actor: str, strategy: ChunkStrategy | None = None
) -> IngestOutcome:
    """Re-index a document from its stored raw bytes (FR-C8 replayable indices).

    Re-runs extract → chunk → embed → index, then removes whatever the new version no longer
    contains — how a tenant adopts a new chunking strategy or contextual retrieval without
    re-uploading. Write-then-swap, so a failure leaves the previous version serving."""
    docs = {d.doc_id: d for d in await get_catalog().list_documents(tenant_id)}
    doc = docs.get(doc_id)
    if doc is None:
        raise NotFoundError("document not found", details={"tenant_id": tenant_id, "doc_id": doc_id})
    # Reproduce this document's OWN chunking unless the caller asked for a change. Reindex used
    # to apply a hardcoded strategy, so a document ingested one way was silently re-chunked
    # another — different boundaries, different chunk ids, no signal. Adopting a new strategy
    # is a real use of reindex; it just must be asked for.
    if strategy is None:
        recorded = doc.metadata.get(META_CHUNK_STRATEGY)
        strategy = ChunkStrategy(recorded) if recorded else None
    # What this document was chunked as BEFORE the layout path took it over, if it ever was. Passed
    # through so that a reindex arriving without a tree — flag turned off, or a re-extraction that
    # produced no markup — restores it rather than the deployment default.
    recorded_base = doc.metadata.get(META_BASE_CHUNK_STRATEGY)
    base_strategy = ChunkStrategy(recorded_base) if recorded_base else None
    try:
        raw = await get_object_store().get(tenant_id, raw_object_key(doc_id))
    except KeyError as exc:
        raise NotFoundError("raw bytes missing for document", details={"doc_id": doc_id}) from exc

    # WRITE, THEN SWAP. The old chunks are NOT purged first: the previous version keeps serving
    # until the new one is indexed, so a failure anywhere in extract -> chunk -> embed -> index
    # leaves the document exactly as it was. Purging first is how the 2026-07-31 re-index left
    # two documents with ZERO chunks — present in the catalog, retrievable by nothing — when the
    # embedder timed out on them. Content-derived ids (P44) make this clean: text that survived
    # keeps its id and is simply re-upserted, so only genuinely-gone ids need removing.
    old_chunk_ids = [c.chunk_id for c in await get_catalog().document_chunks(tenant_id, doc_id)]

    trace_id = str(uuid.uuid4())

    doc_intel = await wiring.doc_intel_service_for(tenant_id)  # held for its masker (see _mask_with)
    extracted = await doc_intel.extract(doc, raw)
    # A replay re-runs extraction, so the provenance stamp must be refreshed with it —
    # otherwise the stored tags describe an extraction that no longer produced this text.
    # A replay re-derives the tags too, not only the provenance. They are D8 retrieval filters
    # and adopting a new taxonomy is one of the two reasons to re-index at all — refreshing
    # half of what a re-extraction produced left a KB filtering on a vocabulary it had already
    # replaced, with the job reporting success.
    derived = await _derive_tags(tenant_id, extracted)
    # CLEAR the previously-derived keys before merging. A plain merge only overwrites keys the new
    # derivation still produces, so one it stops producing survives from the old index -- which is
    # how documents kept a wrong `year` through a re-index that reported success.
    kept = {k: v for k, v in doc.metadata.items() if k not in DERIVED_KEYS}
    doc = doc.model_copy(update={"metadata": {**kept, **derived, **extracted.provenance}})
    pc_child, pc_parent = await _parent_child_sizing(tenant_id)  # FR-C2: per-KB parent/child sizing
    # Slice 6, Phase C: reindex re-derives chemical tokens too (same triple gate) so adopting the
    # feature is a reindex away. None ⇒ byte-identical.
    chem_normalizer = get_molecule_normalizer() if await _chem_normalize_on(tenant_id) else None
    # RAG v2 Phase 5: one parse, two consumers — see the ingest path's call. This is also how an
    # already-ingested corpus ADOPTS layout-aware chunking: with the flag on, a reindex hands the
    # tree to the chunker, which overrides this document's recorded strategy (`index_document`
    # states why) and re-stamps it. Flag off ⇒ None ⇒ the recorded strategy is reproduced exactly.
    tree = _layout_tree(doc_id, extracted, text=extracted.text)
    # RAG v2 Phase 3, same placement and reason as the ingest path's call: the card is built before
    # indexing (it decides chunk size and the `embed_text` prefix) and stored after it. A reindex
    # RE-PROFILES rather than reusing the stored card — the same reason the overlay and the tree are
    # refreshed here: a replay re-runs extraction, so a card built from the previous extraction's
    # outline may describe headings this text no longer has. It is also the only way a corpus
    # ingested before the flag went on acquires a card at all.
    profile = await _document_profile(tenant_id, doc, extracted, tree, mask=_mask_with(doc_intel))
    # RAG v2 Phase 4, same placement and reason as the ingest path's call. A reindex RE-ENRICHES
    # rather than reading back what is stored, for the reason it re-profiles: a replay re-runs
    # extraction, so the section ids and their text can differ from the run that produced the
    # stored summaries. It is also the only way a corpus ingested before the flag went on acquires
    # summaries and propositions at all.
    enrichment, manifest = await _section_enrichment(tenant_id, doc, tree, profile, mask=_mask_with(doc_intel))
    indexed = await (await wiring.indexing_service_for(tenant_id)).index_document(
        doc,
        extracted.text,
        page_map=extracted.page_map,
        strategy=strategy,
        modality=extracted.modality,
        ocr_confidence=extracted.min_confidence,
        child_max_chars=pc_child,
        parent_max_chars=pc_parent,
        chem_normalizer=chem_normalizer,
        layout=_chunkable_layout(tree, text_pages=_text_pages(extracted.page_map)),
        mask=_mask_with(doc_intel),
        base_strategy=base_strategy,
        profile=profile,
        enrichment=enrichment,
        # NOT reused here, deliberately — see `_may_reuse_vectors`. A reindex is the REPAIR path
        # (it is what the auto-reindexer and every runbook reach for when an index is suspect), and
        # the skip rests on the stores still holding what the catalog says they do. A collection
        # recreated or wiped out of band breaks that assumption without changing the fingerprint,
        # and a reindex that silently declined to rebuild it would be the one failure nobody could
        # see. So reindex always re-embeds, exactly as before.
    )
    doc = doc.model_copy(
        update={
            "metadata": {
                **doc.metadata,
                **_strategy_metadata(indexed),
            }
        }
    )
    await get_catalog().register_document(doc)
    # The LAYOUT OVERLAY is refreshed for the same reason as the provenance stamp above: a replay
    # re-runs extraction, so an overlay written by the previous one describes structure that no
    # longer matches this text. It is also the ONLY way to backfill an already-ingested corpus —
    # the other write is on the ingest path, so without this a document ingested before the flag
    # went on could never acquire an overlay and the structure tree would be limited to new uploads.
    #
    # It is placed HERE — after the new chunks are durably indexed and the row re-registered —
    # NOT up beside the provenance refresh, which is only a local `model_copy` and is not durable
    # until this very `register_document`. If `index_document`
    # raises, write-then-swap leaves the OLD chunks serving, and an overlay already overwritten
    # with the new extraction would then describe structure that does not match the text still
    # being served. That is not hypothetical — Chandra is a stochastic VLM, so a re-run's headings
    # and table structure can genuinely differ, and the structure tree is what joins the two. The
    # raw bytes are NOT rewritten on this path either, so the raw/overlay symmetry that justifies
    # the ingest-time placement does not apply here.
    #
    # Flag OFF leaves whatever is stored untouched rather than purging it: the overlay's only
    # consumers are gated on the SAME flag, so with it off a stale artifact is unreadable, and
    # turning the flag back on and re-indexing refreshes it.
    #
    # THE DERIVED ARTIFACTS MUST NOT BE ABLE TO BLOCK THE SWAP. Everything in this block is derived
    # from text that is already durably indexed, and everything after it — the swap, the INDEX
    # audit event, the semantic-cache invalidation, the community rebuild — is what makes the new
    # index COHERENT. Letting a derived write abort the function left the document answering from
    # BOTH chunk generations at once (the old ids were never purged), with no audit record saying a
    # reindex happened and a semantic cache still serving pre-reindex answers. None of that is
    # detectable from outside; a stale derived artifact is, because the overlay carries the raw
    # object's `content_hash` and another reindex rewrites it. So the asymmetry is deliberate:
    # these are logged and skipped, the swap always runs.
    #
    # `_persist_layout_structure` is the concrete risk and the reason this is not theoretical — it
    # is a Postgres delete+insert inside one transaction whose first statement takes an advisory
    # lock, so it can time out under contention. Before this change the exposure also grew by
    # roughly 6x with Phase 2c, because every native PDF now has an overlay and therefore a tree to
    # persist, where previously only the Chandra-markup minority did.
    try:
        await _persist_layout_overlay(tenant_id, doc_id, extracted, content_hash=hashlib.sha256(raw).hexdigest())
        # ...and the structure tree parsed out of it, REPLACING whatever this document's last
        # extraction stored: the ids are content-derived, so a reindex of changed text produces
        # different section/block ids, and appending would leave the previous run's tree behind to
        # double on every replay and describe text no longer served. The Phase-3 card and the
        # Phase-4 section enrichment go with it, inside this same block and for the same reason
        # every other derived write is here: none of them may abort the swap.
        await _persist_v2_derived(tenant_id, doc_id, tree, profile, enrichment, manifest, doc.content_hash or "")
        # Refresh the image's CLIP vector too, so cross-modal recall survives a re-index (the vector
        # is derived from the raw bytes, independent of how the text was re-extracted). No-op off.
        if get_settings().multimodal_enabled and multimodal.is_image(doc):
            # FR-C2: children only — a parent must never receive an image vector (would make it
            # citable).
            await multimodal.index_image_vectors(tenant_id, raw, indexed.child_chunk_ids)
        # Diagram structure (additive/OFF) — same reasoning as ingest_document's call.
        if multimodal.is_image(doc):
            await _index_diagram_graph(tenant_id, doc_id, raw, indexed.child_chunk_ids)
    except Exception as derived_exc:
        _log.error(
            "reindex_derived_artifacts_incomplete",
            tenant_id=tenant_id,
            doc_id=doc_id,
            error=str(derived_exc)[:200],
        )

    # The swap: everything the new version does not contain. Reached only after the new chunks
    # are durably indexed, so there is no window in which the document has neither version.
    #
    # The new ids come from the INDEXER, not from re-reading the catalog. Nothing was purged
    # first, so the catalog holds old AND new together and "old minus what the catalog has" is empty by
    # construction — the swap deleted nothing and stale chunks stayed retrievable with text the
    # document no longer contained. Measured live: 120 reported vs 132 held, 552 vs 627.
    fresh_ids = set(indexed.chunk_ids)
    stale = [cid for cid in old_chunk_ids if cid not in fresh_ids]
    if stale:
        await get_catalog().delete_chunks(tenant_id, stale)
        await _purge_chunk_artifacts(tenant_id, stale)

    audit: AuditService = wiring.audit_service()
    await audit.record(
        tenant_id=tenant_id,
        trace_id=trace_id,
        action=AuditAction.INDEX,
        actor=actor,
        payload={
            "doc_id": doc_id,
            "chunks": indexed.chunk_count,
            "reindex": True,
            "strategy": indexed.strategy.value,
            "replaced_chunks": len(stale),
        },
    )
    semantic_cache.invalidate(tenant_id)
    # rebuild this doc's graph contribution (GraphRAG); same is_image plumbing as ingest_document.
    await _index_graph(tenant_id, doc_id, is_image=multimodal.is_image(doc))
    await _refresh_communities(tenant_id)  # ...and the communities that cite its chunks (P48)
    _log.info("document_reindexed", tenant_id=tenant_id, doc_id=doc_id, chunks=indexed.chunk_count)
    return IngestOutcome(doc_id, indexed.chunk_count, extracted.pii_masked, trace_id)


async def delete_document(tenant_id: str, doc_id: str, *, actor: str, audit_event: bool = True) -> int:
    """Remove a document from every store — catalog rows, dense + sparse index, raw object, and
    the RAG v2 layout overlay (the extraction artifact that rides alongside the raw object; see
    ``_persist_layout_overlay``, which states why the two are purged together).
    Returns the number of chunks purged. Idempotent for an unknown doc. Records a DELETE audit
    event so data removal is attributable (skippable via ``audit_event=False`` when the caller
    records its own aggregate event, e.g. a KB delete)."""
    chunk_ids = await get_catalog().delete_document(tenant_id, doc_id)
    await _purge_chunk_artifacts(tenant_id, chunk_ids)
    await get_object_store().delete(tenant_id, raw_object_key(doc_id))
    # Next to the raw object DELIBERATELY: both are pre-masking extraction artifacts of the same
    # document, so every path that purges one must purge the other. This function is the one DPDP
    # right-to-erasure cascades through (``compliance.erase``) and the one ``delete_knowledge_base``
    # calls per document, so covering it here covers both. Unconditional on the flag: a document
    # ingested while it was ON must still be purgeable after it is turned OFF.
    await get_object_store().delete(tenant_id, layout_object_key(doc_id))
    await _refresh_communities(tenant_id)  # communities cite chunk ids that are now gone
    semantic_cache.invalidate(tenant_id)
    if audit_event:
        await wiring.audit_service().record(
            tenant_id=tenant_id,
            trace_id=str(uuid.uuid4()),
            action=AuditAction.DELETE,
            actor=actor,
            payload={"scope": "document", "doc_id": doc_id, "chunks_purged": len(chunk_ids)},
        )
    _log.info("document_deleted", tenant_id=tenant_id, doc_id=doc_id, chunks=len(chunk_ids), actor=actor)
    return len(chunk_ids)


async def trash_knowledge_base(org_id: str, kb_id: str, *, actor: str) -> None:
    """Soft-delete a KB: move it to Trash (hidden from the active list, data retained and
    restorable). No stores are purged — only an explicit purge does that. Audited."""
    if kb_id == org_id:
        raise ValidationError("the organisation's root namespace cannot be deleted")
    await wiring.kb_store().trash(org_id, kb_id)
    await wiring.audit_service().record(
        tenant_id=kb_id,
        trace_id=str(uuid.uuid4()),
        action=AuditAction.DELETE,
        actor=actor,
        payload={"scope": "kb_trash", "kb_id": kb_id},
    )
    _log.info("kb_trashed", org_id=org_id, kb_id=kb_id, actor=actor)


async def restore_knowledge_base(org_id: str, kb_id: str, *, actor: str) -> None:
    """Restore a KB from Trash — nothing was purged, so it comes back exactly as it was."""
    await wiring.kb_store().restore(org_id, kb_id)
    _log.info("kb_restored", org_id=org_id, kb_id=kb_id, actor=actor)


async def delete_knowledge_base(org_id: str, kb_id: str, *, actor: str) -> dict[str, int]:
    """Delete a whole knowledge base: purge every document from all stores, drop its
    conversations + model connection, and remove it from the org's registry. The org's
    default KB (kb_id == org_id) can't be deleted — it's the org's root namespace.

    Purging the stores (not just the registry) matters: the tenant guard authorises any
    ``{org}__*`` id by prefix, so leftover data would still be reachable."""
    if kb_id == org_id:
        raise ValidationError("the default knowledge base cannot be deleted")
    docs = await get_catalog().list_documents(kb_id)
    chunks = 0
    for doc in docs:  # per-doc DELETE events suppressed — one aggregate KB-delete event is recorded below
        chunks += await delete_document(kb_id, doc.doc_id, actor=actor, audit_event=False)
    # Drop the vector collection + BM25 index outright, not just the per-document chunks: this
    # removes the namespace shells AND any orphan vectors not tracked in the catalog (e.g. a chunk
    # upserted just before a mid-ingest store fault), which a catalog-enumeration purge can't see.
    # Purge is irreversible, so a complete sweep is correct — and it keeps a re-created same-name KB
    # from inheriting stale data.
    await get_vector_store().delete_tenant(kb_id)
    await get_sparse_index().delete_tenant(kb_id)
    await wiring.conversation_store().purge(kb_id)
    await get_graph_store().clear(kb_id)
    await multimodal.purge_image_tenant(kb_id)
    for conn in await wiring.model_connection_store().list(kb_id):
        await wiring.model_connection_store().delete(kb_id, conn.conn_id)
    await wiring.kb_store().remove(org_id, kb_id)
    semantic_cache.invalidate(kb_id)
    await wiring.audit_service().record(
        tenant_id=kb_id,
        trace_id=str(uuid.uuid4()),
        action=AuditAction.DELETE,
        actor=actor,
        payload={"scope": "knowledge_base", "kb_id": kb_id, "documents": len(docs), "chunks_purged": chunks},
    )
    _log.info("knowledge_base_deleted", org_id=org_id, kb_id=kb_id, documents=len(docs), chunks=chunks, actor=actor)
    return {"documents": len(docs), "chunks": chunks}


_THRESHOLD_KEYS = (
    "abstain_threshold",
    "faithful_threshold",
    "flag_threshold",
    # Completeness-guard ceilings (Slice 1) — adaptive per KB via the same override seam. Stored as
    # floats; the generation service casts the count-valued ones to int. Absent ⇒ platform default.
    "completeness_max_continuations",
    "completeness_continuation_budget_factor",
    "completeness_coverage_max_loops",
    "completeness_max_guard_calls",
    "completeness_faithfulness_tolerance",
    # Exact technical-token BM25 boost (Slice 2a) — per-KB adaptive via the same override seam, so a
    # pharma KB can weight exact formula/SMILES/identifier matches higher than an HR KB. Absent ⇒
    # the sparse leg falls back to the global `exact_term_boost` setting.
    "exact_boost",
    # Multi-vector retrieval legs (Slice 2b) — per-KB adaptive via the same override seam, stored
    # 1.0/0.0 (bge learned-sparse leg / ColBERT MaxSim re-scoring) + an int-as-float prefetch pool.
    # A pharma KB turns ColBERT on hard; a prose HR KB may leave both off. Only consulted when the
    # master `multivector_enabled` flag is on AND the KB's embedder emits multi-vectors; absent ⇒
    # the platform defaults (both off, `colbert_prefetch_pool` setting).
    "multivector_sparse_enabled",
    "multivector_colbert_enabled",
    "colbert_prefetch_pool",
    # Notation verbatim-grounding toggle (Slice 4) — per-KB adaptive via the same override seam,
    # stored as 1.0/0.0. A pharma KB turns it on hard; an HR KB may leave it off. Absent ⇒ the
    # platform `notation_grounding_enabled` default. Read in `apply_notation_grounding` (wiring).
    "notation_grounding_enabled",
    # Parent/child sizing + scope (Slice 3, FR-C2) — per-KB adaptive via the same override seam so a
    # dense pharma-table KB and a prose HR KB size children/parents differently. Stored as floats;
    # the two *_max_chars are read at INDEX time (IndexingService.index_document) and max_parents at
    # QUERY time (_parent_expand). Absent ⇒ the platform default. Ingest-time keys need a reindex to
    # take effect (they change what is stored); max_parents takes effect on the next query.
    "parent_child_child_max_chars",
    "parent_child_parent_max_chars",
    "parent_child_max_parents",
    "parent_child_context_max_chars",
    # Typed domain-graph edges (Slice 6, sub-slice B) — per-KB gate stored 1.0/0.0. When set AND
    # the master `domain_profile_enabled` flag is on AND the KB's profile declares a relation
    # vocabulary, `_index_graph` ALSO writes directed, labeled domain-relation edges at ingest.
    # Absent ⇒ off ⇒ only co-occurrence edges are written (byte-identical to today). This is a
    # reindex-class knob: new edges appear on new ingest / opt-in reindex only.
    "domain_relations_enabled",
    # RDKit chemical normalization (Slice 6, Phase C) — per-KB gate stored 1.0/0.0. When set AND the
    # master `chem_normalization_enabled` flag is on AND the KB's profile domain is `chemistry`,
    # ingest stamps a chunk's canonical SMILES/InChIKey as ADDITIONAL searchable exact tokens and the
    # query is augmented to match them, so equivalent-but-differently-drawn molecules match. Absent ⇒
    # off ⇒ byte-identical to today. Reindex-class (changes what is indexed): new tokens appear on new
    # ingest / opt-in reindex only. Documented like `domain_relations_enabled` above.
    "chem_normalization_enabled",
)
_PERSONA_KEY = "persona"


async def _parent_child_sizing(kb_id: str) -> tuple[int | None, int | None]:
    """Per-KB (child_max_chars, parent_max_chars) for parent/child indexing (FR-C2), or (None, None)
    when the KB set no override — IndexingService then uses the platform default. Read from the same
    override seam as the thresholds so sizing a KB's chunks is one governance mechanism, not two."""
    overrides = await kb_threshold_overrides(kb_id)
    child = overrides.get("parent_child_child_max_chars")
    parent = overrides.get("parent_child_parent_max_chars")
    return (int(child) if child is not None else None, int(parent) if parent is not None else None)


async def _index_graph(kb_id: str, doc_id: str, *, is_image: bool = False) -> tuple[int, int]:
    """Extract each of a document's chunks into the KB knowledge graph (FR-C6, GraphRAG). Returns
    ``(entities, edges)`` written (0, 0 when disabled/empty) so the caller can record them as DAG
    metrics. No-op unless GraphRAG is enabled; failures never break ingest (the graph is additive).

    ``is_image`` (default False, so every text/PDF call site is unaffected): for an IMAGE
    document, the same OCR/caption text ALSO gets read here by the ordinary heuristic
    extractor below (unchanged, still needed for hybrid BM25) — but that text is Mermaid
    syntax + captions, not prose, so it surfaces scaffolding fragments and bare
    infra-category words as junk entities/edges alongside the diagram parser's own precise
    edges (see _index_diagram_graph). ``filter_image_text_entities`` drops those, image-only,
    right before co-occurrence edges are written."""
    if not get_settings().graphrag_enabled:
        # Slice 6-B AND-graphrag coupling: typed domain-relation edges are written INSIDE this
        # function (see _index_domain_relations below), so they also require graphrag_enabled.
        # A KB that turned on domain relations while GraphRAG is off would get ZERO typed edges
        # silently — surface that as a one-line WARN. Only checked when the domain-profile master
        # is on (byte-identical otherwise), and only on the rare graphrag-off path.
        if get_settings().domain_profile_enabled:
            overrides = await kb_threshold_overrides(kb_id)
            if overrides.get("domain_relations_enabled", 0.0) >= 1.0:
                _log.warning("domain_relations_enabled_but_graphrag_off", kb_id=kb_id, doc_id=doc_id)
        return (0, 0)
    total_entities = 0
    total_edges = 0
    try:
        extractor = await wiring.resolve_graph_extractor(kb_id)
        store = get_graph_store()
        # FR-C2: parents are a NON-searchable context bank — they must NEVER enter the graph, or
        # _graph_augment/_global_augment would fetch a parent id and merge an 1800-char parent blob
        # into retrieval.spans as a ranked, CITABLE span (the same leak the other four consumers
        # guard against). Children only here.
        chunks = not_parent(await get_catalog().document_chunks(kb_id, doc_id))
        # Run the (slow) per-chunk extraction concurrently, bounded so we neither serialize
        # the model/Sarvam calls nor flood the endpoint. Graph-store WRITES stay sequential.
        sem = asyncio.Semaphore(get_settings().graph_index_concurrency)

        async def _extract(chunk: Chunk) -> GraphTriple:
            async with sem:
                return await extractor.extract(chunk.text)

        # return_exceptions=True: a single chunk whose extractor call raises (the flaky
        # gateway/timeout path GraphRAG is meant to tolerate) must NOT drop the whole
        # document's graph — skip just that chunk and keep every other chunk's contribution.
        # Fail-fast gather would abort all writes on the first raise; the sequential loop this
        # replaced kept the chunks written before the failure, so plain gather was a regression.
        results = await asyncio.gather(*(_extract(chunk) for chunk in chunks), return_exceptions=True)
        for chunk, triple in zip(chunks, results, strict=True):
            if isinstance(triple, BaseException):
                _log.warning(
                    "graph_chunk_extract_failed", kb_id=kb_id, chunk_id=chunk.chunk_id, error=str(triple)[:160]
                )
                continue
            entities, edges = triple.entities, triple.edges
            if is_image and entities:
                # Image-only: filter OCR/caption-artifact junk BEFORE it becomes a
                # co-occurrence edge, then drop any edge that lost an endpoint to the filter.
                # A real text/PDF chunk (is_image=False) never reaches this branch.
                entities = filter_image_text_entities(entities)
                survivors = {e.lower() for e in entities}
                edges = [(a, b) for a, b in edges if a.lower() in survivors and b.lower() in survivors]
            if entities:
                await store.add_chunk(kb_id, chunk.chunk_id, entities, edges)
                total_entities += len(entities)
                total_edges += len(edges)
        _log.info("graph_indexed", kb_id=kb_id, doc_id=doc_id, entities=total_entities, edges=total_edges)
        # Slice 6, sub-slice B: ADDITIVELY write typed domain-relation edges alongside the
        # co-occurrence edges above, when the KB's profile declares a relation vocabulary and the
        # per-KB gate is on. No-op (byte-identical) otherwise. Never breaks ingest.
        await _index_domain_relations(kb_id, chunks)
        # RAG v2 Phase 6: seed the edges the DOCUMENT ITSELF declares (`doc_relations` from Phase 3,
        # `sections.amends` from Phase 4) — better evidence than anything extracted from prose,
        # because the document said it and carries a quote to prove it. Live for `doc_relations`
        # since Phase 3 landed; the `sections.amends` half waits on Phase 4.
        total_edges += await _seed_declared_relations(kb_id, doc_id, chunks)
    except Exception as exc:  # never let graph extraction fail an ingest
        _log.warning("graph_index_failed", kb_id=kb_id, doc_id=doc_id, error=str(exc)[:160])
    return (total_entities, total_edges)


async def _domain_relation_vocab(kb_id: str) -> tuple[str, ...]:
    """The KB's declared domain-relation vocabulary (sub-slice B), or ``()`` when the KB has no
    profile or a ``general`` profile. Read from the same ``domain_profile`` tenant_config key the
    detector writes — no vocab means the typed-relation extractor never runs (byte-identical)."""
    from svc_control_plane.domain_profile import PROFILE_KEY  # noqa: PLC0415 - avoid an import cycle

    raw = (await get_catalog().get_tenant_config(kb_id)).get(PROFILE_KEY)
    if not raw:
        return ()
    try:
        domain = str(json.loads(raw).get("domain", ""))
    except (ValueError, TypeError):
        return ()
    return relation_vocab(domain)


def _profile_is_chemistry(raw: str | None) -> bool:
    """Whether a raw ``domain_profile`` tenant_config value declares the ``chemistry`` domain.

    PURE and fail-safe: False for a missing, non-JSON, or valid-JSON-but-not-an-object value
    (``null`` / a number / an array / a bare string) — the ``isinstance(parsed, dict)`` guard means a
    malformed profile can never raise here, matching the detect-and-suggest contract that a bad
    profile changes nothing."""
    if not raw:
        return False
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return False
    return isinstance(parsed, dict) and str(parsed.get("domain", "")) == "chemistry"


async def _kb_is_chemistry(kb_id: str) -> bool:
    """Whether the KB's ACTIVE domain profile is ``chemistry`` (Slice 6, Phase C). Read from the same
    ``domain_profile`` tenant_config key the detector writes, so chemical normalization applies ONLY
    to chemistry KBs. False (never raises) on a missing or malformed profile."""
    from svc_control_plane.domain_profile import PROFILE_KEY  # noqa: PLC0415 - avoid an import cycle

    return _profile_is_chemistry((await get_catalog().get_tenant_config(kb_id)).get(PROFILE_KEY))


async def _chem_normalize_on(kb_id: str) -> bool:
    """The triple gate for chemical normalization: the master ``chem_normalization_enabled`` flag,
    the per-KB ``chem_normalization_enabled`` gate (>= 1.0), AND a ``chemistry`` domain profile must
    ALL hold. Any one missing ⇒ False ⇒ the ingest/query paths are byte-identical. The master flag is
    checked first so a chem-off deployment does NO tenant_config read; when on, the per-KB gate and
    the profile are read from ONE tenant_config fetch (both live in the same dict)."""
    if not get_settings().chem_normalization_enabled:
        return False
    from svc_control_plane.domain_profile import PROFILE_KEY  # noqa: PLC0415 - avoid an import cycle

    config = await get_catalog().get_tenant_config(kb_id)
    raw_gate = config.get("chem_normalization_enabled")
    try:
        gate = float(raw_gate) if raw_gate is not None else 0.0
    except (TypeError, ValueError):
        gate = 0.0
    if gate < 1.0:
        return False
    return _profile_is_chemistry(config.get(PROFILE_KEY))


async def _seed_declared_relations(kb_id: str, doc_id: str, chunks: list[Chunk]) -> int:
    """Turn a document's DECLARED relations into directed, labeled graph edges (RAG v2 Phase 6).

    The plan: *seed graph edges from ``doc_relations`` and ``sections.amends`` (doc->doc,
    section->section AMENDS/SUPERSEDES/REFERENCES) before or alongside its own extraction.* That
    "before" was literal — this consumer landed ahead of the Phase 3/4 producers that fill those
    tables, which is the plan's own rollout order (§13). Phase 3 has since landed and
    `_persist_document_profile` writes `doc_relations`, so a document that declares an amendment
    now reaches this path with real relations and no change was needed here. `sections.amends` is
    still Phase 4's, so a section-level amendment still returns nothing.

    Each edge is anchored to the chunk whose text contains the relation's evidence quote
    (`rag_core.graph.relation_edges`), so a walk that reaches it lands on citable text. A
    relation whose quote resolves to no chunk is DROPPED and counted, not attached to an arbitrary
    chunk — an amendment edge pointing at the wrong evidence is worse than no edge, because the
    answer it supports would cite text that does not say it.

    Runs inside :func:`_index_graph`, so it inherits that function's GraphRAG gate and its
    never-break-an-ingest guarantee. Returns the number of edges written."""
    if not chunks:
        return 0
    relations = await document_relations(kb_id, doc_id)
    if not relations:
        return 0
    seed = relation_edges(relations, chunks)
    if seed.unresolved or seed.unknown_types:
        _log.warning(
            "declared_relations_partially_seeded",
            kb_id=kb_id,
            doc_id=doc_id,
            seeded=len(seed.edges),
            unresolved_quotes=len(seed.unresolved),
            unknown_types=sorted(set(seed.unknown_types)),
            detail="a relation whose evidence quote matches no chunk is dropped rather than "
            "anchored to an arbitrary one",
        )
    if not seed.edges:
        return 0
    store = get_graph_store()
    # Grouped by anchor chunk because the store writes evidence per chunk. `add_diagram_edges` is
    # the directed-labeled-edge writer, shared with the diagram parser and the Slice 6-B domain
    # extractor (see `GraphTriple.diagram_edges`) — a third producer needs no new store surface.
    by_chunk: dict[str, list[tuple[str, str, str]]] = {}
    for chunk_id, source, target, relation in seed.edges:
        by_chunk.setdefault(chunk_id, []).append((source, target, relation))
    for chunk_id, edges in by_chunk.items():
        names = sorted({name for source, target, _rel in edges for name in (source, target)})
        await store.add_diagram_edges(kb_id, chunk_id, names, edges)
    _log.info(
        "declared_relations_seeded",
        kb_id=kb_id,
        doc_id=doc_id,
        edges=len(seed.edges),
        chunks=len(by_chunk),
    )
    return len(seed.edges)


async def _index_domain_relations(kb_id: str, chunks: list[Chunk]) -> int:
    """Extract DIRECTED, TYPED domain-relation edges from a document's chunks and write them via
    the existing ``add_diagram_edges`` path (Slice 6, sub-slice B). Additive alongside the
    co-occurrence edges. Returns the number of typed edges written (0 when disabled/empty).

    Triple-gated so flag-off is byte-identical: the master ``domain_profile_enabled`` flag, the
    per-KB ``domain_relations_enabled`` knob, AND a non-empty domain vocabulary must all hold —
    otherwise this returns immediately having done no extra work. Failures never break ingest.

    AND-graphrag requirement: this runs ONLY from inside :func:`_index_graph`, which returns early
    when ``graphrag_enabled`` is off — so typed edges require GraphRAG to be on as well (the edges
    live in the same graph store GraphRAG retrieval reads). ``_index_graph`` emits a WARN if a KB
    set the gate while GraphRAG is off, so the coupling is never a silent no-op."""
    settings = get_settings()
    if not settings.domain_profile_enabled or not chunks:
        return 0
    overrides = await kb_threshold_overrides(kb_id)
    if overrides.get("domain_relations_enabled", 0.0) < 1.0:
        return 0
    vocab = await _domain_relation_vocab(kb_id)
    if not vocab:
        return 0
    written = 0
    try:
        extractor = await wiring.resolve_domain_relation_extractor(kb_id, vocab)
        store = get_graph_store()
        sem = asyncio.Semaphore(settings.graph_index_concurrency)

        async def _extract(chunk: Chunk) -> GraphTriple:
            async with sem:
                return await extractor.extract(chunk.text)

        results = await asyncio.gather(*(_extract(chunk) for chunk in chunks), return_exceptions=True)
        for chunk, triple in zip(chunks, results, strict=True):
            if isinstance(triple, BaseException):
                _log.warning(
                    "domain_relation_chunk_failed", kb_id=kb_id, chunk_id=chunk.chunk_id, error=str(triple)[:160]
                )
                continue
            if triple.diagram_edges:
                # Same idempotent, directed typed-edge write the diagram path uses.
                await store.add_diagram_edges(kb_id, chunk.chunk_id, triple.entities, triple.diagram_edges)
                written += len(triple.diagram_edges)
        if written:
            _log.info("domain_relations_indexed", kb_id=kb_id, edges=written)
    except Exception as exc:  # never let domain-relation extraction fail an ingest
        _log.warning("domain_relations_index_failed", kb_id=kb_id, error=str(exc)[:160])
    return written


async def _index_diagram_graph(kb_id: str, doc_id: str, raw: bytes, chunk_ids: list[str]) -> None:
    """For an IMAGE document, ALSO ask the vision LLM to read the diagram's boxes/arrows and
    write those as real graph edges (FR-C6, GraphRAG — pragmatic diagram-structure extension).

    Additive alongside ``_index_graph``, not instead of it: OCR/caption text still indexes and
    still feeds the ordinary co-occurrence/LLM extraction unchanged (hybrid BM25 needs that
    text regardless). This is a SEPARATE vision call, over the raw image bytes, whose output is
    parsed for actual arrow-direction edges — a flowchart's real STRUCTURE, not just its text
    labels landing in the graph as isolated, unconnected entities. No-op unless both
    ``diagram_graph_enabled`` and ``graphrag_enabled`` are on; failures never break ingest."""
    settings = get_settings()
    if not (settings.graphrag_enabled and settings.diagram_graph_enabled) or not chunk_ids:
        return
    try:
        # The KB's own model, exactly like the caption path — reading RAG_VISION_LLM_* alone
        # 401'd here on every image while the credential sat in the console connection.
        gateway = await wiring.resolve_tenant_vision_gateway(kb_id)
        response = await gateway.complete_with_images(build_diagram_structure_prompt(), [raw])
        triple = parse_diagram_structure(response)
        if not triple.entities:
            _log.info("diagram_graph_empty", kb_id=kb_id, doc_id=doc_id, model=gateway.model_name)
            return
        store = get_graph_store()
        # NEW write path (not add_chunk's co-occurrence fan-out): each parsed arrow becomes ONE
        # precise (source, target, relation) row, direction preserved — a 9-node/14-edge diagram
        # writes 14 edges, not a ~142-edge near-complete clique across every node pair.
        for chunk_id in chunk_ids:
            await store.add_diagram_edges(kb_id, chunk_id, triple.entities, triple.diagram_edges)
        _log.info(
            "diagram_graph_indexed",
            kb_id=kb_id,
            doc_id=doc_id,
            model=gateway.model_name,
            entities=len(triple.entities),
            edges=len(triple.diagram_edges),
        )
    except Exception as exc:  # never let diagram-structure extraction fail an ingest
        _log.warning("diagram_graph_failed", kb_id=kb_id, doc_id=doc_id, error=str(exc)[:160])


def _graph_span(chunk: Any) -> RetrievedSpan:
    """A graph-linked chunk as an UNSCORED candidate. It reaches the reranker exactly as a
    fused candidate does — including ``span``/``source_uri``, without which the citation is
    not click-throughable in the console."""
    return RetrievedSpan(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        page=chunk.page,
        section=chunk.metadata.get(META_SECTION),
        category=chunk.metadata.get(META_CATEGORY),
        span=chunk.span,
        source_uri=chunk.source_uri,
        acl=list(chunk.acl),
        classification=chunk.classification,
    )


async def _score_graph_spans(query: RetrievalQuery, spans: list[RetrievedSpan], keep: int) -> list[RetrievedSpan]:
    """Score graph-contributed candidates with the SAME reranker retrieval uses.

    They arrive with no relevance signal whatsoever: entity extraction is a proper-noun regex
    and the graph walk is a pure structural lookup, so a constant here is a number the gate
    would read as confidence that nothing measured. Scoring them also makes
    ``merge_retrieved``'s cross-list comparison sound, and turns an arbitrary sample
    (``linked_chunks`` returns its ids lexicographically) into a relevance-ordered one.

    Their slot in the prompt does not depend on the score: ``merge_retrieved`` interleaves the
    two lists round-robin, so a bridge passage — which a cross-encoder rates low by
    construction, since carrying no trace of the question is exactly why passage search missed
    it — still reaches the model, now with an honest number attached."""
    return await (await wiring.resolve_tenant_reranker(query.tenant_id)).rerank(query.text, spans, keep)


async def _graph_augment(query: RetrievalQuery, retrieval: RetrievalResult) -> tuple[RetrievalResult, int]:
    """Local graph search (FR-D4): link the query's entities to their graph neighbours and
    merge those chunks into hybrid retrieval, lifting multi-hop recall. Returns (result, added)."""
    settings = get_settings()
    if not settings.graphrag_enabled:
        return retrieval, 0
    entities = extract_entities(query.text)
    if not entities:
        return retrieval, 0
    store = get_graph_store()
    chunk_ids = await store.linked_chunks(query.tenant_id, entities, hops=settings.graph_retrieval_hops)
    chunk_ids = [c for c in chunk_ids if c not in {s.chunk_id for s in retrieval.spans}]
    if not chunk_ids:
        return retrieval, 0
    # Fetch WIDE, then let the reranker choose. `linked_chunks` orders lexicographically, so
    # cutting to the final count here would hand the model an arbitrary sample of the neighbourhood.
    chunks = await get_catalog().get_chunks(query.tenant_id, chunk_ids[: settings.graph_retrieval_candidates])
    # ACL- AND filter-match graph chunks exactly like retrieval (never surface a span the
    # caller can't see, and never leak past an active D8/B7 metadata filter).
    acl = set(query.acl)
    graph_spans = [
        _graph_span(c)
        for c in chunks.values()
        if (not c.acl or (set(c.acl) & acl))
        and c.classification <= query.clearance  # banking clearance: never leak an over-classified chunk
        and _chunk_in_scope(c, query)
    ]
    if not graph_spans:
        return retrieval, 0
    graph_spans = await _score_graph_spans(query, graph_spans, settings.graph_retrieval_max_chunks)
    merged, _ = merge_retrieved([retrieval.spans, graph_spans], query.top_k + len(graph_spans))
    return RetrievalResult(
        query=retrieval.query,
        spans=merged,
        # Confidence has to be backed by a span that SURVIVED the merge. Carrying the
        # pre-merge maximum forward can report a number no returned span justifies.
        max_score=max((s.score for s in merged), default=0.0),
        mode_used=retrieval.mode_used,
    ), len(graph_spans)


async def _neighbour_augment(query: RetrievalQuery, retrieval: RetrievalResult) -> tuple[RetrievalResult, int]:
    """Pull the passages either side of the top hits (contiguity recall). Returns (result, added).

    Retrieval ranks each passage on its own merits, which fails any question whose answer is a
    list longer than one passage. Measured on a real HR handbook: the passage holding "Core
    Values" and "six simple ingredients" ranks well, but the one holding values 3-5 begins
    "...assets to ensure there is Luxury in Hard work. 3. Continuous Knowledge Enhancement..." --
    it carries half the answer and shares almost no vocabulary with the question. One value
    appeared in NO retrieved passage at top_k=5 or top_k=10. A list is CONTIGUOUS, so retrieving
    any part of it should bring its neighbours; that is a structural fix where a bigger top_k is
    not one.

    Neighbours arrive UNSCORED. Nothing judged them relevant -- they are here because of where
    they sit, not what they say -- and claiming a score would feed the abstain gate a number no
    reranker produced. `gate_for` reads the top-scoring span, so an unscored neighbour can never
    open the gate on its own."""
    settings = get_settings()
    if not (settings.neighbour_expansion_enabled and settings.neighbour_window and retrieval.spans):
        return retrieval, 0
    have = {s.chunk_id for s in retrieval.spans}
    grouped = anchors_by_document(
        [(s.doc_id, s.chunk_id) for s in retrieval.spans], limit=settings.neighbour_expand_top
    )
    catalog = get_catalog()
    wanted: list[str] = []
    for doc_id, anchors in grouped.items():
        try:
            # FR-C2: drop parent context records from the reading-order list so a parent never
            # enters the neighbour window / fusion pool (a parent is not a retrievable child).
            ordered = not_parent(await catalog.document_chunks(query.tenant_id, doc_id))
        except Exception as exc:  # a document deleted mid-query must not fail the query
            _log.warning("neighbour_lookup_failed", tenant_id=query.tenant_id, doc_id=doc_id, error=str(exc)[:160])
            continue
        wanted += [
            cid
            for cid in neighbour_ids([c.chunk_id for c in ordered], anchors, window=settings.neighbour_window)
            if cid not in have
        ]
    if not wanted:
        return retrieval, 0
    chunks = await catalog.get_chunks(query.tenant_id, wanted)
    # Same visibility rules as retrieval itself. Neighbours inherit the document's ACL and
    # classification, so this should never drop one -- it is here because "should never" is not
    # an argument for skipping the check on a path that puts text in front of a user.
    acl = set(query.acl)
    extra = [
        _graph_span(c)
        for c in chunks.values()
        if (not c.acl or (set(c.acl) & acl)) and c.classification <= query.clearance and _chunk_in_scope(c, query)
    ]
    if not extra:
        return retrieval, 0
    merged, _ = merge_retrieved([retrieval.spans, extra], query.top_k + len(extra))
    return RetrievalResult(
        query=retrieval.query,
        spans=merged,
        # Unchanged by construction: neighbours are unscored, so they cannot raise confidence.
        max_score=max((s.score for s in merged), default=0.0),
        mode_used=retrieval.mode_used,
    ), len(extra)


async def _mark_superseded(query: RetrievalQuery, retrieval: RetrievalResult) -> RetrievalResult:
    """Stamp each span with its document's effective date, and mark the ones a newer document
    replaces, so the answer can lead with what is in force.

    Measured live: two Acme handbooks listed DIFFERENT members of the same committee. The platform
    reported both and said the sources differ -- honest, but a reader cannot tell which list is
    current, and on a thousand-document corpus that becomes the normal answer.

    Superseded sources are KEPT, not hidden. An auditor asking "what did the policy say before?"
    is a real question, and silently dropping the older version answers it wrongly by omission.
    The prompt tells the model to lead with the current one and record what it replaced."""
    settings = get_settings()
    if not settings.version_conflict_enabled or len({s.doc_id for s in retrieval.spans}) < _MIN_COMPARABLE_DOCS:
        return retrieval
    try:
        raw = (await get_catalog().get_tenant_config(query.tenant_id)).get(_PINNED_DOCS_KEY, "[]")
        pinned = set(json.loads(raw))
    except (ValueError, TypeError):
        pinned = set()
    # Dates ride on the spans (stamped onto every chunk at ingest), so this costs no lookup. A
    # per-document catalog read here would be one round trip per document on every question.
    dates: dict[str, str | None] = {}
    for span in retrieval.spans:
        dates.setdefault(span.doc_id, span.doc_date)
        if span.doc_date and not dates[span.doc_id]:
            dates[span.doc_id] = span.doc_date
    # Supersession is only true WITHIN a document, so the resolver groups by title first. The
    # organisation's own name prefixes most of its files ("Acme_..."), which would make every pair
    # of its policies look related; it is derived from the KB id, never a list someone maintains.
    superseded = resolve_supersession(
        [DocumentVersion(doc_id=d, date=dates.get(d), pinned=d in pinned) for d in dates],
        ignore_tokens={query.tenant_id.split("__", 1)[0]},
        min_similarity=settings.version_family_similarity,
    )
    if not superseded and not any(dates.values()):
        return retrieval
    spans = [
        s.model_copy(update={"doc_date": dates.get(s.doc_id), "superseded_by": superseded.get(s.doc_id)})
        for s in retrieval.spans
    ]
    if superseded:
        _log.info("version_conflict", tenant_id=query.tenant_id, superseded=list(superseded))
    return RetrievalResult(
        query=retrieval.query, spans=spans, max_score=retrieval.max_score, mode_used=retrieval.mode_used
    )


async def _parent_expand(query: RetrievalQuery, retrieval: RetrievalResult) -> tuple[RetrievalResult, int]:
    """Attach deduped PARENT CONTEXT to child spans (FR-C2, Option B). Returns ``(result, parents)``.

    Parents are a NON-CITABLE context bank. This fetches the parent behind each surviving top child
    span and rides its text on the span via ``parent_text`` (``span.model_copy``), so ``build_prompt``
    renders a CONTEXT block ABOVE the numbered SOURCES and faithfulness can see parent-only support —
    WITHOUT adding any ranked/cited span and WITHOUT touching ``max_score`` (the abstain gate scores
    CHILDREN only, so a parent can never open or close the gate). Citations stay one-per-child, so
    ``[n]`` still resolves to the precise child span/page.

    No-op — byte-identical — unless parent/child is enabled AND some surviving span carries a
    ``parent_id`` (the legacy self-skip). Applies the SAME ACL/clearance/D8-filter guard as the other
    augment steps, so a parent the caller could not retrieve is never shown. Bounded twice, both per-KB
    adaptive: at most ``parent_child_max_parents`` distinct parents (ordered by best child score), and
    a deterministic CHARACTER budget ``parent_child_context_max_chars`` on the AGGREGATE context (kept
    best-first until the total would overflow; the rest dropped) — rag-core is pure so there is no
    tokenizer at query time, and the char budget keeps the CONTEXT bank from crowding the child SOURCES."""
    settings = get_settings()
    if not settings.parent_child_enabled or not retrieval.spans:
        return retrieval, 0
    head = retrieval.spans[: query.top_k]
    parent_ids = list(dict.fromkeys(s.parent_id for s in head if s.parent_id))
    if not parent_ids:  # legacy / non-parent-child index: nothing to expand
        return retrieval, 0
    overrides = await kb_threshold_overrides(query.tenant_id)
    max_parents = int(overrides.get("parent_child_max_parents", settings.parent_child_max_parents))
    context_budget = int(overrides.get("parent_child_context_max_chars", settings.parent_child_context_max_chars))
    chunks = await get_catalog().get_chunks(query.tenant_id, parent_ids)
    acl = set(query.acl)
    visible: dict[str, str] = {}
    for pid, c in chunks.items():
        # SAME guard as _graph_augment / _neighbour_augment: ACL group ∩, clearance, D8 filter.
        entitled = not c.acl or bool(set(c.acl) & acl)
        if entitled and c.classification <= query.clearance and _chunk_in_scope(c, query):
            visible[pid] = c.text
    if not visible:
        return retrieval, 0
    # Rank parents by the STRONGEST child that references them, cap the COUNT, then apply the
    # aggregate CHAR budget best-first: keep whole parents until the next would overflow. A single
    # top parent that alone exceeds the budget (pathological per-KB sizing) is truncated so the
    # aggregate is ALWAYS within budget and there is always some context.
    best_score: dict[str, float] = {}
    for s in head:
        if s.parent_id in visible:
            best_score[s.parent_id] = max(best_score.get(s.parent_id, 0.0), s.score)
    ranked = sorted(best_score, key=lambda p: best_score[p], reverse=True)[:max_parents]
    kept: dict[str, str] = {}
    used = 0
    for pid in ranked:
        text = visible[pid]
        if used + len(text) <= context_budget:
            kept[pid] = text
            used += len(text)
        elif not kept:
            kept[pid] = text[:context_budget]  # top parent alone over budget: truncate, then stop
            break
        else:
            break
    if not kept:
        return retrieval, 0
    spans = [
        s.model_copy(update={"parent_text": kept[s.parent_id]}) if s.parent_id in kept else s for s in retrieval.spans
    ]
    _log.info(
        "parent_expanded",
        tenant_id=query.tenant_id,
        parents=len(kept),
        context_chars=sum(len(t) for t in kept.values()),
    )
    return (
        RetrievalResult(
            query=retrieval.query,
            spans=spans,
            max_score=retrieval.max_score,  # unchanged: parents are context, never ranked
            mode_used=retrieval.mode_used,
        ),
        len(kept),
    )


async def _attach_followups(
    answer: GroundedAnswer, query: RetrievalQuery, retrieval: RetrievalResult
) -> GroundedAnswer:
    """Offer follow-up questions (FR-I8) built ONLY from the retrieved spans — never the
    model's parametric knowledge. Grounded answers only; failures are swallowed (additive)."""
    settings = get_settings()
    if not (settings.followups_enabled and answer.status is AnswerStatus.GROUNDED and retrieval.spans):
        return answer
    try:
        gen = await wiring.resolve_followup_generator(query.tenant_id)
        context = "\n\n".join(s.text for s in retrieval.spans[:5])
        suggestions = await gen.suggest(query.text, context, settings.followups_count)
        if suggestions:
            return answer.model_copy(update={"suggested_followups": suggestions})
    except Exception as exc:  # follow-ups are a nicety, never a failure path
        _log.warning("followups_failed", tenant_id=query.tenant_id, error=str(exc)[:160])
    return answer


# The 2nd-strongest cluster must be at least this fraction of the top cluster's score for the
# query to count as ambiguous — otherwise one topic clearly dominates (a confident single-topic
# hit) and we offer nothing. Biases hard toward UNDER-triggering: a runaway top hit gets no options.
_CLARIFY_CLUSTER_COMPARABILITY = 0.6


async def _attach_clarifications(
    answer: GroundedAnswer, query: RetrievalQuery, retrieval: RetrievalResult
) -> tuple[GroundedAnswer, int]:
    """Offer grounded "Did you mean …?" options (Layer-2 item 4) when a query is genuinely
    AMBIGUOUS — built ONLY from the retrieved spans, never invented. NON-AUTHORITATIVE: it never
    changes the answer text, citations, status, or the grounded-or-abstain invariant (it may
    accompany a GROUNDED or an ABSTAINED answer). Fully fail-safe: any error → no options, never
    raises. Returns the (possibly-unchanged) answer and the number of options offered.

    BIAS TOWARD UNDER-TRIGGERING: options are offered only when the retrieval forms
    >= clarify_min_clusters distinct doc clusters AND the answer is ambiguous — it abstained, or is
    a low-confidence GROUNDED hit (max_score < ceiling) — AND no single cluster runs away with it."""
    settings = get_settings()
    if not settings.clarify_enabled or not retrieval.spans:
        return answer, 0
    # Only ambiguous answers qualify. A confident grounded answer (max_score >= ceiling) is a clear
    # single-topic query → offer nothing. FLAGGED_FOR_REVIEW is held for a human → offer nothing.
    ambiguous = answer.status is AnswerStatus.ABSTAINED or (
        answer.status is AnswerStatus.GROUNDED and retrieval.max_score < settings.clarify_confidence_ceiling
    )
    if not ambiguous:
        return answer, 0
    try:
        clusters = cluster_spans(retrieval.spans)
        if len(clusters) < settings.clarify_min_clusters:
            return answer, 0
        # Not a runaway top hit: the 2nd-strongest cluster must be reasonably comparable to the 1st,
        # else one topic dominates and there is nothing genuine to disambiguate.
        top = clusters[0].best_score
        if top > 0 and clusters[1].best_score < _CLARIFY_CLUSTER_COMPARABILITY * top:
            return answer, 0
        gen = await wiring.resolve_clarify_generator(query.tenant_id)
        options = await gen.suggest(query.text, retrieval.spans, settings.clarify_count)
        if options:
            return answer.model_copy(update={"clarification_options": options}), len(options)
    except Exception as exc:  # disambiguation is a nicety, never a failure path
        _log.warning("clarify_failed", tenant_id=query.tenant_id, error=str(exc)[:160])
    return answer, 0


async def _purge_chunk_artifacts(tenant_id: str, chunk_ids: list[str]) -> None:
    """Remove everything derived from these chunk ids from every store OUTSIDE the catalog.

    The catalog row is the only thing that can ENUMERATE a document's chunks, so whoever deletes it
    owes these four deletes first — otherwise the vectors and BM25 documents survive with no row
    left to find them by, and neither ``delete_document`` nor DPDP erasure can ever reach them
    again. That is not hypothetical: the failed-ingest rollback deleted the row and discarded the
    ids `delete_document` handed back.

    Stated once because four callers need exactly this list — ``delete_document``, both
    write-then-swap stale purges, and the rollback — and a fifth store added to one of them and not
    the others is how an orphan class gets invented. Caller supplies the catalog delete; this is
    deliberately everything BUT, so a caller that must delete chunk rows without deleting the
    document (the swaps) uses the same purge.
    """
    if not chunk_ids:
        return
    await get_vector_store().delete_chunks(tenant_id, chunk_ids)
    await get_sparse_index().delete_chunks(tenant_id, chunk_ids)
    await multimodal.purge_image_vectors(tenant_id, chunk_ids)
    # DPDP right-to-erasure MUST cascade into the knowledge graph too: without this an erased
    # data-principal survived as API-reachable graph nodes/edges (carrying their own name) plus
    # community evidence, while the signed erasure certificate claimed completion.
    await _forget_graph_chunks(tenant_id, chunk_ids)


async def _forget_graph_chunks(kb_id: str, chunk_ids: list[str]) -> None:
    """Drop dead chunk ids from the knowledge graph. Never fails the write that triggered it."""
    if not chunk_ids or not get_settings().graphrag_enabled:
        return
    try:
        await get_graph_store().remove_chunks(kb_id, chunk_ids)
    except Exception as exc:
        _log.warning("graph_forget_failed", kb_id=kb_id, error=str(exc)[:160])


async def _refresh_communities(kb_id: str) -> None:
    """Rebuild the KB's communities after its graph changed, if it has any (P48).

    Community rows hold EVIDENCE CHUNK IDS. Nodes and edges are rebuilt per document by
    _index_graph, but communities are built on demand and were left untouched, so once a
    document was re-chunked or replaced they cited chunks that no longer existed — and since
    P44 made ids content-derived, changed text changes ids for certain. Global search then
    answers from evidence nothing can resolve.

    Only rebuilt when the KB ALREADY has communities: building them is a graph-wide pass and
    can call the model for summaries, so a KB that never asked for them does not start paying
    for them on every ingest. Never allowed to fail the write that triggered it."""
    try:
        if not await get_graph_store().list_communities(kb_id):
            return
        await build_communities(kb_id)
    except Exception as exc:  # a stale-community refresh must not fail an ingest or a re-index
        _log.warning("community_refresh_failed", kb_id=kb_id, error=str(exc)[:160])


async def build_communities(kb_id: str) -> dict[str, int]:
    """Detect thematic communities over the KB graph and summarise each (GraphRAG slice 2).

    A graph-wide pass — run on demand (never per-ingest). Label-propagation clusters the
    entities; each community keeps its strongest entities + evidence chunks and gets a theme
    summary (heuristic entity list, or KB-model prose when the summary backend is 'llm').
    Returns {communities, entities} counts. No-op unless GraphRAG is enabled."""
    settings = get_settings()
    if not settings.graphrag_enabled:
        return {"communities": 0, "entities": 0}
    store = get_graph_store()
    nodes, edges = await store.graph_data(kb_id)
    if not nodes:
        await store.save_communities(kb_id, [])
        return {"communities": 0, "entities": 0}

    by_key = {n.key: n for n in nodes}
    labels = detect_communities([n.key for n in nodes], edges)
    grouped: dict[int, list[str]] = {}
    for key, cid in labels.items():
        grouped.setdefault(cid, []).append(key)

    summarizer = await wiring.resolve_community_summarizer(kb_id)
    # Qualifying communities IN ORDER (largest first, tie-broken by min member) — the exact
    # order the sequential version appended them, so community_id assignment is unchanged.
    qualifying = [
        members
        for members in sorted(grouped.values(), key=lambda m: (-len(m), min(m)))
        if len(members) >= settings.graph_community_min_size  # singletons/pairs are noise, not a theme
    ]
    # Fan out the (slow) per-community sample-fetch + summarize concurrently, bounded by the
    # same semaphore as graph extraction; the DB write below stays sequential and in order.
    sem = asyncio.Semaphore(get_settings().graph_index_concurrency)

    async def _build(members: list[str], community_id: int) -> GraphCommunity:
        ranked = sorted(members, key=lambda k: (-len(by_key[k].chunk_ids), k))
        top_entities = [by_key[k].display for k in ranked[:12]]
        chunk_ids: list[str] = []
        for k in ranked:
            for c in by_key[k].chunk_ids:
                if c not in chunk_ids:
                    chunk_ids.append(c)
        sample_ids = chunk_ids[:4]
        async with sem:
            samples = await get_catalog().get_chunks(kb_id, sample_ids) if sample_ids else {}
            summary = await summarizer.summarize(
                top_entities, [samples[c].text for c in sample_ids if c in samples], len(members)
            )
        return GraphCommunity(
            community_id=community_id,
            label=community_label(top_entities),
            size=len(members),
            top_entities=top_entities,
            chunk_ids=chunk_ids[:20],
            summary=summary,
            members=list(members),  # ALL member node keys — so the subgraph can tag every node's theme
        )

    # Keyed by position so the assembled list — and the 0,1,2,… community_id — matches the
    # sequential version byte-for-byte regardless of completion order.
    communities: list[GraphCommunity] = list(
        await asyncio.gather(*(_build(members, cid) for cid, members in enumerate(qualifying)))
    )

    await store.save_communities(kb_id, communities)
    _log.info("communities_built", kb_id=kb_id, communities=len(communities), entities=len(nodes))
    return {"communities": len(communities), "entities": len(nodes)}


async def list_communities(kb_id: str) -> list[GraphCommunity]:
    """The KB's thematic communities (for the console Themes card); empty until built."""
    if not get_settings().graphrag_enabled:
        return []
    return await get_graph_store().list_communities(kb_id)


async def _global_augment(query: RetrievalQuery, retrieval: RetrievalResult) -> tuple[RetrievalResult, int]:
    """Global/thematic search (GraphRAG slice 2): a broad question ("what are the main
    themes?") is answered from the most query-relevant communities' evidence chunks — KB-wide
    breadth a narrow top-k misses — merged into retrieval. Citations stay real chunks, so
    grounded-or-abstain is untouched. No-op unless enabled AND the query looks global."""
    settings = get_settings()
    if not (settings.graphrag_enabled and settings.graph_global_search and is_global_query(query.text)):
        return retrieval, 0
    communities = await get_graph_store().list_communities(query.tenant_id)
    if not communities:
        return retrieval, 0
    picked = rank_communities(query.text, communities)[: settings.graph_global_communities]
    seen = {s.chunk_id for s in retrieval.spans}
    want: list[str] = []
    for com in picked:
        for c in com.chunk_ids[: settings.graph_global_chunks_per_community]:
            if c not in seen and c not in want:
                want.append(c)
    if not want:
        return retrieval, 0
    chunks = await get_catalog().get_chunks(query.tenant_id, want)
    acl = set(query.acl)
    global_spans = [
        _graph_span(c)
        for c in chunks.values()
        if (not c.acl or (set(c.acl) & acl))
        and c.classification <= query.clearance  # banking clearance: never leak an over-classified chunk
        and _chunk_in_scope(c, query)
    ]
    if not global_spans:
        return retrieval, 0
    # The community path already selects a bounded set per community, so there is nothing to
    # widen here — scoring is only about the number being real.
    global_spans = await _score_graph_spans(query, global_spans, len(global_spans))
    merged, _ = merge_retrieved([retrieval.spans, global_spans], query.top_k + len(global_spans))
    _log.info("graph_global_augmented", tenant_id=query.tenant_id, communities=len(picked), added=len(global_spans))
    return RetrievalResult(
        query=retrieval.query,
        spans=merged,
        # Same as the local-search path: the confidence must be backed by a surviving span.
        max_score=max((s.score for s in merged), default=0.0),
        mode_used=retrieval.mode_used,
    ), len(global_spans)


def _recency_active(query: RetrievalQuery) -> bool:
    return query.after is not None or query.before is not None or query.freshness > 0


def _chunk_matches_filters(chunk: Any, filters: dict[str, str]) -> bool:
    """A chunk satisfies a D8 metadata filter iff it matches every key (FR-B7). Used so
    graph-augmented chunks respect the query's filters instead of leaking past them."""
    return all(chunk.metadata.get(k) == v for k, v in filters.items())


def _chunk_in_scope(chunk: Any, query: RetrievalQuery) -> bool:
    """Whether an augmented chunk may join the result: it must satisfy BOTH of the query's
    scoping predicates — the D8 metadata filters AND the explicit document scope.

    These are two different predicates over two different data shapes: tags live in a chunk's
    ``metadata``, while ``doc_id`` is a record field. That is precisely why ``doc_ids`` is its own
    channel rather than a reserved filter key (routing it through the tag channel is what made the
    sparse leg silently empty). Both are checked here so a graph / neighbour / parent / community
    augment cannot merge chunks from OUTSIDE the requested document into a document-scoped answer.
    Empty ``doc_ids`` ⇒ filters-only, i.e. unchanged."""
    return _chunk_matches_filters(chunk, query.filters) and (not query.doc_ids or chunk.doc_id in query.doc_ids)


async def _doc_scope_for(query: RetrievalQuery) -> DocScope | None:
    """The document this question names, if it names one unambiguously (see rag_core.docscope)."""
    if not get_settings().doc_scope_enabled:
        return None
    try:
        docs = await get_catalog().list_documents(query.tenant_id)
    except Exception as exc:  # a catalog hiccup must not fail the question
        _log.warning("doc_scope_lookup_failed", tenant_id=query.tenant_id, error=str(exc)[:160])
        return None
    return scope_for_query(query.text, [d.doc_id for d in docs])


def _apply_doc_scope(retrieval: RetrievalResult, scope: DocScope, want: int) -> RetrievalResult:
    """Keep only spans from the named document.

    Filtered HERE rather than pushed down, for the FALLBACK semantics below — not because the
    stores disagree any more. (They used to: the sparse index namespaces tags under ``meta.*``
    while the vector payload is flat, so a ``doc_id`` metadata filter silently emptied the BM25
    leg. ``RetrievalQuery.doc_ids`` now expresses a document scope as a record-field predicate
    both legs implement, so a push-down IS available.) What a push-down cannot replicate in one
    round-trip is returning the UNSCOPED result when the scope turns out to be wrong; that would
    need a second, unscoped retrieval. Post-filtering an over-fetched pool keeps both legs intact
    and gets both behaviours in one trip.

    If the scoped document contributed nothing at all, the UNSCOPED result is returned rather than
    an empty one: the scope is an inference about what the reader meant, and being wrong about it
    must not turn an answerable question into an abstain."""
    kept = [s for s in retrieval.spans if s.doc_id == scope.doc_id]
    if not kept:
        _log.warning("doc_scope_empty", doc_id=scope.doc_id, detail="falling back to the unscoped result")
        return retrieval
    return RetrievalResult(
        query=retrieval.query,
        spans=kept[:want],
        max_score=max((s.score for s in kept[:want]), default=0.0),
        mode_used=retrieval.mode_used,
    )


# How many passages the PROMPT may hold when a question enumerates items.
#
# Coverage fetches per-item passages, and then the prompt threw most of them away: the model sees
# retrieval.query.top_k ranked passages, which defaults to 5. Ten SOPs, five passages -- so the
# answer reported "No detailed explanation is provided in the sources" for the items whose
# passages had been retrieved and then dropped. Truthful about what it was shown, and wrong about
# the corpus, which is the hardest kind of wrong to notice.
#
# Two passages per item is enough for a title plus a detail line without turning a ten-item answer
# into a context-window problem. Bounded three ways: never below what the caller asked for, never
# above what retrieval actually found (the prompt must not claim evidence that does not exist),
# and never above _MAX_ITEM_PROMPT_SPANS.
_PROMPT_SPANS_PER_ITEM = 2
_MAX_ITEM_PROMPT_SPANS = 40


def _prompt_budget_for_items(top_k: int, *, items: int, available: int) -> int:
    if items <= 0:
        return top_k
    wanted = min(items * _PROMPT_SPANS_PER_ITEM, _MAX_ITEM_PROMPT_SPANS)
    return min(max(top_k, wanted), available)


async def _coverage_augment(
    query: RetrievalQuery, retrieval: RetrievalResult, trace: list[AgentStep]
) -> tuple[RetrievalResult, int]:
    """Guarantee evidence for every item a question ENUMERATES (see rag_core.enumeration).

    Retrieval ranks passages by similarity to the whole question, so when ten identifiers are
    named a few of them take the entire budget and the rest are never fetched — the answer then
    truthfully reports them as absent. Measured live on an SOP corpus: two of ten covered, while
    each of the other eight answered in full when asked on its own.

    One retrieval per item, phrased as the item plus the question so that item's own passages rank
    first, keeping ``coverage_per_item`` each; unioned with what ordinary retrieval already found
    (best score per chunk wins, round-robin so no item is crowded out). The ordinary result is
    never shrunk — coverage only ADDS — so a question with no enumeration is untouched and one
    with an enumeration cannot lose a passage it would otherwise have had."""
    settings = get_settings()
    items = enumerated_items(query.text)[: settings.coverage_max_items]
    if not items:
        return retrieval, 0
    svc = await wiring.retrieval_service_for(query.tenant_id)
    per_item = [
        await svc.retrieve(
            query.model_copy(update={"text": f"{item}: {query.text}", "top_k": settings.coverage_per_item})
        )
        for item in items
    ]
    cap = min(80, len(retrieval.spans) + len(items) * settings.coverage_per_item)
    spans, max_score = merge_retrieved([retrieval.spans, *[r.spans for r in per_item]], cap)
    added = max(len(spans) - len(retrieval.spans), 0)
    trace.append(
        AgentStep(
            step="coverage",
            detail=f"Question names {len(items)} items → {settings.coverage_per_item} passages each, +{added} added",
        )
    )
    _log.info("coverage_augmented", tenant_id=query.tenant_id, items=len(items), added=added)
    return (
        RetrievalResult(
            query=retrieval.query,
            spans=spans,
            max_score=max(max_score, retrieval.max_score),
            mode_used=retrieval.mode_used,
        ),
        added,
    )


def _drop_navigation_spans(retrieval: RetrievalResult) -> RetrievalResult:
    """A contents page points at the answer; it never contains one. Dropped from the evidence
    when there is other evidence — see rag_core.navigation.

    max_score is deliberately NOT recomputed: it is the gate's reading of how relevant the best
    thing retrieval FOUND is, and lowering it here would turn "we found a strong match" into an
    abstention because the strongest match happened to be a signpost."""
    kept = drop_navigation(retrieval.spans)
    if len(kept) == len(retrieval.spans):
        return retrieval
    _log.info("navigation_spans_dropped", dropped=len(retrieval.spans) - len(kept), kept=len(kept))
    return retrieval.model_copy(update={"spans": kept})


async def _apply_recency(query: RetrievalQuery, retrieval: RetrievalResult, want: int) -> RetrievalResult:
    """Recency-aware retrieval (FR-D9): filter spans to the query's document time window and/or
    boost fresher documents, then trim to the caller's ``want``. No-op unless requested."""
    if not _recency_active(query):
        return retrieval
    settings = get_settings()
    doc_times = {d.doc_id: d.ingested_at for d in await get_catalog().list_documents(query.tenant_id)}
    spans = apply_recency(
        retrieval.spans,
        doc_times,
        after=query.after,
        before=query.before,
        freshness=query.freshness,
        now=datetime.now(UTC),
        half_life_days=settings.recency_halflife_days,
    )[:want]
    _log.info("recency_applied", tenant_id=query.tenant_id, kept=len(spans), before=len(retrieval.spans))
    return RetrievalResult(
        query=retrieval.query,
        spans=spans,
        # The freshness boost is a SORT key — `apply_recency` keeps each span's own relevance
        # score — so after the re-sort position 0 is the freshness winner, not the relevance
        # winner. Reading confidence off it would let a boost silently tighten the gate.
        max_score=max((s.score for s in spans), default=0.0),
        mode_used=retrieval.mode_used,
    )


_TAXONOMY_KEY = "taxonomy"


async def _record_duplicate_chunks(kb_id: str, doc_id: str) -> None:
    """Report chunks of this document that repeat text already in the knowledge base.

    Ingest already refuses a byte-identical re-upload (FR-A8/C9). This is what that cannot
    see: the same TEXT under a different doc_id. Measured on the real corpus,
    `acme__hr_policies` holds 15 such pairs at similarity 1.0 — the Code of Conduct is
    reproduced verbatim inside the HR Policy Handbook.

    It is why an answer can cite [1], [3] and [5] and say "the same requirement is reiterated
    in..." — one rule printed three times reads as three independent confirmations. Reported,
    never dropped: a reprint is real content and may be the passage a reader wants. Suppressing
    the duplicates in a RESULT SET is a separate, later step, and this is what makes it possible.

    Never fails an ingest — a missing duplication report is not a reason to reject a document."""
    try:
        threshold = get_settings().near_duplicate_threshold
        # FR-C2: a parent's text is the newline-join of its own children, so it would flag as a
        # ~1.0 duplicate of every child on every parent_child ingest — misleading noise. Children only.
        mine = [(c.chunk_id, c.text) for c in not_parent(await get_catalog().document_chunks(kb_id, doc_id))]
        if not mine:
            return
        others = [
            (c.chunk_id, c.text)
            for d in await get_catalog().list_documents(kb_id)
            if d.doc_id != doc_id
            for c in not_parent(await get_catalog().document_chunks(kb_id, d.doc_id))
        ]
        # Against the rest of the KB and against itself: a document that reprints its own
        # boilerplate on every page causes the same harm as two documents that overlap.
        found = near_duplicates(mine, [*others, *mine], threshold)
        if found:
            _log.warning(
                "chunks_duplicate_existing_content",
                kb_id=kb_id,
                doc_id=doc_id,
                duplicates=len(found),
                of_chunks=len(mine),
                sample=[{"chunk": c, "matches": m, "similarity": s} for c, (m, s) in list(found.items())[:3]],
                detail="the same text is indexed more than once, so a single source can be "
                "cited several times and read as independent corroboration",
            )
    except Exception as exc:  # observability must never break an ingest
        _log.warning("duplicate_scan_failed", kb_id=kb_id, doc_id=doc_id, error=str(exc)[:160])


async def _derive_tags(kb_id: str, extracted: Any) -> dict[str, str]:
    """Metadata DERIVED from the extracted text: FR-B2 structured fields, detected language and
    FR-B7 taxonomy tags.

    One function because ingest and reindex must produce the same tags from the same text. They
    did not: reindex refreshed only the extraction provenance, so a re-index after a taxonomy
    change silently kept the old category."""
    derived: dict[str, str] = {**extracted.fields}  # gstin / pan / date / amount
    settings = get_settings()
    if settings.language_detection_enabled and extracted.text.strip():
        derived["lang"] = detect_language(extracted.text)  # Indic multilingual — filterable
    if settings.auto_metadata_enabled:
        derived.update(auto_tags(extracted.text, await _load_taxonomy(kb_id)))
    return derived


async def _load_taxonomy(kb_id: str) -> dict[str, list[str]]:
    """The KB's configured taxonomy ``{label: [keywords]}`` (empty if unset). FR-B7."""
    raw = (await get_catalog().get_tenant_config(kb_id)).get(_TAXONOMY_KEY)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return {str(k): [str(x) for x in v] for k, v in data.items() if isinstance(v, list)}
    except (ValueError, TypeError):
        return {}


async def kb_persona(kb_id: str) -> str | None:
    """The KB's answer-style persona (FR-E7), or None. Shapes wording only — the grounding
    contract in the prompt always wins."""
    persona = (await get_catalog().get_tenant_config(kb_id)).get(_PERSONA_KEY)
    return persona.strip() if persona and persona.strip() else None


async def _semantic_cache_lookup(query: RetrievalQuery) -> tuple[list[float] | None, GroundedAnswer | None]:
    """(query embedding, cached answer) when the cache is on; (None, None) otherwise.

    A hit is only returned if it would STILL pass today's gate (see semantic_cache.still_passes);
    otherwise the query falls through and is answered fresh.

    Recency-scoped (FR-D9) OR metadata-filtered (D8/B7) queries bypass the cache entirely —
    those cannot be expressed in the key, so a cached answer would ignore the window / filter,
    and a scoped answer would poison later un-scoped queries. Returning None here skips BOTH the
    lookup and the put (which is guarded on a non-None vector).

    ``top_k`` CAN be expressed in the key, so it is part of it rather than a bypass: it decides
    how much evidence the generator sees, and an enumerated answer built from 5 passages is
    genuinely less complete than the same one built from 15."""
    settings = get_settings()
    if not settings.semantic_cache_enabled or _recency_active(query) or query.filters or query.doc_ids:
        return None, None
    # The KB's ACTIVE embedder, not the platform one — mixed vector spaces in the cache
    # would compare 384-d entries against 1024-d probes.
    [vector] = await (await wiring.resolve_tenant_embedder(query.tenant_id)).embed([query.text])
    hit = semantic_cache.get(query.tenant_id, vector, list(query.acl), query.clearance, query.top_k)
    if hit is None:
        return vector, None
    # A hit skips the grounding and faithfulness gates that produced it, and those thresholds
    # are per-KB and operator-editable. Re-check the stored numbers against the CURRENT ones so
    # raising a floor takes effect immediately instead of waiting out the entry's lifetime.
    overrides = await kb_threshold_overrides(query.tenant_id)
    if not still_passes(
        hit,
        abstain=overrides.get("abstain_threshold", settings.abstain_threshold),
        faithful=overrides.get("faithful_threshold", settings.faithful_threshold),
        flag=overrides.get("flag_threshold", settings.flag_threshold),
    ):
        _log.info("semantic_cache_stale_gate", tenant_id=query.tenant_id, confidence=hit.confidence)
        return vector, None
    return vector, hit


async def _agentic_retry(
    query: RetrievalQuery, first: RetrievalResult, gate: float
) -> tuple[RetrievalResult, dict[str, object]]:
    """CRAG-lite (FR-F2): the first retrieval was too weak to answer — transform the
    query, retrieve the sub-queries, and keep whichever result is stronger. One bounded
    retry; the decision rides the QUERY audit payload."""
    if not (get_settings().agentic_retry_enabled and should_abstain(first.max_score, gate)):
        return first, {}
    # Hand the transformer what the FAILED retrieval returned. Those passages are from the
    # right corpus in the corpus's own words, even when none of them answers the question — so
    # a reformulation can use the vocabulary the documents actually use instead of guessing
    # synonyms. Measured on a real question: 0.0084 unchanged without them, 0.9976 with.
    transformer = await wiring.query_transformer_for(query.tenant_id)
    subs = await transformer.transform(query, [s.text for s in first.spans])
    if not subs or subs == [query.text]:
        return first, {"agentic_retry": "no_transform"}
    svc = await wiring.retrieval_service_for(query.tenant_id)
    results = [await svc.retrieve(query.model_copy(update={"text": s})) for s in subs]
    spans, max_score = merge_retrieved([r.spans for r in results], query.top_k)
    if max_score <= first.max_score:
        return first, {"agentic_retry": "no_improvement", "retry_score": max_score}
    _log.info(
        "agentic_retry_improved",
        tenant_id=query.tenant_id,
        before=first.max_score,
        after=max_score,
        subqueries=len(subs),
    )
    retried = RetrievalResult(query=query, spans=spans, max_score=max_score, mode_used=get_retriever().mode)
    return retried, {"agentic_retry": "improved", "retry_from": first.max_score, "retry_to": max_score}


def _mode_payload(query: RetrievalQuery) -> dict[str, str]:
    """What the retriever ACTUALLY did, for the tamper-evident QUERY event (P26).

    The audit chain used to record ``query.mode`` — the mode the CALLER asked for — while
    HybridRetriever ignores it and always runs dense + BM25 + RRF. An immutable log that
    faithfully records something that did not happen is worse than one that records nothing:
    it is the artefact a compliance reviewer trusts. When the request differs from what ran,
    both are recorded so the discrepancy is visible rather than quietly resolved."""
    performed = get_retriever().mode
    payload = {"mode": performed.value}
    if query.mode is not performed:
        payload["mode_requested"] = query.mode.value
    return payload


async def kb_threshold_overrides(kb_id: str) -> dict[str, float]:
    """Per-KB grounding/faithfulness threshold overrides (empty when the KB uses the platform
    defaults). Values are validated on write.

    Sources, later winning: (1) the legacy per-KB threshold keys in tenant_config; (2) the
    hierarchical Guardrail POLICY (Guardrails page) — its resolved ``faithful_threshold`` control,
    so a threshold set/locked at the platform/org/KB tier actually drives faithfulness routing.
    The policy is the governance source of truth, so it takes precedence over the legacy key."""
    config = await get_catalog().get_tenant_config(kb_id)
    out: dict[str, float] = {}
    for key in _THRESHOLD_KEYS:
        raw = config.get(key)
        if raw is not None:
            try:
                out[key] = float(raw)
            except ValueError:
                _log.warning("bad_threshold_override", kb_id=kb_id, key=key, value=raw)
    # Guardrail policy overlay — the resolved faithfulness threshold from the Guardrails page.
    #
    # ONLY when a tier actually SET it. resolve_effective returns the CATALOGUE DEFAULT with
    # ``owner=None`` for a control no tier has configured, and applying that default overwrote a
    # threshold the tenant had explicitly set — the per-KB faithfulness control stopped working
    # and answers routed to FLAGGED_FOR_REVIEW against a floor nobody chose. The policy is the
    # governance source of truth when it expresses an intent; a default is not an intent.
    try:
        for entry in await guardrails_policy.resolve_effective(get_catalog(), kb_id):
            if entry.id == "faithful_threshold" and entry.owner is not None and isinstance(entry.value, (int, float)):
                out["faithful_threshold"] = float(entry.value)
    except Exception as exc:
        _log.warning("guardrail_policy_overlay_failed", kb_id=kb_id, error=str(exc)[:160])
    return out


async def _with_exact_boost(query: RetrievalQuery) -> RetrievalQuery:
    """Stamp the per-KB exact-term boost onto the query (Slice 2a) so the sparse leg can weight
    exact technical-token matches for THIS KB. No-op — returns the query unchanged — unless the
    exact-match flag is on AND the KB set an ``exact_boost`` override; otherwise the sparse adapter
    falls back to the global ``exact_term_boost`` setting. Flag OFF ⇒ byte-identical (no extra
    lookup, no model_copy)."""
    if not get_settings().exact_term_match_enabled:
        return query
    boost = (await kb_threshold_overrides(query.tenant_id)).get("exact_boost")
    if boost is None:
        return query
    # Clamp to non-negative: model_copy SKIPS pydantic validation, so RetrievalQuery.exact_boost's
    # ge=0 guard does NOT fire on this server-resolved path. A misconfigured per-KB override
    # (e.g. exact_boost: -1 in tenant_config) would otherwise reach OpenSearch as a negative
    # `boost`, which it rejects — erroring every query for that KB while the flag is on.
    return query.model_copy(update={"exact_boost": max(0.0, boost)})


async def _with_chem_normalization(query: RetrievalQuery) -> RetrievalQuery:
    """Resolve the canonical SMILES/InChIKey of any molecule the query names (Slice 6, Phase C) and
    stamp them on the DEDICATED ``chem_terms`` field, so an equivalent-but-differently-drawn molecule
    matches the canonical tokens ingest indexed. The terms are carried on their own field — NOT
    appended to ``query.text`` — so ONLY the sparse leg sees them: an InChIKey hash is pure noise to
    the dense embedder, the bge-sparse/ColBERT legs and the cross-encoder reranker, which all consume
    ``query.text`` (left exactly as-is). No-op — returns the query unchanged — unless the triple gate
    holds for the KB and a molecule was detected + canonicalized; flag off / non-chemistry KB ⇒
    byte-identical (no lookup, no model_copy)."""
    if not await _chem_normalize_on(query.tenant_id):
        return query
    terms = canonical_terms(query.text, get_molecule_normalizer())
    if not terms:
        return query
    return query.model_copy(update={"chem_terms": tuple(terms)})


async def _with_multivector_flags(query: RetrievalQuery) -> RetrievalQuery:
    """Stamp the per-KB Slice-2b leg flags onto the query so the retriever fuses in the optional
    bge learned-sparse leg and/or ColBERT MaxSim re-scoring for THIS KB. No-op — returns the query
    unchanged — unless the master ``multivector_enabled`` flag is on AND the KB enabled at least one
    leg. The retriever still requires the KB's embedder + store to carry the multivector capability
    (a dense-only KB no-ops regardless). Flag OFF ⇒ byte-identical (no lookup, no model_copy)."""
    settings = get_settings()
    if not settings.multivector_enabled:
        return query
    overrides = await kb_threshold_overrides(query.tenant_id)
    sparse = bool(overrides.get("multivector_sparse_enabled", 0.0))
    colbert = bool(overrides.get("multivector_colbert_enabled", 0.0))
    if not (sparse or colbert):
        return query
    prefetch = int(overrides.get("colbert_prefetch_pool") or settings.colbert_prefetch_pool)
    return query.model_copy(
        update={
            "multivector_sparse": sparse,
            "multivector_colbert": colbert,
            "colbert_prefetch": max(1, prefetch),
        }
    )


async def _with_glossary_expansion(query: RetrievalQuery) -> RetrievalQuery:
    """Bridge plain wording to canonical SOP/GMP terminology BEFORE retrieval (glossary/synonym
    expansion). The canonical terms are appended to ``query.text`` — the field BOTH the dense
    embedder and the cross-encoder reranker read — so a casual question retrieves the right SOP
    chunk. The user's ORIGINAL question is preserved on ``original_text`` for the user-facing
    answer + audit. No-op — returns the query unchanged (byte-identical) — unless the master flag
    is on AND the glossary actually expanded the text; off / nothing matches ⇒ byte-identical."""
    if not get_settings().glossary_expansion_enabled:
        return query
    settings = get_settings()
    expanded = expand_glossary(query.text, load_glossary(settings.glossary_path), mode=settings.glossary_mode)
    if expanded == query.text:
        return query
    return query.model_copy(update={"text": expanded, "original_text": query.original_text or query.text})


async def _guardrail_gate_flags(kb_id: str) -> tuple[bool | None, bool | None]:
    """Resolve the (faithfulness_gate, toxicity_filter) toggles for this KB from the Guardrail
    policy. Returns ``None`` for a toggle the policy does not set, so generation falls back to the
    platform setting. Best-effort — a lookup failure returns ``(None, None)`` (platform defaults)."""
    try:
        resolved = await guardrails_policy.effective_map(get_catalog(), kb_id)
        fg = resolved.get("faithfulness_gate")
        tx = resolved.get("toxicity_filter")
        return (bool(fg) if isinstance(fg, bool) else None, bool(tx) if isinstance(tx, bool) else None)
    except Exception as exc:
        _log.warning("guardrail_gate_flags_lookup_failed", kb_id=kb_id, error=str(exc)[:160])
        return (None, None)


async def _hitl_hold_enabled(kb_id: str) -> bool:
    """Whether a faithfulness-flagged answer is HELD for a human reviewer — the Guardrail
    ``human_review`` toggle, resolved top-down for this KB. Default ON (held). When an org turns
    it OFF via the Guardrails page, a flagged answer is returned to the caller directly instead of
    being queued — the org's explicit governance choice. Best-effort: any lookup failure keeps the
    safe default (hold)."""
    try:
        resolved = await guardrails_policy.effective_map(get_catalog(), kb_id)
        return bool(resolved.get("human_review", True))
    except Exception as exc:
        _log.warning("guardrail_human_review_lookup_failed", kb_id=kb_id, error=str(exc)[:160])
        return True


async def _fuse_structured_evidence(
    query: RetrievalQuery, retrieval: RetrievalResult, *, actor: str
) -> RetrievalResult:
    """Unified KB (documents + connected structured sources): when ``unified_kb_enabled``, gather an
    authoritative cited span from every source connected to the KB (SQL DB, NoSQL, REST API,
    SharePoint — see :mod:`unified`) and merge them with the document spans, so one grounded answer
    draws on documents AND every connected source (each its own citation). Fail-safe: no connected
    source / disabled / errors / empty -> the retrieval is returned unchanged (documents-only). The
    structured spans are score 1.0 so an exact structured answer alone still clears the abstain gate."""
    settings = get_settings()
    if not settings.unified_kb_enabled:
        return retrieval
    spans = await unified.gather_evidence(query.tenant_id, query.text, actor=actor)
    if not spans:
        return retrieval
    # Graph bridge (Option B): put the structured facts INTO the KB knowledge graph so they share
    # nodes with the documents (entity resolution via normalize_entity) and can be shown in contact
    # in GraphRAG. Additive + fail-safe — the graph never fails the answer.
    if settings.graph_bridge_enabled and settings.graphrag_enabled:
        try:
            await graph_bridge.bridge_spans_to_graph(
                query.tenant_id,
                spans,
                store=get_graph_store(),
                extractor=await wiring.resolve_graph_extractor(query.tenant_id),
            )
        except Exception as exc:
            _log.warning("graph_bridge_skipped", tenant_id=query.tenant_id, error=str(exc)[:160])
    _log.info("unified_kb_fused", tenant_id=query.tenant_id, doc_spans=len(retrieval.spans), source_spans=len(spans))
    # Order the merged spans by score DESCENDING. The generator keeps only the first top_k spans
    # (service.py: retrieval.spans[:top_k]); appending the authoritative score-1.0 structured spans
    # LAST let a KB with >= top_k weak document spans push them past the cut, so the model answered
    # "I don't know" on evidence it never saw. Highest score first -> the DB/source answer leads the
    # prompt and survives the slice.
    merged = sorted([*retrieval.spans, *spans], key=lambda s: s.score, reverse=True)
    return retrieval.model_copy(
        update={
            "spans": merged,
            "max_score": merged[0].score if merged else retrieval.max_score,
        }
    )


async def _cross_lingual_retrieve(query: RetrievalQuery) -> RetrievalResult:
    """Standard-path retrieval, optionally expanded across languages (Indic cross-lingual, FR-B1).

    When ``cross_lingual_enabled`` and the query's script differs from the corpus language, the
    query is translated via the tenant's OWN model and retrieval runs on BOTH the original and
    the translation (unioned) — so a Hindi/Marathi question reaches English content. Off or
    same-script: a single retrieve, byte-identical to the prior behaviour. A translation failure
    degrades to the original query (cross_lingual.expand is fail-safe), never failing retrieval."""
    settings = get_settings()
    svc = await wiring.retrieval_service_for(query.tenant_id)
    if not settings.cross_lingual_enabled or not cross_lingual.needs_translation(query.text):
        return await svc.retrieve(query)

    async def _translate(text: str, target_lang: str) -> str:
        gateway = await wiring.resolve_tenant_gateway(query.tenant_id)
        prompt = (
            f"Translate this search query to {target_lang}. "
            f"Output only the translation, no preamble or quotes.\n\nQuery: {text}"
        )
        return (await gateway.complete(prompt, max_tokens=128)).strip()

    variants = await cross_lingual.expand_query_cross_lingual(
        query.text,
        target_lang=settings.cross_lingual_target_lang,
        translate_fn=_translate,
        enabled=True,
    )
    if len(variants) == 1:
        return await svc.retrieve(query)
    results = [await svc.retrieve(query.model_copy(update={"text": v})) for v in variants]
    spans, max_score = merge_retrieved([r.spans for r in results], query.top_k)
    _log.info("cross_lingual_retrieved", tenant_id=query.tenant_id, variants=len(variants), merged=len(spans))
    return RetrievalResult(query=query, spans=spans, max_score=max_score, mode_used=get_retriever().mode)


async def _emit_receipt(audit: AuditService, answer: GroundedAnswer, *, actor: str) -> None:
    """Emit a signed trust receipt for a grounded answer (flag-gated; no-op when disabled).

    Imported locally: provenance -> compliance -> pipeline would form an import cycle at module
    load, so the dependency is deferred to call time (the house pattern for this seam)."""
    from svc_control_plane import provenance  # noqa: PLC0415

    await provenance.emit_answer_receipt(audit, answer, actor=actor)


# --- Semantic intent handlers (flag-gated: intent_routing_enabled) ------------------------------
# Both are NON-AUTHORITATIVE meta-answers that still satisfy the grounded-or-abstain invariant:
# a GROUNDED answer carries >=1 real Citation, an ABSTAINED one carries none.
_CATALOG_ABSTAIN = "This knowledge base has no documents you can access yet."
_SUMMARY_ABSTAIN = "There's nothing in this conversation to summarize yet."


def _doc_label(doc: DocumentEnvelope) -> str:
    """A human label for a catalog entry: its title/filename metadata, else the doc id."""
    return doc.metadata.get("title") or doc.metadata.get("filename") or doc.doc_id


def _doc_tags(doc: DocumentEnvelope) -> str:
    """A short comma tag list (category/year) for a catalog entry, empty when none."""
    return ", ".join(v for k in ("category", "year") if (v := doc.metadata.get(k)))


async def _guard_input(query: RetrievalQuery, *, actor: str, trace_id: str | None = None) -> GroundedAnswer | None:
    """Run the FR-G1 prompt-injection guard for a handler that bypasses ``_answer_query_impl``
    (intent routing on the conversation path). Returns an ABSTAINED answer when blocked (audited
    on a GUARDRAIL event), else None. Mirrors the guard at the top of ``_answer_query_impl`` so a
    block never reaches any handler."""
    if not get_settings().guardrails_enabled:
        return None
    tid = trace_id or str(uuid.uuid4())
    verdict = await wiring.input_guard().check(query.text)
    if verdict.allowed:
        return None
    await wiring.audit_service().record(
        tenant_id=query.tenant_id,
        kb_id=query.tenant_id,
        trace_id=tid,
        action=AuditAction.GUARDRAIL,
        actor=actor,
        payload={"stage": "input", "category": verdict.category, "reason": verdict.reason},
    )
    _log.warning("guardrail_blocked_input", tenant_id=query.tenant_id, category=verdict.category)
    return abstain(query.tenant_id, query.text, 0.0, reason="blocked_prompt_injection", trace_id=tid)


async def _answer_catalog(query: RetrievalQuery, *, actor: str, trace_id: str | None = None) -> GroundedAnswer:
    """Answer a "what documents/files/artifacts are in this KB?" question from the catalog
    inventory (intent routing) instead of an out-of-scope abstain.

    Visibility uses the SAME rule retrieval enforces: a document is listed only if the caller's
    groups entitle its source ACL AND the caller's clearance clears its classification. With >=1
    visible document → GROUNDED with one real :class:`Citation` per listed document (the docs are
    real, so the invariant holds truthfully); with none → ABSTAINED, no citations."""
    tid = trace_id or str(uuid.uuid4())
    audit: AuditService = wiring.audit_service()
    docs = await get_catalog().list_documents(query.tenant_id)
    visible = [
        d for d in docs if is_entitled(d.source_acl, query.acl) and is_cleared(d.classification, query.clearance)
    ]
    await audit.record(
        tenant_id=query.tenant_id,
        kb_id=query.tenant_id,
        trace_id=tid,
        action=AuditAction.QUERY,
        actor=actor,
        payload={"query": query.text, "intent": "catalog", "count": len(visible)},
    )
    trace = [AgentStep(step="intent", detail=f"Intent: catalog → {len(visible)} visible document(s)")]
    if not visible:
        return GroundedAnswer(
            tenant_id=query.tenant_id,
            query=query.text,
            status=AnswerStatus.ABSTAINED,
            answer=_CATALOG_ABSTAIN,
            citations=[],
            confidence=0.0,
            abstain_reason="catalog_empty",
            trace_id=tid,
            agent_trace=trace,
            structured={"intent": "catalog", "count": 0},
        )
    lines = ["Here are the documents in this knowledge base:", ""]
    citations: list[Citation] = []
    for doc in visible:
        label = _doc_label(doc)
        dtype = doc.metadata.get("doc_type") or doc.mime
        tags = _doc_tags(doc)
        lines.append(f"- {label} — {dtype}" + (f" [{tags}]" if tags else ""))
        citations.append(
            Citation(
                chunk_id=f"{doc.doc_id}#catalog",
                doc_id=doc.doc_id,
                source_uri=doc.uri,
                quote=f"{label} ({dtype})"[:200],
                score=1.0,
            )
        )
    return GroundedAnswer(
        tenant_id=query.tenant_id,
        query=query.text,
        status=AnswerStatus.GROUNDED,
        answer="\n".join(lines),
        citations=citations,
        confidence=1.0,
        trace_id=tid,
        agent_trace=trace,
        structured={"intent": "catalog", "count": len(visible)},
    )


def build_summary_prompt(pairs: list[tuple[str, str]]) -> str:
    """Prompt the KB's model to summarize the prior conversation turns (question + answer)."""
    convo = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in pairs)
    return (
        "Summarize the following conversation between a user and a document assistant in 2-4 "
        "concise sentences. Capture what the user asked and what was answered. Do NOT add facts "
        "that are not present.\n\nCONVERSATION:\n" + convo[:4000] + "\n\nSUMMARY:"
    )


def _prior_turns(msgs: list[Message]) -> tuple[list[tuple[str, str]], list[Citation]]:
    """Pair each user message with the assistant reply that immediately follows it, and collect
    those replies' citations. A trailing user turn with no assistant reply yet (the current
    summarize question) is naturally excluded."""
    pairs: list[tuple[str, str]] = []
    citations: list[Citation] = []
    for i, m in enumerate(msgs):
        nxt = msgs[i + 1] if i + 1 < len(msgs) else None
        if m.role == "user" and nxt is not None and nxt.role == "assistant":
            pairs.append((m.content, nxt.content))
            citations.extend(nxt.citations)
    return pairs, citations


def _dedup_citations(cits: list[Citation]) -> list[Citation]:
    """De-duplicate citations by (doc_id, chunk_id), preserving first-seen order."""
    seen: set[tuple[str, str]] = set()
    out: list[Citation] = []
    for c in cits:
        key = (c.doc_id, c.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


async def _answer_summary(
    tenant_id: str, conversation_id: str, question: str, *, actor: str, model: str | None = None
) -> GroundedAnswer:
    """Summarize the current conversation (intent routing, conversation path only).

    With >=1 prior turn → LLM-summarize them and cite the UNION of the prior assistant turns'
    citations (deduped) so the invariant holds and click-through still works; if no prior citation
    exists → ABSTAIN (never GROUNDED-with-0). With no prior turns → ABSTAIN with a clear message."""
    tid = str(uuid.uuid4())
    audit: AuditService = wiring.audit_service()
    store = wiring.conversation_store()
    msgs = await store.messages(tenant_id, conversation_id, get_settings().conversation_history_turns * 2 + 4)
    pairs, prior_citations = _prior_turns(msgs)
    await audit.record(
        tenant_id=tenant_id,
        trace_id=tid,
        action=AuditAction.QUERY,
        actor=actor,
        kb_id=tenant_id,
        payload={"query": question, "intent": "summarize", "turns": len(pairs)},
    )
    trace = [AgentStep(step="intent", detail=f"Intent: summarize → {len(pairs)} prior turn(s)")]
    if not pairs:
        return GroundedAnswer(
            tenant_id=tenant_id,
            query=question,
            status=AnswerStatus.ABSTAINED,
            answer=_SUMMARY_ABSTAIN,
            citations=[],
            confidence=0.0,
            abstain_reason="nothing_to_summarize",
            trace_id=tid,
            agent_trace=trace,
            structured={"intent": "summarize", "turns": 0},
        )
    summary = ""
    try:
        gateway = await wiring.resolve_tenant_gateway(tenant_id, model)
        summary = (
            await gateway.complete(
                build_summary_prompt(pairs), max_tokens=get_settings().llm_max_tokens, temperature=0.3
            )
        ).strip()
    except Exception as exc:  # a model failure must not fail the turn — fall back to a plain recap
        _log.warning("summary_model_failed_fallback", tenant_id=tenant_id, error=str(exc)[:120])
    if not summary:
        summary = "Here's a recap of this conversation:\n" + "\n".join(f"- You asked: {q}" for q, _ in pairs)
    citations = _dedup_citations(prior_citations)
    if not citations:
        # No prior turn carried a citation (all abstained): summarizing is honest but there is
        # nothing to cite, so ABSTAIN rather than violate grounded-or-abstain.
        return GroundedAnswer(
            tenant_id=tenant_id,
            query=question,
            status=AnswerStatus.ABSTAINED,
            answer=summary,
            citations=[],
            confidence=0.0,
            abstain_reason="no_cited_history",
            trace_id=tid,
            agent_trace=trace,
            structured={"intent": "summarize", "turns": len(pairs)},
        )
    return GroundedAnswer(
        tenant_id=tenant_id,
        query=question,
        status=AnswerStatus.GROUNDED,
        answer=summary,
        citations=citations,
        confidence=1.0,
        trace_id=tid,
        agent_trace=trace,
        structured={"intent": "summarize", "turns": len(pairs)},
    )


# --- count / filter / compare meta-handlers (flag-gated: intent_routing_enabled) ---------------
# Same shape as _answer_catalog/_answer_summary: visibility filter, real citations, trace + audit,
# and grounded-or-abstain-safe (a GROUNDED answer carries >=1 Citation; 0 matches → ABSTAINED).
_COUNT_ABSTAIN = "This knowledge base has no documents you can access yet."
_COMPARE_PER_DOC_TOP_K = 4  # small per-document evidence pool for a two-way contrast
_COMPARE_DOC_COUNT = 2  # a comparison contrasts exactly two documents; anything else falls back to rag


def _visible_docs(docs: list[DocumentEnvelope], query: RetrievalQuery) -> list[DocumentEnvelope]:
    """The documents the caller may see — the SAME rule retrieval and ``_answer_catalog`` enforce
    (ACL group entitlement AND classification clearance)."""
    return [d for d in docs if is_entitled(d.source_acl, query.acl) and is_cleared(d.classification, query.clearance)]


def _catalog_citation(doc: DocumentEnvelope) -> Citation:
    """One real citation per listed document (mirrors ``_answer_catalog``'s citation assembly)."""
    label = _doc_label(doc)
    dtype = doc.metadata.get("doc_type") or doc.mime
    return Citation(
        chunk_id=f"{doc.doc_id}#catalog",
        doc_id=doc.doc_id,
        source_uri=doc.uri,
        quote=f"{label} ({dtype})"[:200],
        score=1.0,
    )


def _count_breakdown(docs: list[DocumentEnvelope]) -> list[str]:
    """A short by-doc_type breakdown (falling back to by-year) when the metadata is present; empty
    when neither is tagged."""
    by_type = Counter(t for d in docs if (t := d.metadata.get("doc_type")))
    if by_type:
        return [f"- {t}: **{n}**" for t, n in sorted(by_type.items())]
    by_year = Counter(y for d in docs if (y := d.metadata.get("year")))
    if by_year:
        return [f"- {y}: **{n}**" for y, n in sorted(by_year.items())]
    return []


async def _answer_count(query: RetrievalQuery, *, actor: str, trace_id: str | None = None) -> GroundedAnswer:
    """Answer "how many documents / number of files" — catalog-with-a-number. N = visible docs,
    with a short by-doc_type (or by-year) breakdown when the metadata is present; one real citation
    per counted document. Empty/no-access KB → ABSTAINED (never GROUNDED-with-0)."""
    tid = trace_id or str(uuid.uuid4())
    audit: AuditService = wiring.audit_service()
    docs = await get_catalog().list_documents(query.tenant_id)
    visible = _visible_docs(docs, query)
    await audit.record(
        tenant_id=query.tenant_id,
        kb_id=query.tenant_id,
        trace_id=tid,
        action=AuditAction.QUERY,
        actor=actor,
        payload={"query": query.text, "intent": "count", "count": len(visible)},
    )
    trace = [AgentStep(step="intent", detail=f"Intent: count → {len(visible)} visible document(s)")]
    if not visible:
        return GroundedAnswer(
            tenant_id=query.tenant_id,
            query=query.text,
            status=AnswerStatus.ABSTAINED,
            answer=_COUNT_ABSTAIN,
            citations=[],
            confidence=0.0,
            abstain_reason="catalog_empty",
            trace_id=tid,
            agent_trace=trace,
            structured={"intent": "count", "count": 0},
        )
    lines = [f"This knowledge base contains **{len(visible)}** documents."]
    breakdown = _count_breakdown(visible)
    if breakdown:
        lines.append("")
        lines.extend(breakdown)
    return GroundedAnswer(
        tenant_id=query.tenant_id,
        query=query.text,
        status=AnswerStatus.GROUNDED,
        answer="\n".join(lines),
        citations=[_catalog_citation(d) for d in visible],
        confidence=1.0,
        trace_id=tid,
        agent_trace=trace,
        structured={"intent": "count", "count": len(visible)},
    )


def _doc_matches_criteria(doc: DocumentEnvelope, criteria: dict[str, str]) -> bool:
    """Whether a document satisfies ALL parsed filter criteria (year/doc_type/recency)."""
    if "year" in criteria:
        year = doc.metadata.get("year") or (doc.metadata.get("doc_date") or "")[:4]
        if year != criteria["year"]:
            return False
    if "doc_type" in criteria and doc.metadata.get("doc_type") != criteria["doc_type"]:
        return False
    if "recency_days" in criteria:
        cutoff = datetime.now(UTC) - timedelta(days=int(criteria["recency_days"]))
        if doc.ingested_at < cutoff:
            return False
    return True


def _describe_criteria(criteria: dict[str, str]) -> str:
    """A human phrase for the parsed criteria, for the answer/abstain message."""
    parts: list[str] = []
    if "doc_type" in criteria:
        parts.append(f"type {criteria['doc_type']}")
    if "year" in criteria:
        parts.append(f"year {criteria['year']}")
    if "recency_days" in criteria:
        parts.append(f"ingested in the last {criteria['recency_days']} days")
    return ", ".join(parts) if parts else "the given criteria"


async def _answer_filter(query: RetrievalQuery, *, actor: str, trace_id: str | None = None) -> GroundedAnswer:
    """Answer "which documents from 2026 / policies of type agreement / latest docs" — the visible
    inventory narrowed by criteria extracted heuristically from the query.

    No parseable criteria → fall back to ``_answer_catalog`` (list all), since an unparseable filter
    is not a no-match. 0 matches → ABSTAINED (never GROUNDED-with-0); >=1 match → GROUNDED, citing
    exactly the matched documents."""
    tid = trace_id or str(uuid.uuid4())
    criteria = extract_filter_criteria(query.text)
    if not criteria:
        return await _answer_catalog(query, actor=actor, trace_id=tid)
    audit: AuditService = wiring.audit_service()
    docs = await get_catalog().list_documents(query.tenant_id)
    visible = _visible_docs(docs, query)
    matched = [d for d in visible if _doc_matches_criteria(d, criteria)]
    desc = _describe_criteria(criteria)
    await audit.record(
        tenant_id=query.tenant_id,
        kb_id=query.tenant_id,
        trace_id=tid,
        action=AuditAction.QUERY,
        actor=actor,
        payload={"query": query.text, "intent": "filter", "criteria": desc, "count": len(matched)},
    )
    trace = [AgentStep(step="intent", detail=f"Intent: filter ({desc}) → {len(matched)} document(s)")]
    if not matched:
        return GroundedAnswer(
            tenant_id=query.tenant_id,
            query=query.text,
            status=AnswerStatus.ABSTAINED,
            answer=f"No documents match {desc}.",
            citations=[],
            confidence=0.0,
            abstain_reason="filter_no_match",
            trace_id=tid,
            agent_trace=trace,
            structured={"intent": "filter", "count": 0},
        )
    lines = [f"Documents matching {desc}:", ""]
    citations: list[Citation] = []
    for doc in matched:
        label = _doc_label(doc)
        dtype = doc.metadata.get("doc_type") or doc.mime
        tags = _doc_tags(doc)
        lines.append(f"- {label} — {dtype}" + (f" [{tags}]" if tags else ""))
        citations.append(_catalog_citation(doc))
    return GroundedAnswer(
        tenant_id=query.tenant_id,
        query=query.text,
        status=AnswerStatus.GROUNDED,
        answer="\n".join(lines),
        citations=citations,
        confidence=1.0,
        trace_id=tid,
        agent_trace=trace,
        structured={"intent": "filter", "count": len(matched)},
    )


async def _answer_compare(
    query: RetrievalQuery, *, actor: str, model: str | None = None, trace_id: str | None = None
) -> GroundedAnswer:
    """Answer "difference between A and B / compare A and B" by contrasting two NAMED documents.

    CONSERVATIVE — never fabricates a comparison: it proceeds only when it confidently identifies
    EXACTLY two visible documents (fuzzy token overlap on their labels); anything else falls back to
    normal ``rag`` retrieval. With two, it retrieves the top spans from EACH document (one
    ``doc_ids``-scoped retrieval per doc) and runs them through the UNCHANGED grounded generation
    path (``build_prompt``'s grounded rules + numbered SOURCES from both docs), so grounded-or-
    abstain, faithfulness routing and citation assembly all apply. If either document yields no
    spans → fall back to ``rag``."""
    tid = trace_id or str(uuid.uuid4())
    docs = await get_catalog().list_documents(query.tenant_id)
    visible = _visible_docs(docs, query)
    targets = match_compare_targets(query.text, [(d.doc_id, _doc_label(d)) for d in visible])
    if len(targets) != _COMPARE_DOC_COUNT:
        _log.info("compare_fallback_rag", tenant_id=query.tenant_id, matched=len(targets))
        return await _answer_query_impl(query, actor=actor, model=model, skip_intent=True)

    svc = await wiring.retrieval_service_for(query.tenant_id)
    per_doc_k = min(query.top_k, _COMPARE_PER_DOC_TOP_K)
    spans: list[RetrievedSpan] = []
    for doc_id in targets:
        result = await svc.retrieve(query.model_copy(update={"doc_ids": (doc_id,), "top_k": per_doc_k}))
        if not result.spans:
            _log.info("compare_fallback_no_spans", tenant_id=query.tenant_id, doc_id=doc_id)
            return await _answer_query_impl(query, actor=actor, model=model, skip_intent=True)
        spans.extend(result.spans)

    combined = RetrievalResult(
        query=query,
        spans=spans,
        max_score=max((s.score for s in spans), default=0.0),
        mode_used=get_retriever().mode,
    )
    audit: AuditService = wiring.audit_service()
    await audit.record(
        tenant_id=query.tenant_id,
        kb_id=query.tenant_id,
        trace_id=tid,
        action=AuditAction.QUERY,
        actor=actor,
        payload={"query": query.text, "intent": "compare", "doc_ids": list(targets), "count": len(spans)},
    )
    overrides = await kb_threshold_overrides(query.tenant_id)
    faith_on, guard_on = await _guardrail_gate_flags(query.tenant_id)
    answer = await wiring.generation_service().answer(
        combined,
        actor=actor,
        trace_id=tid,
        model=model,
        threshold=overrides.get("abstain_threshold"),
        faithful_threshold=overrides.get("faithful_threshold"),
        flag_threshold=overrides.get("flag_threshold"),
        faithfulness_enabled=faith_on,
        output_guard_enabled=guard_on,
        persona=await kb_persona(query.tenant_id),
        kb_id=query.tenant_id,
    )
    trace = [AgentStep(step="intent", detail=f"Intent: compare → {targets[0]} vs {targets[1]}")]
    if answer.agent_trace:
        trace.extend(answer.agent_trace)
    if answer.faithfulness_score is not None:
        verify_detail = f"Faithfulness {answer.faithfulness_score:.2f} → {answer.status.value}"
    else:
        verify_detail = f"Grounding check → {answer.status.value}"
    trace.append(AgentStep(step="verify", detail=verify_detail))
    answer = answer.model_copy(update={"agent_trace": trace})
    await _emit_receipt(audit, answer, actor=actor)
    return answer


_DOC_SUMMARY_ABSTAIN = "This knowledge base has no documents you can access yet."
_DOC_SUMMARY_NO_GROUNDING = "I couldn't ground a summary for any document in this knowledge base."
# Max concurrent per-doc generation calls in a doc_summary answer — one LLM call per document, run
# in parallel so an N-doc summary is ~one round-trip, not N. Kept small to stay gentle on the shared
# generation endpoint (doc count is already capped by ``doc_summary_max_docs``).
_DOC_SUMMARY_CONCURRENCY = 4


def _shift_citation_markers(text: str, offset: int) -> str:
    """Renumber a per-doc answer's ``[n]`` citation markers by ``offset`` so they line up with the
    combined union citation list when several per-doc summaries are assembled into one answer."""
    return re.sub(r"\[(\d+)\]", lambda m: f"[{int(m.group(1)) + offset}]", text)


_CITATION_MARKER_RE = re.compile(r"\[\d+\]")


def _dedupe_extract_lines(sections: list[str]) -> list[str]:
    """Drop near-identical lines across map-reduce batch extracts, preserving citation markers.

    A big document map-reduced batch-by-batch produces repetitive scaffolding lines ("The document
    covers…") across batches. Two lines are "the same" when they match after stripping ``[n]``
    citation markers, whitespace and case — so a duplicate whose content is already cited by an
    earlier line's marker is dropped without losing grounding (the kept line still cites it). Never
    an ungrounded synthesis pass — pure line-level de-duplication that keeps the surviving markers."""
    seen: set[str] = set()
    out: list[str] = []
    for section in sections:
        kept: list[str] = []
        for line in section.splitlines():
            norm = _CITATION_MARKER_RE.sub("", line).strip().lower()
            if norm and norm in seen:
                continue
            if norm:
                seen.add(norm)
            kept.append(line)
        joined = "\n".join(kept).strip()
        if joined:
            out.append(joined)
    return out


def _aggregate_doc_completeness(summarized: int, incomplete_docs: list[str], *, enabled: bool) -> Completeness | None:
    """Fold the per-doc completeness signals of a doc-summary into ONE combined signal.

    Returns None when the guard is disabled (so the flag-OFF path stays byte-identical). Otherwise
    the combined answer is complete iff EVERY per-doc summary was complete; when some were cut off it
    names them. Coverage is not measured here — a per-doc summary intentionally omits most chunks —
    so ``expected`` is the number of documents summarized and ``covered`` the ones that came back
    whole, and ``truncated`` reflects that at least one per-doc summary was cut off."""
    if not enabled:
        return None
    return Completeness(
        complete=not incomplete_docs,
        reason=(
            None
            if not incomplete_docs
            else f"{len(incomplete_docs)} document summary(ies) may be cut off: {', '.join(incomplete_docs)}"
        ),
        covered=summarized - len(incomplete_docs),
        expected=summarized,
        truncated=bool(incomplete_docs),
    )


async def _answer_doc_summary(
    query: RetrievalQuery, *, actor: str, model: str | None = None, trace_id: str | None = None
) -> GroundedAnswer:
    """Summarize EACH document in the KB (intent routing) — what a "summarize/explain each doc"
    request actually wants, as opposed to ``catalog``'s bare NAME list.

    Bounded by ``doc_summary_max_docs`` (it is N generation calls, one per doc; more docs → note
    "showing N of M"). FOR EACH visible doc: a ``doc_ids``-scoped retrieval of its
    representative chunks (``doc_summary_chunks_per_doc``) run through the UNCHANGED grounded
    generation path (``build_prompt`` grounded rules, depth="thorough"), so each per-doc summary
    cites THAT doc's real chunks. The per-doc sections are assembled into ONE GroundedAnswer whose
    citations are the union across docs (markers renumbered to the union). Grounded-or-abstain is
    preserved: GROUNDED only when >=1 doc produced a grounded, cited summary; an empty/no-access KB
    (or no doc could be grounded) → ABSTAINED, never GROUNDED-with-0; a doc with no extractable
    chunks gets a one-line note but never fabricates a citation."""
    tid = trace_id or str(uuid.uuid4())
    settings = get_settings()
    audit: AuditService = wiring.audit_service()
    docs = await get_catalog().list_documents(query.tenant_id)
    visible = _visible_docs(docs, query)
    total = len(visible)
    selected = visible[: settings.doc_summary_max_docs]
    truncated = total > len(selected)
    await audit.record(
        tenant_id=query.tenant_id,
        kb_id=query.tenant_id,
        trace_id=tid,
        action=AuditAction.QUERY,
        actor=actor,
        payload={"query": query.text, "intent": "doc_summary", "count": len(selected), "total": total},
    )
    trace = [AgentStep(step="intent", detail=f"doc_summary → {len(selected)} of {total} documents")]
    if not visible:
        return GroundedAnswer(
            tenant_id=query.tenant_id,
            query=query.text,
            status=AnswerStatus.ABSTAINED,
            answer=_DOC_SUMMARY_ABSTAIN,
            citations=[],
            confidence=0.0,
            abstain_reason="catalog_empty",
            trace_id=tid,
            agent_trace=trace,
            structured={"intent": "doc_summary", "count": 0},
        )

    svc = await wiring.retrieval_service_for(query.tenant_id)
    gen = wiring.generation_service()
    overrides = await kb_threshold_overrides(query.tenant_id)
    faith_on, guard_on = await _guardrail_gate_flags(query.tenant_id)
    persona = await kb_persona(query.tenant_id)
    per_doc_k = settings.doc_summary_chunks_per_doc
    # ONE shared guard budget across the whole doc-summary fan-out (per-KB overridable), so the
    # completeness guard's extra calls summed across N docs can never blow up. Coverage is
    # DISABLED per-doc: a per-doc summary intentionally omits most of its chunks, so the coverage
    # loop would fire its full ceiling on every doc and never resolve — pure waste. Truncation
    # continuation still runs (a cut-off per-doc summary is a real defect).
    guard_budget = GuardBudget(
        int(_gc)
        if (_gc := overrides.get("completeness_max_guard_calls")) is not None
        else settings.completeness_max_guard_calls
    )

    async def _summarize_one(doc: DocumentEnvelope) -> tuple[str, GroundedAnswer | None]:
        label = _doc_label(doc)
        # Summarize from THIS doc's OWN excerpts, driven by an excerpt-scoped summary query — NOT
        # the user's meta-instruction ("explain each doc in detail"). Reusing the meta-text as the
        # semantic search string neither selects representative chunks nor holds the generator to
        # what was retrieved: on a large doc the model generalises past the handful of spans and
        # faithfulness (grounded-or-abstain) then rejects the whole summary ("could not produce a
        # grounded summary"). A summary query fixes both: it retrieves the doc's representative
        # chunks AND anchors the answer to what those excerpts actually say, so a genuine per-doc
        # summary passes verification instead of being thrown out.
        summary_query = query.model_copy(
            update={
                "text": f'Summarise the key points of the document titled "{label}" using only the excerpts provided.',
                "doc_ids": (doc.doc_id,),
                "top_k": per_doc_k,
            }
        )
        result = await svc.retrieve(summary_query)
        if not result.spans:
            return label, None
        # threshold=0.0: the task is "summarize THIS doc", so every chunk of it is relevant — the
        # retrieval-confidence gate (a Q&A relevance heuristic) must not abstain here. Faithfulness
        # verification + real citations still enforce grounded-or-abstain on the per-doc summary.
        per_doc = await gen.answer(
            result,
            actor=actor,
            trace_id=tid,
            model=model,
            threshold=0.0,
            faithful_threshold=overrides.get("faithful_threshold"),
            flag_threshold=overrides.get("flag_threshold"),
            faithfulness_enabled=faith_on,
            output_guard_enabled=guard_on,
            persona=persona,
            depth="thorough",
            kb_id=query.tenant_id,
            completeness_coverage=False,  # intentionally-lossy summary → truncation guard only
            guard_budget=guard_budget,  # shared cap across all docs
        )
        return label, per_doc

    # One generation call per doc — run them concurrently (bounded) so an N-doc summary costs one
    # round-trip of latency, not N. Results are assembled in `selected` order below so citation
    # renumbering stays deterministic regardless of which doc's call finishes first.
    sem = asyncio.Semaphore(min(len(selected), _DOC_SUMMARY_CONCURRENCY))

    async def _guarded(doc: DocumentEnvelope) -> tuple[str, GroundedAnswer | None]:
        async with sem:
            return await _summarize_one(doc)

    per_doc_results = await asyncio.gather(*(_guarded(doc) for doc in selected))

    sections: list[str] = []
    citations: list[Citation] = []
    summarized = 0
    for label, per_doc in per_doc_results:
        if per_doc is None:
            sections.append(f"**{label}** — no extractable text to summarize.")
            continue
        if per_doc.status is not AnswerStatus.GROUNDED or not per_doc.citations:
            sections.append(f"**{label}** — could not produce a grounded summary.")
            continue
        summary = _shift_citation_markers(per_doc.answer.strip(), len(citations))
        sections.append(f"**{label}** — {summary}")
        citations.extend(per_doc.citations)
        summarized += 1
    # Propagate each per-doc completeness signal (we pay for it, so surface it): a truncated per-doc
    # summary makes the COMBINED answer incomplete, naming which doc(s).
    incomplete_docs = [
        label
        for label, per_doc in per_doc_results
        if per_doc is not None
        and per_doc.status is AnswerStatus.GROUNDED
        and per_doc.citations
        and per_doc.completeness is not None
        and not per_doc.completeness.complete
    ]

    if summarized == 0 or not citations:
        return GroundedAnswer(
            tenant_id=query.tenant_id,
            query=query.text,
            status=AnswerStatus.ABSTAINED,
            answer=_DOC_SUMMARY_NO_GROUNDING,
            citations=[],
            confidence=0.0,
            abstain_reason="doc_summary_no_grounding",
            trace_id=tid,
            agent_trace=trace,
            structured={"intent": "doc_summary", "count": len(selected)},
        )

    header = f"Summary of {summarized} document{'s' if summarized != 1 else ''} in this knowledge base:"
    if truncated:
        header += f" (showing {len(selected)} of {total})"
    # Aggregate the per-doc completeness the guard produced (only when the guard is enabled, so the
    # flag-OFF path stays byte-identical): the combined answer is incomplete if ANY per-doc summary
    # was cut off, naming which. Coverage is not measured (a per-doc summary is intentionally lossy).
    combined_completeness = _aggregate_doc_completeness(
        summarized, incomplete_docs, enabled=settings.completeness_guard_enabled
    )
    answer = GroundedAnswer(
        tenant_id=query.tenant_id,
        query=query.text,
        status=AnswerStatus.GROUNDED,
        answer=header + "\n\n" + "\n\n".join(sections),
        citations=citations,
        confidence=1.0,
        trace_id=tid,
        agent_trace=trace,
        structured={"intent": "doc_summary", "count": summarized, "total": total},
        completeness=combined_completeness,
    )
    await _emit_receipt(audit, answer, actor=actor)
    return answer


_WHOLE_DOC_ABSTAIN = "There is no accessible content in that document to summarize."
_WHOLE_DOC_NO_GROUNDING = "I couldn't ground a summary of that document from its contents."
# Headroom (tokens) beyond the answer budget for the grounding prompt scaffolding (instructions,
# per-source fences, question) when sizing a single full-context read against the model's window.
_WHOLE_DOC_PROMPT_RESERVE_TOKENS = 2000
# A single full read is chosen only when the estimate is within this fraction of the effective
# budget — the chars/4 heuristic under-counts, so leave margin before risking truncation.
_WHOLE_DOC_FULL_READ_SAFETY = 0.9
# The minimum match count that triggers the combined multi-document whole-doc read (below it is a
# single-doc read; above the per-KB cap it falls through to RAG with clarification options).
_MULTI_TARGET_MIN = 2


def _chunk_to_span(chunk: Chunk) -> RetrievedSpan:
    """A document chunk as an UNSCORED retrieval span for grounded generation.

    Unscored (only a reranker writes a score): the chunk is present because it BELONGS to the
    document being summarized, not because a reranker judged it relevant to a question — so it
    carries no score and the whole-doc path gates on ``threshold=0.0`` instead of the relevance
    gate. Downstream ``_prompt_spans`` keeps every unscored span, so all of a batch's chunks reach
    the prompt (the batch is bounded by ``whole_doc_summary_batch_chunks``)."""
    return RetrievedSpan(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        page=chunk.page,
        section=chunk.metadata.get(META_SECTION),
        span=chunk.span,
        source_uri=chunk.source_uri,
        category=chunk.metadata.get(META_CATEGORY),
        acl=list(chunk.acl),
        classification=chunk.classification,
        modality=chunk.modality,
    )


async def _run_whole_doc_batches(
    batches: list[list[Chunk]],
    *,
    query: RetrievalQuery,
    map_text: str,
    gen: Any,
    actor: str,
    tid: str,
    model: str | None,
    overrides: dict[str, Any],
    faith_on: bool | None,
    guard_on: bool | None,
    persona: str | None,
    guard_budget: GuardBudget,
    cap: int,
) -> tuple[list[GroundedAnswer | None], int]:
    """Run each chunk BATCH through the UNCHANGED grounded ``gen.answer`` seam CONCURRENTLY, bounded by
    ``cap`` (``asyncio.Semaphore``). The batches are independent (each summarises its own slice), so this
    turns N sequential ~30s model calls into ~ceil(N/cap) waves.

    ORDER-STABLE: ``asyncio.gather`` preserves the order of its input coroutine list, so results come
    back in ``batches`` order and the caller's reduce stays document-order-stable (citations renumber by
    the running count in that order). PER-BATCH FAIL-SAFE: a batch that raises yields ``None`` (contributes
    nothing, exactly like an ungrounded batch) and is counted — one bad batch never fails the whole
    answer. GUARD BUDGET under concurrency: the shared ``GuardBudget`` is mutated only through
    ``GuardBudget.take()``, which is synchronous (no ``await`` inside), so on the single-threaded asyncio
    loop concurrent batches cannot interleave mid-``take`` — no double-spend, no deadlock; if it is
    exhausted mid-wave the in-flight batches still finish and their grounded output is kept (best-effort).
    Returns ``(results_in_batch_order, error_count)``.
    """
    sem = asyncio.Semaphore(max(1, min(len(batches), cap)))
    errors = 0

    async def _one(batch: list[Chunk]) -> GroundedAnswer | None:
        nonlocal errors
        async with sem:
            spans = [_chunk_to_span(c) for c in batch]
            batch_query = query.model_copy(update={"text": map_text, "top_k": len(spans)})
            result = RetrievalResult(query=batch_query, spans=spans, max_score=0.0, mode_used=RetrievalMode.HYBRID)
            try:
                # threshold=0.0: every chunk of the target doc is in-scope; faithfulness + real citations
                # still enforce grounded-or-abstain. Coverage re-prompt off (a batch is a deliberate slice).
                answer: GroundedAnswer | None = await gen.answer(
                    result,
                    actor=actor,
                    trace_id=tid,
                    model=model,
                    threshold=0.0,
                    faithful_threshold=overrides.get("faithful_threshold"),
                    flag_threshold=overrides.get("flag_threshold"),
                    faithfulness_enabled=faith_on,
                    output_guard_enabled=guard_on,
                    persona=persona,
                    depth="thorough",
                    kb_id=query.tenant_id,
                    completeness_coverage=False,
                    guard_budget=guard_budget,
                )
                return answer
            except Exception as exc:  # per-batch fail-safe: contribute nothing, never fail the answer
                errors += 1
                _log.warning("whole_doc_batch_failed", tenant_id=query.tenant_id, error=str(exc)[:160])
                return None

    results = list(await asyncio.gather(*(_one(b) for b in batches)))
    return results, errors


async def _verify_whole_doc_coverage(
    *,
    query: RetrievalQuery,
    chunks: list[Chunk],
    sections: list[str],
    citations: list[Citation],
    cited_ids: set[str],
    withheld_ids: set[str],
    withheld: list[dict[str, object]],
    seen_fingerprints: set[str],
    faithful_gate: float,
    map_text: str,
    gen: Any,
    actor: str,
    tid: str,
    model: str | None,
    overrides: dict[str, Any],
    faith_on: bool | None,
    guard_on: bool | None,
    persona: str | None,
    guard_budget: GuardBudget,
    trace: list[AgentStep],
    audit: AuditService,
    max_loops: int,
    batch_size: int,
    map_concurrency: int,
) -> tuple[list[str], list[Citation], set[str], str]:
    """Tier 3 — the CONVERGENCE-BASED coverage-refine loop for the whole-document read (flag-gated).

    The "uncovered" set is simply ALL UNCITED chunks — no boilerplate/substantive pre-filter. Each pass
    RE-READS the uncited chunks through the SAME grounded seam the MAP uses (``gen.answer`` over those
    chunks as unscored spans, ``threshold=0.0``, faithfulness + real citations enforced), BATCHED by
    ``batch_size`` and run CONCURRENTLY (bounded by ``map_concurrency``) exactly like the map, and
    APPENDS any new grounded, cited points to the assembled answer. Passes stay SEQUENTIAL (each depends
    on the prior passes' cited set); only the batches WITHIN a pass run in parallel.

    A pass CONVERGES when it adds ZERO new distinct cited chunk ids — a fixed point. At convergence every
    remaining uncited chunk is one the grounded model could extract NO new cited point from: it is
    non-groundable (a footer / signature / glossary / index row) or already covered — i.e. genuinely
    "nothing missed", established WITHOUT any boilerplate regex. If a real obligation sat in an uncited
    chunk, the grounded pass WOULD have extracted and cited it, so the loop would NOT have converged —
    which is why convergence cannot hide a real obligation. The GROUNDED refine loop is the arbiter.

    Bounded by ``max_loops`` and the shared ``GuardBudget``; fully FAIL-SAFE (a raising batch contributes
    nothing, a raising pass keeps the pre-refine answer, never a 500). Returns ``(sections, citations,
    cited_ids, stop_reason)`` where ``stop_reason`` is one of ``converged`` | ``stopped_on_max_loops`` |
    ``stopped_on_budget`` | ``stopped_on_error`` — the caller drives the completeness signal off it.
    """
    passes = 0
    stop_reason = "stopped_on_max_loops"  # exhausted the loop while still adding new points
    for _loop in range(max(1, max_loops)):
        uncited = [c for c in chunks if c.chunk_id not in cited_ids and c.chunk_id not in withheld_ids]
        if not uncited:
            stop_reason = "converged"  # every chunk is cited — a fixed point
            break
        if guard_budget.remaining <= 0:
            stop_reason = "stopped_on_budget"
            break
        # Re-read the uncited chunks — batched like the MAP and run CONCURRENTLY (bounded).
        uncited_batches = [uncited[i : i + batch_size] for i in range(0, len(uncited), batch_size)]
        extracts, errors = await _run_whole_doc_batches(
            uncited_batches,
            query=query,
            map_text=map_text,
            gen=gen,
            actor=actor,
            tid=tid,
            model=model,
            overrides=overrides,
            faith_on=faith_on,
            guard_on=guard_on,
            persona=persona,
            guard_budget=guard_budget,
            cap=map_concurrency,
        )
        # REDUCE this pass's grounded extracts in BATCH ORDER — order-stable citation renumbering.
        pass_new_ids: set[str] = set()
        for extract in extracts:
            if extract is None or extract.status is not AnswerStatus.GROUNDED or not extract.citations:
                # Same disposition rule as the first-pass reduce: a batch the loop flags is
                # withheld with a recorded review and audit event, not silently dropped. The
                # fingerprint set makes a repeat across passes return the first review rather
                # than opening another, and `withheld_ids` keeps these chunks out of the next
                # pass's uncited set so the loop stops re-reading evidence it already rejected.
                if extract is not None and extract.status is AnswerStatus.FLAGGED_FOR_REVIEW:
                    review_id, outcome = await _withhold_flagged_batch(
                        extract,
                        query=query,
                        actor=actor,
                        tid=tid,
                        threshold=faithful_gate,
                        seen=seen_fingerprints,
                    )
                    if outcome != "deduplicated":
                        withheld.append(
                            {"review_id": review_id, "persistence_outcome": outcome, "chunks": len(extract.citations)}
                        )
                    withheld_ids.update(c.chunk_id for c in extract.citations)
                continue
            new = {c.chunk_id for c in extract.citations} - cited_ids
            if not new:
                continue
            sections.append(_shift_citation_markers(extract.answer.strip(), len(citations)))
            citations.extend(extract.citations)
            cited_ids = cited_ids | {c.chunk_id for c in extract.citations}
            pass_new_ids |= new
        if not pass_new_ids:
            # No new cited chunk this pass. If every batch that could have added one instead ERRORED we
            # cannot claim a true fixed point (not converged); otherwise the grounded model produced
            # nothing new on the uncited chunks → converged ("nothing missed").
            stop_reason = "stopped_on_error" if errors else "converged"
            break
        passes += 1

    detail = {
        "converged": f"Coverage loop → converged after {passes} pass(es)",
        "stopped_on_budget": f"Coverage loop → halted on budget after {passes} pass(es) (did not converge)",
        "stopped_on_error": f"Coverage loop → halted on error after {passes} pass(es) (did not converge)",
        "stopped_on_max_loops": f"Coverage loop → did not converge in {passes} pass(es)",
    }[stop_reason]
    trace.append(AgentStep(step="coverage", detail=detail))
    _log.info(
        "whole_doc_coverage_loop",
        tenant_id=query.tenant_id,
        passes=passes,
        stop_reason=stop_reason,
        covered=len(cited_ids),
    )
    try:
        await audit.record(
            tenant_id=query.tenant_id,
            kb_id=query.tenant_id,
            trace_id=tid,
            action=AuditAction.QUERY,
            actor=actor,
            payload={
                "query": query.text,
                "intent": "whole_document",
                "coverage_loop_passes": passes,
                "coverage_stop_reason": stop_reason,
                "covered": len(cited_ids),
            },
        )
    except Exception as exc:  # an audit failure must never break the answer
        _log.warning("whole_doc_coverage_audit_failed", tenant_id=query.tenant_id, error=str(exc)[:160])
    return sections, citations, cited_ids, stop_reason


def _evidence_fingerprint(trace_id: str, tenant_id: str, decision: str, chunk_ids: Sequence[str]) -> str:
    """Stable identity of one piece of EVIDENCE, for deduplicating its review.

    Deliberately NOT a function of the faithfulness score. The coverage loop re-reads uncited
    chunks and re-verifies the same batch, and a model is not deterministic — scoring 0.6511
    once and 0.6489 the next pass is the same evidence failing the same way, and keying on the
    score would open a second review for it. The chunk ids are the evidence; the score is a
    measurement of it, and belongs in the review's metadata rather than its identity."""
    key = "|".join([tenant_id, trace_id, decision, *sorted(dict.fromkeys(chunk_ids))])
    return hashlib.sha256(key.encode()).hexdigest()[:32]


async def _withhold_flagged_batch(
    batch_answer: GroundedAnswer,
    *,
    query: RetrievalQuery,
    actor: str,
    tid: str,
    threshold: float | None,
    seen: set[str],
) -> tuple[str | None, str]:
    """Give a flagged batch a RECORDED DISPOSITION before the composite discards it.

    The defect this closes: the reduce loops drop every batch that is not GROUNDED, and
    ``flag_for_review`` only CONSTRUCTS a flagged answer — persistence lives on the normal
    ``answer_query`` path, which the whole-document read never reaches. So a batch could fail
    grounding, vanish from the answer, queue no review and leave nothing in the audit trail
    saying it had ever existed. Measured on a forced flag: 4 of 8 batches flagged, zero reviews.

    Returns ``(review_id, outcome)`` where outcome is ``created``, ``deduplicated`` or
    ``failed``. **The audit event is emitted in all three**, failure included — an exclusion
    that cannot be queued is precisely the one the trail must not lose.

    ONLY for FLAGGED batches. An ABSTAINED batch is a correct outcome with nothing to review:
    it asserted nothing, so there is no claim for a human to adjudicate, and queueing it would
    bury the real flags under "we don't know".
    """
    chunk_ids = [c.chunk_id for c in batch_answer.citations]
    fingerprint = _evidence_fingerprint(tid, query.tenant_id, "faithfulness_flagged", chunk_ids)
    audit: AuditService = wiring.audit_service()
    review_id: str | None = None
    outcome = "deduplicated" if fingerprint in seen else "created"
    if fingerprint not in seen:
        try:
            review, outcome = await wiring.hitl_service().create(
                batch_answer,
                actor=actor,
                reason=REASON_FAITHFULNESS,
                warnings=list(batch_answer.unsupported_claims),
                faithfulness_threshold=threshold,
                model_version=batch_answer.model,
                evidence_fingerprint=fingerprint,
            )
            review_id = review.review_id
        except Exception as exc:  # the exclusion is still recorded — see the docstring
            outcome = "failed"
            _log.warning(
                "flagged_batch_review_persist_failed",
                tenant_id=query.tenant_id,
                trace_id=tid,
                error=str(exc)[:200],
            )
        seen.add(fingerprint)
    try:
        await audit.record(
            tenant_id=query.tenant_id,
            kb_id=query.tenant_id,
            trace_id=tid,
            action=AuditAction.FAITHFULNESS_BATCH_EXCLUDED,
            actor=actor,
            payload={
                "query": query.text,
                "evidence_fingerprint": fingerprint,
                "chunk_ids": chunk_ids[:50],
                "chunk_count": len(chunk_ids),
                "faithfulness_score": batch_answer.faithfulness_score,
                "faithfulness_threshold": threshold,
                "unsupported_claims": list(batch_answer.unsupported_claims)[:20],
                "exclusion_reason": "below_faithfulness_threshold",
                "review_id": review_id,
                "persistence_outcome": outcome,
            },
        )
    except Exception as exc:  # never cost the answer its remaining evidence
        _log.warning("flagged_batch_audit_failed", tenant_id=query.tenant_id, error=str(exc)[:160])
    return review_id, outcome


async def _answer_document_full_summary(  # noqa: PLR0912, PLR0915 - a linear map→reduce with guarded stages
    query: RetrievalQuery,
    target_doc: str,
    *,
    actor: str,
    model: str | None = None,
    trace_id: str | None = None,
    plan: QueryPlan | None = None,
    docs: list[DocumentEnvelope] | None = None,
) -> GroundedAnswer:
    """Map-reduce a grounded, fully-cited summary over ALL chunks of ONE document.

    The mechanism the agentic planner's ``whole_document`` strategy needs — comprehensive coverage
    of a single file — as opposed to ``_answer_doc_summary``'s per-doc top-k SAMPLING. It reads the
    WHOLE document, not a representative handful, so a "summarise the key obligations in this
    document" request over a 181-chunk file no longer returns 2 of 10 obligations:

    * **Fetch** ALL of ``target_doc``'s chunks in document order via the SAME catalog path the
      citation viewer uses (:meth:`CatalogAdapter.document_chunks`), ACL- and clearance-filtered to
      the caller. Beyond ``whole_doc_summary_max_chunks`` the read is capped and the completeness
      payload is marked ``truncated`` (honest partial coverage).
    * **MAP** — batch the chunks by ``whole_doc_summary_batch_chunks`` and run each batch through the
      UNCHANGED grounded generation path (``gen.answer`` over that batch's chunks-as-spans), so every
      batch extract is faithfulness-verified and cites its OWN real chunk ids. Coverage re-prompting
      is disabled per batch (a batch is deliberately partial); truncation continuation still applies.
    * **REDUCE** — assemble the grounded batch extracts into ONE answer, renumbering each batch's
      ``[n]`` citation markers into the combined union list (the SAME citation-assembly seam
      ``_answer_doc_summary`` uses). Grounded-or-abstain is preserved end-to-end: GROUNDED only when
      ≥1 batch produced a grounded, cited extract; every marker points at a real chunk; no chunk with
      no accessible content ever fabricates a citation.

    The completeness payload reflects the WHOLE document — ``expected`` = chunks read, ``covered`` =
    chunks in grounded batches — so the completeness guard passes HONESTLY only on full coverage.
    """
    tid = trace_id or str(uuid.uuid4())
    settings = get_settings()
    audit: AuditService = wiring.audit_service()

    # Reuse the caller's already-fetched document list when provided (FIX 7 — avoid a redundant
    # list_documents call on the whole_document path); fall back to a fetch for standalone callers.
    if docs is None:
        docs = await get_catalog().list_documents(query.tenant_id)
    envelope = next((d for d in docs if d.doc_id == target_doc), None)
    label = _doc_label(envelope) if envelope is not None else target_doc

    all_chunks = await get_catalog().document_chunks(query.tenant_id, target_doc)
    # ACL + classification entitlement — the SAME rule retrieval and the citation viewer enforce, so
    # the whole-doc read can never surface a chunk the caller could not retrieve normally.
    chunks = [c for c in all_chunks if is_entitled(c.acl, query.acl) and is_cleared(c.classification, query.clearance)]
    total_chunks = len(chunks)
    cap = settings.whole_doc_summary_max_chunks
    truncated_cap = total_chunks > cap
    if truncated_cap:
        chunks = chunks[:cap]

    # Prefer a SINGLE full-context read (the ENTIRE document in one generation call, like an uploaded
    # doc to a frontier chat model) when it fits the token budget (chars/4 heuristic) — no cross-batch
    # synthesis loss. Only a document too large for the model's window falls back to the batched
    # map-reduce. The effective budget is capped by the MODEL's real context window minus headroom for
    # the grounding prompt scaffolding + the answer tokens, so a single read can never exceed what the
    # model actually accepts (FIX 2); a doc within ~90% of that budget also goes to map-reduce.
    est_tokens = sum(len(c.text) for c in chunks) // 4
    reserve = settings.heavy_max_tokens + _WHOLE_DOC_PROMPT_RESERVE_TOKENS
    effective_budget = min(settings.whole_doc_full_read_token_budget, max(0, settings.llm_context_window - reserve))
    full_read = est_tokens <= int(effective_budget * _WHOLE_DOC_FULL_READ_SAFETY)
    read_mode = "full_read" if full_read else "map_reduce"

    plan_payload: dict[str, object] = (
        {"strategy": plan.strategy, "coverage": plan.coverage, "rationale": plan.rationale} if plan is not None else {}
    )
    await audit.record(
        tenant_id=query.tenant_id,
        kb_id=query.tenant_id,
        trace_id=tid,
        action=AuditAction.QUERY,
        actor=actor,
        payload={
            "query": query.text,
            "intent": "whole_document",
            "target_doc": target_doc,
            "chunks": len(chunks),
            "total_chunks": total_chunks,
            "read_mode": read_mode,
            **plan_payload,
        },
    )
    trace = [AgentStep(step="plan", detail=f"whole_document → reading all {len(chunks)} chunk(s) of {label}")]
    if plan is not None and plan.rationale:
        trace.append(AgentStep(step="rationale", detail=plan.rationale))

    if not chunks:
        return GroundedAnswer(
            tenant_id=query.tenant_id,
            query=query.text,
            status=AnswerStatus.ABSTAINED,
            answer=_WHOLE_DOC_ABSTAIN,
            citations=[],
            confidence=0.0,
            abstain_reason="doc_no_accessible_chunks",
            trace_id=tid,
            agent_trace=trace,
            structured={"intent": "whole_document", "target_doc": target_doc, "chunks": 0},
        )

    gen = wiring.generation_service()
    overrides = await kb_threshold_overrides(query.tenant_id)
    faith_on, guard_on = await _guardrail_gate_flags(query.tenant_id)
    persona = await kb_persona(query.tenant_id)
    # A full read is ONE batch of all chunks; the map-reduce splits by batch size. Both paths share
    # the same grounding/citation assembly and the same whole-document completeness accounting below.
    batch_size = settings.whole_doc_summary_batch_chunks
    batches = [chunks] if full_read else [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
    trace.append(
        AgentStep(
            step="read",
            detail=(
                f"Single full-context read of {len(chunks)} chunk(s) (~{est_tokens} tokens)"
                if full_read
                else f"Map-reduce over {len(batches)} batch(es) of ≤{batch_size} chunks (~{est_tokens} tokens)"
            ),
        )
    )
    # ONE shared guard budget across the whole map fan-out, so the completeness guard's continuation
    # calls summed across batches can never blow up (mirrors _answer_doc_summary's shared budget).
    guard_budget = GuardBudget(
        int(_gc)
        if (_gc := overrides.get("completeness_max_guard_calls")) is not None
        else settings.completeness_max_guard_calls
    )
    # A neutral extraction instruction — NOT the user's meta-phrasing — anchors each batch to its own
    # excerpts (the same reason _answer_doc_summary drives per-doc summaries off an excerpt-scoped
    # query): it holds the model to what the batch actually says, so a genuine extract passes
    # faithfulness instead of being thrown out for generalising past a handful of spans.
    map_text = (
        f"List the key points, obligations, and requirements stated in the excerpts of the document "
        f'titled "{label}", using only the excerpts provided.'
    )

    # MAP — each batch is an INDEPENDENT grounded read of its own chunk slice, so run them CONCURRENTLY
    # (bounded by whole_doc_map_concurrency) instead of one slow sequential call after another. Results
    # come back in BATCH ORDER (gather preserves input order) so the reduce below stays document-order-
    # stable; a batch that raises contributes nothing (per-batch fail-safe) — one bad batch never 500s.
    batch_answers, _map_errors = await _run_whole_doc_batches(
        batches,
        query=query,
        map_text=map_text,
        gen=gen,
        actor=actor,
        tid=tid,
        model=model,
        overrides=overrides,
        faith_on=faith_on,
        guard_on=guard_on,
        persona=persona,
        guard_budget=guard_budget,
        cap=settings.whole_doc_map_concurrency,
    )

    # --- REDUCE: assemble the grounded batch extracts into one cited answer ---
    sections: list[str] = []
    citations: list[Citation] = []
    cited_ids: set[str] = set()
    # Evidence that failed grounding and was withheld. `withheld_ids` is what makes flagged
    # evidence TERMINAL for the coverage loop below: its chunks must not re-enter the uncited
    # set, or the loop re-reads and re-flags the same material every pass (measured: one
    # flagged batch became four). "Flagged and withheld" is a different state from "not yet
    # evaluated", and only the second is worth another read.
    withheld: list[dict[str, object]] = []
    withheld_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    faithful_gate = float(overrides.get("faithful_threshold", settings.faithful_threshold))
    for _batch, per_batch in zip(batches, batch_answers, strict=True):
        if per_batch is None or per_batch.status is not AnswerStatus.GROUNDED or not per_batch.citations:
            if per_batch is not None and per_batch.status is AnswerStatus.FLAGGED_FOR_REVIEW:
                review_id, outcome = await _withhold_flagged_batch(
                    per_batch,
                    query=query,
                    actor=actor,
                    tid=tid,
                    threshold=faithful_gate,
                    seen=seen_fingerprints,
                )
                withheld.append(
                    {"review_id": review_id, "persistence_outcome": outcome, "chunks": len(per_batch.citations)}
                )
                withheld_ids.update(c.chunk_id for c in per_batch.citations)
            continue
        section = _shift_citation_markers(per_batch.answer.strip(), len(citations))
        sections.append(section)
        citations.extend(per_batch.citations)
        cited_ids.update(c.chunk_id for c in per_batch.citations)

    if not sections or not citations:
        return GroundedAnswer(
            tenant_id=query.tenant_id,
            query=query.text,
            status=AnswerStatus.ABSTAINED,
            answer=_WHOLE_DOC_NO_GROUNDING,
            citations=[],
            confidence=0.0,
            abstain_reason="doc_summary_no_grounding",
            trace_id=tid,
            agent_trace=trace,
            structured={"intent": "whole_document", "target_doc": target_doc, "chunks": len(chunks)},
        )

    # FIX 4 / Tier 3 completeness. Default (flag OFF): expected = every chunk read, covered = distinct
    # cited chunk ids — TODAY'S behaviour, byte-identical. Flag ON: run the CONVERGENCE-BASED refine loop
    # (all uncited chunks re-read through the grounded seam) and drive completeness off whether it reached
    # a fixed point — NOT off any boilerplate/substantive ratio.
    covered_chunks = len(cited_ids)
    expected_chunks = len(chunks)
    stop_reason = ""
    max_loops = int(
        _cvl
        if (_cvl := overrides.get("completeness_coverage_max_loops")) is not None
        else settings.completeness_coverage_max_loops
    )
    if settings.coverage_verify_enabled:
        try:
            sections, citations, cited_ids, stop_reason = await _verify_whole_doc_coverage(
                query=query,
                chunks=chunks,
                sections=sections,
                citations=citations,
                cited_ids=cited_ids,
                withheld_ids=withheld_ids,
                withheld=withheld,
                seen_fingerprints=seen_fingerprints,
                faithful_gate=faithful_gate,
                map_text=map_text,
                gen=gen,
                actor=actor,
                tid=tid,
                model=model,
                overrides=overrides,
                faith_on=faith_on,
                guard_on=guard_on,
                persona=persona,
                guard_budget=guard_budget,
                trace=trace,
                audit=audit,
                max_loops=max_loops,
                batch_size=batch_size,
                map_concurrency=settings.whole_doc_map_concurrency,
            )
        except Exception as exc:  # fail-safe: keep the pre-refine answer on any refine failure
            _log.warning("whole_doc_coverage_verify_failed", tenant_id=query.tenant_id, error=str(exc)[:200])
            stop_reason = "stopped_on_error"

    # FIX 6: drop near-identical lines across batch extracts (map-reduce over a big doc otherwise
    # yields N repetitive "the document covers…" blocks), matching on the line WITHOUT its [n]
    # markers so a duplicate whose content is already cited is dropped without losing grounding.
    body = "\n\n".join(_dedupe_extract_lines(sections))

    header = f"Summary of {label} (whole-document read of {len(chunks)} of {total_chunks} passage(s)):"
    if truncated_cap:
        header += f" — showing the first {cap} passages (document exceeds the whole-document read cap)"
    if settings.coverage_verify_enabled:
        # CONVERGENCE-based completeness. `covered` = distinct cited chunk ids. At convergence the loop
        # reached a fixed point — every remaining uncited chunk is one the GROUNDED model could extract
        # no new cited point from (non-groundable or already covered), so nothing real is missed and we
        # set `expected = covered` (complete reads true). A real obligation in an uncited chunk would
        # have been extracted+cited, so the loop would not have converged — this is why convergence
        # cannot hide an obligation, with NO boilerplate heuristic. When it did NOT converge, `expected`
        # includes the still-uncited chunks so the signal honestly shows more may be extractable.
        covered_chunks = len(cited_ids)
        remaining_uncited = len(chunks) - covered_chunks
        converged = stop_reason == "converged"
        expected_chunks = covered_chunks if converged else covered_chunks + remaining_uncited
        complete = converged and not truncated_cap
        if complete:
            _reason: str | None = None
        elif truncated_cap:
            _reason = "document truncated at the whole-document chunk cap; summary is partial"
        elif stop_reason == "stopped_on_budget":
            _reason = "coverage loop halted on budget before converging — more content may be extractable"
        elif stop_reason == "stopped_on_error":
            _reason = "coverage loop halted before converging — more content may be extractable"
        else:  # stopped_on_max_loops
            _reason = f"coverage loop did not converge within {max_loops} pass(es) — more content may be extractable"
    else:
        # Flag OFF — TODAY'S behaviour, byte-identical: expected = every chunk read, covered = distinct
        # cited chunk ids; complete only when every read chunk is cited and the cap did not truncate.
        complete = (not truncated_cap) and covered_chunks >= expected_chunks
        if complete:
            _reason = None
        elif truncated_cap:
            _reason = "document exceeds the whole-document read cap; summary is partial"
        else:
            _missed = expected_chunks - covered_chunks
            _reason = f"{_missed} of {expected_chunks} passage(s) not reflected in the summary"
    completeness = (
        Completeness(
            complete=complete,
            reason=_reason,
            covered=covered_chunks,
            expected=expected_chunks,
            truncated=truncated_cap,
        )
        if settings.completeness_guard_enabled
        else None
    )
    trace.append(AgentStep(step="reduce", detail=f"Combined {len(sections)} grounded batch extract(s)"))
    # WHAT THE COMPOSITE MAY CLAIM. Assembled from the batches that survived grounding, it is
    # the SURVIVING SUBSET — reporting that as plainly GROUNDED tells the reader everything in
    # the document was grounded, which is the one thing that was not established. So when
    # anything was withheld the composite carries FLAGGED_FOR_REVIEW, which already means
    # "cited but not fully supported → review" and which every consumer already renders; a
    # fourth AnswerStatus would be a coordinated release across the console and both SDKs for
    # an outcome the existing one expresses.
    #
    # The DISCLOSURE is built from the counts, never asked of the model — an answer cannot be
    # trusted to describe its own verification failures. And it says "pending review" only
    # where a review really persisted: claiming a reviewer has the material when the insert
    # failed would be a false governance claim, which is worse than saying nothing.
    queued = [w for w in withheld if w["persistence_outcome"] in ("created", "deduplicated")]
    unqueued = [w for w in withheld if w["persistence_outcome"] == "failed"]
    if withheld:
        n = len(withheld)
        passages = "passage" if n == 1 else "passages"
        if unqueued:
            notice = (
                f"**Partial answer:** {n} extracted {passages} could not be sufficiently verified "
                f"against the cited sources and {'has' if n == 1 else 'have'} been withheld. "
                f"{len(unqueued)} could NOT be queued for review — raise this with an operator."
            )
        else:
            notice = (
                f"**Partial answer:** {n} extracted {passages} could not be sufficiently verified "
                f"against the cited sources and {'has' if n == 1 else 'have'} been withheld pending "
                f"review. The information above is limited to the evidence that passed verification."
            )
        body = body + "\n\n" + notice
    answer = GroundedAnswer(
        tenant_id=query.tenant_id,
        query=query.text,
        status=AnswerStatus.FLAGGED_FOR_REVIEW if withheld else AnswerStatus.GROUNDED,
        answer=header + "\n\n" + body,
        citations=citations,
        confidence=1.0,
        trace_id=tid,
        agent_trace=trace,
        review_id=next((str(w["review_id"]) for w in queued if w["review_id"]), None),
        structured={
            "intent": "whole_document",
            "target_doc": target_doc,
            "chunks": len(chunks),
            "covered": covered_chunks,
            "read_mode": read_mode,
            # The aggregate detail, in the additive channel rather than a new status value.
            "withheld_batches": len(withheld),
            "withheld_reviews": [w["review_id"] for w in queued if w["review_id"]],
            "withheld_unqueued": len(unqueued),
            "retained_batches": len(sections),
        },
        completeness=completeness,
    )
    await _emit_receipt(audit, answer, actor=actor)
    return answer


async def _answer_documents_full_summary(
    query: RetrievalQuery,
    target_docs: list[str],
    *,
    actor: str,
    model: str | None = None,
    trace_id: str | None = None,
    plan: QueryPlan | None = None,
    docs: list[DocumentEnvelope],
) -> GroundedAnswer:
    """Whole-document read of SEVERAL matched documents, combined into ONE grounded answer.

    The RAG-faithful answer to an AMBIGUOUS target (e.g. "Vendor Agreement" → the 2023 AND 2024
    files): rather than guessing one and hiding the rest, summarise EACH matched document and return
    one answer with a clearly-labelled section per document. Each section is produced by the SAME
    single-doc whole-document mechanism (:func:`_answer_document_full_summary`), so every section is
    grounded-or-abstain over its own real chunks; a document that grounds nothing is NOTED, never
    fabricated. Citations are attributed per document by renumbering each doc's ``[n]`` markers into
    the combined union list (the same seam the doc-summary path uses)."""
    tid = trace_id or str(uuid.uuid4())
    settings = get_settings()
    audit: AuditService = wiring.audit_service()
    labels = {d.doc_id: _doc_label(d) for d in docs}
    plan_payload: dict[str, object] = (
        {"strategy": plan.strategy, "coverage": plan.coverage, "rationale": plan.rationale} if plan is not None else {}
    )
    await audit.record(
        tenant_id=query.tenant_id,
        kb_id=query.tenant_id,
        trace_id=tid,
        action=AuditAction.QUERY,
        actor=actor,
        payload={
            "query": query.text,
            "intent": "whole_document_multi",
            "targets": list(target_docs),
            "count": len(target_docs),
            **plan_payload,
        },
    )
    trace = [AgentStep(step="plan", detail=f"whole_document → {len(target_docs)} matched documents")]
    if plan is not None and plan.rationale:
        trace.append(AgentStep(step="rationale", detail=plan.rationale))

    sections: list[str] = []
    citations: list[Citation] = []
    covered_total = 0
    expected_total = 0
    any_truncated = False
    all_complete = True
    grounded = 0
    for doc_id in target_docs:
        per = await _answer_document_full_summary(
            query, doc_id, actor=actor, model=model, trace_id=tid, plan=plan, docs=docs
        )
        label = labels.get(doc_id, doc_id)
        if per.status is not AnswerStatus.GROUNDED or not per.citations:
            sections.append(f"## {label}\n\n_No grounded summary could be produced for this document._")
            all_complete = False
            continue
        shifted = _shift_citation_markers(per.answer.strip(), len(citations))
        sections.append(f"## {label}\n\n{shifted}")
        citations.extend(per.citations)
        grounded += 1
        if per.completeness is not None:
            covered_total += per.completeness.covered
            expected_total += per.completeness.expected
            any_truncated = any_truncated or per.completeness.truncated
            all_complete = all_complete and per.completeness.complete
        else:
            all_complete = False  # coverage not measured (guard off) → cannot confirm complete

    if grounded == 0 or not citations:
        return GroundedAnswer(
            tenant_id=query.tenant_id,
            query=query.text,
            status=AnswerStatus.ABSTAINED,
            answer=_WHOLE_DOC_NO_GROUNDING,
            citations=[],
            confidence=0.0,
            abstain_reason="doc_summary_no_grounding",
            trace_id=tid,
            agent_trace=trace,
            structured={"intent": "whole_document_multi", "targets": list(target_docs), "grounded": 0},
        )

    trace.append(AgentStep(step="reduce", detail=f"Combined {grounded} document summary(ies)"))
    completeness = (
        Completeness(
            complete=all_complete and not any_truncated,
            reason=(
                None
                if (all_complete and not any_truncated)
                else "one or more matched documents were only partially covered"
            ),
            covered=covered_total,
            expected=expected_total,
            truncated=any_truncated,
        )
        if settings.completeness_guard_enabled
        else None
    )
    answer = GroundedAnswer(
        tenant_id=query.tenant_id,
        query=query.text,
        status=AnswerStatus.GROUNDED,
        answer=f"Summaries of {grounded} matching document(s):\n\n" + "\n\n".join(sections),
        citations=citations,
        confidence=1.0,
        trace_id=tid,
        agent_trace=trace,
        structured={
            "intent": "whole_document_multi",
            "targets": list(target_docs),
            "grounded": grounded,
            "covered": covered_total,
        },
        completeness=completeness,
    )
    await _emit_receipt(audit, answer, actor=actor)
    return answer


async def _turn_meta_answer(  # noqa: PLR0911 - one guarded return per meta-intent label
    intent: str,
    q: RetrievalQuery,
    tenant_id: str,
    conversation_id: str,
    question: str,
    *,
    actor: str,
    model: str | None,
) -> GroundedAnswer | None:
    """Route a conversational turn to its meta-intent handler (catalog/summarize/count/filter/
    compare), or return None when the intent is plain ``rag`` and the turn should flow through the
    normal condense→retrieve path. Shared by ``answer_turn`` and ``stream_turn``."""
    if intent == "catalog":
        return await _answer_catalog(q, actor=actor)
    if intent == "doc_summary":
        return await _answer_doc_summary(q, actor=actor, model=model)
    if intent == "summarize":
        return await _answer_summary(tenant_id, conversation_id, question, actor=actor, model=model)
    if intent == "count":
        return await _answer_count(q, actor=actor)
    if intent == "filter":
        return await _answer_filter(q, actor=actor)
    if intent == "compare":
        return await _answer_compare(q, actor=actor, model=model)
    return None


# A bounded sample of visible titles handed to the planner as light KB context — enough to resolve
# "this document" / a named title, never the whole corpus.
_PLAN_TITLE_SAMPLE = 30


async def _plan_context(query: RetrievalQuery, visible: list[DocumentEnvelope]) -> PlanContext:
    """The light KB context the planner reads: visible-doc count + a bounded title sample."""
    return PlanContext(
        doc_count=len(visible),
        doc_titles=tuple(_doc_label(d) for d in visible[:_PLAN_TITLE_SAMPLE]),
        has_history=False,
    )


def _match_plan_targets(
    plan: QueryPlan, query: RetrievalQuery, visible: list[DocumentEnvelope], docs: list[DocumentEnvelope]
) -> list[str]:
    """The visible documents a ``whole_document``/summary plan targets — pure, ACL-scoped.

    Returns the matched doc_ids (possibly SEVERAL — the caller decides one-doc vs multi-doc vs
    too-many). Matching:

    (1) ``plan.target_doc`` — prefer EXACT title/doc_id equality (case-insensitive); when none are
        exact, fall to FUZZY token overlap (:func:`match_compare_targets`), returning ALL confident
        matches (e.g. "Vendor Agreement" → both the 2023 and 2024 files). No strict-winner tie-break
        — an ambiguous target yields every match, and the caller summarises each rather than guessing.
    (2) no ``plan.target_doc`` — the sole visible doc when the KB has exactly one, else the document
        the QUESTION itself names unambiguously (``scope_for_query`` over the already-fetched
        ``docs`` — no extra catalog round-trip), else none."""
    if not visible:
        return []
    if plan.target_doc:
        target = plan.target_doc.strip()
        tl = target.lower()
        exact = [d.doc_id for d in visible if d.doc_id.lower() == tl or _doc_label(d).lower() == tl]
        if exact:
            return exact
        return match_compare_targets(target, [(d.doc_id, _doc_label(d)) for d in visible])
    if len(visible) == 1:
        return [visible[0].doc_id]
    if get_settings().doc_scope_enabled:
        scope = scope_for_query(query.text, [d.doc_id for d in docs])
        if scope is not None and any(d.doc_id == scope.doc_id for d in visible):
            return [scope.doc_id]
    return []


async def _execute_query_plan(  # noqa: PLR0911 - one guarded return per strategy → mechanism mapping
    plan: QueryPlan,
    query: RetrievalQuery,
    visible: list[DocumentEnvelope],
    docs: list[DocumentEnvelope],
    *,
    actor: str,
    model: str | None,
    trace_id: str,
) -> tuple[GroundedAnswer | None, list[str]]:
    """Map ``plan.strategy`` → an answer mechanism. Returns ``(answer, clarify_titles)``:

    * a routed answer + ``[]`` when the plan selects a mechanism;
    * ``(None, [])`` to fall through to the UNCHANGED normal RAG path;
    * ``(None, titles)`` when a whole_document target matched TOO MANY documents — the caller answers
      from passages and surfaces ``titles`` as non-authoritative "did you mean…?" options.

    whole_document routing (or ``passages``+``summary``+``exhaustive`` — an LLM plan explicitly asking
    for an exhaustive summary; NEVER ``whole_kb``, so the heuristic doc_summary→whole_kb stays
    today's behaviour — FIX 5):
      * 1 matched doc  → single whole-document read;
      * 2..cap matched → summarise EACH and combine into one grounded answer (FIX 1, multi-target);
      * > cap matched  → normal RAG + the matched titles as clarification options (never dump/guess);
      * 0 matched      → normal RAG.

    Fail-safe: an unresolved strategy returns ``(None, [])`` (normal RAG), so a planner decision can
    never turn an answerable question into an error — worst case is today's behaviour."""
    strategy = plan.strategy
    wants_whole_doc = strategy == "whole_document" or (
        strategy == "passages" and plan.answer_shape == "summary" and plan.coverage == "exhaustive"
    )
    if wants_whole_doc:
        matched = _match_plan_targets(plan, query, visible, docs)
        cap = get_settings().whole_doc_multi_target_cap
        if len(matched) == 1:
            answer = await _answer_document_full_summary(
                query, matched[0], actor=actor, model=model, trace_id=trace_id, plan=plan, docs=docs
            )
            return answer, []
        if _MULTI_TARGET_MIN <= len(matched) <= cap:
            answer = await _answer_documents_full_summary(
                query, matched, actor=actor, model=model, trace_id=trace_id, plan=plan, docs=docs
            )
            return answer, []
        if len(matched) > cap:
            matched_set = set(matched)
            titles = [_doc_label(d) for d in visible if d.doc_id in matched_set]
            return None, titles  # too many → passages + clarify, never guess or dump
        return None, []  # no match → normal RAG (passages)
    if strategy == "catalog":
        return await _answer_catalog(query, actor=actor, trace_id=trace_id), []
    if strategy == "count":
        return await _answer_count(query, actor=actor, trace_id=trace_id), []
    if strategy == "filter":
        return await _answer_filter(query, actor=actor, trace_id=trace_id), []
    if strategy == "compare":
        return await _answer_compare(query, actor=actor, model=model, trace_id=trace_id), []
    # whole_kb → normal RAG path (which runs _global_augment when GraphRAG global search is on);
    # passages / anything else → the unchanged normal RAG path.
    return None, []


# --- RAG v2 Phase 12: per-stage query timing + the query log -------------------------------------


async def _record_query_latency(
    query: RetrievalQuery,
    answer: GroundedAnswer | None,
    timeline: StageTimeline,
    *,
    trace_id: str,
    path: str,
    verification: str | None = None,
    router_json: str | None = None,
    selected_sections: Sequence[str] | None = None,
) -> None:
    """Emit this question's per-stage timing, and write the ``query_log`` row when it is enabled.

    THE PLAN'S §0 CRITERIA ARE LATENCY TARGETS and discovery found them unmeasurable: the query
    path recorded one number, the round trip, so a slow answer could not be attributed to routing,
    retrieval or generation (§8.2 — which is also why the Phase-0 baseline was never run). This is
    the measurement those targets need.

    THE LOG LINE IS ALWAYS EMITTED and carries no question text — stage durations, the trace id and
    the status, nothing a log shipper has to be trusted with. The `query_log` ROW is what carries
    the question and the answer, and that is flag-gated (`query_log_enabled`, default off): the row
    is a derived record of document text in exactly the way a chat turn is, which is why the DPDP
    erasure path for it was built before anything wrote it.

    NO AUDIT EVENT IS ADDED, deliberately. The obvious place to hang a latency number is the QUERY
    audit payload, and it is the wrong one twice over: `AuditSink.count_actions` METERS those
    events for FinOps showback (FR-H4), so a second one per question would silently inflate every
    tenant's bill, and the existing QUERY event is recorded mid-flight — before generation — so it
    is not a place a total can honestly go.

    NEVER RAISES. An answer that is correct, grounded and already paid for must not fail because
    its latency could not be recorded.
    """
    stages = timeline.payload()
    _log.info(
        "query_stage_timing",
        tenant_id=query.tenant_id,
        trace_id=trace_id,
        path=path,
        status=answer.status.value if answer is not None else "none",
        **stages,
    )
    settings = get_settings()
    if not settings.query_log_enabled or answer is None:
        return
    try:
        await record_query_log(
            query.tenant_id,
            query=_clip(query.original_text or query.text, settings.query_log_max_query_chars),
            answer=_clip(answer.answer, settings.query_log_max_answer_chars),
            # THE ERASURE KEY. A row whose answer quotes a document must be reachable by a
            # document-scoped delete, and this array is the only thing that makes it so — see
            # `PostgresCatalog.delete_document`, where this closed a gap flagged in Phase 1.
            cited_doc_ids=sorted({c.doc_id for c in answer.citations if c.doc_id}),
            latency_ms=stages,
            # Phase 10's column, filled when `v2_verify` is on and NULL otherwise — which is
            # the honest distinction between "verified and clean" and "not verified".
            verification=verification,
            # ...and Phase 7's, on the same terms: NULL means the router did not run.
            router_json=router_json,
            # ...and Phase 8's. NULL means no section pass ran — the flag is off, the intent is
            # not an aggregate-class one, or there were no profiled sections in scope. An EMPTY
            # array means the pass ran and selected nothing, which is a different and much more
            # interesting fact, so the two are deliberately not collapsed.
            selected_sections=selected_sections,
        )
    except Exception as exc:  # the answer is already correct; losing its log row must not fail it
        _log.warning("query_log_write_failed", tenant_id=query.tenant_id, trace_id=trace_id, error=str(exc)[:160])


# --- RAG v2 Phase 7: the query router -------------------------------------------------------------


async def _router_scope(tenant_id: str, query: RetrievalQuery) -> RouterScope:
    """The documents this question may be about, as Phase-3 cards (plan §7's router input).

    EXPLICIT when the caller named documents (`RetrievalQuery.doc_ids`, the first-class predicate
    D1's fix introduced), a SAMPLE otherwise. The plan asks for "scope = explicit doc filter, else
    top docs by doc-level retrieval", and there is no doc-level retrieval to ask: the `doc` chunk
    level has no producer (§8.17). So the fallback is the newest profiled documents, and
    `RouterScope.explicit` carries which of the two happened — a decision made over a named
    document is about that document, while one made over a sample is only as good as the sample,
    and a consumer reading a stored decision has to be able to tell them apart.

    An empty scope is ORDINARY, not an error: the card is written only on the v2 ingest path, so
    every KB has no cards until a document is ingested with the flag on, and the router is built
    to decide an intent without any."""
    settings = get_settings()
    explicit = tuple(query.doc_ids)
    cards = await document_cards(tenant_id, doc_ids=explicit or None, limit=settings.v2_router_max_cards)
    return RouterScope(cards=tuple(cards), explicit=bool(explicit))


async def _route_v2(tenant_id: str, query: RetrievalQuery) -> RouterDecision | None:
    """Route one question (Phase 7). Gated on ``v2_router``; NEVER raises.

    None when the flag is off, which leaves the query path byte-identical. With it on there is
    ALWAYS a decision — the heuristic backend is deterministic and needs nothing — which is the
    plan's own fallback posture ("if router output fails validation, default to lookup and log")
    and the reason this returns a decision rather than None on model trouble.

    The query's language is the platform's own detection and is handed in rather than asked for,
    for the reason `parse_doc_profile` records: the platform answers it deterministically in
    ISO-639-1 and a measured real-model run returned the prose word "english"."""
    settings = get_settings()
    if not settings.v2_router:
        return None
    try:
        text_for_router = query.original_text or query.text
        # Phase 12's router cache. A decision is a function of the question's words and the
        # cards in scope, so the DOC SET is part of the key: the same question over a different
        # scope is a different decision, and a cache keyed on the query alone would scope one
        # question's answer to another question's documents. The LLM backend is where this pays
        # — it is a gateway call on the latency path before every answer.
        cached = v2_cache.router(tenant_id, text_for_router, tuple(query.doc_ids))
        if cached is not None:
            return cached
        scope = await _router_scope(tenant_id, query)
        lang = detect_language(text_for_router)
        if settings.v2_router_backend == "llm":
            router = LLMQueryRouter(
                await wiring.resolve_tenant_gateway(tenant_id), max_tokens=settings.v2_router_max_tokens
            )
            decided = await router.route(text_for_router, scope, query_lang=lang)
        else:
            decided = heuristic_decision(text_for_router, scope, query_lang=lang)
        v2_cache.put_router(tenant_id, text_for_router, tuple(query.doc_ids), decided)
        return decided
    except Exception as exc:  # fail-safe: a routing failure must never cost a question its answer
        _log.warning("router_skipped", tenant_id=tenant_id, error=str(exc)[:160])
        return None


def _router_payload(decision: RouterDecision | None) -> dict[str, object]:
    """The router's decision on the QUERY audit payload, beside `route_payload` and `plan_payload`.

    Non-authoritative and additive. NOT a second audit EVENT, for the reason
    `_record_query_latency` states: `count_actions` meters QUERY events for FinOps showback, so
    one extra per question would inflate every tenant's bill. The full decision goes to
    `query_log.router_json`."""
    if decision is None:
        return {}
    return {
        "rag_v2_intent": decision.intent,
        "rag_v2_docs": len(decision.doc_ids),
        "rag_v2_facets": len(decision.facets),
        "rag_v2_sub_queries": len(decision.sub_queries),
        "rag_v2_amendment_view": decision.needs_amendment_view,
    }


# --- RAG v2 Phase 8: retrieval strategies ---------------------------------------------------------


def _section_probes(query: str, decision: RouterDecision, taxonomy: Sequence[tuple[str, str]] = ()) -> tuple[str, ...]:
    """Everything the section cards are scored against (plan step 2's "the main query AND every
    sub-query"), plus one probe per matched facet.

    THE FACET PROBE IS AN ADDITION, and it is here because it measured best. Scored against the
    real AEGCL corrigendum's enriched section cards on this deployment's BGE cross-encoder:

    ======================================  =========  ==================================
    probe                                   best       ratios over the relevant sections
    ======================================  =========  ==================================
    the bare aggregate query                 3.4e-4     0.049 / 0.064 / 0.044 — no signal
    "What are the <facet> requirements?"     0.356      1.0 / 0.366 / 0.047
    the facet's own words                    0.631      1.0 / 0.994 / 0.413
    ======================================  =========  ==================================

    The facet's own words win on both numbers that matter: the highest absolute score, and —
    more importantly for recall — they keep the THIRD relevant section at 0.413 of the best
    where the question form drops it to 0.047, which is the difference between selecting it and
    not. A facet is a topic label, and a cross-encoder reads a topic label as a topic.

    They are NOT made the router's ``sub_queries``, though, because that field has a second
    consumer: it drives the pipeline's decomposition block, where each entry is used as a
    RETRIEVAL query, and a bare keyword is a poor dense query. Two consumers, two shapes — and
    the difference in COST is the other half of the reason: a probe is one batched rerank call
    over candidates already in memory, while a sub-query is a full hybrid retrieval.

    ``taxonomy`` is the EXHAUSTIVE intents' extra scope (``StrategyPlan.exhaustive``). An
    aggregate deliberately names no facet, so the router matches almost none by word overlap:
    on the real AEGCL corrigendum, "give me the compliance details from this document" matched
    exactly one of the document's eight facets, giving two probes and reaching 6 of the
    acceptance's 14 section groups. Probing all eight reached 11. It is passed in rather than
    read from the decision because the applied taxonomy lives on the SECTIONS — see
    ``strategies.taxonomy_of``."""
    probes: dict[str, None] = {(query or "").strip(): None}
    probes.update(dict.fromkeys(decision.sub_queries))
    for _doc_id, facet in (*decision.facets, *taxonomy):
        probes.setdefault(facet.replace("_", " "), None)
    return tuple(p for p in probes if p)


async def _section_scores(tenant_id: str, probes: Sequence[str], cards: Sequence[SectionCard]) -> dict[str, float]:
    """Rerank every section against every probe and keep each section's best score (plan step 2).

    ONE RERANK CALL PER PROBE, each over the whole candidate set — not one per (probe, section)
    pair. The cross-encoder's batch interface is what makes a section pass affordable at all: a
    tender with 200 sections and 6 probes is 7 calls, not 1400.

    THE PROBES CARRY THIS, and that is the measured reason the loop runs over all of them rather
    than the main query alone. Against this deployment's BGE reranker, the acceptance query
    "give me the compliance details from this document" scores every one of the real document's
    78 sections between 1.2e-5 and 7.2e-5 — a 6x spread with no relationship to relevance,
    because a vague whole-document question is not "about" any one passage. See
    :func:`_section_probes` for what does separate them.

    The spans handed to the reranker are SYNTHETIC — one per section, carrying the section's
    card text (breadcrumb + heading + summary), with the section_id in `chunk_id` so the scores
    come back addressable. They never leave this function and never become citations.
    """
    reranker = await wiring.resolve_tenant_reranker(tenant_id)
    texts = {c.section_id: c.rerank_text() for c in cards}
    by_id = {c.section_id: c for c in cards}
    per_query: list[dict[str, float]] = []
    for text_q in probes:
        if not text_q.strip():
            continue
        # Phase 12's score cache. A score is a function of (probe, candidate text) and nothing
        # else, so a hit is the number the cross-encoder would have returned — no revalidation,
        # unlike the ANSWER cache. Eight of this document's nine probes are its own facet names,
        # identical for every question asked of it, which is where the saving comes from.
        cached = v2_cache.scores(tenant_id, text_q, texts)
        todo = [by_id[sid] for sid in texts if sid not in cached]
        fresh: dict[str, float] = {}
        if todo:
            spans = [RetrievedSpan(chunk_id=c.section_id, doc_id=c.doc_id, text=texts[c.section_id]) for c in todo]
            fresh = {sp.chunk_id: sp.score for sp in await reranker.rerank(text_q, spans, len(spans))}
            v2_cache.put_scores(tenant_id, text_q, texts, fresh)
        per_query.append({**cached, **fresh})
    return max_scores(per_query)


async def _select_sections(
    tenant_id: str, query: RetrievalQuery, decision: RouterDecision | None
) -> SectionSelection | None:
    """The plan's §8 section pass for the aggregate-class strategies. NEVER raises.

    Returns None — meaningfully distinct from an empty selection — when the flag is off, when no
    router decision exists, when the intent's strategy has no section pass, when no document is
    in scope, or when the in-scope documents have no `sections` rows. All five mean "no section
    pass ran", which is what a NULL `query_log.selected_sections` records; an empty SELECTION
    means the pass ran and chose nothing, which is a real and much more interesting result.

    SCOPE COMES FROM THE DECISION, then the caller's own `doc_ids`. The strategy reads EVERY
    section of the documents in scope, so an unscoped aggregate would load the whole KB onto the
    latency path — and would also be answering a question about "this document" from documents
    the asker never mentioned.

    FAIL-SAFE like every other derived artifact on this path. A failure here costs the section
    pass; the pipeline's existing passage retrieval runs either way and still answers."""
    settings = get_settings()
    if not settings.v2_retrieval or decision is None:
        return None
    strategy = plan_strategy(decision.intent)
    if not strategy.section_pass:
        return None
    scope = tuple(decision.doc_ids) or tuple(query.doc_ids)
    if not scope:
        # SAY SO. Returning None here turns off Phases 8 and 9 for this question, and the caller
        # gets passage retrieval or the whole-document read instead — a different algorithm, with
        # its own truncation, and nothing in the answer or the events says which one ran. An
        # operator who switched the v2 flags on and saw no change had no way to find out why.
        _log.info(
            "rag_v2_section_pass_skipped",
            tenant_id=tenant_id,
            reason="no document in scope",
            detail="name the document(s) in the request — the section pass needs a scope",
        )
        return None
    try:
        cards = await document_sections(tenant_id, scope)
        if not cards:
            return None
        main = query.original_text or query.text
        # Bound what reaches the cross-encoder. Every probe scores every candidate, so this is
        # the one place in the section pass whose cost grows with the SIZE OF THE DOCUMENT on
        # the latency path. The cards past the cap are still CONSIDERED — they can be selected
        # by the facet or amendment unions, which need no score — they are simply not scored.
        candidates = cards[: settings.v2_max_section_candidates]
        # An EXHAUSTIVE intent's scope is the document's whole applied facet taxonomy, not the
        # facets the query's words happened to match — see `StrategyPlan.exhaustive`. It widens
        # both halves of the selection: every facet becomes a scoring probe, and step 4's union
        # covers every section the enrichment gave a facet to.
        taxonomy = taxonomy_of(candidates) if strategy.exhaustive else ()
        facets = taxonomy or decision.facets
        # `compare` splits the budget per document (the plan's "run aggregate per entity/doc"),
        # because one global cap lets a large document crowd a small one out of its own
        # comparison. Every other section-pass strategy takes the single-pass selection.
        selector = select_per_target if strategy.per_target else select_sections
        selection = selector(
            cards,
            await _section_scores(tenant_id, _section_probes(main, decision, taxonomy), candidates),
            facets=facets,
            amendment_view=decision.needs_amendment_view,
            threshold=settings.v2_section_threshold,
            relative=settings.v2_section_relative,
            max_sections=settings.v2_max_sections,
        )
        _log.info(
            "rag_v2_sections_selected",
            tenant_id=tenant_id,
            intent=decision.intent,
            docs=len(scope),
            considered=selection.considered,
            # Below `considered` when the document is larger than the candidate cap: those
            # sections were weighed on the unions alone. A silent shortfall here would read as
            # "the reranker found nothing in them".
            scored=len(candidates),
            selected=len(selection.selected),
            dropped=len(selection.dropped),
            # How many got in on a reason the reranker did NOT discover. A pass where this is the
            # whole selection means the scores contributed nothing, which is worth seeing.
            decided=sum(1 for sec in selection.selected if sec.decided),
        )
        return selection
    except Exception as exc:  # fail-safe: a selection failure must never cost a question its answer
        _log.warning("rag_v2_sections_skipped", tenant_id=tenant_id, error=str(exc)[:160])
        return None


def _strategy_payload(decision: RouterDecision | None, selection: SectionSelection | None) -> dict[str, object]:
    """The Phase-8 strategy on the QUERY audit payload, beside `_router_payload`.

    Additive and non-authoritative, and NOT a second audit EVENT — same reason as Phase 7, 10 and
    12: `count_actions` meters QUERY events for FinOps showback."""
    if decision is None:
        return {}
    strategy = plan_strategy(decision.intent)
    payload: dict[str, object] = {"rag_v2_strategy": strategy.intent}
    if strategy.levels:
        payload["rag_v2_levels"] = [lv.value for lv in strategy.levels]
    if selection is not None:
        payload["rag_v2_sections_considered"] = selection.considered
        payload["rag_v2_sections_selected"] = len(selection.selected)
        payload["rag_v2_sections_dropped"] = len(selection.dropped)
    return payload


# Per-item overhead in the reduce prompt's items block beyond the statement and quote: the
# marker, the breadcrumb, the facet, the clause ref and the rendered values. Measured against
# `items_block` on real items rather than guessed, and deliberately generous — the budget it
# feeds is spent against a safety fraction, and underestimating it is what overflows a prompt.
_ITEM_OVERHEAD_CHARS = 180

# An INTERNAL stream event type, emitted by `_stream_answer_impl` and consumed by the
# `stream_answer` wrapper. Underscore-prefixed so it cannot be mistaken for part of the
# client-facing protocol (`phase`, `token`, `done`), and the wrapper never forwards it.
_V2_META = "_rag_v2_meta"

# The confidence a map-reduce answer carries. There is no reranker score to report: this answer
# was not built from retrieved passages, and every span under it is an extracted item whose
# quote was matched against its section verbatim (`scorer="exact"`, score 1.0). Reporting a
# retrieval max_score here would be a number about a search this answer did not use.
_MAP_REDUCE_CONFIDENCE = 1.0

# Chunk levels that hold text this PLATFORM wrote rather than text the document contains: the
# Phase-4 section summary and its propositions. See `_section_chunk_text`.
_DERIVED_LEVELS = frozenset({ChunkLevel.SECTION, ChunkLevel.PROPOSITION})

# The reduce prompt's citation markers: `[m3]` or `[m3,m7]`, the `item_id`s it was given.
#
# MARKDOWN EMPHASIS INSIDE THE BRACKETS IS TOLERATED, because the presentation contract the
# reduce prompt now carries says to bold every number and an `item_id` ends in one. `**[m3]**`
# always matched -- the emphasis is outside the brackets -- but `[**m3**]` did not, and an
# unmatched marker is not dropped like an unresolvable one: it is left in the published answer
# verbatim, which leaks this prompt's internal marker vocabulary to the reader. The prompt also
# says not to bold a marker; this is the half that does not depend on the model obeying. Widening
# can only make previously-unmatched text match, and the id is still looked up in `_by_id`, so
# nothing the map did not produce can resolve.
_ITEM_MARKER_RE = re.compile(r"\[(\*{0,2}m\d+\*{0,2}(?:\s*,\s*\*{0,2}m\d+\*{0,2})*)\]")


# --- RAG v2 Phase 9: generation by map then reduce --------------------------------------------------


async def _section_chunk_text(
    tenant_id: str, section_ids: Sequence[str], doc_ids: Sequence[str]
) -> dict[str, list[tuple[str, str]]]:
    """``section_id -> [(chunk_id, text), ...]`` in document order, for the sections in scope.

    THE TEXT IS ``Chunk.text``, WHICH IS MASKED. The plan says to map over "section text (tables
    as HTML)", and the obvious source for that is ``blocks.text`` — but those blocks are captured
    PRE-masking by design (see :func:`_section_enrichment`), so they carry PII as the extractor
    read it. A map prompt is a model call over document text, so it gets the masked form, exactly
    as the Phase-3 and Phase-4 prompts do. The cost is that an HTML table arrives as its chunked
    text rather than as markup; the alternative is sending unmasked PII to a model on the query
    path, which is not a trade this codebase makes anywhere else.

    ONE READ PER DOCUMENT, not per section: ``document_chunks`` is a whole-document query, and a
    40-section selection over one tender would otherwise run it forty times for the same rows.

    Chunks with no ``section_id`` (every v1 chunk) are skipped rather than pooled under a null
    section — they belong to no section the selector chose.

    THE PLATFORM'S OWN DERIVED UNITS ARE EXCLUDED. A section's chunk set also holds its Phase-4
    SUMMARY (``level='section'``) and its PROPOSITIONS (``level='proposition'``), both of which
    are restatements this platform wrote, not text the document contains. Feeding them to the
    map step would ask a model to extract requirements from another model's paraphrase and then
    quote it as if the document had said it — and the quote check could not tell the difference,
    because the paraphrase really is in the text it was handed. It would also mean a stale
    summary, left behind by a re-enrichment that has not been reindexed, silently becomes
    evidence. `None` is kept: a v1 chunk has no level and is the document's own text."""
    wanted = set(section_ids)
    out: dict[str, list[tuple[str, str]]] = {}
    for doc_id in dict.fromkeys(doc_ids):
        for chunk in await get_catalog().document_chunks(tenant_id, doc_id):
            if chunk.section_id in wanted and chunk.level not in _DERIVED_LEVELS:
                out.setdefault(chunk.section_id, []).append((chunk.chunk_id, chunk.text))
    return out


def _map_jobs(
    selection: SectionSelection, cards: Mapping[str, SectionCard], texts: Mapping[str, list[tuple[str, str]]]
) -> list[MapJob]:
    """One job per selected section that actually has text to read.

    A selected section with no chunks is dropped here rather than sent as an empty prompt: it
    contributes nothing either way, and an empty map call is a model call paid for a guaranteed
    empty answer. It still counts in the coverage denominator, which is the honest record —
    the section was selected and did not contribute."""
    jobs: list[MapJob] = []
    for selected in selection.selected:
        chunks = texts.get(selected.section_id, [])
        if not chunks:
            continue
        card = cards.get(selected.section_id)
        jobs.append(
            MapJob(
                section_id=selected.section_id,
                doc_id=selected.doc_id,
                breadcrumb=card.breadcrumb if card else "",
                text="\n\n".join(text for _cid, text in chunks),
            )
        )
    return jobs


class _MarkerResolver:
    """Rewrites the reduce output's ``[mN]`` markers into the ``[n]`` markers ``ground`` reads,
    collecting the spans they point at — INCREMENTALLY, so the same resolver can be fed one
    whole answer or the answer in pieces and produce identical output either way.

    THAT EQUIVALENCE IS THE POINT, and it is why this is a class and not a function. The
    streaming path emits the answer section by section as each reduce call lands, and the
    non-streaming path rewrites it in one go; if the two numbered citations differently the same
    question would answer differently depending on transport. Numbering by FIRST APPEARANCE
    makes them agree, because first appearance and streaming order are the same order.

    WHY REWRITE RATHER THAN TEACH ``ground`` A SECOND MARKER SYNTAX. Everything downstream —
    the cited-only filter, the citation assembly, the console's click-through, the Phase-10
    verifier — reads ``[n]`` against a span list. A map-reduce answer that carried its own
    marker vocabulary would need every one of them to learn it, and would quietly bypass the
    ones that did not.

    AN UNRESOLVABLE MARKER LOSES ITS CITATION, NOT ITS SENTENCE. An item's quote was validated
    against the SECTION text, which is the concatenation of its chunks, so a quote straddling
    the join between two chunks is in the section and in neither chunk. Dropping the marker
    leaves a true statement uncited — which the grounded-or-abstain gate then judges on the
    citations that remain — where dropping the sentence would delete a requirement the document
    does state."""

    def __init__(self, items: Sequence[MappedItem], texts: Mapping[str, list[tuple[str, str]]]) -> None:
        self._by_id = {item.item_id: item for item in items}
        self._texts = texts
        self._index: dict[str, int] = {}
        self.spans: list[RetrievedSpan] = []

    def rewrite(self, text: str) -> str:
        """Rewrite one piece. Safe to call repeatedly; state carries across calls."""
        return _ITEM_MARKER_RE.sub(self._replace, text)

    def _replace(self, match: re.Match[str]) -> str:
        out: list[str] = []
        for raw in match.group(1).split(","):
            item = self._by_id.get(raw.strip().strip("*"))
            if item is None:
                continue
            if item.item_id not in self._index:
                chunk_id = resolve_item_chunk(item, self._texts.get(item.section_id, []))
                if chunk_id is None:
                    continue
                self.spans.append(
                    RetrievedSpan(
                        chunk_id=chunk_id,
                        doc_id=item.doc_id,
                        text=item.quote,
                        score=1.0,
                        # The cross-encoder never saw this span: it is an extracted item whose
                        # quote was checked against the section verbatim. `exact` is the scorer
                        # this codebase already has for "matched, not ranked".
                        scorer="exact",
                    )
                )
                self._index[item.item_id] = len(self.spans)
            out.append(str(self._index[item.item_id]))
        return "[" + ",".join(out) + "]" if out else ""


def _map_citations(
    answer_text: str, items: Sequence[MappedItem], texts: Mapping[str, list[tuple[str, str]]]
) -> tuple[str, list[RetrievedSpan]]:
    """Resolve a WHOLE map-reduce answer's citations in one pass (the non-streaming path).

    A thin wrapper over :class:`_MarkerResolver` so the two paths cannot drift: this is one
    ``rewrite`` call where the streaming path makes several."""
    resolver = _MarkerResolver(items, texts)
    return resolver.rewrite(answer_text), resolver.spans


async def _map_inputs(
    tenant_id: str, selection: SectionSelection
) -> tuple[dict[str, SectionCard], dict[str, list[tuple[str, str]]], list[MapJob]] | None:
    """The cards, the per-section chunk text and the map jobs for a selection — or None when
    there is nothing to map over.

    Its own function so :func:`_map_reduce_answer` reads as the sequence it is (prepare, map,
    reduce, cite) rather than as a chain of early exits, and so the "selected but empty" case
    has one place to be explained: a selected section with no chunks is dropped from the JOBS
    but still counts in the coverage DENOMINATOR, because it was selected and did not
    contribute, and an answer that quietly shrank its own denominator would overstate its
    coverage."""
    docs = tuple(dict.fromkeys(s.doc_id for s in selection.selected))
    cards = {card.section_id: card for card in await document_sections(tenant_id, docs)}
    texts = await _section_chunk_text(
        tenant_id, [s.section_id for s in selection.selected], [s.doc_id for s in selection.selected]
    )
    jobs = _map_jobs(selection, cards, texts)
    if not jobs:
        _log.info("rag_v2_map_no_text", tenant_id=tenant_id, selected=len(selection.selected))
        return None
    return cards, texts, jobs


def _reduce_batches(
    data: ReduceInput, breadcrumbs: Mapping[str, str], *, max_tokens: int
) -> list[tuple[str, Sequence[MappedItem]]]:
    """The reduce calls to make, as ``(facet, items)`` in ANSWER ORDER.

    ONE FACET CAN STILL OVERFLOW. Grouping by facet bounded the prompt by "how much the document
    says about one topic", which is not a bound at all on a tender: measured on the real
    corrigendum, ``soc_noc_infrastructure`` alone held 278 of 523 items. So each facet's items
    are packed into batches that fit the model's window, and a facet that needs several
    contributes several calls under one heading (``assemble_answer`` merges them).

    SHARED BY BOTH PATHS, deliberately. The streaming path emits these in this order and the
    non-streaming path gathers them in this order, so the two answers are the same text — if
    each computed its own plan, a change to the packing would silently make them diverge."""
    budget = prompt_budget_chars(
        context_window=get_settings().llm_context_window,
        max_output_tokens=max_tokens,
        # The fixed part of the prompt, measured rather than guessed, by rendering it empty.
        scaffolding_chars=len(build_reduce_prompt(data, "x", (), breadcrumbs)),
    )
    # AND BOUND THE REPLY, not just the prompt. Packing by prompt size alone produced batches the
    # model could not possibly answer: measured on the real AEGCL corrigendum, the
    # `infrastructure_hardware` facet arrived as ONE call carrying 446 items, which fits the
    # 32k window with room to spare and needs ~13,000 output tokens against a 4,000-token reply
    # budget. The model rendered what fitted and the rest was silently absent — two gold items
    # (a TPM requirement and a firewall OEM-diversity clause) were extracted correctly by the map,
    # survived the quote check, and then simply never appeared in the answer. Nothing logged it,
    # because from the reduce's point of view the call succeeded.
    return [
        (facet, batch)
        for facet, items in group_by_facet(data.items)
        for batch in batch_to_budget(items, _item_cost, budget, max_items=_reduce_item_cap(max_tokens))
    ]


#: Output tokens one rendered requirement costs in the answer. MEASURED, not assumed: a 984-item
#: run produced a 54,124-character answer across 11 calls, and the facets that fitted their budget
#: rendered at ~30 tokens per item. 40 is that figure with headroom, because an item carrying a
#: long clause reference or a restated old -> new value runs longer than the median one.
_REDUCE_OUTPUT_TOKENS_PER_ITEM = 40


def _reduce_item_cap(max_tokens: int) -> int:
    """How many items one reduce call can actually RENDER inside its reply budget.

    Derived from ``max_tokens`` rather than configured separately so the two can never disagree:
    raising the reply budget raises the cap with it, and an operator who lowers it does not
    silently reintroduce the overflow this exists to prevent."""
    return max(1, max_tokens // _REDUCE_OUTPUT_TOKENS_PER_ITEM)


def _item_cost(item: MappedItem) -> int:
    """How much of a reduce prompt one item takes (``items_block``'s line for it)."""
    return len(item.statement) + len(item.quote) + _ITEM_OVERHEAD_CHARS


async def _reduce_and_cite(
    tenant_id: str,
    gateway: Any,
    data: ReduceInput,
    breadcrumbs: Mapping[str, str],
    texts: Mapping[str, list[tuple[str, str]]],
    *,
    max_tokens: int,
    concurrency: int,
) -> tuple[str, list[RetrievedSpan]] | None:
    """The reduce calls, the output guard and the citation resolution. None declines.

    ONE CALL PER FACET, in parallel, assembled by code — see
    :func:`~rag_core.mapreduce.build_reduce_prompt` for the measurement that forced it. A
    facet whose call fails contributes no section; the rest of the answer is still written, and
    the coverage line still reports what it reports, because the facet's items came from
    sections that DID contribute.

    FR-G3 OUTPUT GUARDRAIL is applied HERE because this path does not go through
    ``generation_service().answer()``, which is where every other answer meets it. Declining on
    a block rather than abstaining is deliberate: the caller then produces the ordinary
    single-pass answer, which goes through the guard itself and abstains properly if the same
    content comes back — so a blocked reduce costs the map-reduce shape, not the safety control.

    NO RESOLVABLE CITATION also declines. Grounded-or-abstain forbids a GROUNDED answer with no
    citations, and publishing this as an ABSTAIN would be worse than the passage answer the
    caller is about to produce instead — the document does say these things."""
    # ONE FACET CAN STILL OVERFLOW. Grouping by facet bounded the prompt by "how much the
    # document says about one topic", which is not a bound at all on a tender: measured on the
    # real corrigendum, `soc_noc_infrastructure` alone held 278 of 523 items. So each facet's
    # items are packed into batches that fit the model's window, and a facet that needs several
    # contributes several calls under one heading.
    batches = _reduce_batches(data, breadcrumbs, max_tokens=max_tokens)
    gate = asyncio.Semaphore(max(1, concurrency))

    async def one(facet: str, items: Sequence[MappedItem]) -> tuple[str, str]:
        async with gate:
            try:
                prompt = build_reduce_prompt(data, facet, items, breadcrumbs)
                return facet, await gateway.complete(prompt, max_tokens=max_tokens)
            except Exception as exc:  # one topic lost, not the answer
                _log.warning("rag_v2_reduce_facet_failed", tenant_id=tenant_id, facet=facet, error=str(exc)[:120])
                return facet, ""

    sections = await asyncio.gather(*(one(facet, items) for facet, items in batches))
    _log.info(
        "rag_v2_reduce_batched",
        tenant_id=tenant_id,
        facets=len({facet for facet, _ in batches}),
        calls=len(batches),
    )
    reply = assemble_answer(data, sections)
    if not reply.strip():
        _log.info("rag_v2_reduce_empty", tenant_id=tenant_id, calls=len(batches))
        return None
    if (await _guardrail_gate_flags(tenant_id))[1]:
        verdict = await get_output_guard().check(reply)
        if not verdict.allowed:
            _log.warning("rag_v2_reduce_blocked", tenant_id=tenant_id, category=verdict.category)
            return None
    text, spans = _map_citations(reply.strip(), data.items, texts)
    if not spans:
        _log.info("rag_v2_reduce_uncited", tenant_id=tenant_id, items=len(data.items))
        return None
    return text, spans


async def _route_map_reduce_faithfulness(  # noqa: PLR0911 -- each early return is a distinct
    # verification outcome: gate off, no checker, checker raised, verdict unavailable, grounded,
    # flagged, abstained. Collapsing any two of them is how an unverified answer starts reading
    # as a verified one, which is the defect this function was just corrected for.
    answer: GroundedAnswer,
    spans: Sequence[RetrievedSpan],
    query: RetrievalQuery,
    *,
    actor: str,
    trace_id: str,
    display_query: str,
) -> GroundedAnswer:
    """Run FR-E4 over a map-reduce answer and route it (RAG v2 Phase 12).

    THE ONE PLACE BOTH PATHS DO THIS. ``_map_reduce_answer`` short-circuits
    ``generation_service().answer()`` and the streamed reduce never reaches it either, so neither
    inherited the faithfulness stage — and for a while only the streaming path had a version of
    it, which made the SAME question route to human review through the console and not through
    the API. Two implementations of a safety control is one implementation and one liability, so
    there is a single function and both call it.

    PARITY WITH THE GENERATION SERVICE, step for step, because a control that is almost the same
    is the hard kind of bug to find:

    * the tenant's own checker when configured (FR-E1), then the per-KB notation twin (Slice 4);
    * ``with_parent_context`` on the spans, so parent-only support is not false-flagged (FR-C2);
    * per-KB ``faithful_threshold`` / ``flag_threshold`` overrides, not the platform defaults;
    * a distinct ``AuditAction.FAITHFULNESS`` event with the same payload shape — which is NOT
      the inflation the QUERY event would be: the ordinary path records one per answer, so
      recording it here is what makes the meters agree rather than what skews them;
    * GROUNDED keeps the answer with its score; FLAGGED_FOR_REVIEW rebuilds it through
      ``flag_for_review`` so the caller's existing HITL hook opens the review; below the flag
      threshold it ABSTAINS.

    WHAT IS DELIBERATELY NOT COPIED is the generation service's self-correction and completeness
    re-prompting. Both rewrite the answer by re-prompting over the same spans — and this answer
    was not written from spans. It was assembled from quote-validated items by N calls, so
    "write it again using only what the sources support" has no single prompt to re-issue, and
    the honest equivalent (re-run the map with the unsupported items dropped) is a larger change
    than a safety gap needs. Recorded here rather than left as a silent difference.

    NEVER RAISES. A checker fault must not unground an answer whose every statement is already
    tied to a verbatim quote — it leaves the answer as it was, logged."""
    settings = get_settings()
    if (await _guardrail_gate_flags(query.tenant_id))[0] is False or not settings.faithfulness_enabled:
        return answer
    checker = await wiring.resolve_tenant_faithfulness(query.tenant_id) or get_faithfulness_checker()
    if checker is None:
        return answer
    checker = await wiring.apply_notation_grounding(checker, query.tenant_id)
    overrides = await kb_threshold_overrides(query.tenant_id)
    faithful_gate = overrides.get("faithful_threshold", settings.faithful_threshold)
    flag_gate = overrides.get("flag_threshold", settings.flag_threshold)
    try:
        verdict = await checker.check(answer.answer, with_parent_context(list(spans)), query.text)
    except Exception as exc:  # a cited answer stays cited
        _log.warning("rag_v2_faithfulness_failed", tenant_id=query.tenant_id, error=str(exc)[:160])
        return answer
    # UNAVAILABLE IS NOT A SCORE. `None` means the checker made no judgement -- unreachable with
    # no usable fallback -- and routing a number for it would either ship the answer as verified
    # having checked nothing or abstain on non-evidence. The answer's citations are intact, which
    # is what the status is about, so it stays GROUNDED with no score recorded and the provenance
    # on the audit event says why.
    if verdict.score is None:
        _log.warning(
            "rag_v2_verification_unavailable",
            tenant_id=query.tenant_id,
            method=verdict.method,
            reason=verdict.degraded_reason,
        )
        # The answer still ships -- its citations are intact -- but it now SAYS that nothing
        # verified it, instead of looking identical to one that verified cleanly.
        return answer.model_copy(update={"verification": verification_block([verdict])})
    status = route_faithfulness(verdict.score, faithful_gate, flag_gate)
    await wiring.audit_service().record(
        tenant_id=query.tenant_id,
        kb_id=query.tenant_id,
        trace_id=trace_id,
        action=AuditAction.FAITHFULNESS,
        actor=actor,
        payload={
            "query": query.text,
            "score": verdict.score,
            "outcome": status.value,
            "unsupported_claims": verdict.unsupported_claims,
            # Provenance on the audit event, not only in the answer: a governance reader working
            # from the trail must be able to tell an NLI verdict from a fallback one.
            "method": verdict.method,
            "degraded": verdict.degraded,
            "degraded_reason": verdict.degraded_reason,
            "faithful_threshold": faithful_gate,
            "flag_threshold": flag_gate,
            "self_corrected": False,
            "path": "rag_v2_map_reduce",
        },
    )
    _log.info(
        "rag_v2_faithfulness",
        tenant_id=query.tenant_id,
        trace_id=trace_id,
        score=verdict.score,
        outcome=status.value,
    )
    if status is AnswerStatus.GROUNDED:
        # `unsupported_claims` travels WITH a grounded answer now. The numeric guard can set
        # `faithful=False` and build a claims list on an answer that routes GROUNDED on its
        # score; those findings used to stop at the audit payload, so GROUNDED read as "every
        # check passed" at every surface a reader has.
        return answer.model_copy(
            update={
                "faithfulness_score": verdict.score,
                "unsupported_claims": list(verdict.unsupported_claims),
                "verification": verification_block([verdict]),
            }
        )
    if status is AnswerStatus.FLAGGED_FOR_REVIEW and verdict.score is not None:
        return flag_for_review(
            query.tenant_id,
            display_query,
            answer.answer,
            list(spans),
            _MAP_REDUCE_CONFIDENCE,
            faithfulness_score=verdict.score,
            unsupported_claims=verdict.unsupported_claims,
            trace_id=trace_id,
        )
    return abstain(
        query.tenant_id,
        display_query,
        _MAP_REDUCE_CONFIDENCE,
        reason="faithfulness_below_threshold",
        trace_id=trace_id,
        faithfulness_score=verdict.score,
    )


async def _map_reduce_answer(
    tenant_id: str,
    query: RetrievalQuery,
    decision: RouterDecision | None,
    selection: SectionSelection | None,
    *,
    actor: str,
    trace_id: str,
) -> tuple[GroundedAnswer | None, dict[str, object]]:
    """Answer an aggregate-class question by mapping over its selected sections and reducing
    (RAG v2 Phase 9). Returns ``(answer, payload)``; the answer is None when this path declines.

    IT DECLINES RATHER THAN FAILS, in every case: the flag is off, no section pass ran, nothing
    was selected, no section had text, no item survived its quote check, or the reduce call
    raised. Each returns None and the caller runs the single-pass generation it has always run.
    That is the plan's own fall-through rule and the reason this can be enabled on a live KB: the
    worst outcome is the answer that would have been given anyway, one map pass later.

    The one thing it does NOT do is hide a partial result. A section whose map call failed is
    counted in the coverage statement's denominator and not its numerator, so an answer built
    from half the sections says so in its last line.

    WHICH CONTROLS THIS PATH MEETS, stated because it short-circuits
    ``generation_service().answer()`` and that is where the others live:

    * **Grounded-or-abstain** — enforced, by :func:`~rag_core.grounding.ground`. An answer
      with no resolvable citation is declined here rather than published uncited.
    * **The output guardrail (FR-G3)** — enforced, applied explicitly below.
    * **Phase-10 verification** — enforced, by the caller, which runs ``_verify_answer`` on
      whatever this returns exactly as it does on a single-pass answer.
    * **The abstain gate** — NOT applied, and it should not be: that gate is a judgement about
      RETRIEVAL quality (``should_abstain(retrieval.max_score, ...)``) and this path answers
      from the sections Phase 8 selected, not from retrieved passages. Its equivalent here is
      "no item survived its quote check", which declines.
    * **FR-E4 faithfulness routing** — enforced, by
      :func:`_route_map_reduce_faithfulness`, which the STREAMING path calls too so the two
      cannot reach different verdicts. It was the one real gap in this path until Phase 12: a
      map-reduce answer carried no ``faithfulness_score`` and never routed to HITL review, so an
      operator who enabled this flag on a KB that relies on the review queue was turning that
      queue off for aggregate questions without being told. What is still NOT copied from the
      generation service is its self-correction and completeness re-prompting — both rewrite the
      answer by re-prompting over the same spans, and this answer was not written from spans.
      See that function."""
    settings = get_settings()
    if not settings.v2_generate or decision is None or selection is None or not selection.selected:
        return None, {}
    try:
        prepared = await _map_inputs(tenant_id, selection)
        if prepared is None:
            return None, {}
        cards, texts, jobs = prepared
        question = query.original_text or query.text
        facet_ids = sorted({f for _doc, f in decision.facets} | {f for c in cards.values() for f in c.facet_ids})
        gateway = await wiring.resolve_tenant_gateway(tenant_id)
        items, report, cached = await _mapped_items(
            tenant_id, jobs, question=question, facet_ids=facet_ids, gateway=gateway, settings=settings
        )
        if not items:
            _log.info("rag_v2_map_no_items", tenant_id=tenant_id, sections=len(jobs), failed=report.sections_failed)
            return None, {}

        coverage = Coverage(
            used=report.sections_contributed + cached,
            selected=len(selection.selected),
            total=selection.considered,
            # A section the map could only read part of still contributed, and saying so without
            # this qualifier overstates the answer — see `Coverage.line`.
            partial=report.sections_truncated,
            # A section whose call never came back is not a section with nothing in it.
            failed=report.sections_failed,
        )
        written = await _reduce_and_cite(
            tenant_id,
            gateway,
            ReduceInput(
                query=question,
                profiles=_profiles_block(cards),
                items=items,
                coverage=coverage,
                answer_lang=decision.answer_lang or "en",
            ),
            {sid: card.breadcrumb for sid, card in cards.items()},
            texts,
            max_tokens=settings.v2_reduce_max_tokens,
            concurrency=settings.v2_map_concurrency,
        )
        if written is None:
            return None, {}
        text, spans = written
        payload: dict[str, object] = {
            "rag_v2_map_sections": len(jobs),
            "rag_v2_map_cached": cached,
            "rag_v2_map_failed": report.sections_failed,
            # The kinds, not just the count: `read_timeout` and `unusable_reply` arrive at the
            # same `sections_failed` and ask for opposite responses.
            "rag_v2_map_failure_kinds": dict(report.failures),
            "rag_v2_map_items": len(items),
            "rag_v2_map_dropped_unquoted": report.dropped_unquoted,
            "rag_v2_map_dropped_over_cap": report.dropped_over_cap,
            "rag_v2_coverage_used": coverage.used,
            # The four measures below are INDEPENDENT and none of them substitutes for another.
            # `windows`/`calls` are the denominators (a retried window cost two calls, so they
            # differ); `truncated` is COVERAGE truncation, text never shown to the model;
            # `at_cap` is OUTPUT truncation, text shown and the reply cut before it could be
            # reported. A low `at_cap` is not a completeness score and a clean pass is not
            # evidence the output budget is sufficient — see `MapReport`.
            "rag_v2_map_windows": report.windows,
            "rag_v2_map_calls": report.calls,
            "rag_v2_map_retries": report.retries,
            "rag_v2_map_truncated": report.sections_truncated,
            "rag_v2_map_at_cap": report.calls_at_cap,
            "rag_v2_map_completion_tokens": report.completion_tokens,
        }
        _log.info("rag_v2_map_reduce", tenant_id=tenant_id, trace_id=trace_id, **payload)
        if report.dropped_over_cap:
            # Its own WARNING, not just a field on the line above: this one says the answer is
            # INCOMPLETE for a reason the operator can act on, and a count buried among six
            # others is how the earlier silent truncations went unnoticed for weeks.
            _log.warning(
                "rag_v2_map_items_over_cap",
                tenant_id=tenant_id,
                trace_id=trace_id,
                dropped=report.dropped_over_cap,
                detail=(
                    "the model returned more items than one section may contribute and the "
                    "surplus was discarded; lower RAG_V2_MAP_SECTION_CHARS so a window "
                    "holds fewer requirements, or raise the cap"
                ),
            )
        grounded = ground(tenant_id, question, text, spans, _MAP_REDUCE_CONFIDENCE, trace_id=trace_id)
        # FR-E4, the same way the streaming path does it and through the same function. Without
        # this the non-streaming map-reduce answer carried no faithfulness score and never
        # routed to HITL review — so a tenant relying on that queue had it silently off for
        # aggregate questions asked through the API while the console's streamed answers were
        # being reviewed normally.
        routed = await _route_map_reduce_faithfulness(
            grounded, spans, query, actor=actor, trace_id=trace_id, display_query=question
        )
        return routed, {**payload, "rag_v2_faithfulness": routed.faithfulness_score}
    except Exception as exc:  # fail-safe: the caller still answers the question the usual way
        _log.warning("rag_v2_map_reduce_skipped", tenant_id=tenant_id, error=str(exc)[:160])
        return None, {}


async def _mapped_items(
    tenant_id: str,
    jobs: Sequence[MapJob],
    *,
    question: str,
    facet_ids: Sequence[str],
    gateway: Any,
    settings: Any,
) -> tuple[list[MappedItem], MapReport, int]:
    """The map pass, through the cache. Returns ``(items, report, sections served from cache)``.

    THE CACHE IS CHECKED PER SECTION, not per question: two questions over the same document
    share nothing, but the same question asked twice re-reads every section for free. A hit and
    a miss produce the same item shape, so the reduce step cannot tell them apart — which is the
    property that makes caching safe to turn off."""
    signature = query_signature(question, facet_ids)
    model = getattr(gateway, "model_name", "")
    keys = {job.section_id: map_cache_key(job.section_id, signature, model) for job in jobs}
    hits = await read_map_cache(tenant_id, list(keys.values())) if settings.v2_map_cache_enabled else {}

    cached_items: list[MappedItem] = []
    misses: list[MapJob] = []
    for job in jobs:
        stored = hits.get(keys[job.section_id])
        if stored is None:
            misses.append(job)
            continue
        cached_items.extend(MappedItem.from_json(raw) for raw in stored)
    served = len(jobs) - len(misses)

    fresh: list[MappedItem] = []
    report = MapReport()
    if misses:
        mapper = LLMSectionMapper(
            gateway,
            max_tokens=settings.v2_map_max_tokens,
            concurrency=settings.v2_map_concurrency,
            section_chars=settings.v2_map_section_chars,
            max_windows=settings.v2_map_max_windows,
        )
        fresh, report = await mapper.map_sections(misses, query=question, profile_json="", facet_ids=facet_ids)
        if settings.v2_map_cache_enabled:
            by_section: dict[str, list[dict[str, object]]] = {job.section_id: [] for job in misses}
            for item in fresh:
                by_section.setdefault(item.section_id, []).append(item.as_json())
            await write_map_cache(
                tenant_id,
                [
                    MapCacheEntry(
                        cache_key=keys[section_id],
                        section_id=section_id,
                        query_sig=signature,
                        model_version=model,
                        items=stored,
                    )
                    for section_id, stored in by_section.items()
                ],
            )

    # Renumbered across the whole set, because the cached half and the fresh half each numbered
    # from m1 and the reduce prompt cites one flat list.
    merged = [*cached_items, *fresh]
    return (
        [
            MappedItem(
                item_id=f"m{i}",
                section_id=item.section_id,
                doc_id=item.doc_id,
                facet_id=item.facet_id,
                statement=item.statement,
                quote=item.quote,
                values=item.values,
                clause_ref=item.clause_ref,
                is_amendment=item.is_amendment,
                old_value=item.old_value,
                new_value=item.new_value,
            )
            for i, item in enumerate(merged, start=1)
        ],
        report,
        served,
    )


def _profiles_block(cards: Mapping[str, SectionCard]) -> str:
    """The documents in scope, one line each, for the reduce prompt's opening rule ("what the
    document is and its purpose").

    Built from the SECTION cards already in hand rather than re-reading the profile cards: the
    reduce step needs to name the documents, and a second catalog round trip on the latency path
    to fetch a title the section rows already imply is a cost with no answer attached."""
    docs = sorted({card.doc_id for card in cards.values()})
    return "\n".join(f"- {doc_id}" for doc_id in docs)


@dataclass(slots=True)
class StreamedMapReduce:
    """What a streamed map-reduce run produced, for the caller's post-processing.

    The generator yields TEXT and fills this, rather than yielding a tagged union: the caller
    needs the assembled text and the spans only once the stream is done, and a union would make
    every consumer of the token stream pattern-match on something it does not care about."""

    text: str = ""
    spans: list[RetrievedSpan] = field(default_factory=list)
    items: int = 0
    calls: int = 0
    coverage: str = ""
    verified: str | None = None


@dataclass(frozen=True, slots=True)
class _StreamPlan:
    """The reduce plan for a streamed answer. A named record rather than a seven-tuple, because
    six of the seven fields are collections and a transposed pair would be silent."""

    gateway: LLMGateway
    data: ReduceInput
    breadcrumbs: dict[str, str]
    batches: list[tuple[str, Sequence[MappedItem]]]
    texts: dict[str, list[tuple[str, str]]]
    jobs: list[MapJob]
    cached: int


async def _stream_plan(
    tenant_id: str,
    query: RetrievalQuery,
    decision: RouterDecision | None,
    selection: SectionSelection | None,
    *,
    settings: Any,
) -> _StreamPlan | None:
    """Everything the streamed reduce needs, or None when this path declines.

    The MAP pass happens here, before a single token is emitted, and that is not a choice: an
    item's citation has to resolve before the bullet citing it goes on screen, and the reduce
    prompt cannot be built until the items exist. So what streams is the REDUCE, which is what
    the plan asks for — the map is a fixed cost paid up front, and it is the part the Phase-12
    cache makes free on a repeated question.

    Its own function so the generator is the emission loop and nothing else; the decline
    reasons are identical to the non-streaming path's."""
    if not settings.v2_generate or decision is None or selection is None or not selection.selected:
        return None
    prepared = await _map_inputs(tenant_id, selection)
    if prepared is None:
        return None
    cards, texts, jobs = prepared
    question = query.original_text or query.text
    facet_ids = sorted({f for _doc, f in decision.facets} | {f for c in cards.values() for f in c.facet_ids})
    gateway = await wiring.resolve_tenant_gateway(tenant_id)
    items, report, cached = await _mapped_items(
        tenant_id, jobs, question=question, facet_ids=facet_ids, gateway=gateway, settings=settings
    )
    if not items:
        _log.info("rag_v2_stream_no_items", tenant_id=tenant_id, sections=len(jobs))
        return None
    data = ReduceInput(
        query=question,
        profiles=_profiles_block(cards),
        items=items,
        coverage=Coverage(
            used=report.sections_contributed + cached,
            selected=len(selection.selected),
            total=selection.considered,
            partial=report.sections_truncated,
            failed=report.sections_failed,
        ),
        answer_lang=decision.answer_lang or "en",
    )
    breadcrumbs = {sid: card.breadcrumb for sid, card in cards.items()}
    batches = _reduce_batches(data, breadcrumbs, max_tokens=settings.v2_reduce_max_tokens)
    if not batches:
        return None
    return _StreamPlan(gateway, data, breadcrumbs, batches, texts, jobs, cached)


async def _stream_first_batch(
    gateway: LLMGateway,
    data: ReduceInput,
    facet: str,
    batch: Sequence[MappedItem],
    breadcrumbs: Mapping[str, str],
    heading: str,
    *,
    max_tokens: int,
    tenant_id: str,
) -> AsyncIterator[str]:
    """Stream the FIRST reduce batch from the model, heading first, yielding nothing on failure.

    The first batch is the only one worth streaming: time-to-first-token is the whole point, and
    nothing can block the first batch because it is first in the answer's order. The remaining
    batches run as ordinary completions behind it — their tokens could not be emitted until
    their turn came anyway.

    THE HEADING WAITS FOR THE FIRST REAL TOKEN. A model that returns nothing for this facet must
    not leave an empty heading on the reader's screen with the next facet's bullets under it."""
    body = ""
    head_sent = False
    held = ""  # trailing whitespace, not yet emitted — see below
    try:
        async for token in gateway.stream(build_reduce_prompt(data, facet, batch, breadcrumbs), max_tokens=max_tokens):
            body += token
            if not head_sent and body.strip():
                yield heading
                head_sent = True
            if not head_sent:
                continue
            # TRAILING WHITESPACE IS HELD BACK until a token with content follows it. The
            # non-streaming path strips each section body (`assemble_answer`), so emitting the
            # model's final space or newline verbatim would make the streamed answer differ from
            # the same question's unstreamed answer by exactly that whitespace — and the whole
            # point of the ordering work above is that the two are the SAME text. Leading
            # whitespace of the first token is dropped for the same reason.
            piece = held + token
            stripped = piece.rstrip()
            held = piece[len(stripped) :]
            if not body[: len(body) - len(token)].strip():
                stripped = stripped.lstrip()
            if stripped:
                yield stripped
    except Exception as exc:  # the rest of the answer still writes
        _log.warning("rag_v2_reduce_stream_failed", tenant_id=tenant_id, error=str(exc)[:120])


async def _stream_map_reduce(
    tenant_id: str,
    query: RetrievalQuery,
    decision: RouterDecision | None,
    selection: SectionSelection | None,
    *,
    trace_id: str,
    out: StreamedMapReduce,
) -> AsyncIterator[str]:
    """Stream a map-reduce answer piece by piece (RAG v2 Phase 12's "stream the reduce output").

    YIELDS NOTHING AND LEAVES ``out`` EMPTY when this path declines, which is how the caller
    tells "no map-reduce answer" from "a map-reduce answer that happened to be short". Every
    decline reason is the same as the non-streaming path's.

    THE ORDER IS THE ANSWER'S ORDER, NOT COMPLETION ORDER. The reduce calls run concurrently, so
    a later facet routinely finishes first — but emitting it first would make the streamed text
    differ from the text the same question returns unstreamed, and then the answer a reader sees
    depends on which transport their console happened to use. So completions are awaited in
    answer order: a facet that finished early waits its turn. What that costs is nothing, because
    the calls were already running; what it buys is that the two paths are the same answer.

    THE FIRST BATCH IS STREAMED FROM THE MODEL, the rest are not. Time-to-first-token is the
    only thing streaming is for, and the first batch is the one nothing can block — so it is
    consumed token by token through ``gateway.stream`` while the remaining batches run as
    ordinary completions in parallel behind it. Streaming all of them would buy nothing: their
    tokens could not be emitted until their turn came anyway.

    CITATION NUMBERS ARE ASSIGNED AS THE TEXT IS EMITTED, by the same
    :class:`_MarkerResolver` the non-streaming path uses in one call. First appearance and
    streaming order are the same order, which is what makes the numbering identical."""
    settings = get_settings()
    planned = await _stream_plan(tenant_id, query, decision, selection, settings=settings)
    if planned is None:
        return
    gateway, data, batches = planned.gateway, planned.data, planned.batches
    items = data.items
    resolver = _MarkerResolver(items, planned.texts)
    emitted: list[str] = []

    def emit(piece: str) -> str:
        """Resolve this piece's markers, record it, and hand it to the stream."""
        resolved = resolver.rewrite(piece)
        emitted.append(resolved)
        return resolved

    # The opening line is a FACT the caller already holds, so it goes out before any model has
    # answered — which is most of what makes this feel streamed at all.
    yield emit(data.profiles.strip() + "\n\n")

    max_out = settings.v2_reduce_max_tokens
    gate = asyncio.Semaphore(max(1, settings.v2_map_concurrency))

    async def complete_one(facet: str, batch: Sequence[MappedItem]) -> str:
        async with gate:
            try:
                return await gateway.complete(
                    build_reduce_prompt(data, facet, batch, planned.breadcrumbs), max_tokens=max_out
                )
            except Exception as exc:  # one topic lost, not the answer
                _log.warning("rag_v2_reduce_facet_failed", tenant_id=tenant_id, facet=facet, error=str(exc)[:120])
                return ""

    # Everything after the first batch starts NOW and runs while the first is streaming.
    tail = [asyncio.create_task(complete_one(facet, batch)) for facet, batch in batches[1:]]
    try:
        last_facet = ""
        for position, (facet, batch) in enumerate(batches):
            heading = "" if facet == last_facet else f"### {facet.replace('_', ' ')}\n"
            last_facet = facet
            if position == 0:
                produced = False
                async for piece in _stream_first_batch(
                    gateway, data, facet, batch, planned.breadcrumbs, heading, max_tokens=max_out, tenant_id=tenant_id
                ):
                    produced = True
                    yield emit(piece)
                if not produced:
                    last_facet = ""  # nothing went out under this heading, so the next re-emits it
                continue
            body = await tail[position - 1]
            if not body.strip():
                last_facet = ""
                continue
            yield emit("\n\n" + heading + body.strip())
    finally:
        for task in tail:
            task.cancel()

    if not resolver.spans:
        # Grounded-or-abstain forbids a GROUNDED answer with no citations. The tokens are
        # already on the reader's screen, so the caller cannot un-send them — it emits a final
        # `done` carrying the honest status, exactly as the streaming path does for a blocked
        # or abstained answer. `out` stays empty so the caller knows to do that.
        _log.info("rag_v2_stream_uncited", tenant_id=tenant_id, items=len(items))
        return

    yield emit("\n\n" + data.coverage.line())
    # Filled AFTER the last yield, so `out.text` is the WHOLE answer including the coverage
    # line. The consumer's loop asks for one more value, the generator resumes here and
    # finishes — so this runs before the caller sees the stream end. Setting it earlier is the
    # bug this ordering exists to avoid: the caller would then ground an answer whose text was
    # missing the line the reader had already been shown.
    out.text = "".join(emitted).strip()
    out.spans = resolver.spans
    out.items = len(items)
    out.calls = len(batches)
    out.coverage = data.coverage.line()
    _log.info(
        "rag_v2_stream_map_reduce",
        tenant_id=tenant_id,
        trace_id=trace_id,
        sections=len(planned.jobs),
        cached=planned.cached,
        items=len(items),
        calls=len(batches),
        spans=len(resolver.spans),
    )


def _verify_answer(answer: GroundedAnswer, *, trace_id: str) -> tuple[GroundedAnswer, str | None]:
    """Verify a grounded answer's claims against the spans they CITE (RAG v2 Phase 10).

    Returns the answer — possibly with a "could not be verified" footnote, or with the failing
    claims dropped, per ``v2_verify_action`` — and the report as JSON for
    ``query_log.verification`` (the plan's §10 item 6, whose column got its writer in Phase 12).

    GATED ON HAVING CITATIONS rather than on the status. GROUNDED and FLAGGED_FOR_REVIEW both
    carry citations and both reach a reader; ABSTAINED carries none by contract, and "every claim
    failed" would be a false reading of "there was nothing to check against".

    THE STATUS IS NEVER CHANGED HERE, deliberately. The plan's remedy for a failing claim is
    editorial — footnote it or drop it — and both are stronger consequences than a score nudge,
    which is what the pre-existing numeric guard does and what discovery criticised it for having
    no teeth. Routing a DETERMINISTIC check straight into ABSTAIN would make one false positive
    withhold a correct answer; the notation guard already shows the care that needs (it caps into
    the FLAGGED band rather than failing outright, see `faithfulness.notation_flag_band_score`).

    NEVER RAISES. The answer is already grounded and already paid for.
    """
    settings = get_settings()
    if not settings.v2_verify or not answer.citations:
        return answer, None
    try:
        report = verify_claims(answer.answer, [c.quote for c in answer.citations])
    except Exception as exc:  # fail-safe, like every other derived artifact on this path
        _log.warning("verification_skipped", trace_id=trace_id, error=str(exc)[:160])
        return answer, None
    if report.checked:
        _log.info(
            "answer_verified",
            trace_id=trace_id,
            checked=report.checked,
            failed=len(report.failures),
            action=settings.v2_verify_action,
            reasons=sorted({r for v in report.failures for r in v.reasons}),
        )
    if report.failures:
        _log.warning(
            "answer_claims_unverified",
            trace_id=trace_id,
            failed=len(report.failures),
            claims=[v.claim[:120] for v in report.failures][:5],
        )
    action = settings.v2_verify_action
    if action == "footnote" and report.failures:
        answer = answer.model_copy(update={"answer": answer.answer + unverified_footnote(report)})
    elif action == "drop" and report.failures:
        answer = answer.model_copy(update={"answer": without_unverified(answer.answer, report)})
    return answer, report.as_json()


def _clip(text: str, limit: int) -> str:
    """Bound a stored free-text column, SAYING when it was cut rather than truncating silently."""
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


async def _answer_query_impl(  # noqa: PLR0915, PLR0912, PLR0911
    query: RetrievalQuery, *, actor: str, model: str | None = None, skip_intent: bool = False
) -> GroundedAnswer:
    """Retrieve -> ground-or-abstain -> cited answer (TDD §5.2). Fully tenant-scoped.

    ``skip_intent`` bypasses intent routing for this call — set by ``_answer_compare`` when it
    falls back to normal retrieval, so the fallback does not re-classify as ``compare`` and recurse.
    """
    trace_id = str(uuid.uuid4())
    audit: AuditService = wiring.audit_service()
    # RAG v2 Phase 12: per-stage durations for this question. One object per request, passed down
    # the call rather than a global — see `rag_common.timing`.
    timeline = StageTimeline()

    settings = get_settings()
    # Slice 2a: resolve the per-KB exact-term boost onto the query (no-op unless the flag is on and
    # the KB overrode it). Downstream `effective = query` copies inherit it via model_copy.
    query = await _with_exact_boost(query)
    # Slice 6, Phase C: augment the query with the canonical form of any molecule it names (no-op
    # unless the triple gate holds for the KB and a molecule was detected). Byte-identical off.
    query = await _with_chem_normalization(query)
    # Slice 2b: stamp the per-KB multi-vector leg flags (no-op unless the master flag is on and the
    # KB enabled a leg). Inherited by downstream `effective = query` copies via model_copy.
    query = await _with_multivector_flags(query)
    # Glossary/synonym expansion: bridge plain wording to canonical SOP/GMP terms on query.text
    # BEFORE retrieval (no-op off / nothing matches). original_text preserves the user's question.
    query = await _with_glossary_expansion(query)

    # FR-G1: prompt-injection guard runs BEFORE any model call — a block never reaches
    # retrieval or the LLM. Surfaces as ABSTAINED with a distinct GUARDRAIL audit event.
    if settings.guardrails_enabled:
        verdict = await wiring.input_guard().check(query.text)
        if not verdict.allowed:
            await audit.record(
                tenant_id=query.tenant_id,
                kb_id=query.tenant_id,
                trace_id=trace_id,
                action=AuditAction.GUARDRAIL,
                actor=actor,
                payload={"stage": "input", "category": verdict.category, "reason": verdict.reason},
            )
            _log.warning("guardrail_blocked_input", tenant_id=query.tenant_id, category=verdict.category)
            return abstain(
                query.tenant_id,
                query.original_text or query.text,
                0.0,
                reason="blocked_prompt_injection",
                trace_id=trace_id,
            )

    timeline.mark("guard")

    # Architecture-intent is no longer a REPLACE-the-answer route: the public answer_query wrapper
    # layers the diagram onto the normal grounded answer (auto+confirm) — see _finalize_diagram.

    # Process-intent: "explain/optimize the process/workflow of X" routes to the process-flow
    # understanding + gap-analysis flow (diagrams + gaps in structured). Flag-gated; off = unchanged.
    if settings.process_intent_enabled and process_flow.is_process_query(query.text):
        return await process_flow.process_grounded_answer(query, actor=actor, trace_id=trace_id)

    # FR-E5 semantic cache: a near-identical question (same KB, same ACL visibility)
    # returns its cached GROUNDED answer instantly — audited as a cache hit.
    query_vector, cached = await _semantic_cache_lookup(query)
    timeline.mark("cache")
    if cached is not None:
        await audit.record(
            tenant_id=query.tenant_id,
            kb_id=query.tenant_id,
            trace_id=trace_id,
            action=AuditAction.QUERY,
            actor=actor,
            payload={"query": query.text, "top_k": query.top_k, "cache_hit": True},
        )
        # Timed and logged like any other answer. A cache hit is part of the p95 the plan's §0
        # targets are about, and excluding it would make the number flattering rather than true.
        await _record_query_latency(query, cached, timeline, trace_id=trace_id, path="cache_hit")
        return cached

    # Agentic decision trace (FR-F): a transparent record of every step the pipeline takes for
    # this question. NON-AUTHORITATIVE — attached to the answer for explainability, never changes it.
    trace: list[AgentStep] = []

    # Agentic query planner (flag-gated MASTER, default OFF → the intent block below is UNCHANGED).
    # When ON it SUPERSEDES the intent classifier as the PRIMARY routing decision: the KB's model
    # reads the COMPLETE query + light KB context and returns a structured plan; the pipeline selects
    # its answer mechanism FROM the plan. Fail-safe: the planner falls back to the heuristic plan on
    # any error, an unresolved/ambiguous strategy falls through to normal RAG, and it never 500s.
    plan_payload: dict[str, object] = {}
    plan_clarify_options: list[str] = []
    planner_on = settings.query_planner_enabled and not skip_intent
    if planner_on:
        # FIX 3: the ENTIRE planner block is fail-safe — catalog fetch, context build, the plan call,
        # AND plan execution (incl. the whole-doc map-reduce) are wrapped so a gateway input-limit
        # error / catalog hiccup / audit failure FALLS THROUGH to the normal RAG path instead of
        # 500-ing the query. Worst case is today's behaviour.
        try:
            docs = await get_catalog().list_documents(query.tenant_id)
            visible = _visible_docs(docs, query)
            ctx = await _plan_context(query, visible)
            plan = await (await wiring.resolve_tenant_query_planner(query.tenant_id)).plan(query.text, ctx)
            trace.append(AgentStep(step="plan", detail=f"Plan: {plan.strategy}/{plan.coverage} — {plan.rationale}"))
            routed, plan_clarify_options = await _execute_query_plan(
                plan, query, visible, docs, actor=actor, model=model, trace_id=trace_id
            )
            if routed is not None:
                return routed
            # passages / whole_kb / ambiguous → normal RAG (unchanged); carry the plan onto the audit.
            plan_payload = {
                "plan_strategy": plan.strategy,
                "plan_coverage": plan.coverage,
                "plan_rationale": plan.rationale,
            }
            if plan_clarify_options:  # too many whole-doc matches → surface the ambiguity
                trace.append(AgentStep(step="plan", detail=f"Ambiguous target → {len(plan_clarify_options)} documents"))
                plan_payload["plan_ambiguous_targets"] = plan_clarify_options
        except Exception as exc:  # never 500 on a planner fault — fall through to normal RAG
            _log.warning("query_planner_failed_fallthrough_rag", tenant_id=query.tenant_id, error=str(exc)[:200])
            plan_payload, plan_clarify_options = {}, []

    # Semantic intent routing (flag-gated, default OFF → byte-identical): classify BEFORE
    # retrieval. Skipped when the planner is the primary router (planner_on). "catalog"
    # short-circuits to the KB inventory; "summarize" falls through; "rag" is unchanged.
    intent = "rag"
    if settings.intent_routing_enabled and not skip_intent and not planner_on:
        intent = await (await wiring.resolve_tenant_intent_classifier(query.tenant_id)).classify(
            query.text, has_history=False
        )
        trace.append(AgentStep(step="intent", detail=f"Intent: {intent}"))
        # catalog/count/filter/compare all work on the query path; summarize has no conversation
        # here and falls through to normal RAG. Each handler self-records the QUERY audit + intent.
        if intent == "catalog":
            return await _answer_catalog(query, actor=actor, trace_id=trace_id)
        if intent == "doc_summary":
            return await _answer_doc_summary(query, actor=actor, model=model, trace_id=trace_id)
        if intent == "count":
            return await _answer_count(query, actor=actor, trace_id=trace_id)
        if intent == "filter":
            return await _answer_filter(query, actor=actor, trace_id=trace_id)
        if intent == "compare":
            return await _answer_compare(query, actor=actor, model=model, trace_id=trace_id)

    # FR-D5: adaptive router picks the retrieval path (cheap standard vs heavier, deeper).
    # Default OFF -> unchanged behaviour. The decision rides the QUERY audit payload.
    route_payload: dict[str, object] = {}
    # RAG v2 Phase 7: ONE decision per question, read off the in-scope documents' Phase-3 cards.
    # Placed here, after the cache miss and before every existing routing block, because its
    # `doc_ids` must narrow the query the rest of them build on. None with the flag off.
    # Named `v2_decision`, not `decision`: the FR-D5 adaptive router below binds `decision` to
    # its own depth choice, and one name for two routing decisions is how a later edit silently
    # reads the wrong one.
    v2_decision = await _route_v2(query.tenant_id, query)
    if v2_decision is not None and v2_decision.doc_ids:
        # Applied through the FIRST-CLASS `doc_ids` predicate, not a metadata filter — D1's whole
        # finding was that a `doc_id` TAG silently empties the BM25 leg. A decision scoping to a
        # document is the one part of the plan's §7 output that has a consumer today.
        query = query.model_copy(update={"doc_ids": v2_decision.doc_ids})
    # RAG v2 Phase 8: the strategy the decision's intent selects. The SECTION PASS (aggregate,
    # compare, summarize) reads every section of the in-scope documents and reranks them —
    # Phase 9's map step is its consumer; here it is decided, audited and logged. The GRAIN
    # (lookup, procedural) narrows the hybrid legs to the v2 chunk levels. Both are None/empty
    # with the flag off, which leaves everything below byte-identical.
    v2_selection = await _select_sections(query.tenant_id, query, v2_decision)
    if settings.v2_retrieval and v2_decision is not None:
        v2_levels = plan_strategy(v2_decision.intent).levels
        if v2_levels:
            query = query.model_copy(update={"levels": v2_levels})
    effective = query
    heavy = not settings.adaptive_router_enabled  # if router off, "heavy" gate is open
    thorough = False  # comprehensive answer depth (heavy path only) — FR-D5 adaptive depth
    if settings.adaptive_router_enabled:
        decision = await wiring.query_router().route(query)
        heavy = decision.path == "heavy"
        thorough = heavy
        top_k = settings.heavy_top_k if heavy else query.top_k  # more evidence for a complex question
        effective = query.model_copy(update={"top_k": top_k})
        route_payload = {"route_path": decision.path, "route_reason": decision.reason, "route_top_k": top_k}
        trace.append(AgentStep(step="plan", detail=f"Planned route: {decision.path} ({decision.reason})"))
        _log.info("query_routed", tenant_id=query.tenant_id, path=decision.path, reason=decision.reason)

    # FR-D9: when a recency window/boost is requested, over-fetch so the post-filter keeps recall.
    want = query.top_k
    if _recency_active(query):
        effective = effective.model_copy(update={"top_k": min(50, want * settings.recency_overfetch)})
    # A question that NAMES a document is answered from that document. Over-fetch FIRST, because
    # the named document's passages may sit anywhere in the ranking — retrieving top_k and then
    # filtering would usually leave one span or none. See _apply_doc_scope.
    doc_scope = await _doc_scope_for(query)
    if doc_scope is not None:
        effective = effective.model_copy(update={"top_k": min(50, max(effective.top_k, want * 5))})
    # Everything from the cache miss to here is ROUTING: the agentic planner, the intent
    # classifier, the adaptive router and the document-scope resolver, each of which may call a
    # model. The sub-intent routes (catalog / doc_summary / count / filter / compare) have already
    # returned by this point into their own functions, which carry no timeline — see
    # `_record_query_latency`'s own limits.
    timeline.mark("route")

    # FR-D6: on the heavy path, transform the query into sub-queries, retrieve each, and
    # merge/dedupe by chunk keeping the best score (union lifts recall, not noise).
    transform_payload: dict[str, object] = {}
    # The Phase-7 router's facet-specific rewrites drive the EXISTING decomposition machinery
    # rather than a parallel one (discovery §8.4). They are produced only for `aggregate` and
    # `compare`, which is where the plan puts them, and they bypass the `heavy`/flag gate because
    # a router that asked for sub-queries has already made that decision — the gate exists to
    # decide WHETHER to decompose, and a router reading the document's facets is a better answer
    # to that than a depth label.
    router_subs = tuple(v2_decision.sub_queries) if v2_decision else ()
    if router_subs or (settings.query_transform_enabled and heavy):
        sub_queries = list(router_subs) or await (await wiring.query_transformer_for(effective.tenant_id)).transform(
            effective
        )
        results = [
            await (await wiring.retrieval_service_for(effective.tenant_id)).retrieve(
                effective.model_copy(update={"text": sq})
            )
            for sq in sub_queries
        ]
        # The budget scales with the number of sub-questions: capping at top_k meant a question
        # naming ten things could never hold more than ten passages, so most of them got none.
        # The merge is round-robin, so the extra room is shared out rather than taken by the
        # loudest sub-question (see merge_retrieved).
        budget = min(effective.top_k * max(1, len(sub_queries)), settings.transform_span_budget)
        spans, max_score = merge_retrieved([r.spans for r in results], budget)
        retrieval = RetrievalResult(query=effective, spans=spans, max_score=max_score, mode_used=get_retriever().mode)
        transform_payload = {
            "transform": "v2_router" if router_subs else settings.query_transform_backend,
            "subqueries": len(sub_queries),
            "span_budget": budget,
        }
        trace.append(AgentStep(step="decompose", detail=f"Decomposed into {len(sub_queries)} sub-questions"))
        _log.info("query_transformed", tenant_id=query.tenant_id, subqueries=len(sub_queries), merged=len(spans))
    else:
        retrieval = await _cross_lingual_retrieve(effective)
    # A GRAIN THAT MATCHED NOTHING FALLS BACK TO NO GRAIN. A v2 level is present only on chunks
    # written by the v2 INGEST path, which is a separate flag, so every document in every
    # deployment today is level-less — and a `lookup` narrowed to the three v2 levels would
    # retrieve zero spans and ABSTAIN on a question the pipeline could answer perfectly well a
    # moment before the flag was turned on. That is the worst shape a failure can take here:
    # silent, total, and indistinguishable from "the corpus does not say".
    #
    # So the narrowing is an OPTIMISATION, not a precondition, and it is retried without. One
    # extra retrieval, paid only in the case that is already broken. Same posture as the plan's
    # own fall-through-to-RAG rule: a strategy SELECTS an existing mechanism and can never leave
    # a question unanswered.
    if effective.levels and not retrieval.spans:
        _log.info("rag_v2_grain_empty_refetching_unscoped", tenant_id=query.tenant_id, levels=len(effective.levels))
        effective = effective.model_copy(update={"levels": ()})
        retrieval = await _cross_lingual_retrieve(effective)
    detail = f"Retrieved {len(retrieval.spans)} passages (top {retrieval.max_score:.2f})"
    trace.append(AgentStep(step="retrieve", detail=detail))

    await audit.record(
        tenant_id=query.tenant_id,
        kb_id=query.tenant_id,
        trace_id=trace_id,
        action=AuditAction.QUERY,
        actor=actor,
        payload={
            # The user's ORIGINAL question (retrieval ran on the possibly glossary-expanded text).
            "query": query.original_text or query.text,
            "top_k": effective.top_k,
            **_mode_payload(query),
            **route_payload,
            **transform_payload,
            **({"intent": intent} if settings.intent_routing_enabled else {}),
            **plan_payload,
            **_router_payload(v2_decision),
            **_strategy_payload(v2_decision, v2_selection),
        },
    )
    overrides = await kb_threshold_overrides(query.tenant_id)
    abstain_gate = overrides.get("abstain_threshold", settings.abstain_threshold)

    # FR-F2 agentic loop: too weak to answer? transform + retry once before giving up.
    retrieval, retry_payload = await _agentic_retry(effective, retrieval, abstain_gate)
    if retry_payload:
        await audit.record(
            tenant_id=query.tenant_id,
            kb_id=query.tenant_id,
            trace_id=trace_id,
            action=AuditAction.RETRIEVE,
            actor=actor,
            payload={"query": query.text, **retry_payload},
        )
        improved = retry_payload.get("agentic_retry") == "improved"
        detail = "Retried weak retrieval → stronger evidence" if improved else "Retried weak retrieval → no improvement"
        trace.append(AgentStep(step="retry", detail=detail))
    retrieval, graph_added = await _graph_augment(query, retrieval)  # GraphRAG local search (FR-D4)
    if graph_added:
        trace.append(AgentStep(step="graph", detail=f"Knowledge graph → +{graph_added} linked passages"))
        _log.info("graph_augmented", tenant_id=query.tenant_id, added=graph_added)
    retrieval, global_added = await _global_augment(query, retrieval)  # GraphRAG global search (slice 2)
    if global_added:
        trace.append(AgentStep(step="themes", detail=f"KB-wide themes → +{global_added} passages"))
        _log.info("graph_global_search", tenant_id=query.tenant_id, added=global_added)
    retrieval, nbr_added = await _neighbour_augment(query, retrieval)  # contiguity recall
    if nbr_added:
        trace.append(AgentStep(step="expand", detail=f"Adjacent passages → +{nbr_added} for continuity"))
        _log.info("neighbour_augmented", tenant_id=query.tenant_id, added=nbr_added)
    if _recency_active(query):
        trace.append(AgentStep(step="recency", detail="Applied time-window filter + freshness boost"))
    retrieval = await _apply_recency(query, retrieval, want)  # FR-D9 time-window filter + freshness boost
    retrieval, cov_added = await _coverage_augment(effective, retrieval, trace)
    if cov_added:
        want = max(want, len(retrieval.spans))  # a per-item quota must survive the later top_k slices
        # ...and must survive the PROMPT too. Widening top_k here is what puts the per-item
        # passages in front of the model; without it they are fetched and then discarded.
        budget = _prompt_budget_for_items(
            retrieval.query.top_k, items=len(enumerated_items(effective.text)), available=len(retrieval.spans)
        )
        if budget > retrieval.query.top_k:
            retrieval = retrieval.model_copy(update={"query": retrieval.query.model_copy(update={"top_k": budget})})
    if doc_scope is not None:
        retrieval = _apply_doc_scope(retrieval, doc_scope, want)
        trace.append(
            AgentStep(
                step="scope",
                detail=f"Question names a document → answering from {doc_scope.doc_id} "
                f"({len(retrieval.spans)} passages)",
            )
        )
    retrieval = _drop_navigation_spans(retrieval)

    await enforce_budget(query.tenant_id)  # FR-H7: refuse once the monthly token cap is hit
    retrieval = await _fuse_structured_evidence(query, retrieval, actor=actor)  # unified KB: docs + sources
    retrieval = await _mark_superseded(query, retrieval)  # version conflict: lead with what is in force
    retrieval, pc_added = await _parent_expand(query, retrieval)  # FR-C2: non-citable parent context bank
    if pc_added:
        trace.append(AgentStep(step="expand", detail=f"Parent context → +{pc_added} background section(s)"))
    # Retrieval proper plus every augmentation that adds or removes evidence: the agentic retry,
    # the graph and global legs, neighbour and coverage expansion, recency, structured fusion,
    # supersession marking and parent expansion. One stage, because they are one question —
    # "how long did it take to assemble the evidence".
    timeline.mark("retrieve")
    _fg, _tx = await _guardrail_gate_flags(query.tenant_id)  # Guardrail toggles: faithfulness-gate / toxicity
    # RAG v2 Phase 9: an aggregate-class question is answered by MAPPING over the sections Phase 8
    # selected and REDUCING the extracted items, rather than by one pass over the retrieved
    # passages. It DECLINES (None) whenever it cannot do better — flag off, no selection, no
    # section text, no item survived its quote check, nothing citable — and the ordinary
    # single-pass generation below runs instead. That fall-through is what makes it safe to turn
    # on: the worst outcome is the answer that would have been given anyway, one map pass later.
    mapped_answer, map_payload = await _map_reduce_answer(
        query.tenant_id, query, v2_decision, v2_selection, actor=actor, trace_id=trace_id
    )
    if mapped_answer is not None:
        trace.append(AgentStep(step="map", detail=f"Extracted {map_payload['rag_v2_map_items']} items"))
    answer = mapped_answer or await wiring.generation_service().answer(
        retrieval,
        actor=actor,
        trace_id=trace_id,
        model=model,
        threshold=overrides.get("abstain_threshold"),
        faithful_threshold=overrides.get("faithful_threshold"),
        flag_threshold=overrides.get("flag_threshold"),
        faithfulness_enabled=_fg,
        output_guard_enabled=_tx,
        persona=await kb_persona(query.tenant_id),
        depth="thorough" if thorough else "brief",
        max_tokens=settings.heavy_max_tokens if thorough else None,
        kb_id=query.tenant_id,  # single-KB query path: self-attribute GENERATE/FAITHFULNESS to the KB
        # Completeness-guard ceilings, resolved per-KB from the override seam (None ⇒ platform default).
        max_continuations=(int(_mc) if (_mc := overrides.get("completeness_max_continuations")) is not None else None),
        continuation_budget_factor=overrides.get("completeness_continuation_budget_factor"),
        coverage_max_loops=(
            int(_cl) if (_cl := overrides.get("completeness_coverage_max_loops")) is not None else None
        ),
        max_guard_calls=(int(_gc) if (_gc := overrides.get("completeness_max_guard_calls")) is not None else None),
        faithfulness_tolerance=overrides.get("completeness_faithfulness_tolerance"),
    )
    # A coverage-aware re-prompt (flag-gated in generation) records its own step; fold it into the
    # pipeline trace so the final overwrite below preserves it. Empty when the flag is off.
    # Generation AND the faithfulness verdict, which runs inside `generation_service().answer`.
    # They are not separable here without changing that service's contract; Phase 10 gives
    # verification its own stage when it becomes a step of its own.
    timeline.mark("generate")
    # RAG v2 Phase 10: verify each claim against the span it CITES, then footnote or drop what
    # does not check out. Placed here, immediately after generation, so everything downstream —
    # the followups, the semantic cache, the receipt — sees the answer a reader will see. No-op
    # with the flag off.
    answer, verification = _verify_answer(answer, trace_id=trace_id)
    timeline.mark("verify")
    if answer.agent_trace:
        trace.extend(answer.agent_trace)
    answer = await _attach_followups(answer, query, retrieval)  # FR-I8 (grounded only, from context)
    answer, clarify_n = await _attach_clarifications(answer, query, retrieval)  # Layer-2 item 4 (grounded only)
    # FIX 1: a whole_document target that matched TOO MANY docs surfaces those titles as
    # non-authoritative "did you mean…?" options (only when the RAG path didn't already offer some).
    if plan_clarify_options and not answer.clarification_options:
        answer = answer.model_copy(update={"clarification_options": plan_clarify_options})
    if clarify_n:
        trace.append(AgentStep(step="clarify", detail=f"offered {clarify_n} options"))
        await audit.record(
            tenant_id=query.tenant_id,
            kb_id=query.tenant_id,
            trace_id=trace_id,
            action=AuditAction.QUERY,
            actor=actor,
            payload={"query": query.text, "clarify_offered": clarify_n},
        )

    # Final step: the grounding/faithfulness verdict — always recorded so the trace shows how
    # the answer was gated (grounded / held for review / abstained).
    if answer.faithfulness_score is not None:
        verify_detail = f"Faithfulness {answer.faithfulness_score:.2f} → {answer.status.value}"
    else:
        verify_detail = f"Grounding check → {answer.status.value}"
    trace.append(AgentStep(step="verify", detail=verify_detail))
    answer = answer.model_copy(update={"agent_trace": trace})

    if query_vector is not None:
        semantic_cache.put(
            query.tenant_id, query.text, query_vector, list(query.acl), answer, query.clearance, query.top_k
        )

    # FR-E4: a faithfulness-borderline answer is NOT final — open a HITL review (reusing
    # the existing HITL flow) and return it with the review_id so the caller knows it is
    # pending human review, never surfacing it to the end user as a final answer.
    if answer.status is AnswerStatus.FLAGGED_FOR_REVIEW and await _hitl_hold_enabled(query.tenant_id):
        review, _outcome = await wiring.hitl_service().create(
            answer, actor=actor, reason=REASON_FAITHFULNESS, warnings=answer.unsupported_claims
        )
        answer = answer.model_copy(update={"review_id": review.review_id})
        _log.info("answer_flagged_for_review", tenant_id=query.tenant_id, review_id=review.review_id)
    await _emit_receipt(audit, answer, actor=actor)
    timeline.mark("post")
    await _record_query_latency(
        query,
        answer,
        timeline,
        trace_id=trace_id,
        path="answer",
        verification=verification,
        router_json=v2_decision.as_json() if v2_decision is not None else None,
        # Phase 8's column. `None` (no pass ran) and `()` (the pass ran and chose nothing) are
        # deliberately different values — see `_select_sections`.
        selected_sections=v2_selection.section_ids if v2_selection is not None else None,
    )
    return answer


async def _finish_streamed_map_reduce(
    streamed: StreamedMapReduce,
    query: RetrievalQuery,
    retrieval: RetrievalResult,
    trace: list[AgentStep],
    *,
    actor: str,
    trace_id: str,
    display_query: str,
) -> AsyncIterator[dict[str, Any]]:
    """Close out a streamed map-reduce answer: guard, audit, verify, ground, emit ``done``.

    THE SAME CONTROLS THE ORDINARY STREAM MEETS, in the same order, because the tokens are
    already on the reader's screen and a control that cannot un-send them has to be able to
    change the final STATUS instead. That is how the existing stream path handles a toxic
    answer, and this follows it exactly:

    * **The output guardrail (FR-G3)** — a block emits a ``done`` carrying an ABSTAIN rather
      than a grounded answer, so the console shows the real status under the streamed text.
    * **FR-E4 faithfulness** — the NLI check runs over the assembled answer against the spans
      its own citations resolved to, and routes grounded / flagged / abstained. Through
      :func:`_route_map_reduce_faithfulness`, which the NON-streaming path calls as well: for a
      while only this path had it, which meant the same question routed to human review through
      the console and not through the API.
    * **Phase-10 verification** — applied by the ``stream_answer`` WRAPPER to the ``done`` this
      yields, exactly as it is to every other ``done`` the impl emits. Not repeated here.

    PROGRESSIVE VERIFICATION IS NOT DONE HERE and the plan's own wording is why: "run
    verification on completed bullets progressively IF UI SUPPORTS IT, else verify before
    display (config)". No console in this repo consumes a per-bullet verdict event, so the
    second branch is the one that exists — and it is what this is. Emitting verdicts no client
    reads would be a protocol nobody implements; the config knob arrives with the client that
    needs it."""
    answer_text = streamed.text
    spans = streamed.spans

    if (await _guardrail_gate_flags(query.tenant_id))[1]:
        verdict = await get_output_guard().check(answer_text)
        if not verdict.allowed:
            await wiring.audit_service().record(
                tenant_id=query.tenant_id,
                kb_id=query.tenant_id,
                trace_id=trace_id,
                action=AuditAction.GUARDRAIL,
                actor=actor,
                payload={"stage": "output", "category": verdict.category, "reason": verdict.reason},
            )
            blocked = abstain(
                query.tenant_id, display_query, retrieval.max_score, reason="blocked_toxic_output", trace_id=trace_id
            ).model_copy(update={"agent_trace": trace})
            yield {"type": "done", "answer": blocked.model_dump(mode="json")}
            return

    await wiring.audit_service().record(
        tenant_id=query.tenant_id,
        kb_id=query.tenant_id,
        trace_id=trace_id,
        action=AuditAction.GENERATE,
        actor=actor,
        payload={
            "query": query.text,
            "output_sha256": hashlib.sha256(answer_text.encode()).hexdigest(),
            "citations": [s.chunk_id for s in spans],
            "rag_v2_map_items": streamed.items,
            "rag_v2_reduce_calls": streamed.calls,
        },
    )

    final = ground(query.tenant_id, display_query, answer_text, spans, _MAP_REDUCE_CONFIDENCE, trace_id=trace_id)
    final = await _route_map_reduce_faithfulness(
        final, spans, query, actor=actor, trace_id=trace_id, display_query=display_query
    )
    if final.faithfulness_score is not None:
        trace.append(
            AgentStep(step="verify", detail=f"Faithfulness {final.faithfulness_score:.2f} → {final.status.value}")
        )

    # NO `_verify_answer` HERE. The `stream_answer` WRAPPER applies Phase 10 to whatever `done`
    # this yields, and to every other `done` the impl emits — verifying again would footnote an
    # already-footnoted answer and double-count the dropped claims.
    final = final.model_copy(update={"agent_trace": trace})
    yield {"type": "done", "answer": final.model_dump(mode="json")}


async def _stream_answer_impl(  # noqa: PLR0915, PLR0912, PLR0911
    query: RetrievalQuery, *, actor: str, model: str | None = None
) -> AsyncIterator[dict[str, Any]]:
    """Streaming variant of :func:`answer_query` (SSE for the chat UI). Same grounded-or-
    abstain + faithfulness routing, but generation tokens are emitted as they arrive.
    Yields ``{"type": "phase"|"token"|"done", ...}`` dicts."""
    trace_id = str(uuid.uuid4())
    audit: AuditService = wiring.audit_service()
    settings = get_settings()
    # Slice 2a: per-KB exact-term boost (no-op unless the flag is on and the KB overrode it).
    query = await _with_exact_boost(query)
    # Slice 6, Phase C: augment the query with canonical molecule forms (no-op off/non-chemistry).
    query = await _with_chem_normalization(query)
    # Slice 2b: per-KB multi-vector leg flags (no-op unless the master flag is on and a leg is set).
    query = await _with_multivector_flags(query)
    # Glossary/synonym expansion: canonical SOP/GMP terms onto query.text before retrieval (no-op
    # off / nothing matches). original_text preserves the user's question for the answer + audit.
    query = await _with_glossary_expansion(query)
    # The user's ORIGINAL question for the USER-FACING answer (GroundedAnswer.query), while
    # retrieval/generation run on `query.text`. None original_text ⇒ `query.text` (path unchanged).
    display_query = query.original_text or query.text

    # G1 input guard — a block never reaches retrieval or the model.
    if settings.guardrails_enabled:
        verdict = await wiring.input_guard().check(query.text)
        if not verdict.allowed:
            await audit.record(
                tenant_id=query.tenant_id,
                kb_id=query.tenant_id,
                trace_id=trace_id,
                action=AuditAction.GUARDRAIL,
                actor=actor,
                payload={"stage": "input", "category": verdict.category, "reason": verdict.reason},
            )
            blocked = abstain(query.tenant_id, display_query, 0.0, reason="blocked_prompt_injection", trace_id=trace_id)
            yield {"type": "done", "answer": blocked.model_dump(mode="json")}
            return

    # Architecture-intent is no longer a REPLACE-the-answer route (which swapped the rich prose for a
    # drier caption): the public stream_answer wrapper layers the diagram onto the normal grounded
    # answer's done event (auto+confirm) — see stream_answer / _finalize_diagram.

    query_vector, cached = await _semantic_cache_lookup(query)
    if cached is not None:
        await audit.record(
            tenant_id=query.tenant_id,
            kb_id=query.tenant_id,
            trace_id=trace_id,
            action=AuditAction.QUERY,
            actor=actor,
            payload={"query": query.text, "top_k": query.top_k, "cache_hit": True},
        )
        yield {"type": "done", "answer": cached.model_dump(mode="json")}
        return

    yield {"type": "phase", "phase": "retrieving"}

    # D5 router + D6 transform + retrieval (mirrors answer_query; defaults are no-ops).
    trace: list[AgentStep] = []  # FR-F agentic decision trace, attached to the final answer

    # Agentic query planner (flag-gated MASTER, default OFF → this whole block is skipped and the
    # stream path is byte-identical). When ON it routes as the PRIMARY decision, mirroring the
    # non-streaming path. A non-streamable mechanism (whole-document map-reduce, catalog/count/
    # filter/compare) is answered non-streaming and emitted as a SINGLE chunk (parity with how the
    # conversation stream path emits a meta-answer whole). passages/whole_kb/ambiguous → normal stream.
    plan_payload: dict[str, object] = {}
    plan_clarify_options: list[str] = []
    if settings.query_planner_enabled:
        # FIX 3: the WHOLE planner block is fail-safe. A routed non-streamable answer is computed
        # first and only emitted if it succeeds; any fault (catalog / plan call / map-reduce) falls
        # through to the normal streaming path rather than 500-ing — worst case is today's behaviour.
        routed: GroundedAnswer | None = None
        try:
            docs = await get_catalog().list_documents(query.tenant_id)
            visible = _visible_docs(docs, query)
            ctx = await _plan_context(query, visible)
            plan = await (await wiring.resolve_tenant_query_planner(query.tenant_id)).plan(query.text, ctx)
            routed, plan_clarify_options = await _execute_query_plan(
                plan, query, visible, docs, actor=actor, model=model, trace_id=trace_id
            )
            if routed is None:
                trace.append(AgentStep(step="plan", detail=f"Plan: {plan.strategy}/{plan.coverage} — {plan.rationale}"))
                plan_payload = {
                    "plan_strategy": plan.strategy,
                    "plan_coverage": plan.coverage,
                    "plan_rationale": plan.rationale,
                }
                if plan_clarify_options:
                    plan_payload["plan_ambiguous_targets"] = plan_clarify_options
        except Exception as exc:  # never 500 on a planner fault — fall through to normal streaming
            _log.warning("query_planner_failed_fallthrough_rag", tenant_id=query.tenant_id, error=str(exc)[:200])
            routed, plan_payload, plan_clarify_options = None, {}, []
        if routed is not None:
            yield {"type": "token", "text": routed.answer}
            yield {"type": "done", "answer": routed.model_dump(mode="json")}
            return

    route_payload: dict[str, object] = {}
    # RAG v2 Phases 7 and 8 on the STREAMING path, placed exactly where the non-streaming path
    # places them and for the same reason: the decision's `doc_ids` must narrow the query every
    # block below builds on.
    #
    # They were missing here until Phase 12 wired the streamed reduce, and the gap mattered more
    # than it looks: the console's Ask streams, so a tenant who turned the v2 flags on got the
    # whole v2 query path through the API and NONE of it through the UI — the same question,
    # answered two different ways, with nothing saying so.
    v2_decision = await _route_v2(query.tenant_id, query)
    if v2_decision is not None and v2_decision.doc_ids:
        query = query.model_copy(update={"doc_ids": v2_decision.doc_ids})
    v2_selection = await _select_sections(query.tenant_id, query, v2_decision)
    if settings.v2_retrieval and v2_decision is not None:
        v2_levels = plan_strategy(v2_decision.intent).levels
        if v2_levels:
            query = query.model_copy(update={"levels": v2_levels})
    effective = query
    heavy = not settings.adaptive_router_enabled
    thorough = False  # FR-D5 adaptive depth — comprehensive answer on the heavy path
    if settings.adaptive_router_enabled:
        decision = await wiring.query_router().route(query)
        heavy = decision.path == "heavy"
        thorough = heavy
        top_k = settings.heavy_top_k if heavy else query.top_k
        effective = query.model_copy(update={"top_k": top_k})
        route_payload = {"route_path": decision.path, "route_reason": decision.reason, "route_top_k": top_k}
        trace.append(AgentStep(step="plan", detail=f"Planned route: {decision.path} ({decision.reason})"))
    want = query.top_k
    if _recency_active(query):  # FR-D9 over-fetch before the post-filter
        effective = effective.model_copy(update={"top_k": min(50, want * settings.recency_overfetch)})
    # A question that NAMES a document is answered from that document. Over-fetch FIRST, because
    # the named document's passages may sit anywhere in the ranking — retrieving top_k and then
    # filtering would usually leave one span or none. See _apply_doc_scope.
    doc_scope = await _doc_scope_for(query)
    if doc_scope is not None:
        effective = effective.model_copy(update={"top_k": min(50, max(effective.top_k, want * 5))})
    if settings.query_transform_enabled and heavy:
        subs = await (await wiring.query_transformer_for(effective.tenant_id)).transform(effective)
        svc = await wiring.retrieval_service_for(effective.tenant_id)
        results = [await svc.retrieve(effective.model_copy(update={"text": s})) for s in subs]
        # Same budget as the non-streaming path — the console streams, so capping here and not
        # there would make a chat answer thinner than the identical question through the API.
        budget = min(effective.top_k * max(1, len(subs)), settings.transform_span_budget)
        spans_m, max_score = merge_retrieved([r.spans for r in results], budget)
        retrieval = RetrievalResult(query=effective, spans=spans_m, max_score=max_score, mode_used=get_retriever().mode)
        trace.append(AgentStep(step="decompose", detail=f"Decomposed into {len(subs)} sub-questions"))
    else:
        retrieval = await (await wiring.retrieval_service_for(effective.tenant_id)).retrieve(effective)
    # A grain that matched nothing falls back to no grain — every document ingested before the
    # v2 flag is level-less, so a narrowed lookup would retrieve zero spans and abstain on a
    # question that was answerable a moment before the flag went on. Same rule as answer_query.
    if effective.levels and not retrieval.spans:
        _log.info("rag_v2_grain_empty_refetching_unscoped", tenant_id=query.tenant_id, levels=len(effective.levels))
        effective = effective.model_copy(update={"levels": ()})
        retrieval = await (await wiring.retrieval_service_for(effective.tenant_id)).retrieve(effective)
    rdetail = f"Retrieved {len(retrieval.spans)} passages (top {retrieval.max_score:.2f})"
    trace.append(AgentStep(step="retrieve", detail=rdetail))

    await audit.record(
        tenant_id=query.tenant_id,
        kb_id=query.tenant_id,
        trace_id=trace_id,
        action=AuditAction.QUERY,
        actor=actor,
        payload={
            # The user's ORIGINAL question (retrieval ran on the possibly glossary-expanded text).
            "query": query.original_text or query.text,
            "top_k": effective.top_k,
            **_mode_payload(query),
            **route_payload,
            **plan_payload,
            # Parity with the non-streaming path: the v2 decision and strategy on the SAME
            # QUERY event, never a second one — `count_actions` meters QUERY for FinOps showback.
            **_router_payload(v2_decision),
            **_strategy_payload(v2_decision, v2_selection),
        },
    )
    # The impl's locals are the only place the v2 decision exists, and the `query_log` row is
    # written by the WRAPPER (it is the only observer that can time a stream). Without this the
    # streaming path would log a NULL `router_json` and NULL `selected_sections` for every
    # answer — leaving the Phase-1 columns blind to exactly the path the console uses, which is
    # the opposite of what Phase 12 is for. An INTERNAL event carries them out; the wrapper
    # consumes it and never forwards it, so the client-facing protocol is unchanged.
    yield {
        "type": _V2_META,
        "router_json": v2_decision.as_json() if v2_decision is not None else None,
        "selected_sections": list(v2_selection.section_ids) if v2_selection is not None else None,
    }

    # Retrieval-confidence abstain (no model call). Per-KB override wins over global.
    overrides = await kb_threshold_overrides(query.tenant_id)
    abstain_gate = overrides.get("abstain_threshold", settings.abstain_threshold)
    retrieval, retry_payload = await _agentic_retry(effective, retrieval, abstain_gate)
    if retry_payload:
        await audit.record(
            tenant_id=query.tenant_id,
            kb_id=query.tenant_id,
            trace_id=trace_id,
            action=AuditAction.RETRIEVE,
            actor=actor,
            payload={"query": query.text, **retry_payload},
        )
        improved = retry_payload.get("agentic_retry") == "improved"
        rd = "Retried weak retrieval → stronger evidence" if improved else "Retried weak retrieval → no improvement"
        trace.append(AgentStep(step="retry", detail=rd))
    retrieval, graph_added = await _graph_augment(query, retrieval)  # GraphRAG local search (FR-D4)
    if graph_added:
        trace.append(AgentStep(step="graph", detail=f"Knowledge graph → +{graph_added} linked passages"))
    retrieval, global_added = await _global_augment(query, retrieval)  # GraphRAG global search (slice 2)
    if global_added:
        trace.append(AgentStep(step="themes", detail=f"KB-wide themes → +{global_added} passages"))
    retrieval, nbr_added = await _neighbour_augment(query, retrieval)  # contiguity recall
    if nbr_added:
        trace.append(AgentStep(step="expand", detail=f"Adjacent passages → +{nbr_added} for continuity"))
    if _recency_active(query):
        trace.append(AgentStep(step="recency", detail="Applied time-window filter + freshness boost"))
    retrieval = await _apply_recency(query, retrieval, want)  # FR-D9 recency filter + boost
    retrieval, cov_added = await _coverage_augment(effective, retrieval, trace)
    if cov_added:
        want = max(want, len(retrieval.spans))
    if doc_scope is not None:
        retrieval = _apply_doc_scope(retrieval, doc_scope, want)
        trace.append(
            AgentStep(
                step="scope",
                detail=f"Question names a document → answering from {doc_scope.doc_id} "
                f"({len(retrieval.spans)} passages)",
            )
        )
    retrieval = _drop_navigation_spans(retrieval)
    retrieval = await _fuse_structured_evidence(query, retrieval, actor=actor)  # unified KB: docs + sources
    retrieval = await _mark_superseded(query, retrieval)  # version conflict: lead with what is in force
    retrieval, pc_added = await _parent_expand(query, retrieval)  # FR-C2: non-citable parent context bank
    if pc_added:
        trace.append(AgentStep(step="expand", detail=f"Parent context → +{pc_added} background section(s)"))
    # THE RETRIEVAL-CONFIDENCE GATE DOES NOT APPLY TO A MAP-REDUCE ANSWER, and `v2_mappable`
    # is what exempts it. That gate is a judgement about how good the RETRIEVED PASSAGES are;
    # this path does not answer from them — it reads the sections Phase 8 selected, straight out
    # of the catalog. Abstaining here because a passage search scored badly would refuse a
    # question whose evidence was never going to come from a passage search, which on a
    # document-scoped aggregate is most of them. Its own equivalent of "no evidence" is "no item
    # survived its quote check", and that declines inside `_stream_map_reduce`.
    v2_mappable = settings.v2_generate and v2_selection is not None and bool(v2_selection.selected)
    if not v2_mappable and (not retrieval.spans or should_abstain(retrieval.max_score, abstain_gate)):
        await audit.record(
            tenant_id=query.tenant_id,
            kb_id=query.tenant_id,
            trace_id=trace_id,
            action=AuditAction.ABSTAIN,
            actor=actor,
            payload={
                "query": query.text,
                "max_score": retrieval.max_score,
                "reason": "retrieval_confidence_below_threshold",
            },
        )
        ans = abstain(query.tenant_id, display_query, retrieval.max_score, trace_id=trace_id)
        trace.append(AgentStep(step="verify", detail="Evidence below confidence threshold → abstained"))
        ans = ans.model_copy(update={"agent_trace": trace})
        yield {"type": "done", "answer": ans.model_dump(mode="json")}
        return

    # Stream generation from the tenant's model.
    await enforce_budget(query.tenant_id)  # FR-H7: refuse once the monthly token cap is hit
    yield {"type": "phase", "phase": "generating"}
    spans = retrieval.spans[: effective.top_k]
    persona = await kb_persona(query.tenant_id)
    prompt = build_prompt(query.text, spans, persona=persona, depth="thorough" if thorough else "brief")
    gateway = await wiring.resolve_tenant_gateway(query.tenant_id, model)
    model_name = gateway.model_name
    acc = ""
    # RAG v2 Phase 12: stream the REDUCE. An aggregate-class question is answered from the
    # sections Phase 8 selected, emitted in the answer's own order as each reduce call lands,
    # with the first batch streamed token by token so time-to-first-token is one model call
    # rather than all of them.
    #
    # WHAT HAPPENS AFTERWARDS IS THE EXISTING TAIL, UNCHANGED, and that is the whole reason this
    # is wired here rather than returning its own answer: the output guard, the GENERATE audit,
    # FR-E4 faithfulness routing and `ground` all run over `acc`/`spans` below. So a streamed
    # map-reduce answer is faithfulness-routed, which the non-streaming path still is not
    # (see `_map_reduce_answer`'s stated limits) — the streaming path is the stronger one here.
    streamed = StreamedMapReduce()
    async for piece in _stream_map_reduce(
        query.tenant_id, query, v2_decision, v2_selection, trace_id=trace_id, out=streamed
    ):
        acc += piece
        yield {"type": "token", "text": piece}
    if streamed.text:
        trace.append(AgentStep(step="map", detail=f"Extracted {streamed.items} items in {streamed.calls} call(s)"))
        async for ev in _finish_streamed_map_reduce(
            streamed, query, retrieval, trace, actor=actor, trace_id=trace_id, display_query=display_query
        ):
            yield ev
        return
    if acc:
        # Tokens went out and the run then declined — the only way here is "no resolvable
        # citation", because every earlier decline yields nothing. They cannot be un-sent, so the
        # honest close is an abstain carrying the reason, exactly as a blocked answer does.
        _log.info("rag_v2_stream_declined_after_tokens", tenant_id=query.tenant_id, chars=len(acc))
        ans = abstain(
            query.tenant_id, display_query, retrieval.max_score, reason="uncited_map_reduce", trace_id=trace_id
        ).model_copy(update={"agent_trace": trace})
        yield {"type": "done", "answer": ans.model_dump(mode="json")}
        return
    # Live table-loop guard: a greedy self-hosted model can stream duplicate table rows forever
    # (slow AND ugly in the UI). Buffer whole lines, drop any repeated pipe-row from the STREAM, and
    # if it repeats 3+ in a row the model is stuck — stop consuming (closes the upstream stream).
    line_buf = ""
    seen_rows: set[str] = set()
    dup_streak = 0
    async for token in gateway.stream(prompt, max_tokens=settings.heavy_max_tokens if thorough else None):
        acc += token
        line_buf += token
        emit = ""
        looping = False
        while "\n" in line_buf:
            line, line_buf = line_buf.split("\n", 1)
            s = line.strip()
            if s.startswith("|") and "---" not in s:  # a Markdown table row (not the |---| rule)
                if s in seen_rows:  # duplicate -> drop; count the streak
                    dup_streak += 1
                    if dup_streak >= 3:  # noqa: PLR2004  # 3 identical rows in a row = a generation loop
                        looping = True
                    continue
                seen_rows.add(s)
                dup_streak = 0
            emit += line + "\n"
        if emit:
            yield {"type": "token", "text": emit}
        if looping:
            break
    if line_buf:  # flush the trailing partial line
        yield {"type": "token", "text": line_buf}
    # Empty-generation recovery: a reasoning model (e.g. sarvam-105b) can spend its whole token
    # budget inside <think> and stream back NO final answer. Retry ONCE with a non-stream
    # completion at the larger budget (that path strips reasoning + maps null content to empty).
    if not acc.strip():
        _log.info("empty_stream_generation_retry", tenant_id=query.tenant_id, model=model_name, trace_id=trace_id)
        try:
            retried = (await gateway.complete(prompt, max_tokens=settings.heavy_max_tokens)).strip()
        except Exception as exc:  # a failed retry just falls through to abstain
            _log.warning("empty_generation_retry_failed", error=str(exc)[:160])
            retried = ""
        if retried:
            acc = retried
            yield {"type": "token", "text": retried}
    # No answer from the model → ABSTAIN. Never publish the empty fallback as GROUNDED: the
    # canned abstain text scores faithfulness ~1.0 (it contradicts nothing) and would route to
    # GROUNDED with citations attached — the "Grounded / out-of-scope" grounded-or-abstain bug.
    if not acc.strip():
        await audit.record(
            tenant_id=query.tenant_id,
            kb_id=query.tenant_id,
            trace_id=trace_id,
            action=AuditAction.ABSTAIN,
            actor=actor,
            payload={"query": query.text, "max_score": retrieval.max_score, "reason": "empty_generation"},
        )
        ans = abstain(
            query.tenant_id, display_query, retrieval.max_score, reason="empty_generation", trace_id=trace_id
        ).model_copy(update={"model": model_name})
        trace.append(AgentStep(step="verify", detail="Model returned no answer → abstained"))
        ans = ans.model_copy(update={"agent_trace": trace})
        yield {"type": "done", "answer": ans.model_dump(mode="json")}
        return
    answer_text = acc.strip()

    # The model emitted its designed "I don't know" refusal — abstain honestly (parity with the
    # non-streaming path), so a refusal is not streamed as a grounded answer.
    if is_refusal_text(answer_text):
        ans = abstain(
            query.tenant_id, display_query, retrieval.max_score, reason="out_of_scope", trace_id=trace_id
        ).model_copy(update={"model": model_name, "agent_trace": trace})
        yield {"type": "done", "answer": ans.model_dump(mode="json")}
        return

    # Completeness signal (Slice 1) — on the STREAMING path we DETECT and MARK only: bounded
    # continuation is harder mid-stream (the tokens are already on screen) and is DEFERRED to the
    # non-streaming path. NOTE(defer): stream continuation. We still never ship a cut-off answer
    # silently — a structural truncation cutoff or a still-omitted relevant point is surfaced as a
    # non-authoritative completeness field so the console can warn "answer may be incomplete".
    # finish_reason is not exposed on the stream, so truncation uses the structural heuristic.
    completeness_signal = None
    if settings.completeness_guard_enabled:
        completeness_signal = build_completeness(
            answer_text,
            spans,
            min_score=settings.coverage_min_score,
            coverage_ratio=settings.coverage_ratio,
            max_spans=settings.coverage_max_spans,
            truncated=looks_truncated(answer_text),
        )
        if not completeness_signal.complete:
            trace.append(AgentStep(step="completeness", detail=f"Incomplete (stream): {completeness_signal.reason}"))

    # Guardrail toggles for this KB (parity with the non-streaming path): faithfulness-gate + toxicity.
    _fg, _tx = await _guardrail_gate_flags(query.tenant_id)
    guard_on = _tx if _tx is not None else settings.guardrails_enabled
    faith_on = _fg if _fg is not None else settings.faithfulness_enabled

    # Output (toxicity) guardrail — the non-streaming path blocks toxic output; do the same here.
    # The tokens were already streamed, so on a block we emit a final `done` carrying an ABSTAIN
    # (blocked_toxic_output) — the console shows the final status, exactly as for flagged/abstained.
    if guard_on and get_output_guard() is not None:
        og = await get_output_guard().check(answer_text)
        if not og.allowed:
            await audit.record(
                tenant_id=query.tenant_id,
                kb_id=query.tenant_id,
                trace_id=trace_id,
                action=AuditAction.GUARDRAIL,
                actor=actor,
                payload={"stage": "output", "category": og.category, "reason": og.reason},
            )
            blocked = abstain(
                query.tenant_id, display_query, retrieval.max_score, reason="blocked_toxic_output", trace_id=trace_id
            ).model_copy(update={"model": model_name, "agent_trace": trace})
            yield {"type": "done", "answer": blocked.model_dump(mode="json")}
            return

    # Streamed generation: servers rarely report usage on streams, so meter with the
    # ~4-chars/token estimate and mark it estimated (FR-H4 token metering).
    usage = _estimate_usage(prompt, acc)
    await audit.record(
        tenant_id=query.tenant_id,
        kb_id=query.tenant_id,
        trace_id=trace_id,
        action=AuditAction.GENERATE,
        actor=actor,
        payload={
            "query": query.text,
            "model": model_name,
            "output_sha256": hashlib.sha256(answer_text.encode()).hexdigest(),
            "citations": [c.chunk_id for c in spans],
            "confidence": retrieval.max_score,
            **usage,
        },
    )

    # Faithfulness routing → grounded / flagged (HITL) / abstained. Gated on the resolved
    # Guardrail faithfulness-gate toggle (parity with the non-streaming path).
    if not faith_on:
        _off_update: dict[str, Any] = {"model": model_name}
        if completeness_signal is not None:
            _off_update["completeness"] = completeness_signal
        final = ground(
            query.tenant_id, display_query, answer_text, spans, retrieval.max_score, trace_id=trace_id
        ).model_copy(update=_off_update)
        trace.append(AgentStep(step="verify", detail=f"Grounding check → {final.status.value}"))
        final = await _stream_attach_clarifications(
            final, query, retrieval, actor=actor, trace_id=trace_id, trace=trace
        )
        if plan_clarify_options and not final.clarification_options:  # FIX 1: ambiguous whole-doc target
            final = final.model_copy(update={"clarification_options": plan_clarify_options})
        final = final.model_copy(update={"agent_trace": trace})
        yield {"type": "done", "answer": final.model_dump(mode="json")}
        async for ev in _stream_followups(final, query, retrieval):
            yield ev
        return

    checker = await wiring.resolve_tenant_faithfulness(query.tenant_id) or get_faithfulness_checker()
    # Slice 4: toggle the notation twin for this KB (no-op / same checker when the flag is off).
    checker = await wiring.apply_notation_grounding(checker, query.tenant_id)
    # FR-C2: give the checker the parent CONTEXT as extra NON-citation spans so a sentence supported
    # by parent-only background isn't falsely flagged; citations are still built from `spans` only.
    fv = await checker.check(answer_text, with_parent_context(spans), query.text)
    # AN UNAVAILABLE VERDICT CARRIES NO SCORE, and no number may be invented for it: 1.0 would
    # certify an answer nothing examined, 0.0 would condemn it on the same non-evidence. The
    # citations are intact, which is what the status is about, so it routes GROUNDED with the
    # score left unset -- exactly where a disabled gate leaves it -- and the provenance is logged
    # because a degradation with no named cause cannot be triaged.
    if fv.score is None:
        _log.warning(
            "faithfulness_unavailable",
            tenant_id=query.tenant_id,
            method=fv.method,
            reason=fv.degraded_reason,
        )
        status = AnswerStatus.GROUNDED
    else:
        status = route_faithfulness(
            fv.score,
            overrides.get("faithful_threshold", settings.faithful_threshold),
            overrides.get("flag_threshold", settings.flag_threshold),
        )
    await audit.record(
        tenant_id=query.tenant_id,
        kb_id=query.tenant_id,
        trace_id=trace_id,
        action=AuditAction.FAITHFULNESS,
        actor=actor,
        payload={
            "query": query.text,
            "score": fv.score,
            "outcome": status.value,
            "unsupported_claims": fv.unsupported_claims,
            "method": fv.method,
            "degraded": fv.degraded,
            "degraded_reason": fv.degraded_reason,
        },
    )
    _verdict_detail = "unavailable" if fv.score is None else f"{fv.score:.2f}"
    trace.append(AgentStep(step="verify", detail=f"Faithfulness {_verdict_detail} → {status.value}"))
    if status is AnswerStatus.GROUNDED:
        _grounded_update: dict[str, Any] = {
            "model": model_name,
            "agent_trace": list(trace),
            "unsupported_claims": list(fv.unsupported_claims),
            "verification": verification_block([fv]),
        }
        if completeness_signal is not None:
            _grounded_update["completeness"] = completeness_signal
        final = ground(
            query.tenant_id,
            display_query,
            answer_text,
            spans,
            retrieval.max_score,
            trace_id=trace_id,
            faithfulness_score=fv.score,
        ).model_copy(update=_grounded_update)
        if query_vector is not None:
            # Cache the answer WITH its agent_trace so a later cache-hit still surfaces the
            # pipeline flow in the UI (the put previously ran before the trace was attached,
            # so every streamed cache-hit came back with an empty trace / no "flow" panel).
            semantic_cache.put(
                query.tenant_id, query.text, query_vector, list(query.acl), final, query.clearance, query.top_k
            )
    elif status is AnswerStatus.FLAGGED_FOR_REVIEW and fv.score is not None:
        flagged = flag_for_review(
            query.tenant_id,
            display_query,
            answer_text,
            spans,
            retrieval.max_score,
            faithfulness_score=fv.score,
            unsupported_claims=fv.unsupported_claims,
            trace_id=trace_id,
        ).model_copy(update={"model": model_name})
        if await _hitl_hold_enabled(query.tenant_id):
            review, _outcome = await wiring.hitl_service().create(
                flagged, actor=actor, reason=REASON_FAITHFULNESS, warnings=flagged.unsupported_claims
            )
            final = flagged.model_copy(update={"review_id": review.review_id})
        else:
            final = flagged  # human_review OFF for this KB — return the flagged answer directly
    else:
        final = abstain(
            query.tenant_id,
            display_query,
            retrieval.max_score,
            reason="faithfulness_below_threshold",
            trace_id=trace_id,
        )
    final = await _stream_attach_clarifications(final, query, retrieval, actor=actor, trace_id=trace_id, trace=trace)
    if plan_clarify_options and not final.clarification_options:  # FIX 1: ambiguous whole-doc target
        final = final.model_copy(update={"clarification_options": plan_clarify_options})
    final = final.model_copy(update={"agent_trace": trace})
    yield {"type": "done", "answer": final.model_dump(mode="json")}
    async for ev in _stream_followups(final, query, retrieval):
        yield ev


async def _stream_attach_clarifications(
    answer: GroundedAnswer,
    query: RetrievalQuery,
    retrieval: RetrievalResult,
    *,
    actor: str,
    trace_id: str,
    trace: list[AgentStep],
) -> GroundedAnswer:
    """Streaming-path wrapper for :func:`_attach_clarifications`: attach the grounded options to
    the answer and, when any are offered, record the clarify trace step + QUERY audit note (so a
    streamed turn matches the non-streaming path). Fail-safe via ``_attach_clarifications``."""
    answer, clarify_n = await _attach_clarifications(answer, query, retrieval)
    if clarify_n:
        trace.append(AgentStep(step="clarify", detail=f"offered {clarify_n} options"))
        await wiring.audit_service().record(
            tenant_id=query.tenant_id,
            kb_id=query.tenant_id,
            trace_id=trace_id,
            action=AuditAction.QUERY,
            actor=actor,
            payload={"query": query.text, "clarify_offered": clarify_n},
        )
    return answer


async def _stream_followups(
    answer: GroundedAnswer, query: RetrievalQuery, retrieval: RetrievalResult
) -> AsyncIterator[dict[str, Any]]:
    """Emit suggested follow-ups AFTER the answer is already on screen (FR-I8).

    Generating them costs an extra model call, and on a reasoning model that is seconds —
    blocking the ``done`` event made the answer look finished while sources and follow-ups
    were still loading. They now stream as their own event."""
    if answer.status is not AnswerStatus.GROUNDED:
        return
    with_followups = await _attach_followups(answer, query, retrieval)
    if with_followups.suggested_followups:
        yield {"type": "followups", "items": with_followups.suggested_followups}


async def answer_across(
    org_id: str,
    kb_ids: list[str],
    text: str,
    *,
    top_k: int,
    acl: list[str],
    clearance: int = 0,
    actor: str,
    model: str | None = None,
) -> GroundedAnswer:
    """Answer one question across SEVERAL of the org's knowledge bases (federated ask).

    Retrieval runs per KB (each in its own namespace, ACL-filtered), results are merged
    keeping the best score per chunk, and citations carry a ``{kb}/`` prefix so the reader
    can see which knowledge base each source came from. Generation, guardrails, and
    faithfulness routing run once. When exactly ONE KB is actually in play (the common
    case: this is the org's generic ask endpoint, called for a single named KB), that
    audit trail — GUARDRAIL/QUERY/GENERATE/FAITHFULNESS — is recorded in THAT KB's own
    schema so its Quality dashboard sees the answer, matching the KB-scoped audit every
    other retrieval/generation call site already uses. A genuine multi-KB merge has no
    single owning KB and stays at the org level (unchanged)."""
    trace_id = str(uuid.uuid4())
    audit: AuditService = wiring.audit_service()
    settings = get_settings()
    audit_tenant_id = kb_ids[0] if len(kb_ids) == 1 else org_id

    if settings.guardrails_enabled:
        verdict = await wiring.input_guard().check(text)
        if not verdict.allowed:
            await audit.record(
                tenant_id=audit_tenant_id,
                trace_id=trace_id,
                action=AuditAction.GUARDRAIL,
                actor=actor,
                payload={"stage": "input", "category": verdict.category, "reason": verdict.reason},
            )
            return abstain(audit_tenant_id, text, 0.0, reason="blocked_prompt_injection", trace_id=trace_id)

    # Retrieve per KB CONCURRENTLY, then merge. Chunk ids are prefixed per KB before merging so
    # two KBs holding an identically-named document can never collapse into one entry. Was a
    # sequential loop (N KBs = N round-trips added up); now every KB's own-namespace retrieval
    # fires at once, so the wall-clock cost is the single slowest KB, not the sum of all of
    # them — this matters more now that SHARED_KB_ID rides along on every default ask.
    async def _retrieve_one(kb: str) -> list[RetrievedSpan]:
        result = await (await wiring.retrieval_service_for(kb)).retrieve(
            RetrievalQuery(tenant_id=kb, text=text, top_k=top_k, acl=acl, clearance=clearance)
        )
        label = kb.split("__", 1)[1] if "__" in kb else "default"
        return [
            s.model_copy(update={"chunk_id": f"{kb}::{s.chunk_id}", "doc_id": f"{label}/{s.doc_id}"})
            for s in result.spans
        ]

    per_kb: list[list[RetrievedSpan]] = await asyncio.gather(*(_retrieve_one(kb) for kb in kb_ids))
    spans, max_score = merge_retrieved(per_kb, top_k)

    await audit.record(
        tenant_id=audit_tenant_id,
        trace_id=trace_id,
        action=AuditAction.QUERY,
        actor=actor,
        payload={"query": text, "top_k": top_k, "federated_kbs": kb_ids},
    )

    retrieval = RetrievalResult(
        query=RetrievalQuery(tenant_id=audit_tenant_id, text=text, top_k=top_k, acl=acl, clearance=clearance),
        spans=spans,
        max_score=max_score,
        mode_used=RetrievalMode.HYBRID,
    )
    overrides = await kb_threshold_overrides(org_id)
    await enforce_budget(org_id)  # FR-H7
    _fg, _tx = await _guardrail_gate_flags(audit_tenant_id)  # Guardrail toggles: faithfulness-gate / toxicity
    answer = await wiring.generation_service().answer(
        retrieval,
        actor=actor,
        trace_id=trace_id,
        model=model,
        threshold=overrides.get("abstain_threshold"),
        faithful_threshold=overrides.get("faithful_threshold"),
        flag_threshold=overrides.get("flag_threshold"),
        faithfulness_enabled=_fg,
        output_guard_enabled=_tx,
        persona=await kb_persona(org_id),
        # Only a single-KB ask has one owning KB to self-attribute to; a real multi-KB merge stays org-level.
        kb_id=kb_ids[0] if len(kb_ids) == 1 else None,
    )
    if answer.status is AnswerStatus.FLAGGED_FOR_REVIEW and await _hitl_hold_enabled(audit_tenant_id):
        review, _outcome = await wiring.hitl_service().create(
            answer, actor=actor, reason=REASON_FAITHFULNESS, warnings=answer.unsupported_claims
        )
        answer = answer.model_copy(update={"review_id": review.review_id})
    await _emit_receipt(audit, answer, actor=actor)
    return answer


# --- Federated RAG federated query (cross-tenant, grant-gated) -----------------
# Scatter-gather across the caller's own KBs plus grant-authorised peer departments,
# reached through the FederationTransport seam: LocalTransport for CO-LOCATED peers
# (in-process, intra-org shared compute) and the libp2p MeshTransport for REMOTE peers —
# the one authoritative remote transport (ADR-0017/0020). Unlike answer_across (own KBs
# only, foreign KBs are a 403),
# this reaches ANOTHER department's KBs — but ONLY where that department has published
# an active grant to the caller, and ONLY the KBs/topics that grant names. A department
# with no grant is skipped SILENTLY (never a 403), so the caller learns nothing about
# un-granted departments. Cross-dept retrieval runs at least privilege: public-ACL
# content only (the grant authorises the crossing, not the caller's internal groups).


def federation_denial_alert(denied: list[str], threshold: int) -> bool:
    """A caller denied by >= threshold peers in one query looks like grant probing (SIEM).
    threshold <= 0 disables the alert."""
    return threshold > 0 and len(denied) >= threshold


def _federation_note(caller_dept: str, reached_peers: list[str], unreachable: list[str]) -> str:
    """Human-readable completeness caveat for a federated answer (T1)."""
    reached = ", ".join([caller_dept, *reached_peers])
    if not unreachable:
        return f"Answered from all granted sources: {reached}."
    return f"Answered from {reached}; unreachable (evidence missing): {', '.join(unreachable)}."


async def answer_federated(  # noqa: PLR0915, PLR0912
    caller_dept: str,
    own_kb_ids: list[str],
    federate_from: list[str],
    text: str,
    *,
    top_k: int,
    acl: list[str],
    clearance: int = 0,
    actor: str,
    model: str | None = None,
) -> GroundedAnswer:
    """Answer across the caller's own KBs PLUS any foreign departments that granted access.

    ``federate_from`` names foreign department ids to attempt. For each, active grants
    (issuer=foreign, subject=caller) authorise retrieval from ``grant.kb_ids``, capped at
    ``grant.max_results_per_query``. Merge → single grounded-or-abstain answer. The set of
    peers contacted, grants used, and departments skipped-for-no-grant is audited at the
    department level (FEDERATED_QUERY/FEDERATION_DENIED/FEDERATION_ALERT — a fact about
    caller_dept's relationship to its peers, not about any one KB). The answer-quality
    trail (GUARDRAIL/GENERATE/FAITHFULNESS) is different: when no peer federation was even
    attempted and exactly ONE local KB is in play, it is really a single-KB query and is
    recorded in THAT KB's own schema so its Quality dashboard sees it — matching every
    other KB-scoped retrieval/generation call site. A real multi-source merge has no single
    owning KB and stays at the department level (unchanged).
    """
    trace_id = str(uuid.uuid4())
    audit: AuditService = wiring.audit_service()
    settings = get_settings()
    audit_tenant_id = own_kb_ids[0] if len(own_kb_ids) == 1 and not federate_from else caller_dept

    if settings.guardrails_enabled:
        verdict = await wiring.input_guard().check(text)
        if not verdict.allowed:
            await audit.record(
                tenant_id=audit_tenant_id,
                trace_id=trace_id,
                action=AuditAction.GUARDRAIL,
                actor=actor,
                payload={"stage": "input", "category": verdict.category, "reason": verdict.reason},
            )
            return abstain(audit_tenant_id, text, 0.0, reason="blocked_prompt_injection", trace_id=trace_id)

    per_source: list[list[RetrievedSpan]] = []

    # 1) Local: the caller's own KBs, with the caller's real ACL groups.
    for kb in own_kb_ids:
        result = await (await wiring.retrieval_service_for(kb)).retrieve(
            RetrievalQuery(tenant_id=kb, text=text, top_k=top_k, acl=acl, clearance=clearance)
        )
        label = kb.split("__", 1)[1] if "__" in kb else "default"
        per_source.append(
            [
                s.model_copy(update={"chunk_id": f"{kb}::{s.chunk_id}", "doc_id": f"{label}/{s.doc_id}"})
                for s in result.spans
            ]
        )

    # 2) Federated: each foreign department, gated by an active publisher grant, reached
    #    through the configured transport — in-process for co-located peers, libp2p mesh for
    #    remote deployments (svc_control_plane.federation / ADR-0017). Grant resolution stays
    #    here: a department must hold a grant to attempt a peer, and that grant is the token
    #    the mesh presents to the remote node (which re-validates fail-closed).
    grant_store = wiring.grant_store()
    registry = wiring.federation_registry()
    used_grants: list[dict[str, object]] = []
    skipped: list[str] = []
    unreachable: list[str] = []
    reached_peers: list[str] = []
    denied: list[str] = []
    tracer = get_tracer("svc_control_plane.federation")

    async def _fetch_peer(foreign: str) -> PeerFetch | None:
        """Resolve grants and fetch one peer. Returns the PeerFetch, or None when the peer is
        skipped for holding no active grant. Raising is left to the caller to classify.

        Wrapped in its own OTel span so a distributed trace SHOWS the mesh fan-out: one child
        span per department reached, nested under the scatter-gather parent (each concurrent
        task inherits the parent context). No-op when tracing is disabled (dev/tests)."""
        with tracer.start_as_current_span("federation.peer") as _peer_span:
            _peer_span.set_attribute("federation.peer_dept", foreign)
            grants = await grant_store.get_active(issuer_dept_id=foreign, subject_dept_id=caller_dept)
            # Transitive revocation (A->B->C): a RE-DELEGATED grant is usable only while its WHOLE
            # ancestry is live — a revoked or expired root kills every descendant. Direct grants
            # (no parent) have no chain to walk, so they take the cheap path unchanged.
            grants = [g for g in grants if not g.parent_grant_id or await grant_store.is_chain_active(g.grant_id)]
            if not grants:
                _peer_span.set_attribute("federation.decision", "skipped")
                return None
            peer = await registry.transport_for(foreign).fetch(
                caller_dept=caller_dept, peer_dept_id=foreign, grants=grants, query=text, top_k=top_k
            )
            _peer_span.set_attribute("federation.reachable", peer.reachable)
            _peer_span.set_attribute("federation.reason", peer.reason)
            _peer_span.set_attribute("federation.span_count", sum(len(s) for s in peer.sources))
            return peer

    # Scatter-gather: fan out to every granted peer CONCURRENTLY (ADR-0017/0019). A federated
    # answer's latency is then the SLOWEST peer, not the SUM — and a slow or unreachable peer
    # can no longer serialize-block the peers queued behind it. Availability stays fail-open
    # (unreachable -> no spans, reported) and authorisation stays fail-closed. Results are
    # re-assembled in federate_from ORDER, so the merge (and therefore the answer) is identical
    # to the former sequential loop; only the wall-clock and blast-radius of one bad peer change.
    targets = [f for f in federate_from if f != caller_dept]
    if targets:
        with tracer.start_as_current_span("federation.scatter_gather") as _fed_span:
            _fed_span.set_attribute("federation.caller_dept", caller_dept)
            _fed_span.set_attribute("federation.peers_requested", len(targets))
            outcomes = await asyncio.gather(*(_fetch_peer(f) for f in targets), return_exceptions=True)
            for foreign, outcome in zip(targets, outcomes, strict=True):
                if isinstance(outcome, BaseException):
                    # An unexpected error in ONE peer's fetch degrades THAT peer to unreachable
                    # (an availability gap, reported), never aborting the whole federated answer.
                    _log.warning("federation_peer_fetch_failed", peer=foreign, error=str(outcome))
                    unreachable.append(foreign)
                    continue
                if outcome is None:
                    skipped.append(foreign)
                    continue
                peer = outcome
                per_source.extend(peer.sources)
                used_grants.extend(peer.used_grants)
                if not peer.reachable:
                    unreachable.append(foreign)
                elif peer.reason == "grant_rejected":
                    # Reached, but the peer refused (grant revoked/invalid at use). SOC-visible; kept
                    # OUT of the caller-facing coverage — an authorisation 'no' stays silent to the
                    # caller (same rule as an un-granted department), it is not an availability gap.
                    denied.append(foreign)
                else:
                    reached_peers.append(foreign)
            _fed_span.set_attribute("federation.reached", len(reached_peers))
            _fed_span.set_attribute("federation.unreachable", len(unreachable))
            _fed_span.set_attribute("federation.denied", len(denied))
            _fed_span.set_attribute("federation.skipped", len(skipped))

    spans, max_score = merge_retrieved(per_source, top_k)
    # Originating-node cross-encoder RE-RANK over the scatter-gathered union (Federated RAG data
    # plane): each peer scored its own spans independently, so a single score-merge can't compare
    # them fairly. When federation actually pulled peer evidence, the caller re-ranks the merged
    # set with ITS OWN reranker before synthesis — improving cross-peer precision. Best-effort: a
    # reranker hiccup falls back to the score-merge, never blocking the answer.
    if reached_peers and len(spans) > 1:
        try:
            reranker = await wiring.resolve_tenant_reranker(caller_dept)
            spans = await reranker.rerank(text, spans, top_k)
            max_score = spans[0].score if spans else max_score
        except Exception as exc:
            _log.warning("federation_rerank_failed", caller_dept=caller_dept, error=str(exc)[:160])

    await audit.record(
        tenant_id=caller_dept,
        trace_id=trace_id,
        action=AuditAction.FEDERATED_QUERY,
        actor=actor,
        payload={
            "query": text,
            "top_k": top_k,
            "own_kbs": own_kb_ids,
            "federate_from": federate_from,
            "grants_used": used_grants,
            "skipped_no_grant": skipped,
            "unreachable": unreachable,
            "denied": denied,
        },
    )
    # Denials get their OWN audit action so the SIEM/SOC can alert on them directly; a burst
    # of denials for one caller looks like grant probing and additionally raises an alert.
    if denied:
        await audit.record(
            tenant_id=caller_dept,
            trace_id=trace_id,
            action=AuditAction.FEDERATION_DENIED,
            actor=actor,
            payload={"denied_by": denied, "federate_from": federate_from},
        )
    if federation_denial_alert(denied, get_settings().federation_denial_alert_threshold):
        _log.warning("federation_denial_anomaly", caller=caller_dept, denied=denied)
        await audit.record(
            tenant_id=caller_dept,
            trace_id=trace_id,
            action=AuditAction.FEDERATION_ALERT,
            actor=actor,
            payload={"reason": "denial_burst", "denied_by": denied},
        )

    retrieval = RetrievalResult(
        query=RetrievalQuery(tenant_id=audit_tenant_id, text=text, top_k=top_k, acl=acl, clearance=clearance),
        spans=spans,
        max_score=max_score,
        mode_used=RetrievalMode.HYBRID,
    )
    overrides = await kb_threshold_overrides(caller_dept)
    await enforce_budget(caller_dept)  # FR-H7
    _fg, _tx = await _guardrail_gate_flags(audit_tenant_id)  # Guardrail toggles: faithfulness-gate / toxicity
    answer = await wiring.generation_service().answer(
        retrieval,
        actor=actor,
        trace_id=trace_id,
        model=model,
        threshold=overrides.get("abstain_threshold"),
        faithful_threshold=overrides.get("faithful_threshold"),
        flag_threshold=overrides.get("flag_threshold"),
        faithfulness_enabled=_fg,
        output_guard_enabled=_tx,
        persona=await kb_persona(caller_dept),
        # A single local KB with no peer federation is really a single-KB ask → self-attribute to it.
        kb_id=own_kb_ids[0] if len(own_kb_ids) == 1 and not federate_from else None,
    )
    answer = answer.model_copy(
        update={
            "federation_coverage": FederationCoverage(
                reached=[caller_dept, *reached_peers],
                unreachable=unreachable,
                complete=not unreachable,
                note=_federation_note(caller_dept, reached_peers, unreachable),
            )
        }
    )
    if answer.status is AnswerStatus.FLAGGED_FOR_REVIEW and await _hitl_hold_enabled(audit_tenant_id):
        review, _outcome = await wiring.hitl_service().create(
            answer, actor=actor, reason=REASON_FAITHFULNESS, warnings=answer.unsupported_claims
        )
        answer = answer.model_copy(update={"review_id": review.review_id})
    await _emit_receipt(audit, answer, actor=actor)
    return answer


# --- Conversational RAG (FR-I1) ---------------------------------------------
# A chat turn is condensed into a standalone query, then flows through the UNCHANGED
# answer pipeline (guardrails, grounding, thresholds, semantic cache, agentic retry all
# apply per turn), and both the user turn and the cited answer are persisted to the thread.


async def _condense_turn(tenant_id: str, history_user_turns: list[str], question: str) -> tuple[str, bool]:
    """Rewrite a follow-up into a standalone query. Returns ``(query_text, was_condensed)``.

    Heuristic backend is deterministic and adds no latency; the ``llm`` backend rewrites via
    the tenant's own gateway, falling back to the heuristic on any error (condensation must
    never break a turn)."""
    settings = get_settings()
    # "off" → conversational memory disabled: answer each question standalone, never fuse in prior
    # chat. (The thread still SHOWS earlier turns, but they no longer enter the query/context.)
    if settings.conversation_condense_backend == "off" or not history_user_turns:
        return question.strip(), False
    if settings.conversation_condense_backend == "llm":
        try:
            gateway = await wiring.resolve_tenant_gateway(tenant_id)
            prompt = build_condense_prompt(history_user_turns, question)
            rewritten = (await gateway.complete(prompt, max_tokens=64)).strip()
            if rewritten:
                return rewritten, rewritten.lower() != question.strip().lower()
        except Exception:  # never let condensation break the answer path
            _log.warning("condense_llm_failed_fallback_heuristic", tenant_id=tenant_id)
    condensed = heuristic_condense(history_user_turns, question)
    return condensed, condensed.lower() != question.strip().lower()


async def start_conversation(tenant_id: str, first_question: str) -> Conversation:
    """Open a new conversation thread titled from its first question."""
    store = wiring.conversation_store()
    conv = Conversation(conversation_id=str(uuid.uuid4()), tenant_id=tenant_id, title=default_title(first_question))
    await store.create(conv)
    _log.info("conversation_started", tenant_id=tenant_id, conversation_id=conv.conversation_id)
    return conv


async def _prepare_turn(tenant_id: str, conversation_id: str, question: str) -> list[str]:
    """Load prior user turns for condensation, then persist THIS user turn. Order matters:
    history is read before the new turn is added so condensation sees only the past."""
    store = wiring.conversation_store()
    if await store.get(tenant_id, conversation_id) is None:
        raise NotFoundError("conversation not found", details={"conversation_id": conversation_id})
    prior = await store.messages(tenant_id, conversation_id, get_settings().conversation_history_turns * 2)
    history_user = [m.content for m in prior if m.role == "user"]
    await store.add_message(
        tenant_id,
        Message(message_id=str(uuid.uuid4()), conversation_id=conversation_id, role="user", content=question),
    )
    return history_user


async def _finish_turn(tenant_id: str, conversation_id: str, answer: GroundedAnswer) -> None:
    store = wiring.conversation_store()
    await store.add_message(
        tenant_id,
        Message(
            message_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role="assistant",
            content=answer.answer,
            trace_id=answer.trace_id,
            status=answer.status.value,
            citations=answer.citations,
            faithfulness=answer.faithfulness_score,
        ),
    )
    await store.touch(tenant_id, conversation_id)


_EMOJI_RE = re.compile("[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff\U00002b00-\U00002bff\ufe0f]+")

_SMALLTALK_FALLBACK = (
    "Hello. Ask me anything about the documents in this knowledge base and I'll answer with "
    "citations — or tell you honestly when the sources don't cover it."
)


def build_smalltalk_prompt(text: str) -> str:
    """Prompt the KB's model to answer the ACTUAL pleasantry (a greeting, a thank-you, a
    sign-off each deserve a different reply) — never a canned line."""
    return (
        "You are a sovereign document-assistant. The user said something conversational rather "
        "than asking a question about their documents:\n\n"
        f'USER: "{text.strip()}"\n\n'
        "Reply to what they actually said in ONE short, warm sentence, then invite them to ask "
        "about their documents. Do NOT state any facts about documents and do NOT ask them to "
        "repeat themselves. Reply with the sentence only.\n\nREPLY:"
    )


async def _smalltalk_answer(tenant_id: str, question: str, *, model: str | None = None) -> GroundedAnswer:
    """A greeting is not a knowledge question: reply conversationally, with no retrieval and no
    QUERY audit event (so pleasantries never inflate usage or answer with an unrelated
    document). The reply is written by the KB's own model from what the user actually said;
    a static line is used only if that call fails."""
    reply = _SMALLTALK_FALLBACK
    try:
        gateway = await wiring.resolve_tenant_gateway(tenant_id, model)
        text = await gateway.complete(
            build_smalltalk_prompt(question), max_tokens=get_settings().llm_max_tokens, temperature=0.4
        )
        # Strip emoji — the console is an institutional surface, not a chat toy.
        cleaned = " ".join(_EMOJI_RE.sub("", text).strip().split())
        if cleaned:
            reply = cleaned[:400]
    except Exception as exc:  # a greeting must never surface an error
        _log.warning("smalltalk_reply_failed_fallback", tenant_id=tenant_id, error=str(exc)[:120])
    return GroundedAnswer(
        tenant_id=tenant_id,
        query=question,
        status=AnswerStatus.ABSTAINED,
        answer=reply,
        citations=[],
        confidence=0.0,
        abstain_reason="smalltalk",
    )


def _dedupe_table_rows(text: str) -> str:
    """Drop duplicate Markdown-table rows. A greedy self-hosted model can loop and repeat table
    rows forever (temperature only mitigates it); this is the deterministic guarantee. Keeps the
    FIRST occurrence of each pipe-row (header, separator, data) and leaves every non-table line
    untouched, so ordinary prose is never altered."""
    seen: set[str] = set()
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|"):
            if s in seen:
                continue
            seen.add(s)
        out.append(line)
    return "\n".join(out)


async def _finalize_diagram(
    answer: GroundedAnswer, query: RetrievalQuery, *, strong: bool | None = None, model: str | None = None
) -> GroundedAnswer:
    """Layer the auto+confirm architecture-diagram affordance onto ANY grounded answer, WITHOUT
    replacing its prose. STRONG architecture intent -> render Mermaid/PlantUML into ``structured``
    (the console draws it beneath the answer). Weaker POTENTIAL intent -> flip ``diagram_available``
    so the console offers an on-demand "generate diagram" chip. No signal / an abstain / a diagram
    already present -> untouched. Fully fail-safe (attach_diagram never raises)."""
    # Deterministic table-loop guard on EVERY grounded answer (before the arch early-return below).
    if answer.answer:
        deduped = _dedupe_table_rows(answer.answer)
        if deduped != answer.answer:
            answer = answer.model_copy(update={"answer": deduped})
    settings = get_settings()
    if not settings.architecture_intent_enabled or answer.status != AnswerStatus.GROUNDED:
        return answer
    if (answer.structured or {}).get("mermaid"):
        return answer  # a diagram is already attached (e.g. the explicit /architecture path)
    if strong is None:
        strong = architecture.is_architecture_query(query.text)
    if strong:
        return await architecture.attach_diagram(answer, query.tenant_id, query.text, model)
    if architecture.has_architecture_potential(query.text):
        return answer.model_copy(update={"diagram_available": True})
    return answer


async def _attach_processing(answer: GroundedAnswer, query: RetrievalQuery) -> GroundedAnswer:
    """Attach the document-processing completeness dimension. NEVER raises.

    THE ASSESSED SCOPE IS THE RESOLVED ``doc_ids`` UNION THE CITED DOCUMENTS. Neither alone works.
    Selection cannot be the basis -- a scoped document whose every section failed enrichment
    contributes no evidence and no citation, so it would vanish from its own assessment exactly
    when the failure is total. The whole KB cannot be either: one partial document would mark
    every answer the KB ever gives, which teaches a reader to ignore the warning.

    WHAT ``COMPLETE`` THEREFORE MEANS is complete for the ASSESSED SOURCES. For an unscoped
    question the cited set is not the universe of documents that could have contributed, and
    `scope_basis` records which basis was used so the claim can be read correctly. Closing that
    gap needs the whole-document reader to report the inventory it considered, including what it
    could not process.

    FAIL-SAFE AND NON-AUTHORITATIVE. A failure here leaves `processing` None, which a consumer
    must read as UNKNOWN rather than as a clean result; it never touches `status`, the citations,
    the faithfulness score or any verification field, and none of those touch it."""
    if not get_settings().v2_retrieval:
        return answer
    try:
        scoped = tuple(query.doc_ids or ())
        cited = tuple({c.doc_id for c in answer.citations if c.doc_id})
        if not scoped and not cited:
            return answer
        manifests = await factory_enrichment_manifests(query.tenant_id, sorted({*scoped, *cited}))
        assessed = assess_processing(scoped_doc_ids=scoped, cited_doc_ids=cited, manifests=manifests)
        # THE FIELD IS SET EVEN WHEN THERE IS NOTHING TO DISCLOSE, because "assessed and clean" is
        # a different statement from "nobody looked", and `processing=None` is the second one.
        report = ProcessingReport(
            status=assessed.status,
            scope_basis=assessed.scope_basis,
            documents_assessed=assessed.documents_assessed,
            documents_partial=assessed.documents_partial,
            documents_unknown=assessed.documents_unknown,
            sections_offered=assessed.sections_offered,
            sections_failed=assessed.sections_failed,
            note=assessed.disclosure(),
        )
        if assessed.status != "COMPLETE":
            _log.info(
                "processing_completeness",
                tenant_id=query.tenant_id,
                status=assessed.status,
                scope_basis=assessed.scope_basis,
                documents_assessed=assessed.documents_assessed,
                documents_partial=assessed.documents_partial,
                documents_unknown=assessed.documents_unknown,
                sections_failed=assessed.sections_failed,
                failure_classes=dict(assessed.failure_classes) or None,
                # DIAGNOSTIC ONLY, and never in the answer: the platform knows which section
                # RECORDS failed and has no extracted evidence of what they contain.
                failed_section_ids=list(assessed.failed_section_ids[:20]),
            )
        return answer.model_copy(update={"processing": report})
    except Exception as exc:  # fail-safe: a disclosure dimension must not be able to fail a query
        _log.warning("processing_completeness_failed", tenant_id=query.tenant_id, error=str(exc)[:160])
        return answer


async def answer_query(query: RetrievalQuery, *, actor: str, model: str | None = None) -> GroundedAnswer:
    """Public entry point: run the grounded-or-abstain pipeline, then layer the auto+confirm
    architecture-diagram affordance on the result (see :func:`_finalize_diagram`)."""
    answer = await _answer_query_impl(query, actor=actor, model=model)
    return await _attach_processing(await _finalize_diagram(answer, query, model=model), query)


async def stream_answer(
    query: RetrievalQuery, *, actor: str, model: str | None = None
) -> AsyncIterator[dict[str, Any]]:
    """Public streaming entry point: pass phase/token events through untouched, then finalize the
    ``done`` answer with the auto+confirm architecture-diagram affordance — emitting a
    ``diagramming the architecture`` phase first when a diagram will actually be rendered (typed
    extraction is a slower call than a normal answer, so the UI shows progress)."""
    strong = get_settings().architecture_intent_enabled and architecture.is_architecture_query(query.text)
    # RAG v2 Phase 12. The streaming path is timed HERE, at the wrapper, and not with the
    # stage-by-stage marks `_answer_query_impl` carries. Two reasons, and they are the same
    # reason: `_stream_answer_impl` is a generator with half a dozen `yield done; return` exits,
    # so instrumenting it inside would mean six call sites that must each stay correct; and the
    # number that actually matters for a stream is TIME TO FIRST TOKEN, which only an observer of
    # the event sequence can see. So this records the two honest numbers for a stream —
    # `first_token` and the total — and does not pretend to the breakdown.
    timeline = StageTimeline()
    first_token_seen = False
    v2_meta: dict[str, Any] = {}
    async for event in _stream_answer_impl(query, actor=actor, model=model):
        if event.get("type") == _V2_META:
            # Internal, never forwarded — see the note where it is emitted.
            v2_meta = {k: v for k, v in event.items() if k != "type"}
            continue
        if event.get("type") != "done":
            if not first_token_seen and event.get("type") == "token":
                first_token_seen = True
                timeline.mark("first_token")
            yield event
            continue
        answer = GroundedAnswer.model_validate(event["answer"])
        if strong and answer.status == AnswerStatus.GROUNDED and not (answer.structured or {}).get("mermaid"):
            yield {"type": "phase", "phase": "diagramming the architecture"}
        final = await _attach_processing(await _finalize_diagram(answer, query, strong=strong, model=model), query)
        # Phase 10 on the streaming path. HONEST LIMIT: the tokens have already been sent, so the
        # footnote reaches the final `done` answer and not the token stream — a console that
        # renders `done` shows it, one that only concatenates tokens does not. The plan's own
        # remedy is "verify before display (config)", which is a UI contract, not a pipeline
        # change. Recorded in discovery §8.22.
        final, verification = _verify_answer(final, trace_id=final.trace_id or "")
        timeline.mark("stream")
        # Before the yield: a client that disconnects mid-stream would otherwise abandon the
        # generator here and lose the row for exactly the slow answers worth measuring.
        await _record_query_latency(
            query,
            final,
            timeline,
            trace_id=final.trace_id or "",
            path="stream",
            verification=verification,
            router_json=v2_meta.get("router_json"),
            selected_sections=v2_meta.get("selected_sections"),
        )
        yield {"type": "done", "answer": final.model_dump(mode="json")}


async def answer_turn(
    tenant_id: str,
    conversation_id: str,
    question: str,
    *,
    actor: str,
    acl: list[str],
    clearance: int = 0,
    top_k: int = 5,
    mode: RetrievalMode = RetrievalMode.HYBRID,
    model: str | None = None,
    after: datetime | None = None,
    before: datetime | None = None,
    freshness: float = 0.0,
) -> GroundedAnswer:
    """Answer one conversational turn (non-streaming) and persist it to the thread."""
    history_user = await _prepare_turn(tenant_id, conversation_id, question)
    if is_smalltalk(question):
        answer = await _smalltalk_answer(tenant_id, question, model=model)
        await _finish_turn(tenant_id, conversation_id, answer)
        return answer
    # Semantic intent routing (flag-gated; guardrails run FIRST, before any handler).
    if get_settings().intent_routing_enabled:
        q = RetrievalQuery(tenant_id=tenant_id, text=question, top_k=top_k, mode=mode, acl=acl, clearance=clearance)
        blocked = await _guard_input(q, actor=actor)
        if blocked is not None:
            await _finish_turn(tenant_id, conversation_id, blocked)
            return blocked
        intent = await (await wiring.resolve_tenant_intent_classifier(tenant_id)).classify(question, has_history=True)
        meta_answer = await _turn_meta_answer(intent, q, tenant_id, conversation_id, question, actor=actor, model=model)
        if meta_answer is not None:
            await _finish_turn(tenant_id, conversation_id, meta_answer)
            return meta_answer
    condensed, _ = await _condense_turn(tenant_id, history_user, question)
    query = RetrievalQuery(
        tenant_id=tenant_id,
        text=condensed,
        top_k=top_k,
        mode=mode,
        acl=acl,
        clearance=clearance,
        after=after,
        before=before,
        freshness=freshness,
    )
    answer = await answer_query(query, actor=actor, model=model)
    await _finish_turn(tenant_id, conversation_id, answer)
    return answer


async def stream_turn(
    tenant_id: str,
    conversation_id: str,
    question: str,
    *,
    actor: str,
    acl: list[str],
    clearance: int = 0,
    top_k: int = 5,
    mode: RetrievalMode = RetrievalMode.HYBRID,
    model: str | None = None,
    after: datetime | None = None,
    before: datetime | None = None,
    freshness: float = 0.0,
) -> AsyncIterator[dict[str, Any]]:
    """Streaming conversational turn (SSE). Emits a ``condensed`` event when the follow-up
    was rewritten, then the normal phase/token/done stream, then persists the answer."""
    history_user = await _prepare_turn(tenant_id, conversation_id, question)
    if is_smalltalk(question):
        answer = await _smalltalk_answer(tenant_id, question, model=model)
        await _finish_turn(tenant_id, conversation_id, answer)
        yield {"type": "token", "text": answer.answer}
        yield {"type": "done", "answer": answer.model_dump(mode="json")}
        return
    # Semantic intent routing (flag-gated; guardrails run FIRST, before any handler). A catalog /
    # summarize meta-answer is not streamed token-by-token — it is emitted whole.
    if get_settings().intent_routing_enabled:
        q = RetrievalQuery(tenant_id=tenant_id, text=question, top_k=top_k, mode=mode, acl=acl, clearance=clearance)
        blocked = await _guard_input(q, actor=actor)
        if blocked is not None:
            await _finish_turn(tenant_id, conversation_id, blocked)
            yield {"type": "token", "text": blocked.answer}
            yield {"type": "done", "answer": blocked.model_dump(mode="json")}
            return
        intent = await (await wiring.resolve_tenant_intent_classifier(tenant_id)).classify(question, has_history=True)
        # A catalog/summarize/count/filter/compare meta-answer is emitted WHOLE, not streamed.
        meta_answer = await _turn_meta_answer(intent, q, tenant_id, conversation_id, question, actor=actor, model=model)
        if meta_answer is not None:
            await _finish_turn(tenant_id, conversation_id, meta_answer)
            yield {"type": "token", "text": meta_answer.answer}
            yield {"type": "done", "answer": meta_answer.model_dump(mode="json")}
            return
    condensed, was_condensed = await _condense_turn(tenant_id, history_user, question)
    if was_condensed:
        yield {"type": "condensed", "text": condensed}
    query = RetrievalQuery(
        tenant_id=tenant_id,
        text=condensed,
        top_k=top_k,
        mode=mode,
        acl=acl,
        clearance=clearance,
        after=after,
        before=before,
        freshness=freshness,
    )
    final: GroundedAnswer | None = None
    async for event in stream_answer(query, actor=actor, model=model):
        if event.get("type") == "done":
            final = GroundedAnswer.model_validate(event["answer"])
        yield event
    if final is not None:
        await _finish_turn(tenant_id, conversation_id, final)
