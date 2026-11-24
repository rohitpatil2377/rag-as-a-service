"""Dependency wiring — construct concrete adapters from configuration.

Services depend on the rag-core Protocols; this factory is the ONE place that knows
which concrete adapter backs each Protocol for the current config. Swapping a backend
(Qdrant->Milvus, TEI->vLLM, extractive_dev->vllm) is a change here only.

Adapters are cached per-process (they hold connection pools). The factory also builds
the :class:`HybridRetriever` with a catalog-backed chunk resolver.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from rag_clients.chandra import ChandraVisionGateway
from rag_clients.contextual import build_contextualizer
from rag_clients.external_fetcher import MinioExternalFetcher
from rag_clients.faithfulness import build_faithfulness_checker
from rag_clients.graph_store import InMemoryGraphStore, PostgresGraphStore
from rag_clients.guardrails import build_input_guard, build_output_guard
from rag_clients.intent_classifier import build_intent_classifier
from rag_clients.llm_gateway import build_llm_gateway
from rag_clients.memory import (
    InMemoryAuditSink,
    InMemoryCatalog,
    InMemoryExternalFetcher,
    InMemoryObjectStore,
    InMemorySparseIndex,
    InMemoryVectorStore,
    LocalHashEmbedder,
)
from rag_clients.minio_store import MinioObjectStore
from rag_clients.opensearch_index import OpenSearchSparseIndex
from rag_clients.postgres_audit import PostgresAuditSink
from rag_clients.postgres_catalog import PostgresCatalog
from rag_clients.qdrant_store import QdrantVectorStore
from rag_clients.reranker import LexicalReranker, build_reranker
from rag_clients.routing import build_query_router
from rag_clients.tei_embedder import TeiEmbedder
from rag_clients.transform import build_query_transformer
from rag_clients.vision import (
    EchoVisionGateway,
    LocalHashVisionEmbedder,
    OpenAICompatVisionGateway,
    TEIVisionEmbedder,
)
from rag_common import ConfigError, Settings, get_logger, get_settings, vision_collection
from rag_core import (
    ChunkLookup,
    Contextualizer,
    FaithfulnessChecker,
    Guardrail,
    HybridRetriever,
    QueryRouter,
    QueryTransformer,
)
from rag_core.chem import MoleculeNormalizer, NullMoleculeNormalizer
from rag_core.doc_profile import DocProfile
from rag_core.facet_vocab import FacetVocabulary
from rag_core.graph import DeclaredRelation
from rag_core.interfaces import (
    AuditSink,
    CatalogAdapter,
    Embedder,
    ExternalObjectFetcher,
    GraphStore,
    IntentClassifier,
    LLMGateway,
    ObjectStore,
    Reranker,
    SparseIndex,
    VectorStore,
)
from rag_core.layout import LayoutTree
from rag_core.mapreduce import MapCacheEntry
from rag_core.router import DocCard
from rag_core.section_enrich import EnrichmentManifest, SectionEnrichment
from rag_core.strategies import SectionCard
from rag_core.vision import VisionEmbedder, VisionLLMGateway
from rag_schemas import Chunk

_log = get_logger(__name__)


def _inmemory() -> bool:
    """True when the in-memory profile is active (dev/test/CI). Never in prod."""
    settings = get_settings()
    if settings.inmemory and settings.env == "prod":
        raise ConfigError("RAG_INMEMORY is forbidden in prod")
    return settings.inmemory


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    settings = get_settings()
    if _inmemory():
        return LocalHashEmbedder()
    if settings.embed_backend == "tei":
        return TeiEmbedder()
    if settings.embed_backend == "openai_compat":
        from rag_clients.openai_embedder import OpenAICompatEmbedder  # noqa: PLC0415

        return OpenAICompatEmbedder(
            settings.tei_url,
            settings.embed_model,
            dim=settings.embed_dim,
            api_key=settings.embed_api_key,
            api_key_header=settings.embed_api_key_header,
        )
    if settings.embed_backend == "bge_m3_multi":
        # Slice 2b: dense+sparse+ColBERT from one /embed call (Bearer auth). A DIFFERENT backend
        # type, not a URL swap (shape + auth + outputs differ), per the ADR-0016 per-KB seam. It
        # behaves as a plain dense embedder for every path that does not opt into embed_multi.
        from rag_clients.bge_m3_embedder import BgeM3MultiEmbedder  # noqa: PLC0415

        return BgeM3MultiEmbedder(
            settings.embed_multi_url or settings.tei_url,
            settings.embed_model,
            dim=settings.embed_dim,
            api_key=settings.embed_api_key,
            api_key_header=settings.embed_api_key_header,
            auth=settings.embed_api_auth,
        )
    raise ConfigError(f"unknown embed backend: {settings.embed_backend}")


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    settings = get_settings()
    if _inmemory():
        return InMemoryVectorStore()
    if settings.vector_backend == "qdrant":
        return QdrantVectorStore()
    raise ConfigError(f"unknown vector backend: {settings.vector_backend}")


@lru_cache(maxsize=1)
def get_vision_vector_store() -> VectorStore:
    """Cross-modal (CLIP) IMAGE-vector index — the SAME vector backend as the text store, but
    namespaced to a separate per-tenant collection (``vision_collection``). The vision embedder
    has its own dimensionality, so image vectors cannot share the text collection. Additive:
    used only when ``vision_recall_enabled``; the text store is untouched."""
    settings = get_settings()
    if _inmemory():
        return InMemoryVectorStore(namespace=vision_collection)
    if settings.vector_backend == "qdrant":
        return QdrantVectorStore(namespace=vision_collection)
    raise ConfigError(f"unknown vector backend: {settings.vector_backend}")


@lru_cache(maxsize=1)
def get_sparse_index() -> SparseIndex:
    settings = get_settings()
    if _inmemory():
        return InMemorySparseIndex()
    if settings.sparse_backend == "opensearch":
        return OpenSearchSparseIndex()
    raise ConfigError(f"unknown sparse backend: {settings.sparse_backend}")


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """Reranker (FR-D2). Cross-encoder for every profile that has real stores.

    The in-memory profile substitutes a twin for every backend (hash embedder, dict vector
    store); the reranker is no different — there is no model server to reach. That is a
    profile chosen by an explicit flag, NOT by the environment name, which is what the old
    ``env != "prod"`` branch got wrong: a staging deployment with real stores silently got
    bag-of-words ranking."""
    if _inmemory():
        _log.warning(
            "reranker_inmemory_twin",
            detail="RAG_INMEMORY: ranking is term overlap and max_score is an overlap fraction, "
            "NOT comparable to a cross-encoder abstain threshold",
        )
        return LexicalReranker()
    return build_reranker()


@lru_cache(maxsize=1)
def get_object_store() -> ObjectStore:
    settings = get_settings()
    if _inmemory():
        return InMemoryObjectStore()
    if settings.object_backend == "minio":
        return MinioObjectStore()
    raise ConfigError(f"unknown object backend: {settings.object_backend}")


@lru_cache(maxsize=1)
def get_external_object_fetcher() -> ExternalObjectFetcher:
    """Fetcher for external tenant-provided S3 buckets (FR-A2b). In-memory twin under CI.

    Signature deliberately unchanged: this is a process-wide cached singleton that tests and the
    connector registry seed through the no-argument call, so a provider parameter here would key
    the cache differently and hand the sync handler a fresh, empty twin. Other providers get
    their own getter below; callers choose with :func:`external_fetcher_for`."""
    if _inmemory():
        return InMemoryExternalFetcher()
    return MinioExternalFetcher()


@lru_cache(maxsize=1)
def get_azure_object_fetcher() -> ExternalObjectFetcher:
    """Fetcher for Azure Blob Storage containers (provider "azure"). The in-memory twin serves
    under CI exactly as for S3 -- the seam is the same, only the wire differs."""
    if _inmemory():
        return get_external_object_fetcher()  # one shared twin, so a test's seeded objects are visible
    from rag_clients.azure_fetcher import AzureBlobExternalFetcher  # noqa: PLC0415 -- optional SDK

    return AzureBlobExternalFetcher()


def external_fetcher_for(provider: str) -> ExternalObjectFetcher:
    """The fetcher for an ``ExternalObjectSource.provider`` value."""
    return get_azure_object_fetcher() if provider == "azure" else get_external_object_fetcher()


@lru_cache(maxsize=1)
def get_catalog() -> CatalogAdapter:
    if _inmemory():
        return InMemoryCatalog()
    return PostgresCatalog()


async def document_cards(tenant_id: str, *, doc_ids: Sequence[str] | None = None, limit: int = 20) -> list[DocCard]:
    """The Phase-3 profile cards the Phase-7 router routes on (RAG v2 Phase 7).

    Here for the same reason as the other v2-only catalog operations: not part of the
    ``CatalogAdapter`` Protocol, and ``factory`` is the one place allowed to know which concrete
    adapter is in play.

    FAIL-OPEN on an unknown adapter, like :func:`record_query_log` and unlike the erasure seams.
    An empty scope is a state the router is built for — it is what every KB looks like until a
    document is profiled — so a catalog this function does not recognise costs routing quality,
    not the answer. Failing closed here would turn an unrecognised backend into an outage on
    every question."""
    catalog = get_catalog()
    if isinstance(catalog, PostgresCatalog | InMemoryCatalog):
        return await catalog.document_cards(tenant_id, doc_ids=doc_ids, limit=limit)
    _log.warning("router_cards_adapter_unsupported", adapter=type(catalog).__name__, tenant_id=tenant_id)
    return []


async def document_sections(tenant_id: str, doc_ids: Sequence[str]) -> list[SectionCard]:
    """Every section of ``doc_ids``, for the Phase-8 aggregate selector (RAG v2 Phase 8).

    FAIL-OPEN, like :func:`document_cards` and for a sharper version of its reason: an empty
    section list means the aggregate strategy selects nothing and the pipeline falls through to
    the passage retrieval it has always run. A catalog this function does not recognise therefore
    costs the section pass, not the answer."""
    catalog = get_catalog()
    if isinstance(catalog, PostgresCatalog | InMemoryCatalog):
        return await catalog.document_sections(tenant_id, doc_ids)
    _log.warning("strategy_sections_adapter_unsupported", adapter=type(catalog).__name__, tenant_id=tenant_id)
    return []


async def stale_profiles(tenant_id: str, *, below: int) -> list[str]:
    """Documents whose stored profile card predates ``below`` (RAG v2 Phase 12).

    FAIL-CLOSED, unlike the other v2 catalog seams. The others feed a QUERY and an empty result
    costs answer quality; this feeds a re-enrichment run, and an empty result there reads as
    "nothing is stale" — an operator would conclude the corpus is current and move on. Raising
    is the honest answer to "I cannot tell you"."""
    catalog = get_catalog()
    if isinstance(catalog, PostgresCatalog | InMemoryCatalog):
        return await catalog.stale_profiles(tenant_id, below=below)
    raise ConfigError(f"stale_profiles is not supported by {type(catalog).__name__}")


async def read_map_cache(tenant_id: str, keys: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    """Cached Phase-9 map items (RAG v2 Phase 9).

    FAIL-OPEN, and the cheapest fail-open in this module: an unreadable cache is a cache miss,
    and a miss is the ordinary case. The caller maps the sections."""
    catalog = get_catalog()
    if isinstance(catalog, PostgresCatalog | InMemoryCatalog):
        try:
            return await catalog.read_map_cache(tenant_id, keys)
        except Exception as exc:
            _log.warning("map_cache_read_failed", tenant_id=tenant_id, error=str(exc)[:160])
            return {}
    _log.warning("map_cache_adapter_unsupported", adapter=type(catalog).__name__, tenant_id=tenant_id)
    return {}


async def write_map_cache(tenant_id: str, entries: Sequence[MapCacheEntry]) -> int:
    """Store Phase-9 map items. Returns rows written, 0 on any trouble.

    FAIL-OPEN for the same reason, from the other side: the items are already extracted and the
    answer is already being written from them. Losing the cache row costs the NEXT identical
    question one model call per section; raising here would cost THIS question its answer."""
    if not entries:
        return 0
    catalog = get_catalog()
    if isinstance(catalog, PostgresCatalog | InMemoryCatalog):
        try:
            return await catalog.write_map_cache(tenant_id, entries)
        except Exception as exc:
            _log.warning("map_cache_write_failed", tenant_id=tenant_id, error=str(exc)[:160])
            return 0
    _log.warning("map_cache_adapter_unsupported", adapter=type(catalog).__name__, tenant_id=tenant_id)
    return 0


async def record_query_log(
    tenant_id: str,
    *,
    query: str,
    answer: str,
    cited_doc_ids: Sequence[str],
    latency_ms: Mapping[str, int],
    router_json: str | None = None,
    selected_sections: Sequence[str] | None = None,
    verification: str | None = None,
) -> int | None:
    """Append one row to the RAG v2 ``query_log`` (Phase 12). Returns the row id, or None.

    Here for the same reason as :func:`purge_query_log_citing`: a v2-only catalog operation that
    is not part of the ``CatalogAdapter`` Protocol, and ``factory`` is the one place allowed to
    know which concrete adapter is in play.

    FAIL-OPEN on an unknown adapter, and this is the ONE v2 seam in this module that does not fail
    closed — the opposite of :func:`purge_query_log_citing` directly above it, deliberately. That
    one issues a certificate asserting an erasure happened, so an unreachable store has to stop
    it. This one is an observability record on the live query path: a deployment whose catalog
    this function does not recognise would otherwise have every ANSWER fail because its latency
    could not be logged. The miss is logged once per call and the answer is served."""
    catalog = get_catalog()
    if isinstance(catalog, PostgresCatalog | InMemoryCatalog):
        return await catalog.record_query_log(
            tenant_id,
            query=query,
            answer=answer,
            cited_doc_ids=cited_doc_ids,
            latency_ms=latency_ms,
            router_json=router_json,
            selected_sections=selected_sections,
            verification=verification,
        )
    _log.warning("query_log_adapter_unsupported", adapter=type(catalog).__name__, tenant_id=tenant_id)
    return None


async def purge_query_log_citing(tenant_id: str, doc_ids: set[str]) -> int:
    """DPDP erasure scrub of the RAG v2 ``query_log`` — rows quoting an erased document go too.
    Returns rows removed.

    Lives here rather than on the ``CatalogAdapter`` Protocol for the reason every other v2-only
    catalog operation does: ``factory`` is the ONE place allowed to know which concrete adapter is
    in play (see CLAUDE.md), which is what keeps ``compliance`` free of a concrete-adapter import.

    **Corrected in Phase 12.** This used to answer 0 for the in-memory profile, truthfully, on the
    grounds that nothing wrote the query log there. :func:`record_query_log` now does, under both
    profiles, so the in-memory branch purges for real — an erasure path that silently did nothing
    under the profile the unit and e2e gates run on would be untested where it matters most.

    See :meth:`rag_clients.postgres_catalog.PostgresCatalog.delete_query_log_citing` for the
    retention/purge contract and what is deliberately NOT erased.

    FAIL-CLOSED on an adapter it does not recognise. Returning 0 for "I don't know how to purge
    this backend" would be the same silent failure as duck-typing the method: erasure would finish
    and a SIGNED ErasureCertificate would assert a completeness nobody verified. A certificate is
    evidence handed to a data principal and to a regulator, so an unreachable store has to stop the
    erasure, not be rounded down to zero. The two known adapters are answered explicitly —
    Postgres purges, in-memory genuinely has no query log — and a third one is a deliberate
    decision somebody must come here and make."""
    catalog = get_catalog()
    if isinstance(catalog, PostgresCatalog):
        return await catalog.delete_query_log_citing(tenant_id, doc_ids)
    if isinstance(catalog, InMemoryCatalog):
        return await catalog.delete_query_log_citing(tenant_id, doc_ids)
    raise ConfigError(
        f"cannot scrub the query log for catalog adapter {type(catalog).__name__}: erasure would "
        "issue a certificate claiming a completeness it cannot verify"
    )


async def record_layout_key(tenant_id: str, doc_id: str, key: str) -> bool:
    """Record where a document's LAYOUT OVERLAY is stored (RAG v2 ``documents.minio_layout_key``).
    Returns False when the document has no catalog row.

    Lives here for the same reason as :func:`purge_query_log_citing`: this is a v2-only catalog
    column, not part of the ``CatalogAdapter`` Protocol that every service holds, and ``factory``
    is the ONE place allowed to know which concrete adapter is in play (see CLAUDE.md).

    FAIL-CLOSED on an adapter it does not recognise, rather than shrugging: the caller has by then
    WRITTEN an object to the store, and a swallowed failure would leave that object with nothing
    in the catalog pointing at it — an orphan no purge path enumerates. An unknown adapter is a
    decision somebody must come here and make."""
    catalog = get_catalog()
    if isinstance(catalog, PostgresCatalog | InMemoryCatalog):
        return await catalog.set_layout_key(tenant_id, doc_id, key)
    raise ConfigError(
        f"cannot record the layout-overlay key for catalog adapter {type(catalog).__name__}: the "
        "stored overlay would be an orphan with nothing in the catalog pointing at it"
    )


async def persist_document_structure(tenant_id: str, tree: LayoutTree) -> tuple[int, int]:
    """REPLACE a document's stored structure tree (RAG v2 ``sections`` + ``blocks``). Returns
    ``(sections, blocks)`` written.

    Lives here for the same reason as :func:`purge_query_log_citing` and
    :func:`record_layout_key`: these are v2-only catalog operations, not part of the
    ``CatalogAdapter`` Protocol that every service holds, and ``factory`` is the ONE place allowed
    to know which concrete adapter is in play (see CLAUDE.md). Unlike the query log, this one DOES
    have an in-memory twin — the structure tables are written by the ingest path, which the
    in-memory profile exercises in full, so a twin that silently wrote nothing would make every
    unit test on this path evidence about nothing.

    FAIL-CLOSED on an adapter it does not recognise. Not for the orphan reason
    ``record_layout_key`` fails closed on — nothing is written before this call — but because
    "structure was persisted" is the precondition every later phase reads, and a backend that
    silently stored nothing would surface as a document that simply has no structure, which is
    also what a plain-text document legitimately looks like. An unknown adapter is a decision
    somebody must come here and make."""
    catalog = get_catalog()
    if isinstance(catalog, PostgresCatalog | InMemoryCatalog):
        return await catalog.replace_document_structure(tenant_id, tree)
    raise ConfigError(
        f"cannot persist the document structure tree for catalog adapter {type(catalog).__name__}: "
        "a backend that stored nothing is indistinguishable from a document with no structure"
    )


async def persist_document_profile(tenant_id: str, doc_id: str, profile: DocProfile, *, revision: str = "") -> bool:
    """REPLACE a document's profile card — `documents` columns + `facets` + `doc_relations` (RAG v2
    Phase 3). Returns False when the document has no catalog row.

    Here for the same reason as :func:`persist_document_structure` and :func:`document_relations`:
    a v2-only catalog operation that is not part of the ``CatalogAdapter`` Protocol every service
    holds, and ``factory`` is the one place allowed to know which concrete adapter is in play.

    FAIL-CLOSED on an adapter it does not recognise, and this one is the sharpest case of the three.
    A silently unstored card is indistinguishable from a document the model declined to profile —
    which is a legitimate, expected outcome here (``LLMDocProfiler`` returns None on any model
    trouble) — so a backend that quietly wrote nothing would hide behind the fail-safe this phase
    deliberately has. An unknown adapter is a decision somebody must come here and make."""
    catalog = get_catalog()
    if isinstance(catalog, PostgresCatalog | InMemoryCatalog):
        return await catalog.replace_document_profile(tenant_id, doc_id, profile, revision=revision)
    raise ConfigError(
        f"cannot persist the document profile card for catalog adapter {type(catalog).__name__}: "
        "a backend that stored nothing is indistinguishable from a document that was never profiled"
    )


async def persist_section_enrichment(tenant_id: str, doc_id: str, enrichments: Mapping[str, SectionEnrichment]) -> int:
    """Write Phase-4 section enrichment onto this document's ``sections`` rows. Returns rows hit.

    Here for the same reason as :func:`persist_document_structure` and
    :func:`persist_document_profile`: a v2-only catalog operation that is not part of the
    ``CatalogAdapter`` Protocol every service holds, and ``factory`` is the one place allowed to
    know which concrete adapter is in play.

    FAIL-CLOSED on an adapter it does not recognise, for the reason the profile seam states and
    with the same sharpness: enrichment is fail-safe PER SECTION, so "nothing was stored" is an
    expected outcome here, and a backend that quietly wrote nothing would hide behind that
    fail-safe indefinitely. It would also take `sections.amends` with it, which is half of what
    Phase 6 seeds the graph from."""
    catalog = get_catalog()
    if isinstance(catalog, PostgresCatalog | InMemoryCatalog):
        return await catalog.replace_section_enrichment(tenant_id, doc_id, enrichments)
    raise ConfigError(
        f"cannot persist section enrichment for catalog adapter {type(catalog).__name__}: "
        "a backend that stored nothing is indistinguishable from a document that was never enriched"
    )


async def persist_enrichment_manifest(tenant_id: str, doc_id: str, manifest: EnrichmentManifest) -> bool:
    """Store what Phase 4 managed to do to this document. Returns whether a row was hit.

    Here for the same reason as :func:`persist_section_enrichment`: a v2-only catalog operation
    outside the ``CatalogAdapter`` Protocol, and ``factory`` is the one place allowed to know
    which concrete adapter is in play.

    FAIL-CLOSED on an unrecognised adapter, and here the reason is sharper than elsewhere. The
    manifest is the record that a document was only PARTLY readable. A backend that silently
    stored nothing would leave every document looking like one with no manifest -- which a
    completeness check must treat as UNKNOWN, so the failure would degrade every answer's
    disclosure rather than announcing itself."""
    catalog = get_catalog()
    if isinstance(catalog, PostgresCatalog | InMemoryCatalog):
        return await catalog.replace_enrichment_manifest(tenant_id, doc_id, manifest)
    raise ConfigError(
        f"cannot persist the enrichment manifest for catalog adapter {type(catalog).__name__}: "
        "a backend that stored nothing cannot be told from a document that was never enriched, "
        "and this record is what an answer's completeness disclosure is built from"
    )


async def enrichment_manifests(tenant_id: str, doc_ids: Sequence[str]) -> dict[str, tuple[EnrichmentManifest, str]]:
    """Stored manifest + the document's CURRENT content hash, for each doc that has a readable one.

    READ-SIDE, SO IT DOES NOT FAIL CLOSED. A completeness check that raised would turn a
    diagnostic dimension into an outage, and the answer path treats a missing manifest as UNKNOWN
    rather than as complete — which is the correct reading of an adapter that cannot answer.
    `{}` from an unrecognised backend therefore degrades the disclosure honestly instead of
    taking the query down with it."""
    if not doc_ids:
        return {}
    catalog = get_catalog()
    if isinstance(catalog, PostgresCatalog | InMemoryCatalog):
        return await catalog.enrichment_manifests(tenant_id, doc_ids)
    _log.warning(
        "enrichment_manifests_unavailable",
        adapter=type(catalog).__name__,
        detail="completeness will report UNKNOWN for every document in scope",
    )
    return {}


async def facet_vocabulary(tenant_id: str, doc_id: str) -> FacetVocabulary | None:
    """The facet vocabulary stored for this document, or None when there is nothing to reuse.

    READ-SIDE, SO IT DOES NOT FAIL CLOSED, and the degraded behaviour is the OLD behaviour: None
    means "nothing to reuse", so Phase 3's freshly proposed taxonomy is adopted — exactly what
    every ingest did before a vocabulary was persisted. An unrecognised adapter therefore loses
    the stability and nothing else, which is the honest degradation; raising here would turn a
    reproducibility improvement into an ingest outage."""
    catalog = get_catalog()
    if isinstance(catalog, PostgresCatalog | InMemoryCatalog):
        return await catalog.facet_vocabulary(tenant_id, doc_id)
    _log.warning(
        "facet_vocabulary_unavailable",
        adapter=type(catalog).__name__,
        detail="each ingest will adopt a freshly proposed taxonomy, as it did before pinning",
    )
    return None


async def document_relations(tenant_id: str, doc_id: str) -> list[DeclaredRelation]:
    """The relationships a document declares about itself (RAG v2 `doc_relations` +
    `sections.amends`), for seeding the knowledge graph.

    Here for the same reason as :func:`persist_document_structure`: a v2-only catalog operation
    that is not part of the ``CatalogAdapter`` Protocol every service holds, and ``factory`` is the
    one place allowed to know which concrete adapter is in play.

    FAIL-CLOSED on an unknown adapter rather than returning an empty list. Both tables are empty
    until Phases 3-4, so "no relations" is the correct answer for every document today — which is
    exactly why a backend that cannot answer must not be allowed to say the same thing. The caller
    (`pipeline._index_graph`) treats a raise as a graph-stage failure, which never breaks an
    ingest."""
    catalog = get_catalog()
    if isinstance(catalog, PostgresCatalog | InMemoryCatalog):
        return await catalog.document_relations(tenant_id, doc_id)
    raise ConfigError(
        f"cannot read declared document relations for catalog adapter {type(catalog).__name__}: "
        "a backend that returned nothing is indistinguishable from a document that declares nothing"
    )


@lru_cache(maxsize=1)
def get_graph_store() -> GraphStore:
    """Per-KB knowledge graph for GraphRAG (in-memory under CI; Postgres otherwise)."""
    if _inmemory():
        return InMemoryGraphStore()
    return PostgresGraphStore()


@lru_cache(maxsize=1)
def get_audit_sink() -> AuditSink:
    if _inmemory():
        return InMemoryAuditSink()
    return PostgresAuditSink()


@lru_cache(maxsize=1)
def get_llm_gateway() -> LLMGateway:
    return build_llm_gateway()


@lru_cache(maxsize=1)
def get_vision_llm_gateway() -> VisionLLMGateway:
    """Vision LLM (multimodal, reads image bytes). echo twin under CI; Chandra or a self-hosted
    VLM in prod. ``chandra`` = the one vision model that also serves RAG_OCR_BACKEND=chandra."""
    settings = get_settings()
    if _inmemory() or settings.vision_llm_backend == "echo":
        return EchoVisionGateway()
    if settings.vision_llm_backend == "chandra":
        return ChandraVisionGateway()
    if settings.vision_llm_backend == "chandra_api":
        from rag_clients.datalab import DatalabVisionGateway  # noqa: PLC0415 — Chandra via Datalab's cloud

        return DatalabVisionGateway()
    if settings.vision_llm_backend == "openai_compat":
        return OpenAICompatVisionGateway()
    raise ConfigError(f"unknown vision LLM backend: {settings.vision_llm_backend}")


@lru_cache(maxsize=1)
def get_vision_embedder() -> VisionEmbedder:
    """CLIP-style cross-modal embedder (Phase-2 visual retrieval). Twin under CI/local."""
    settings = get_settings()
    if _inmemory() or settings.vision_embed_backend == "local":
        return LocalHashVisionEmbedder()
    if settings.vision_embed_backend == "tei":
        return TEIVisionEmbedder()
    raise ConfigError(f"unknown vision embed backend: {settings.vision_embed_backend}")


@lru_cache(maxsize=1)
def get_faithfulness_checker() -> FaithfulnessChecker:
    """Faithfulness checker (FR-E4). LLM-judge reuses the sovereign gateway; lexical is CI default."""
    if get_settings().faithfulness_backend == "llm":
        return build_faithfulness_checker(get_llm_gateway())
    return build_faithfulness_checker()


@lru_cache(maxsize=1)
def get_contextualizer() -> Contextualizer:
    """Contextualizer (FR-C2). LLM backend reuses the sovereign gateway; metadata is the default."""
    if get_settings().contextual_retrieval_backend == "llm":
        return build_contextualizer(get_llm_gateway())
    return build_contextualizer()


@lru_cache(maxsize=1)
def get_query_router() -> QueryRouter:
    """Adaptive query router (FR-D5). LLM backend reuses the sovereign gateway; heuristic default."""
    if get_settings().router_backend == "llm":
        return build_query_router(get_llm_gateway())
    return build_query_router()


@lru_cache(maxsize=1)
def get_query_transformer() -> QueryTransformer:
    """Query transformer (FR-D6). Every backend except the purely deterministic one needs the
    sovereign gateway — including the default, which composes deterministic decomposition with
    model reformulation. Withholding the gateway here is what silently reduced `auto` to
    decomposition alone."""
    if get_settings().query_transform_backend == "decompose":
        return build_query_transformer()
    return build_query_transformer(get_llm_gateway())


@lru_cache(maxsize=1)
def get_intent_classifier() -> IntentClassifier:
    """Semantic intent classifier. LLM backend reuses the sovereign gateway; heuristic default."""
    if get_settings().intent_backend == "llm":
        return build_intent_classifier(get_llm_gateway())
    return build_intent_classifier()


def _guard_gateway() -> LLMGateway | None:
    return get_llm_gateway() if get_settings().guardrails_backend == "llm" else None


@lru_cache(maxsize=1)
def get_input_guard() -> Guardrail:
    """Prompt-injection guard (FR-G1). Heuristic default; LLM reuses the sovereign gateway."""
    return build_input_guard(_guard_gateway())


@lru_cache(maxsize=1)
def get_output_guard() -> Guardrail:
    """Toxicity guard (FR-G3). Heuristic default; LLM reuses the sovereign gateway."""
    return build_output_guard(_guard_gateway())


def build_molecule_normalizer(settings: Settings) -> MoleculeNormalizer:
    """The SMILES normalizer for chemical normalization (Slice 6, Phase C): the RDKit adapter when
    ``chem_normalization_enabled`` is on AND RDKit imports, else the fail-safe :class:`Null` twin.

    Fail-safe: flag off → Null (no lookup); flag on but RDKit absent → Null + a one-line WARN (the
    feature was asked for but the optional ``rag-clients[chem]`` extra is not installed). Either
    way :func:`rag_core.chem.canonical_terms` yields ``[]`` and every path stays byte-identical."""
    if not settings.chem_normalization_enabled:
        return NullMoleculeNormalizer()
    from rag_clients.chem_normalize import RDKitMoleculeNormalizer  # noqa: PLC0415 - optional rdkit extra

    normalizer = RDKitMoleculeNormalizer()
    if not normalizer.available:
        _log.warning(
            "chem_normalization_enabled_but_rdkit_missing",
            detail="RAG_CHEM_NORMALIZATION_ENABLED is on but RDKit is not installed "
            "(pip install 'rag-clients[chem]'); falling back to the no-op normalizer — no canonical "
            "chemical tokens will be written or matched.",
        )
        return NullMoleculeNormalizer()
    return normalizer


@lru_cache(maxsize=1)
def get_molecule_normalizer() -> MoleculeNormalizer:
    """Cached SMILES normalizer (mirrors the other ``get_*`` seams). Null unless chem normalization
    is enabled and RDKit is importable."""
    return build_molecule_normalizer(get_settings())


def build_retriever(embedder: Embedder, reranker: Reranker) -> HybridRetriever:
    """A hybrid retriever over the SHARED stores with the given embedder/reranker — the
    per-KB attach path (FR-E1 roles). The catalog-backed chunk resolver is identical to
    the platform retriever's; only the two model seams differ."""
    catalog = get_catalog()

    async def _resolve(tenant_id: str, chunk_ids: list[str]) -> dict[str, Chunk]:
        return await catalog.get_chunks(tenant_id, chunk_ids)

    settings = get_settings()
    # Cross-modal recall (additive/OFF): wire the CLIP embedder + image index as a third leg
    # only when enabled, so the text retrieval path is byte-identical when it is off.
    recall = settings.vision_recall_enabled
    return HybridRetriever(
        embedder=embedder,
        vector_store=get_vector_store(),
        sparse_index=get_sparse_index(),
        reranker=reranker,
        chunk_lookup=ChunkLookup(_resolve),
        candidate_pool=settings.rerank_candidates,
        rerank_top_k=settings.rerank_top_k,
        vision_embedder=get_vision_embedder() if recall else None,
        vision_store=get_vision_vector_store() if recall else None,
        # Slice 2b: the multivector legs additionally require the embedder + store to carry the
        # capability (checked in the retriever); the master flag gates the whole path.
        multivector_enabled=settings.multivector_enabled,
    )


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    """Build the hybrid retriever with a catalog-backed chunk resolver."""
    return build_retriever(get_embedder(), get_reranker())


async def close_all() -> None:
    """Best-effort close of every real adapter's connection pool.

    In-memory adapters hold no real connections and have no ``aclose``, so this only
    does work against the live sovereign stack. Call on API shutdown and between test
    runs against real stores — without it, real-backend clients (aiohttp/httpx
    sessions, SQLAlchemy engines) leak, which is invisible with the in-memory profile
    since it holds no such resources.
    """
    for getter in (
        get_embedder,
        get_vector_store,
        get_sparse_index,
        get_reranker,
        get_object_store,
        get_catalog,
        get_audit_sink,
        get_llm_gateway,
    ):
        if not getter.cache_info().currsize:
            continue
        adapter = getter()
        aclose = getattr(adapter, "aclose", None)
        if aclose is not None:
            await aclose()
