"""Dependency wiring for the control plane (modular monolith).

Builds the per-service objects from the concrete-adapter factory in ``rag_clients``.
In the monolith these are in-process calls; when a service is split out, this wiring
is replaced by an HTTP/gRPC client for that service — the orchestration code is
unchanged. Cached per-process (services hold connection pools).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from svc_control_plane.federation import FederationRegistry
    from svc_control_plane.tenant_control import TenantRegistry

from svc_doc_intelligence import DocIntelligenceService
from svc_doc_intelligence.ocr import OCREngine, build_ocr_engine
from svc_doc_intelligence.pii import build_pii_masker, parse_entities
from svc_generation import GenerationService
from svc_guardrails_eval import AuditService
from svc_indexing import IndexingService
from svc_ingestion import IngestionService
from svc_retrieval import RetrievalService

from rag_clients.clarify import LLMClarifyGenerator
from rag_clients.factory import (
    build_retriever,
    get_audit_sink,
    get_catalog,
    get_contextualizer,
    get_embedder,
    get_external_object_fetcher,
    get_faithfulness_checker,
    get_input_guard,
    get_llm_gateway,
    get_object_store,
    get_output_guard,
    get_query_router,
    get_query_transformer,
    get_reranker,
    get_retriever,
    get_sparse_index,
    get_vector_store,
)
from rag_clients.faithfulness import LLMFaithfulnessChecker, NumericallyGroundedChecker
from rag_clients.followups import LLMFollowupGenerator
from rag_clients.graph_extract import LLMDomainRelationExtractor, LLMGraphExtractor
from rag_clients.graph_summary import LLMCommunitySummarizer
from rag_clients.intent_classifier import HeuristicIntentClassifier, LLMIntentClassifier
from rag_clients.llm_gateway import OpenAICompatGateway
from rag_clients.openai_embedder import OpenAICompatEmbedder
from rag_clients.query_planner_llm import LLMQueryPlanner
from rag_clients.reranker import OpenAICompatReranker
from rag_clients.transform import build_query_transformer
from rag_clients.translation import LLMTranslator
from rag_common import get_logger, get_settings
from rag_core import (
    FaithfulnessChecker,
    GraphExtractor,
    Guardrail,
    HeuristicGraphExtractor,
    IntentClassifier,
    QueryPlanner,
    QueryRouter,
    QueryTransformer,
)
from rag_core.clarify import ClarifyGenerator, HeuristicClarifyGenerator
from rag_core.followups import FollowupGenerator, HeuristicFollowupGenerator
from rag_core.graph import CommunitySummarizer, HeuristicCommunitySummarizer, HeuristicDomainRelationExtractor
from rag_core.interfaces import Embedder, LLMGateway, Reranker
from rag_core.language import NoopTranslator, Translator
from rag_core.query_plan import HeuristicQueryPlanner
from rag_core.vision import VisionLLMGateway
from svc_control_plane import dual_control
from svc_control_plane.api_keys import ApiKeyStore, InMemoryApiKeyStore, PostgresApiKeyStore
from svc_control_plane.auth_tokens import AuthTokenStore, InMemoryAuthTokenStore, PostgresAuthTokenStore
from svc_control_plane.conversation import (
    ConversationStore,
    InMemoryConversationStore,
    PostgresConversationStore,
)
from svc_control_plane.egress_requests import (
    EgressRequestStore,
    InMemoryEgressRequestStore,
    PostgresEgressRequestStore,
)
from svc_control_plane.eval_schedule import (
    EvalScheduleStore,
    InMemoryEvalScheduleStore,
    PostgresEvalScheduleStore,
)
from svc_control_plane.feature_flags import FlagStore, InMemoryFlagStore, PostgresFlagStore
from svc_control_plane.grants import GrantStore, InMemoryGrantStore, PostgresGrantStore
from svc_control_plane.hitl import HitlService, HitlStore, InMemoryHitlStore, PostgresHitlStore
from svc_control_plane.identity import IdentityStore, InMemoryIdentityStore, PostgresIdentityStore
from svc_control_plane.ingest_jobs import IngestJobStore, InMemoryIngestJobStore, PostgresIngestJobStore
from svc_control_plane.knowledge_base import (
    InMemoryKnowledgeBaseStore,
    KnowledgeBaseStore,
    PostgresKnowledgeBaseStore,
)
from svc_control_plane.model_connection import (
    InMemoryModelConnectionStore,
    ModelConnectionStore,
    PostgresModelConnection,
    StoredModelConnection,
)
from svc_control_plane.org_units import InMemoryOrgUnitStore, OrgUnitStore, PostgresOrgUnitStore
from svc_control_plane.sql_connection import InMemorySqlConnectionStore, PostgresSqlConnectionStore, SqlConnectionStore

_log = get_logger(__name__)


@lru_cache(maxsize=1)
def audit_service() -> AuditService:
    from rag_common.siem import build_siem_forwarder  # noqa: PLC0415

    return AuditService(get_audit_sink(), build_siem_forwarder())


@lru_cache(maxsize=1)
def ingestion_service() -> IngestionService:
    return IngestionService(get_object_store(), get_catalog(), get_external_object_fetcher())


@lru_cache(maxsize=1)
def doc_intel_service() -> DocIntelligenceService:
    """The platform extraction service, shared by every tenant that has not registered its own
    OCR model (see ``doc_intel_service_for``)."""
    return build_doc_intel_service(platform_ocr_engine())


def build_doc_intel_service(ocr_engine: OCREngine) -> DocIntelligenceService:
    """Extraction with a GIVEN OCR engine. Everything else about extraction — masking, field
    extraction, the embedded-image thresholds — is platform behaviour and identical for every
    tenant; the engine is the one part an org can bring its own of."""
    s = get_settings()
    return DocIntelligenceService(
        masker=build_pii_masker(
            s.pii_masking_backend,
            entities=parse_entities(s.pii_entities),
            strategy=s.pii_strategy,
            hash_salt=s.pii_hash_salt,
        ),
        ocr_engine=ocr_engine,
        extract_fields_enabled=s.field_extraction_enabled,
        scanned_chars_per_page=s.ocr_scanned_chars_per_page,
        pdf_image_pages=s.pdf_image_pages,
        image_describer=describe_image_for_ingest,
        embedded_image_ocr=s.ocr_embedded_images_enabled,
        min_image_edge_px=s.ocr_embedded_min_edge_px,
        min_image_page_share=s.ocr_embedded_min_page_share,
        max_ocr_images=s.ocr_embedded_max_images,
        ocr_concurrency=s.ocr_concurrency,
        # RAG v2: capture the born-digital PDF layout overlay. Gated here, at the CAPTURE, not
        # only at `pipeline._persist_layout_overlay` — see the field comment on the service.
        pdf_layout_overlay=s.v2_ingest,
    )


# Per-tenant extraction services, keyed by the OCR engine they were built with. An org that adds
# its own OCR model gets its own service; every other tenant keeps sharing the platform one.
_tenant_doc_intel: dict[str, tuple[str, DocIntelligenceService]] = {}


async def resolve_tenant_ocr(tenant_id: str) -> OCREngine:
    """The OCR engine for a KB: the ORG-WIDE ocr connection if one is registered, else the
    deployment's configured engine.

    OCR is stateless — nothing stored depends on which engine read a page (the TEXT is stored, and
    a re-ingest re-reads it), so an org-wide OCR model takes effect immediately, like generation
    and reranking. An org connection is an OpenAI-compatible vision-chat endpoint, which is the
    Chandra engine's wire shape, and it is used even when the deployment default is a local engine:
    registering a model org-wide is an instruction to use it."""
    conn = await _resolve_org_default_connection(tenant_id, "ocr")
    if conn is None:
        return platform_ocr_engine()

    from svc_doc_intelligence.ocr import ChandraOCREngine  # noqa: PLC0415 — lazy: heavy optional deps

    from svc_control_plane import tenant_crypto  # noqa: PLC0415 — avoid import cycle at module load

    key = await tenant_crypto.decrypt(conn.tenant_id, conn.key_ciphertext) if conn.key_ciphertext else ""
    return ChandraOCREngine(conn.endpoint, conn.model, api_key=key, api_key_header=conn.header)


@lru_cache(maxsize=1)
def platform_ocr_engine() -> OCREngine:
    """The deployment's configured OCR engine (RAG_OCR_*), shared by every tenant without one."""
    s = get_settings()
    return build_ocr_engine(
        s.ocr_backend,
        lang=s.ocr_lang,
        preprocess=s.ocr_preprocess,
        binarize=s.ocr_binarize,
        psm=s.ocr_psm,
        oem=s.ocr_oem,
        model_dir=s.ocr_model_dir,
    )


async def doc_intel_service_for(tenant_id: str) -> DocIntelligenceService:
    """The extraction service a KB's ingests go through — identical to the platform one except
    for the OCR engine, which the org may have registered itself."""
    conn = await _resolve_org_default_connection(tenant_id, "ocr")
    if conn is None:
        return doc_intel_service()
    sig = f"{conn.endpoint}|{conn.model}|{conn.header}"
    cached = _tenant_doc_intel.get(tenant_id)
    if cached and cached[0] == sig:
        return cached[1]
    service = build_doc_intel_service(await resolve_tenant_ocr(tenant_id))
    _tenant_doc_intel[tenant_id] = (sig, service)
    return service


async def describe_image_for_ingest(image: bytes, mime: str, tenant_id: str) -> str:
    """The vision caption ADDED to an image's OCR text at ingest ("" when multimodal is off).

    The flag is read per call, not when the cached service is built, so turning multimodal on or
    off takes effect without rebuilding the extraction service. OCR runs either way."""
    settings = get_settings()
    # image_extract_backend picks what ACCOMPANIES the OCR text, not whether OCR runs: "ocr" means
    # the engine's reading alone (screenshots, forms, scanned pages — a caption adds noise there),
    # "caption" adds the vision model's description of what the picture shows. OCR itself is never
    # skipped: skipping it is what indexed a scanned e-ticket as 32 characters of caption.
    if not settings.multimodal_enabled or settings.image_extract_backend == "ocr":
        return ""
    from svc_control_plane import multimodal  # noqa: PLC0415 — multimodal resolves gateways through wiring

    return await multimodal.caption_image(image, mime=mime, tenant_id=tenant_id)


@lru_cache(maxsize=1)
def indexing_service() -> IndexingService:
    return IndexingService(get_embedder(), get_vector_store(), get_sparse_index(), get_catalog(), get_contextualizer())


@lru_cache(maxsize=1)
def retrieval_service() -> RetrievalService:
    return RetrievalService(get_retriever())


@lru_cache(maxsize=1)
def query_router() -> QueryRouter:
    return get_query_router()


@lru_cache(maxsize=1)
def query_transformer() -> QueryTransformer:
    """The PLATFORM-default transformer (RAG_LLM_*). Prefer :func:`query_transformer_for` —
    a transformer that cannot reach a model degrades to returning the query unchanged."""
    return get_query_transformer()


async def query_transformer_for(tenant_id: str) -> QueryTransformer:
    """The query transformer built on the KB's OWN model, like generation and image captioning.

    Reformulation is what rescues a question the reranker scored too low to answer, and it needs
    a model to do it. Bound to the platform default it reached a gateway this deployment has no
    working key for: every rewrite 401'd, the agentic retry reported "no_transform", and the
    question abstained. Measured on the live KB that exposed it — "explain me the whole
    architecture" against a KB whose one document IS an architecture diagram — retrieval scored
    0.0165 alone and 0.9132 once the KB's own model rewrote it.

    Falls back to the platform default when a KB has no org connection, which is the same order
    ``resolve_tenant_gateway`` uses."""
    if get_settings().query_transform_backend == "decompose":
        return query_transformer()  # deterministic: no model, nothing to bind
    try:
        return build_query_transformer(await resolve_tenant_gateway(tenant_id))
    except Exception as exc:  # a KB without a usable connection still gets the platform default
        _log.warning("query_transformer_tenant_gateway_unavailable", tenant_id=tenant_id, error=str(exc)[:200])
        return query_transformer()


@lru_cache(maxsize=1)
def identity_store() -> IdentityStore:
    """Accounts + organisations (in-memory under CI; Postgres otherwise)."""
    return InMemoryIdentityStore() if get_settings().inmemory else PostgresIdentityStore()


@lru_cache(maxsize=1)
def auth_token_store() -> AuthTokenStore:
    """One-time activation / password-reset tokens (in-memory under CI; Postgres otherwise)."""
    return InMemoryAuthTokenStore() if get_settings().inmemory else PostgresAuthTokenStore()


@lru_cache(maxsize=1)
def kb_store() -> KnowledgeBaseStore:
    """Registry of a tenant's knowledge bases (in-memory under CI; Postgres otherwise)."""
    return InMemoryKnowledgeBaseStore() if get_settings().inmemory else PostgresKnowledgeBaseStore()


@lru_cache(maxsize=1)
def model_connection_store() -> ModelConnectionStore:
    """Per-tenant model connection store (in-memory under CI; Postgres otherwise)."""
    return InMemoryModelConnectionStore() if get_settings().inmemory else PostgresModelConnection()


@lru_cache(maxsize=1)
def sql_connection_store() -> SqlConnectionStore:
    """Per-KB SQL connection registry, one or many per KB (in-memory under CI; Postgres otherwise)."""
    return InMemorySqlConnectionStore() if get_settings().inmemory else PostgresSqlConnectionStore()


@lru_cache(maxsize=1)
def conversation_store() -> ConversationStore:
    """Per-tenant conversation/message store (in-memory under CI; Postgres otherwise)."""
    return InMemoryConversationStore() if get_settings().inmemory else PostgresConversationStore()


@lru_cache(maxsize=1)
def api_key_store() -> ApiKeyStore:
    """Global service-account API-key store (in-memory under CI; Postgres otherwise)."""
    return InMemoryApiKeyStore() if get_settings().inmemory else PostgresApiKeyStore()


@lru_cache(maxsize=1)
def eval_schedule_store() -> EvalScheduleStore:
    """Global continuous-evaluation enrolment registry (FR-G9; in-memory under CI)."""
    return InMemoryEvalScheduleStore() if get_settings().inmemory else PostgresEvalScheduleStore()


@lru_cache(maxsize=1)
def grant_store() -> GrantStore:
    """Global Federated RAG federation-grant registry (cross-tenant; in-memory under CI)."""
    return InMemoryGrantStore() if get_settings().inmemory else PostgresGrantStore()


@lru_cache(maxsize=1)
def org_unit_store() -> OrgUnitStore:
    """Global per-tenant org-unit tree registry (RBAC scope; in-memory under CI)."""
    return InMemoryOrgUnitStore() if get_settings().inmemory else PostgresOrgUnitStore()


@lru_cache(maxsize=1)
def tenant_registry() -> TenantRegistry:
    """Global managed-cloud tenant registry (provider control plane; in-memory under CI). Imported
    locally: tenant_control imports wiring, so a top-level import here would be circular."""
    from svc_control_plane.tenant_control import (  # noqa: PLC0415 — avoids an import cycle
        InMemoryTenantRegistry,
        PostgresTenantRegistry,
    )

    return InMemoryTenantRegistry() if get_settings().inmemory else PostgresTenantRegistry()


def federation_registry() -> FederationRegistry:
    """Resolve the federation transport per peer from settings (ADR-0017). Not cached: it
    reads live settings so a config change (or a test) takes effect without a reset."""
    from svc_control_plane.federation import FederationRegistry  # noqa: PLC0415 — avoids import cycle

    return FederationRegistry.from_settings()


@lru_cache(maxsize=1)
def feature_flag_store() -> FlagStore:
    """Global runtime feature-flag override store (admin toggles; in-memory under CI)."""
    return InMemoryFlagStore() if get_settings().inmemory else PostgresFlagStore()


@lru_cache(maxsize=1)
def egress_request_store() -> EgressRequestStore:
    """Global egress-host approval-request store (in-memory under CI; Postgres otherwise)."""
    return InMemoryEgressRequestStore() if get_settings().inmemory else PostgresEgressRequestStore()


@lru_cache(maxsize=1)
def ingest_job_store() -> IngestJobStore:
    """Global async bulk-ingestion job store (in-memory under CI; Postgres otherwise)."""
    return InMemoryIngestJobStore() if get_settings().inmemory else PostgresIngestJobStore()


@lru_cache(maxsize=1)
def approval_store() -> dual_control.ApprovalStore:
    """Global maker-checker approval-request store (in-memory under CI; Postgres otherwise)."""
    return dual_control.InMemoryApprovalStore() if get_settings().inmemory else dual_control.PostgresApprovalStore()


# Per-(tenant, model) gateway cache -> (config signature, gateway). Rebuilt when a
# tenant's connection changes (signature covers endpoint/model/key/header).
_tenant_gateways: dict[tuple[str, str], tuple[str, LLMGateway]] = {}

# Per-tenant embedder / retrieval / indexing caches -> (config signature, object).
# Same lifecycle as _tenant_gateways: a model-connection mutation invalidates all four.
_tenant_embedders: dict[str, tuple[str, Embedder]] = {}
_tenant_retrieval: dict[str, tuple[str, RetrievalService]] = {}
_tenant_indexing: dict[str, tuple[str, IndexingService]] = {}


def invalidate_tenant_gateway(tenant_id: str) -> None:
    """Drop a tenant's cached model objects (gateways, embedder, retrieval/indexing
    services) so the next call picks up a changed connection."""
    for key in [k for k in _tenant_gateways if k[0] == tenant_id]:
        _tenant_gateways.pop(key, None)
    _tenant_embedders.pop(tenant_id, None)
    _tenant_retrieval.pop(tenant_id, None)
    _tenant_indexing.pop(tenant_id, None)


async def _resolve_org_default_connection(kb_id: str, role: str) -> StoredModelConnection | None:
    """The ORG-WIDE DEFAULT connection of ``role`` for the org that owns ``kb_id``, if any.

    Generation models are org-wide only: EVERY KB in the org uses the org-wide default of that role.
    The connection lives in the ORG's ``org_model_connections`` registry (its key encrypted with the
    ORG's DEK). We derive the org id from the ``{org}__{slug}`` KB id convention and return the org
    connection (owning tenant_id = org) whose role matches and ``is_default`` is set, or None. Kept
    here (rather than importing api.py) to avoid an import cycle."""
    import json  # noqa: PLC0415

    org_id = kb_id.split("__", 1)[0]
    org_raw = (await get_catalog().get_tenant_config(org_id)).get("org_model_connections")
    if not org_raw:
        return None
    try:
        rows = json.loads(org_raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(rows, list):
        return None
    for r in rows:
        if isinstance(r, dict) and r.get("role") == role and r.get("is_default"):
            try:
                return StoredModelConnection(**{**r, "tenant_id": org_id})
            except (ValueError, TypeError):
                return None
    return None


async def _resolve_org_connection_by_model(kb_id: str, role: str, model: str) -> StoredModelConnection | None:
    """The org-wide connection of ``role`` whose model id equals ``model`` — a per-question pick of a
    NON-default org model. Unlike ``_resolve_org_default_connection`` (default only), this lets the Ask
    picker switch to a DIFFERENT connection's FULL endpoint/creds, not just swap the model name on the
    default endpoint (which sent e.g. a local-GPU model id to the hosted API → 400)."""
    import json  # noqa: PLC0415

    org_id = kb_id.split("__", 1)[0]
    org_raw = (await get_catalog().get_tenant_config(org_id)).get("org_model_connections")
    if not org_raw:
        return None
    try:
        rows = json.loads(org_raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(rows, list):
        return None
    for r in rows:
        if isinstance(r, dict) and r.get("role") == role and r.get("model") == model:
            try:
                return StoredModelConnection(**{**r, "tenant_id": org_id})
            except (ValueError, TypeError):
                return None
    return None


async def _resolve_org_connection_by_id(kb_id: str, conn_id: str) -> StoredModelConnection | None:
    """One org-wide connection by its id — how a KB's ``embedder_active`` record, which may point
    into the ORG registry rather than the KB's own store, is read back."""
    import json  # noqa: PLC0415

    org_id = kb_id.split("__", 1)[0]
    org_raw = (await get_catalog().get_tenant_config(org_id)).get("org_model_connections")
    if not org_raw:
        return None
    try:
        rows = json.loads(org_raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(rows, list):
        return None
    for r in rows:
        if isinstance(r, dict) and r.get("conn_id") == conn_id:
            try:
                return StoredModelConnection(**{**r, "tenant_id": org_id})
            except (ValueError, TypeError):
                return None
    return None


async def resolve_tenant_vision_gateway(tenant_id: str) -> VisionLLMGateway:
    """The VISION gateway for a KB, preferring the org's own model connection.

    The vision path used to read RAG_VISION_LLM_* only, so a deployment that configured its
    model in the CONSOLE (FR-E1) had a working text model and a 401 on every image: the console
    connection carries the credential, the env variable was never filled, and nothing tied the
    two together. Observed live — Gemma-31B answered questions while every JPG ingest failed
    "401 Unauthorized" against the same endpoint.

    An org-wide connection of role VISION is used outright: registering a model for a role is an
    instruction to use it for that role, and describing an image depends on nothing stored, so it
    takes effect immediately.

    Failing that, the org's GENERATION connection is used only when it is the SAME endpoint the
    vision config names — that is the 401 above, not a model choice — so a deployment that
    deliberately points vision at a separate VLM keeps doing so. In that borrowed case the vision
    MODEL still comes from config when set: an org's chat deployment and its vision deployment are
    not always the same id."""
    from rag_clients.factory import get_vision_llm_gateway  # noqa: PLC0415 — cycle at module load
    from rag_clients.vision import OpenAICompatVisionGateway  # noqa: PLC0415
    from svc_control_plane import tenant_crypto  # noqa: PLC0415 — avoid import cycle at module load

    settings = get_settings()
    own = await _resolve_org_default_connection(tenant_id, "vision")
    if own is not None:
        key = await tenant_crypto.decrypt(own.tenant_id, own.key_ciphertext) if own.key_ciphertext else ""
        _log.info("vision_using_org_vision_model", tenant_id=tenant_id, endpoint=own.endpoint, model=own.model)
        return OpenAICompatVisionGateway(url=own.endpoint, model=own.model, api_key=key, api_key_header=own.header)

    if settings.vision_llm_backend != "openai_compat":
        return get_vision_llm_gateway()
    conn = await _resolve_org_default_connection(tenant_id, "generation")
    if conn is None or conn.endpoint.rstrip("/") != settings.vision_llm_url.rstrip("/"):
        return get_vision_llm_gateway()

    key = await tenant_crypto.decrypt(conn.tenant_id, conn.key_ciphertext) if conn.key_ciphertext else ""
    _log.info("vision_using_org_connection", tenant_id=tenant_id, endpoint=conn.endpoint)
    return OpenAICompatVisionGateway(
        url=conn.endpoint,
        model=settings.vision_llm_model or conn.model,
        api_key=key,
        api_key_header=conn.header,
    )


async def resolve_tenant_gateway(tenant_id: str, model: str | None = None) -> LLMGateway:
    """The LLM gateway for a KB. Generation models are ORG-WIDE only: every KB uses the org-wide
    DEFAULT generation connection if one is set, else the platform default. ``model`` (a per-question
    pick) selects the org-wide connection whose model id matches; with no match it overrides the
    deployment on the org-wide default. Embedding/reranking connections are never eligible here — they
    are not chat models.

    Any residual per-KB generation connections (legacy, no longer creatable in the UI) are still
    honoured for backward compatibility, but the org-wide default takes precedence for the plain
    (no per-question model) path."""
    conns = await model_connection_store().list(tenant_id, "generation")
    chosen: StoredModelConnection | None = None
    if model:
        chosen = next((c for c in conns if c.model == model), None)

    # Org-wide default generation model — the primary source now. Consulted for the plain path
    # (no per-question model) and when a per-question model matches the org-wide connection.
    if chosen is None:
        org_conn = await _resolve_org_default_connection(tenant_id, "generation")
        if org_conn is not None and (model is None or org_conn.model == model):
            chosen = org_conn

    # Per-question pick of a NON-default org connection: match by model id across ALL org connections
    # and use its FULL endpoint/creds — so the Ask picker actually switches endpoint (e.g. to a local
    # GPU), not just swaps the model name onto the default/platform endpoint (which caused the 400).
    if chosen is None and model:
        chosen = await _resolve_org_connection_by_model(tenant_id, "generation", model)

    # Legacy per-KB generation connections (no longer creatable): honour a matching default only if
    # no org-wide default resolved above.
    if chosen is None:
        chosen = next((c for c in conns if c.is_default), None) or (conns[0] if conns else None)

    if chosen is None:  # no KB connection → platform default (+ optional deployment override)
        if model and get_settings().llm_backend in {"openai_compat", "vllm", "llamacpp"}:
            cache_key = (tenant_id, f"default|{model}")
            cached = _tenant_gateways.get(cache_key)
            if cached and cached[0] == f"default|{model}":
                return cached[1]
            gw: LLMGateway = OpenAICompatGateway(model=model)
            _tenant_gateways[cache_key] = (f"default|{model}", gw)
            return gw
        return get_llm_gateway()

    from svc_control_plane import tenant_crypto  # noqa: PLC0415 — avoid import cycle at module load

    effective_model = model or chosen.model
    # An org-wide connection's key is encrypted with the ORG's DEK, not the KB's — decrypt with the
    # connection's OWNING tenant (chosen.tenant_id: the org for an adopted org model, else this KB).
    key = await tenant_crypto.decrypt(chosen.tenant_id, chosen.key_ciphertext) if chosen.key_ciphertext else ""
    sig = f"{chosen.endpoint}|{effective_model}|{chosen.header}|{len(key)}:{key[-4:]}"
    cache_key = (tenant_id, f"{chosen.endpoint}|{effective_model}")
    cached = _tenant_gateways.get(cache_key)
    if cached and cached[0] == sig:
        return cached[1]
    # The self-hosted Sarvam-30B (served as "/model" on the local vLLM/gpu1) is a reasoning model that
    # ignores reasoning_effort and always emits a long <think> block — slow, and empty when the budget
    # runs out. Its chain-of-thought is suppressed at the SERVER (gpu1's chat template primes a closed
    # <think></think>), so no prefill is sent; this flag just marks it so the gateway nudges sampling
    # temperature (a no-reasoning greedy decode loops on tables). Scoped by model id so qwen and the
    # hosted Sarvam API are unaffected.
    skip_reasoning = effective_model == "/model"
    # Badge/citation display: show the connection's friendly name (e.g. "Sarvam-30B (local GPU)")
    # instead of the raw deployment id (e.g. "/model"). Display-only; API calls use effective_model.
    gw = OpenAICompatGateway(
        url=chosen.endpoint,
        model=effective_model,
        api_key=key,
        api_key_header=chosen.header,
        skip_reasoning=skip_reasoning,
        # The friendly name belongs to the CONNECTION, so it is only the right label while the
        # connection's own deployment is what runs. A per-question pick overrides the deployment
        # on that connection's endpoint, and showing the connection's name then attributes the
        # answer to a model that did not produce it — on an audited platform that is a wrong
        # record, not a cosmetic one.
        display_name=(
            (getattr(chosen, "name", None) or effective_model) if effective_model == chosen.model else effective_model
        ),
    )
    _tenant_gateways[cache_key] = (sig, gw)
    return gw


def _build_embedder_from_conn(conn: StoredModelConnection, key: str, dim: int) -> Embedder:
    """The embedding adapter a connection names. A ``bge_m3_multi`` attach (the bge-m3 /embed
    endpoint, Bearer auth) gets the multivector adapter — so a KB or org that ATTACHED it activates
    the Slice-2b dense+sparse+ColBERT legs through the SAME resolver the router and indexer use,
    not only via the global env backend. Every other connection stays ``openai_compat`` (raw apikey
    header), byte-identical to before."""
    if conn.backend == "bge_m3_multi":
        from rag_clients.bge_m3_embedder import BgeM3MultiEmbedder  # noqa: PLC0415

        return BgeM3MultiEmbedder(
            conn.endpoint, conn.model, dim=dim, api_key=key, api_key_header=conn.header, auth="bearer"
        )
    return OpenAICompatEmbedder(conn.endpoint, conn.model, dim=dim, api_key=key, api_key_header=conn.header)


async def _org_embedder_or_platform(tenant_id: str) -> Embedder:
    """The ORG-WIDE embedding connection when it is safe to use, else the platform embedder.

    An org-wide model added in the console should be what the platform uses — that is what the
    registry is for, and it is already true of generation. An embedder is the one role where that
    cannot be unconditional: it owns the vector space, so adopting a DIFFERENT one for a KB whose
    vectors were built elsewhere would not switch the embedder, it would invalidate every vector
    in it and silently ruin retrieval.

    So the org connection is adopted here only when it is COMPATIBLE with what already embedded
    this KB — same model and same dimension as the platform embedder that built it, which is the
    same vector space, so only the endpoint and credential change.

    A DIFFERENT org embedder is not adopted here; it is STAGED when it is registered
    (``embedder_switch.on_org_embedding_change``) and takes effect when the operator applies the
    re-embed. Reaching the warning below therefore means an apply is still outstanding — the KB
    keeps answering with the embedder that built its vectors until then, which is the point."""
    from svc_control_plane import tenant_crypto  # noqa: PLC0415 — avoid import cycle at module load

    conn = await _resolve_org_default_connection(tenant_id, "embedding")
    if conn is None or conn.embed_dim is None:
        return get_embedder()
    settings = get_settings()
    if (conn.model, conn.embed_dim) != (settings.embed_model, settings.embed_dim):
        _log.warning(
            "org_embedder_not_adopted",
            tenant_id=tenant_id,
            org_model=conn.model,
            org_dim=conn.embed_dim,
            corpus_model=settings.embed_model,
            corpus_dim=settings.embed_dim,
            detail="an org-wide embedder differs from the one this KB's vectors were built with; "
            "it stays staged until the re-embed is applied (POST /v1/org/embedder/apply)",
        )
        return get_embedder()
    key = await tenant_crypto.decrypt(conn.tenant_id, conn.key_ciphertext) if conn.key_ciphertext else ""
    sig = f"org|{conn.backend}|{conn.endpoint}|{conn.model}|{conn.embed_dim}|{conn.header}|{len(key)}:{key[-4:]}"
    cached = _tenant_embedders.get(tenant_id)
    if cached and cached[0] == sig:
        return cached[1]
    embedder = _build_embedder_from_conn(conn, key, conn.embed_dim)
    _tenant_embedders[tenant_id] = (sig, embedder)
    return embedder


async def resolve_tenant_embedder(tenant_id: str) -> Embedder:
    """The embedder for a KB: its ACTIVE per-KB embedding attach, or the platform default.

    Reads the ``embedder_active`` tenant-config record — NOT the connection store. That
    asymmetry is the safety mechanism: saving an embedding connection does not change
    what embeds queries/ingests until the operator applies the re-embed (the corpus and
    the query must stay in ONE vector space). See ``embedder_switch``."""
    import json  # noqa: PLC0415

    from svc_control_plane import tenant_crypto  # noqa: PLC0415 — avoid import cycle at module load

    cfg = await get_catalog().get_tenant_config(tenant_id)
    raw = cfg.get("embedder_active", "")
    if not raw:
        return await _org_embedder_or_platform(tenant_id)
    active = json.loads(raw)
    conn_id = str(active.get("conn_id", ""))
    # The active record may name an ORG-WIDE embedder (one entry serving every KB in the org) or
    # this KB's own attach. A conn_id alone cannot say which store holds it, so the record says.
    conn = (
        await _resolve_org_connection_by_id(tenant_id, conn_id)
        if active.get("scope") == "org"
        else await model_connection_store().get(tenant_id, conn_id)
    )
    if conn is None or conn.role != "embedding" or conn.embed_dim is None:
        # The record points at nothing usable (deletion is guarded, but fail safe):
        # the platform embedder indexed nothing here, yet it is the only honest fallback.
        _log.warning("embedder_active_dangling", tenant_id=tenant_id)
        return get_embedder()
    # An org-wide connection's key is encrypted with the ORG's DEK, a KB's with its own.
    key = await tenant_crypto.decrypt(conn.tenant_id, conn.key_ciphertext) if conn.key_ciphertext else ""
    sig = f"{conn.backend}|{conn.endpoint}|{conn.model}|{conn.embed_dim}|{conn.header}|{len(key)}:{key[-4:]}"
    cached = _tenant_embedders.get(tenant_id)
    if cached and cached[0] == sig:
        return cached[1]
    embedder = _build_embedder_from_conn(conn, key, conn.embed_dim)
    _tenant_embedders[tenant_id] = (sig, embedder)
    return embedder


async def resolve_tenant_reranker(tenant_id: str) -> Reranker:
    """The reranker for a KB: its default reranking attach, or the platform default.

    Unlike the embedder this reads the connection store DIRECTLY — a reranker is
    stateless with respect to the index (no stored vectors depend on it), so it takes
    effect immediately on attach, and an ORG-WIDE reranking connection can be adopted
    the moment it is added (no vector space depends on it, so there is nothing to
    invalidate — contrast ``_org_embedder_or_platform``). The KB's own attach wins over
    the org default. Score-scale caveat: recalibrate the KB's abstain threshold after
    attaching (per-KB overrides exist)."""
    from svc_control_plane import tenant_crypto  # noqa: PLC0415 — avoid import cycle at module load

    conn = await model_connection_store().get_default(tenant_id, "reranking")
    if conn is None:
        conn = await _resolve_org_default_connection(tenant_id, "reranking")
    if conn is None or not conn.api_shape:
        return get_reranker()
    key = await tenant_crypto.decrypt(conn.tenant_id, conn.key_ciphertext) if conn.key_ciphertext else ""
    return OpenAICompatReranker(
        conn.endpoint,
        conn.model,
        api_shape=conn.api_shape,
        api_key=key,
        api_key_header=conn.header,
    )


async def retrieval_service_for(tenant_id: str) -> RetrievalService:
    """The retrieval service a KB's queries go through: the shared platform singleton
    unless the KB has its own embedder/reranker attached, in which case a per-tenant
    HybridRetriever over the SAME shared stores (cached per tenant + config)."""
    embedder = await resolve_tenant_embedder(tenant_id)
    reranker = await resolve_tenant_reranker(tenant_id)
    if embedder is get_embedder() and reranker is get_reranker():
        return retrieval_service()  # identity-preserving default path
    sig = f"{id(type(embedder))}|{getattr(embedder, '_url', '')}|{getattr(reranker, '_url', '')}"
    cached = _tenant_retrieval.get(tenant_id)
    if cached and cached[0] == sig:
        return cached[1]
    service = RetrievalService(build_retriever(embedder, reranker))
    _tenant_retrieval[tenant_id] = (sig, service)
    return service


async def indexing_service_for(tenant_id: str) -> IndexingService:
    """The indexing service a KB's ingests go through — the platform singleton unless the
    KB has an ACTIVE per-KB embedder (then its collection is built in that vector space)."""
    embedder = await resolve_tenant_embedder(tenant_id)
    if embedder is get_embedder():
        return indexing_service()
    sig = f"{getattr(embedder, '_url', '')}|{embedder.dim}"
    cached = _tenant_indexing.get(tenant_id)
    if cached and cached[0] == sig:
        return cached[1]
    service = IndexingService(embedder, get_vector_store(), get_sparse_index(), get_catalog(), get_contextualizer())
    _tenant_indexing[tenant_id] = (sig, service)
    return service


async def resolve_graph_extractor(tenant_id: str) -> GraphExtractor:
    """The GraphRAG entity/relationship extractor: the KB's own model when the backend is
    'llm', else the deterministic heuristic (FR-C6)."""
    settings = get_settings()
    if settings.graph_extract_backend == "llm":
        # A generous token budget: reasoning models (e.g. Sarvam) spend the first several hundred
        # tokens THINKING before emitting the entity JSON — the old 512 default starved them,
        # yielding an empty/sparse graph (worst on non-English text the heuristic can't parse).
        return LLMGraphExtractor(await resolve_tenant_gateway(tenant_id), max_tokens=settings.graph_extract_max_tokens)
    if settings.graph_extract_backend == "translate":
        # Any-Indic path: translate a non-English chunk to English, extract where NER is reliable
        # (the heuristic), then map the entities back to the detected source language. English
        # chunks are extracted directly. The translator is chosen by translate_backend — the
        # dedicated Sarvam MT model when set (the chat model is unreliable at translation).
        from rag_clients.graph_translate import TranslateThenExtract  # noqa: PLC0415

        return TranslateThenExtract(await _build_translator(tenant_id), HeuristicGraphExtractor())
    return HeuristicGraphExtractor()


async def resolve_domain_relation_extractor(tenant_id: str, vocab: tuple[str, ...]) -> GraphExtractor:
    """The typed domain-relation extractor (Slice 6, sub-slice B): the KB's own model constrained
    to ``vocab`` when the backend is 'llm', else the deterministic verb-pattern heuristic. Mirrors
    :func:`resolve_graph_extractor`; both implement ``extract(text) -> GraphTriple``."""
    settings = get_settings()
    if settings.domain_relation_backend == "llm":
        return LLMDomainRelationExtractor(
            await resolve_tenant_gateway(tenant_id), vocab, max_tokens=settings.domain_relation_max_tokens
        )
    return HeuristicDomainRelationExtractor(vocab)


async def _build_translator(tenant_id: str) -> Translator:
    """The translator behind translate-then-extract and Indic helpers. ``translate_backend=sarvam``
    uses Sarvam's dedicated MT endpoint (deterministic, reliable on Indic); ``llm`` reuses the KB
    chat model. One place so every translation path picks the same backend."""
    settings = get_settings()
    if settings.translate_backend == "sarvam":
        from rag_clients.sarvam_translate import SarvamTranslator  # noqa: PLC0415

        return SarvamTranslator()
    return LLMTranslator(await resolve_tenant_gateway(tenant_id), max_tokens=settings.graph_translate_max_tokens)


async def resolve_community_summarizer(tenant_id: str) -> CommunitySummarizer:
    """The GraphRAG community summariser (slice 2): the KB's model writes prose themes when
    the backend is 'llm', else the entity-list heuristic (CPU-free)."""
    if get_settings().graph_summary_backend == "llm":
        return LLMCommunitySummarizer(await resolve_tenant_gateway(tenant_id))
    return HeuristicCommunitySummarizer()


async def resolve_translator(tenant_id: str) -> Translator:
    """Indic translation: the dedicated Sarvam MT model or the KB chat model (see
    ``translate_backend``) when translation is enabled, else a no-op."""
    if get_settings().translation_backend == "llm":
        return await _build_translator(tenant_id)
    return NoopTranslator()


async def resolve_followup_generator(tenant_id: str) -> FollowupGenerator:
    """Suggested follow-ups (FR-I8): the KB's model over the retrieved context when the
    backend is 'llm', else the deterministic entity-anchored heuristic."""
    if get_settings().followups_backend == "llm":
        return LLMFollowupGenerator(
            await resolve_tenant_gateway(tenant_id), max_tokens=get_settings().followups_max_tokens
        )
    return HeuristicFollowupGenerator()


async def resolve_clarify_generator(tenant_id: str) -> ClarifyGenerator:
    """Grounded query disambiguation (Layer-2 item 4): the KB's model over the retrieved spans
    when the backend is 'llm', else the deterministic doc-cluster heuristic. Mirrors follow-ups."""
    if get_settings().clarify_backend == "llm":
        return LLMClarifyGenerator(await resolve_tenant_gateway(tenant_id))
    return HeuristicClarifyGenerator()


async def resolve_tenant_intent_classifier(tenant_id: str) -> IntentClassifier:
    """Semantic intent classifier for a KB: the KB's own model when the backend is 'llm'
    (the real semantic path), else the deterministic keyword heuristic."""
    if get_settings().intent_backend == "llm":
        return LLMIntentClassifier(await resolve_tenant_gateway(tenant_id))
    return HeuristicIntentClassifier()


async def resolve_tenant_query_planner(tenant_id: str) -> QueryPlanner:
    """Agentic query planner for a KB (mirrors :func:`resolve_tenant_intent_classifier`).

    Returns the real semantic :class:`LLMQueryPlanner` — the KB's own sovereign gateway reads the
    complete query + light KB context and returns a structured plan — ONLY when the master
    ``query_planner_enabled`` flag is on AND that gateway resolves. Otherwise, and on ANY resolution
    error, returns the deterministic :class:`HeuristicQueryPlanner`, so the pipeline is fail-safe to
    today's heuristic routing (a planner failure never 500s a query)."""
    if not get_settings().query_planner_enabled:
        return HeuristicQueryPlanner()
    try:
        return LLMQueryPlanner(await resolve_tenant_gateway(tenant_id))
    except Exception as exc:  # fail-safe: a gateway-resolution error falls back to the heuristic
        _log.warning("query_planner_resolve_failed_fallback_heuristic", tenant_id=tenant_id, error=str(exc)[:160])
        return HeuristicQueryPlanner()


async def resolve_tenant_faithfulness(tenant_id: str) -> FaithfulnessChecker | None:
    """Judge faithfulness with the KB's own model when one is connected (FR-E1), so a real
    generative answer is verified semantically by an LLM instead of the lexical fallback.
    Returns None to use the platform-default checker (lexical on the CPU dev tier)."""
    if not get_settings().faithfulness_tenant_model_judge:
        return None
    if await model_connection_store().get_default(tenant_id) is None:
        return None
    # Use the tenant's LLM directly — NOT build_faithfulness_checker, which honours the
    # platform's (lexical) backend and would ignore the gateway.
    return LLMFaithfulnessChecker(await resolve_tenant_gateway(tenant_id))


async def _notation_enabled_for(kb_id: str) -> bool:
    """The effective notation-grounding toggle for this KB (Slice 4): the per-KB override in
    tenant_config (`notation_grounding_enabled`, stored 1.0/0.0 in the `_THRESHOLD_KEYS` seam)
    when set, else the platform `notation_grounding_enabled` default. So a pharma KB can turn it
    on hard while an HR KB leaves it off — no static-global-only behaviour."""
    # Lazy import: `pipeline` imports `wiring` at module load, so importing `kb_threshold_overrides`
    # here (at call time, both modules already loaded) reuses the ONE per-KB override seam without a
    # circular import.
    from svc_control_plane.pipeline import kb_threshold_overrides  # noqa: PLC0415

    override = (await kb_threshold_overrides(kb_id)).get("notation_grounding_enabled")
    if override is not None:
        # Stored as a float in the threshold-override seam (1.0 = on, 0.0 = off); treat any
        # non-zero as enabled.
        return override != 0.0
    return get_settings().notation_grounding_enabled


async def apply_notation_grounding(checker: FaithfulnessChecker, kb_id: str) -> FaithfulnessChecker:
    """Toggle the notation twin on the faithfulness checker for this KB (Slice 4). The composition
    seam for per-KB adaptivity: only a :class:`NumericallyGroundedChecker` carries the notation
    check, so a bare tenant-model judge (which likewise skips the numeric check) is returned
    unchanged — notation coverage equals numeric coverage. When the flag resolves False the wrapper
    is returned untouched, so behaviour is byte-identical to today.

    The KB's routing gates are bound onto the wrapper too, so a notation hit caps its own score
    into THIS KB's FLAGGED band (from the SAME `kb_threshold_overrides` seam the generation service
    and stream path route by, incl. the Guardrail-policy overlay), routing to FLAGGED_FOR_REVIEW."""
    if not isinstance(checker, NumericallyGroundedChecker):
        return checker
    from svc_control_plane.pipeline import kb_threshold_overrides  # noqa: PLC0415

    overrides = await kb_threshold_overrides(kb_id)
    settings = get_settings()
    return checker.with_notation(
        await _notation_enabled_for(kb_id),
        faithful_threshold=overrides.get("faithful_threshold", settings.faithful_threshold),
        flag_threshold=overrides.get("flag_threshold", settings.flag_threshold),
    )


@lru_cache(maxsize=1)
def generation_service() -> GenerationService:
    return GenerationService(
        get_llm_gateway(),
        audit_service(),
        get_faithfulness_checker(),
        output_guard=get_output_guard(),
        gateway_resolver=resolve_tenant_gateway,
        faithfulness_resolver=resolve_tenant_faithfulness,
        notation_resolver=apply_notation_grounding,
    )


@lru_cache(maxsize=1)
def input_guard() -> Guardrail:
    return get_input_guard()


@lru_cache(maxsize=1)
def hitl_store() -> HitlStore:
    """In-memory HITL store under the in-memory profile; Postgres otherwise."""
    return InMemoryHitlStore() if get_settings().inmemory else PostgresHitlStore()


@lru_cache(maxsize=1)
def hitl_service() -> HitlService:
    return HitlService(audit_service(), hitl_store())


async def provision_kb(tenant_id: str) -> None:
    """Ensure all per-tenant namespaces exist (vector collection, BM25 index, PG schema).
    Uses the tenant-aware indexing service so a KB with an active per-KB embedder gets
    its vector collection created at that embedder's dimension from day one."""
    await (await indexing_service_for(tenant_id)).ensure_namespaces(tenant_id)
    await get_catalog().get_tenant_config(tenant_id)  # ensures the tenant schema exists
