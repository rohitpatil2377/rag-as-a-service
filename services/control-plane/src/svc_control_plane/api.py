"""Control-plane REST API (`/v1`) — TDD §7.

The public, tenant-scoped, RBAC/ABAC-enforced surface. ``kb_id`` in the path IS the
tenant namespace in Phase-0; the :func:`tenant_guard` dependency asserts the
authenticated principal owns that namespace (ABAC), binds the tenant into the logging
context, and fails closed on mismatch — API-layer tenant isolation.

Handlers stay thin over :mod:`svc_control_plane.pipeline` so an MCP server (FR-H2) can
wrap the same capabilities later without a refactor.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import re
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal, cast
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, File, Form, Path, Query, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from rag_clients.external_creds import external_credentials
from rag_clients.factory import (
    external_fetcher_for,
    get_catalog,
    get_embedder,
    get_graph_store,
    get_llm_gateway,
    get_reranker,
    get_vector_store,
)
from rag_clients.llm_gateway import OpenAICompatGateway
from rag_common import (
    ApprovalRequiredError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    PlatformError,
    Principal,
    Settings,
    TenantIsolationError,
    UnauthorizedError,
    ValidationError,
    assert_sovereign,
    get_logger,
    get_principal,
    get_secrets_provider,
    get_settings,
    hash_password,
    issue_dev_jwt,
    register_api_key_resolver,
    verify_dummy,
    verify_password,
)
from rag_common.airgap import airgap_preflight
from rag_common.email import send_email
from rag_common.model_sbom import ModelSBOM
from rag_common.rbac import Clearance, can_admin_unit, can_grant_clearance, can_ingest_at, can_read_kb
from rag_common.tenancy import assert_registrable, reset_tenant, set_tenant
from rag_common.trust_receipt import TrustReceipt
from rag_core.access import is_cleared, is_entitled
from rag_core.parent_child import not_parent
from rag_core.workflow import WorkflowDef, WorkflowResult
from rag_schemas import (
    AuditEvent,
    DocumentEnvelope,
    ExternalObjectSource,
    Grant,
    GroundedAnswer,
    RetrievalMode,
    RetrievalQuery,
    RetrievedSpan,
)
from rag_schemas.classification import ClassificationLevel, classification_label, parse_classification
from rag_schemas.common import AuditAction, ChunkStrategy, SourceKind
from rag_schemas.retrieval import reject_reserved_filter_keys
from svc_control_plane import (
    agent_orchestrator,
    agent_types,
    architecture,
    billing,
    bots,
    budgets,
    business_value,
    compliance,
    continuous_eval,
    crawl_jobs,
    domain_profile,
    dual_control,
    egress_requests,
    embedder_switch,
    feature_flags,
    federated_eval,
    golden_gen,
    graph_sync,
    guardrails_policy,
    ingest_dag,
    ingest_jobs,
    kb_roles,
    multimodal,
    nosql_query,
    pipeline,
    process_flow,
    provenance,
    reindex,
    responsible_ai,
    rest_source,
    sharepoint_source,
    sql_query,
    sync_jobs,
    tenant_control,
    tenant_crypto,
    widget,
    wiring,
    workflows,
)
from svc_control_plane.analytics import AnalyticsGaps, analytics_gaps
from svc_control_plane.api_keys import SCOPE_ROLES, ApiKeyRecord, mint_key
from svc_control_plane.eval_schedule import EvalSchedule
from svc_control_plane.hitl import HitlAction, HitlReview, HitlStatus
from svc_control_plane.identity import CAP_ROLES, ROLE_CAPS, Org, User, effective_roles
from svc_control_plane.knowledge_base import SHARED_KB_ID, SHARED_KB_NAME, slug_kb_id
from svc_control_plane.mcp import mcp_manifest
from svc_control_plane.model_connection import StoredModelConnection
from svc_control_plane.org_units import OrgUnit, parents_map
from svc_control_plane.preflight import BackendHealth
from svc_control_plane.preflight import health as probe_backend_health
from svc_control_plane.quality import EvalQuestion, EvalReport, QualityReport, quality_report, run_eval
from svc_control_plane.quota import check_document_cap, check_ingest_rate, check_query_rate
from svc_control_plane.rbac_context import is_platform_admin, subject_of
from svc_control_plane.usage import UsageReport, usage_report

_log = get_logger(__name__)
router = APIRouter(prefix="/v1", tags=["control-plane"])


async def _resolve_api_key(token: str) -> Principal | None:
    """Resolve a service-account key (FR-I4) to a Principal — org + scope-derived roles."""
    rec = await wiring.api_key_store().resolve(token)
    if rec is None:
        return None
    return Principal(
        subject=f"apikey:{rec.name}",
        tenant_id=rec.org_id,
        roles=frozenset(SCOPE_ROLES.get(rec.scope, [])),
        attrs={"api_key": rec.key_id},
    )


register_api_key_resolver(_resolve_api_key)


# --- Tenancy guard (ABAC + context binding) ---------------------------------
async def tenant_guard(
    kb_id: Annotated[str, Path(description="Knowledge base id (owned by the caller's tenant)")],
    principal: Annotated[Principal, Depends(get_principal)],
) -> AsyncIterator[Principal]:
    """Assert the principal's tenant OWNS ``kb_id`` (or ``kb_id`` is the one platform-wide
    SHARED_KB_ID) and bind the KB namespace for the request.

    A tenant (org) owns its default KB (``kb_id == tenant_id``) and any KB it created
    (``kb_id`` starts with ``{tenant_id}-``). Ownership is by prefix, so no lookup is
    needed and a KB can never be addressed across tenants — EXCEPT SHARED_KB_ID, which
    every tenant may read AND ingest into by design (Federated RAG Tier C): a document
    ends up there only when someone with ``ingestor`` role explicitly chooses that id
    over their own KB's at ingest time, never by inference from tenant identity."""
    _require_member(principal)  # FR-RBAC-01: a no-role token reaches no KB endpoint, read or write
    tid = principal.tenant_id
    # SIM109 is suppressed below ("use `kb_id in (...)`"): that collapse would silently drop the
    # startswith prefix check, which IS the tenant-ownership boundary — not equivalent.
    if not (kb_id == tid or kb_id.startswith(f"{tid}__") or kb_id == SHARED_KB_ID):  # noqa: SIM109
        raise TenantIsolationError(
            "principal may not access this knowledge base",
            details={"kb_id": kb_id, "principal_tenant": tid},
        )
    # Per-user KB access (RBAC): a restricted user (non-empty kb_access) may only reach the KBs
    # granted to them. Owner/admin carry an empty list = access to every KB in the org.
    allowed = principal.attrs.get("kb_access") or []
    if allowed and kb_id not in allowed and kb_id != SHARED_KB_ID:
        # A LIVE cross-KB access grant (approved access-request) lets a restricted caller in without
        # re-login — grant/revoke take effect immediately since this is read fresh, not off the JWT.
        grants = await kb_roles.load_access_grants(get_catalog(), kb_id)
        if principal.subject.strip().lower() not in grants:
            raise ForbiddenError(
                "your account is not permitted to access this knowledge base", details={"kb_id": kb_id}
            )
    await tenant_control.assert_active(principal.tenant_id)
    await _assert_kb_access(principal, kb_id)
    token = set_tenant(kb_id)
    try:
        yield principal
    finally:
        reset_tenant(token)


async def _assert_kb_access(principal: Principal, kb_id: str) -> None:
    """Hierarchical read gate at the chokepoint: a KB above the caller's CLEARANCE, or outside its
    unit SCOPE, is hidden — 404 (not 403) so a higher-classified KB's existence is not revealed. An
    unknown KB defers to the handler's own 404. Additive: with the defaults (no units, Public floor)
    every same-tenant KB stays visible, so nothing changes until governance is configured."""
    kb = await wiring.kb_store().get(principal.tenant_id, kb_id)
    if kb is None:
        return
    # A trashed (soft-deleted) KB must be inaccessible everywhere — not merely hidden from the
    # active list. Otherwise a saved deep link, MCP client, widget/bot, or SDK script still bound
    # to its id could query or ingest into a "deleted" KB. 404 (like the clearance gate) so its
    # continued existence isn't revealed. Restore/purge use get_principal (not this guard), so
    # they still work on a trashed KB.
    if kb.trashed_at is not None:
        raise NotFoundError("knowledge base not found", details={"kb_id": kb_id})
    subject = await kb_roles.elevated_subject(get_catalog(), kb_id, principal)
    if not can_read_kb(
        subject,
        kb_classification=Clearance(kb.classification_floor),
        kb_unit_id=kb.unit_id or principal.tenant_id,
        parents=await parents_map(wiring.org_unit_store(), principal.tenant_id),
    ):
        raise NotFoundError("knowledge base not found", details={"kb_id": kb_id})


def _principal_groups(principal: Principal) -> list[str]:
    """The caller's ACL group memberships for permission-aware retrieval (FR-A6)."""
    groups = principal.attrs.get("groups", [])
    return [str(g) for g in groups] if isinstance(groups, list) else []


def _principal_clearance(principal: Principal) -> int:
    """The caller's data-classification clearance 0=Public..3=Restricted (banking pack).
    Defaults to 0 (Public only) — least privilege — for any token without a clearance claim."""
    return parse_classification(principal.attrs.get("clearance"))


def _resolve_pii_policy(toggle: bool | None) -> str:
    """Resolve the effective PII-classification-floor policy for one ingest: ``off`` unless the
    platform master flag is on AND this ingest opted in (``toggle`` True, or None = default-on when
    the feature is enabled). Returns the configured mode (``raise``/``block``) or ``off``."""
    settings = get_settings()
    if not settings.pii_classification_enabled:
        return "off"
    enabled = True if toggle is None else toggle
    return settings.pii_classification_policy if enabled else "off"


async def _gate(
    action: dual_control.Action,
    *,
    tenant_id: str,
    kb_id: str | None,
    params: dict[str, Any],
    summary: str,
    principal: Principal,
) -> None:
    """Maker-checker gate (banking pack): if this action requires dual control, STAGE it as a
    pending approval and raise 202 ApprovalRequired instead of executing. No-op when disabled."""
    if not dual_control.is_gated(action):
        return
    req = await dual_control.submit(
        wiring.approval_store(),
        action=action,
        tenant_id=tenant_id,
        kb_id=kb_id,
        params=params,
        summary=summary,
        requested_by=principal.subject,
    )
    raise ApprovalRequiredError(
        f"{summary} — staged for a second admin's approval (maker-checker).",
        details={"request_id": req.id, "action": action, "status": "pending"},
    )


async def _require_published(kb_id: str, principal: Principal) -> None:
    """Block end-user querying of a KB that is still a draft (unpublished, banking pack). A KB
    is unpublished only under the maker-checker publish gate; normal KBs are published."""
    rec = await wiring.kb_store().get(principal.tenant_id, kb_id)
    if rec is not None and not rec.published:
        raise ForbiddenError(
            "this knowledge base is not published yet — it is pending publish approval",
            details={"kb_id": kb_id},
        )


def _require_role(principal: Principal, role: str) -> None:
    """RBAC gate — used so a scoped API key (read/ingest/admin, FR-I4) actually restricts.

    Admin-inclusive: an ``admin`` token satisfies any non-admin capability role (matching the
    admin-implies-broader-capability convention used by the owner/admin checks elsewhere in this
    file) — but a plain ``role="admin"`` gate still requires the ``admin`` role itself.
    """
    if principal.has_role(role):
        return
    # owner is the top of the built-in ladder (FRS Table 4: "owner — Everything, all four
    # capabilities, permanently"), so it satisfies EVERY capability gate, admin included.
    if principal.has_role("owner"):
        return
    if role != "admin" and principal.has_role("admin"):
        return
    raise ForbiddenError(f"this action requires the '{role}' role", details={"have": sorted(principal.roles)})


_CAPABILITY_ROLES = frozenset({"reader", "ingestor", "reviewer", "admin", "owner"})


def _require_member(principal: Principal) -> None:
    """Baseline gate: the caller must hold at least one built-in capability role to read or ask.

    FR-RBAC-01 — a token with no assigned role can do nothing but sign in; an empty (or
    unrecognised) role list must never be treated as access, not even read/ask on the shared KB.
    The write/admin gates (:func:`_require_role`) sit ON TOP of this for mutating actions.
    """
    if not (_CAPABILITY_ROLES & set(principal.roles)):
        raise ForbiddenError("this action requires an assigned role", details={"have": sorted(principal.roles)})


# --- Request/response models -------------------------------------------------
class CreateKBRequest(BaseModel):
    display_name: str = Field(..., min_length=1)


class IngestRequest(BaseModel):
    """Ingest a document either inline (``content_b64``) or by-reference from the
    tenant's object store (``object_key``, FR-A2). Exactly one, keyed by ``source_kind``."""

    doc_id: str
    mime: str = "application/pdf"
    content_b64: str | None = Field(default=None, description="Inline base64 bytes (source_kind=upload).")
    object_key: str | None = Field(
        default=None,
        description="Object key to pull by reference (source_kind=object_store, FR-A2). In the tenant's own "
        "namespace when external_source is unset (A2a), or a key in that external bucket when set (A2b).",
    )
    external_source: ExternalObjectSource | None = Field(
        default=None,
        description="External tenant-provided S3 bucket to pull object_key from (FR-A2b). Requires "
        "source_kind=object_store. Endpoint must be egress-allowlisted; credentials are a SecretsProvider ref.",
    )
    url: str | None = Field(
        default=None,
        description="Web page / file URL to fetch (source_kind=web, FR-A3). Host must be egress-allowlisted.",
    )
    source_kind: SourceKind = SourceKind.UPLOAD
    acl: list[str] = Field(default_factory=list)
    classification: str | int = Field(
        default=0, description="Data classification: 0/public, 1/internal, 2/confidential, 3/restricted (banking)."
    )
    pii_autoclassify: bool | None = Field(
        default=None,
        description="Enable the PII-driven classification floor for THIS ingest (raise/block a doc "
        "whose detected PII implies a higher classification). None = platform default when enabled.",
    )
    metadata: dict[str, str] = Field(default_factory=dict)
    # None -> the platform default (RAG_CHUNK_STRATEGY). Four entry points used to carry
    # four different literals, so how a document was chunked depended on which door it
    # came through.
    strategy: ChunkStrategy | None = None

    @model_validator(mode="after")
    def _check_payload(self) -> IngestRequest:
        if self.source_kind is SourceKind.UPLOAD:
            if not self.content_b64:
                raise ValueError("source_kind=upload requires content_b64")
            if self.object_key or self.external_source or self.url:
                raise ValueError("object_key/external_source/url are not valid for source_kind=upload")
        elif self.source_kind is SourceKind.OBJECT_STORE:
            if not self.object_key:
                raise ValueError("source_kind=object_store requires object_key")
            if self.content_b64 or self.url:
                raise ValueError("content_b64/url are not used for source_kind=object_store")
        elif self.source_kind is SourceKind.WEB:
            if not self.url or not self.url.startswith(("http://", "https://")):
                raise ValueError("source_kind=web requires an http(s) url")
            if self.content_b64 or self.object_key or self.external_source:
                raise ValueError("only url is used for source_kind=web")
        else:
            raise ValueError(f"source_kind={self.source_kind.value} is not yet supported for direct ingest")
        return self


class IngestResponse(BaseModel):
    doc_id: str
    chunk_count: int
    pii_masked: int
    trace_id: str
    outcome: str = "created"  # created | updated | unchanged | quarantined | blocked_pii (FR-A8 delta)
    duplicate_of: str | None = None  # identical content already under another doc (FR-C9)
    classification: int | None = None  # final data-classification (raised by the PII floor when applicable)
    reason: str | None = None  # detail when outcome == "blocked_pii" (which PII, required level)


class QueryRequest(BaseModel):
    # max_length: a query this long has no legitimate answer use-case, and an unbounded
    # string reaches the OpenSearch BM25 match query as one boolean clause PER TOKEN --
    # past maxClauseCount (1024) that raises an uncaught RequestError -> unhandled 500.
    # 2000 chars is already generous for a real question; it fails this the same clean
    # 422 shape as the existing min_length=1 (empty-query) validation.
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(
        default_factory=lambda: get_settings().query_top_k, ge=1, le=50
    )  # recall floor, measured on the golden set — see Settings.query_top_k
    mode: RetrievalMode = RetrievalMode.HYBRID
    filters: dict[str, str] = Field(default_factory=dict)
    # The DOCUMENT SCOPE, and the first-class predicate `filters` has always redirected callers
    # to: `filters={"doc_id": ...}` is rejected 422 with the hint "Use `doc_ids` for a document
    # scope", and until now that hint named a field this model did not have. A `doc_id` TAG
    # silently empties the BM25 leg (the D1 finding), which is why this is its own predicate
    # rather than a filter.
    #
    # It is also the only way to reach the Phase-8 section pass and Phase-9 map-reduce: those
    # read EVERY section of the documents in scope, so `_select_sections` declines an unscoped
    # question ("no document in scope") rather than load a whole KB onto the latency path, and
    # `_route_v2` cannot infer a scope because the `doc` chunk level has no producer (§8.17).
    # Empty — the default — leaves every existing caller byte-identical.
    #
    # NOT AN AUTHORISATION CHANNEL. It NARROWS a search that is already bound to this KB's
    # namespace and already filtered by the caller's ACL groups and clearance; it can never
    # widen one. `_assert_doc_scope_visible` still checks each id against the documents this
    # caller may see, so naming a document that does not exist and naming one they may not read
    # fail the SAME way — a request must not become an existence oracle for restricted files.
    doc_ids: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="Restrict the answer to these documents. Required to reach the section pass.",
    )
    model: str | None = Field(default=None, description="Per-question model override (same endpoint/creds).")
    # Recency-aware retrieval (FR-D9) — all optional, default-off.
    after: datetime | None = Field(default=None, description="Only documents ingested at/after this instant.")
    before: datetime | None = Field(default=None, description="Only documents ingested at/before this instant.")
    recency_days: int | None = Field(default=None, ge=1, le=3650, description="Shorthand: only the last N days.")
    freshness: float = Field(default=0.0, ge=0.0, le=1.0, description="Freshness-boost weight (0=off).")

    @field_validator("filters")
    @classmethod
    def _no_record_fields_as_filters(cls, v: dict[str, str]) -> dict[str, str]:
        """Reject a chunk RECORD field used as a metadata filter key, with the same clean 422 the
        min_length/max_length validations above give. `RetrievalQuery` enforces this too, but only
        AFTER the handler has started; validating on the request model means an external caller is
        told what to use instead (`doc_ids`) before any retrieval runs. See
        `rag_schemas.retrieval.RESERVED_FILTER_KEYS` for why this was previously silent."""
        return reject_reserved_filter_keys(v)

    def window_after(self) -> datetime | None:
        """Resolve the effective lower bound (recency_days shorthand wins if set)."""
        if self.recency_days is not None:
            return datetime.now(UTC) - timedelta(days=self.recency_days)
        return self.after


class HitlActRequest(BaseModel):
    action: HitlAction
    reviewer: str
    note: str | None = None
    modified_answer: str | None = None


# --- Knowledge base lifecycle (FR-H1) — an org owns MANY KBs -----------------
class KbInfo(BaseModel):
    kb_id: str
    name: str
    is_default: bool = False
    published: bool = True
    personal: bool = False  # a user-owned KB (creator is its manager), not a governed dept KB
    owner: str = ""  # email of the personal-KB owner (empty for governed dept KBs)


@router.get("/knowledge-bases", response_model=list[KbInfo])
async def list_knowledge_bases(principal: Annotated[Principal, Depends(get_principal)]) -> list[KbInfo]:
    """List the knowledge bases the caller's org owns, plus the one platform-wide SHARED_KB_ID
    (Federated RAG Tier C — every tenant reads and may ingest into it, unconditionally, no
    grant). No auto-provisioned 'Default' KB otherwise — an org-owned KB exists only after
    it is explicitly created, so a new org starts with just the shared KB in this list."""
    tenant_id = principal.tenant_id
    registered = await wiring.kb_store().list(tenant_id)
    allowed = principal.attrs.get("kb_access") or []  # [] = all KBs; else restrict to granted ones
    subject = subject_of(principal)
    me = principal.subject.strip().lower()
    parents = await parents_map(wiring.org_unit_store(), tenant_id)
    personal_map = await _load_personal_kbs(tenant_id)
    own: list[KbInfo] = []
    for r in registered:
        if r.kb_id == tenant_id:
            continue
        # Personal KBs are PRIVATE: visible only to their owner, or to someone they've shared with
        # (a live grant) — never through all-KB scoping, MLS, or admin. This gate REPLACES the normal
        # one for personal KBs so "only the creator sees it until they share it" holds absolutely.
        if r.kb_id in personal_map:
            grants = await kb_roles.load_access_grants(get_catalog(), r.kb_id)
            if me == personal_map[r.kb_id] or me in grants:
                own.append(
                    KbInfo(
                        kb_id=r.kb_id, name=r.name, published=r.published, personal=True, owner=personal_map[r.kb_id]
                    )
                )
            continue
        # Governed dept KB: visible if in the caller's base kb_access AND it passes the MLS/unit read
        # gate — OR the caller holds a LIVE cross-KB access GRANT (authorises access even when the KB
        # is outside their base kb_access; without it an approved request never showed in the switcher).
        in_base = (not allowed or r.kb_id in allowed) and can_read_kb(
            subject,
            kb_classification=Clearance(r.classification_floor),
            kb_unit_id=r.unit_id or tenant_id,
            parents=parents,
        )
        # The second branch is the cross-KB GRANT only — NOT _has_kb_access, which returns True
        # for every caller whose kb_access list is empty ("all KBs", the normal case). Using it
        # here ORed the MLS/unit gate away for almost everyone, so a Confidential KB was listed
        # to an under-cleared reader: the query path correctly 404'd, but discovery still leaked
        # its existence and its NAME. The "all KBs" case is already covered inside `in_base`.
        if in_base or await _has_access_grant(principal, r.kb_id):
            own.append(KbInfo(kb_id=r.kb_id, name=r.name, published=r.published))
    # Unconditional: the shared KB isn't subject to this org's own kb_access/unit governance —
    # it's a platform-wide resource, not one of this tenant's own KBs.
    return [*own, KbInfo(kb_id=SHARED_KB_ID, name=SHARED_KB_NAME, published=True)]


@router.post("/knowledge-bases", status_code=201, response_model=KbInfo)
async def create_kb(body: CreateKBRequest, principal: Annotated[Principal, Depends(get_principal)]) -> KbInfo:
    """Create a NEW knowledge base for the org — its own isolated namespace (FR-H1). When the
    publish gate is on (maker-checker), the KB is created as a DRAFT and does not answer
    end-user queries until a second admin approves publishing it."""
    if not (principal.has_role("owner") or principal.has_role("admin")):
        raise ForbiddenError("only an owner or admin can create a knowledge base")
    tenant_id = principal.tenant_id
    kb_id = slug_kb_id(tenant_id, body.display_name)
    # Fail-closed on the non-injective tenant slug: never provision a KB whose derived store
    # namespace would collide with the org or an existing sibling KB (differs only by case or -/_).
    existing_kbs = {r.kb_id for r in await wiring.kb_store().list(tenant_id)}
    assert_registrable(kb_id, existing_kbs | {tenant_id})
    # A slug that collides with a TRASHED KB would otherwise upsert-revive that row — and its
    # RETAINED documents/graph/conversations — as a surprise "new" KB pre-filled with old content
    # under old ACLs. Reject with 409 and make the caller restore or purge it first (explicit, no
    # silent resurrection). list() excludes trashed, so assert_registrable above can't catch this.
    if kb_id in {r.kb_id for r in await wiring.kb_store().list_trashed(tenant_id)}:
        raise ConflictError(
            "a deleted knowledge base with this name is in Trash — restore it or permanently delete it first",
            details={"kb_id": kb_id},
        )
    published = not dual_control.is_gated("publish_kb")  # draft when the publish gate is active
    await wiring.provision_kb(kb_id)  # provision this KB's own vector/BM25/object/schema namespace
    await wiring.kb_store().add(tenant_id, kb_id, body.display_name, published=published)
    _log.info("kb_created", tenant_id=tenant_id, kb_id=kb_id, display_name=body.display_name, published=published)
    return KbInfo(kb_id=kb_id, name=body.display_name, published=published)


# --- Platform control plane (SUPER_ADMIN: managed-cloud tenant lifecycle) -----
def _require_platform_admin(principal: Principal) -> None:
    """Gate on the platform-operator allowlist. Separation of duties: this authorises FLEET
    lifecycle, never tenant data (a super admin cannot read a KB — see rbac.can_read_kb)."""
    if not is_platform_admin(principal.subject):
        raise ForbiddenError("platform-admin only", details={"subject": principal.subject})


class TenantProvisionRequest(BaseModel):
    org_id: str = Field(..., min_length=2, max_length=63)
    name: str = Field(..., min_length=1)
    admin_email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=8)
    plan: str = Field(default="standard")


class TenantPlanRequest(BaseModel):
    plan: str


@router.post("/platform/tenants", response_model=tenant_control.TenantRecord, status_code=201)
async def provision_tenant(
    body: TenantProvisionRequest, principal: Annotated[Principal, Depends(get_principal)]
) -> tenant_control.TenantRecord:
    """Provision a managed-cloud tenant end to end (org + owner + namespaces + fleet record)."""
    _require_platform_admin(principal)
    return await tenant_control.provision(
        org_id=body.org_id,
        name=body.name,
        admin_email=body.admin_email,
        password=body.password,
        plan=body.plan,
        actor=principal.subject,
    )


@router.get("/platform/tenants", response_model=list[tenant_control.TenantRecord])
async def list_tenants(principal: Annotated[Principal, Depends(get_principal)]) -> list[tenant_control.TenantRecord]:
    """Every org the platform knows about: fleet-provisioned tenants (managed) PLUS any org that
    exists in the identity store but was never fleet-provisioned (self-serve signup / legacy),
    surfaced read-only as ``managed=false``. Keeps the super-admin view from ever being blind to
    an org the ``provision`` guard already treats as existing."""
    _require_platform_admin(principal)
    ident = wiring.identity_store()
    managed = await wiring.tenant_registry().list_all()
    known = {t.org_id for t in managed}
    extra: list[tenant_control.TenantRecord] = []
    for org in await ident.list_orgs():
        if org.org_id in known:
            continue
        owner = next((u.email for u in await ident.list_users(org.org_id) if u.role == "owner"), "")
        extra.append(tenant_control.TenantRecord(org_id=org.org_id, name=org.name, admin_email=owner, managed=False))
    return [*managed, *sorted(extra, key=lambda t: t.org_id)]


@router.get("/platform/tenants/{org_id}", response_model=tenant_control.TenantSummary)
async def get_tenant(
    org_id: str, principal: Annotated[Principal, Depends(get_principal)]
) -> tenant_control.TenantSummary:
    """One tenant's record + a live KB/user rollup."""
    _require_platform_admin(principal)
    summ = await tenant_control.summary(org_id)
    if summ is None:
        raise NotFoundError("tenant not provisioned", details={"org_id": org_id})
    return summ


@router.post("/platform/tenants/{org_id}/suspend", response_model=tenant_control.TenantRecord)
async def suspend_tenant(
    org_id: str, principal: Annotated[Principal, Depends(get_principal)]
) -> tenant_control.TenantRecord:
    """Withhold service from a tenant (non-payment / hold). Its users are refused KB access."""
    _require_platform_admin(principal)
    return await tenant_control.set_status(org_id, tenant_control.TenantStatus.SUSPENDED, actor=principal.subject)


@router.post("/platform/tenants/{org_id}/resume", response_model=tenant_control.TenantRecord)
async def resume_tenant(
    org_id: str, principal: Annotated[Principal, Depends(get_principal)]
) -> tenant_control.TenantRecord:
    """Restore a suspended tenant."""
    _require_platform_admin(principal)
    return await tenant_control.set_status(org_id, tenant_control.TenantStatus.ACTIVE, actor=principal.subject)


@router.post("/platform/tenants/{org_id}/plan", response_model=tenant_control.TenantRecord)
async def set_tenant_plan(
    org_id: str, body: TenantPlanRequest, principal: Annotated[Principal, Depends(get_principal)]
) -> tenant_control.TenantRecord:
    """Change a tenant's plan (its resolved caps are re-stamped)."""
    _require_platform_admin(principal)
    return await tenant_control.set_plan(org_id, body.plan, actor=principal.subject)


@router.delete("/platform/tenants/{org_id}")
async def deprovision_tenant(org_id: str, principal: Annotated[Principal, Depends(get_principal)]) -> dict[str, int]:
    """Permanently purge a tenant's KBs and tombstone its fleet record."""
    _require_platform_admin(principal)
    return await tenant_control.deprovision(org_id, actor=principal.subject)


@router.get("/platform/tenants/{org_id}/invoice", response_model=billing.Invoice)
async def tenant_invoice(
    org_id: str,
    period: Annotated[str, Query(pattern=r"^\d{4}-\d{2}$", description="Billing month, YYYY-MM.")],
    principal: Annotated[Principal, Depends(get_principal)],
) -> billing.Invoice:
    """A tenant invoice for a calendar month (metered usage x rate card across its KBs), recorded
    on the audit chain. Platform-admin only."""
    _require_platform_admin(principal)
    return await billing.invoice(org_id, period, actor=principal.subject)


@router.get("/billing/invoice", response_model=billing.Invoice)
async def my_invoice(
    period: Annotated[str, Query(pattern=r"^\d{4}-\d{2}$", description="Billing month, YYYY-MM.")],
    principal: Annotated[Principal, Depends(get_principal)],
) -> billing.Invoice:
    """A tenant pulls its OWN invoice for a calendar month (FinOps self-service)."""
    return await billing.invoice(principal.tenant_id, period, actor=principal.subject)


# --- Provenance: signed SBOM, residency attestation, receipt verify ----------
@router.get("/sbom", response_model=ModelSBOM)
async def model_sbom(principal: Annotated[Principal, Depends(get_principal)]) -> ModelSBOM:
    """The deployment's SIGNED model bill of materials — the India-origin, self-hosted claim made
    verifiable. 404 when no models are declared (``model_sbom_models`` unset)."""
    _ = principal  # any authenticated caller may read (the signature, not access, is the trust)
    sbom = provenance.signed_sbom()
    if sbom is None:
        raise NotFoundError("model SBOM not configured", details={"hint": "set model_sbom_models"})
    return sbom


@router.get("/platform/residency/node", response_model=compliance.NodeResidencyAttestation)
async def node_residency(
    principal: Annotated[Principal, Depends(get_principal)],
) -> compliance.NodeResidencyAttestation:
    """A SIGNED per-node data-residency attestation an operator hands a regulator. Platform-admin only."""
    _require_platform_admin(principal)
    return await provenance.signed_residency()


class ReceiptVerifyResult(BaseModel):
    valid: bool


@router.post("/receipts/verify", response_model=ReceiptVerifyResult)
async def verify_receipt_endpoint(
    receipt: TrustReceipt, principal: Annotated[Principal, Depends(get_principal)]
) -> ReceiptVerifyResult:
    """Verify a supplied trust receipt against the provider public key — stateless, fail-closed."""
    _ = principal  # authenticated; verification itself needs no tenant scope
    return ReceiptVerifyResult(valid=provenance.verify_supplied_receipt(receipt))


@router.get("/receipts/{answer_id}", response_model=TrustReceipt)
async def get_receipt(
    answer_id: Annotated[str, Path(description="The answer's trace id.")],
    principal: Annotated[Principal, Depends(get_principal)],
) -> TrustReceipt:
    """Pull the signed trust receipt for one answer (by trace id) from the tenant's audit chain.
    404 when the answer produced no receipt (abstained, or trust receipts disabled at the time)."""
    # The RECEIPT event is recorded under the ANSWER's tenant (== the queried kb_id, e.g.
    # "acme__kb"), which differs from the JWT tenant (the org, "acme") on a normal per-KB
    # query — so a single-namespace lookup 404'd the owner on their own receipt. Search the
    # caller's org namespace AND every KB they own; each is a namespace they are entitled to.
    namespaces = {principal.tenant_id}
    # KB enumeration is best-effort; the org namespace is always searched.
    with contextlib.suppress(Exception):
        namespaces |= {k.kb_id for k in await list_knowledge_bases(principal)}
    for ns in namespaces:
        for ev in await wiring.audit_service().list(ns, trace_id=answer_id):
            receipt = ev.payload.get("receipt") if ev.action == AuditAction.RECEIPT else None
            if receipt:
                return TrustReceipt(**receipt)
    raise NotFoundError("no trust receipt for this answer", details={"answer_id": answer_id})


@router.get("/platform/airgap/preflight")
async def airgap_preflight_report(
    principal: Annotated[Principal, Depends(get_principal)],
) -> dict[str, object]:
    """Air-gap preflight: report any config that would breach a fully-offline deployment
    (remote egress, off-box backends, missing update-verify key). Platform-admin only."""
    _require_platform_admin(principal)
    return airgap_preflight(get_settings())


# --- Org units (hierarchical RBAC scope) -------------------------------------
class UnitCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    parent_id: str | None = Field(default=None, description="Parent unit; omit for a top-level unit under root.")


class UnitInfo(BaseModel):
    unit_id: str
    name: str
    parent_id: str | None


def _unit_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "unit"


@router.get("/units", response_model=list[UnitInfo])
async def list_units(principal: Annotated[Principal, Depends(get_principal)]) -> list[UnitInfo]:
    """The tenant's org-unit tree (RBAC scope). Everyone in the tenant may READ the shape."""
    await wiring.org_unit_store().ensure_root(principal.tenant_id)
    units = await wiring.org_unit_store().list_for_tenant(principal.tenant_id)
    return [UnitInfo(unit_id=u.unit_id, name=u.name, parent_id=u.parent_id) for u in units]


@router.post("/units", response_model=UnitInfo, status_code=201)
async def create_unit(body: UnitCreate, principal: Annotated[Principal, Depends(get_principal)]) -> UnitInfo:
    """Create an org unit under ``parent_id`` (root by default). Only an admin OF THAT PARENT
    subtree may (a TENANT_ADMIN, or a UNIT_ADMIN whose unit is an ancestor) — no one grows a
    branch they do not govern."""
    store = wiring.org_unit_store()
    await store.ensure_root(principal.tenant_id)
    parent = body.parent_id or principal.tenant_id
    parents = await parents_map(store, principal.tenant_id)
    if parent not in parents:
        raise NotFoundError("parent unit not found", details={"parent_id": parent})
    if not can_admin_unit(subject_of(principal), unit_id=parent, parents=parents):
        raise ForbiddenError("you may not create a unit under this parent", details={"parent_id": parent})
    unit_id = _unit_slug(body.name)
    if unit_id == principal.tenant_id or unit_id in parents:
        raise ValidationError("a unit with this name already exists", details={"unit_id": unit_id})
    await store.create(OrgUnit(unit_id=unit_id, tenant_id=principal.tenant_id, name=body.name, parent_id=parent))
    _log.info("org_unit_created", tenant_id=principal.tenant_id, unit_id=unit_id, parent=parent)
    return UnitInfo(unit_id=unit_id, name=body.name, parent_id=parent)


@router.delete("/units/{unit_id}", status_code=204)
async def delete_unit(unit_id: str, principal: Annotated[Principal, Depends(get_principal)]) -> None:
    """Delete an EMPTY, non-root unit. Only an admin of the unit may. The store refuses the root
    or a unit that still has children."""
    store = wiring.org_unit_store()
    parents = await parents_map(store, principal.tenant_id)
    if unit_id not in parents:
        raise NotFoundError("unit not found", details={"unit_id": unit_id})
    if not can_admin_unit(subject_of(principal), unit_id=unit_id, parents=parents):
        raise ForbiddenError("you may not delete this unit", details={"unit_id": unit_id})
    await store.delete(principal.tenant_id, unit_id)
    _log.info("org_unit_deleted", tenant_id=principal.tenant_id, unit_id=unit_id)


@router.post("/kb/{kb_id}/publish", response_model=KbInfo)
async def publish_kb(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> KbInfo:
    """Publish a draft KB so it answers end-user queries. Gated by maker-checker: when enabled,
    this stages the publish for a second admin's approval (the executor flips the flag)."""
    if not (principal.has_role("owner") or principal.has_role("admin")):
        raise ForbiddenError("only an owner or admin can publish a knowledge base")
    rec = await wiring.kb_store().get(principal.tenant_id, kb_id)
    if rec is None:
        raise NotFoundError("knowledge base not found", details={"kb_id": kb_id})
    await _gate(
        "publish_kb",
        tenant_id=principal.tenant_id,
        kb_id=kb_id,
        params={"kb_id": kb_id},
        summary=f"Publish knowledge base '{rec.name}' to end-users",
        principal=principal,
    )
    await wiring.kb_store().set_published(principal.tenant_id, kb_id, True)
    return KbInfo(kb_id=kb_id, name=rec.name, published=True)


class DeleteKbResult(BaseModel):
    kb_id: str
    documents: int
    chunks: int


class TrashKbResult(BaseModel):
    kb_id: str
    status: str = "trashed"


class TrashedKb(BaseModel):
    kb_id: str
    name: str
    trashed_at: str | None = None


class PurgeKbRequest(BaseModel):
    confirm_name: str = Field(..., description="Must exactly match the KB's name — a typed-confirmation guard.")


def _assert_owns_kb(kb_id: str, org_id: str) -> None:
    if not kb_id.startswith(f"{org_id}__"):
        raise TenantIsolationError(
            "principal may not act on this knowledge base", details={"kb_id": kb_id, "principal_tenant": org_id}
        )


@router.delete("/knowledge-bases/{kb_id}", response_model=TrashKbResult)
async def delete_kb(kb_id: str, principal: Annotated[Principal, Depends(get_principal)]) -> TrashKbResult:
    """Move a knowledge base to Trash (soft-delete). Data is retained and restorable — nothing is
    purged. Use the purge endpoint (with typed confirmation) to permanently delete it."""
    org_id = principal.tenant_id
    if not (principal.has_role("owner") or principal.has_role("admin")):
        raise ForbiddenError("only an owner or admin can trash a knowledge base")
    _assert_owns_kb(kb_id, org_id)
    await pipeline.trash_knowledge_base(org_id, kb_id, actor=principal.subject)
    return TrashKbResult(kb_id=kb_id)


@router.get("/knowledge-bases/trash", response_model=list[TrashedKb])
async def list_trashed_kbs(principal: Annotated[Principal, Depends(get_principal)]) -> list[TrashedKb]:
    """Knowledge bases in Trash — restorable, or permanently purgeable."""
    rows = await wiring.kb_store().list_trashed(principal.tenant_id)
    return [
        TrashedKb(kb_id=r.kb_id, name=r.name, trashed_at=r.trashed_at.isoformat() if r.trashed_at else None)
        for r in rows
    ]


@router.post("/knowledge-bases/{kb_id}/restore", response_model=KbInfo)
async def restore_kb(kb_id: str, principal: Annotated[Principal, Depends(get_principal)]) -> KbInfo:
    """Restore a knowledge base from Trash — it returns exactly as it was (nothing was purged)."""
    org_id = principal.tenant_id
    if not (principal.has_role("owner") or principal.has_role("admin")):
        raise ForbiddenError("only an owner or admin can restore a knowledge base")
    _assert_owns_kb(kb_id, org_id)
    rec = await wiring.kb_store().get(org_id, kb_id)
    if rec is None:
        raise NotFoundError("knowledge base not found", details={"kb_id": kb_id})
    await pipeline.restore_knowledge_base(org_id, kb_id, actor=principal.subject)
    return KbInfo(kb_id=kb_id, name=rec.name)


@router.post("/knowledge-bases/{kb_id}/purge", response_model=DeleteKbResult)
async def purge_kb(
    kb_id: str, body: PurgeKbRequest, principal: Annotated[Principal, Depends(get_principal)]
) -> DeleteKbResult:
    """Permanently delete a knowledge base and ALL its data — irreversible. Requires the caller to
    type the KB's exact name as confirmation (guards against accidental destruction)."""
    org_id = principal.tenant_id
    _assert_owns_kb(kb_id, org_id)
    if not (principal.has_role("owner") or principal.has_role("admin")):
        raise ForbiddenError("only an owner or admin can permanently delete a knowledge base")
    rec = await wiring.kb_store().get(org_id, kb_id)
    if rec is None:
        raise NotFoundError("knowledge base not found", details={"kb_id": kb_id})
    if body.confirm_name.strip() != rec.name:
        raise ValidationError(
            "confirmation text does not match the knowledge base name", details={"expected": rec.name}
        )
    await _gate(
        "purge",
        tenant_id=org_id,
        kb_id=kb_id,
        params={"kb_id": kb_id},
        summary=f"Permanently purge knowledge base '{rec.name}'",
        principal=principal,
    )
    result = await pipeline.delete_knowledge_base(org_id, kb_id, actor=principal.subject)
    return DeleteKbResult(kb_id=kb_id, documents=result["documents"], chunks=result["chunks"])


@router.post("/kb/{kb_id}/ingest", response_model=IngestResponse)
async def ingest(
    kb_id: str, body: IngestRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> IngestResponse:
    """Ingest → OCR/parse/mask → chunk → embed → index (TDD §5.1)."""
    _require_role(principal, "ingestor")
    check_ingest_rate(principal.tenant_id)
    await check_document_cap(kb_id)
    # Maker-checker (banking pack): a Confidential/Restricted UPLOAD is staged for a second
    # admin's approval (the executor re-runs the ingest from the stored request).
    cls = parse_classification(body.classification)
    # The shared KB (Federated RAG Tier C) is readable by every tenant, unconditionally — so
    # it may never hold anything above Public. This is the one hard backstop against a
    # mis-click accidentally broadcasting Confidential/Restricted content company-wide;
    # everything else about who may ingest here is the normal ingestor-role check above.
    if kb_id == SHARED_KB_ID and cls > 0:
        raise ForbiddenError(
            "the shared KB is company-wide readable and may only hold Public (0) content",
            details={"kb_id": kb_id, "requested_classification": cls},
        )
    if kb_id == SHARED_KB_ID:
        # Unlike a normal KB, SHARED_KB_ID is never created via POST /knowledge-bases (it isn't
        # owned by any one tenant), so nothing has provisioned its vector/BM25/schema namespace
        # yet the first time any tenant ingests into it. Idempotent — ensure_namespaces mirrors
        # the CREATE-IF-NOT-EXISTS pattern every other store in this codebase already uses.
        await wiring.provision_kb(SHARED_KB_ID)
    _kb = await wiring.kb_store().get(principal.tenant_id, kb_id)
    # Additive KB-role elevation: a member's assigned KB role can raise their clearance ceiling for
    # THIS KB (never lower it), and its can_ingest flag additionally gates writing here.
    _ingest_subject = await kb_roles.elevated_subject(get_catalog(), kb_id, principal)
    if not await kb_roles.role_allows_ingest(get_catalog(), kb_id, principal):
        raise ForbiddenError(
            "your assigned role for this KB does not permit ingestion",
            details={"kb_id": kb_id},
        )
    if not can_ingest_at(
        _ingest_subject,
        kb_floor=Clearance(_kb.classification_floor if _kb else 0),
        requested=Clearance(cls),
        kb_unit_id=(_kb.unit_id if _kb else None) or principal.tenant_id,
        parents=await parents_map(wiring.org_unit_store(), principal.tenant_id),
    ):
        raise ForbiddenError(
            "your clearance/role may not ingest a document at this classification into this KB",
            details={"kb_id": kb_id, "requested_classification": cls},
        )
    if body.source_kind is SourceKind.UPLOAD and cls >= int(ClassificationLevel.CONFIDENTIAL):
        await _gate(
            "ingest_classified",
            tenant_id=kb_id,
            kb_id=kb_id,
            params={
                "doc_id": body.doc_id,
                "mime": body.mime,
                "content_b64": body.content_b64,
                "acl": body.acl,
                "classification": cls,
                "metadata": body.metadata,
                "strategy": (body.strategy.value if body.strategy else None),
            },
            summary=f"Ingest {classification_label(cls)} document '{body.doc_id}'",
            principal=principal,
        )
    by_ref = body.source_kind is SourceKind.OBJECT_STORE
    is_web = body.source_kind is SourceKind.WEB
    # Web pages default to text/html extraction unless the caller says otherwise.
    mime = "text/html" if is_web and body.mime == "application/pdf" else body.mime
    doc = DocumentEnvelope(
        tenant_id=kb_id,
        doc_id=body.doc_id,
        source_id="api-web" if is_web else ("api-object-store" if by_ref else "api-upload"),
        source_kind=body.source_kind,
        mime=mime,
        content_b64=body.content_b64,
        uri=body.url if is_web else (body.object_key if by_ref else None),
        external_source=body.external_source if by_ref else None,
        source_acl=body.acl,
        classification=parse_classification(body.classification),
        metadata=body.metadata,
        ingested_at=datetime.now(UTC),
    )
    outcome = await pipeline.ingest_document(
        doc,
        actor=principal.subject,
        strategy=body.strategy,
        # Per-KB role elevation applies: a member elevated to (say) Confidential in THIS KB may
        # ingest Confidential content here, so the PII floor caps against their effective KB clearance.
        clearance=await kb_roles.effective_kb_clearance(get_catalog(), kb_id, principal),
        pii_policy=_resolve_pii_policy(body.pii_autoclassify),
    )
    return IngestResponse(
        doc_id=outcome.doc_id,
        chunk_count=outcome.chunk_count,
        pii_masked=outcome.pii_masked,
        trace_id=outcome.trace_id,
        outcome=outcome.outcome,
        duplicate_of=outcome.duplicate_of,
        classification=outcome.classification,
        reason=outcome.reason,
    )


# --- Per-KB settings: threshold overrides (grounding + faithfulness routing) --
class KbSettings(BaseModel):
    """Per-KB overrides; null = use the platform default (shown in `defaults`)."""

    abstain_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    faithful_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    flag_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class KbSettingsInfo(BaseModel):
    overrides: KbSettings
    defaults: dict[str, float]


def _platform_threshold_defaults() -> dict[str, float]:
    s = get_settings()
    return {
        "abstain_threshold": s.abstain_threshold,
        "faithful_threshold": s.faithful_threshold,
        "flag_threshold": s.flag_threshold,
    }


@router.get("/kb/{kb_id}/settings", response_model=KbSettingsInfo)
async def get_kb_settings(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> KbSettingsInfo:
    """This KB's threshold overrides + the platform defaults they fall back to."""
    overrides = await pipeline.kb_threshold_overrides(kb_id)
    return KbSettingsInfo(overrides=KbSettings(**overrides), defaults=_platform_threshold_defaults())


@router.put("/kb/{kb_id}/settings", response_model=KbSettingsInfo)
async def set_kb_settings(
    kb_id: str, body: KbSettings, principal: Annotated[Principal, Depends(tenant_guard)]
) -> KbSettingsInfo:
    """Set (or clear, with null) this KB's threshold overrides. Takes effect immediately."""
    # These overrides move the grounding gate (abstain/faithfulness thresholds), so restrict to
    # the KB's management tier — never an ordinary member (the console already hides the page).
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only a KB manager or org admin can change this KB's settings")
    catalog = get_catalog()
    changed = False
    for key, value in body.model_dump().items():
        # Clearing = write the platform default? No — store "" to mean cleared is ambiguous;
        # we store the value when given and skip nulls (explicit clear = set to default value).
        if value is not None:
            await catalog.set_tenant_config(kb_id, key, str(value))
            changed = True
    # A manual threshold/knob edit changes how answers are produced right now; a stale semantic-cache
    # entry would keep serving the pre-edit behaviour. Invalidate so the edit takes effect immediately
    # (same class of change — and the same fix — as domain_profile.apply_suggestion).
    if changed:
        from svc_control_plane.semantic_cache import cache as _semantic_cache  # noqa: PLC0415

        _semantic_cache.invalidate(kb_id)
    _log.info("kb_settings_updated", kb_id=kb_id, overrides=body.model_dump(exclude_none=True))
    overrides = await pipeline.kb_threshold_overrides(kb_id)
    return KbSettingsInfo(overrides=KbSettings(**overrides), defaults=_platform_threshold_defaults())


# --- Domain-adaptive profile (Slice 6, sub-slice A) — detect-and-SUGGEST ------
class DomainProfileView(BaseModel):
    """A KB's active domain profile + its PENDING knob suggestion + the applied-knob audit +
    the KB's CURRENT knob values. Detection writes profile + suggestion but moves no knob — the
    operator confirms via ``/profile/apply``."""

    enabled: bool = Field(description="Whether domain profiling is enabled (RAG_DOMAIN_PROFILE_ENABLED).")
    profile: dict[str, Any] = Field(default_factory=dict, description="Active profile: domain/score/mix/evidence.")
    suggestion: dict[str, Any] = Field(default_factory=dict, description="Pending, unconfirmed knob recommendation.")
    applied: dict[str, Any] = Field(default_factory=dict, description="Audit of knobs an operator confirmed.")
    current_knobs: dict[str, float] = Field(default_factory=dict, description="The KB's current knob overrides.")


class DomainProfileApplyRequest(BaseModel):
    """Which suggested knobs the operator confirms. Reindex-class knobs are refused unless
    ``allow_reindex`` is set (and the caller is warned a reindex is required)."""

    query_knobs: dict[str, float] = Field(default_factory=dict, description="Query-time knobs to apply now.")
    reindex_knobs: dict[str, float] = Field(default_factory=dict, description="Reindex-class knobs (opt-in only).")
    allow_reindex: bool = Field(default=False, description="Explicit opt-in to apply reindex-class knobs.")


class DomainProfileApplyResult(BaseModel):
    applied: dict[str, float] = Field(default_factory=dict)
    refused_reindex: dict[str, float] = Field(default_factory=dict)
    allow_reindex: bool = False
    reindex_warning: str | None = None
    current_knobs: dict[str, float] = Field(default_factory=dict)


def _require_domain_profile_enabled() -> None:
    if not get_settings().domain_profile_enabled:
        raise ValidationError("domain profiling is disabled (set RAG_DOMAIN_PROFILE_ENABLED to enable it)")


@router.post("/kb/{kb_id}/profile/detect", response_model=DomainProfileView)
async def detect_domain_profile(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> DomainProfileView:
    """Detect the KB's domain from its corpus and compute a knob SUGGESTION. Writes the profile +
    suggestion; moves NO knob (detect-and-suggest). Gated like ``set_kb_settings`` — a profile
    moves the grounding gate, the same sensitivity class."""
    _require_domain_profile_enabled()
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only a KB manager or org admin can detect this KB's domain profile")
    await domain_profile.detect(kb_id)
    return await _domain_profile_view(kb_id)


@router.get("/kb/{kb_id}/profile", response_model=DomainProfileView)
async def get_domain_profile(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> DomainProfileView:
    """The KB's current profile + pending suggestion + applied-knob audit + current knob values."""
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only a KB manager or org admin can view this KB's domain profile")
    return await _domain_profile_view(kb_id)


@router.post("/kb/{kb_id}/profile/apply", response_model=DomainProfileApplyResult)
async def apply_domain_profile(
    kb_id: str, body: DomainProfileApplyRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> DomainProfileApplyResult:
    """Operator confirmation: apply the selected QUERY-TIME knobs via the same ``set_tenant_config``
    call ``set_kb_settings`` uses. Reindex-class knobs are REFUSED unless ``allow_reindex`` is set —
    the profile never silently triggers a reindex or flips multivector/parent_child sizing."""
    _require_domain_profile_enabled()
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only a KB manager or org admin can apply this KB's domain profile")
    result = await domain_profile.apply_suggestion(
        kb_id,
        query_knobs=body.query_knobs,
        reindex_knobs=body.reindex_knobs,
        allow_reindex=body.allow_reindex,
        actor=principal.subject,
    )
    return DomainProfileApplyResult(
        applied=result.applied,
        refused_reindex=result.refused_reindex,
        allow_reindex=result.allow_reindex,
        reindex_warning=result.reindex_warning,
        current_knobs=await pipeline.kb_threshold_overrides(kb_id),
    )


async def _domain_profile_view(kb_id: str) -> DomainProfileView:
    return DomainProfileView(
        enabled=get_settings().domain_profile_enabled,
        profile=await domain_profile.load_profile(kb_id),
        suggestion=await domain_profile.load_suggestion(kb_id),
        applied=await domain_profile.load_applied(kb_id),
        current_knobs=await pipeline.kb_threshold_overrides(kb_id),
    )


# --- Per-KB persona / answer style (FR-E7) -----------------------------------
_PERSONA_KEY = "persona"
_PERSONA_MAX = 1200


class PersonaInfo(BaseModel):
    persona: str = Field(default="", description="Answer-style instruction; empty = platform default voice.")


@router.get("/kb/{kb_id}/persona", response_model=PersonaInfo)
async def get_persona(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> PersonaInfo:
    """The KB's answer-style persona (tone, format, output language)."""
    return PersonaInfo(persona=(await get_catalog().get_tenant_config(kb_id)).get(_PERSONA_KEY, ""))


@router.put("/kb/{kb_id}/persona", response_model=PersonaInfo)
async def set_persona(
    kb_id: str, body: PersonaInfo, principal: Annotated[Principal, Depends(tenant_guard)]
) -> PersonaInfo:
    """Set the KB's persona. It shapes wording only — grounded-or-abstain and citations
    always win. Screened by the prompt-injection guard so it can't smuggle instructions."""
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only a KB manager or org admin can change this KB's persona")
    persona = body.persona.strip()[:_PERSONA_MAX]
    if persona and get_settings().guardrails_enabled:
        verdict = await wiring.input_guard().check(persona)
        if not verdict.allowed:
            raise ValidationError(
                "this persona looks like a prompt-injection attempt and was rejected",
                details={"category": verdict.category},
            )
    await get_catalog().set_tenant_config(kb_id, _PERSONA_KEY, persona)
    _log.info("persona_set", kb_id=kb_id, length=len(persona), actor=principal.subject)
    return PersonaInfo(persona=persona)


# --- Per-KB custom named roles (extends the RBAC spine, additive-only) --------
@router.get("/kb/{kb_id}/roles", response_model=kb_roles.KbRoles)
async def get_kb_roles(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> kb_roles.KbRoles:
    """The KB-admin-defined named roles for this KB (empty if none configured)."""
    return await kb_roles.load_roles(get_catalog(), kb_id)


@router.put("/kb/{kb_id}/roles", response_model=kb_roles.KbRoles)
async def set_kb_roles(
    kb_id: str, body: kb_roles.KbRoles, principal: Annotated[Principal, Depends(tenant_guard)]
) -> kb_roles.KbRoles:
    """Define/replace this KB's named roles (admin only, escalation-gated). A role can only raise
    an assigned member's data-clearance ceiling within THIS KB — it never grants management tier,
    and a KB admin may not define a role above their OWN effective clearance."""
    try:
        kb_roles.assert_can_define(principal, body)
    except kb_roles.RoleEscalationError as exc:
        raise ForbiddenError(str(exc), details={"kb_id": kb_id}) from exc
    await kb_roles.save_roles(get_catalog(), kb_id, body)
    await wiring.audit_service().record(
        tenant_id=kb_id,
        trace_id=str(uuid.uuid4()),
        action=AuditAction.ROLE_ASSIGN,
        actor=principal.subject,
        payload={"op": "define", "roles": sorted(body.roles)},
    )
    _log.info("kb_roles_defined", kb_id=kb_id, actor=principal.subject, roles=sorted(body.roles))
    return body


@router.get("/kb/{kb_id}/roles/assignments", response_model=kb_roles.KbRoleAssignments)
async def get_kb_role_assignments(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> kb_roles.KbRoleAssignments:
    """Current member -> role assignments for this KB (empty if none)."""
    return await kb_roles.load_assignments(get_catalog(), kb_id)


@router.put("/kb/{kb_id}/roles/assignments", response_model=kb_roles.KbRoleAssignments)
async def set_kb_role_assignments(
    kb_id: str, body: kb_roles.KbRoleAssignments, principal: Annotated[Principal, Depends(tenant_guard)]
) -> kb_roles.KbRoleAssignments:
    """Set member -> role assignments (admin only, escalation-gated). Every referenced role must
    be defined (422 otherwise); an admin may not assign a role above their own clearance (403)."""
    defined = await kb_roles.load_roles(get_catalog(), kb_id)
    try:
        kb_roles.assert_can_assign(principal, defined, body)
    except kb_roles.UnknownRoleError as exc:
        raise ValidationError(str(exc), details={"kb_id": kb_id}) from exc
    except kb_roles.RoleEscalationError as exc:
        raise ForbiddenError(str(exc), details={"kb_id": kb_id}) from exc
    await kb_roles.save_assignments(get_catalog(), kb_id, body)
    await wiring.audit_service().record(
        tenant_id=kb_id,
        trace_id=str(uuid.uuid4()),
        action=AuditAction.ROLE_ASSIGN,
        actor=principal.subject,
        payload={"op": "assign", "members": sorted(body.assignments)},
    )
    _log.info("kb_roles_assigned", kb_id=kb_id, actor=principal.subject, count=len(body.assignments))
    return body


# --- DPDP compliance surface (FR-H8): erasure, retention, attestation --------
class ErasureRequest(BaseModel):
    doc_ids: list[str] = Field(default_factory=list, description="Specific documents to erase.")
    subject: str | None = Field(default=None, description="Erase every document mentioning this data principal.")
    reason: str = Field(default="data_principal_request", max_length=200)

    @model_validator(mode="after")
    def _need_target(self) -> ErasureRequest:
        if not self.doc_ids and not (self.subject and self.subject.strip()):
            raise ValueError("provide doc_ids and/or a subject to erase")
        return self


@router.post("/kb/{kb_id}/erasure", response_model=compliance.ErasureCertificate)
async def erase_data(
    kb_id: str, body: ErasureRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> compliance.ErasureCertificate:
    """DPDP right-to-erasure: purge documents (by id and/or data-principal subject) from
    every store and the derived chat history, returning a tamper-evident certificate."""
    # Destructive: without this an ordinary member could erase KB data. The dual-control `_gate`
    # below only checks WHETHER maker-checker is enabled, not the caller's role — so gate first.
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only a KB manager or org admin can erase KB data")
    await _gate(
        "erasure",
        tenant_id=kb_id,
        kb_id=kb_id,
        params={"kb_id": kb_id, "doc_ids": body.doc_ids, "subject": body.subject, "reason": body.reason},
        summary=f"Erase data from '{kb_id}' ({len(body.doc_ids)} doc(s)"
        + (f", subject='{body.subject}'" if body.subject else "")
        + ")",
        principal=principal,
    )
    return await compliance.erase(
        kb_id, actor=principal.subject, doc_ids=body.doc_ids, subject=body.subject, reason=body.reason
    )


class RetentionRequest(BaseModel):
    retention_days: int = Field(..., ge=0, le=36500, description="0 = keep indefinitely.")


@router.get("/kb/{kb_id}/retention", response_model=compliance.RetentionInfo)
async def get_retention_policy(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> compliance.RetentionInfo:
    """This KB's retention window (0 = keep forever)."""
    return await compliance.get_retention(kb_id)


@router.put("/kb/{kb_id}/retention", response_model=compliance.RetentionInfo)
async def set_retention_policy(
    kb_id: str, body: RetentionRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> compliance.RetentionInfo:
    """Set this KB's retention window. Enforced on demand via /retention/apply."""
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only a KB manager or org admin can change this KB's retention policy")
    return await compliance.set_retention(kb_id, body.retention_days)


@router.post("/kb/{kb_id}/retention/apply", response_model=compliance.RetentionResult)
async def apply_retention_policy(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> compliance.RetentionResult:
    """Erase documents older than the retention window now (the scheduled worker is FR-F7)."""
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only a KB manager or org admin can apply retention")
    return await compliance.apply_retention(kb_id, actor=principal.subject)


@router.get("/kb/{kb_id}/attestation", response_model=compliance.Attestation)
async def get_attestation(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> compliance.Attestation:
    """Residency + protection attestation for this KB (data location, sovereignty,
    encryption, audit-chain integrity, retention) — an auditor-ready report."""
    return await compliance.attestation(kb_id)


# --- Per-KB golden set (saved evaluation questions, FR-G5) --------------------
_GOLDEN_KEY = "golden_set"


class GoldenSet(BaseModel):
    questions: list[EvalQuestion] = Field(default_factory=list, max_length=50)


@router.get("/kb/{kb_id}/golden-set", response_model=GoldenSet)
async def get_golden_set(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> GoldenSet:
    """The KB's saved evaluation questions (empty if never saved)."""
    raw = (await get_catalog().get_tenant_config(kb_id)).get(_GOLDEN_KEY)
    if not raw:
        return GoldenSet()
    try:
        return GoldenSet.model_validate_json(raw)
    except ValueError:
        return GoldenSet()


@router.put("/kb/{kb_id}/golden-set", response_model=GoldenSet)
async def set_golden_set(
    kb_id: str, body: GoldenSet, principal: Annotated[Principal, Depends(tenant_guard)]
) -> GoldenSet:
    """Save the KB's evaluation questions — rerun them after any chunking/model change."""
    # Match the sibling generate endpoint (ingestor+): the golden set is the quality baseline,
    # not something a bare reader should overwrite.
    _require_role(principal, "ingestor")
    await get_catalog().set_tenant_config(kb_id, _GOLDEN_KEY, body.model_dump_json())
    return body


@router.post("/kb/{kb_id}/golden-set/generate", response_model=GoldenSet)
async def generate_golden_set(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)], count: int = 5
) -> GoldenSet:
    """Draft evaluation Q&A from the KB's own documents (FR-G8) — a quality baseline with
    zero effort. Returns a DRAFT for review/edit; the tenant saves it via PUT. Not saved here."""
    _require_role(principal, "ingestor")
    check_query_rate(principal.tenant_id)
    questions = await golden_gen.generate_golden(kb_id, count)
    return GoldenSet(questions=questions)


# --- Quality dashboards + on-demand evaluation (FR-G5) -----------------------
@router.get("/kb/{kb_id}/quality", response_model=QualityReport)
async def get_quality(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)], window: str = "7d"
) -> QualityReport:
    """Answer-quality report from live traffic (24h | 7d | 30d | all)."""
    try:
        return await quality_report(kb_id, window)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


@router.get("/kb/{kb_id}/analytics/gaps", response_model=AnalyticsGaps)
async def get_analytics_gaps(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)], window: str = "7d"
) -> AnalyticsGaps:
    """Abstain & feedback analytics — the demand signal behind every out-of-scope ask (24h | 7d |
    30d | all): abstain rate, clustered abstained queries, thumbs-down questions, and
    low-faithfulness answers. Reads via the KB-scoped union audit read, so a multi-KB org sees its
    query-time activity."""
    _require_role(principal, "reader")
    try:
        return await analytics_gaps(kb_id, window)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


# --- End-user feedback (FR-I2) ----------------------------------------------
class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., description="Trace id of the answer being rated (from GroundedAnswer.trace_id).")
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=2000)


@router.post("/kb/{kb_id}/feedback", status_code=204)
async def submit_feedback(
    kb_id: str, body: FeedbackRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> None:
    """Record an up/down vote (+ optional comment) on an answer. Trace-correlated, audited, and
    aggregated into the Quality dashboard's satisfaction rate."""
    await wiring.audit_service().record(
        tenant_id=kb_id,
        trace_id=body.trace_id,
        action=AuditAction.FEEDBACK,
        actor=principal.subject,
        payload={"rating": body.rating, "comment": (body.comment or "")[:2000]},
    )
    _log.info("feedback_recorded", kb_id=kb_id, rating=body.rating, trace_id=body.trace_id)


class EvalRequest(BaseModel):
    questions: list[EvalQuestion] = Field(..., min_length=1, max_length=20)


@router.post("/kb/{kb_id}/eval", response_model=EvalReport)
async def run_kb_eval(
    kb_id: str, body: EvalRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> EvalReport:
    """Run a golden-question set through the REAL pipeline and score it (FR-G5)."""
    # Quality is a reviewer/admin surface; an eval run exercises the pipeline, so require an
    # operator role (ingestor/reviewer) or above — not a bare reader.
    if not any(principal.has_role(r) for r in ("ingestor", "reviewer", "admin", "owner")):
        raise ForbiddenError("running an evaluation requires an ingestor, reviewer, or admin role")
    check_query_rate(principal.tenant_id)
    return await run_eval(kb_id, body.questions, acl=_principal_groups(principal), actor=principal.subject)


class FederatedEvalRequest(BaseModel):
    federate_from: list[str] = Field(..., min_length=1, description="Peer departments to evaluate across the mesh.")
    questions: list[EvalQuestion] | None = Field(
        default=None, max_length=20, description="Golden questions; the KB's stored golden set is used if omitted."
    )


@router.post("/kb/{kb_id}/federation/eval", response_model=federated_eval.FederatedEvalReport)
async def run_federation_eval(
    kb_id: str, body: FederatedEvalRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> federated_eval.FederatedEvalReport:
    """Per-grant mesh-quality attribution (FR-G9): score a federated golden with each peer
    ablated, so a peer whose content DEGRADES the joint answer is caught and tied to its grant.
    Admin only; federation must be enabled. Records a FEDERATED_EVAL audit event (+ a
    FEDERATION_QUALITY_ALERT per degrading peer)."""
    _require_federation(principal)
    _require_role(principal, "admin")
    check_query_rate(principal.tenant_id)
    questions = body.questions if body.questions is not None else await continuous_eval._load_golden(kb_id)
    if not questions:
        raise ValidationError("no golden questions: provide `questions` or store a golden set for this KB")
    caller_dept = kb_id.split("__", 1)[0]
    return await federated_eval.run_federated_eval(
        caller_dept, [kb_id], body.federate_from, list(questions), actor=principal.subject
    )


class BusinessValueRequest(BaseModel):
    questions: list[EvalQuestion] = Field(..., min_length=1, max_length=200, description="The workload.")
    baseline: business_value.BizBaseline | None = Field(
        default=None, description="The before the claim is measured against; omit for measurement only."
    )


@router.post("/kb/{kb_id}/business-value", response_model=business_value.BusinessValueReport)
async def run_business_value_bench(
    kb_id: str, body: BusinessValueRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> business_value.BusinessValueReport:
    """Business-value benchmark (FR-H4): run a workload against this KB, MEASURE cost-per-answer,
    latency (p50/p95/p99) and quality (grounded/hit/faithfulness), and compare to a supplied
    baseline - turning the proposal estimates into measured numbers. Admin only; every report
    carries its own caveats (indicative rate card, productivity proxy, single-node)."""
    _require_role(principal, "admin")
    check_query_rate(principal.tenant_id)
    return await business_value.run_business_value(kb_id, body.questions, body.baseline, actor=principal.subject)


# --- Continuous evaluation (FR-G9) -------------------------------------------
class EvalScheduleRequest(BaseModel):
    enabled: bool = False
    interval_hours: int = Field(default=24, ge=1, le=720)
    min_grounded_rate: float = Field(default=0.6, ge=0.0, le=1.0)
    min_faithfulness: float = Field(default=0.5, ge=0.0, le=1.0)
    federate_from: list[str] = Field(default_factory=list, description="Peers to evaluate across the mesh (FR-G9).")


class EvalRunsResponse(BaseModel):
    enabled: bool
    schedule: EvalSchedule | None = None
    runs: list[dict[str, Any]] = Field(default_factory=list)


@router.get("/kb/{kb_id}/eval/schedule", response_model=EvalRunsResponse)
async def get_eval_schedule(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> EvalRunsResponse:
    """The KB's continuous-evaluation enrolment + recent runs/alerts (FR-G9)."""
    sched = await wiring.eval_schedule_store().get(kb_id)
    runs = await continuous_eval.recent_runs(kb_id)
    return EvalRunsResponse(enabled=get_settings().continuous_eval_enabled, schedule=sched, runs=runs)


@router.put("/kb/{kb_id}/eval/schedule", response_model=EvalSchedule)
async def set_eval_schedule(
    kb_id: str, body: EvalScheduleRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> EvalSchedule:
    """Enrol/update this KB in continuous evaluation (admin). The scheduler must also be
    enabled platform-side (RAG_CONTINUOUS_EVAL_ENABLED) for scheduled runs to fire."""
    _require_role(principal, "admin")
    existing = await wiring.eval_schedule_store().get(kb_id)
    sched = EvalSchedule(
        kb_id=kb_id,
        enabled=body.enabled,
        interval_hours=body.interval_hours,
        min_grounded_rate=body.min_grounded_rate,
        min_faithfulness=body.min_faithfulness,
        federate_from=body.federate_from,
        last_run_at=existing.last_run_at if existing else None,
    )
    await wiring.eval_schedule_store().put(sched)
    _log.info("eval_schedule_set", kb_id=kb_id, enabled=body.enabled, interval_hours=body.interval_hours)
    return sched


@router.post("/kb/{kb_id}/eval/run-now", response_model=continuous_eval.EvalRunResult)
async def run_eval_now(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> continuous_eval.EvalRunResult:
    """Trigger a continuous-eval run immediately (golden set + sampled live traffic),
    independent of the schedule. Writer-gated; result also lands on the audit trail."""
    _require_role(principal, "ingestor")
    check_query_rate(principal.tenant_id)
    sched = await wiring.eval_schedule_store().get(kb_id) or EvalSchedule(
        kb_id=kb_id,
        min_grounded_rate=get_settings().continuous_eval_min_grounded_rate,
        min_faithfulness=get_settings().continuous_eval_min_faithfulness,
    )
    return await continuous_eval.run_once(kb_id, sched, actor=principal.subject)


# --- Embeddable chat widget (FR-I6) ------------------------------------------
@router.get("/kb/{kb_id}/widget/config", response_model=widget.WidgetConfig)
async def get_widget_config(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> widget.WidgetConfig:
    """The KB's embeddable-widget theming (disabled by default)."""
    return await widget.get_widget_config(kb_id)


@router.put("/kb/{kb_id}/widget/config", response_model=widget.WidgetConfig)
async def set_widget_config(
    kb_id: str, body: widget.WidgetConfig, principal: Annotated[Principal, Depends(tenant_guard)]
) -> widget.WidgetConfig:
    """Enable + white-label the widget for this KB (admin)."""
    _require_role(principal, "admin")
    return await widget.set_widget_config(kb_id, body)


class WidgetSessionRequest(BaseModel):
    kb_id: str = Field(..., description="The KB this widget answers from (must be owned by the caller).")


class WidgetSessionResponse(BaseModel):
    token: str = Field(description="Short-lived, reader-scoped, single-KB JWT for the browser widget.")
    expires_in: int
    kb_id: str
    config: widget.WidgetConfig


@router.post("/widget/session", response_model=WidgetSessionResponse)
async def create_widget_session(
    body: WidgetSessionRequest, principal: Annotated[Principal, Depends(get_principal)]
) -> WidgetSessionResponse:
    """Exchange a read-scoped API key (held by the host page's SERVER) for a short-lived JWT
    the browser widget can use (FR-I6). The token is bound to exactly one KB, reader-only, and
    public-ACL — the API key itself never reaches the browser."""
    if not principal.has_role("reader"):
        raise ForbiddenError("widget sessions require a read-scoped credential")
    if not widget.owns_kb(principal.tenant_id, body.kb_id):
        raise TenantIsolationError("credential may not open a widget for this knowledge base", details={})
    config = await widget.get_widget_config(body.kb_id)
    if not config.enabled:
        raise NotFoundError("the widget is not enabled for this knowledge base")
    ttl = get_settings().widget_session_ttl_seconds
    # tenant_id == kb_id binds the token to this ONE KB; empty groups => public content only.
    token = issue_dev_jwt(
        f"widget:{body.kb_id}", body.kb_id, roles=["reader"], attrs={"groups": [], "widget": True}, ttl_seconds=ttl
    )
    _log.info("widget_session_issued", kb_id=body.kb_id, actor=principal.subject, ttl=ttl)
    return WidgetSessionResponse(token=token, expires_in=ttl, kb_id=body.kb_id, config=config)


# --- Chat-platform bots (FR-I7) ----------------------------------------------
@router.get("/kb/{kb_id}/bot/config", response_model=bots.BotConfigView)
async def get_bot_config(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> bots.BotConfigView:
    """The KB's chat-bot binding (never returns the shared token)."""
    return await bots.get_bot_config(kb_id)


@router.put("/kb/{kb_id}/bot/config", response_model=bots.BotConfigView)
async def set_bot_config(
    kb_id: str, body: bots.BotConfig, principal: Annotated[Principal, Depends(tenant_guard)]
) -> bots.BotConfigView:
    """Enable/configure the chat bot for this KB (admin). Set the token separately."""
    _require_role(principal, "admin")
    return await bots.set_bot_config(kb_id, body)


class BotTokenResponse(BaseModel):
    token: str = Field(description="Shared webhook token — shown ONCE; paste it into the platform's outgoing webhook.")
    webhook_path: str


@router.post("/kb/{kb_id}/bot/token", response_model=BotTokenResponse)
async def rotate_bot_token(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> BotTokenResponse:
    """Generate (or rotate) the bot's shared token and enable it (admin). Shown once."""
    _require_role(principal, "admin")
    token = await bots.rotate_token(kb_id)
    return BotTokenResponse(token=token, webhook_path=f"/v1/bots/mattermost/{kb_id}")


@router.post("/bots/mattermost/{kb_id}", response_model=bots.MattermostResponse)
async def mattermost_bot(kb_id: str, request: Request) -> bots.MattermostResponse:
    """Mattermost outgoing-webhook endpoint (FR-I7). Authenticated by the shared token in the
    payload; answers grounded-or-abstain SYNCHRONOUSLY (no outbound call → sovereign)."""
    ctype = request.headers.get("content-type", "")
    if "application/json" in ctype:
        payload = bots.MattermostInbound.model_validate(await request.json())
    else:  # Mattermost default is application/x-www-form-urlencoded
        form = await request.form()
        payload = bots.MattermostInbound.model_validate({k: str(v) for k, v in form.items()})
    if not await bots.verify_token(kb_id, payload.token):
        raise UnauthorizedError("invalid or disabled bot token")
    check_query_rate(kb_id)
    cfg = await bots.get_bot_config(kb_id)
    question = bots.strip_trigger(payload.text, payload.trigger_word or cfg.trigger)
    if not question:
        return bots.MattermostResponse(text="Ask me a question about the knowledge base.")
    rq = RetrievalQuery(tenant_id=kb_id, text=question, top_k=5, acl=[])
    answer = await pipeline.answer_query(rq, actor=f"bot:{payload.user_name or 'mattermost'}")
    _log.info("bot_answer", kb_id=kb_id, status=answer.status.value, user=payload.user_name)
    return bots.MattermostResponse(text=bots.format_answer(answer))


# --- Federated ask: one question across several of the org's KBs -------------
class AskAcrossRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)  # see QueryRequest.query
    kb_ids: list[str] = Field(default_factory=list, description="KBs to search; empty = ALL the org's KBs.")
    top_k: int = Field(
        default_factory=lambda: get_settings().query_top_k, ge=1, le=50
    )  # recall floor, measured on the golden set — see Settings.query_top_k
    model: str | None = Field(default=None, description="Per-question model override.")
    federate_from: list[str] = Field(
        default_factory=list,
        description=(
            "Federated RAG: foreign department ids to also query. Each is included ONLY if it "
            "has published an active grant to this department; un-granted departments are "
            "skipped silently. Requires RAG_FEDERATION_ENABLED."
        ),
    )


@router.post("/ask", response_model=GroundedAnswer)
async def ask_across(body: AskAcrossRequest, principal: Annotated[Principal, Depends(get_principal)]) -> GroundedAnswer:
    """Grounded-or-abstain answer federated across the org's knowledge bases. Citations
    are prefixed with the knowledge base they came from.

    With ``federate_from`` (and RAG_FEDERATION_ENABLED), the answer also draws on foreign
    departments' KBs that have granted this department access (Federated RAG) — grant-gated,
    least-privilege (public content only), and fully audited."""
    _require_member(principal)  # FR-RBAC-01: a no-role token cannot ask, even the shared KB
    check_query_rate(principal.tenant_id)
    org = principal.tenant_id
    owned = {k.kb_id for k in await list_knowledge_bases(principal)}
    kb_ids = body.kb_ids or sorted(owned)
    foreign = [k for k in kb_ids if k not in owned]
    if foreign:
        raise TenantIsolationError("kb_ids outside your organisation", details={"kb_ids": foreign})

    if body.federate_from:
        if not get_settings().federation_enabled:
            raise ForbiddenError("federation is not enabled", details={"flag": "federation_enabled"})
        return await pipeline.answer_federated(
            org,
            kb_ids,
            body.federate_from,
            body.query,
            top_k=body.top_k,
            acl=_principal_groups(principal),
            clearance=_principal_clearance(principal),
            actor=principal.subject,
            model=body.model,
        )

    return await pipeline.answer_across(
        org,
        kb_ids,
        body.query,
        top_k=body.top_k,
        acl=_principal_groups(principal),
        clearance=_principal_clearance(principal),
        actor=principal.subject,
        model=body.model,
    )


# --- Federated RAG federation grants (cross-tenant P2P) ------------------------
# A grant is the ONLY mechanism that opens the hard tenant-isolation wall: an issuer
# department cryptographically authorises a subject department to query specific topics
# from specific KBs. Feature-gated (federation_enabled); admin/owner only to issue/revoke.


class GrantCreateRequest(BaseModel):
    subject_dept_id: str = Field(..., min_length=1, description="Department granted read access.")
    topics: list[str] = Field(..., min_length=1, description="Topics the subject may query.")
    kb_ids: list[str] = Field(..., min_length=1, description="KBs (of the issuer) exposed for these topics.")
    max_results_per_query: int = Field(default=5, ge=1, le=20)
    ttl_days: int = Field(default=180, ge=1, le=3650, description="Grant validity window from now.")


class GrantView(BaseModel):
    grant_id: str
    issuer_dept_id: str
    subject_dept_id: str
    topics: list[str]
    kb_ids: list[str]
    max_results_per_query: int
    not_before: datetime
    expires_at: datetime
    revoked_at: datetime | None
    active: bool
    parent_grant_id: str | None = None


def _grant_view(g: Grant) -> GrantView:
    return GrantView(
        grant_id=g.grant_id,
        issuer_dept_id=g.issuer_dept_id,
        subject_dept_id=g.subject_dept_id,
        topics=g.topics,
        kb_ids=g.kb_ids,
        max_results_per_query=g.max_results_per_query,
        not_before=g.not_before,
        expires_at=g.expires_at,
        revoked_at=g.revoked_at,
        active=g.is_active,
        parent_grant_id=g.parent_grant_id,
    )


def _require_federation(principal: Principal) -> None:
    if not get_settings().federation_enabled:
        raise ForbiddenError("federation is not enabled", details={"flag": "federation_enabled"})


@router.post("/grants", response_model=GrantView, status_code=201)
async def create_grant(body: GrantCreateRequest, principal: Annotated[Principal, Depends(get_principal)]) -> GrantView:
    """Publish a cross-department grant. The issuer is the caller's tenant; only an
    owner/admin can issue. The grant's KBs must belong to the issuer, and the subject
    cannot be the issuer itself. Signed with the issuer's Ed25519 key, then audited."""
    from svc_control_plane.grants import current_key_id, sign_grant  # noqa: PLC0415

    _require_federation(principal)
    if not (principal.has_role("owner") or principal.has_role("admin")):
        raise ForbiddenError("only an owner/admin can issue federation grants")

    issuer = principal.tenant_id
    if body.subject_dept_id == issuer:
        raise ValidationError("subject_dept_id cannot be the issuer (self-grant is meaningless)")
    owned = {k.kb_id for k in await list_knowledge_bases(principal)}
    foreign = [k for k in body.kb_ids if k not in owned]
    if foreign:
        raise TenantIsolationError("kb_ids outside your organisation", details={"kb_ids": foreign})

    # Whole-second precision: the Go SovereignNode's canonical-claims formatter
    # (grant.go isoUTC) always renders timestamps as "...T15:04:05+00:00" with NO
    # fractional seconds, by design/documented assumption (see the CRITICAL interop
    # note there). datetime.now(UTC) carries microseconds, which would make Python's
    # _canonical_claims() sign a DIFFERENT byte string than what the Go responder
    # recomputes when verifying — an always-fails Ed25519 signature check. Truncate
    # here so both sides sign/verify the identical whole-second ISO-8601 string.
    now = datetime.now(UTC).replace(microsecond=0)
    grant = Grant(
        grant_id=f"grant_{uuid.uuid4().hex[:16]}",
        issuer_dept_id=issuer,
        subject_dept_id=body.subject_dept_id,
        topics=body.topics,
        kb_ids=body.kb_ids,
        max_results_per_query=body.max_results_per_query,
        not_before=now,
        expires_at=now + timedelta(days=body.ttl_days),
        signed_token="",  # filled below
        created_at=now,
    )
    grant = grant.model_copy(update={"signed_token": sign_grant(grant)})
    await wiring.grant_store().create(grant)
    await wiring.audit_service().record(
        tenant_id=issuer,
        trace_id=str(uuid.uuid4()),
        action=AuditAction.GRANT_ISSUE,
        actor=principal.subject,
        payload={
            "grant_id": grant.grant_id,
            "subject_dept_id": grant.subject_dept_id,
            "topics": grant.topics,
            "kb_ids": grant.kb_ids,
            "expires_at": grant.expires_at.isoformat(),
            "signing_key_id": current_key_id(),
        },
    )
    _log.info("grant_issued", grant_id=grant.grant_id, issuer=issuer, subject=grant.subject_dept_id)
    return _grant_view(grant)


@router.get("/grants", response_model=list[GrantView])
async def list_issued_grants(principal: Annotated[Principal, Depends(get_principal)]) -> list[GrantView]:
    """List grants this department has PUBLISHED (including revoked/expired)."""
    _require_federation(principal)
    grants = await wiring.grant_store().list_issued(principal.tenant_id)
    return [_grant_view(g) for g in grants]


@router.get("/grants/received", response_model=list[GrantView])
async def list_received_grants(principal: Annotated[Principal, Depends(get_principal)]) -> list[GrantView]:
    """List ACTIVE grants OTHER departments have published FOR this department."""
    _require_federation(principal)
    grants = await wiring.grant_store().list_received(principal.tenant_id)
    return [_grant_view(g) for g in grants]


@router.delete("/grants/{grant_id}", status_code=204)
async def revoke_grant(grant_id: str, principal: Annotated[Principal, Depends(get_principal)]) -> None:
    """Revoke a grant this department issued. Only the issuer (owner/admin) can revoke."""
    _require_federation(principal)
    if not (principal.has_role("owner") or principal.has_role("admin")):
        raise ForbiddenError("only an owner/admin can revoke federation grants")
    try:
        found = await wiring.grant_store().revoke(grant_id, principal.tenant_id)
    except PermissionError as exc:
        raise ForbiddenError("only the issuer can revoke this grant") from exc
    if not found:
        raise NotFoundError("grant not found", details={"grant_id": grant_id})
    await wiring.audit_service().record(
        tenant_id=principal.tenant_id,
        trace_id=str(uuid.uuid4()),
        action=AuditAction.GRANT_REVOKE,
        actor=principal.subject,
        payload={"grant_id": grant_id},
    )
    _log.info("grant_revoked", grant_id=grant_id, issuer=principal.tenant_id)


class GrantDelegateRequest(BaseModel):
    subject_dept_id: str = Field(..., min_length=1, description="Department to RE-DELEGATE to.")
    topics: list[str] = Field(..., min_length=1, description="Subset of the parent grant's topics.")
    kb_ids: list[str] = Field(..., min_length=1, description="Subset of the parent grant's kb_ids.")
    max_results_per_query: int = Field(default=5, ge=1, le=20)
    ttl_days: int = Field(default=180, ge=1, le=3650, description="Validity window (bounded by the parent).")


@router.post("/grants/{grant_id}/delegate", response_model=GrantView, status_code=201)
async def delegate_grant(
    grant_id: str, body: GrantDelegateRequest, principal: Annotated[Principal, Depends(get_principal)]
) -> GrantView:
    """Re-delegate a grant this department RECEIVED (transitive A->B->C). The caller must be the
    parent grant's subject; the new grant may only NARROW it (topics/kb_ids subset, cap and
    validity no larger) and the chain may not exceed ``federation_max_grant_depth``. Revoking
    any ancestor later kills this grant too (transitive revocation)."""
    from svc_control_plane.grants import DelegationError, attenuate, current_key_id, sign_grant  # noqa: PLC0415

    _require_federation(principal)
    if not (principal.has_role("owner") or principal.has_role("admin")):
        raise ForbiddenError("only an owner/admin can delegate federation grants")
    store = wiring.grant_store()
    parent = await store.get(grant_id)
    if parent is None:
        raise NotFoundError("grant not found", details={"grant_id": grant_id})
    if parent.subject_dept_id != principal.tenant_id:
        raise ForbiddenError("only the grant's subject can re-delegate it")
    if not parent.is_active:
        raise ValidationError("cannot delegate a revoked or expired grant")
    if body.subject_dept_id == principal.tenant_id:
        raise ValidationError("subject_dept_id cannot be the delegator itself")
    max_depth = get_settings().federation_max_grant_depth
    parent_depth = len(await store.resolve_chain(grant_id))
    if parent_depth + 1 > max_depth:
        raise ValidationError(
            f"delegation would exceed the max chain depth ({max_depth})",
            details={"parent_depth": parent_depth, "max_depth": max_depth},
        )

    # Whole-second precision — see the matching comment in create_grant: the Go
    # responder's canonical-claims formatter always drops fractional seconds, so
    # signing with a microsecond-precision "now" produces a signature the Go side can
    # never reproduce when re-verifying.
    now = datetime.now(UTC).replace(microsecond=0)
    # Clamp the child window INSIDE the parent: a delegation inherits the parent ceiling, so a
    # same-TTL re-share is bounded by the parent rather than rejected for outliving it by a hair.
    not_before = max(now, parent.not_before)
    expires_at = min(now + timedelta(days=body.ttl_days), parent.expires_at)
    child = Grant(
        grant_id=f"grant_{uuid.uuid4().hex[:16]}",
        issuer_dept_id=principal.tenant_id,
        subject_dept_id=body.subject_dept_id,
        topics=body.topics,
        kb_ids=body.kb_ids,
        max_results_per_query=body.max_results_per_query,
        not_before=not_before,
        expires_at=expires_at,
        signed_token="",
        created_at=now,
        parent_grant_id=grant_id,
    )
    try:
        attenuate(parent, child)  # bounded trust: the child may only narrow the parent
    except DelegationError as exc:
        raise ValidationError(str(exc)) from exc
    child = child.model_copy(update={"signed_token": sign_grant(child)})
    await store.create(child)
    await wiring.audit_service().record(
        tenant_id=principal.tenant_id,
        trace_id=str(uuid.uuid4()),
        action=AuditAction.GRANT_ISSUE,
        actor=principal.subject,
        payload={
            "grant_id": child.grant_id,
            "subject_dept_id": child.subject_dept_id,
            "topics": child.topics,
            "kb_ids": child.kb_ids,
            "expires_at": child.expires_at.isoformat(),
            "parent_grant_id": grant_id,
            "delegation_depth": parent_depth + 1,
            "signing_key_id": current_key_id(),
        },
    )
    _log.info("grant_delegated", grant_id=child.grant_id, parent=grant_id, issuer=principal.tenant_id)
    return _grant_view(child)


@router.get("/grants/{grant_id}/export", response_model=compliance.GrantLineageExport)
async def export_grant_lineage(
    grant_id: str, principal: Annotated[Principal, Depends(get_principal)]
) -> compliance.GrantLineageExport:
    """Per-grant compliance export (FR-H8 federation edge): the consent lineage (A->B->C), the
    transitive validity, and the audit-sourced record of who answered under it. Available to the
    grant's issuer or subject department (either party to the consent), or an owner/admin."""
    _require_federation(principal)
    grant = await wiring.grant_store().get(grant_id)
    if grant is None:
        raise NotFoundError("grant not found", details={"grant_id": grant_id})
    is_party = principal.tenant_id in {grant.issuer_dept_id, grant.subject_dept_id}
    if not (is_party or principal.has_role("owner") or principal.has_role("admin")):
        raise ForbiddenError("only a party to this grant (issuer/subject) or an admin can export it")
    return await compliance.grant_export(grant_id)


# --- Folder sync (FR-A2b batch): ingest every object under a bucket prefix ---
class SyncRequest(BaseModel):
    external_source: ExternalObjectSource
    prefix: str = Field(default="", description="Only keys under this prefix are ingested.")
    mime: str = Field(default="application/pdf", description="MIME applied to keys without a known extension.")
    strategy: ChunkStrategy | None = None
    limit: int = Field(
        default=50,
        ge=1,
        le=1000,
        description=(
            "Max objects THIS call ingests. The sync resumes from where it stopped, so a bucket "
            "larger than this is finished by running it again (or on the schedule) rather than by "
            "raising the cap — one run should not hold an hour of extraction open."
        ),
    )


class SyncItem(BaseModel):
    doc_id: str
    ok: bool
    chunks: int = 0
    outcome: str = "created"  # created | updated | unchanged (FR-A8 delta)
    error: str | None = None


class SyncResponse(BaseModel):
    found: int
    ingested: int
    failed: int
    unchanged: int = 0  # skipped by delta (content unchanged) — FR-A8
    updated: int = 0
    # True when the bucket holds more objects than this run took. The caller MUST be told:
    # a partial sync that reports only "found: 200" is indistinguishable from a complete one,
    # which is how a 2000-object bucket could silently index a tenth of itself.
    more: bool = False
    items: list[SyncItem]


_EXT_MIME = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".html": "text/html",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


@router.post("/kb/{kb_id}/ingest/sync", response_model=SyncResponse)
async def ingest_sync(
    kb_id: str, body: SyncRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> SyncResponse:
    """Batch-ingest every object under a prefix of an external bucket (folder sync).

    Sovereignty-gated like any A2b pull; per-file failures are reported, not fatal."""
    _require_role(principal, "ingestor")
    check_ingest_rate(principal.tenant_id)
    src = body.external_source
    assert_sovereign(f"{'https' if src.secure else 'http'}://{src.endpoint}")
    # Provider-aware, like every other external pull. This endpoint carried its own inline copy
    # of the S3 credential lookup -- the THIRD copy of the same four lines -- so an Azure sync
    # asked for {ref}_access_key, a name Azure does not have.
    access_key, secret_key = external_credentials(src, get_secrets_provider())
    # Resume where the last sync of this KB stopped, and share the cursor with the scheduled
    # runner so the two drain ONE bucket together instead of each restarting at the top.
    cursor = await sync_jobs.get_cursor(kb_id)
    keys = await external_fetcher_for(getattr(src, "provider", "s3")).list_keys(
        endpoint=src.endpoint,
        bucket=src.bucket,
        prefix=body.prefix,
        access_key=access_key,
        secret_key=secret_key,
        secure=src.secure,
        limit=body.limit,
        start_after=cursor,
    )
    more = len(keys) >= body.limit
    await sync_jobs.set_cursor(kb_id, keys[-1] if more and keys else "")

    items: list[SyncItem] = []
    for key in keys:
        try:
            await check_document_cap(kb_id)
            ext = "." + key.rsplit(".", 1)[-1].lower() if "." in key else ""
            doc = DocumentEnvelope(
                tenant_id=kb_id,
                doc_id=key,
                source_id="api-folder-sync",
                source_kind=SourceKind.OBJECT_STORE,
                mime=_EXT_MIME.get(ext, body.mime),
                uri=key,
                external_source=src,
                ingested_at=datetime.now(UTC),
            )
            outcome = await pipeline.ingest_document(
                doc,
                actor=principal.subject,
                strategy=body.strategy,
                clearance=_principal_clearance(principal),
                pii_policy=_resolve_pii_policy(None),  # folder sync uses the platform default when enabled
            )
            items.append(SyncItem(doc_id=key, ok=True, chunks=outcome.chunk_count, outcome=outcome.outcome))
        except PlatformError as exc:
            items.append(SyncItem(doc_id=key, ok=False, error=exc.message))
        except Exception as exc:
            items.append(SyncItem(doc_id=key, ok=False, error=str(exc)[:160]))
    ok = sum(1 for i in items if i.ok)
    unchanged = sum(1 for i in items if i.outcome == "unchanged")
    updated = sum(1 for i in items if i.outcome == "updated")
    # Remember this source so it can be re-synced on a schedule (FR-A8) without re-specifying it.
    await sync_jobs.set_sync_source(
        kb_id,
        sync_jobs.SyncSource(
            external_source=src,
            prefix=body.prefix,
            mime=body.mime,
            # Resolved AT ENQUEUE, not at run: a queued job records the strategy it will use,
            # so a config change mid-flight cannot silently re-chunk a half-finished batch.
            strategy=(body.strategy or ChunkStrategy(get_settings().chunk_strategy)).value,
            limit=body.limit,
        ),
    )
    _log.info("folder_sync_done", kb_id=kb_id, found=len(keys), ingested=ok, unchanged=unchanged)
    return SyncResponse(
        found=len(keys),
        ingested=ok,
        failed=len(items) - ok,
        unchanged=unchanged,
        updated=updated,
        more=more,
        items=items,
    )


# --- Web crawler (FR-A9) -----------------------------------------------------
class CrawlRequest(BaseModel):
    seed_url: str = Field(..., description="Starting URL; its host must be egress-allowlisted (sovereign).")
    max_depth: int = Field(default=1, ge=0, le=3)
    max_pages: int = Field(default=20, ge=1, le=100)
    same_host_only: bool = True
    path_prefix: str | None = Field(default=None, description="Restrict to URLs under this path, e.g. /docs.")
    respect_robots: bool = True


@router.post("/kb/{kb_id}/ingest/crawl", response_model=crawl_jobs.CrawlResult)
async def crawl_website(
    kb_id: str, body: CrawlRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> crawl_jobs.CrawlResult:
    """Crawl a website from a seed URL and ingest each in-scope page (FR-A9). Bounded by
    depth/page limits, respects robots.txt, and every hop is sovereignty-gated."""
    _require_role(principal, "ingestor")
    check_ingest_rate(principal.tenant_id)
    return await crawl_jobs.crawl_site(
        kb_id,
        body.seed_url,
        actor=principal.subject,
        max_depth=body.max_depth,
        max_pages=body.max_pages,
        same_host_only=body.same_host_only,
        path_prefix=body.path_prefix,
        respect_robots=body.respect_robots,
    )


# --- Async bulk ingestion (job queue for 1000+ docs) -------------------------
class CreateJobRequest(BaseModel):
    pii_autoclassify: bool | None = Field(
        default=None, description="Enable the PII-driven classification floor for this batch (see /ingest)."
    )
    classification: str | int = Field(
        default=0,
        description="Data classification for every doc in this batch: 0/public..3/restricted. Capped at the "
        "ingestor's clearance (mirrors the direct /ingest cap).",
    )


@router.post("/kb/{kb_id}/ingest/jobs", response_model=ingest_jobs.IngestJob)
async def create_ingest_job(
    kb_id: str,
    principal: Annotated[Principal, Depends(tenant_guard)],
    body: CreateJobRequest | None = None,
) -> ingest_jobs.IngestJob:
    """Start a bulk-ingestion job. Add documents to it, then start it — the worker ingests them
    in the background so the request never blocks (FR: volume ingestion). The PII-floor choice is
    captured here (the async worker has no request principal) and applied to every doc in the batch."""
    _require_role(principal, "ingestor")
    toggle = body.pii_autoclassify if body else None
    # Per-KB role elevation applies to the whole batch (captured now for the async worker).
    clearance = await kb_roles.effective_kb_clearance(get_catalog(), kb_id, principal)
    requested_cls = parse_classification(body.classification if body else 0)
    # Same cap as the direct /ingest path (can_ingest_at): a batch classification above the
    # ingestor's clearance is rejected — you can't publish content more secret than you may read.
    if requested_cls > clearance:
        raise ForbiddenError(
            "your clearance may not ingest documents at this classification into this KB",
            details={"kb_id": kb_id, "requested_classification": requested_cls, "clearance": clearance},
        )
    # The shared KB is company-wide readable — it may only ever hold Public content (mirrors /ingest).
    if kb_id == SHARED_KB_ID and requested_cls > 0:
        raise ForbiddenError(
            "the shared KB is company-wide readable and may only hold Public (0) content",
            details={"kb_id": kb_id, "requested_classification": requested_cls},
        )
    return await ingest_jobs.create_job(
        wiring.ingest_job_store(),
        kb_id,
        principal.subject,
        clearance=clearance,
        pii_policy=_resolve_pii_policy(toggle),
        classification=requested_cls,
    )


class BulkDocument(BaseModel):
    doc_id: str = Field(..., min_length=1)
    content_b64: str = Field(..., description="Base64 file bytes; staged to the object store.")
    mime: str = "application/pdf"
    strategy: ChunkStrategy | None = None


@router.post("/kb/{kb_id}/ingest/jobs/{job_id}/items", status_code=204)
async def add_ingest_job_item(
    kb_id: str, job_id: str, body: BulkDocument, principal: Annotated[Principal, Depends(tenant_guard)]
) -> None:
    """Stage one file into a building job (fast — bytes are stored, not yet processed)."""
    _require_role(principal, "ingestor")
    await ingest_jobs.add_document(
        wiring.ingest_job_store(),
        job_id,
        doc_id=body.doc_id,
        content_b64=body.content_b64,
        mime=body.mime,
        # Resolved AT ENQUEUE, not at run: a queued job records the strategy it will use,
        # so a config change mid-flight cannot silently re-chunk a half-finished batch.
        strategy=(body.strategy or ChunkStrategy(get_settings().chunk_strategy)).value,
    )


@router.post("/kb/{kb_id}/ingest/jobs/{job_id}/items/file", status_code=204)
async def add_ingest_job_item_file(
    kb_id: str,
    job_id: str,
    principal: Annotated[Principal, Depends(tenant_guard)],
    file: Annotated[UploadFile, File(description="Raw file bytes (multipart) — NOT base64.")],
    doc_id: Annotated[str, Form()],
    mime: Annotated[str, Form()] = "application/pdf",
    strategy: Annotated[str | None, Form()] = None,
) -> None:
    """Stage one file into a building job by STREAMING raw binary (no base64) — the large-file
    path. The file goes straight to the object store (the job's existing staging area), so the
    unchanged worker ingests it by reference. Avoids the ~1.37x base64 bloat and holding the whole
    file in memory, so it scales past the base64 upload ceiling."""
    _require_role(principal, "ingestor")
    f = file.file
    f.seek(0, 2)  # measure length without reading the (possibly multi-GB, disk-spooled) body in
    length = f.tell()
    f.seek(0)
    await ingest_jobs.add_document_stream(
        wiring.ingest_job_store(),
        job_id,
        doc_id=doc_id,
        stream=f,
        length=length,
        mime=mime,
        strategy=(ChunkStrategy(strategy) if strategy else ChunkStrategy(get_settings().chunk_strategy)).value,
    )


@router.post("/kb/{kb_id}/ingest/jobs/{job_id}/start", response_model=ingest_jobs.IngestJob)
async def start_ingest_job(
    kb_id: str, job_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> ingest_jobs.IngestJob:
    """Queue the job — the background worker picks it up and processes every document."""
    _require_role(principal, "ingestor")
    return await ingest_jobs.start_job(wiring.ingest_job_store(), job_id)


@router.get("/kb/{kb_id}/ingest/jobs", response_model=list[ingest_jobs.IngestJob])
async def list_ingest_jobs(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> list[ingest_jobs.IngestJob]:
    """Recent bulk-ingestion jobs for this KB (newest first)."""
    return await wiring.ingest_job_store().list_for_kb(kb_id, 20)


@router.get("/kb/{kb_id}/ingest/jobs/{job_id}", response_model=ingest_jobs.JobView)
async def get_ingest_job(
    kb_id: str, job_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> ingest_jobs.JobView:
    """Live progress for a bulk-ingestion job (done/failed/total + failure reasons)."""
    return await ingest_jobs.job_view(wiring.ingest_job_store(), job_id)


@router.post("/kb/{kb_id}/ingest/jobs/{job_id}/retry", response_model=ingest_jobs.IngestJob)
async def retry_ingest_job(
    kb_id: str, job_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> ingest_jobs.IngestJob:
    """Re-queue only the failed documents of a job."""
    _require_role(principal, "ingestor")
    store = wiring.ingest_job_store()
    if await store.retry_failed(job_id) == 0:
        raise ValidationError("no failed documents to retry")
    job = await store.get(job_id)
    if job is None:
        raise NotFoundError("ingest job not found", details={"job_id": job_id})
    return job


@router.post("/kb/{kb_id}/ingest/jobs/{job_id}/cancel", status_code=204)
async def cancel_ingest_job(kb_id: str, job_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> None:
    """Cancel a job — in-flight items finish, the rest are skipped."""
    _require_role(principal, "ingestor")
    await wiring.ingest_job_store().set_status(job_id, "cancelled")


# --- Scheduled connector sync (FR-A8) ----------------------------------------
class SyncScheduleRequest(BaseModel):
    enabled: bool = False
    interval_hours: int = Field(default=24, ge=1, le=720)
    # Calendar-precise mode (banking pack): a 5-field cron ("0 2 * * 1" = Mon 02:00) evaluated in
    # `timezone`. When set, it takes precedence over interval_hours.
    cron: str = ""
    timezone: str = "UTC"


class SyncSourceView(BaseModel):
    """The captured folder source, so the console can SHOW what is connected.

    Non-secret fields only. ``credentials_ref`` is a secret NAME the operator typed, never a
    credential -- the platform resolves it server-side and the value never leaves the process.
    """

    provider: str
    endpoint: str
    bucket: str
    prefix: str
    secure: bool
    credentials_ref: str
    limit: int


class SyncScheduleView(BaseModel):
    enabled: bool
    interval_hours: int
    cron: str = ""
    timezone: str = "UTC"
    last_run_at: str | None = None
    next_run_at: str | None = None
    source_configured: bool
    # What is connected, not merely THAT something is. A bare boolean left the operator with no
    # way to see the bucket they had just added, or to notice they had pointed a KB at the wrong
    # one -- reported from the Connectors page.
    source: SyncSourceView | None = None
    scheduler_on: bool


def _next_run_at(sched: sync_jobs.SyncSchedule | None) -> str | None:
    """Compute when the schedule fires next (UTC ISO), for display."""
    if sched is None or not sched.enabled:
        return None
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415

    if sched.cron:
        from svc_control_plane.cron import next_run  # noqa: PLC0415

        anchor = sched.last_run_at or datetime.now(UTC)
        try:
            return next_run(sched.cron, sched.timezone, anchor).astimezone(UTC).isoformat()
        except ValueError:
            return None
    base = sched.last_run_at or datetime.now(UTC)
    return (base + timedelta(hours=sched.interval_hours)).astimezone(UTC).isoformat()


@router.get("/kb/{kb_id}/sync/schedule", response_model=SyncScheduleView)
async def get_sync_schedule(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> SyncScheduleView:
    """This KB's scheduled-sync enrolment + whether a source has been captured (FR-A8)."""
    sched = await sync_jobs.sync_schedule_store().get(kb_id)
    source = await sync_jobs.get_sync_source(kb_id)
    return SyncScheduleView(
        enabled=bool(sched and sched.enabled),
        interval_hours=sched.interval_hours if sched else 24,
        cron=sched.cron if sched else "",
        timezone=sched.timezone if sched else "UTC",
        last_run_at=sched.last_run_at.isoformat() if sched and sched.last_run_at else None,
        next_run_at=_next_run_at(sched),
        source_configured=source is not None,
        source=(
            SyncSourceView(
                provider=getattr(source.external_source, "provider", "s3"),
                endpoint=source.external_source.endpoint,
                bucket=source.external_source.bucket,
                prefix=source.prefix,
                secure=source.external_source.secure,
                credentials_ref=source.external_source.credentials_ref,
                limit=source.limit,
            )
            if source is not None
            else None
        ),
        scheduler_on=get_settings().sync_scheduler_enabled,
    )


@router.put("/kb/{kb_id}/sync/schedule", response_model=SyncScheduleView)
async def set_sync_schedule(
    kb_id: str, body: SyncScheduleRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> SyncScheduleView:
    """Enrol this KB for scheduled delta re-sync (admin). A source must have been captured by a
    prior folder sync. Choose an hour interval OR a cron expression (evaluated in `timezone`),
    which re-runs the source re-indexing only what changed."""
    _require_role(principal, "admin")
    if body.enabled and await sync_jobs.get_sync_source(kb_id) is None:
        raise ValidationError("run a folder sync first so the source is captured, then enable the schedule")
    if body.cron:
        from svc_control_plane.cron import resolve_tz, validate_cron  # noqa: PLC0415

        try:
            validate_cron(body.cron)
        except ValueError as exc:
            raise ValidationError(f"invalid cron expression: {exc}", details={"cron": body.cron}) from exc
        if resolve_tz(body.timezone).key != (body.timezone or "UTC"):
            raise ValidationError(
                "unknown timezone (use an IANA name, e.g. Asia/Kolkata)", details={"tz": body.timezone}
            )
    store = sync_jobs.sync_schedule_store()
    await store.put(
        sync_jobs.SyncSchedule(
            kb_id=kb_id,
            enabled=body.enabled,
            interval_hours=body.interval_hours,
            cron=body.cron,
            timezone=body.timezone,
        )
    )
    # Anchor a cron schedule at enable time so it fires at the NEXT occurrence, never immediately.
    if body.enabled and body.cron:
        from datetime import UTC, datetime  # noqa: PLC0415

        await store.mark_run(kb_id, datetime.now(UTC))
    return await get_sync_schedule(kb_id, principal)


@router.post("/kb/{kb_id}/sync/run-now", response_model=sync_jobs.SyncResult)
async def run_sync_now(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> sync_jobs.SyncResult:
    """Re-run the KB's captured sync source immediately (delta-aware). Writer-gated."""
    _require_role(principal, "ingestor")
    check_ingest_rate(principal.tenant_id)
    source = await sync_jobs.get_sync_source(kb_id)
    if source is None:
        raise NotFoundError("no sync source captured for this KB — run a folder sync first")
    return await sync_jobs.run_sync(kb_id, source, actor=principal.subject)


async def _assert_doc_scope_visible(kb_id: str, doc_ids: Sequence[str], *, acl: Sequence[str], clearance: int) -> None:
    """Every named document must be one THIS caller may read, or the request is refused.

    Not a tenant-isolation boundary — that is already closed twice over before this runs:
    `tenant_guard` binds the request to the KB, and `RetrievalQuery.doc_ids` is a predicate
    applied INSIDE the KB's own namespace, so a foreign id narrows the search to nothing rather
    than reaching another tenant's data. What this adds is the two things that property does not
    give on its own:

    * a CLEAN REFUSAL instead of a silently empty answer. An unscoped typo would otherwise come
      back as a confident "the sources do not cover that", which is indistinguishable from the
      document genuinely not saying it.
    * no EXISTENCE ORACLE. The check is against the documents this caller may SEE — the same
      ACL-entitlement and clearance rule retrieval and `_answer_catalog` enforce — so naming a
      document that does not exist and naming a Restricted one the caller is not cleared for
      fail identically. Distinguishing them would let an unprivileged caller enumerate the
      titles of classified files by watching which ids are accepted.

    Only runs when a scope was actually supplied, so an ordinary unscoped query pays nothing.
    """
    if not doc_ids:
        return
    docs = await get_catalog().list_documents(kb_id)
    visible = {
        d.doc_id for d in docs if is_entitled(d.source_acl, list(acl)) and is_cleared(d.classification, clearance)
    }
    unknown = [d for d in dict.fromkeys(doc_ids) if d not in visible]
    if unknown:
        raise ValidationError(f"doc_ids: no such document in this knowledge base: {', '.join(sorted(unknown)[:10])}")


@router.post("/kb/{kb_id}/query", response_model=GroundedAnswer)
async def query(
    kb_id: str, body: QueryRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> GroundedAnswer:
    """Grounded-or-abstain answer with citations (TDD §5.2)."""
    check_query_rate(principal.tenant_id)
    await _require_published(kb_id, principal)  # block draft (unpublished) KBs (banking pack)
    acl = _principal_groups(principal)
    clearance = await kb_roles.effective_kb_clearance(get_catalog(), kb_id, principal)
    await _assert_doc_scope_visible(kb_id, body.doc_ids, acl=acl, clearance=clearance)
    rq = RetrievalQuery(
        tenant_id=kb_id,
        text=body.query,
        top_k=body.top_k,
        mode=body.mode,
        filters=body.filters,
        doc_ids=tuple(body.doc_ids),
        acl=acl,
        clearance=clearance,
        after=body.window_after(),
        before=body.before,
        freshness=body.freshness,
    )
    return await pipeline.answer_query(rq, actor=principal.subject, model=body.model)


@router.post("/kb/{kb_id}/query/stream")
async def query_stream(
    kb_id: str, body: QueryRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> StreamingResponse:
    """Grounded-or-abstain answer, streamed token-by-token as Server-Sent Events."""
    check_query_rate(principal.tenant_id)
    await _require_published(kb_id, principal)  # block draft (unpublished) KBs (banking pack)
    acl = _principal_groups(principal)
    clearance = await kb_roles.effective_kb_clearance(get_catalog(), kb_id, principal)
    await _assert_doc_scope_visible(kb_id, body.doc_ids, acl=acl, clearance=clearance)
    rq = RetrievalQuery(
        tenant_id=kb_id,
        text=body.query,
        top_k=body.top_k,
        mode=body.mode,
        filters=body.filters,
        doc_ids=tuple(body.doc_ids),
        acl=acl,
        clearance=clearance,
        after=body.window_after(),
        before=body.before,
        freshness=body.freshness,
    )
    actor = principal.subject

    async def _events() -> AsyncIterator[bytes]:
        try:
            async for event in pipeline.stream_answer(rq, actor=actor, model=body.model):
                yield f"data: {json.dumps(event)}\n\n".encode()
        except Exception as exc:
            _log.warning("stream_answer_failed", kb_id=kb_id, error=str(exc)[:200])
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)[:200]})}\n\n".encode()

    return StreamingResponse(_events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@router.post("/kb/{kb_id}/retrieve")
async def retrieve(
    kb_id: str, body: QueryRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> dict[str, Any]:
    """Retrieval-only (for agents): ranked cited spans, no generation (TDD §7)."""
    check_query_rate(principal.tenant_id)
    await _require_published(kb_id, principal)  # block draft (unpublished) KBs (banking pack)
    acl = _principal_groups(principal)
    clearance = await kb_roles.effective_kb_clearance(get_catalog(), kb_id, principal)
    await _assert_doc_scope_visible(kb_id, body.doc_ids, acl=acl, clearance=clearance)
    rq = RetrievalQuery(
        tenant_id=kb_id,
        text=body.query,
        top_k=body.top_k,
        mode=body.mode,
        filters=body.filters,
        doc_ids=tuple(body.doc_ids),
        acl=acl,
        clearance=clearance,
    )
    result = await (await wiring.retrieval_service_for(kb_id)).retrieve(rq)
    return result.model_dump()


# --- Conversational RAG (FR-I1) ---------------------------------------------
class ConversationInfo(BaseModel):
    conversation_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class MessageInfo(BaseModel):
    message_id: str
    role: str
    content: str
    trace_id: str | None = None
    status: str | None = None
    citations: list[Any] = Field(default_factory=list)
    faithfulness: float | None = None
    created_at: datetime


class ConversationDetail(BaseModel):
    conversation: ConversationInfo
    messages: list[MessageInfo]


class TurnRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)  # see QueryRequest.query
    top_k: int = Field(
        default_factory=lambda: get_settings().query_top_k, ge=1, le=50
    )  # recall floor, measured on the golden set — see Settings.query_top_k
    mode: RetrievalMode = RetrievalMode.HYBRID
    model: str | None = Field(default=None, description="Per-turn model override (same endpoint/creds).")
    # Recency-aware retrieval (FR-D9) — optional, default-off.
    after: datetime | None = None
    before: datetime | None = None
    recency_days: int | None = Field(default=None, ge=1, le=3650)
    freshness: float = Field(default=0.0, ge=0.0, le=1.0)

    def window_after(self) -> datetime | None:
        if self.recency_days is not None:
            return datetime.now(UTC) - timedelta(days=self.recency_days)
        return self.after


def _conv_info(conv: Any) -> ConversationInfo:
    return ConversationInfo(
        conversation_id=conv.conversation_id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


def _msg_info(msg: Any) -> MessageInfo:
    return MessageInfo(
        message_id=msg.message_id,
        role=msg.role,
        content=msg.content,
        trace_id=msg.trace_id,
        status=msg.status,
        citations=[c.model_dump(mode="json") for c in msg.citations],
        faithfulness=msg.faithfulness,
        created_at=msg.created_at,
    )


@router.get("/kb/{kb_id}/conversations", response_model=list[ConversationInfo])
async def list_conversations(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)], limit: int = 50
) -> list[ConversationInfo]:
    """The KB's conversation threads, most-recently-active first."""
    convs = await wiring.conversation_store().recent(kb_id, limit)
    return [_conv_info(c) for c in convs]


@router.post("/kb/{kb_id}/conversations", status_code=201, response_model=ConversationInfo)
async def create_conversation(
    kb_id: str, body: TurnRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> ConversationInfo:
    """Open a new conversation, titled from its first question (no answer yet)."""
    conv = await pipeline.start_conversation(kb_id, body.query)
    return _conv_info(conv)


@router.get("/kb/{kb_id}/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    kb_id: str, conversation_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> ConversationDetail:
    """Full thread — the conversation plus every message in order."""
    store = wiring.conversation_store()
    conv = await store.get(kb_id, conversation_id)
    if conv is None:
        raise NotFoundError("conversation not found", details={"conversation_id": conversation_id})
    msgs = await store.messages(kb_id, conversation_id, 500)
    return ConversationDetail(conversation=_conv_info(conv), messages=[_msg_info(m) for m in msgs])


@router.delete("/kb/{kb_id}/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    kb_id: str, conversation_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> None:
    """Delete one conversation thread from the KB's history."""
    await wiring.conversation_store().delete(kb_id, conversation_id)
    _log.info("conversation_deleted", kb_id=kb_id, conversation_id=conversation_id)


@router.post("/kb/{kb_id}/conversations/{conversation_id}/ask", response_model=GroundedAnswer)
async def conversation_ask(
    kb_id: str, conversation_id: str, body: TurnRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> GroundedAnswer:
    """One conversational turn (non-streaming). Follow-ups resolve against thread history."""
    check_query_rate(principal.tenant_id)
    await _require_published(kb_id, principal)  # block draft (unpublished) KBs (banking pack)
    return await pipeline.answer_turn(
        kb_id,
        conversation_id,
        body.query,
        actor=principal.subject,
        acl=_principal_groups(principal),
        clearance=await kb_roles.effective_kb_clearance(get_catalog(), kb_id, principal),
        top_k=body.top_k,
        mode=body.mode,
        model=body.model,
        after=body.window_after(),
        before=body.before,
        freshness=body.freshness,
    )


@router.post("/kb/{kb_id}/conversations/{conversation_id}/ask/stream")
async def conversation_ask_stream(
    kb_id: str, conversation_id: str, body: TurnRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> StreamingResponse:
    """One conversational turn, streamed as SSE (emits a ``condensed`` event first when the
    follow-up was rewritten, then phase/token/done)."""
    check_query_rate(principal.tenant_id)
    await _require_published(kb_id, principal)  # block draft (unpublished) KBs (banking pack)
    actor = principal.subject
    acl = _principal_groups(principal)
    clearance = await kb_roles.effective_kb_clearance(get_catalog(), kb_id, principal)

    async def _events() -> AsyncIterator[bytes]:
        try:
            async for event in pipeline.stream_turn(
                kb_id,
                conversation_id,
                body.query,
                actor=actor,
                acl=acl,
                clearance=clearance,
                top_k=body.top_k,
                mode=body.mode,
                model=body.model,
                after=body.window_after(),
                before=body.before,
                freshness=body.freshness,
            ):
                yield f"data: {json.dumps(event)}\n\n".encode()
        except Exception as exc:
            _log.warning("stream_turn_failed", kb_id=kb_id, error=str(exc)[:200])
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)[:200]})}\n\n".encode()

    return StreamingResponse(_events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# --- HITL (FR-F6) ------------------------------------------------------------
@router.get("/kb/{kb_id}/hitl", response_model=list[HitlReview])
async def list_reviews(
    kb_id: str,
    principal: Annotated[Principal, Depends(tenant_guard)],
    status: HitlStatus | None = None,
    limit: int = 100,
) -> list[HitlReview]:
    """HITL review queue for the console (optionally filtered by status)."""
    return await wiring.hitl_service().list(kb_id, status=status, limit=limit)


@router.get("/kb/{kb_id}/hitl/{review_id}", response_model=HitlReview)
async def get_review(kb_id: str, review_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> HitlReview:
    return await wiring.hitl_service().get(kb_id, review_id)


@router.post("/kb/{kb_id}/hitl/{review_id}/act", response_model=HitlReview)
async def act_review(
    kb_id: str, review_id: str, body: HitlActRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> HitlReview:
    # FR-F6: only a reviewer (or admin) may act on a HITL review — was missing entirely, letting
    # a bare reader token accept/modify/flag/override a pending answer (security gap).
    _require_role(principal, "reviewer")
    return await wiring.hitl_service().act(
        kb_id,
        review_id,
        body.action,
        reviewer=body.reviewer,
        note=body.note,
        modified_answer=body.modified_answer,
    )


# --- Audit trail (FR-G4) — compliance viewer -------------------------------
@router.get("/kb/{kb_id}/audit", response_model=list[AuditEvent])
async def list_audit(
    kb_id: str,
    principal: Annotated[Principal, Depends(tenant_guard)],
    trace_id: str | None = None,
    limit: int = 100,
) -> list[AuditEvent]:
    """Read the tamper-evident audit trail (optionally one trace) for the console."""
    # The audit trail exposes every actor/query/ingest/faithfulness event for the KB — a
    # governance read, restricted to the KB's management tier (the Dashboard's recent-questions
    # widget is guarded the same way, so an ordinary member never triggers a doomed request).
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only a KB manager or org admin can read the audit trail")
    return await wiring.audit_service().list(kb_id, trace_id=trace_id, limit=limit)


# --- Job centre: recent long-running operations (NFR: job visibility) --------
class ActivityItem(BaseModel):
    action: str  # ingest | index (document processed)
    label: str  # the document / target
    detail: str = ""  # e.g. "12 chunks"
    at: datetime


_ACTIVITY_ACTIONS = {AuditAction.INGEST, AuditAction.INDEX, AuditAction.ERASURE, AuditAction.EVAL_RUN}


@router.get("/kb/{kb_id}/activity", response_model=list[ActivityItem])
async def list_activity(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)], limit: int = 20
) -> list[ActivityItem]:
    """Recent long-running operations (ingest/index/erasure/eval) for the job-centre — a
    focused, human-readable slice of the audit trail."""
    events = await wiring.audit_service().list(kb_id, limit=max(limit * 4, 40))
    items: list[ActivityItem] = []
    for e in events:
        if e.action not in _ACTIVITY_ACTIONS:
            continue
        p = e.payload
        if e.action is AuditAction.INDEX:
            n = int(p.get("chunks", 0) or 0)
            detail = f"{n} chunk{'' if n == 1 else 's'}" + (" · reindex" if p.get("reindex") else "")
        elif e.action is AuditAction.EVAL_RUN:
            detail = f"{p.get('total', 0)} scored" + (" · breach" if p.get("breached") else "")
        elif e.action is AuditAction.ERASURE:
            detail = "erasure certificate"
        else:
            detail = ""
        label = str(p.get("doc_id") or p.get("kb_id") or kb_id)
        items.append(ActivityItem(action=e.action.value, label=label, detail=detail, at=e.at))
        if len(items) >= limit:
            break
    return items


# --- Usage metering / FinOps showback (FR-H4) -------------------------------
@router.get("/kb/{kb_id}/usage", response_model=UsageReport)
async def get_usage(
    kb_id: str,
    principal: Annotated[Principal, Depends(tenant_guard)],
    window: str = "7d",
) -> UsageReport:
    """Per-tenant usage + indicative showback over a window (24h | 7d | 30d | all)."""
    _require_role(principal, "reader")
    try:
        return await usage_report(kb_id, window)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


# --- Knowledge graph (GraphRAG, FR-C6/D4) ------------------------------------
class GraphStatsInfo(BaseModel):
    enabled: bool
    nodes: int
    edges: int
    top_entities: list[dict[str, Any]] = Field(default_factory=list)


@router.get("/kb/{kb_id}/graph", response_model=GraphStatsInfo)
async def get_graph(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> GraphStatsInfo:
    """The KB's knowledge-graph size + most-connected entities (GraphRAG)."""
    enabled = get_settings().graphrag_enabled
    if not enabled:
        return GraphStatsInfo(enabled=False, nodes=0, edges=0)
    stats = await get_graph_store().stats(kb_id, top=15)
    return GraphStatsInfo(
        enabled=True,
        nodes=stats.nodes,
        edges=stats.edges,
        top_entities=[{"name": n, "mentions": m} for n, m in stats.top_entities],
    )


class GraphVizNode(BaseModel):
    id: str
    label: str
    degree: int
    community: int  # community_id for colour grouping, or -1 if unclustered


class GraphSubgraph(BaseModel):
    """A bounded, renderable slice of the KB graph: the most-connected entities + the edges
    among them (the full graph can be tens of thousands of nodes — too many for a browser)."""

    enabled: bool
    nodes: list[GraphVizNode] = Field(default_factory=list)
    edges: list[tuple[int, int]] = Field(default_factory=list)  # index pairs into `nodes`
    total_nodes: int = 0
    total_edges: int = 0


@router.get("/kb/{kb_id}/graph/subgraph", response_model=GraphSubgraph)
async def get_graph_subgraph(
    kb_id: str,
    principal: Annotated[Principal, Depends(tenant_guard)],
    limit: int = Query(default=60, ge=10, le=5000),
) -> GraphSubgraph:
    """Top-``limit`` most-connected entities + the edges between them, coloured by community —
    a renderable subgraph for the console's knowledge-graph visualisation. The 2D/3D views
    request a small subgraph (~60-80); the GPU "Overview" asks for the whole graph (hence the
    high cap — the store already loads every node/edge, this only widens the top-N slice)."""
    if not get_settings().graphrag_enabled:
        return GraphSubgraph(enabled=False)
    store = get_graph_store()
    nodes, edges = await store.graph_data(kb_id)
    degree: dict[str, int] = {}
    for src, tgt in edges:
        degree[src] = degree.get(src, 0) + 1
        degree[tgt] = degree.get(tgt, 0) + 1
    top = sorted(nodes, key=lambda n: degree.get(n.key, 0), reverse=True)[:limit]
    idx = {n.key: i for i, n in enumerate(top)}
    seen: set[tuple[int, int]] = set()
    viz_edges: list[tuple[int, int]] = []
    for src, tgt in edges:
        if src in idx and tgt in idx and src != tgt:
            pair = (min(idx[src], idx[tgt]), max(idx[src], idx[tgt]))
            if pair not in seen:
                seen.add(pair)
                viz_edges.append(pair)
    # Colour by community: map an entity's display name to its community via top_entities.
    # Tag each node with its REAL community via the persisted per-node membership (members). Older
    # KBs whose communities predate the members column fall back to the top_entities mapping so the
    # colouring still works until the themes are rebuilt.
    comms = await store.list_communities(kb_id)
    key_to_comm: dict[str, int] = {}
    disp_to_comm: dict[str, int] = {}
    for c in comms:
        for k in c.members:
            key_to_comm.setdefault(k, c.community_id)
        for ent in c.top_entities:
            disp_to_comm.setdefault(ent, c.community_id)
    viz_nodes = [
        GraphVizNode(
            id=n.key,
            label=n.display,
            degree=degree.get(n.key, 0),
            community=key_to_comm.get(n.key, disp_to_comm.get(n.display, -1)),
        )
        for n in top
    ]
    return GraphSubgraph(enabled=True, nodes=viz_nodes, edges=viz_edges, total_nodes=len(nodes), total_edges=len(edges))


class CommunityInfo(BaseModel):
    community_id: int
    label: str
    size: int
    top_entities: list[str] = Field(default_factory=list)
    summary: str = ""


class CommunitiesResponse(BaseModel):
    enabled: bool
    communities: list[CommunityInfo] = Field(default_factory=list)


class BuildCommunitiesResponse(BaseModel):
    communities: int
    entities: int


@router.get("/kb/{kb_id}/graph/communities", response_model=CommunitiesResponse)
async def get_graph_communities(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> CommunitiesResponse:
    """The KB's thematic communities (GraphRAG slice 2) — the 'Themes' the graph groups
    into. Empty until built via POST. Read-only; safe for any KB reader."""
    if not get_settings().graphrag_enabled:
        return CommunitiesResponse(enabled=False)
    coms = await pipeline.list_communities(kb_id)
    return CommunitiesResponse(
        enabled=True,
        communities=[
            CommunityInfo(
                community_id=c.community_id,
                label=c.label,
                size=c.size,
                top_entities=c.top_entities,
                summary=c.summary,
            )
            for c in coms
        ],
    )


@router.post("/kb/{kb_id}/graph/communities", response_model=BuildCommunitiesResponse)
async def build_graph_communities(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> BuildCommunitiesResponse:
    """Detect communities over the KB graph and summarise each (GraphRAG slice 2). A graph-
    wide pass, so it's on-demand (never per-ingest) and gated to writers."""
    _require_role(principal, "ingestor")
    result = await pipeline.build_communities(kb_id)
    return BuildCommunitiesResponse(communities=result["communities"], entities=result["entities"])


# --- Token budgets / spend caps (FR-H7) --------------------------------------
class BudgetRequest(BaseModel):
    budget_tokens: int = Field(..., ge=0, description="Monthly hard cap in total tokens; 0 = unlimited.")
    alert_pct: float = Field(default=0.8, ge=0.1, le=1.0)


@router.get("/kb/{kb_id}/budget", response_model=budgets.BudgetInfo)
async def get_budget(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> budgets.BudgetInfo:
    """This KB's monthly token budget + how much of it is used."""
    return await budgets.get_budget(kb_id)


@router.put("/kb/{kb_id}/budget", response_model=budgets.BudgetInfo)
async def set_budget(
    kb_id: str, body: BudgetRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> budgets.BudgetInfo:
    """Set the monthly token budget (admin). Over the cap, answers are refused (429)."""
    _require_role(principal, "admin")
    return await budgets.set_budget(kb_id, body.budget_tokens, body.alert_pct)


# --- Documents in a knowledge base (console visibility) ----------------------
class DocumentSummary(BaseModel):
    doc_id: str
    mime: str
    source_kind: str
    uri: str | None = None
    ingested_at: datetime
    metadata: dict[str, str] = Field(default_factory=dict)  # incl. auto-tags (FR-B7)


@router.get("/kb/{kb_id}/documents", response_model=list[DocumentSummary])
async def list_kb_documents(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> list[DocumentSummary]:
    """List the documents ingested into this knowledge base (newest first)."""
    docs = await get_catalog().list_documents(kb_id)
    return [
        DocumentSummary(
            doc_id=d.doc_id,
            mime=d.mime,
            source_kind=d.source_kind.value,
            uri=d.uri,
            ingested_at=d.ingested_at,
            metadata=d.metadata,
        )
        for d in docs
    ]


# --- Per-KB taxonomy for auto-tagging (FR-B7) --------------------------------
_TAXONOMY_KEY = "taxonomy"


class Taxonomy(BaseModel):
    taxonomy: dict[str, list[str]] = Field(
        default_factory=dict, description="Label -> keywords; a doc gets the label if any keyword appears."
    )


@router.get("/kb/{kb_id}/taxonomy", response_model=Taxonomy)
async def get_taxonomy(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> Taxonomy:
    """The KB's auto-tagging taxonomy (empty if unset)."""
    return Taxonomy(taxonomy=await pipeline._load_taxonomy(kb_id))


@router.put("/kb/{kb_id}/taxonomy", response_model=Taxonomy)
async def set_taxonomy(kb_id: str, body: Taxonomy, principal: Annotated[Principal, Depends(tenant_guard)]) -> Taxonomy:
    """Configure the KB's taxonomy (admin). Applied to newly ingested/reindexed documents."""
    _require_role(principal, "admin")
    await get_catalog().set_tenant_config(kb_id, _TAXONOMY_KEY, json.dumps(body.taxonomy))
    return body


# --- Authoritative-document pins (version conflict) --------------------------
_PINNED_DOCS_KEY = "authoritative_docs"


class AuthoritativeDocs(BaseModel):
    """Documents a KB manager has marked as the version that governs.

    A pin overrules the effective dates read from the documents themselves. That is the point:
    the manager knows things the platform cannot read off a page — that a file's printed date is
    wrong, that it was reissued without a new date, or that the older wording still governs
    pending a review."""

    doc_ids: list[str] = Field(default_factory=list)


@router.get("/kb/{kb_id}/authoritative-docs", response_model=AuthoritativeDocs)
async def get_authoritative_docs(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> AuthoritativeDocs:
    """Which documents are pinned as authoritative for this knowledge base."""
    _require_role(principal, "reader")
    raw = (await get_catalog().get_tenant_config(kb_id)).get(_PINNED_DOCS_KEY, "[]")
    try:
        return AuthoritativeDocs(doc_ids=list(json.loads(raw)))
    except (ValueError, TypeError):
        return AuthoritativeDocs()


@router.put("/kb/{kb_id}/authoritative-docs", response_model=AuthoritativeDocs)
async def set_authoritative_docs(
    kb_id: str, body: AuthoritativeDocs, principal: Annotated[Principal, Depends(tenant_guard)]
) -> AuthoritativeDocs:
    """Pin the documents that govern when versions conflict (KB manager or org admin).

    Gated to whoever manages the KB rather than to any member: deciding which version of a policy
    is in force is exactly the judgement a KB manager is appointed to make, and exactly the one an
    ordinary reader must not be able to change for everyone else.

    Pinning MORE THAN ONE document is accepted and deliberately does nothing — there is then no
    single operator decision to honour, so resolution falls back to the documents' own dates
    rather than picking a winner nobody chose."""
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only an org admin or a manager of this knowledge base may pin documents")
    await get_catalog().set_tenant_config(kb_id, _PINNED_DOCS_KEY, json.dumps(body.doc_ids))
    _log.info("authoritative_docs_set", kb_id=kb_id, count=len(body.doc_ids), actor=principal.subject)
    return body


# --- In-document citation viewer (FR-I3) ------------------------------------
class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    page: int | None = None


class DocumentContent(BaseModel):
    """A document's readable source text, reconstructed from its ordered chunks, so the
    console can open a citation in context and highlight the supporting span."""

    doc_id: str
    mime: str
    source_uri: str | None = None
    chunks: list[SourceChunk]


@router.get("/kb/{kb_id}/documents/{doc_id}/content", response_model=DocumentContent)
async def get_document_content(
    kb_id: str, doc_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> DocumentContent:
    """Source text of a document in reading order (for the in-document citation viewer).

    ACL-aware: a chunk is visible only if it is tenant-public or intersects the caller's
    groups — the viewer can never surface a span the caller could not retrieve (FR-A6)."""
    groups = _principal_groups(principal)
    # FR-C2: parents are a non-citable context bank — never a document's own citable content.
    chunks = not_parent(await get_catalog().document_chunks(kb_id, doc_id))
    visible = [c for c in chunks if not c.acl or (set(c.acl) & set(groups))]
    if not visible:
        raise NotFoundError("document has no readable content", details={"doc_id": doc_id})
    docs = {d.doc_id: d for d in await get_catalog().list_documents(kb_id)}
    envelope = docs.get(doc_id)
    return DocumentContent(
        doc_id=doc_id,
        mime=envelope.mime if envelope else "text/plain",
        source_uri=envelope.uri if envelope else None,
        chunks=[SourceChunk(chunk_id=c.chunk_id, text=c.text, page=c.page) for c in visible],
    )


class ImageAnalysisRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    interpret: bool = Field(
        default=True,
        description=(
            "True (default): the model may use its own subject knowledge to EXPLAIN the image "
            "— not a grounded answer. False: strictly what is visible, the platform's normal rule."
        ),
    )


class ImageAnalysis(BaseModel):
    doc_id: str
    question: str
    answer: str
    model: str
    grounded: bool = Field(
        description="False whenever interpret=True — the answer may contain knowledge that is in "
        "no document in this knowledge base, and no citation backs it."
    )


@router.post("/kb/{kb_id}/documents/{doc_id}/analyse", response_model=ImageAnalysis)
async def analyse_document_image(
    kb_id: str, doc_id: str, body: ImageAnalysisRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> ImageAnalysis:
    """Ask a vision model about a stored IMAGE, optionally letting it interpret what it sees.

    This is the one path in the platform that is deliberately NOT grounded-or-abstain, and it
    exists because some documents are useless under that rule. A page of chemical formulas, a
    circuit diagram or a score carries meaning that is not written on it: asked "what does this
    reaction do?", the grounded pipeline correctly abstains, because the answer is chemistry
    knowledge rather than document content.

    The exception is made explicit rather than by loosening the gate: a separate endpoint, a
    distinct IMAGE_ANALYSIS audit action, and ``grounded: false`` on every interpreted response.
    Nothing about the normal ask path changes.

    Two things a caller must understand:
      * with ``interpret=true`` there are NO citations and the faithfulness gate does not run.
        Treat the output as a knowledgeable reader's opinion, not as sourced fact.
      * the model reads RAW PIXELS, which bypasses the PII masking applied to extracted text.
        An image holding an Aadhaar or PAN number can therefore be described even though the
        indexed text masked it — which is why every call is audited with its question.
    """
    _require_role(principal, "reader")
    envelope = next((d for d in await get_catalog().list_documents(kb_id) if d.doc_id == doc_id), None)
    if envelope is None:
        raise NotFoundError("document not found", details={"doc_id": doc_id})
    if not multimodal.is_image(envelope):
        raise ValidationError(
            "this endpoint reads IMAGES; for a text document the ordinary grounded ask already "
            "sees its full extracted text",
            details={"doc_id": doc_id, "mime": envelope.mime},
        )
    # Same visibility rule as the citation viewer: a caller who cannot retrieve any of this
    # document's chunks must not be able to have a model read its pixels aloud instead.
    groups = set(_principal_groups(principal))
    chunks = not_parent(await get_catalog().document_chunks(kb_id, doc_id))  # FR-C2: gate on children only
    if chunks and not any((not c.acl) or (set(c.acl) & groups) for c in chunks):
        raise NotFoundError("document has no readable content", details={"doc_id": doc_id})
    try:
        answer = await multimodal.describe_image(kb_id, doc_id, body.question, interpret=body.interpret)
    except RuntimeError as exc:  # multimodal disabled on this deployment
        raise ValidationError(str(exc), details={"doc_id": doc_id}) from exc
    gateway = await wiring.resolve_tenant_vision_gateway(kb_id)
    await wiring.audit_service().record(
        tenant_id=kb_id,
        trace_id=str(uuid.uuid4()),
        action=AuditAction.IMAGE_ANALYSIS,
        actor=principal.subject,
        payload={"doc_id": doc_id, "question": body.question, "interpret": body.interpret},
    )
    return ImageAnalysis(
        doc_id=doc_id,
        question=body.question,
        answer=answer,
        model=gateway.model_name,
        grounded=not body.interpret,
    )


class ReindexRequest(BaseModel):
    # None -> reproduce the chunking this document already has. Reindex used to hardcode a
    # strategy, so re-indexing silently converted documents ingested any other way.
    strategy: ChunkStrategy | None = None


@router.post("/kb/{kb_id}/documents/{doc_id}/reindex", response_model=IngestResponse)
async def reindex_kb_document(
    kb_id: str,
    doc_id: str,
    principal: Annotated[Principal, Depends(tenant_guard)],
    body: ReindexRequest | None = None,
) -> IngestResponse:
    """Re-index a document from its stored raw bytes (FR-C8) — e.g. to adopt a new
    chunking strategy or contextual retrieval without re-uploading."""
    _require_role(principal, "ingestor")
    check_ingest_rate(principal.tenant_id)
    outcome = await pipeline.reindex_document(
        kb_id, doc_id, actor=principal.subject, strategy=body.strategy if body else None
    )
    return IngestResponse(
        doc_id=outcome.doc_id,
        chunk_count=outcome.chunk_count,
        pii_masked=outcome.pii_masked,
        trace_id=outcome.trace_id,
    )


class IndexStatusResponse(BaseModel):
    """Whether this KB's documents were built by the pipeline that is running now."""

    kb_id: str
    current_version: str = Field(..., description="Fingerprint of the running indexing pipeline.")
    total: int
    stale: int = Field(..., description="Documents indexed by a different pipeline, or carrying no stamp.")
    stale_doc_ids: list[str]
    auto_reindex: bool = Field(..., description="True when an operator opted this KB into automatic re-indexing.")


@router.get("/kb/{kb_id}/index-status", response_model=IndexStatusResponse)
async def kb_index_status(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> IndexStatusResponse:
    """How much of the KB is still indexed by an older pipeline (FR-C8).

    A read, so any reader may ask: "are the answers I am getting built by the code running
    here?" is not privileged information about someone else's data."""
    _require_role(principal, "reader")
    report = await reindex.stale_report(kb_id)
    return IndexStatusResponse(
        kb_id=report.kb_id,
        current_version=report.current_version,
        total=report.total,
        stale=report.stale,
        stale_doc_ids=report.stale_doc_ids[:200],
        auto_reindex=kb_id in reindex.auto_reindex_kbs(),
    )


class ReindexKbRequest(BaseModel):
    stale_only: bool = Field(default=True, description="Rebuild only documents the running pipeline did not index.")
    limit: int | None = Field(default=None, ge=1, le=10_000, description="Cap on documents this pass rebuilds.")


class ReindexKbResponse(BaseModel):
    kb_id: str
    requested: int
    reindexed: int
    skipped_current: int
    failed: list[dict[str, str]]


@router.post("/kb/{kb_id}/reindex", response_model=ReindexKbResponse)
async def reindex_whole_kb(
    kb_id: str,
    principal: Annotated[Principal, Depends(tenant_guard)],
    body: ReindexKbRequest | None = None,
) -> ReindexKbResponse:
    """Rebuild the KB's stale documents (FR-C8). Sequential and capped, because a re-index
    competes with live questions for the same model endpoints."""
    _require_role(principal, "ingestor")
    check_ingest_rate(principal.tenant_id)
    req = body or ReindexKbRequest()
    result = await reindex.reindex_kb(
        kb_id,
        actor=principal.subject,
        stale_only=req.stale_only,
        limit=req.limit if req.limit is not None else get_settings().reindex_batch_limit,
    )
    return ReindexKbResponse(
        kb_id=result.kb_id,
        requested=result.requested,
        reindexed=result.reindexed,
        skipped_current=result.skipped_current,
        failed=result.failed,
    )


@router.delete("/kb/{kb_id}/documents/{doc_id}", status_code=204)
async def delete_kb_document(kb_id: str, doc_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> None:
    """Delete a document and purge it from every store (catalog, vectors, BM25, object)."""
    _require_role(principal, "ingestor")
    await pipeline.delete_document(kb_id, doc_id, actor=principal.subject)


# --- Model connection (FR-E1 — bring your own sovereign model) ---------------
class ModelConnectionRequest(BaseModel):
    """Configure a tenant's own OpenAI-compatible model endpoint from the console."""

    endpoint: str = Field(..., min_length=1, description="OpenAI-compatible base URL (must be egress-allowlisted).")
    model: str = Field(..., min_length=1, description="Model / deployment name.")
    header: str = Field(default="apikey", min_length=1, description="Header carrying the API key.")
    api_key: str | None = Field(default=None, description="API key; omit on update to keep the stored key.")
    active: bool = True
    role: str = Field(
        default="generation",
        pattern="^(generation|embedding|reranking|ocr|vision)$",
        description=(
            "What this model DOES. Each role is a different job and they are not interchangeable: "
            "generation writes the answer, embedding turns text into vectors, reranking re-orders "
            "retrieved passages, ocr reads scanned pages and images, vision describes an image. "
            "The faithfulness (NLI) judge is deliberately NOT here — it is the platform's own "
            "grounding gate, and a tenant choosing their own judge could weaken the check that "
            "protects them."
        ),
    )
    embed_dim: int | None = Field(
        default=None, gt=0, description="Embedding dimension (required for role=embedding; the Test probe reports it)."
    )
    api_shape: str = Field(
        default="", description="Reranker wire shape (required for role=reranking; the Test probe detects it)."
    )
    backend: str = Field(
        default="openai_compat",
        pattern="^(openai_compat|bge_m3_multi)$",
        description=(
            "Embedding adapter type (role=embedding only): 'openai_compat' = /embeddings + apikey "
            "(default); 'bge_m3_multi' = the bge-m3 /embed endpoint (Bearer) that also yields "
            "learned-sparse + ColBERT, enabling multivector retrieval once the KB is reindexed."
        ),
    )


class ModelConnectionInfo(BaseModel):
    """A tenant's model connection as shown in the console — never returns the key."""

    configured: bool
    endpoint: str | None = None
    model: str | None = None
    header: str | None = None
    active: bool = False
    has_key: bool = False


class ModelTestResult(BaseModel):
    ok: bool
    available: list[str] = Field(default_factory=list)
    error: str | None = None
    dim: int | None = Field(default=None, description="Embedding dimension the endpoint returned (embedding role).")
    api_shape: str | None = Field(default=None, description="Detected rerank wire shape (reranking role).")
    score_sample: float | None = Field(default=None, description="Relevant-doc probe score (reranking role).")


def _conn_info(conn: StoredModelConnection | None) -> ModelConnectionInfo:
    """Back-compat single-connection view (the KB's default) for Ask/Dashboard."""
    if conn is None:
        return ModelConnectionInfo(configured=False)
    return ModelConnectionInfo(
        configured=True,
        endpoint=conn.endpoint,
        # The Ask/Dashboard "answering with" line shows this — use the connection's friendly NAME
        # (e.g. "qwen-32b", "Sarvam-30B (local GPU)") rather than the raw deployment id ("qwendeploy",
        # "/model"), matching the topbar indicator. The raw id is never user-facing.
        model=conn.name or conn.model,
        header=conn.header,
        active=conn.is_default,
        has_key=bool(conn.key_ciphertext),
    )


class ModelConnectionItem(BaseModel):
    """One registered model connection (no secret returned)."""

    conn_id: str
    name: str
    endpoint: str
    model: str
    header: str
    has_key: bool
    is_default: bool
    role: str = "generation"
    embed_dim: int | None = None
    api_shape: str = ""
    backend: str = "openai_compat"
    scope: str = "kb"  # "kb" = registered on this KB; "org" = an org-wide model, READ-ONLY here.


def _conn_item(conn: StoredModelConnection, scope: str = "kb") -> ModelConnectionItem:
    return ModelConnectionItem(
        conn_id=conn.conn_id,
        name=conn.name,
        endpoint=conn.endpoint,
        model=conn.model,
        header=conn.header,
        has_key=bool(conn.key_ciphertext),
        is_default=conn.is_default,
        role=conn.role,
        embed_dim=conn.embed_dim,
        api_shape=conn.api_shape,
        backend=conn.backend,
        scope=scope,
    )


def _require_role_fields(body: ModelConnectionRequest) -> None:
    """An embedder without its dimension (or a reranker without its wire shape) cannot
    run deterministically — the console's Test step supplies both."""
    if body.role == "embedding" and body.embed_dim is None:
        raise ValidationError("embed_dim is required for an embedding model — run Test to detect it")
    if body.role == "reranking" and not body.api_shape:
        raise ValidationError("api_shape is required for a reranking model — run Test to detect it")


# --- Org-wide model registry (org-admin managed) ---------------------------------------------
# Org-wide model connections live in the ORG's tenant_config under `org_model_connections`
# (same record shape as the per-KB StoredModelConnection; the key is encrypted at rest with the
# ORG's DEK exactly like per-KB keys). Option B — "available, not auto-default": these surface in
# EVERY KB's model list (tagged scope="org", READ-ONLY there) and the Ask per-question picker, but
# they NEVER change a KB's default resolution. A KB only *uses* one when it is explicitly chosen
# per question, or when a KB adopts it as its default (see `org_default_conn` below).
_ORG_MODEL_CONNECTIONS_KEY = "org_model_connections"


async def _load_org_connections(org_id: str) -> list[StoredModelConnection]:
    raw = (await get_catalog().get_tenant_config(org_id)).get(_ORG_MODEL_CONNECTIONS_KEY)
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    out: list[StoredModelConnection] = []
    for r in data:
        if isinstance(r, dict):
            try:
                out.append(StoredModelConnection(**{**r, "tenant_id": org_id}))
            except (ValueError, TypeError):
                continue
    return sorted(out, key=lambda c: (not c.is_default, c.name))


async def _save_org_connections(org_id: str, conns: list[StoredModelConnection]) -> None:
    await get_catalog().set_tenant_config(
        org_id, _ORG_MODEL_CONNECTIONS_KEY, json.dumps([c.model_dump() for c in conns])
    )


async def _get_org_connection(org_id: str, conn_id: str) -> StoredModelConnection | None:
    return next((c for c in await _load_org_connections(org_id) if c.conn_id == conn_id), None)


@router.get("/kb/{kb_id}/model-connection", response_model=ModelConnectionInfo)
async def get_model_connection(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> ModelConnectionInfo:
    """The KB's DEFAULT model connection (no secret returned) — used by Ask/Dashboard."""
    # KB-level default, else the ORG-WIDE default (generation is org-wide) — so Ask's "answering
    # with" line shows the real model (qwen) instead of falling back to the dead platform default.
    conn = await wiring.model_connection_store().get_default(kb_id)
    if conn is None:
        conn = await wiring._resolve_org_default_connection(kb_id, "generation")
    return _conn_info(conn)


@router.get("/kb/{kb_id}/model-connections", response_model=list[ModelConnectionItem])
async def list_model_connections(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> list[ModelConnectionItem]:
    """Every model connection AVAILABLE to this KB: the KB's own connections (scope="kb", default
    first) PLUS the org-wide connections (scope="org", READ-ONLY here). Option B — an org-wide
    model is available for per-question selection or KB adoption, but never becomes a KB default
    on its own. Readable by any tenant member (writes stay gated)."""
    kb_conns = [_conn_item(c, "kb") for c in await wiring.model_connection_store().list(kb_id)]
    # The org registry lives on the ORG (principal.tenant_id); is_default there is ORG-scoped and
    # irrelevant to this KB, so it is flattened off when surfaced in a KB's list.
    org_raw = await _load_org_connections(principal.tenant_id)
    # Consolidated model: the org-wide default IS every KB's effective generation model, so its
    # is_default is meaningful here (kept, not flattened) — the picker marks the real default.
    org_conns = [_conn_item(c, "org") for c in org_raw]
    # The built-in platform default (Sarvam) is no longer surfaced as a separate selectable
    # "Platform default" entry — the org-wide Sarvam default already serves every KB, so the extra
    # option was redundant/confusing. Resolution still falls back to the platform default
    # server-side when no org-wide generation model is marked default (see wiring.resolve_*).
    return kb_conns + org_conns


class AddModelConnectionRequest(ModelConnectionRequest):
    name: str = Field(..., min_length=1, description="Human label for this connection.")
    make_default: bool = Field(default=True, description="Answer with this connection by default.")


# --- Model-change approval (KB-manager request → org-admin approve → applied) ----------------
# Model-connection WRITE endpoints (add/update/default/delete + embedder/apply) are gated by
# _can_manage_kb (org owner/admin OR a manager of this KB). An org owner/admin's write APPLIES
# IMMEDIATELY; a KB manager's write instead creates a PENDING model-request in the ORG's
# tenant_config (mirrors the access-request pattern) — it applies only when an org admin approves,
# and the approve path dispatches back through the same _apply_* functions the direct write uses.
# "test connection" stays directly allowed for managers (it probes, changes nothing).
_MODEL_REQUESTS_KEY = "model_requests"


def _is_org_admin(principal: Principal) -> bool:
    """An org owner/admin — the approver, who also applies model changes immediately."""
    return principal.has_role("owner") or principal.has_role("admin")


async def _load_model_requests(org_id: str) -> list[dict[str, object]]:
    raw = (await get_catalog().get_tenant_config(org_id)).get(_MODEL_REQUESTS_KEY)
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
    except (ValueError, TypeError):
        return []


async def _save_model_requests(org_id: str, rows: list[dict[str, object]]) -> None:
    await get_catalog().set_tenant_config(org_id, _MODEL_REQUESTS_KEY, json.dumps(rows))


class ModelRequestInfo(BaseModel):
    """A per-KB model-change request awaiting (or having received) an org-admin decision. Returned
    (HTTP 202) when a KB manager submits a model write, and listed on the approval queue."""

    id: str
    kb_id: str  # "" for an org-wide (org_add) request — it targets the org registry, not one KB
    kb_name: str = ""
    action: str  # add | update | default | delete | embedder_apply | org_add
    conn_id: str | None = None
    payload: dict[str, Any] | None = None
    requested_by: str
    requested_at: str = ""
    status: str = "pending"  # pending | approved | rejected
    decided_by: str = ""
    decided_at: str = ""
    reason: str = ""


class ModelDecisionBody(BaseModel):
    decision: str = Field(..., pattern="^(approve|reject)$")
    reason: str = Field(default="", max_length=500)


async def _submit_model_request(
    kb_id: str,
    action: str,
    conn_id: str | None,
    payload: dict[str, Any] | None,
    principal: Principal,
    response: Response,
) -> ModelRequestInfo:
    """Persist a KB manager's model write as a PENDING request on the ORG (not applied), audit it,
    and answer 202 Accepted with the request record for the console's 'awaiting approval' state."""
    org = principal.tenant_id
    now = datetime.now(UTC).isoformat()
    row: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "kb_id": kb_id,
        "action": action,
        "conn_id": conn_id,
        "payload": payload,
        "requested_by": principal.subject.strip().lower(),
        "requested_at": now,
        "status": "pending",
        "decided_by": "",
        "decided_at": "",
        "reason": "",
    }
    rows = await _load_model_requests(org)
    rows.append(row)
    await _save_model_requests(org, rows)
    await wiring.audit_service().record(
        tenant_id=kb_id,
        trace_id=str(uuid.uuid4()),
        action=AuditAction.MODEL_REQUEST,
        actor=principal.subject,
        payload={"op": "request", "request_id": row["id"], "action": action, "conn_id": conn_id},
    )
    _log.info(
        "model_request_created",
        org_id=org,
        kb_id=kb_id,
        request_id=row["id"],
        action=action,
        requested_by=principal.subject,
    )
    response.status_code = 202
    return ModelRequestInfo(kb_name=await _kb_name(org, kb_id), **row)


async def _submit_org_model_request(
    payload: dict[str, Any], principal: Principal, response: Response
) -> ModelRequestInfo:
    """Persist a KB manager's ORG-WIDE model request as a PENDING request on the ORG (not applied),
    audit it, and answer 202. Stored in the same `model_requests` list as per-KB requests but with
    action=``org_add`` and an empty kb_id (it targets the org registry, not one KB)."""
    org = principal.tenant_id
    now = datetime.now(UTC).isoformat()
    row: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "kb_id": "",
        "action": "org_add",
        "conn_id": None,
        "payload": payload,
        "requested_by": principal.subject.strip().lower(),
        "requested_at": now,
        "status": "pending",
        "decided_by": "",
        "decided_at": "",
        "reason": "",
    }
    rows = await _load_model_requests(org)
    rows.append(row)
    await _save_model_requests(org, rows)
    await wiring.audit_service().record(
        tenant_id=org,
        trace_id=str(uuid.uuid4()),
        action=AuditAction.MODEL_REQUEST,
        actor=principal.subject,
        payload={"op": "request", "request_id": row["id"], "action": "org_add"},
    )
    _log.info("org_model_request_created", org_id=org, request_id=row["id"], requested_by=principal.subject)
    response.status_code = 202
    return ModelRequestInfo(**row)


async def _apply_add_connection(
    kb_id: str, body: AddModelConnectionRequest, principal: Principal
) -> ModelConnectionItem:
    """Apply an 'add model connection' — the shared body run by a direct org-admin write AND by
    the approve path when an org admin approves a KB manager's pending request."""
    assert_sovereign(body.endpoint)
    _require_role_fields(body)
    store = wiring.model_connection_store()
    existing = await store.list(kb_id, body.role)
    conn = StoredModelConnection(
        tenant_id=kb_id,
        conn_id=str(uuid.uuid4()),
        name=body.name,
        endpoint=body.endpoint,
        model=body.model,
        header=body.header,
        key_ciphertext=(await tenant_crypto.encrypt(kb_id, body.api_key) if body.api_key else ""),
        is_default=body.make_default or not existing,
        role=body.role,
        embed_dim=body.embed_dim,
        api_shape=body.api_shape,
        backend=body.backend,
    )
    await store.put(conn)
    wiring.invalidate_tenant_gateway(kb_id)
    if conn.role == "embedding":
        await embedder_switch.on_embedding_default_change(kb_id)
    _log.info("model_connection_added", kb_id=kb_id, conn_id=conn.conn_id, endpoint=body.endpoint, model=body.model)
    return _conn_item(conn)


@router.post(
    "/kb/{kb_id}/model-connections",
    status_code=201,
    response_model=ModelConnectionItem | ModelRequestInfo,
)
async def add_model_connection(
    kb_id: str,
    body: AddModelConnectionRequest,
    response: Response,
    principal: Annotated[Principal, Depends(tenant_guard)],
) -> ModelConnectionItem | ModelRequestInfo:
    """Register a new model connection. Endpoint must be egress-allowlisted (403 otherwise);
    the key is encrypted at rest. The first connection OF ITS ROLE becomes that role's default.
    An org owner/admin applies immediately; a KB manager creates a PENDING request (202) that an
    org admin must approve."""
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only an owner/admin or a manager of this KB can configure its models")
    if _is_org_admin(principal):
        return await _apply_add_connection(kb_id, body, principal)
    return await _submit_model_request(kb_id, "add", None, body.model_dump(), principal, response)


async def _apply_update_connection(
    kb_id: str, conn_id: str, body: AddModelConnectionRequest, principal: Principal
) -> ModelConnectionItem:
    """Apply an 'update model connection' — shared by the direct org-admin write and the approve path."""
    assert_sovereign(body.endpoint)
    store = wiring.model_connection_store()
    existing = await store.get(kb_id, conn_id)
    if existing is None:
        raise NotFoundError("model connection not found", details={"conn_id": conn_id})
    if body.role != existing.role:
        raise ValidationError(
            "a connection's role cannot change — remove it and add a new one",
            details={"role": existing.role},
        )
    merged_dim = body.embed_dim if body.embed_dim is not None else existing.embed_dim
    merged_shape = body.api_shape or existing.api_shape
    if existing.role == "embedding" and merged_dim is None:
        raise ValidationError("embed_dim is required for an embedding model — run Test to detect it")
    if existing.role == "reranking" and not merged_shape:
        raise ValidationError("api_shape is required for a reranking model — run Test to detect it")
    ciphertext = await tenant_crypto.encrypt(kb_id, body.api_key) if body.api_key else existing.key_ciphertext
    conn = StoredModelConnection(
        tenant_id=kb_id,
        conn_id=conn_id,
        name=body.name,
        endpoint=body.endpoint,
        model=body.model,
        header=body.header,
        key_ciphertext=ciphertext,
        is_default=body.make_default or existing.is_default,
        role=existing.role,
        embed_dim=merged_dim,
        api_shape=merged_shape,
        backend=body.backend,
    )
    await store.put(conn)
    wiring.invalidate_tenant_gateway(kb_id)
    if conn.role == "embedding":
        await embedder_switch.on_embedding_default_change(kb_id)
    return _conn_item(conn)


@router.put("/kb/{kb_id}/model-connections/{conn_id}", response_model=ModelConnectionItem | ModelRequestInfo)
async def update_model_connection(
    kb_id: str,
    conn_id: str,
    body: AddModelConnectionRequest,
    response: Response,
    principal: Annotated[Principal, Depends(tenant_guard)],
) -> ModelConnectionItem | ModelRequestInfo:
    """Update a connection (blank api_key keeps the stored key). The role is fixed at
    creation — changing what a saved endpoint does would silently repoint a pipeline seam.
    An org owner/admin applies immediately; a KB manager creates a PENDING request (202)."""
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only an owner/admin or a manager of this KB can configure its models")
    if await _get_org_connection(principal.tenant_id, conn_id) is not None:
        raise ForbiddenError("this is an org-wide model — edit it under Org models, not per KB")
    if _is_org_admin(principal):
        return await _apply_update_connection(kb_id, conn_id, body, principal)
    return await _submit_model_request(kb_id, "update", conn_id, body.model_dump(), principal, response)


# A KB can adopt an ORG-WIDE connection as its default (Option B (b)). Because that connection
# does not live in the KB's own store, the adoption is recorded per-role in the KB's tenant_config
# under `org_default_conn` = {role: conn_id}; the resolution path falls back to the org registry
# when a KB's chosen default isn't found locally. Setting a KB-local default clears this so the two
# never conflict.
_ORG_DEFAULT_CONN_KEY = "org_default_conn"


async def _load_org_default_conns(kb_id: str) -> dict[str, str]:
    raw = (await get_catalog().get_tenant_config(kb_id)).get(_ORG_DEFAULT_CONN_KEY)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


async def _set_org_default_conn(kb_id: str, role: str, conn_id: str | None) -> None:
    mapping = await _load_org_default_conns(kb_id)
    if conn_id is None:
        mapping.pop(role, None)
    else:
        mapping[role] = conn_id
    await get_catalog().set_tenant_config(kb_id, _ORG_DEFAULT_CONN_KEY, json.dumps(mapping))


async def _apply_default_connection(kb_id: str, conn_id: str, principal: Principal) -> ModelConnectionItem:
    """Apply a 'make default' — shared by the direct org-admin write and the approve path. The
    conn_id may be a KB-local connection OR an org-wide one this KB is adopting (Option B (b))."""
    store = wiring.model_connection_store()
    conn = await store.get(kb_id, conn_id)
    if conn is not None:
        # Adopting a KB-local connection clears any org-wide adoption for that role.
        await store.set_default(kb_id, conn_id)
        await _set_org_default_conn(kb_id, conn.role, None)
        wiring.invalidate_tenant_gateway(kb_id)
        if conn.role == "embedding":
            await embedder_switch.on_embedding_default_change(kb_id)
        return _conn_item(conn.model_copy(update={"is_default": True}), "kb")
    # Not a KB-local connection — it must be an org-wide one this KB is adopting as its default.
    org_conn = await _get_org_connection(principal.tenant_id, conn_id)
    if org_conn is None:
        raise NotFoundError("model connection not found", details={"conn_id": conn_id})
    # Clear the KB-local default of that role so the org adoption is unambiguous, then record it.
    for c in await store.list(kb_id, org_conn.role):
        if c.is_default:
            await store.put(c.model_copy(update={"is_default": False}))
    await _set_org_default_conn(kb_id, org_conn.role, conn_id)
    wiring.invalidate_tenant_gateway(kb_id)
    if org_conn.role == "embedding":
        await embedder_switch.on_embedding_default_change(kb_id)
    return _conn_item(org_conn.model_copy(update={"is_default": True}), "org")


@router.post(
    "/kb/{kb_id}/model-connections/{conn_id}/default",
    response_model=ModelConnectionItem | ModelRequestInfo,
)
async def default_model_connection(
    kb_id: str, conn_id: str, response: Response, principal: Annotated[Principal, Depends(tenant_guard)]
) -> ModelConnectionItem | ModelRequestInfo:
    """Make this connection the KB's default (the model answers use unless overridden).
    An org owner/admin applies immediately; a KB manager creates a PENDING request (202)."""
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only an owner/admin or a manager of this KB can configure its models")
    if _is_org_admin(principal):
        return await _apply_default_connection(kb_id, conn_id, principal)
    return await _submit_model_request(kb_id, "default", conn_id, None, principal, response)


async def _apply_delete_connection(kb_id: str, conn_id: str, principal: Principal) -> None:
    """Apply a 'delete connection' — shared by the direct org-admin write and the approve path."""
    store = wiring.model_connection_store()
    victim = await store.get(kb_id, conn_id)
    if victim is not None:
        # 409 while the KB's documents live in this embedder's vector space.
        await embedder_switch.guard_delete(kb_id, victim)
    await store.delete(kb_id, conn_id)
    if victim and victim.is_default:
        remaining = await store.list(kb_id, victim.role)
        if remaining:
            await store.set_default(kb_id, remaining[0].conn_id)
    wiring.invalidate_tenant_gateway(kb_id)
    if victim and victim.role == "embedding":
        await embedder_switch.on_embedding_default_change(kb_id)


@router.delete(
    "/kb/{kb_id}/model-connections/{conn_id}",
    responses={204: {"description": "Deleted."}, 202: {"model": ModelRequestInfo}},
)
async def remove_model_connection(
    kb_id: str, conn_id: str, response: Response, principal: Annotated[Principal, Depends(tenant_guard)]
) -> ModelRequestInfo | None:
    """Remove a connection. If it was the default, another is promoted; with none left,
    answers revert to the platform default model. An org owner/admin applies immediately (204); a
    KB manager creates a PENDING request (202) an org admin must approve."""
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only an owner/admin or a manager of this KB can configure its models")
    if await _get_org_connection(principal.tenant_id, conn_id) is not None:
        raise ForbiddenError("this is an org-wide model — remove it under Org models, not per KB")
    if _is_org_admin(principal):
        await _apply_delete_connection(kb_id, conn_id, principal)
        response.status_code = 204
        return None
    return await _submit_model_request(kb_id, "delete", conn_id, None, principal, response)


@router.post("/kb/{kb_id}/model-connections/test", response_model=ModelTestResult)
async def test_model_connection(
    kb_id: str, body: ModelConnectionRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> ModelTestResult:
    """Test a candidate endpoint before saving. Role-aware: generation lists the deployed
    models; embedding probes the dimension the endpoint actually returns; reranking probes
    which wire shape it speaks and on what scale it scores. What the probe learns is what
    the console stores on the connection — the runtime never guesses."""
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only an owner/admin or a manager of this KB can configure its models")
    return await _probe_model(body)


async def _probe_model(body: ModelConnectionRequest) -> ModelTestResult:
    """Role-aware endpoint probe, sovereignty-gated. Shared by the per-KB and org-wide test
    endpoints so both learn the same thing (models list / embedding dim / rerank wire shape)."""
    assert_sovereign(body.endpoint)
    if body.role == "embedding":
        from rag_clients.openai_embedder import probe_embedding_dim  # noqa: PLC0415

        try:
            dim = await probe_embedding_dim(
                body.endpoint, body.model, api_key=body.api_key or "", api_key_header=body.header
            )
        except Exception as exc:
            return ModelTestResult(ok=False, error=str(exc)[:200])
        return ModelTestResult(ok=True, dim=dim, available=await _list_models_best_effort(body))
    if body.role == "reranking":
        from rag_clients.reranker import probe_rerank_shape  # noqa: PLC0415

        try:
            shape, sample = await probe_rerank_shape(
                body.endpoint, body.model, api_key=body.api_key or "", api_key_header=body.header
            )
        except Exception as exc:
            return ModelTestResult(ok=False, error=str(exc)[:200])
        return ModelTestResult(
            ok=True, api_shape=shape, score_sample=round(sample, 4), available=await _list_models_best_effort(body)
        )
    gw = OpenAICompatGateway(
        url=body.endpoint, model=body.model, api_key=body.api_key or "", api_key_header=body.header
    )
    try:
        return ModelTestResult(ok=True, available=await gw.list_models())
    except Exception as exc:
        return ModelTestResult(ok=False, error=str(exc)[:200])
    finally:
        await gw.aclose()


async def _list_models_best_effort(body: ModelConnectionRequest) -> list[str]:
    """The /models catalog for the console's picker chips — nice to have, never fatal
    (many embedding/rerank servers don't expose it)."""
    gw = OpenAICompatGateway(
        url=body.endpoint, model=body.model, api_key=body.api_key or "", api_key_header=body.header
    )
    try:
        return await gw.list_models()
    except Exception:
        return []
    finally:
        await gw.aclose()


# --- Org-wide model connections (ORG ADMIN ONLY, NOT approval-gated) --------------------------
# An org admin is the authority for org-wide models: these writes apply immediately (no request
# queue). Every write enforces the sovereignty egress allowlist on the endpoint host exactly like
# the per-KB add. Keys are encrypted at rest with the ORG's DEK. Availability, not default: these
# surface READ-ONLY in every KB's model list (scope="org", see list_model_connections) and never
# change a KB's default resolution unless a KB explicitly adopts one (see _apply_default_connection).


@router.get("/org/model-connections", response_model=list[ModelConnectionItem])
async def list_org_model_connections(
    principal: Annotated[Principal, Depends(get_principal)],
) -> list[ModelConnectionItem]:
    """Every org-wide model connection (org admin only)."""
    if not _is_org_admin(principal):
        raise ForbiddenError("only an owner or admin can manage org-wide models")
    return [_conn_item(c, "org") for c in await _load_org_connections(principal.tenant_id)]


async def _apply_org_add_connection(org: str, body: AddModelConnectionRequest) -> ModelConnectionItem:
    """Add a connection to the ORG registry — shared by a direct org-admin write AND the approve
    path when an org admin approves a KB manager's pending org_add request."""
    assert_sovereign(body.endpoint)
    _require_role_fields(body)
    conns = await _load_org_connections(org)
    role_conns = [c for c in conns if c.role == body.role]
    conn = StoredModelConnection(
        tenant_id=org,
        conn_id=str(uuid.uuid4()),
        name=body.name,
        endpoint=body.endpoint,
        model=body.model,
        header=body.header,
        key_ciphertext=(await tenant_crypto.encrypt(org, body.api_key) if body.api_key else ""),
        is_default=body.make_default or not role_conns,
        role=body.role,
        embed_dim=body.embed_dim,
        api_shape=body.api_shape,
        backend=body.backend,
    )
    if conn.is_default:
        conns = [c.model_copy(update={"is_default": False}) if c.role == conn.role else c for c in conns]
    conns.append(conn)
    await _save_org_connections(org, conns)
    _log.info(
        "org_model_connection_added",
        org_id=org,
        conn_id=conn.conn_id,
        role=conn.role,
        endpoint=body.endpoint,
        model=body.model,
    )
    if conn.role == "embedding" and conn.is_default:
        # Every other role is in force the moment it is added. An embedder cannot be, because the
        # vectors already in each KB came from a different model — so this stages the switch per KB
        # and the operator applies it once (POST /org/embedder/apply), which re-embeds. KBs with
        # nothing indexed, or already on that vector space, take it immediately.
        await embedder_switch.on_org_embedding_change(org, conn)
    return _conn_item(conn, "org")


@router.post(
    "/org/model-connections",
    status_code=201,
    response_model=ModelConnectionItem | ModelRequestInfo,
)
async def add_org_model_connection(
    body: AddModelConnectionRequest,
    response: Response,
    principal: Annotated[Principal, Depends(get_principal)],
) -> ModelConnectionItem | ModelRequestInfo:
    """Register an org-wide GENERATION model. Endpoint must be egress-allowlisted (403 otherwise);
    the key is encrypted at rest with the org DEK. An org owner/admin applies IMMEDIATELY. A KB
    manager (manages at least one KB but is not an org admin) instead creates a PENDING org-wide
    request (202) that an org admin must approve. Anyone else → 403."""
    if _is_org_admin(principal):
        return await _apply_org_add_connection(principal.tenant_id, body)
    # A KB manager may REQUEST an org-wide model; a plain member cannot.
    if await _managed_kbs(principal.tenant_id, principal.subject):
        return await _submit_org_model_request(body.model_dump(), principal, response)
    raise ForbiddenError("only an owner/admin or a KB manager can add or request an org-wide model")


@router.put("/org/model-connections/{conn_id}", response_model=ModelConnectionItem)
async def update_org_model_connection(
    conn_id: str,
    body: AddModelConnectionRequest,
    principal: Annotated[Principal, Depends(get_principal)],
) -> ModelConnectionItem:
    """Update an org-wide connection (org admin only). Blank api_key keeps the stored key; the
    role is fixed at creation (as per-KB)."""
    if not _is_org_admin(principal):
        raise ForbiddenError("only an owner or admin can manage org-wide models")
    org = principal.tenant_id
    assert_sovereign(body.endpoint)
    conns = await _load_org_connections(org)
    existing = next((c for c in conns if c.conn_id == conn_id), None)
    if existing is None:
        raise NotFoundError("model connection not found", details={"conn_id": conn_id})
    if body.role != existing.role:
        raise ValidationError(
            "a connection's role cannot change — remove it and add a new one",
            details={"role": existing.role},
        )
    merged_dim = body.embed_dim if body.embed_dim is not None else existing.embed_dim
    merged_shape = body.api_shape or existing.api_shape
    if existing.role == "embedding" and merged_dim is None:
        raise ValidationError("embed_dim is required for an embedding model — run Test to detect it")
    if existing.role == "reranking" and not merged_shape:
        raise ValidationError("api_shape is required for a reranking model — run Test to detect it")
    ciphertext = await tenant_crypto.encrypt(org, body.api_key) if body.api_key else existing.key_ciphertext
    updated = StoredModelConnection(
        tenant_id=org,
        conn_id=conn_id,
        name=body.name,
        endpoint=body.endpoint,
        model=body.model,
        header=body.header,
        key_ciphertext=ciphertext,
        is_default=body.make_default or existing.is_default,
        role=existing.role,
        embed_dim=merged_dim,
        api_shape=merged_shape,
        backend=body.backend,
    )
    new_conns: list[StoredModelConnection] = []
    for c in conns:
        if c.conn_id == conn_id:
            new_conns.append(updated)
        elif updated.is_default and c.role == updated.role:
            new_conns.append(c.model_copy(update={"is_default": False}))
        else:
            new_conns.append(c)
    await _save_org_connections(org, new_conns)
    if updated.role == "embedding" and updated.is_default:
        await embedder_switch.on_org_embedding_change(org, updated)
    return _conn_item(updated, "org")


@router.delete("/org/model-connections/{conn_id}", status_code=204)
async def remove_org_model_connection(
    conn_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
) -> None:
    """Remove an org-wide connection (org admin only). KBs that had adopted it as their default
    fall back to their own default/platform default on the next resolution (dangling adoption is
    handled safely there — the org lookup simply returns None)."""
    if not _is_org_admin(principal):
        raise ForbiddenError("only an owner or admin can manage org-wide models")
    org = principal.tenant_id
    conns = await _load_org_connections(org)
    victim = next((c for c in conns if c.conn_id == conn_id), None)
    if victim is None:
        raise NotFoundError("model connection not found", details={"conn_id": conn_id})
    if victim.role == "embedding":
        await _guard_org_embedder_delete(org, victim)
    remaining = [c for c in conns if c.conn_id != conn_id]
    # Promote another connection of that role to default if the removed one was the org default.
    if victim.is_default:
        for c in remaining:
            if c.role == victim.role:
                remaining = [
                    rc.model_copy(update={"is_default": rc.conn_id == c.conn_id}) if rc.role == victim.role else rc
                    for rc in remaining
                ]
                break
    await _save_org_connections(org, remaining)


async def _guard_org_embedder_delete(org: str, victim: StoredModelConnection) -> None:
    """Refuse removing the org embedder that BUILT a KB's vectors while that KB holds documents.

    Deleting it does not un-embed anything: the collection stays full of vectors only that model
    can produce queries for, and the platform embedder cannot answer against a space it did not
    build. The operator must register a replacement and apply the re-embed first."""
    held: list[str] = []
    for kb_id in await embedder_switch.org_kb_ids(org):
        active = embedder_switch.parse_record((await get_catalog().get_tenant_config(kb_id)).get("embedder_active", ""))
        if active and active.get("scope") == "org" and active.get("conn_id") == victim.conn_id:
            if await get_catalog().list_documents(kb_id):
                held.append(kb_id)
            else:  # empty KB: nothing depends on the space, release it
                await get_catalog().set_tenant_config(kb_id, "embedder_active", "")
                await get_catalog().set_tenant_config(kb_id, "embedder_pending", "")
                wiring.invalidate_tenant_gateway(kb_id)
    if held:
        raise ConflictError(
            "this embedding model built the vectors of knowledge bases that still hold documents — "
            "register a replacement and apply the re-embed before removing it",
            details={"conn_id": victim.conn_id, "model": victim.model, "knowledge_bases": held},
        )


class OrgEmbedderStatus(BaseModel):
    """Whether an org-wide embedder switch is waiting to be applied, and what it would move."""

    pending: dict[str, Any] | None = Field(None, description="The staged embedder; null when none is waiting.")
    kbs: list[dict[str, Any]] = Field(default_factory=list, description="Each KB waiting, with its document count.")
    documents: int = Field(0, description="Total documents that would be re-embedded.")
    apply_enabled: bool = True


class OrgEmbedderApplyResult(BaseModel):
    applied: list[dict[str, Any]] = Field(default_factory=list)
    failed: list[dict[str, Any]] = Field(default_factory=list)


@router.get("/org/embedder/status", response_model=OrgEmbedderStatus)
async def org_embedder_status(principal: Annotated[Principal, Depends(get_principal)]) -> OrgEmbedderStatus:
    """What the console's org-wide banner shows before an operator commits to a re-embed."""
    if not _is_org_admin(principal):
        raise ForbiddenError("only an owner or admin can manage org-wide models")
    return OrgEmbedderStatus(**await embedder_switch.org_status(principal.tenant_id))


@router.post("/org/embedder/apply", response_model=OrgEmbedderApplyResult)
async def org_embedder_apply(principal: Annotated[Principal, Depends(get_principal)]) -> OrgEmbedderApplyResult:
    """Re-embed every KB of the org onto the staged org-wide embedder.

    Deliberately one explicit action rather than a side effect of adding the model: it rewrites
    every vector the org holds, and during each KB's rebuild dense retrieval degrades to
    BM25+rerank. One KB failing does not stop the others — its error is reported against it."""
    if not _is_org_admin(principal):
        raise ForbiddenError("only an owner or admin can manage org-wide models")
    return OrgEmbedderApplyResult(**await embedder_switch.apply_org(principal.tenant_id, actor=principal.subject))


@router.post("/org/model-connections/test", response_model=ModelTestResult)
async def test_org_model_connection(
    body: ModelConnectionRequest,
    principal: Annotated[Principal, Depends(get_principal)],
) -> ModelTestResult:
    """Probe a candidate org-wide endpoint before saving. Same role-aware probe as the per-KB test.

    Open to an org admin AND to a KB manager, because a manager may REQUEST an org-wide model and
    an embedding/reranking request is INVALID without the probed dimension/wire shape — gating the
    probe above the request would leave the manager with a form they cannot complete. The probe
    itself is a read-only reachability check against an already egress-allowlisted endpoint."""
    if not _is_org_admin(principal) and not await _managed_kbs(principal.tenant_id, principal.subject):
        raise ForbiddenError("only an owner/admin or a KB manager can test an org-wide model")
    return await _probe_model(body)


class EmbedderStatus(BaseModel):
    """The KB's embedding state: which embedder owns the vector space (null = platform),
    what is staged, and whether the apply path is enabled."""

    active: dict[str, Any] | None = None
    pending: dict[str, Any] | None = None
    documents: int = 0
    apply_enabled: bool = True


class EmbedderApplyResult(BaseModel):
    active: dict[str, Any]
    reindexed: int
    failed: list[str] = Field(default_factory=list)


@router.get("/kb/{kb_id}/embedder/status", response_model=EmbedderStatus)
async def embedder_status(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> EmbedderStatus:
    """Whether an embedder switch is pending for this KB (console banner)."""
    return EmbedderStatus(**await embedder_switch.status(kb_id))


async def _apply_embedder_apply(kb_id: str, principal: Principal) -> EmbedderApplyResult:
    """Apply the pending embedder switch — shared by the direct org-admin write and the approve path."""
    return EmbedderApplyResult(**await embedder_switch.apply(kb_id, actor=principal.subject))


@router.post("/kb/{kb_id}/embedder/apply", response_model=EmbedderApplyResult | ModelRequestInfo)
async def embedder_apply(
    kb_id: str, response: Response, principal: Annotated[Principal, Depends(tenant_guard)]
) -> EmbedderApplyResult | ModelRequestInfo:
    """Apply the pending embedder switch: recreate the vector collection at the new
    dimension and re-embed every document through the write-then-swap reindex. Dense
    retrieval degrades to BM25+rerank during the rebuild. An org owner/admin applies
    immediately; a KB manager creates a PENDING request (202) an org admin must approve."""
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only an owner/admin or a manager of this KB can apply its embedder")
    if _is_org_admin(principal):
        return await _apply_embedder_apply(kb_id, principal)
    return await _submit_model_request(kb_id, "embedder_apply", None, None, principal, response)


@router.get("/org/model-requests", response_model=list[ModelRequestInfo])
async def list_org_model_requests(
    principal: Annotated[Principal, Depends(get_principal)],
) -> list[ModelRequestInfo]:
    """ORG-WIDE (org_add) model requests. An org owner/admin sees ALL of them (the approval queue);
    a KB manager sees only their OWN (filtered by requested_by) so they can watch the status. A
    plain member gets 403."""
    org = principal.tenant_id
    is_admin = _is_org_admin(principal)
    if not is_admin and not await _managed_kbs(org, principal.subject):
        raise ForbiddenError("only an owner/admin or a KB manager can view org model requests")
    me = principal.subject.strip().lower()
    rows = [r for r in await _load_model_requests(org) if r.get("action") == "org_add"]
    if not is_admin:
        rows = [r for r in rows if str(r.get("requested_by", "")).lower() == me]
    return [ModelRequestInfo(**r) for r in rows]


@router.post("/org/model-requests/{request_id}/decide", response_model=ModelRequestInfo)
async def decide_org_model_request(
    request_id: str,
    body: ModelDecisionBody,
    principal: Annotated[Principal, Depends(get_principal)],
) -> ModelRequestInfo:
    """Approve or reject a KB manager's pending ORG-WIDE (org_add) model request. Org owner/admin
    ONLY. On approve the requested model is added to the org registry via _apply_org_add_connection;
    on reject it is simply marked rejected."""
    if not _is_org_admin(principal):
        raise ForbiddenError("only an owner or admin can decide org model requests")
    org = principal.tenant_id
    rows = await _load_model_requests(org)
    row = next((r for r in rows if r.get("id") == request_id and r.get("action") == "org_add"), None)
    if row is None:
        raise NotFoundError("org model request not found", details={"id": request_id})
    if row.get("status") != "pending":
        raise ConflictError(f"request already {row.get('status')}", details={"id": request_id})
    if body.decision == "approve":
        payload = row.get("payload")
        await _apply_org_add_connection(org, AddModelConnectionRequest(**(payload or {})))  # type: ignore[arg-type]
        row["status"] = "approved"
    else:
        row["status"] = "rejected"
    row["decided_by"] = principal.subject
    row["decided_at"] = datetime.now(UTC).isoformat()
    row["reason"] = body.reason.strip()
    await _save_model_requests(org, rows)
    await wiring.audit_service().record(
        tenant_id=org,
        trace_id=str(uuid.uuid4()),
        action=AuditAction.MODEL_REQUEST,
        actor=principal.subject,
        payload={"op": "decide", "request_id": request_id, "status": row["status"], "action": "org_add"},
    )
    _log.info(
        "org_model_request_decided",
        org_id=org,
        request_id=request_id,
        status=row["status"],
        by=principal.subject,
    )
    return ModelRequestInfo(**row)


@router.get("/kb/{kb_id}/model-requests", response_model=list[ModelRequestInfo])
async def list_kb_model_requests(
    kb_id: str, principal: Annotated[Principal, Depends(get_principal)]
) -> list[ModelRequestInfo]:
    """This KB's model-change requests — visible to an org owner/admin OR a manager of this KB
    (so a manager can watch the status of what they submitted)."""
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only an owner/admin or a manager of this KB may view its model requests")
    name = await _kb_name(principal.tenant_id, kb_id)
    return [
        ModelRequestInfo(kb_name=name, **r)
        for r in await _load_model_requests(principal.tenant_id)
        if r.get("kb_id") == kb_id
    ]


async def _apply_model_request(row: dict[str, object], principal: Principal) -> None:
    """Re-apply a stored, approved model request by dispatching on its recorded action + payload +
    conn_id to the same _apply_* function the direct org-admin write uses."""
    kb_id = str(row["kb_id"])
    action = str(row["action"])
    conn_id = row.get("conn_id")
    payload = row.get("payload")
    if action == "add":
        await _apply_add_connection(kb_id, AddModelConnectionRequest(**(payload or {})), principal)  # type: ignore[arg-type]
    elif action == "update":
        await _apply_update_connection(kb_id, str(conn_id), AddModelConnectionRequest(**(payload or {})), principal)  # type: ignore[arg-type]
    elif action == "default":
        await _apply_default_connection(kb_id, str(conn_id), principal)
    elif action == "delete":
        await _apply_delete_connection(kb_id, str(conn_id), principal)
    elif action == "embedder_apply":
        await _apply_embedder_apply(kb_id, principal)
    else:  # pragma: no cover - guarded by the pattern at submit time
        raise ValidationError("unknown model-request action", details={"action": action})


@router.post("/kb/{kb_id}/model-requests/{request_id}/decide", response_model=ModelRequestInfo)
async def decide_model_request(
    kb_id: str,
    request_id: str,
    body: ModelDecisionBody,
    principal: Annotated[Principal, Depends(get_principal)],
) -> ModelRequestInfo:
    """Approve or reject a KB manager's pending model-change request. Org owner/admin ONLY (they are
    the approver — no self-approval since a manager cannot reach this). On approve the stored action
    is re-applied via the matching _apply_* function; on reject it is simply marked rejected."""
    if not _is_org_admin(principal):
        raise ForbiddenError("only an owner or admin can decide model requests")
    org = principal.tenant_id
    rows = await _load_model_requests(org)
    row = next((r for r in rows if r.get("id") == request_id and r.get("kb_id") == kb_id), None)
    if row is None:
        raise NotFoundError("model request not found", details={"id": request_id})
    if row.get("status") != "pending":
        raise ConflictError(f"request already {row.get('status')}", details={"id": request_id})
    if body.decision == "approve":
        await _apply_model_request(row, principal)
        row["status"] = "approved"
    else:
        row["status"] = "rejected"
    row["decided_by"] = principal.subject
    row["decided_at"] = datetime.now(UTC).isoformat()
    row["reason"] = body.reason.strip()
    await _save_model_requests(org, rows)
    await wiring.audit_service().record(
        tenant_id=kb_id,
        trace_id=str(uuid.uuid4()),
        action=AuditAction.MODEL_REQUEST,
        actor=principal.subject,
        payload={"op": "decide", "request_id": request_id, "status": row["status"], "action": row["action"]},
    )
    _log.info(
        "model_request_decided",
        org_id=org,
        kb_id=kb_id,
        request_id=request_id,
        status=row["status"],
        by=principal.subject,
    )
    return ModelRequestInfo(kb_name=await _kb_name(org, kb_id), **row)


class ModelStatus(BaseModel):
    """Which model answers this KB, and whether it is reachable right now (console indicator)."""

    source: str  # "connected" = the KB's own attached model; "default" = platform fallback
    name: str  # connection label, or "Platform default"
    model: str  # model / deployment id
    endpoint: str | None = None  # host only (never the full URL/key) for a connected model
    online: bool | None = None  # True reachable · False unreachable · None not probeable
    latency_ms: int | None = None
    detail: str = ""


# Short TTL so the topbar indicator can poll without hammering an external endpoint.
_model_status_cache: dict[str, tuple[float, ModelStatus]] = {}
_MODEL_STATUS_TTL = 20.0


@router.get("/kb/{kb_id}/model-status", response_model=ModelStatus)
async def model_status(
    kb_id: str,
    principal: Annotated[Principal, Depends(tenant_guard)],
    refresh: Annotated[bool, Query(description="Bypass the cache and re-probe now.")] = False,
) -> ModelStatus:
    """Live status of the model answering this KB: is it the KB's own connected model or the
    platform default, and is that endpoint reachable right now. Powers the console indicator."""
    now = time.monotonic()
    if not refresh:
        cached = _model_status_cache.get(kb_id)
        if cached and now - cached[0] < _MODEL_STATUS_TTL:
            return cached[1]

    # KB-level default first, then the ORG-WIDE default (generation models are org-wide now) — so the
    # indicator matches what actually answers (resolve_tenant_gateway below), not a dead platform
    # default. Without this the subtitle showed "Platform default: <llm_model>" even though the org
    # had connected its own model.
    conn = await wiring.model_connection_store().get_default(kb_id)
    if conn is None:
        conn = await wiring._resolve_org_default_connection(kb_id, "generation")
    if conn is not None:
        source, name, model = "connected", conn.name, conn.model
        host = urlparse(conn.endpoint).hostname
        endpoint: str | None = host or conn.endpoint
    else:
        source, name, model = "default", "Platform default", get_settings().llm_model
        endpoint = None

    gw = await wiring.resolve_tenant_gateway(kb_id)
    online: bool | None = None
    latency_ms: int | None = None
    detail = ""
    probe = getattr(gw, "list_models", None)
    if probe is None:
        detail = "no health endpoint"
    else:
        t = time.monotonic()
        try:
            models = await asyncio.wait_for(probe(), timeout=8.0)
            online = True
            latency_ms = int((time.monotonic() - t) * 1000)
            detail = "reachable · " + ", ".join(models[:3]) if models else "reachable"
        except Exception as exc:  # unreachable / auth / timeout → offline, never 500
            online = False
            latency_ms = int((time.monotonic() - t) * 1000)
            detail = str(exc)[:140] or "unreachable"

    status = ModelStatus(
        source=source,
        name=name,
        model=model,
        endpoint=endpoint,
        online=online,
        latency_ms=latency_ms,
        detail=detail,
    )
    _model_status_cache[kb_id] = (now, status)
    return status


class ConnectionStatus(BaseModel):
    """Live reachability of ONE registered connection, probed in its own protocol."""

    conn_id: str
    online: bool
    latency_ms: int | None = None
    detail: str = ""


_conn_status_cache: dict[tuple[str, str], tuple[float, ConnectionStatus]] = {}


@router.get("/kb/{kb_id}/model-connections/{conn_id}/status", response_model=ConnectionStatus)
async def model_connection_status(
    kb_id: str,
    conn_id: str,
    principal: Annotated[Principal, Depends(tenant_guard)],
    refresh: Annotated[bool, Query(description="Bypass the cache and re-probe now.")] = False,
) -> ConnectionStatus:
    """Probe a stored connection with the STORED (decrypted server-side) key, each role in
    its own protocol: generation lists models, embedding embeds a probe (and confirms the
    stored dimension still holds), reranking scores a probe pair (and confirms the stored
    wire shape). Unreachable/auth failures return online=false — never a 500."""
    now = time.monotonic()
    if not refresh:
        cached = _conn_status_cache.get((kb_id, conn_id))
        if cached and now - cached[0] < _MODEL_STATUS_TTL:
            return cached[1]

    conn = await wiring.model_connection_store().get(kb_id, conn_id)
    # Fall back to the org-wide registry so an org-wide connection surfaced in this KB's list
    # (scope="org") can be probed too; its key is encrypted with the ORG's DEK.
    crypto_tenant = kb_id
    if conn is None:
        conn = await _get_org_connection(principal.tenant_id, conn_id)
        crypto_tenant = principal.tenant_id
    if conn is None:
        raise NotFoundError("model connection not found", details={"conn_id": conn_id})
    key = await tenant_crypto.decrypt(crypto_tenant, conn.key_ciphertext) if conn.key_ciphertext else ""

    t = time.monotonic()
    online, detail = False, ""
    try:
        if conn.role == "embedding":
            from rag_clients.openai_embedder import probe_embedding_dim  # noqa: PLC0415

            dim = await asyncio.wait_for(
                probe_embedding_dim(conn.endpoint, conn.model, api_key=key, api_key_header=conn.header),
                timeout=10.0,
            )
            online = True
            detail = f"reachable · {dim}-dim"
            if conn.embed_dim is not None and dim != conn.embed_dim:
                detail = f"DIMENSION DRIFT: endpoint returns {dim}, connection stored {conn.embed_dim}"
        elif conn.role == "reranking":
            from rag_clients.reranker import probe_rerank_shape  # noqa: PLC0415

            shape, _sample = await asyncio.wait_for(
                probe_rerank_shape(conn.endpoint, conn.model, api_key=key, api_key_header=conn.header),
                timeout=10.0,
            )
            online = True
            detail = f"reachable · {shape}"
            if conn.api_shape and shape != conn.api_shape:
                detail = f"SHAPE DRIFT: endpoint speaks {shape}, connection stored {conn.api_shape}"
        else:
            gw = OpenAICompatGateway(url=conn.endpoint, model=conn.model, api_key=key, api_key_header=conn.header)
            try:
                models = await asyncio.wait_for(gw.list_models(), timeout=8.0)
            finally:
                await gw.aclose()
            online = True
            detail = "reachable" if conn.model in models or not models else f"model not in catalog: {conn.model}"
    except Exception as exc:  # unreachable / auth / timeout → offline, never 500
        online = False
        detail = str(exc)[:140] or "unreachable"

    status = ConnectionStatus(
        conn_id=conn_id, online=online, latency_ms=int((time.monotonic() - t) * 1000), detail=detail
    )
    _conn_status_cache[(kb_id, conn_id)] = (now, status)
    return status


# --- Envelope encryption / BYOK (FR-H9) --------------------------------------
class CryptoStatus(BaseModel):
    """Non-secret crypto posture for a KB (console Compliance + attestation)."""

    envelope_encryption: bool
    kms_backend: str
    byok: bool
    kms_key_id_configured: bool
    dek_versions: int
    provisioned: bool


@router.get("/kb/{kb_id}/crypto/status", response_model=CryptoStatus)
async def get_crypto_status(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> CryptoStatus:
    """This KB's key posture: envelope encryption, KMS/BYOK backend, DEK versions."""
    return CryptoStatus(**await tenant_crypto.status(kb_id))


class CryptoRotateResult(BaseModel):
    rotated: bool
    dek_versions: int
    reencrypted_connections: int


@router.post("/kb/{kb_id}/crypto/rotate", response_model=CryptoRotateResult)
async def rotate_crypto_key(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> CryptoRotateResult:
    """Rotate the KB's Data Encryption Key: mint a new current DEK, re-encrypt the stored
    model keys with it, then retire the old DEK. Admin-only; audited."""
    _require_role(principal, "admin")
    versions = await tenant_crypto.rotate(kb_id)
    # Re-encrypt every stored secret with the new current DEK so the old DEK can be retired.
    store = wiring.model_connection_store()
    reencrypted = 0
    for conn in await store.list(kb_id):
        if conn.key_ciphertext:
            plaintext = await tenant_crypto.decrypt(kb_id, conn.key_ciphertext)
            await store.put(conn.model_copy(update={"key_ciphertext": await tenant_crypto.encrypt(kb_id, plaintext)}))
            reencrypted += 1
    await tenant_crypto.prune(kb_id)  # drop retired DEKs now that nothing references them
    wiring.invalidate_tenant_gateway(kb_id)
    await wiring.audit_service().record(
        tenant_id=kb_id,
        trace_id=str(uuid.uuid4()),
        action=AuditAction.KEY_ROTATION,
        actor=principal.subject,
        payload={"reencrypted_connections": reencrypted, "dek_versions": versions},
    )
    _log.info("crypto_key_rotated", kb_id=kb_id, reencrypted=reencrypted)
    return CryptoRotateResult(rotated=True, dek_versions=versions, reencrypted_connections=reencrypted)


# --- Responsible-AI compliance pack (FR-H8+) ---------------------------------
@router.get("/kb/{kb_id}/compliance/frameworks", response_model=responsible_ai.ComplianceReport)
async def get_compliance_frameworks(
    kb_id: str,
    principal: Annotated[Principal, Depends(tenant_guard)],
    framework: str | None = None,
) -> responsible_ai.ComplianceReport:
    """Responsible-AI framework attestation: the platform's live controls mapped to NITI Aayog,
    DPDP, RBI, SEBI, IRDAI, ISO 42001 and OWASP-LLM requirements.
    Optional ``?framework=niti|dpdp|rbi|sebi|irdai|iso42001|owasp``."""
    return await responsible_ai.report(kb_id, only=framework)


# --- Governed NL→SQL over a tenant RDBMS (FR-D10) — one KB, many DB connections ----
@router.get("/kb/{kb_id}/sql/connection", response_model=sql_query.SqlConnectionInfo)
async def get_sql_connection(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> sql_query.SqlConnectionInfo:
    """The KB's DEFAULT DB connection (non-secret) — quick-check/back-compat single view.
    Use GET .../sql/connections for the full list."""
    return await sql_query.get_default_info(kb_id)


@router.get("/kb/{kb_id}/sql/connections", response_model=list[sql_query.SqlConnectionItem])
async def list_sql_connections(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> list[sql_query.SqlConnectionItem]:
    """Every DB connection registered for this KB (default first)."""
    return await sql_query.list_connections(kb_id)


class AddSqlConnectionRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Human label for this connection, e.g. 'Sales DB'.")
    dsn: str = Field(..., description="SQLAlchemy DSN; host must be egress-allowlisted (SQLite exempt).")
    tables: list[str] = Field(..., min_length=1, description="Allowlist of readable tables.")
    dialect: str = Field(default="postgres")
    make_default: bool = Field(default=True, description="Query with this connection when none is named.")


async def _autosync_sources_to_graph(kb_id: str) -> None:
    """Best-effort: seed the KB graph with the connected sources' structural entities (table names)
    the moment a source is connected/updated, so the schema is in GraphRAG without a manual
    graph-sync step. No-op unless graphrag is enabled; never fails the connect (additive)."""
    try:
        await graph_sync.sync_kb_sources(kb_id)
    except Exception as exc:
        _log.warning("auto_graph_sync_failed", kb_id=kb_id, error=str(exc)[:160])


@router.post("/kb/{kb_id}/sql/connections", status_code=201, response_model=sql_query.SqlConnectionItem)
async def add_sql_connection(
    kb_id: str, body: AddSqlConnectionRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> sql_query.SqlConnectionItem:
    """Register a new read-only DB connection (admin). DSN stored envelope-encrypted."""
    _require_role(principal, "admin")
    item = await sql_query.add_connection(kb_id, body.name, body.dsn, body.tables, body.dialect, body.make_default)
    await _autosync_sources_to_graph(kb_id)
    return item


class UpdateSqlConnectionRequest(BaseModel):
    name: str = Field(..., min_length=1)
    dsn: str | None = Field(default=None, description="Omit to keep the existing DSN; set to rotate it.")
    tables: list[str] = Field(..., min_length=1)
    dialect: str = Field(default="postgres")


@router.put("/kb/{kb_id}/sql/connections/{conn_id}", response_model=sql_query.SqlConnectionItem)
async def update_sql_connection(
    kb_id: str, conn_id: str, body: UpdateSqlConnectionRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> sql_query.SqlConnectionItem:
    """Update a connection's label/table allowlist, and optionally rotate its DSN (admin)."""
    _require_role(principal, "admin")
    item = await sql_query.update_connection(kb_id, conn_id, body.name, body.dsn, body.tables, body.dialect)
    await _autosync_sources_to_graph(kb_id)  # re-seed: the table allowlist may have changed
    return item


@router.post("/kb/{kb_id}/sql/connections/{conn_id}/default", response_model=sql_query.SqlConnectionItem)
async def default_sql_connection(
    kb_id: str, conn_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> sql_query.SqlConnectionItem:
    """Make this the connection a query uses when none is named explicitly (admin)."""
    _require_role(principal, "admin")
    return await sql_query.set_default_connection(kb_id, conn_id)


@router.delete("/kb/{kb_id}/sql/connections/{conn_id}", status_code=204)
async def remove_sql_connection(
    kb_id: str, conn_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> None:
    """Remove a DB connection (admin)."""
    _require_role(principal, "admin")
    await sql_query.delete_connection(kb_id, conn_id)


class SqlAskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    connection_id: str | None = Field(default=None, description="Which connection to ask; default when omitted.")


@router.post("/kb/{kb_id}/sql/query", response_model=sql_query.SqlAnswer)
async def ask_sql(
    kb_id: str, body: SqlAskRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> sql_query.SqlAnswer:
    """Ask a natural-language question over a registered DB (NL→SQL, governed, audited)."""
    _require_role(principal, "ingestor")
    check_query_rate(principal.tenant_id)
    return await sql_query.ask(kb_id, body.question, connection_id=body.connection_id, actor=principal.subject)


class SqlExecuteRequest(BaseModel):
    sql: str = Field(..., min_length=1, description="A read-only SELECT; validated + bounded before it runs.")
    connection_id: str | None = Field(default=None, description="Which connection to run against; default if omitted.")


@router.post("/kb/{kb_id}/sql/execute", response_model=sql_query.SqlAnswer)
async def execute_sql(
    kb_id: str, body: SqlExecuteRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> sql_query.SqlAnswer:
    """Run an analyst-supplied SELECT through the SAME governance (validated, bounded, audited)."""
    _require_role(principal, "ingestor")
    check_query_rate(principal.tenant_id)
    return await sql_query.execute_sql(kb_id, body.sql, connection_id=body.connection_id, actor=principal.subject)


# --- Unified KB connectors: NoSQL / REST / SharePoint (docs+sources fusion) ---
class NoSqlConnectionRequest(BaseModel):
    uri: str = Field(..., min_length=1)
    database: str = Field(..., min_length=1)
    collections: list[str] = Field(..., min_length=1)


@router.get("/kb/{kb_id}/nosql/connection", response_model=nosql_query.NoSqlConnectionInfo)
async def get_kb_nosql_connection(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> nosql_query.NoSqlConnectionInfo:
    """The KB's registered read-only NoSQL connection (non-secret: db + collection allowlist)."""
    return await nosql_query.get_nosql_connection(kb_id)


@router.put("/kb/{kb_id}/nosql/connection", response_model=nosql_query.NoSqlConnectionInfo)
async def set_kb_nosql_connection(
    kb_id: str, body: NoSqlConnectionRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> nosql_query.NoSqlConnectionInfo:
    """Register/replace the KB's read-only Mongo connection (admin). URI stored envelope-encrypted."""
    _require_role(principal, "admin")
    return await nosql_query.set_nosql_connection(kb_id, body.uri, body.database, body.collections)


@router.delete("/kb/{kb_id}/nosql/connection", status_code=204)
async def delete_kb_nosql_connection(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> None:
    """Disconnect the KB's NoSQL source (admin)."""
    _require_role(principal, "admin")
    await nosql_query.clear_nosql_connection(kb_id)


class RestConnectionRequest(BaseModel):
    base_url: str = Field(..., min_length=1)
    path: str = "/search"
    query_param: str = "q"
    token: str = ""


@router.get("/kb/{kb_id}/rest/connection", response_model=rest_source.RestConnectionInfo)
async def get_kb_rest_connection(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> rest_source.RestConnectionInfo:
    """The KB's registered read-only REST source (non-secret: base URL + path; token never returned)."""
    return await rest_source.get_rest_connection(kb_id)


@router.put("/kb/{kb_id}/rest/connection", response_model=rest_source.RestConnectionInfo)
async def set_kb_rest_connection(
    kb_id: str, body: RestConnectionRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> rest_source.RestConnectionInfo:
    """Register/replace the KB's read-only REST source (admin). Token stored envelope-encrypted."""
    _require_role(principal, "admin")
    return await rest_source.set_rest_connection(
        kb_id, body.base_url, path=body.path, query_param=body.query_param, token=body.token
    )


@router.delete("/kb/{kb_id}/rest/connection", status_code=204)
async def delete_kb_rest_connection(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> None:
    """Disconnect the KB's REST source (admin)."""
    _require_role(principal, "admin")
    await rest_source.clear_rest_connection(kb_id)


class SharePointConnectionRequest(BaseModel):
    site_url: str = Field(..., min_length=1)
    token: str = ""


@router.get("/kb/{kb_id}/sharepoint/connection", response_model=sharepoint_source.SharePointConnectionInfo)
async def get_kb_sharepoint_connection(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> sharepoint_source.SharePointConnectionInfo:
    """The KB's registered SharePoint site (non-secret: site URL; token never returned)."""
    return await sharepoint_source.get_sharepoint_connection(kb_id)


@router.put("/kb/{kb_id}/sharepoint/connection", response_model=sharepoint_source.SharePointConnectionInfo)
async def set_kb_sharepoint_connection(
    kb_id: str, body: SharePointConnectionRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> sharepoint_source.SharePointConnectionInfo:
    """Register/replace the KB's SharePoint site (admin). OAuth token stored envelope-encrypted."""
    _require_role(principal, "admin")
    return await sharepoint_source.set_sharepoint_connection(kb_id, body.site_url, token=body.token)


@router.delete("/kb/{kb_id}/sharepoint/connection", status_code=204)
async def delete_kb_sharepoint_connection(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> None:
    """Disconnect the KB's SharePoint site (admin)."""
    _require_role(principal, "admin")
    await sharepoint_source.clear_sharepoint_connection(kb_id)


@router.post("/kb/{kb_id}/sources/graph-sync")
async def sync_kb_sources_to_graph(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> dict[str, int]:
    """Seed the KB knowledge graph with every connected source's declared structure (SQL tables,
    NoSQL collections, REST/SharePoint host), so the sources are in contact in GraphRAG up front —
    not only once a query touches them. Admin. Returns per-source entity counts."""
    _require_role(principal, "admin")
    return await graph_sync.sync_kb_sources(kb_id)


# --- Architecture extraction (technical docs <-> understandable diagrams) -----
class ArchitectureRequest(BaseModel):
    topic: str = Field(..., min_length=1, description="What to explain/diagram, e.g. a product or subsystem.")
    audience: str = Field(default="non-technical", pattern="^(non-technical|technical)$")


@router.post("/kb/{kb_id}/architecture", response_model=architecture.ArchitectureAnswer)
async def kb_architecture(
    kb_id: str, body: ArchitectureRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> architecture.ArchitectureAnswer:
    """Explain + diagram the architecture of a topic from the KB (documents + diagram-image captions),
    in plain language for a non-technical user by default. Returns editable Mermaid + PlantUML, a
    grounded explanation, and citations to the source documents."""
    check_query_rate(principal.tenant_id)
    return await architecture.answer_architecture(kb_id, body.topic, audience=body.audience, actor=principal.subject)


# --- Workflow understanding + optimization (SOP -> process flow + gap analysis) ---
class ProcessRequest(BaseModel):
    topic: str = Field(..., min_length=1, description="The process/workflow to explain or optimize.")
    audience: str = Field(default="business", pattern="^(business|operations)$")


@router.post("/kb/{kb_id}/process", response_model=process_flow.ProcessAnswer)
async def kb_process(
    kb_id: str, body: ProcessRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> process_flow.ProcessAnswer:
    """Understand + optimize a business process from the KB (SOPs + flowchart-image captions): a typed
    process flow, editable Mermaid + PlantUML, detected structural gaps, and a plain-language
    explanation of steps/dependencies/business-logic with recommended improvements — all cited."""
    check_query_rate(principal.tenant_id)
    return await process_flow.answer_process(kb_id, body.topic, audience=body.audience, actor=principal.subject)


# --- Ingestion pipeline DAG (backend traceability, audit-chain verified) ------
class IngestDagResponse(BaseModel):
    dag: ingest_dag.IngestDag
    mermaid: str
    trust: dict[str, Any]


@router.get("/ingest/dag", response_model=IngestDagResponse)
async def ingest_pipeline_dag(principal: Annotated[Principal, Depends(get_principal)]) -> IngestDagResponse:
    """The canonical ingestion pipeline DAG — what the backend runs after a document is ingested
    (doc-intelligence -> sanitize -> chunk -> embed -> index fan-out -> graph). The static template."""
    _ = principal
    dag = ingest_dag.pipeline_dag()
    return IngestDagResponse(
        dag=dag, mermaid=ingest_dag.to_mermaid(dag), trust=ingest_dag.trust_summary(dag, chain_intact=True)
    )


@router.get("/kb/{kb_id}/documents/{doc_id}/dag", response_model=IngestDagResponse)
async def document_ingest_dag(
    kb_id: str, doc_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> IngestDagResponse:
    """The ingestion DAG for ONE document, overlaid with which stages actually ran and their metrics,
    each confirmed stage carrying its tamper-evident audit event hash. Visual + verifiable proof of
    the backend work — the trust summary reports whether the shown audit events are un-tampered."""
    _ = doc_id
    audit = wiring.audit_service()
    dag = await ingest_dag.document_trace(kb_id, doc_id, audit=audit)
    events = await audit.list(kb_id, trace_id=None, limit=200)
    chain_intact = (not events) or all(e.event_hash and e.compute_hash() == e.event_hash for e in events)
    return IngestDagResponse(
        dag=dag,
        mermaid=ingest_dag.to_mermaid(dag),
        trust=ingest_dag.trust_summary(dag, chain_intact=chain_intact),
    )


# --- Egress-host approval requests (sovereign self-service) -------------------
class EgressRequestBody(BaseModel):
    host: str = Field(..., description="Host or URL to request approval for.")
    reason: str = Field(default="", max_length=500)
    kb_id: str | None = None


@router.post("/egress/requests", response_model=egress_requests.EgressRequest)
async def request_egress(
    body: EgressRequestBody, principal: Annotated[Principal, Depends(get_principal)]
) -> egress_requests.EgressRequest:
    """Request approval to fetch an external host (any signed-in user). Owner/admin approves."""
    return await egress_requests.submit(
        wiring.egress_request_store(),
        host_or_url=body.host,
        tenant_id=principal.tenant_id,
        requested_by=principal.subject,
        kb_id=body.kb_id,
        reason=body.reason,
    )


@router.get("/egress/requests", response_model=list[egress_requests.EgressRequest])
async def list_egress(
    principal: Annotated[Principal, Depends(get_principal)], status: str | None = None
) -> list[egress_requests.EgressRequest]:
    """List egress requests (pending queue for admins; the requester sees their own)."""
    st: egress_requests.Status | None = (
        status if status in ("pending", "approved", "rejected") else None  # type: ignore[assignment]
    )
    return await wiring.egress_request_store().list(st)


@router.post("/egress/requests/{req_id}/{action}", response_model=egress_requests.EgressRequest)
async def decide_egress(
    req_id: str, action: str, principal: Annotated[Principal, Depends(get_principal)]
) -> egress_requests.EgressRequest:
    """Approve, reject, or revoke an egress request. An owner/admin OR the manager of the KB the
    request was raised for may decide it (the sovereignty boundary — approval adds ONLY that host
    to the allowlist). Revoke removes a previously-approved host again."""
    if action not in ("approve", "reject", "revoke"):
        raise ValidationError("action must be 'approve', 'reject', or 'revoke'")
    store = wiring.egress_request_store()
    req = await store.get(req_id)
    if req is None:
        raise NotFoundError("egress request not found", details={"id": req_id})
    is_admin = principal.has_role("owner") or principal.has_role("admin")
    is_kb_manager = req.kb_id is not None and await _can_manage_kb(principal, req.kb_id)
    if not (is_admin or is_kb_manager):
        raise ForbiddenError("only an owner/admin or the KB's manager can decide egress requests")
    if action == "revoke":
        return await egress_requests.revoke(store, req_id=req_id, decided_by=principal.subject)
    return await egress_requests.decide(store, req_id=req_id, approve=action == "approve", decided_by=principal.subject)


# --- Runtime feature flags (admin-toggleable capabilities) --------------------
@router.get("/admin/features", response_model=feature_flags.FeatureFlagsView)
async def get_features(principal: Annotated[Principal, Depends(get_principal)]) -> feature_flags.FeatureFlagsView:
    """The platform capability posture: toggleable (live) + locked (operator-managed, read-only)."""
    return feature_flags.snapshot()


class FeatureToggleRequest(BaseModel):
    enabled: bool


@router.put("/admin/features/{name}", response_model=feature_flags.FeatureFlagsView)
async def set_feature(
    name: str, body: FeatureToggleRequest, principal: Annotated[Principal, Depends(get_principal)]
) -> feature_flags.FeatureFlagsView:
    """Toggle a behavioural capability at runtime (admin only). Security/infra flags are locked."""
    _require_role(principal, "admin")
    await _gate(
        "security_change",
        tenant_id=principal.tenant_id,
        kb_id=None,
        params={"name": name, "enabled": body.enabled},
        summary=f"Change security/feature setting '{name}' to {body.enabled}",
        principal=principal,
    )
    return await feature_flags.set_flag(name, body.enabled, wiring.feature_flag_store())


# --- Maker-checker (dual control) approvals (banking pack) --------------------
def _assert_approver(principal: Principal) -> None:
    if not (principal.has_role("owner") or principal.has_role("admin")):
        raise ForbiddenError("only an owner or admin can review approvals")


@router.get("/admin/approvals", response_model=list[dual_control.ApprovalRequest])
async def list_approvals(
    principal: Annotated[Principal, Depends(get_principal)], status: str | None = None
) -> list[dual_control.ApprovalRequest]:
    """Pending (or filtered) maker-checker requests for this tenant. Owner/admin only."""
    _assert_approver(principal)
    return await wiring.approval_store().list(principal.tenant_id, status)  # type: ignore[arg-type]


@router.post("/admin/approvals/{req_id}/approve", response_model=dual_control.ApprovalRequest)
async def approve_request(
    req_id: str, principal: Annotated[Principal, Depends(get_principal)]
) -> dual_control.ApprovalRequest:
    """Approve a pending request — runs the action. The maker cannot approve their own (SoD)."""
    _assert_approver(principal)
    return await dual_control.decide(wiring.approval_store(), req_id=req_id, approve=True, decided_by=principal.subject)


@router.post("/admin/approvals/{req_id}/reject", response_model=dual_control.ApprovalRequest)
async def reject_request(
    req_id: str, principal: Annotated[Principal, Depends(get_principal)]
) -> dual_control.ApprovalRequest:
    """Reject a pending request (owner/admin)."""
    _assert_approver(principal)
    return await dual_control.decide(
        wiring.approval_store(), req_id=req_id, approve=False, decided_by=principal.subject
    )


# --- Agent orchestrator (Agentic RAG + MCP: plan / critic / approval-gate / execute) ---
class AgentRunRequest(BaseModel):
    goal: str = Field(..., min_length=1, description="What the agent should accomplish.")


@router.post("/agent/run", response_model=agent_types.AgentRunResult)
async def run_agent_endpoint(
    body: AgentRunRequest, principal: Annotated[Principal, Depends(get_principal)]
) -> agent_types.AgentRunResult:
    """Run a dynamic plan-and-execute agent: an LLM plans over the in-process tool catalog, a critic
    reviews the plan, CRITICAL tool calls are gated behind dual-control human approval, safe steps run
    (tenant-scoped + audited), and the results are synthesised. The run PAUSES with a
    pending_approval_id when a critical action needs sign-off. 404 when the orchestrator is disabled."""
    if not get_settings().agent_orchestrator_enabled:
        raise NotFoundError("agent orchestrator is disabled", details={"hint": "set agent_orchestrator_enabled"})
    check_query_rate(principal.tenant_id)
    gateway = await wiring.resolve_tenant_gateway(principal.tenant_id)
    return await agent_orchestrator.run_agent(
        body.goal,
        tenant_id=principal.tenant_id,
        actor=principal.subject,
        gateway=gateway,
        store=wiring.approval_store(),
    )


# --- Agentic-workflow runtime (FR-F) -----------------------------------------
@router.get("/kb/{kb_id}/workflows", response_model=list[WorkflowDef])
async def list_workflows(kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> list[WorkflowDef]:
    """List the KB's registered agentic workflows."""
    return await workflows.list_defs(kb_id)


@router.put("/kb/{kb_id}/workflows/{workflow_id}", response_model=WorkflowDef)
async def save_workflow(
    kb_id: str, workflow_id: str, body: WorkflowDef, principal: Annotated[Principal, Depends(tenant_guard)]
) -> WorkflowDef:
    """Register/replace a workflow definition (admin). The path id is authoritative."""
    _require_role(principal, "admin")
    return await workflows.save(kb_id, body.model_copy(update={"id": workflow_id}))


@router.delete("/kb/{kb_id}/workflows/{workflow_id}", status_code=204)
async def delete_workflow(kb_id: str, workflow_id: str, principal: Annotated[Principal, Depends(tenant_guard)]) -> None:
    _require_role(principal, "admin")
    await workflows.delete(kb_id, workflow_id)


class WorkflowRunRequest(BaseModel):
    inputs: dict[str, Any] = Field(default_factory=dict, description="Initial context, e.g. {'query': '…'}.")


@router.post("/kb/{kb_id}/workflows/{workflow_id}/run", response_model=WorkflowResult)
async def run_workflow_endpoint(
    kb_id: str, workflow_id: str, body: WorkflowRunRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> WorkflowResult:
    """Execute a workflow and return its result + decision trace (audited).

    Running a pre-existing workflow has no separate role gate of its own — like ``/v1/ask``, any
    authenticated tenant member may run one (authoring/saving stays admin-only, see
    ``save_workflow``). What keeps this safe is that every step handler is bound with the REAL
    calling principal's own clearance/ACL/roles (see ``workflows.run``), so a step can never see
    more or do more than this same caller could by hitting the underlying endpoint directly.
    """
    check_query_rate(principal.tenant_id)
    return await workflows.run(
        kb_id,
        workflow_id,
        body.inputs,
        actor=principal.subject,
        groups=_principal_groups(principal),
        # KB-scoped run -> per-KB role elevation applies, same as the direct query/retrieve endpoints.
        clearance=await kb_roles.effective_kb_clearance(get_catalog(), kb_id, principal),
        roles=principal.roles,
    )


# --- MCP discoverability (Phase-0 read-only stub) ----------------------------
@router.get("/mcp/manifest")
async def get_mcp_manifest(principal: Annotated[Principal, Depends(get_principal)]) -> dict[str, Any]:
    """Advertise the MCP tools + transports agents can use."""
    return {
        "tools": mcp_manifest(),
        "transports": {
            "streamable_http": {"path": "/mcp", "auth": "bearer (platform JWT)", "stateless": True},
            "stdio": {"command": "rag-mcp", "tenant_binding": "RAG_MCP_TENANT"},
        },
        "status": "live",
    }


# --- Platform info (non-secret) — powers the console Settings + status badge --
class ModelEntry(BaseModel):
    """One active model in the roster (non-secret): what it does and where it runs."""

    role: str  # "Generation" | "Embedding" | "Reranking" | "OCR / Vision" | "Faithfulness (NLI)" | "Translation"
    backend: str  # the configured backend id (e.g. openai_compat, chandra_api, nli, sarvam)
    model: str  # the model / deployment name
    endpoint: str  # host only (no scheme, no creds, no key) — or "in-cluster" for local services
    detail: str = ""  # role-specific extra (dim, wire shape, max_tokens, fallback, …)


class SystemInfo(BaseModel):
    """Non-secret platform configuration surfaced to the console.

    Deliberately excludes EVERY secret (JWT signing key, store credentials, DSNs) —
    only feature flags, thresholds, and backend/model *names* are exposed."""

    env: str
    sovereign_enforce: bool
    llm_backend: str
    llm_model: str
    embed_model: str
    # The abstain threshold below is only meaningful against a named reranker, so report which
    # one the number belongs to rather than the threshold alone.
    rerank_model: str
    vector_backend: str
    sparse_backend: str
    guardrails_backend: str
    faithfulness_backend: str
    features: dict[str, bool]
    thresholds: dict[str, float]
    # The full active-model roster (FR-H6 transparency): every role the platform runs, its model
    # name, backend, and endpoint — read straight from config so the console shows what is ACTUALLY
    # in use, defaults included. Non-secret: endpoints are hosts only, keys are NEVER included.
    models: list[ModelEntry] = Field(default_factory=list)


class ModelsInfo(BaseModel):
    """The configured model endpoint + the models it exposes (FR-E1/H6). Non-secret."""

    endpoint: str
    backend: str
    active_model: str
    available: list[str] = Field(default_factory=list)
    error: str | None = None


@router.get("/system/models", response_model=ModelsInfo)
async def system_models(principal: Annotated[Principal, Depends(get_principal)]) -> ModelsInfo:
    """Discover the deployed models on the configured LLM endpoint (e.g. the AIOps platform)."""
    s = get_settings()
    info = ModelsInfo(endpoint=s.vllm_url, backend=s.llm_backend, active_model=s.llm_model)
    try:
        info.available = await get_llm_gateway().list_models()
    except Exception as exc:
        info.error = str(exc)[:200]
        _log.warning("model_discovery_failed", endpoint=s.vllm_url, error=info.error)
    return info


def _endpoint_host(url: str) -> str:
    """The bare host of a model URL — non-secret (keys live in headers, never the URL). Falls
    back to 'in-cluster' for internal service names (no dots), which never leave the deployment."""
    from urllib.parse import urlparse  # noqa: PLC0415

    host = urlparse(url).netloc or url
    host = host.split("@")[-1]  # defensively drop any user:pass@ if ever present
    return host or "in-cluster"


def _model_roster(s: Settings) -> list[ModelEntry]:
    """The active model per role, read straight from config (defaults included) so the console
    shows what is ACTUALLY running. Non-secret: names + endpoint HOSTS only, never keys."""
    roster = [
        ModelEntry(
            role="Generation",
            backend=s.llm_backend,
            model=s.llm_model,
            endpoint=_endpoint_host(s.vllm_url),
            detail=f"max_tokens {s.llm_max_tokens}",
        ),
        ModelEntry(
            role="Embedding",
            backend=s.embed_backend,
            model=s.embed_model,
            endpoint=_endpoint_host(s.tei_url),
            detail=f"dim {s.embed_dim}",
        ),
        ModelEntry(
            role="Reranking",
            backend=s.rerank_backend,
            model=s.rerank_model,
            endpoint=_endpoint_host(s.rerank_url),
            detail=f"wire {s.rerank_api_shape}"
            + (f" · fallback {s.rerank_fallback_model}" if s.rerank_fallback_backend else ""),
        ),
    ]
    if s.faithfulness_backend == "nli":
        roster.append(
            ModelEntry(
                role="Faithfulness (NLI)",
                backend="nli",
                model=s.nli_model,
                endpoint=_endpoint_host(s.nli_url),
                detail="1 - P(contradiction)",
            )
        )
    if s.ocr_backend != "noop":
        # The endpoint and model of the engine ACTUALLY in use. Naming a host the OCR calls never
        # touch is how www.datalab.to appeared on the Models page of a deployment running
        # self-hosted Chandra: the row fell back to the Datalab URL for every backend that was not
        # "chandra", including the LOCAL engines, which have no endpoint at all.
        remote = {"chandra": (s.chandra_url, s.chandra_model), "chandra_api": (s.datalab_url, "chandra")}
        url, model = remote.get(s.ocr_backend, ("", s.ocr_backend))
        roster.append(
            ModelEntry(
                role="OCR / Vision",
                backend=s.ocr_backend,
                model=model,
                endpoint=_endpoint_host(url) if url else "in-cluster",
                detail="layout-aware document OCR" if url else "local engine, no network call",
            )
        )
    if s.translation_backend != "noop" or s.graph_extract_backend == "translate":
        is_sarvam = s.translate_backend == "sarvam"
        roster.append(
            ModelEntry(
                role="Translation",
                backend=s.translate_backend,
                model=s.translate_sarvam_model if is_sarvam else s.llm_model,
                endpoint=_endpoint_host(s.translate_sarvam_url if is_sarvam else s.vllm_url),
                detail="Indic↔English (graph entities)",
            )
        )
    return roster


def _generation_only(roster: list[ModelEntry]) -> list[ModelEntry]:
    """The generation row alone — what a KB member can actually pick between."""
    return [m for m in roster if m.role == "Generation"]


class ModelHealth(BaseModel):
    """One backend's LIVE state — what the console shows beside its row."""

    name: str = Field(..., description='The probed backend, e.g. "Generation LLM", "Reranker (fallback)".')
    role: str = Field(..., description="The roster role it serves, as /system/info spells it.")
    endpoint: str = Field(..., description="Bare host. Non-secret: keys travel in headers, never the URL.")
    online: bool = Field(..., description="Did the endpoint answer past authentication?")
    latency_ms: int | None = Field(None, description="Probe round trip; null when it never answered.")
    detail: str = Field("", description="Empty when healthy, else why not (wrong credential, unreachable).")
    model: str = Field("", description="The model the probe actually reached — may differ from the configured one.")
    last_online: datetime | None = Field(
        None, description="When this backend was last observed online (from the poller). Null if never seen up."
    )
    consecutive_failures: int = Field(
        0, description="Consecutive failed probes; the dot flips offline only past the debounce threshold."
    )


class ModelHealthReport(BaseModel):
    checked_at: datetime
    cached: bool = Field(..., description="True when served from the short cache rather than freshly probed.")
    source: Literal["poller", "live"] = Field(
        "live", description='"poller" = the background snapshot (no probe this request); "live" = probed now.'
    )
    stale: bool = Field(False, description="True when a served poller snapshot is older than several poll intervals.")
    backends: list[ModelHealth] = Field(default_factory=list)
    retrieval_degraded: bool = Field(
        False,
        description="Slice-2b: true when this KB's multivector legs have failed open (answering on "
        "BM25/dense only) — a reindex to the named collection is required. Per-KB probe only.",
    )


# A 1x1 transparent PNG — the smallest valid image, used to probe a vision/OCR endpoint without
# shipping a page of content to it.
_PROBE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

# A probe is a real request to a real model endpoint, so a page refresh must not fan out ten of
# them. Cached briefly; ?refresh=true forces a new round (the console's "Re-check" button).
_HEALTH_TTL_SECONDS = 30.0
# A one-slot cache held in a mutable container rather than rebound through `global`.
_health_cache: dict[str, tuple[float, list[BackendHealth]]] = {}


async def _effective_generation_health(kb_id: str) -> BackendHealth:
    """Probe the generation backend that ACTUALLY answers this KB, through its own adapter.

    The env-level probe reads RAG_LLM_*, but a KB answers on its org-wide model CONNECTION when
    one is set — different endpoint, different credential. Reported live, the topbar badge said
    "online, 85 ms" (it resolves the connection) while this table said "offline (401)" (it read
    the platform default) for the same row. The row shows the connection's model, so it must show
    the connection's health: resolved through ``resolve_tenant_gateway``, the same object a
    question is answered by, so it passes or fails for exactly the reason a question would."""
    gw = await wiring.resolve_tenant_gateway(kb_id)
    started = time.perf_counter()
    try:
        await gw.complete("ping", max_tokens=1)
    except Exception as exc:
        return BackendHealth(
            name="Generation LLM",
            role="Generation",
            endpoint=_endpoint_host(getattr(gw, "url", "") or ""),
            online=False,
            latency_ms=None,
            detail=str(exc)[:200],
            model=gw.model_name,
        )
    return BackendHealth(
        name="Generation LLM",
        role="Generation",
        endpoint=_endpoint_host(getattr(gw, "url", "") or ""),
        online=True,
        latency_ms=int((time.perf_counter() - started) * 1000),
        detail=f"reachable · {gw.model_name}",
        model=gw.model_name,
    )


async def _effective_infra_health(kb_id: str, role: str) -> BackendHealth | None:
    """Probe the embedder/reranker that ACTUALLY serves this KB, when it is not the platform one.

    Same reason as ``_effective_generation_health``: the env probe reads RAG_EMBED_*/RAG_RERANK_*,
    but a KB resolves its own attach or the ORG-WIDE default of that role, on a different endpoint
    with a different credential. Reporting the env probe beside a row that names the org model is
    the drift that already bit the generation row once.

    Returns None when the platform singleton is what serves this KB — the env probe already
    describes it exactly, and re-probing it would only add a second, slower answer to the same
    question."""
    # Names match the boot preflight's own probes ("Embedder", "Reranker", "OCR (chandra)") so the
    # console and the boot log keep naming the same thing the same way.
    from rag_clients.factory import get_vision_llm_gateway  # noqa: PLC0415 — cycle at module load

    resolvers: dict[str, tuple[Any, Any, str]] = {
        "Embedding": (wiring.resolve_tenant_embedder, get_embedder(), "Embedder"),
        "Reranking": (wiring.resolve_tenant_reranker, get_reranker(), "Reranker"),
        "OCR / Vision": (wiring.resolve_tenant_ocr, wiring.platform_ocr_engine(), "OCR (org model)"),
        "Vision": (wiring.resolve_tenant_vision_gateway, get_vision_llm_gateway(), "Vision LLM (org model)"),
    }
    resolver, platform, name = resolvers[role]
    adapter = await resolver(kb_id)
    if adapter is platform:
        return None
    client = getattr(adapter, "_client", None)
    endpoint = _endpoint_host(str(getattr(adapter, "_url", None) or getattr(client, "_url", "") or ""))
    model = str(getattr(adapter, "_model", None) or getattr(client, "_model", "") or "")
    started = time.perf_counter()
    try:
        if role == "Embedding":
            vectors = await adapter.embed(["ping"])
            detail = f"reachable · {model} · dim {len(vectors[0]) if vectors else 0}"
        elif role == "Reranking":
            span = RetrievedSpan(chunk_id="probe", doc_id="probe", text="ping", score=0.0)
            await adapter.rerank("ping", [span], 1)
            detail = f"reachable · {model}"
        elif role == "Vision":
            await adapter.complete_with_images("Reply with OK.", [_PROBE_PNG], max_tokens=1)
            detail = f"reachable · {model}"
        elif client is None:
            # An OCR engine with no HTTP client is a LOCAL one (tesseract/paddle): there is no
            # endpoint to reach, so reporting it as a network probe would be a fiction.
            return None
        else:
            # A 1x1 PNG: the cheapest request that still exercises the real vision-chat path
            # (endpoint, credential, model id) the way an ingested page would.
            await client.run("Reply with OK.", [_PROBE_PNG], mime="image/png", max_tokens=1)
            detail = f"reachable · {model}"
    except Exception as exc:
        return BackendHealth(
            name=name, role=role, endpoint=endpoint, online=False, latency_ms=None, detail=str(exc)[:200], model=model
        )
    return BackendHealth(
        name=name,
        role=role,
        endpoint=endpoint,
        online=True,
        latency_ms=int((time.perf_counter() - started) * 1000),
        detail=detail,
        model=model,
    )


@router.get("/system/model-health", response_model=ModelHealthReport)
async def system_model_health(
    principal: Annotated[Principal, Depends(get_principal)],
    refresh: bool = False,
    kb_id: str | None = None,
) -> ModelHealthReport:
    """Is every configured model endpoint actually answering, and how fast?

    The roster in ``/system/info`` reads CONFIGURATION — it says what is set, not what works. A
    wrong header or an expired key on embedding, reranking, OCR or NLI was invisible until a
    query or an ingest failed on it; measured live, a reranker fallback sat misconfigured with no
    indicator anywhere in the product. This probes the same endpoints the boot preflight does,
    with the same prober, so the console and the boot log cannot disagree.

    Scoped like the roster: an ordinary member sees the generation model that answers their
    questions; org owners/admins see every backend, since they are who fixes one."""
    # Continuous poller: when the background self-check has published a platform snapshot, serve it
    # WITHOUT probing on the request path — every replica reads the same shared snapshot, so all
    # show identical fresh status. Gated on the flag so a poller-off deployment is byte-identical to
    # on-demand probing. refresh=true still forces a live probe (the manual "Re-check" button), and
    # the per-KB path stays on-demand (the poller covers the platform slot only).
    if kb_id is None and not refresh and get_settings().health_poll_enabled:
        from svc_control_plane import health_poller  # noqa: PLC0415 — only when the poller is on

        snap = await health_poller.read_snapshot()
        if snap is not None:
            healths = [
                ModelHealth(
                    name=b.name,
                    role=b.role,
                    endpoint=b.endpoint,
                    online=b.online,
                    latency_ms=b.latency_ms,
                    detail=b.detail,
                    model=b.model,
                    last_online=b.last_online,
                    consecutive_failures=b.consecutive_failures,
                )
                for b in snap.backends
            ]
            if not _is_org_admin(principal):
                healths = [h for h in healths if h.name.startswith("Generation")]
            return ModelHealthReport(
                checked_at=snap.polled_at,
                cached=True,
                source="poller",
                stale=health_poller.is_stale(snap, get_settings()),
                backends=healths,
            )

    now = time.monotonic()
    slot = f"kb:{kb_id}" if kb_id else "platform"
    previous = _health_cache.get(slot)
    cached = previous is not None and not refresh and (now - previous[0]) < _HEALTH_TTL_SECONDS
    if not cached:
        probed = await probe_backend_health(get_settings())
        if kb_id:
            await _assert_kb_access(principal, kb_id)
            generation = await _effective_generation_health(kb_id)
            settings = get_settings()
            # Translation on the llm backend runs on that SAME resolved gateway
            # (wiring.resolve_translator), so it is reachable exactly when generation is.
            translation = (
                [replace(generation, name="Translation (generation model)", role="Translation")]
                if settings.translation_backend != "noop" and settings.translate_backend == "llm"
                else []
            )
            probed = [generation, *translation, *(b for b in probed if b.role not in ("Generation", "Translation"))]
            # Same for the infrastructure roles a KB can now resolve for itself (its own attach,
            # or the org-wide default of that role). None = the platform singleton serves it, so
            # the env probe already in `probed` is the right answer.
            for infra in ("Embedding", "Reranking", "OCR / Vision", "Vision"):
                effective = await _effective_infra_health(kb_id, infra)
                if effective is not None:
                    # Every env row of that role goes, the env fallback included: a KB on its own
                    # reranker does not fall back to the platform's secondary.
                    probed = [effective, *(b for b in probed if b.role != infra)]
        _health_cache[slot] = (now, probed)
    backends = _health_cache[slot][1]
    if not _is_org_admin(principal):
        backends = [b for b in backends if b.name.startswith("Generation")]
    # Slice-2b degraded-mode indicator (per-KB probe only): the KB's vector store reports whether a
    # named multivector leg has failed open this process. getattr-guarded — the in-memory twin and a
    # non-multivector store simply lack it, so the flag stays False.
    degraded = False
    if kb_id:
        probe = getattr(get_vector_store(), "multivector_degraded", None)
        degraded = bool(probe(kb_id)) if callable(probe) else False
    return ModelHealthReport(
        checked_at=datetime.now(UTC),
        cached=cached,
        backends=[ModelHealth(**asdict(b)) for b in backends],
        retrieval_degraded=degraded,
    )


@router.get("/system/info", response_model=SystemInfo)
async def system_info(principal: Annotated[Principal, Depends(get_principal)]) -> SystemInfo:
    """Non-secret platform config for the console (flags, thresholds, model/backend names)."""
    s = get_settings()
    return SystemInfo(
        env=s.env.value,
        sovereign_enforce=s.sovereign_enforce,
        llm_backend=s.llm_backend,
        llm_model=s.llm_model,
        embed_model=s.embed_model,
        rerank_model=s.rerank_model,
        vector_backend=s.vector_backend,
        sparse_backend=s.sparse_backend,
        guardrails_backend=s.guardrails_backend,
        faithfulness_backend=s.faithfulness_backend,
        features={
            "contextual_retrieval": s.contextual_retrieval_enabled,
            "adaptive_router": s.adaptive_router_enabled,
            "query_transform": s.query_transform_enabled,
            "guardrails": s.guardrails_enabled,
            "faithfulness": s.faithfulness_enabled,
            "semantic_cache": s.semantic_cache_enabled,
            "agentic_retry": s.agentic_retry_enabled,
            "ocr": s.ocr_backend != "noop",
            "field_extraction": s.field_extraction_enabled,
            "language_detection": s.language_detection_enabled,
            "translation": s.translation_backend != "noop",
            "oidc_sso": s.auth_mode == "oidc",
            "mfa_required": s.oidc_require_mfa,
            "multi_replica_scale": bool(s.redis_url),
            "pii_masking": True,
            "byok_kms": s.kms_backend != "local",
            "nl2sql": s.nl2sql_enabled,
            "agentic_workflows": s.agentic_workflows_enabled,
            "integrations": s.ui_integrations_enabled,
            "personal_kb": s.personal_kb_enabled,
        },
        thresholds={
            "abstain": s.abstain_threshold,
            "faithful": s.faithful_threshold,
            "flag": s.flag_threshold,
        },
        # Generation is the only role a KB actually chooses; embedding, reranking, OCR,
        # faithfulness and translation are platform infrastructure, used during ingestion and
        # retrieval and identical for every KB. Ordinary members see just the model answering
        # their questions. Org owners/admins keep the full roster: it is the only place the
        # RUNNING model configuration is visible, and reading it is what surfaced a reranker and
        # an OCR backend still pointed at foreign hosted APIs. Names and endpoint HOSTS only —
        # never keys — for either audience.
        models=_model_roster(s) if _is_org_admin(principal) else _generation_only(_model_roster(s)),
    )


# --- Indic multilingual: translate + detect (content) ------------------------
class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)
    target_lang: str = Field(..., min_length=2, max_length=8, description="Target language code, e.g. hi, or, ta.")


class TranslateResponse(BaseModel):
    text: str
    detected_lang: str
    target_lang: str


@router.post("/kb/{kb_id}/translate", response_model=TranslateResponse)
async def translate_text(
    kb_id: str, body: TranslateRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> TranslateResponse:
    """Translate text into a target language via the KB's own sovereign model (Indic-ready).
    Returns the source language (script-detected) too. No-op unless the LLM backend is on."""
    from rag_core.language import detect_language  # noqa: PLC0415

    check_query_rate(principal.tenant_id)
    translator = await wiring.resolve_translator(kb_id)
    out = await translator.translate(body.text, body.target_lang)
    return TranslateResponse(text=out, detected_lang=detect_language(body.text), target_lang=body.target_lang)


# --- Agent connectivity: MCP server info (FR-H2) -----------------------------
class McpTool(BaseModel):
    name: str
    description: str


class McpInfo(BaseModel):
    enabled: bool
    endpoint_path: str  # mounted path; the full URL is {origin}{endpoint_path}
    transport: str
    auth: str
    tenancy: str
    tools: list[McpTool]


@router.get("/agent/mcp", response_model=McpInfo)
async def mcp_info(principal: Annotated[Principal, Depends(get_principal)]) -> McpInfo:
    """How an AI agent connects to this platform over MCP (FR-H2). The tools wrap the same
    pipeline, so an agent inherits guardrails, grounding, faithfulness, tenancy and audit."""
    from rag_mcp import tool_catalog  # noqa: PLC0415 — lazy import avoids load-time coupling

    return McpInfo(
        enabled=True,
        endpoint_path="/mcp",
        transport="streamable-http",
        auth="Bearer a platform JWT or a service-account API key (ek_…) in the Authorization header.",
        tenancy="Scoped to the credential's organisation for the request; the tenant is never client-supplied.",
        tools=[McpTool(**t) for t in tool_catalog()],
    )


# --- Accounts & organisations (self-serve multi-tenant SaaS) -----------------
_ORG_RESERVED = {"api", "v1", "auth", "admin", "rag", "rag_identity"}


def _slug_org_id(name: str) -> str:
    slug = "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")
    slug = "-".join(p for p in slug.split("-") if p)
    return slug or "org"


async def _unique_org_id(name: str) -> str:
    base = _slug_org_id(name)
    ident = wiring.identity_store()
    candidate, n = base, 1
    while candidate in _ORG_RESERVED or await ident.get_org(candidate) is not None:
        n += 1
        candidate = f"{base}-{n}"
    return candidate


class SignupRequest(BaseModel):
    org_name: str = Field(..., min_length=1, description="Your organisation's name.")
    name: str = Field(..., min_length=1, description="Your full name.")
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=8, description="At least 8 characters.")


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=1)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105 - OAuth token type, not a secret
    email: str
    name: str
    org_id: str
    org_name: str
    role: str
    roles: list[str] = Field(default_factory=list)  # fine-grained capability roles
    kb_access: list[str] = Field(default_factory=list)  # allowed kb_ids ([] = all in org)


def _token_for(user: User) -> str:
    # Carry the fine-grained capability roles + per-KB access in the JWT so every RBAC/tenant
    # check (and kb_access guard) works off the token, no per-request user lookup needed.
    return issue_dev_jwt(
        user.email,
        user.org_id,
        roles=effective_roles(user),
        attrs={"kb_access": user.kb_access, "clearance": user.clearance, "unit": user.unit_id or user.org_id},
    )


@router.post("/auth/signup", response_model=AuthResponse, status_code=201)
async def signup(body: SignupRequest) -> AuthResponse:
    """Create an organisation + its owner account. The org is a fully isolated tenant."""
    ident = wiring.identity_store()
    email = body.email.strip().lower()
    if await ident.get_user(email) is not None:
        raise ValidationError("an account with this email already exists")
    org_id = await _unique_org_id(body.org_name)
    await ident.create_org(Org(org_id=org_id, name=body.org_name))
    # The owner is the org's root of trust and is intentionally un-editable via /org/members, so it
    # must be seeded at the top clearance here — otherwise a self-serve org is stuck at Public(0) with
    # no in-product way to raise ANY clearance (grants are capped by the grantor's own). Mirrors the
    # platform-provision path (tenant_control.py), which already seeds the owner at RESTRICTED.
    user = User(
        email=email,
        org_id=org_id,
        name=body.name,
        role="owner",
        password_hash=hash_password(body.password),
        clearance=int(Clearance.RESTRICTED),
    )
    await ident.create_user(user)
    await wiring.provision_kb(org_id)  # the org's default KB namespace
    _log.info("org_created", org_id=org_id, owner=email)
    return AuthResponse(
        access_token=_token_for(user),
        email=email,
        name=body.name,
        org_id=org_id,
        org_name=body.org_name,
        role="owner",
        roles=effective_roles(user),
        kb_access=user.kb_access,
    )


async def _ldap_login(username: str, password: str) -> User:
    """Authenticate against on-prem AD/LDAP and get-or-provision the platform user (banking pack).
    AD is the source of truth: the user is created on first login into the configured org, with a
    role mapped from their AD groups (no local password stored — LDAP binds every time)."""
    from rag_common.ldap_auth import authenticate_ldap, role_for_groups  # noqa: PLC0415

    identity = authenticate_ldap(username, password)
    if identity is None:
        raise UnauthorizedError("invalid directory credentials")
    settings = get_settings()
    ident = wiring.identity_store()
    email = identity.email.strip().lower()
    existing = await ident.get_user(email)
    if existing is not None:
        return existing
    org_id = settings.ldap_default_org
    if await ident.get_org(org_id) is None:
        await ident.create_org(Org(org_id=org_id, name=org_id))
        await wiring.provision_kb(org_id)
    role = role_for_groups(identity.groups, settings.ldap_role_map)
    user = User(email=email, org_id=org_id, name=identity.name or email, role=role, password_hash="")
    await ident.create_user(user)
    _log.info("ldap_user_provisioned", email=email, org_id=org_id, role=role)
    return user


@router.post("/auth/login", response_model=AuthResponse)
async def login(body: LoginRequest) -> AuthResponse:
    """Sign in with email + password (local), or via on-prem AD/LDAP bind when enabled."""
    ident = wiring.identity_store()
    settings = get_settings()
    if settings.ldap_enabled:
        user = await _ldap_login(body.email.strip(), body.password)
    else:
        candidate = await ident.get_user(body.email.strip().lower())
        if candidate is None:
            # Do equivalent scrypt work so the response time doesn't reveal whether the email
            # is registered (timing-based account enumeration, QA ISS-008).
            verify_dummy(body.password)
            raise UnauthorizedError("invalid email or password")
        if not verify_password(body.password, candidate.password_hash):
            raise UnauthorizedError("invalid email or password")
        # Gate on lifecycle AFTER the password check, so status is only revealed to someone who
        # already holds the correct credential (not an anonymous prober).
        if candidate.status == "suspended":
            raise UnauthorizedError("this account has been suspended. Contact your administrator.")
        if candidate.status != "active":
            raise UnauthorizedError("this account isn't activated yet. Use your activation link to set a password.")
        user = candidate
    org = await ident.get_org(user.org_id)
    return AuthResponse(
        access_token=_token_for(user),
        email=user.email,
        name=user.name,
        org_id=user.org_id,
        org_name=org.name if org else user.org_id,
        role=user.role,
        roles=effective_roles(user),
        kb_access=user.kb_access,
    )


class MeResponse(BaseModel):
    email: str
    name: str
    org_id: str
    org_name: str
    role: str
    roles: list[str] = Field(default_factory=list)
    kb_access: list[str] = Field(default_factory=list)


@router.get("/auth/me", response_model=MeResponse)
async def me(principal: Annotated[Principal, Depends(get_principal)]) -> MeResponse:
    """The signed-in user + their organisation."""
    ident = wiring.identity_store()
    user = await ident.get_user(principal.subject)
    org = await ident.get_org(principal.tenant_id)
    return MeResponse(
        email=principal.subject,
        name=user.name if user else principal.subject,
        org_id=principal.tenant_id,
        org_name=org.name if org else principal.tenant_id,
        role=user.role if user else ("owner" if "owner" in principal.roles else "member"),
        roles=effective_roles(user) if user else list(principal.roles),
        kb_access=user.kb_access if user else list(principal.attrs.get("kb_access", [])),
    )


class MemberInfo(BaseModel):
    email: str
    name: str
    role: str
    status: str = "active"  # invited | active | suspended
    roles: list[str] = Field(default_factory=list)  # fine-grained capability roles
    kb_access: list[str] = Field(default_factory=list)  # allowed kb_ids ([] = all in org)


def _member_info(u: User) -> MemberInfo:
    return MemberInfo(
        email=u.email,
        name=u.name,
        role=u.role,
        status=u.status,
        roles=effective_roles(u),
        kb_access=u.kb_access,
    )


class AddMemberRequest(BaseModel):
    name: str = Field(..., min_length=1)
    email: str = Field(..., min_length=3)
    # Optional: the member sets their own password via the activation link. If a temporary
    # password IS supplied it must be >= 8 chars; either way the member starts `invited`.
    password: str | None = Field(default=None)
    # Fine-grained capability roles to grant (reader/ingestor/reviewer/admin). Falls back to the
    # coarse `role` bundle when omitted (backward-compatible).
    roles: list[str] | None = None
    role: str = Field(default="member")
    kb_access: list[str] = Field(default_factory=list)  # allowed kb_ids ([] = all in org)


class UpdateMemberRequest(BaseModel):
    roles: list[str] = Field(default_factory=list)
    kb_access: list[str] = Field(default_factory=list)
    clearance: str | int = Field(default=0, description="Data clearance 0..3 or a level name (public..restricted).")
    unit_id: str | None = Field(default=None, description="Org unit the member belongs to (None = tenant root).")


def _clean_cap_roles(roles: list[str]) -> list[str]:
    """Validate + dedupe capability roles; owner is never grantable via member management."""
    bad = [r for r in roles if r not in CAP_ROLES]
    if bad:
        raise ValidationError(f"unknown role(s): {', '.join(bad)}", details={"roles": list(CAP_ROLES)})
    clean = list(dict.fromkeys(roles))
    if not clean:
        raise ValidationError("at least one role is required", details={"roles": list(CAP_ROLES)})
    return clean


@router.get("/org/members", response_model=list[MemberInfo])
async def list_members(principal: Annotated[Principal, Depends(get_principal)]) -> list[MemberInfo]:
    """List the users in the caller's organisation (owner/admin only). A KB manager sees only their
    KB's users via GET /kb/{id}/members — never the org-wide roster."""
    if not (principal.has_role("owner") or principal.has_role("admin")):
        raise ForbiddenError("only an owner or admin can list organisation members")
    users = await wiring.identity_store().list_users(principal.tenant_id)
    return [_member_info(u) for u in users]


@router.post("/org/members", response_model=MemberInfo, status_code=201)
async def add_member(body: AddMemberRequest, principal: Annotated[Principal, Depends(get_principal)]) -> MemberInfo:
    """Add a user to the caller's organisation with explicit capability roles + KB access
    (owner/admin only)."""
    if not (principal.has_role("owner") or principal.has_role("admin")):
        raise ForbiddenError("only an owner or admin can add members")
    if body.roles is not None:
        roles = _clean_cap_roles(body.roles)
    else:  # backward-compatible: derive from the coarse role bundle (owner is never granted here)
        if body.role not in ROLE_CAPS:
            raise ValidationError(f"unknown role '{body.role}'", details={"roles": sorted(ROLE_CAPS)})
        roles = [r for r in ROLE_CAPS[body.role] if r in CAP_ROLES]
    ident = wiring.identity_store()
    email = body.email.strip().lower()
    # Server-side email validation — the console validates too, but API/MCP callers bypass it, and
    # an invalid address means the member can never sign in.
    if not re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email):
        raise ValidationError("enter a valid email address (e.g. you@org.in)", details={"email": body.email})
    if await ident.get_user(email) is not None:
        raise ValidationError("an account with this email already exists")
    if body.password is not None and len(body.password) < _MIN_PASSWORD_LEN:
        raise ValidationError(f"password must be at least {_MIN_PASSWORD_LEN} characters")
    # New members start `invited`: they can't sign in until activated. A temporary password is
    # optional — the member normally sets their own via the activation link.
    password_hash = hash_password(body.password) if body.password else hash_password(secrets.token_urlsafe(24))
    user = User(
        email=email,
        org_id=principal.tenant_id,
        name=body.name,
        role="member",
        password_hash=password_hash,
        status="invited",
        roles=roles,
        kb_access=body.kb_access,
    )
    await ident.create_user(user)
    _log.info("member_added", org_id=principal.tenant_id, email=email, roles=roles, kbs=len(body.kb_access))
    return _member_info(user)


@router.patch("/org/members/{email}", response_model=MemberInfo)
async def update_member(
    email: str, body: UpdateMemberRequest, principal: Annotated[Principal, Depends(get_principal)]
) -> MemberInfo:
    """Change a member's capability roles + KB access (owner/admin only). The owner is protected."""
    if not (principal.has_role("owner") or principal.has_role("admin")):
        raise ForbiddenError("only an owner or admin can change member access")
    roles = _clean_cap_roles(body.roles)
    ident = wiring.identity_store()
    target = await ident.get_user(email.strip().lower())
    if target is None or target.org_id != principal.tenant_id:
        raise NotFoundError("member not found", details={"email": email})
    if target.role == "owner":
        raise ForbiddenError("the owner's access cannot be changed")
    # No-escalation on the two hierarchical dimensions: an admin may not grant a CLEARANCE above
    # its own, nor place a member into a UNIT it does not govern. (Role/kb-access management keeps
    # its existing owner/admin gate above.)
    grantor = subject_of(principal)
    target_clearance = Clearance(parse_classification(body.clearance))
    if not can_grant_clearance(grantor.tier, grantor.clearance, target_clearance):
        raise ForbiddenError(
            "you cannot grant a clearance above your own", details={"your_clearance": int(grantor.clearance)}
        )
    unit_parents = await parents_map(wiring.org_unit_store(), principal.tenant_id)
    target_unit = body.unit_id or principal.tenant_id
    if body.unit_id is not None and target_unit not in unit_parents:
        raise NotFoundError("unit not found", details={"unit_id": body.unit_id})
    if not can_admin_unit(grantor, unit_id=target_unit, parents=unit_parents):
        raise ForbiddenError(
            "you cannot assign a member into a unit you do not govern", details={"unit_id": target_unit}
        )
    await ident.set_user_access(target.email, roles=roles, kb_access=body.kb_access)
    await ident.set_governance(target.email, clearance=int(target_clearance), unit_id=body.unit_id)
    updated = await ident.get_user(target.email)
    assert updated is not None
    _log.info("member_updated", org_id=principal.tenant_id, email=target.email, roles=roles)
    return _member_info(updated)


# --- KB managers (delegated, KB-scoped member management) --------------------
# An org owner/admin can appoint members as MANAGERS of a specific KB. A KB manager can view and
# add users scoped to THAT KB only — they never gain org-wide or admin powers. Stored org-wide in
# tenant_config as {kb_id: [emails]}. The org's member list (/org/members) stays owner/admin-only;
# this is a separate, narrower surface managed from the KB page.
_KB_MANAGERS_KEY = "kb_managers"
# The capability roles a KB manager may grant to users they create — never admin/owner.
_KB_GRANTABLE_ROLES = ("reader", "ingestor", "reviewer")


async def _load_kb_managers(org_id: str) -> dict[str, list[str]]:
    cfg = await get_catalog().get_tenant_config(org_id)
    raw = cfg.get(_KB_MANAGERS_KEY)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return {str(k): [str(e).lower() for e in v] for k, v in data.items() if isinstance(v, list)}


async def _managed_kbs(org_id: str, email: str) -> list[str]:
    e = email.strip().lower()
    return [kb for kb, emails in (await _load_kb_managers(org_id)).items() if e in emails]


async def _can_manage_kb(principal: Principal, kb_id: str) -> bool:
    if principal.has_role("owner") or principal.has_role("admin"):
        return True
    return kb_id in await _managed_kbs(principal.tenant_id, principal.subject)


class KbManagers(BaseModel):
    managers: list[str] = Field(default_factory=list, description="Member emails who manage this KB's users.")


@router.get("/kb/{kb_id}/managers", response_model=KbManagers)
async def get_kb_managers(kb_id: str, principal: Annotated[Principal, Depends(get_principal)]) -> KbManagers:
    """The members who manage this KB's users. Visible to an org owner/admin or a manager of this KB."""
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only an owner/admin or a manager of this KB may view its managers")
    return KbManagers(managers=(await _load_kb_managers(principal.tenant_id)).get(kb_id, []))


@router.put("/kb/{kb_id}/managers", response_model=KbManagers)
async def set_kb_managers(
    kb_id: str, body: KbManagers, principal: Annotated[Principal, Depends(get_principal)]
) -> KbManagers:
    """Assign the members who manage this KB's users (org owner/admin only). Each must be an existing
    member of the org; a KB manager can add/see users scoped to THIS KB only."""
    if not (principal.has_role("owner") or principal.has_role("admin")):
        raise ForbiddenError("only an owner or admin can assign KB managers")
    ident = wiring.identity_store()
    clean: list[str] = []
    for raw in body.managers:
        e = raw.strip().lower()
        u = await ident.get_user(e)
        if u is None or u.org_id != principal.tenant_id:
            raise ValidationError(f"'{raw}' is not a member of this organisation", details={"email": raw})
        if e not in clean:
            clean.append(e)
    allm = await _load_kb_managers(principal.tenant_id)
    if clean:
        allm[kb_id] = clean
    else:
        allm.pop(kb_id, None)
    await get_catalog().set_tenant_config(principal.tenant_id, _KB_MANAGERS_KEY, json.dumps(allm))
    # Ensure each manager can actually reach this KB: if they're scoped (non-empty kb_access) and
    # don't yet have it, grant access. An All-KB member (empty kb_access) already sees it.
    for e in clean:
        u = await ident.get_user(e)
        if u and u.kb_access and kb_id not in u.kb_access:
            await ident.set_user_access(e, roles=u.roles, kb_access=[*u.kb_access, kb_id])
    _log.info("kb_managers_set", org_id=principal.tenant_id, kb_id=kb_id, count=len(clean), actor=principal.subject)
    return KbManagers(managers=clean)


class KbAddMemberRequest(BaseModel):
    name: str = Field(..., min_length=1)
    email: str = Field(..., min_length=3)
    password: str | None = Field(default=None)
    roles: list[str] = Field(default_factory=lambda: ["reader"])


@router.get("/kb/{kb_id}/members", response_model=list[MemberInfo])
async def list_kb_members(kb_id: str, principal: Annotated[Principal, Depends(get_principal)]) -> list[MemberInfo]:
    """Members scoped to this KB — the users a KB manager can see. Org owner/admin or a manager of
    this KB. A member counts as scoped when this KB is in their kb_access (an empty kb_access means
    org-wide All-KB access — those org-level members are NOT shown to a KB manager)."""
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only an owner/admin or a manager of this KB may view its members")
    users = await wiring.identity_store().list_users(principal.tenant_id)
    return [_member_info(u) for u in users if kb_id in u.kb_access]


@router.post("/kb/{kb_id}/members", response_model=MemberInfo, status_code=201)
async def add_kb_member(
    kb_id: str, body: KbAddMemberRequest, principal: Annotated[Principal, Depends(get_principal)]
) -> MemberInfo:
    """Add a user scoped to THIS KB (org owner/admin or a manager of this KB). The new member's
    access is forced to this KB, and only non-admin capability roles may be granted."""
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only an owner/admin or a manager of this KB can add its members")
    roles = [r for r in dict.fromkeys(body.roles) if r in _KB_GRANTABLE_ROLES]
    if not roles:
        raise ValidationError("select at least one role", details={"roles": list(_KB_GRANTABLE_ROLES)})
    ident = wiring.identity_store()
    email = body.email.strip().lower()
    if not re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email):
        raise ValidationError("enter a valid email address (e.g. you@org.in)", details={"email": body.email})
    if await ident.get_user(email) is not None:
        raise ValidationError("an account with this email already exists")
    if body.password is not None and len(body.password) < _MIN_PASSWORD_LEN:
        raise ValidationError(f"password must be at least {_MIN_PASSWORD_LEN} characters")
    # A KB manager can't reach the owner/admin-gated activation flow, so a password given here makes
    # the member immediately ACTIVE (the manager shares the credential). No password -> invited, to
    # be activated by an org admin.
    password_hash = hash_password(body.password) if body.password else hash_password(secrets.token_urlsafe(24))
    status = "active" if body.password else "invited"
    user = User(
        email=email,
        org_id=principal.tenant_id,
        name=body.name,
        role="member",
        password_hash=password_hash,
        status=status,
        roles=roles,
        kb_access=[kb_id],
    )
    await ident.create_user(user)
    _log.info(
        "kb_member_added",
        org_id=principal.tenant_id,
        kb_id=kb_id,
        email=email,
        roles=roles,
        status=status,
        actor=principal.subject,
    )
    return _member_info(user)


@router.delete("/kb/{kb_id}/members/{email}", status_code=204)
async def remove_kb_member(kb_id: str, email: str, principal: Annotated[Principal, Depends(get_principal)]) -> None:
    """Remove a KB-scoped user (org owner/admin or a manager of this KB). A KB manager may only
    remove a member whose access is THIS KB alone — never an org-wide/multi-KB member or the owner."""
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only an owner/admin or a manager of this KB can remove its members")
    ident = wiring.identity_store()
    target = await ident.get_user(email.strip().lower())
    if target is None or target.org_id != principal.tenant_id or kb_id not in target.kb_access:
        raise NotFoundError("member not found in this KB", details={"email": email})
    if target.role == "owner":
        raise ForbiddenError("the owner cannot be removed")
    is_org_admin = principal.has_role("owner") or principal.has_role("admin")
    if not is_org_admin and (target.email == principal.subject or set(target.kb_access) != {kb_id}):
        raise ForbiddenError("a KB manager can only remove users scoped to this KB alone")
    await ident.delete_user(target.email)
    _log.info("kb_member_removed", org_id=principal.tenant_id, kb_id=kb_id, email=target.email, actor=principal.subject)


@router.get("/me/managed-kbs", response_model=list[str])
async def my_managed_kbs(principal: Annotated[Principal, Depends(get_principal)]) -> list[str]:
    """The KBs the caller manages (empty for a normal member). The console uses this to surface the
    per-KB management + Models affordances to a KB manager who isn't an org admin."""
    return await _managed_kbs(principal.tenant_id, principal.subject)


# --- Cross-KB access requests (request → KB-manager approve → live, revocable grant) ---------
# A member requests access to another KB at a clearance level; the KB's MANAGER approves at a level
# bounded by their own effective clearance in that KB. On approve a LIVE grant (kb_roles
# access_grants, in the KB's tenant_config) both admits the caller and sets their clearance ceiling
# there — so grant/revoke take effect with no re-login. Requests (the audit/overview source of
# truth, with status) live in the ORG tenant_config. Revocable by the requester OR the KB manager.
_ACCESS_REQUESTS_KEY = "access_requests"


async def _load_access_requests(org_id: str) -> list[dict[str, object]]:
    raw = (await get_catalog().get_tenant_config(org_id)).get(_ACCESS_REQUESTS_KEY)
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
    except (ValueError, TypeError):
        return []


async def _save_access_requests(org_id: str, rows: list[dict[str, object]]) -> None:
    await get_catalog().set_tenant_config(org_id, _ACCESS_REQUESTS_KEY, json.dumps(rows))


async def _set_access_grant(kb_id: str, email: str, clearance: int | None) -> None:
    """Write (clearance) or clear (None) the live access grant for ``email`` in ``kb_id``."""
    grants = dict(await kb_roles.load_access_grants(get_catalog(), kb_id))
    e = email.strip().lower()
    if clearance is None:
        grants.pop(e, None)
    else:
        grants[e] = int(clearance)
    await get_catalog().set_tenant_config(kb_id, kb_roles.ACCESS_GRANTS_KEY, json.dumps(grants))


async def _kb_name(org_id: str, kb_id: str) -> str:
    rec = await wiring.kb_store().get(org_id, kb_id)
    return rec.name if rec else kb_id


class AccessRequestInfo(BaseModel):
    id: str
    kb_id: str
    kb_name: str = ""
    requester: str
    requested_clearance: int = 0
    granted_clearance: int | None = None
    reason: str = ""
    status: str = "pending"  # pending | approved | rejected | revoked
    decided_by: str = ""
    created_at: str = ""
    decided_at: str = ""


class AccessRequestBody(BaseModel):
    clearance: int = Field(default=0, ge=0, le=3)
    reason: str = Field(default="", max_length=500)


class AccessDecisionBody(BaseModel):
    approve: bool
    clearance: int = Field(
        default=0, ge=0, le=3, description="Granted level on approve (<= approver's own in this KB)."
    )


class KbDirectoryEntry(BaseModel):
    kb_id: str
    name: str
    has_access: bool
    request_status: str = ""  # ''|pending|approved|rejected|revoked
    granted_clearance: int | None = None


async def _has_access_grant(principal: Principal, kb_id: str) -> bool:
    """Does this caller hold a LIVE cross-KB access grant for ``kb_id``?

    Distinct from :func:`_has_kb_access`, which also answers True for the "no kb_access
    restriction = every KB" case. That is the right answer for "may I reach this KB at all",
    and the wrong one for "may I see it despite the clearance gate"."""
    return principal.subject.strip().lower() in await kb_roles.load_access_grants(get_catalog(), kb_id)


async def _has_kb_access(principal: Principal, kb_id: str) -> bool:
    allowed = principal.attrs.get("kb_access") or []
    if not allowed or kb_id in allowed:
        return True
    return principal.subject.strip().lower() in await kb_roles.load_access_grants(get_catalog(), kb_id)


@router.get("/org/kb-directory", response_model=list[KbDirectoryEntry])
async def kb_directory(principal: Annotated[Principal, Depends(get_principal)]) -> list[KbDirectoryEntry]:
    """Every KB in the caller's org, flagged with whether they already have access and the status of
    any access-request they've made — drives the 'Request access' browser."""
    org = principal.tenant_id
    me = principal.subject.strip().lower()
    kbs = [r for r in await wiring.kb_store().list(org) if r.kb_id != org]
    personal_map = await _load_personal_kbs(org)
    mine = {r["kb_id"]: r for r in await _load_access_requests(org) if str(r.get("requester", "")).lower() == me}
    out: list[KbDirectoryEntry] = []
    for r in kbs:
        # A personal KB is private: it must NEVER appear in another member's directory unless they
        # own it or have already been shared with (a live grant). This keeps it undiscoverable until
        # the owner shares it — so it can't be browsed or requested by others.
        if r.kb_id in personal_map and me != personal_map[r.kb_id] and not await _has_kb_access(principal, r.kb_id):
            continue
        req = mine.get(r.kb_id)
        out.append(
            KbDirectoryEntry(
                kb_id=r.kb_id,
                name=r.name,
                has_access=await _has_kb_access(principal, r.kb_id),
                request_status=str(req.get("status", "")) if req else "",
                granted_clearance=(
                    int(cast("int | str", req["granted_clearance"]))
                    if req and req.get("granted_clearance") is not None
                    else None
                ),
            )
        )
    return out


@router.post("/kb/{kb_id}/access-requests", response_model=AccessRequestInfo, status_code=201)
async def create_access_request(
    kb_id: str, body: AccessRequestBody, principal: Annotated[Principal, Depends(get_principal)]
) -> AccessRequestInfo:
    """Request access to a KB at a clearance level. Any org member who doesn't already have access."""
    org = principal.tenant_id
    if await wiring.kb_store().get(org, kb_id) is None:
        raise NotFoundError("knowledge base not found", details={"kb_id": kb_id})
    if await _has_kb_access(principal, kb_id):
        raise ValidationError("you already have access to this knowledge base")
    rows = await _load_access_requests(org)
    me = principal.subject.strip().lower()
    if any(
        str(r.get("requester", "")).lower() == me and r.get("kb_id") == kb_id and r.get("status") == "pending"
        for r in rows
    ):
        raise ValidationError("you already have a pending request for this knowledge base")
    now = datetime.now(UTC).isoformat()
    row: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "kb_id": kb_id,
        "requester": me,
        "requested_clearance": int(body.clearance),
        "granted_clearance": None,
        "reason": body.reason.strip(),
        "status": "pending",
        "decided_by": "",
        "created_at": now,
        "decided_at": "",
    }
    rows.append(row)
    await _save_access_requests(org, rows)
    _log.info("access_requested", org_id=org, kb_id=kb_id, requester=me, level=int(body.clearance))
    return AccessRequestInfo(kb_name=await _kb_name(org, kb_id), **row)


@router.get("/kb/{kb_id}/access-requests", response_model=list[AccessRequestInfo])
async def list_access_requests(
    kb_id: str, principal: Annotated[Principal, Depends(get_principal)]
) -> list[AccessRequestInfo]:
    """Access-requests for this KB — the manager's queue (owner/admin or a manager of this KB)."""
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only an owner/admin or a manager of this KB may view its access requests")
    name = await _kb_name(principal.tenant_id, kb_id)
    return [
        AccessRequestInfo(kb_name=name, **r)
        for r in await _load_access_requests(principal.tenant_id)
        if r.get("kb_id") == kb_id
    ]


@router.post("/kb/{kb_id}/access-requests/{req_id}/decide", response_model=AccessRequestInfo)
async def decide_access_request(
    kb_id: str, req_id: str, body: AccessDecisionBody, principal: Annotated[Principal, Depends(get_principal)]
) -> AccessRequestInfo:
    """Approve (at a granted level <= the approver's own clearance in this KB) or reject a request.
    Only a manager of this KB (or org owner/admin). On approve a live grant is written."""
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only an owner/admin or a manager of this KB can decide access requests")
    org = principal.tenant_id
    rows = await _load_access_requests(org)
    row = next((r for r in rows if r.get("id") == req_id and r.get("kb_id") == kb_id), None)
    if row is None:
        raise NotFoundError("access request not found", details={"id": req_id})
    if row.get("status") != "pending":
        raise ValidationError(f"request already {row.get('status')}")
    if body.approve:
        ceiling = await kb_roles.effective_kb_clearance(get_catalog(), kb_id, principal)
        if int(body.clearance) > ceiling:
            raise ForbiddenError(
                "you cannot grant a clearance above your own in this KB", details={"your_clearance": ceiling}
            )
        await _set_access_grant(kb_id, str(row["requester"]), int(body.clearance))
        row["status"], row["granted_clearance"] = "approved", int(body.clearance)
    else:
        row["status"], row["granted_clearance"] = "rejected", None
    row["decided_by"] = principal.subject
    row["decided_at"] = datetime.now(UTC).isoformat()
    await _save_access_requests(org, rows)
    _log.info("access_decided", org_id=org, kb_id=kb_id, req_id=req_id, status=row["status"], by=principal.subject)
    return AccessRequestInfo(kb_name=await _kb_name(org, kb_id), **row)


@router.post("/kb/{kb_id}/access/{email}/revoke", status_code=204)
async def revoke_access(kb_id: str, email: str, principal: Annotated[Principal, Depends(get_principal)]) -> None:
    """Revoke a live access grant. The requester may revoke their OWN; a manager of this KB (or org
    owner/admin) may revoke anyone's. Live — the caller loses access immediately, no re-login."""
    org = principal.tenant_id
    target = email.strip().lower()
    is_self = target == principal.subject.strip().lower()
    if not (is_self or await _can_manage_kb(principal, kb_id)):
        raise ForbiddenError("you can only revoke your own access, or one for a KB you manage")
    await _set_access_grant(kb_id, target, None)
    rows = await _load_access_requests(org)
    for r in rows:
        if r.get("kb_id") == kb_id and str(r.get("requester", "")).lower() == target and r.get("status") == "approved":
            r["status"], r["decided_by"], r["decided_at"] = "revoked", principal.subject, datetime.now(UTC).isoformat()
    await _save_access_requests(org, rows)
    _log.info("access_revoked", org_id=org, kb_id=kb_id, subject=target, by=principal.subject)


@router.get("/me/access", response_model=list[AccessRequestInfo])
async def my_access(principal: Annotated[Principal, Depends(get_principal)]) -> list[AccessRequestInfo]:
    """The caller's own access-requests + their status (pending/approved/rejected/revoked)."""
    org = principal.tenant_id
    me = principal.subject.strip().lower()
    rows = [r for r in await _load_access_requests(org) if str(r.get("requester", "")).lower() == me]
    return [AccessRequestInfo(kb_name=await _kb_name(org, str(r["kb_id"])), **r) for r in rows]


@router.get("/org/access-overview", response_model=list[AccessRequestInfo])
async def access_overview(principal: Annotated[Principal, Depends(get_principal)]) -> list[AccessRequestInfo]:
    """Every access-request in the org + status — the org admin's read-only visibility of
    who-has-what-access. Owner/admin only."""
    if not (principal.has_role("owner") or principal.has_role("admin")):
        raise ForbiddenError("only an owner or admin can view the org access overview")
    org = principal.tenant_id
    names: dict[str, str] = {}
    out: list[AccessRequestInfo] = []
    for r in await _load_access_requests(org):
        kid = str(r["kb_id"])
        if kid not in names:
            names[kid] = await _kb_name(org, kid)
        out.append(AccessRequestInfo(kb_name=names[kid], **r))
    return out


# --- Personal (user-owned) knowledge bases (RAG_PERSONAL_KB_ENABLED) ----------------------------
# Any org member may create their OWN KB and share it with other members of the SAME org. The
# creator is auto-appointed the KB's MANAGER, so sharing / members / egress / models all reuse the
# existing manager flow — the governed dept-KB path (owner/admin-only create) stays untouched.
_PERSONAL_KBS_KEY = "personal_kbs"


async def _load_personal_kbs(org_id: str) -> dict[str, str]:
    """{kb_id: owner_email} for this org's personal KBs (stored in the org tenant_config)."""
    raw = (await get_catalog().get_tenant_config(org_id)).get(_PERSONAL_KBS_KEY)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return {str(k): str(v).lower() for k, v in data.items()} if isinstance(data, dict) else {}


async def _grant_kb_access_attr(email: str, kb_id: str) -> None:
    """Append kb_id to a SCOPED member's kb_access so the KB shows in their switcher. An all-KB
    member (empty kb_access) already sees every KB, so leave them untouched — mirrors set_kb_managers."""
    ident = wiring.identity_store()
    e = email.strip().lower()
    u = await ident.get_user(e)
    if u and u.kb_access and kb_id not in u.kb_access:
        await ident.set_user_access(e, roles=u.roles, kb_access=[*u.kb_access, kb_id])


class CreatePersonalKBRequest(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=80)


@router.post("/me/kbs", status_code=201, response_model=KbInfo)
async def create_personal_kb(
    body: CreatePersonalKBRequest, principal: Annotated[Principal, Depends(get_principal)]
) -> KbInfo:
    """Create a PERSONAL, user-owned knowledge base. Any authenticated member may create one when
    RAG_PERSONAL_KB_ENABLED is on. The creator becomes its KB manager (so they can share it) and is
    granted full clearance within it. Namespace-isolated exactly like every KB."""
    if not get_settings().personal_kb_enabled:
        raise ForbiddenError("personal knowledge bases are not enabled on this deployment")
    org = principal.tenant_id
    owner = principal.subject.strip().lower()
    # Namespace the id under the owner so a personal KB can't collide with a governed dept KB and its
    # ownership is legible in the id itself.
    local = owner.split("@", 1)[0]
    kb_id = slug_kb_id(org, f"{local} {body.display_name}")
    existing_kbs = {r.kb_id for r in await wiring.kb_store().list(org)}
    assert_registrable(kb_id, existing_kbs | {org})
    await wiring.provision_kb(kb_id)
    await wiring.kb_store().add(org, kb_id, body.display_name, published=True)
    # Record ownership + auto-appoint the creator as this KB's manager.
    personal = await _load_personal_kbs(org)
    personal[kb_id] = owner
    await get_catalog().set_tenant_config(org, _PERSONAL_KBS_KEY, json.dumps(personal))
    managers = await _load_kb_managers(org)
    managers[kb_id] = [owner]
    await get_catalog().set_tenant_config(org, _KB_MANAGERS_KEY, json.dumps(managers))
    # Give the owner full clearance in their own KB + make it visible in their switcher.
    await _set_access_grant(kb_id, owner, int(Clearance.RESTRICTED))
    await _grant_kb_access_attr(owner, kb_id)
    _log.info("personal_kb_created", org_id=org, kb_id=kb_id, owner=owner)
    return KbInfo(kb_id=kb_id, name=body.display_name, published=True, personal=True, owner=owner)


class ShareKBRequest(BaseModel):
    email: str = Field(..., min_length=3)
    clearance: int = Field(default=0, ge=0, le=3)


@router.post("/kb/{kb_id}/share", response_model=AccessRequestInfo, status_code=201)
async def share_kb(
    kb_id: str, body: ShareKBRequest, principal: Annotated[Principal, Depends(get_principal)]
) -> AccessRequestInfo:
    """Directly share a KB with another member of the SAME org (owner-push). Only a manager of the KB
    (or org owner/admin) may share, at a clearance <= their own in this KB. Writes a live grant and
    records an APPROVED access-request row so the recipient sees it (My access) and either side can
    revoke it — identical semantics to approving a request, just owner-initiated."""
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only an owner/admin or a manager of this KB can share it")
    org = principal.tenant_id
    if await wiring.kb_store().get(org, kb_id) is None:
        raise NotFoundError("knowledge base not found", details={"kb_id": kb_id})
    target = body.email.strip().lower()
    u = await wiring.identity_store().get_user(target)
    if u is None or u.org_id != org:
        raise ValidationError(f"'{body.email}' is not a member of this organisation", details={"email": body.email})
    ceiling = await kb_roles.effective_kb_clearance(get_catalog(), kb_id, principal)
    if int(body.clearance) > ceiling:
        raise ForbiddenError(
            "you cannot share at a clearance above your own in this KB", details={"your_clearance": ceiling}
        )
    await _set_access_grant(kb_id, target, int(body.clearance))
    await _grant_kb_access_attr(target, kb_id)
    # Record/refresh an approved request row so the share shows in the overview + the recipient's
    # My-access and can be revoked through the normal path.
    now = datetime.now(UTC).isoformat()
    rows = await _load_access_requests(org)
    row = next((r for r in rows if r.get("kb_id") == kb_id and str(r.get("requester", "")).lower() == target), None)
    if row is None:
        row = {
            "id": str(uuid.uuid4()),
            "kb_id": kb_id,
            "requester": target,
            "requested_clearance": int(body.clearance),
            "reason": f"shared by {principal.subject}",
            "created_at": now,
        }
        rows.append(row)
    row["granted_clearance"] = int(body.clearance)
    row["status"] = "approved"
    row["decided_by"] = principal.subject
    row["decided_at"] = now
    await _save_access_requests(org, rows)
    _log.info("kb_shared", org_id=org, kb_id=kb_id, target=target, clearance=int(body.clearance), by=principal.subject)
    return AccessRequestInfo(
        id=str(row["id"]),
        kb_id=kb_id,
        kb_name=await _kb_name(org, kb_id),
        requester=target,
        requested_clearance=int(body.clearance),
        granted_clearance=int(body.clearance),
        reason=str(row.get("reason", "")),
        status="approved",
        decided_by=principal.subject,
        created_at=str(row.get("created_at", now)),
        decided_at=now,
    )


# --- Org-wide role display labels (FR-E-role-labels) -------------------------
# The 4 capability roles (reader/ingestor/reviewer/admin) are fixed permission tiers —
# this only customizes what they're CALLED in the UI, per org. Owner is a fixed structural
# concept (the account creator), not a capability tier, so it is never customizable.
_ROLE_LABEL_KEY_PREFIX = "role_label__"
_ROLE_LABEL_MAX = 40
ROLE_LABEL_DEFAULTS: dict[str, str] = {
    "reader": "Associate",
    "ingestor": "Contributor",
    "reviewer": "Reviewer",
    "admin": "Manager",
    "owner": "Owner",
}


class RoleLabels(BaseModel):
    labels: dict[str, str] = Field(
        default_factory=lambda: dict(ROLE_LABEL_DEFAULTS),
        description="role -> display label, for all 5 roles (owner included, read-only).",
    )


@router.get("/org/role-labels", response_model=RoleLabels)
async def get_role_labels(principal: Annotated[Principal, Depends(get_principal)]) -> RoleLabels:
    """The caller's org's custom display labels for the fixed capability roles, falling back to
    the platform defaults for anything not yet customized. Permission values themselves
    (reader/ingestor/reviewer/admin/owner) never change — this is display-only."""
    stored = await get_catalog().get_tenant_config(principal.tenant_id)
    labels = dict(ROLE_LABEL_DEFAULTS)
    for role in CAP_ROLES:
        value = stored.get(f"{_ROLE_LABEL_KEY_PREFIX}{role}")
        if value:
            labels[role] = value
    return RoleLabels(labels=labels)


@router.put("/org/role-labels", response_model=RoleLabels)
async def set_role_labels(body: RoleLabels, principal: Annotated[Principal, Depends(get_principal)]) -> RoleLabels:
    """Set custom display labels for the org's capability roles (owner/admin only). Only the 4
    known capability role keys are accepted; ``owner`` may not be customized — it is a fixed
    structural concept, not a capability tier."""
    if not (principal.has_role("owner") or principal.has_role("admin")):
        raise ForbiddenError("only an owner or admin can change role labels")
    if "owner" in body.labels:
        raise ValidationError("the 'owner' label is fixed and cannot be customized", details={"roles": list(CAP_ROLES)})
    bad = [r for r in body.labels if r not in CAP_ROLES]
    if bad:
        raise ValidationError(f"unknown role(s): {', '.join(bad)}", details={"roles": list(CAP_ROLES)})
    for role, label in body.labels.items():
        clean = label.strip()
        if not clean:
            raise ValidationError(f"label for '{role}' must not be empty")
        if len(clean) > _ROLE_LABEL_MAX:
            raise ValidationError(f"label for '{role}' must be at most {_ROLE_LABEL_MAX} characters")
        await get_catalog().set_tenant_config(principal.tenant_id, f"{_ROLE_LABEL_KEY_PREFIX}{role}", clean)
    _log.info("role_labels_updated", org_id=principal.tenant_id, roles=list(body.labels), actor=principal.subject)
    return await get_role_labels(principal)


# --- Custom named roles (org-wide) ------------------------------------------
# A custom role is a saved LABEL + a chosen subset of the 4 capability primitives
# (reader/ingestor/reviewer/admin). Pure sugar: assigning one just sets those caps on the member
# (RBAC enforcement is unchanged), and the label is reverse-matched from a member's cap-set for
# display. Stored org-wide as one JSON blob in tenant_config.
_CUSTOM_ROLES_KEY = "custom_roles"
_CUSTOM_ROLE_LABEL_MAX = 40
_CUSTOM_ROLES_MAX = 40


class CustomRole(BaseModel):
    id: str = Field(..., min_length=1, max_length=48, description="Stable slug id.")
    label: str = Field(..., min_length=1, max_length=_CUSTOM_ROLE_LABEL_MAX)
    caps: list[str] = Field(default_factory=list, description="Subset of reader/ingestor/reviewer/admin.")


class CustomRoles(BaseModel):
    roles: list[CustomRole] = Field(default_factory=list)


@router.get("/org/roles", response_model=CustomRoles)
async def get_custom_roles(principal: Annotated[Principal, Depends(get_principal)]) -> CustomRoles:
    """The org's custom named roles (label + capability bundle). Any tenant member may READ them
    (needed to display a member's role); only owner/admin may change them."""
    stored = await get_catalog().get_tenant_config(principal.tenant_id)
    raw = stored.get(_CUSTOM_ROLES_KEY)
    if not raw:
        return CustomRoles()
    try:
        data = json.loads(raw)
        return CustomRoles(roles=[CustomRole(**r) for r in data if isinstance(r, dict)])
    except (ValueError, TypeError):
        return CustomRoles()


@router.put("/org/roles", response_model=CustomRoles)
async def set_custom_roles(body: CustomRoles, principal: Annotated[Principal, Depends(get_principal)]) -> CustomRoles:
    """Replace the org's custom named roles (owner/admin only). Each role's caps must be a non-empty
    subset of the 4 fixed capability primitives; ``owner`` is never a capability and is rejected."""
    if not (principal.has_role("owner") or principal.has_role("admin")):
        raise ForbiddenError("only an owner or admin can manage custom roles")
    if len(body.roles) > _CUSTOM_ROLES_MAX:
        raise ValidationError(f"at most {_CUSTOM_ROLES_MAX} custom roles")
    seen: set[str] = set()
    for r in body.roles:
        rid = r.id.strip().lower()
        if not re.match(r"^[a-z0-9][a-z0-9-]*$", rid):
            raise ValidationError("role id must be a slug (lowercase letters/digits/hyphens)", details={"id": r.id})
        if rid in seen:
            raise ValidationError(f"duplicate role id '{rid}'")
        seen.add(rid)
        if not r.label.strip():
            raise ValidationError("role label must not be empty")
        bad = [c for c in r.caps if c not in CAP_ROLES]
        if bad:
            raise ValidationError(f"unknown capabilities: {', '.join(bad)}", details={"caps": list(CAP_ROLES)})
        if not r.caps:
            raise ValidationError(f"role '{r.label}' must grant at least one capability")
    cleaned = [CustomRole(id=r.id.strip().lower(), label=r.label.strip(), caps=sorted(set(r.caps))) for r in body.roles]
    await get_catalog().set_tenant_config(
        principal.tenant_id, _CUSTOM_ROLES_KEY, json.dumps([r.model_dump() for r in cleaned])
    )
    _log.info("custom_roles_updated", org_id=principal.tenant_id, count=len(cleaned), actor=principal.subject)
    return CustomRoles(roles=cleaned)


_MIN_PASSWORD_LEN = 8
_INVITE_TTL_SECONDS = 7 * 24 * 3600  # activation link valid 7 days
_RESET_TTL_SECONDS = 3600  # password reset link valid 1 hour


def _build_link(token: str) -> str:
    """Absolute set-password link when console_base_url is configured, else a relative path the
    console turns into an absolute URL from its own origin."""
    base = get_settings().console_base_url.rstrip("/")
    return f"{base}/set-password?token={token}" if base else f"/set-password?token={token}"


class ActivationResult(BaseModel):
    """Result of activating/inviting a member or requesting a reset."""

    email_sent: bool  # True if the platform mailed the link (SMTP configured)
    link: str  # the set-password link — shown to the admin to share when email wasn't sent


async def _issue_and_notify(user: User, purpose: str, ttl: int, *, subject: str, intro: str) -> ActivationResult:
    """Issue a one-time token for the user, email the link if SMTP is configured, and always
    return the link so an admin on an air-gapped site can copy and share it."""
    token = await wiring.auth_token_store().issue(user.email, purpose, ttl)
    link = _build_link(token)
    body = (
        f"Hello {user.name},\n\n{intro}\n\n{link}\n\n"
        "This link is single-use and will expire. If you didn't expect this, ignore this email.\n\n"
        "— RAG as a Service"
    )
    sent = send_email(user.email, subject, body)
    return ActivationResult(email_sent=sent, link=link)


@router.post("/org/members/{email}/activate", response_model=ActivationResult)
async def activate_member(email: str, principal: Annotated[Principal, Depends(get_principal)]) -> ActivationResult:
    """Send (or produce) an activation link so an invited member can set a password and sign in
    (owner/admin only). Returns the link for the admin to share when email isn't configured."""
    if not (principal.has_role("owner") or principal.has_role("admin")):
        raise ForbiddenError("only an owner or admin can activate members")
    ident = wiring.identity_store()
    target = await ident.get_user(email.strip().lower())
    if target is None or target.org_id != principal.tenant_id:
        raise NotFoundError("member not found", details={"email": email})
    _log.info("member_activation_sent", org_id=principal.tenant_id, email=target.email)
    return await _issue_and_notify(
        target,
        "invite",
        _INVITE_TTL_SECONDS,
        subject="You've been given access to RAG as a Service",
        intro="An administrator has created your account. Set your password to activate it and sign in:",
    )


class MemberStatusRequest(BaseModel):
    status: str = Field(..., pattern="^(active|suspended)$")


@router.post("/org/members/{email}/status", response_model=MemberInfo)
async def set_member_status(
    email: str, body: MemberStatusRequest, principal: Annotated[Principal, Depends(get_principal)]
) -> MemberInfo:
    """Suspend or reactivate a member (owner/admin only). The owner is protected."""
    if not (principal.has_role("owner") or principal.has_role("admin")):
        raise ForbiddenError("only an owner or admin can change member status")
    ident = wiring.identity_store()
    target = await ident.get_user(email.strip().lower())
    if target is None or target.org_id != principal.tenant_id:
        raise NotFoundError("member not found", details={"email": email})
    if target.role == "owner":
        raise ForbiddenError("the owner cannot be suspended")
    await ident.set_status(target.email, body.status)
    _log.info("member_status_changed", org_id=principal.tenant_id, email=target.email, status=body.status)
    updated = await ident.get_user(target.email)
    assert updated is not None
    return _member_info(updated)


@router.delete("/org/members/{email}", status_code=204)
async def revoke_member(email: str, principal: Annotated[Principal, Depends(get_principal)]) -> None:
    """Permanently remove a member from the org (owner/admin only). The owner is protected, and an
    admin cannot revoke themselves. Suspension (reversible) is the softer option; this deletes the
    account. KBs/data the member authored are unaffected (owned by the org, not the user)."""
    if not (principal.has_role("owner") or principal.has_role("admin")):
        raise ForbiddenError("only an owner or admin can remove members")
    ident = wiring.identity_store()
    target = await ident.get_user(email.strip().lower())
    if target is None or target.org_id != principal.tenant_id:
        raise NotFoundError("member not found", details={"email": email})
    if target.role == "owner":
        raise ForbiddenError("the owner cannot be removed")
    if target.email == principal.subject:
        raise ForbiddenError("you cannot remove your own account")
    await ident.delete_user(target.email)
    _log.info("member_revoked", org_id=principal.tenant_id, email=target.email, actor=principal.subject)


class SetPasswordRequest(BaseModel):
    token: str = Field(..., min_length=1)
    password: str = Field(..., min_length=_MIN_PASSWORD_LEN)


@router.post("/auth/set-password", status_code=204)
async def set_password(body: SetPasswordRequest) -> None:
    """Set a password from a one-time activation/reset token, activating the account. Public —
    the token is the credential."""
    consumed = await wiring.auth_token_store().consume(body.token)
    if consumed is None:
        raise UnauthorizedError("this link is invalid or has expired — ask your admin for a new one")
    email, _purpose = consumed
    ident = wiring.identity_store()
    if await ident.get_user(email) is None:
        raise UnauthorizedError("this link is invalid or has expired — ask your admin for a new one")
    await ident.set_password(email, hash_password(body.password))
    _log.info("password_set", email=email)


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., min_length=3)


@router.post("/auth/forgot-password", status_code=202)
async def forgot_password(body: ForgotPasswordRequest) -> dict[str, bool]:
    """Request a password-reset link. Always returns the same response whether or not the email
    exists (no account enumeration); the link is delivered by email only."""
    ident = wiring.identity_store()
    user = await ident.get_user(body.email.strip().lower())
    if user is not None and user.status != "suspended":
        await _issue_and_notify(
            user,
            "reset",
            _RESET_TTL_SECONDS,
            subject="Reset your RAG as a Service password",
            intro="We received a request to reset your password. Set a new one here:",
        )
    return {"ok": True}


# --- Service-account API keys (FR-I4) ----------------------------------------
class ApiKeyInfo(BaseModel):
    key_id: str
    name: str
    prefix: str
    scope: str
    created_at: datetime


class CreateApiKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    scope: Literal["read", "ingest", "admin"] = "read"


class CreatedApiKey(ApiKeyInfo):
    key: str = Field(description="The full key — shown ONCE. Store it now; it can't be retrieved again.")


def _key_info(rec: ApiKeyRecord) -> ApiKeyInfo:
    return ApiKeyInfo(key_id=rec.key_id, name=rec.name, prefix=rec.prefix, scope=rec.scope, created_at=rec.created_at)


@router.get("/api-keys", response_model=list[ApiKeyInfo])
async def list_api_keys(principal: Annotated[Principal, Depends(get_principal)]) -> list[ApiKeyInfo]:
    """The org's API keys (never the secret)."""
    return [_key_info(r) for r in await wiring.api_key_store().list(principal.tenant_id)]


@router.post("/api-keys", response_model=CreatedApiKey, status_code=201)
async def create_api_key(
    body: CreateApiKeyRequest, principal: Annotated[Principal, Depends(get_principal)]
) -> CreatedApiKey:
    """Mint a scoped service-account key for programmatic access. The full key is returned
    ONCE — only its hash is stored. Owner/admin only."""
    if not principal.has_role("admin"):  # noqa: SIM102  # inner guard kept nested for the explanatory comment
        # A KB manager may mint a READ-ONLY agent token (least privilege). ingest/admin-scoped keys
        # stay owner/admin-only. (Frontend Agents page offers this to managers; enforced here too.)
        if not (body.scope == "read" and await _managed_kbs(principal.tenant_id, principal.subject)):
            raise ForbiddenError("only an owner/admin can create API keys (KB managers may mint read-only tokens)")
    full, prefix, key_hash = mint_key()
    rec = ApiKeyRecord(
        key_id=str(uuid.uuid4()),
        org_id=principal.tenant_id,
        name=body.name.strip(),
        prefix=prefix,
        scope=body.scope,
        created_at=datetime.now(UTC),
    )
    await wiring.api_key_store().create(rec, key_hash)
    _log.info("api_key_created", org_id=principal.tenant_id, key_id=rec.key_id, scope=body.scope)
    return CreatedApiKey(**_key_info(rec).model_dump(), key=full)


@router.delete("/api-keys/{key_id}", status_code=204)
async def revoke_api_key(key_id: str, principal: Annotated[Principal, Depends(get_principal)]) -> None:
    """Revoke a key immediately (owner/admin only)."""
    if not principal.has_role("admin"):
        raise ForbiddenError("only an owner or admin can revoke API keys")
    await wiring.api_key_store().revoke(principal.tenant_id, key_id)
    _log.info("api_key_revoked", org_id=principal.tenant_id, key_id=key_id)


# --- Dev auth token (dev_jwt mode only; forbidden in prod) -------------------
class DevTokenRequest(BaseModel):
    subject: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    roles: list[str] = Field(default_factory=lambda: ["reader", "ingestor", "reviewer"])
    groups: list[str] = Field(default_factory=list, description="ACL group memberships (FR-A6).")
    clearance: str | int = Field(default=0, description="Classification clearance 0..3 (banking pack).")


@router.post("/dev/token")
async def dev_token(body: DevTokenRequest) -> dict[str, str]:
    """Mint a dev JWT for the console login. DEV ONLY — replaced by OIDC in Phase-1."""
    settings = get_settings()
    if settings.env == "prod" or settings.auth_mode != "dev_jwt":
        raise ForbiddenError("dev token endpoint disabled", details={"env": settings.env})
    token = issue_dev_jwt(
        body.subject,
        body.tenant_id,
        roles=body.roles,
        attrs={"groups": body.groups, "clearance": parse_classification(body.clearance)},
    )
    _log.warning("dev_token_issued", subject=body.subject, tenant_id=body.tenant_id, note="dev-only; not for prod")
    return {"access_token": token, "token_type": "bearer", "tenant_id": body.tenant_id}


# --- Hierarchical guardrail policy (top-down RBAC lock model) -----------------
# One governance surface over the platform's live safety controls. PLATFORM (SUPER_ADMIN) sets the
# sovereign floor and locks the non-negotiables; ORG (owner/admin) and KB (manager) tune what is
# left unlocked, within the envelope handed down. See guardrails_policy.py for the resolver + catalog.
class GuardrailUpdateRequest(BaseModel):
    """A partial policy write: ``{control_id: {value, locked}}`` for the caller's own tier."""

    updates: dict[str, guardrails_policy.PolicyEntry] = Field(default_factory=dict)


class GuardrailCatalogItem(BaseModel):
    id: str
    label: str
    category: str
    kind: str
    constitutional: bool
    default_owner: str
    help: str


@router.get("/guardrails/catalog", response_model=list[GuardrailCatalogItem])
async def guardrail_catalog(_principal: Annotated[Principal, Depends(get_principal)]) -> list[GuardrailCatalogItem]:
    """The static catalog of guardrail controls (id/label/category/constitutional). Any member may read."""
    return [
        GuardrailCatalogItem(
            id=c.id,
            label=c.label,
            category=c.category,
            kind=c.kind,
            constitutional=c.constitutional,
            default_owner=c.default_owner,
            help=c.help,
        )
        for c in guardrails_policy.CATALOG
    ]


async def _record_policy_change(tenant_id: str, actor: str, tier: str, updates: Mapping[str, object]) -> None:
    await wiring.audit_service().record(
        tenant_id=tenant_id,
        trace_id=str(uuid.uuid4()),
        action=AuditAction.GUARDRAIL,
        actor=actor,
        payload={"event": "policy_change", "tier": tier, "controls": sorted(updates.keys())},
    )


@router.get("/platform/guardrails", response_model=guardrails_policy.ScopePolicy)
async def get_platform_guardrails(
    principal: Annotated[Principal, Depends(get_principal)],
) -> guardrails_policy.ScopePolicy:
    """The fleet-wide platform guardrail policy (SUPER_ADMIN)."""
    _require_platform_admin(principal)
    return await guardrails_policy.load_policy(get_catalog(), guardrails_policy.PLATFORM_SCOPE)


@router.put("/platform/guardrails", response_model=guardrails_policy.ScopePolicy)
async def set_platform_guardrails(
    body: GuardrailUpdateRequest, principal: Annotated[Principal, Depends(get_principal)]
) -> guardrails_policy.ScopePolicy:
    """SUPER_ADMIN sets/locks fleet-wide guardrail defaults."""
    _require_platform_admin(principal)
    try:
        policy = await guardrails_policy.apply_policy_update(
            get_catalog(), tier="platform", org_id=None, kb_id=None, updates=body.updates
        )
    except guardrails_policy.PolicyWriteError as exc:
        raise ValidationError(str(exc)) from exc
    await _record_policy_change(guardrails_policy.PLATFORM_SCOPE, principal.subject, "platform", body.updates)
    return policy


@router.get("/org/guardrails", response_model=guardrails_policy.ScopePolicy)
async def get_org_guardrails(principal: Annotated[Principal, Depends(get_principal)]) -> guardrails_policy.ScopePolicy:
    """The org-wide guardrail policy (owner/admin read; any member may read to display it)."""
    return await guardrails_policy.load_policy(get_catalog(), principal.tenant_id)


@router.put("/org/guardrails", response_model=guardrails_policy.ScopePolicy)
async def set_org_guardrails(
    body: GuardrailUpdateRequest, principal: Annotated[Principal, Depends(get_principal)]
) -> guardrails_policy.ScopePolicy:
    """owner/admin set/lock org-wide guardrails. A platform-locked control is refused (422)."""
    if not (principal.has_role("owner") or principal.has_role("admin")):
        raise ForbiddenError("only an owner or admin can change org guardrails")
    try:
        policy = await guardrails_policy.apply_policy_update(
            get_catalog(), tier="org", org_id=principal.tenant_id, kb_id=None, updates=body.updates
        )
    except guardrails_policy.PolicyWriteError as exc:
        raise ValidationError(str(exc)) from exc
    await _record_policy_change(principal.tenant_id, principal.subject, "org", body.updates)
    return policy


@router.get("/kb/{kb_id}/guardrails/effective", response_model=list[guardrails_policy.EffectiveEntry])
async def get_kb_guardrails_effective(
    kb_id: str, principal: Annotated[Principal, Depends(tenant_guard)]
) -> list[guardrails_policy.EffectiveEntry]:
    """The RESOLVED guardrail policy for this KB (platform -> org -> kb, honouring locks), with the
    owner tier and locked-by tier per control — for the Guardrails page and for enforcement."""
    return await guardrails_policy.resolve_effective(get_catalog(), kb_id)


@router.put("/kb/{kb_id}/guardrails", response_model=guardrails_policy.ScopePolicy)
async def set_kb_guardrails(
    kb_id: str, body: GuardrailUpdateRequest, principal: Annotated[Principal, Depends(tenant_guard)]
) -> guardrails_policy.ScopePolicy:
    """A KB manager (or org admin) sets guardrails for THIS KB. A platform- or org-locked control
    is refused (422)."""
    if not await _can_manage_kb(principal, kb_id):
        raise ForbiddenError("only a KB manager or org admin can change this KB's guardrails")
    org_id = kb_id.split("__", 1)[0] if "__" in kb_id else kb_id
    try:
        policy = await guardrails_policy.apply_policy_update(
            get_catalog(), tier="kb", org_id=org_id, kb_id=kb_id, updates=body.updates
        )
    except guardrails_policy.PolicyWriteError as exc:
        raise ValidationError(str(exc)) from exc
    await _record_policy_change(kb_id, principal.subject, "kb", body.updates)
    return policy
