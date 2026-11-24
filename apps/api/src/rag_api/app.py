"""RAG as a Service — modular-monolith FastAPI application (ADR-0001).

One deployable process mounts every service module's router. Package boundaries and
the rag-core Protocols mirror the TDD's 8-layer microservice target, so splitting a
layer into its own container later is a deployment change, not a rewrite.

Cross-cutting concerns (structured logging, OTel tracing, tenancy/error middleware)
are installed once here for the whole app.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response

# Service module routers (each is independently split-able later).
from svc_control_plane.router import router as control_plane_router
from svc_doc_intelligence.router import router as doc_intel_router
from svc_generation.router import router as generation_router
from svc_guardrails_eval.router import router as guardrails_router
from svc_indexing.router import router as indexing_router
from svc_ingestion.router import router as ingestion_router
from svc_retrieval.router import router as retrieval_router

from rag_clients.factory import close_all
from rag_common import configure_logging, get_logger, get_settings, init_tracing
from rag_common.airgap import assert_air_gap
from rag_common.middleware import install as install_middleware
from rag_common.tracing import instrument_app
from rag_mcp.server import build_http_server, http_app

# Multi-tenant MCP server (FR-H2, streamable HTTP): built PER APP in create_app and kept on
# app.state, because its session manager may be run exactly once per instance (the SDK raises on a
# second run()). A module-level singleton therefore made create_app a one-shot factory — the second
# app in a process died at startup, which is why two e2e tests could not both enter the /mcp
# lifespan. One app per process hid it in production; it is still the factory being wrong.

_SERVICE_ROUTERS = [
    control_plane_router,  # public /v1 API surface
    ingestion_router,
    doc_intel_router,
    indexing_router,
    retrieval_router,
    generation_router,
    guardrails_router,
]


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # The MCP streamable-HTTP transport needs its session manager running for the
    # lifetime of the app (mounted sub-app lifespans don't run automatically). THIS app's
    # server, not a shared one — see the note above build_http_server's use in create_app.
    async with app.state.mcp_server.session_manager.run():
        from svc_control_plane.dual_control_actions import register_executors  # noqa: PLC0415

        register_executors()  # bind maker-checker action executors (banking pack)
        await _load_runtime_flags()
        await _ensure_shared_kb()
        await _preflight()
        stop, task = _maybe_start_eval_scheduler()
        bstop, btask = _maybe_start_bulk_ingest_worker()
        erstop, ertask = _maybe_start_erasure_scheduler()
        ristop, ritask = _maybe_start_auto_reindexer()
        hstop, htask = _maybe_start_health_poller()
        try:
            yield
        finally:
            for ev, tk in (
                (stop, task),
                (bstop, btask),
                (erstop, ertask),
                (ristop, ritask),
                (hstop, htask),
            ):
                if ev is not None and tk is not None:
                    ev.set()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await asyncio.wait_for(tk, timeout=5)
    # Real adapters (Qdrant/OpenSearch/TEI/vLLM/Postgres) hold connection pools that
    # must be closed on shutdown; in-memory adapters are no-ops here.
    await close_all()


async def _preflight() -> None:
    """Report any model backend that is configured but cannot serve. Non-blocking by default;
    in strict mode (RAG_PREFLIGHT_STRICT) a critical-backend failure fails boot."""
    from svc_control_plane import preflight  # noqa: PLC0415 — keep app import light

    try:
        await preflight.run()
    except preflight.PreflightStrictError:
        raise  # strict mode: a critical backend is unusable — fail boot loudly, don't degrade
    except Exception as exc:  # a diagnostic must never be the reason the app fails to start
        get_logger(__name__).warning("preflight_failed", error=str(exc)[:160])


async def _load_runtime_flags() -> None:
    """Apply persisted admin overrides (feature flags + approved egress hosts) before serving."""
    with contextlib.suppress(Exception):  # a store hiccup must never block startup
        from svc_control_plane import egress_requests, feature_flags, wiring  # noqa: PLC0415

        await feature_flags.load_overrides(wiring.feature_flag_store())
        await egress_requests.load_approved(wiring.egress_request_store())


async def _ensure_shared_kb() -> None:
    """Provision SHARED_KB_ID's namespace at boot (Federated RAG Tier C), so the very first
    query against it — before any tenant has ingested anything — hits a real empty
    collection/index rather than a not-yet-provisioned one. Idempotent; a store hiccup here
    must never block startup (first ingest into it would provision it anyway, see api.py)."""
    with contextlib.suppress(Exception):
        from svc_control_plane import wiring  # noqa: PLC0415
        from svc_control_plane.knowledge_base import SHARED_KB_ID  # noqa: PLC0415

        await wiring.provision_kb(SHARED_KB_ID)


def _maybe_start_bulk_ingest_worker() -> tuple[asyncio.Event | None, asyncio.Task[None] | None]:
    """Start the in-process async bulk-ingestion worker (drains the job queue)."""
    if not get_settings().bulk_ingest_worker_enabled:
        return None, None
    from svc_control_plane import ingest_jobs, wiring  # noqa: PLC0415

    stop = asyncio.Event()
    return stop, asyncio.create_task(ingest_jobs.worker_loop(stop, wiring.ingest_job_store()))


def _maybe_start_eval_scheduler() -> tuple[asyncio.Event | None, asyncio.Task[None] | None]:
    """Start the in-process scheduler loop iff any scheduled job is enabled — continuous
    evaluation (FR-G9) and/or delta connector sync (FR-A8). One loop drives both."""
    settings = get_settings()
    if not (settings.continuous_eval_enabled or settings.sync_scheduler_enabled):
        return None, None
    from svc_control_plane.continuous_eval import scheduler_loop  # noqa: PLC0415  (avoid import cost when off)

    stop = asyncio.Event()
    return stop, asyncio.create_task(scheduler_loop(stop))


def _maybe_start_erasure_scheduler() -> tuple[asyncio.Event | None, asyncio.Task[None] | None]:
    """Start the retention-driven auto-erasure worker iff enabled (DPDP FR-H8; default off)."""
    if not get_settings().erasure_schedule_enabled:
        return None, None
    from svc_control_plane.erasure_worker import worker_loop  # noqa: PLC0415  (avoid import cost + cycle when off)

    stop = asyncio.Event()
    return stop, asyncio.create_task(worker_loop(stop))


def _maybe_start_auto_reindexer() -> tuple[asyncio.Event | None, asyncio.Task[None] | None]:
    """Start the stale-index re-indexer iff an operator named KBs to keep current (FR-C8).

    Empty list = off, which is the default: staleness is always REPORTED, but rebuilding a
    tenant's corpus on a schedule is an opt-in write, chosen per knowledge base."""
    from svc_control_plane.reindex import auto_reindex_kbs, worker_loop  # noqa: PLC0415  (cost + cycle when off)

    if not auto_reindex_kbs():
        return None, None
    stop = asyncio.Event()
    return stop, asyncio.create_task(worker_loop(stop))


def _maybe_start_health_poller() -> tuple[asyncio.Event | None, asyncio.Task[None] | None]:
    """Start the continuous model-health poller iff enabled (default off). The Redis leader probes
    the platform backends every health_poll_seconds and publishes a snapshot the endpoint serves
    without probing — replacing the console's click-to-recheck."""
    if not get_settings().health_poll_enabled:
        return None, None
    from svc_control_plane.health_poller import poller_loop  # noqa: PLC0415  (avoid import cost when off)

    stop = asyncio.Event()
    return stop, asyncio.create_task(poller_loop(stop))


def create_app() -> FastAPI:
    """Build and configure the monolith app."""
    settings = get_settings()
    # Air-gap preflight: on a fully-offline deployment, refuse to start if any config would
    # breach air-gap (remote egress / off-box backends / missing update-verify key). No-op off.
    assert_air_gap(settings)
    configure_logging()
    init_tracing(settings.otel_service_name)

    # Ops visibility: log which Phase-1 features + backends are active at startup.
    get_logger(__name__).info(
        "phase1_features",
        env=settings.env,
        llm_backend=settings.llm_backend,
        contextual_retrieval=settings.contextual_retrieval_enabled,
        adaptive_router=settings.adaptive_router_enabled,
        query_transform=settings.query_transform_enabled,
        faithfulness=settings.faithfulness_enabled,
        guardrails=settings.guardrails_enabled,
    )

    app = FastAPI(
        title="RAG as a Service",
        version="0.1.0-phase0",
        summary="Sovereign, multi-tenant Agentic RAG-as-a-Service (Phase-0 core).",
        lifespan=_lifespan,
    )
    install_middleware(app)
    instrument_app(app)

    # CORS for the dev console SPA (localhost:5173). Prod serves the SPA same-origin
    # behind the ingress, so this allowlist is dev-focused and configurable.
    origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
    if origins:
        from fastapi.middleware.cors import CORSMiddleware  # noqa: PLC0415

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    for router in _SERVICE_ROUTERS:
        app.include_router(router)

    # Agent surface (FR-H2): multi-tenant MCP over streamable HTTP at /mcp — every
    # request authenticated with a platform JWT and scoped to that principal's org.
    app.state.mcp_server = build_http_server()
    app.mount("/mcp", http_app(app.state.mcp_server))

    @app.get("/health", tags=["health"], operation_id="health_root")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": settings.otel_service_name, "env": settings.env}

    @app.get("/livez", tags=["health"], operation_id="liveness")
    async def livez() -> dict[str, str]:
        """Liveness: the PROCESS is up. Dependency-free — a failing liveness restarts the pod, so
        it must never check a downstream store (that would restart-loop every replica on a blip)."""
        return {"status": "alive"}

    @app.get("/readyz", tags=["health"], operation_id="readiness")
    async def readyz(response: Response) -> dict[str, object]:
        """Readiness: can this replica SERVE? Checks the stores a query needs. A failing readiness
        sheds traffic (pulled from the Service) WITHOUT a restart, so a store blip self-heals."""
        from rag_api import health as _health  # noqa: PLC0415 — avoid import at module load

        ready, checks = await _health.readiness()
        if not ready:
            response.status_code = 503
        return {"ready": ready, "checks": checks}

    return app


app = create_app()
