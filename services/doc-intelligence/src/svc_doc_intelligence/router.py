"""Document Intelligence router — health only in Phase-0.

Extraction is invoked in-process by the control-plane ingest pipeline (modular
monolith); no public HTTP extract endpoint is exposed yet. The health router keeps
the module independently observable and ready for the microservice split.
"""

from __future__ import annotations

from fastapi import APIRouter

from rag_common.health import health_router

router = APIRouter(prefix="/internal/doc-intelligence")
router.include_router(health_router("doc-intelligence"))
