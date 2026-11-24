"""Retrieval router — health only in Phase-0 (invoked in-process by control-plane).

The public retrieval endpoint (``POST /v1/kb/{id}/retrieve``, for agent consumers) is
exposed by the control-plane API surface (TDD §7); this module stays internal.
"""

from __future__ import annotations

from fastapi import APIRouter

from rag_common.health import health_router

router = APIRouter(prefix="/internal/retrieval")
router.include_router(health_router("retrieval"))
