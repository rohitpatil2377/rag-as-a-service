"""Control-plane router aggregate: public `/v1` API + health."""

from __future__ import annotations

from fastapi import APIRouter

from rag_common.health import health_router
from svc_control_plane.api import router as api_router

router = APIRouter()
router.include_router(api_router)
router.include_router(health_router("control-plane"))
