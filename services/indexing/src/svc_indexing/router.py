"""Indexing router — health only in Phase-0 (invoked in-process by control-plane)."""

from __future__ import annotations

from fastapi import APIRouter

from rag_common.health import health_router

router = APIRouter(prefix="/internal/indexing")
router.include_router(health_router("indexing"))
