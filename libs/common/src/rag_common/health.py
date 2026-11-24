"""Shared health/readiness router.

Every service module exposes ``/health`` (liveness) and the monolith aggregates a
top-level ``/health`` and ``/ready``. Readiness checks are pluggable so a service
can assert its stores are reachable before receiving traffic.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import APIRouter

ReadyCheck = Callable[[], Awaitable[bool]]


def health_router(component: str, ready_checks: list[ReadyCheck] | None = None) -> APIRouter:
    """Build a health router for ``component``."""
    router = APIRouter(tags=["health"])
    checks = ready_checks or []
    slug = component.replace("-", "_")

    @router.get("/health", operation_id=f"health_{slug}")
    async def health() -> dict[str, str]:
        return {"status": "ok", "component": component}

    @router.get("/ready", operation_id=f"ready_{slug}")
    async def ready() -> dict[str, object]:
        results = {}
        healthy = True
        for check in checks:
            name = getattr(check, "__name__", "check")
            try:
                ok = await check()
            except Exception:
                ok = False
            results[name] = ok
            healthy = healthy and ok
        return {"status": "ok" if healthy else "degraded", "checks": results}

    return router
