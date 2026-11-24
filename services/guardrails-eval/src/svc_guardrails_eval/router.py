"""Guardrails & Eval router — health only in Phase-0.

Guardrails G1 (prompt-injection) and G3 (toxicity) are Phase-1 (NeMo/LlamaGuard);
their inline hook attaches at the LLM gateway. Eval results (``GET /v1/eval/{kb_id}``)
are surfaced via the control-plane API.
"""

from __future__ import annotations

from fastapi import APIRouter

from rag_common.health import health_router

router = APIRouter(prefix="/internal/guardrails-eval")
router.include_router(health_router("guardrails-eval"))
