"""Query reformulation must use the KB's OWN model, like generation and image captioning.

Measured live on a KB whose single document IS an architecture diagram: "explain me the whole
architecture" scored 0.0165 — below any usable abstain gate — and the agentic retry that exists
to rescue exactly that reported "no_transform". The rewrite was calling the PLATFORM-default
gateway, which this deployment has no working key for, so every reformulation 401'd and silently
degraded to the unchanged question. Bound to the KB's own model the same query retrieved at
0.9132.
"""

from __future__ import annotations

import pytest
from svc_control_plane import wiring

from rag_core.interfaces import LLMGateway


class _Gateway:
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def model_name(self) -> str:
        return self._name

    async def complete(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> str:
        return ""


def _gateway_of(transformer: object) -> LLMGateway | None:
    """The gateway the composed transformer reformulates with, if any."""
    for part in getattr(transformer, "_transformers", (transformer,)):
        gw = getattr(part, "_gw", None)
        if gw is not None:
            return gw  # type: ignore[no-any-return]
    return None


async def test_reformulation_is_bound_to_the_kbs_own_model(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _tenant(tenant_id: str, model: str | None = None) -> LLMGateway:
        assert tenant_id == "acme__hr"
        return _Gateway("kb-model")  # type: ignore[return-value]

    monkeypatch.setattr(wiring, "resolve_tenant_gateway", _tenant)
    gw = _gateway_of(await wiring.query_transformer_for("acme__hr"))
    assert gw is not None and gw.model_name == "kb-model"


async def test_a_kb_with_no_usable_connection_still_gets_a_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Degrade visibly, never fail the ask: without a tenant gateway the platform default stands
    in, and the deterministic decomposition works either way."""

    async def _broken(tenant_id: str, model: str | None = None) -> LLMGateway:
        raise RuntimeError("no model connection")

    monkeypatch.setattr(wiring, "resolve_tenant_gateway", _broken)
    assert await wiring.query_transformer_for("acme__hr") is not None
