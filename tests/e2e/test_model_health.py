"""Live model health: is every configured backend actually answering, and how fast?

The roster in /system/info reads CONFIGURATION — it says what is SET, not what works. Measured
live, a reranker fallback sat pointed at a real endpoint with no credential and nothing in the
product showed it: the boot log knew, the console did not, and the first symptom would have been
a failed query during an outage. This endpoint probes the same endpoints the boot preflight does,
with the same prober.
"""

from __future__ import annotations

import httpx
import pytest
from svc_control_plane import api as api_module
from svc_control_plane.preflight import BackendHealth

from rag_common import issue_dev_jwt

pytestmark = pytest.mark.e2e


def _client(roles: list[str]) -> httpx.AsyncClient:
    from rag_api.app import create_app

    token = issue_dev_jwt("u", "acme", roles=roles)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()),
        base_url="http://t",
        headers={"Authorization": f"Bearer {token}"},
    )


@pytest.fixture(autouse=True)
def _probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The live situation this endpoint exists for: the PLATFORM-default generation credential is
    wrong (401) while the infrastructure backends answer — and one fallback is misconfigured."""

    async def _fake(settings: object) -> list[BackendHealth]:
        return [
            BackendHealth(
                name="Generation LLM",
                role="Generation",
                endpoint="platform.example",
                online=False,
                latency_ms=None,
                detail="Generation LLM at https://platform.example rejected the configured credential (HTTP 401)",
            ),
            BackendHealth(
                name="Embedder",
                role="Embedding",
                endpoint="models.example",
                online=True,
                latency_ms=231,
                detail="",
            ),
            BackendHealth(
                name="Reranker (fallback)",
                role="Reranking",
                endpoint="models.example",
                online=False,
                latency_ms=None,
                detail="Reranker (fallback) rejected the configured credential (HTTP 401)",
            ),
        ]

    monkeypatch.setattr(api_module, "probe_backend_health", _fake)
    monkeypatch.setattr(api_module, "_health_cache", {})


async def test_an_admin_sees_every_backend_with_its_latency(inmemory: None) -> None:
    async with _client(["admin"]) as c:
        body = (await c.get("/v1/system/model-health")).json()

    by_name = {b["name"]: b for b in body["backends"]}
    assert by_name["Embedder"]["online"] is True
    assert by_name["Embedder"]["latency_ms"] == 231
    broken = by_name["Reranker (fallback)"]
    assert broken["online"] is False
    assert "credential" in broken["detail"]
    # A failed probe reports NO latency: its elapsed time is the timeout, not a round trip, and
    # printing it beside an endpoint that never answered would read as a slow-but-working backend.
    assert broken["latency_ms"] is None


async def test_a_member_sees_only_the_model_that_answers_them(inmemory: None) -> None:
    async with _client(["reader"]) as c:
        body = (await c.get("/v1/system/model-health")).json()
    assert [b["name"] for b in body["backends"]] == ["Generation LLM"]


async def test_the_probe_is_cached_and_refresh_forces_a_new_round(inmemory: None) -> None:
    """Each probe is a real request to a real model endpoint, so a page refresh must not fan out
    one per backend — but the Re-check button has to mean it."""
    async with _client(["admin"]) as c:
        first = (await c.get("/v1/system/model-health")).json()
        second = (await c.get("/v1/system/model-health")).json()
        forced = (await c.get("/v1/system/model-health?refresh=true")).json()

    assert first["cached"] is False
    assert second["cached"] is True, "a second read inside the TTL must not re-probe"
    assert forced["cached"] is False, "refresh=true must probe again"


async def test_the_generation_row_reports_the_backend_that_actually_answers(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reported live: the topbar badge said "online, 85 ms" while this table said "offline (401)"
    for the SAME generation row. The badge resolves the KB's org-wide model CONNECTION; the table
    was reading the platform-default env (a different endpoint and credential). A row that shows
    the connection's model must show the connection's health."""

    class _Gateway:
        url = "https://models.example/openai/v1"
        model_name = "gemma4-31b"

        async def complete(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> str:
            return "ok"

    async def _resolve(kb_id: str, model: str | None = None) -> object:
        return _Gateway()

    from svc_control_plane import wiring

    monkeypatch.setattr(wiring, "resolve_tenant_gateway", _resolve)

    async with _client(["admin"]) as c:
        await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
        body = (await c.get("/v1/system/model-health?kb_id=acme__kb")).json()

    generation = next(b for b in body["backends"] if b["role"] == "Generation")
    assert generation["online"] is True, "the platform default 401s here; the KB's connection answers"
    assert generation["endpoint"] == "models.example"
    assert generation["latency_ms"] is not None
