"""Billing — invoice from recorded usage (integration) + endpoint gating (e2e)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import httpx
import pytest

pytestmark = pytest.mark.e2e


def _period() -> str:
    return datetime.now(UTC).strftime("%Y-%m")


async def test_invoice_reflects_recorded_usage(inmemory: None) -> None:
    from svc_control_plane import billing, wiring

    from rag_schemas.common import AuditAction

    audit = wiring.audit_service()
    org = "acme"
    await audit.record(tenant_id=org, trace_id="t", action=AuditAction.QUERY, actor="u", payload={"query": "hi"})
    await audit.record(tenant_id=org, trace_id="t", action=AuditAction.INGEST, actor="u", payload={"doc_id": "d"})
    await audit.record(
        tenant_id=org, trace_id="t", action=AuditAction.GENERATE, actor="u", payload={"total_tokens": 5000}
    )

    inv = await billing.invoice(org, _period(), actor="billing")
    assert inv.queries == 1 and inv.documents == 1 and inv.total_tokens == 5000
    # 1*0.5 + 1*2.0 + 5*0.1 = 3.0 at the default rate card.
    assert inv.subtotal == 3.0
    # The invoice itself is recorded on the tamper-evident chain (auditable statement).
    actions = {e.action for e in await audit.list(org, limit=50)}
    assert AuditAction.BILLING in actions


@pytest.fixture
def platform(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_PLATFORM_ADMIN_EMAILS", "ops@acme.example")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _client(subject: str, tenant: str, roles: list[str]) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    token = issue_dev_jwt(subject, tenant, roles=roles, attrs={})
    transport = httpx.ASGITransport(app=create_app())
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def test_invoice_endpoints_and_gating(inmemory: None, platform: None) -> None:
    period = _period()
    # Platform admin can pull any tenant's invoice.
    async with _client("ops@acme.example", "platform", ["owner"]) as ops:
        r = await ops.get(f"/v1/platform/tenants/acme/invoice?period={period}")
        assert r.status_code == 200, r.text
        assert r.json()["org_id"] == "acme"
        assert (await ops.get("/v1/platform/tenants/acme/invoice?period=bad")).status_code == 422  # bad period

    # A non-platform-admin cannot use the platform endpoint.
    async with _client("nobody@x.com", "acme", ["owner", "admin"]) as user:
        assert (await user.get(f"/v1/platform/tenants/acme/invoice?period={period}")).status_code == 403
        # ...but may pull its OWN bill.
        own = await user.get(f"/v1/billing/invoice?period={period}")
        assert own.status_code == 200 and own.json()["org_id"] == "acme"
