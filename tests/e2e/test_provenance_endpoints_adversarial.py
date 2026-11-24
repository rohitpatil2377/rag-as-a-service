"""ADVERSARIAL e2e for the provenance API surface — receipt verify/get, SBOM, residency.

Run: uv run pytest tests/e2e/test_provenance_endpoints_adversarial.py -q -m e2e
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest

pytestmark = pytest.mark.e2e


async def _client(tenant: str, *, subject: str = "u", roles: list[str] | None = None) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt(subject, tenant, roles=roles or ["reader", "ingestor", "admin"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def _ingest(c: httpx.AsyncClient, tenant: str, doc_id: str, text: str) -> None:
    body = base64.b64encode(text.encode()).decode()
    await c.post(f"/v1/kb/{tenant}/ingest", json={"doc_id": doc_id, "mime": "text/plain", "content_b64": body})


def _m(name: str, role: str, wh: str = "a", lic: str = "MIT") -> dict[str, str]:
    return {"name": name, "role": role, "source_origin": "India", "license": lic, "weight_sha256": wh * 64}


# ---------------------------------------------------------------------------
# 1. /receipts/verify round-trips a genuine receipt and rejects a tampered one.
# ---------------------------------------------------------------------------


async def test_verify_endpoint_true_for_genuine_false_for_tampered(inmemory: None) -> None:
    from svc_control_plane import provenance

    from rag_common.trust_receipt import CitedSpan, Contributor, build_receipt, sign_receipt

    receipt = build_receipt(
        tenant_id="acme",
        answer_id="a1",
        query="q",
        answer_text="a",
        model_id="m",
        model_weight_sha256="a" * 64,
        contributors=[Contributor(dept_id="acme")],
        cited_spans=[CitedSpan(text="t", chunk_id="c1")],
        prev_audit_hash=None,
        issued_at=__import__("datetime").datetime(2026, 8, 5, tzinfo=__import__("datetime").UTC),
        signer_kid=provenance.signer_kid(),
        receipt_id="r1",
    )
    signed = sign_receipt(receipt, provenance.signing_key_b64())

    c = await _client("acme")
    async with c:
        good = await c.post("/v1/receipts/verify", json=json.loads(signed.model_dump_json()))
        assert good.status_code == 200, good.text
        assert good.json() == {"valid": True}

        tampered = signed.model_copy(update={"answer_sha256": "0" * 64})
        bad = await c.post("/v1/receipts/verify", json=json.loads(tampered.model_dump_json()))
        assert bad.status_code == 200
        assert bad.json() == {"valid": False}


async def test_verify_endpoint_rejects_receipt_tampered_in_each_field(inmemory: None) -> None:
    from svc_control_plane import provenance

    from rag_common.trust_receipt import CitedSpan, Contributor, build_receipt, sign_receipt

    receipt = build_receipt(
        tenant_id="acme",
        answer_id="a1",
        query="q",
        answer_text="a",
        model_id="m",
        model_weight_sha256="a" * 64,
        contributors=[Contributor(dept_id="acme")],
        cited_spans=[CitedSpan(text="t", chunk_id="c1")],
        prev_audit_hash=None,
        issued_at=__import__("datetime").datetime(2026, 8, 5, tzinfo=__import__("datetime").UTC),
        signer_kid=provenance.signer_kid(),
        receipt_id="r1",
    )
    signed = sign_receipt(receipt, provenance.signing_key_b64())
    c = await _client("acme")
    async with c:
        for field, val in [("model_id", "evil"), ("tenant_id", "attacker"), ("span_digests", [])]:
            t = signed.model_copy(update={field: val})
            r = await c.post("/v1/receipts/verify", json=json.loads(t.model_dump_json()))
            assert r.json() == {"valid": False}, field


# ---------------------------------------------------------------------------
# 2. Full loop: ask → receipt emitted → GET /receipts/{id} → verify offline.
#    And ABSTAINED answers produce NO receipt (404).
# ---------------------------------------------------------------------------


# FIXED: get_receipt now searches the caller's org namespace AND every KB they own, so the
# owner can pull a receipt emitted under the kb_id namespace (was a 404-for-owner bug).
async def test_grounded_answer_emits_pullable_verifiable_receipt(inmemory: None, monkeypatch) -> None:
    monkeypatch.setenv("RAG_TRUST_RECEIPTS_ENABLED", "true")
    from rag_common import get_settings

    get_settings.cache_clear()
    from svc_control_plane import provenance

    tenant = "acme"
    c = await _client(tenant)
    async with c:
        kb = (await c.post("/v1/knowledge-bases", json={"display_name": "KB"})).json()
        kb_id = kb["kb_id"]
        await c.put(f"/v1/kb/{kb_id}/settings", json={"faithful_threshold": 0.01})
        await _ingest(c, kb_id, "d1", "The uptime target is 99.5 percent for all tiers.")
        ans = (await c.post(f"/v1/kb/{kb_id}/query", json={"query": "uptime target percent"})).json()
        trace = ans.get("trace_id")
        assert trace, ans
        if ans["status"] not in ("grounded", "flagged_for_review"):
            pytest.skip(f"answer not grounded ({ans['status']}); receipt only for grounded/flagged")

        # The receipt is audited under the ANSWER's tenant_id (== kb_id), but GET /receipts/{id}
        # lists under principal.tenant_id (== the JWT tenant 'acme'). Probe BOTH to catch a
        # scope mismatch.
        got = await c.get(f"/v1/receipts/{trace}")
        assert got.status_code == 200, (
            f"legitimate owner got {got.status_code} pulling own receipt — "
            f"scope mismatch (kb_id={kb_id} vs principal.tenant_id={tenant}); body={got.text}"
        )
        from rag_common.trust_receipt import TrustReceipt, sha256_hex

        receipt = TrustReceipt(**got.json())
        assert provenance.verify_supplied_receipt(receipt) is True
        assert receipt.answer_sha256 == sha256_hex(ans["answer"])


async def test_get_receipt_404_for_unknown_answer(inmemory: None) -> None:
    c = await _client("acme")
    async with c:
        r = await c.get("/v1/receipts/no-such-trace-id")
        assert r.status_code == 404


async def test_abstained_answer_has_no_receipt(inmemory: None, monkeypatch) -> None:
    monkeypatch.setenv("RAG_TRUST_RECEIPTS_ENABLED", "true")
    from rag_common import get_settings

    get_settings.cache_clear()
    tenant = "acme"
    c = await _client(tenant)
    async with c:
        kb = (await c.post("/v1/knowledge-bases", json={"display_name": "KB"})).json()
        kb_id = kb["kb_id"]
        # No documents ingested → the platform must abstain.
        ans = (await c.post(f"/v1/kb/{kb_id}/query", json={"query": "What is the airspeed of a swallow?"})).json()
        assert ans["status"] == "abstained", ans
        trace = ans.get("trace_id")
        if trace:
            r = await c.get(f"/v1/receipts/{trace}")
            assert r.status_code == 404


# ---------------------------------------------------------------------------
# 3. CROSS-TENANT: tenant B cannot pull tenant A's receipt (scoped by principal).
# ---------------------------------------------------------------------------


async def test_receipt_get_is_tenant_scoped(inmemory: None, monkeypatch) -> None:
    monkeypatch.setenv("RAG_TRUST_RECEIPTS_ENABLED", "true")
    from rag_common import get_settings

    get_settings.cache_clear()
    # Tenant A produces a grounded answer. Use kb_id == tenant so the receipt is reliably
    # stored AND retrievable by A (avoids the separate namespace-mismatch bug) — this isolates
    # the isolation check from that bug.
    kb_id = "acme"
    ca = await _client(kb_id)
    async with ca:
        await ca.put(f"/v1/kb/{kb_id}/settings", json={"faithful_threshold": 0.01})
        await _ingest(ca, kb_id, "d1", "Acme secret uptime is 99.999 percent internal only.")
        ans = (await ca.post(f"/v1/kb/{kb_id}/query", json={"query": "uptime percent internal"})).json()
        trace = ans.get("trace_id")
        if ans["status"] not in ("grounded", "flagged_for_review") or not trace:
            pytest.skip("answer not grounded; cannot test receipt isolation")
        # sanity: A CAN pull its own receipt here
        assert (await ca.get(f"/v1/receipts/{trace}")).status_code == 200

    # Tenant B, presenting acme's trace id, must NOT get acme's receipt.
    cb = await _client("evilcorp")
    async with cb:
        r = await cb.get(f"/v1/receipts/{trace}")
        assert r.status_code == 404, "tenant B pulled tenant A's receipt — ISOLATION LEAK"


async def test_receipt_retrievable_when_jwt_tenant_equals_kb_id(inmemory: None, monkeypatch) -> None:
    """Proves the namespace theory behind the round-trip bug: if the caller's JWT tenant is the
    SAME string as the kb_id used for the query, GET /receipts/{id} DOES find the receipt —
    because the audit list scope (principal.tenant_id) then matches the store scope (answer.tenant_id).
    This is the workaround/confirmation, not the intended API shape."""
    monkeypatch.setenv("RAG_TRUST_RECEIPTS_ENABLED", "true")
    from rag_common import get_settings

    get_settings.cache_clear()
    from svc_control_plane import provenance

    from rag_common.trust_receipt import TrustReceipt

    # Use a bare tenant id as the kb path (tenant_guard allows kb_id==tenant_id).
    kb_id = "acme"
    c = await _client(kb_id)
    async with c:
        await c.put(f"/v1/kb/{kb_id}/settings", json={"faithful_threshold": 0.01})
        await _ingest(c, kb_id, "d1", "The uptime target is 99.5 percent for all tiers.")
        ans = (await c.post(f"/v1/kb/{kb_id}/query", json={"query": "uptime target percent"})).json()
        if ans.get("status") not in ("grounded", "flagged_for_review"):
            pytest.skip("answer not grounded")
        got = await c.get(f"/v1/receipts/{ans['trace_id']}")
        assert got.status_code == 200, got.text
        assert provenance.verify_supplied_receipt(TrustReceipt(**got.json())) is True


# ---------------------------------------------------------------------------
# 4. /v1/sbom — signed, verifiable; 404 when unconfigured; any auth caller can read.
# ---------------------------------------------------------------------------


async def test_sbom_404_when_unconfigured(inmemory: None, monkeypatch) -> None:
    monkeypatch.setenv("RAG_MODEL_SBOM_MODELS", "")
    from rag_common import get_settings

    get_settings.cache_clear()
    c = await _client("acme")
    async with c:
        r = await c.get("/v1/sbom")
        assert r.status_code == 404


async def test_sbom_signed_and_drops_invalid_entries(inmemory: None, monkeypatch) -> None:
    monkeypatch.setenv(
        "RAG_MODEL_SBOM_MODELS",
        json.dumps([_m("bge-m3", "embed"), _m("junk", "translate", "b", "X")]),
    )
    from rag_common import get_settings

    get_settings.cache_clear()
    from svc_control_plane import provenance

    from rag_common.model_sbom import ModelSBOM, verify_sbom

    c = await _client("acme")
    async with c:
        r = await c.get("/v1/sbom")
        assert r.status_code == 200, r.text
        sbom = ModelSBOM(**r.json())
        assert [m.name for m in sbom.models] == ["bge-m3"], "invalid role entry not dropped"
        assert verify_sbom(sbom, provenance.public_key_b64()) is True
        # tamper origin -> fails
        bad = sbom.model_copy(update={"models": [sbom.models[0].model_copy(update={"source_origin": "China"})]})
        assert verify_sbom(bad, provenance.public_key_b64()) is False


async def test_sbom_requires_auth(inmemory: None, monkeypatch) -> None:
    monkeypatch.setenv("RAG_MODEL_SBOM_MODELS", json.dumps([_m("m", "embed")]))
    from rag_common import get_settings

    get_settings.cache_clear()
    from rag_api.app import create_app

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as anon:
        r = await anon.get("/v1/sbom")  # no Authorization header
        assert r.status_code in (401, 403), r.status_code


# ---------------------------------------------------------------------------
# 5. /platform/residency/node — platform-admin gated (403 when allowlist empty).
# ---------------------------------------------------------------------------


async def test_residency_node_forbidden_without_platform_admin(inmemory: None, monkeypatch) -> None:
    monkeypatch.setenv("RAG_PLATFORM_ADMIN_EMAILS", "")  # empty allowlist on VM parity
    from rag_common import get_settings

    get_settings.cache_clear()
    c = await _client("acme", subject="nobody@x.com")
    async with c:
        r = await c.get("/v1/platform/residency/node")
        assert r.status_code == 403, r.text


async def test_residency_node_signed_for_platform_admin(inmemory: None, monkeypatch) -> None:
    monkeypatch.setenv("RAG_PLATFORM_ADMIN_EMAILS", "boss@acme.example")
    monkeypatch.setenv("RAG_RESIDENCY_NODE_ID", "node-blr-1")
    monkeypatch.setenv("RAG_RESIDENCY_REGION", "India")
    from rag_common import get_settings

    get_settings.cache_clear()
    from svc_control_plane import provenance
    from svc_control_plane.compliance import NodeResidencyAttestation, verify_node_attestation

    c = await _client("acme", subject="boss@acme.example")
    async with c:
        r = await c.get("/v1/platform/residency/node")
        assert r.status_code == 200, r.text
        att = NodeResidencyAttestation(**r.json())
        assert att.no_foreign_egress is True
        assert att.region == "India"
        assert verify_node_attestation(att, provenance.public_key_b64()) is True
        # tamper region -> fails offline verification
        bad = att.model_copy(update={"region": "Singapore"})
        assert verify_node_attestation(bad, provenance.public_key_b64()) is False
