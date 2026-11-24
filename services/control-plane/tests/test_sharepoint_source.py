"""Unit tests for the SharePoint connector (unified KB, docs + connected structured sources).

Mirrors ``test_unified_kb.py``'s SQL-connector coverage: not-configured -> None, a real search
response -> a span, fail-safe on non-200/malformed/no-hits, set/get/clear round-trip, the token
never leaking back out of ``get_sharepoint_connection``, and the sovereignty guard enforced on
registration. All HTTP is faked via ``httpx.MockTransport`` injected through the module-level
``_client_factory`` seam — no real network call is ever made.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from svc_control_plane import sharepoint_source
from svc_control_plane.sharepoint_source import (
    SharePointConnectionInfo,
    SharePointHit,
    render_sharepoint_evidence,
    sharepoint_evidence_span,
)

from rag_common.errors import SovereigntyViolationError
from rag_schemas.common import Modality

_KB = "acme__eng"
_SITE = "https://localhost/sites/eng"  # "localhost" is in the default egress allowlist


def _mock_client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _search_json(*, wrap_results: bool = False) -> dict[str, Any]:
    """A realistic SharePoint Search REST API (odata=nometadata) payload with two hits."""
    rows = [
        {
            "Cells": [
                {"Key": "Title", "Value": "Onboarding Guide"},
                {"Key": "Path", "Value": "https://localhost/sites/eng/docs/onboarding.docx"},
                {"Key": "HitHighlightedSummary", "Value": "Welcome to the <c0>team</c0>, start here<ddd/>"},
            ]
        },
        {
            "Cells": [
                {"Key": "Title", "Value": "VPN Setup"},
                {"Key": "Path", "Value": "https://localhost/sites/eng/docs/vpn.docx"},
                {"Key": "HitHighlightedSummary", "Value": "Install the <c0>VPN</c0> client"},
            ]
        },
    ]
    table_rows: Any = {"results": rows} if wrap_results else rows
    return {"PrimaryQueryResult": {"RelevantResults": {"Table": {"Rows": table_rows}}}}


# --- render_sharepoint_evidence --------------------------------------------------------------


def test_render_sharepoint_evidence_bounds_and_formats() -> None:
    hits = [SharePointHit(title=f"Doc {i}", snippet="s", path="p") for i in range(15)]
    text = render_sharepoint_evidence(hits, max_results=10)
    assert "15 hit(s)" in text
    assert "Doc 0" in text and "Doc 9" in text
    assert "Doc 10" not in text  # truncated beyond max_results
    assert "... (5 more hit(s))" in text


# --- not configured -------------------------------------------------------------------------


async def test_not_configured_returns_none(inmemory: None) -> None:
    assert await sharepoint_evidence_span(_KB, "how do I connect the VPN", actor="u") is None


# --- set / get / clear round-trip, token never leaked ----------------------------------------


async def test_set_get_clear_roundtrip(inmemory: None) -> None:
    info = await sharepoint_source.set_sharepoint_connection(_KB, _SITE, token="secret-tok-123")
    assert info == SharePointConnectionInfo(configured=True, site_url=_SITE, has_token=True)

    fetched = await sharepoint_source.get_sharepoint_connection(_KB)
    assert fetched == info
    assert "token" not in fetched.model_dump()  # the model has no field that could carry it
    assert "secret-tok-123" not in str(fetched.model_dump())  # belt-and-braces

    # the raw stored ciphertext is not the plaintext token
    from rag_clients.factory import get_catalog

    cfg = await get_catalog().get_tenant_config(_KB)
    assert cfg["sharepoint_token"] != "secret-tok-123"

    await sharepoint_source.clear_sharepoint_connection(_KB)
    cleared = await sharepoint_source.get_sharepoint_connection(_KB)
    assert cleared == SharePointConnectionInfo(configured=False)


async def test_set_without_token_leaves_has_token_false(inmemory: None) -> None:
    info = await sharepoint_source.set_sharepoint_connection(_KB, _SITE)
    assert info.configured is True
    assert info.has_token is False
    assert info.site_url == _SITE


# --- sovereignty guard ------------------------------------------------------------------------


async def test_assert_sovereign_enforced_on_registration(inmemory: None) -> None:
    with pytest.raises(SovereigntyViolationError):
        await sharepoint_source.set_sharepoint_connection(_KB, "https://evil.example.com/sites/x", token="t")
    # rejected registration must not have persisted a connection
    assert (await sharepoint_source.get_sharepoint_connection(_KB)).configured is False


# --- sharepoint_evidence_span: happy path ------------------------------------------------------


async def test_valid_search_response_returns_span(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        assert request.url.path.endswith("/_api/search/query")
        return httpx.Response(200, json=_search_json())

    await sharepoint_source.set_sharepoint_connection(_KB, _SITE, token="secret-tok-123")
    monkeypatch.setattr(sharepoint_source, "_client_factory", lambda: _mock_client(handler))

    span = await sharepoint_evidence_span(_KB, "how do I set up the vpn", actor="u")

    assert span is not None
    assert span.scorer == "exact"
    assert span.score == 1.0
    assert span.modality is Modality.SHAREPOINT
    assert span.chunk_id == f"sharepoint::{_KB}"
    assert span.doc_id == "sharepoint://localhost"
    assert span.source_uri == _SITE
    assert span.category == "sharepoint"
    assert "Onboarding Guide" in span.text
    assert "VPN Setup" in span.text
    assert "<c0>" not in span.text  # highlight markup stripped

    # bearer token forwarded, question URL-encoded, results bounded at the query level
    assert seen["headers"]["authorization"] == "Bearer secret-tok-123"
    assert seen["headers"]["accept"] == "application/json;odata=nometadata"
    assert "vpn" in seen["url"]
    assert "rowlimit=10" in seen["url"]


async def test_valid_search_response_handles_results_wrapper(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """SharePoint sometimes nests collections under a ``results`` key even w/ nometadata."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_search_json(wrap_results=True))

    await sharepoint_source.set_sharepoint_connection(_KB, _SITE, token="t")
    monkeypatch.setattr(sharepoint_source, "_client_factory", lambda: _mock_client(handler))

    span = await sharepoint_evidence_span(_KB, "q", actor="u")
    assert span is not None
    assert "Onboarding Guide" in span.text


async def test_no_token_omits_authorization_header(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json=_search_json())

    await sharepoint_source.set_sharepoint_connection(_KB, _SITE)  # no token
    monkeypatch.setattr(sharepoint_source, "_client_factory", lambda: _mock_client(handler))

    span = await sharepoint_evidence_span(_KB, "q", actor="u")
    assert span is not None
    assert "authorization" not in seen["headers"]


# --- sharepoint_evidence_span: fail-safe (returns None, never raises) -------------------------


async def test_non_200_returns_none(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    await sharepoint_source.set_sharepoint_connection(_KB, _SITE, token="t")
    monkeypatch.setattr(sharepoint_source, "_client_factory", lambda: _mock_client(handler))
    assert await sharepoint_evidence_span(_KB, "q", actor="u") is None


async def test_malformed_json_returns_none(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all", headers={"content-type": "text/plain"})

    await sharepoint_source.set_sharepoint_connection(_KB, _SITE, token="t")
    monkeypatch.setattr(sharepoint_source, "_client_factory", lambda: _mock_client(handler))
    assert await sharepoint_evidence_span(_KB, "q", actor="u") is None


async def test_unrecognised_shape_returns_none(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    await sharepoint_source.set_sharepoint_connection(_KB, _SITE, token="t")
    monkeypatch.setattr(sharepoint_source, "_client_factory", lambda: _mock_client(handler))
    assert await sharepoint_evidence_span(_KB, "q", actor="u") is None


async def test_no_hits_returns_none(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"PrimaryQueryResult": {"RelevantResults": {"Table": {"Rows": []}}}})

    await sharepoint_source.set_sharepoint_connection(_KB, _SITE, token="t")
    monkeypatch.setattr(sharepoint_source, "_client_factory", lambda: _mock_client(handler))
    assert await sharepoint_evidence_span(_KB, "q", actor="u") is None


async def test_transport_raises_returns_none(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    await sharepoint_source.set_sharepoint_connection(_KB, _SITE, token="t")
    monkeypatch.setattr(sharepoint_source, "_client_factory", lambda: _mock_client(handler))
    assert await sharepoint_evidence_span(_KB, "q", actor="u") is None


async def test_span_none_when_site_no_longer_sovereign(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A KB registered while the host was allowlisted, then the allowlist tightens: the span
    path must still fail safe rather than call out to a now-disallowed host."""
    await sharepoint_source.set_sharepoint_connection(_KB, _SITE, token="t")

    async def _stale_info(_kb: str) -> SharePointConnectionInfo:
        return SharePointConnectionInfo(configured=True, site_url="https://evil.example.com/sites/x", has_token=True)

    monkeypatch.setattr(sharepoint_source, "get_sharepoint_connection", _stale_info)
    assert await sharepoint_evidence_span(_KB, "q", actor="u") is None
