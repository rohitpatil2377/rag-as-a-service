"""WebConnector network-failure handling — a reachable-but-dead host is a clean 502, not a 500."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from svc_ingestion.connectors import WebConnector

from rag_common.config import get_settings
from rag_common.errors import ConnectorError, SovereigntyViolationError
from rag_schemas import DocumentEnvelope
from rag_schemas.common import SourceKind


class _TimeoutClient:
    def __init__(self, *a: object, **k: object) -> None: ...
    async def __aenter__(self) -> _TimeoutClient:
        return self

    async def __aexit__(self, *a: object) -> bool:
        return False

    async def get(self, url: str) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")


def _doc(uri: str) -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id="t",
        doc_id="d",
        source_id="web",
        source_kind=SourceKind.WEB,
        mime="text/html",
        uri=uri,
        ingested_at=datetime.now(UTC),
    )


async def test_unreachable_host_maps_to_connector_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "unreachable.local")
    get_settings.cache_clear()
    monkeypatch.setattr("svc_ingestion.connectors.httpx.AsyncClient", _TimeoutClient)
    with pytest.raises(ConnectorError, match="Couldn't reach"):
        await WebConnector().fetch(_doc("https://unreachable.local/page"))
    get_settings.cache_clear()


async def test_non_allowlisted_host_still_403(monkeypatch: pytest.MonkeyPatch) -> None:
    # The sovereignty block must stay a distinct 403, never masked as a connector error.
    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost")
    get_settings.cache_clear()
    with pytest.raises(SovereigntyViolationError):
        await WebConnector().fetch(_doc("https://not-approved.example.com/page"))
    get_settings.cache_clear()
