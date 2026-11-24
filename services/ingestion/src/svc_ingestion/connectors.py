"""Connector seam (FR-A1/A2/A3/A7).

Phase-0 supports UPLOAD (inline bytes) and OBJECT_STORE (by-reference) sources. Each
connector yields normalised :class:`DocumentEnvelope`s. Lakehouse (Iceberg/Delta,
FR-A1), RDBMS/CDC, REST, email, web-crawler, and streaming (FR-A3/A4) are Phase-1
stubs behind this same Protocol so new sources need no core changes (FR-A5).
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit

import httpx

from rag_clients.external_creds import external_credentials
from rag_clients.factory import external_fetcher_for
from rag_common import (
    ConnectorError,
    NotFoundError,
    ValidationError,
    assert_sovereign,
    get_logger,
    get_secrets_provider,
)
from rag_core.interfaces import ExternalObjectFetcher, ObjectStore
from rag_schemas import DocumentEnvelope

_log = get_logger(__name__)


@runtime_checkable
class Connector(Protocol):
    """Resolves an envelope to its raw bytes (pulling from the source if by-reference)."""

    async def fetch(self, doc: DocumentEnvelope) -> bytes: ...


# --- Connector SDK (FR-A5) ---------------------------------------------------
@dataclass(slots=True)
class ConnectorContext:
    """Everything a connector factory may need — grows without breaking factories."""

    object_store: ObjectStore
    external_fetcher: ExternalObjectFetcher | None


ConnectorFactory = Callable[[ConnectorContext], Connector]

_REGISTRY: dict[str, ConnectorFactory] = {}
_REDIRECTS = frozenset({301, 302, 307, 308})


def register_connector(source_kind: str, factory: ConnectorFactory) -> None:
    """Register a connector for a ``SourceKind`` value — the FR-A5 extension point.

    A new source is a new module that calls this at import time; the ingestion service
    and core pipeline need no changes."""
    _REGISTRY[source_kind] = factory
    _log.info("connector_registered", source_kind=source_kind)


def build_connectors(ctx: ConnectorContext) -> dict[str, Connector]:
    """Instantiate every registered connector for one ingestion-service instance."""
    return {kind: factory(ctx) for kind, factory in _REGISTRY.items()}


class UploadConnector:
    """Inline-bytes upload (FR-A7). Bytes travel in ``content_b64``."""

    async def fetch(self, doc: DocumentEnvelope) -> bytes:
        if not doc.content_b64:
            raise ValidationError("upload envelope missing content_b64")
        # A malformed base64 body is a bad request, not a server fault — map it to a clean 422
        # instead of letting binascii.Error surface as an opaque 500 on the ingest API/SDK surface.
        try:
            return base64.b64decode(doc.content_b64)
        except (binascii.Error, ValueError) as exc:
            raise ValidationError("content_b64 is not valid base64") from exc


class ObjectStoreConnector:
    """By-reference pull from S3-compatible object storage (FR-A2, ADR-0012/0013).

    The envelope's ``uri`` is the object key. Two modes, keyed by ``external_source``:

    * **A2a (external_source is None):** pull from the tenant's OWN namespace on the
      platform object store — tenant-isolated by construction (``ObjectStore.get`` scopes
      the key under the tenant prefix), sovereign (platform store is on-allowlist).
    * **A2b (external_source set):** pull from an EXTERNAL, tenant-provided S3 bucket. The
      endpoint is enforced against the sovereignty egress allowlist BEFORE any credential
      is resolved or network touched; credentials arrive by reference (a secret name) and
      are resolved via the ``SecretsProvider`` — never present in the envelope or logs.
    """

    def __init__(self, store: ObjectStore, external: ExternalObjectFetcher | None = None) -> None:
        self._store = store
        self._external = external

    async def fetch(self, doc: DocumentEnvelope) -> bytes:
        if not doc.uri:
            raise ValueError("object_store envelope missing uri (object key)")
        if doc.external_source is None:
            return await self._fetch_platform(doc)
        return await self._fetch_external(doc)

    async def _fetch_platform(self, doc: DocumentEnvelope) -> bytes:
        try:
            return await self._store.get(doc.tenant_id, doc.uri or "")
        except KeyError as exc:
            raise NotFoundError(
                "referenced object not found in tenant namespace",
                details={"tenant_id": doc.tenant_id, "object_key": doc.uri},
            ) from exc

    async def _fetch_external(self, doc: DocumentEnvelope) -> bytes:
        src = doc.external_source
        assert src is not None  # guarded by caller
        # Sovereignty FIRST: refuse a non-allowlisted endpoint before resolving any secret
        # or opening any connection. The guard is never loosened to make a pull succeed.
        assert_sovereign(f"{'https' if src.secure else 'http'}://{src.endpoint}")
        provider = getattr(src, "provider", "s3")
        # The injected fetcher is the S3 one; any other provider resolves its own from the factory.
        fetcher = self._external if provider == "s3" else external_fetcher_for(provider)
        if fetcher is None:
            raise ValueError("external object fetcher is not configured")
        access_key, secret_key = external_credentials(src, get_secrets_provider())
        try:
            return await fetcher.fetch(
                endpoint=src.endpoint,
                bucket=src.bucket,
                key=doc.uri or "",
                access_key=access_key,
                secret_key=secret_key,
                secure=src.secure,
            )
        except KeyError as exc:
            raise NotFoundError(
                "referenced object not found in external bucket",
                details={"endpoint": src.endpoint, "bucket": src.bucket, "object_key": doc.uri},
            ) from exc


# A descriptive User-Agent — many sites (Wikipedia, docs portals, WAFs) reject requests with
# no UA (HTTP 403). Identifies the platform and points to a contact, per good-crawler etiquette.
WEB_USER_AGENT = "Mozilla/5.0 (compatible; RagService/1.0; +https://acme.example)"
_WEB_HEADERS = {
    "User-Agent": WEB_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en,hi;q=0.8",
}


class WebConnector:
    """Fetch a public web page / file by URL (FR-A3 web source).

    Sovereignty: the URL's host must be on the egress allowlist — the platform never
    crawls an arbitrary internet host; enabling a site is an operator action, exactly
    like an external bucket. Redirect hops are re-checked against the allowlist."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    async def fetch(self, doc: DocumentEnvelope) -> bytes:
        if not doc.uri:
            raise ValueError("web envelope missing uri (the page URL)")
        assert_sovereign(doc.uri)  # SovereigntyViolationError (403) stays distinct from a reach failure
        host = urlsplit(doc.uri).hostname or doc.uri
        try:
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=False, headers=_WEB_HEADERS) as client:
                url = doc.uri
                for _ in range(3):  # bounded redirects, each hop re-checked against the allowlist
                    resp = await client.get(url)
                    if resp.status_code in _REDIRECTS and resp.headers.get("location"):
                        url = str(resp.next_request.url) if resp.next_request else resp.headers["location"]
                        assert_sovereign(url)
                        continue
                    break
                if resp.status_code == httpx.codes.NOT_FOUND:
                    raise NotFoundError("page not found", details={"url": doc.uri, "status": resp.status_code})
                resp.raise_for_status()
                _log.info("web_fetched", url=doc.uri, status=resp.status_code, bytes=len(resp.content))
                return resp.content
        except httpx.HTTPStatusError as exc:
            raise ConnectorError(
                f"{host} returned HTTP {exc.response.status_code} when fetching the page.",
                details={"url": doc.uri, "status": exc.response.status_code},
            ) from exc
        except httpx.HTTPError as exc:
            # ConnectTimeout / ConnectError / ReadTimeout / … — host allowlisted but unreachable.
            raise ConnectorError(
                f"Couldn't reach {host} ({type(exc).__name__}). Check the host is reachable "
                "from the server (network/firewall) — it's on the allowlist but did not respond.",
                details={"url": doc.uri, "reason": type(exc).__name__},
            ) from exc


# --- Built-in connectors (registered via the same SDK seam new sources use) ---
register_connector("upload", lambda ctx: UploadConnector())
register_connector("object_store", lambda ctx: ObjectStoreConnector(ctx.object_store, ctx.external_fetcher))
register_connector("web", lambda ctx: WebConnector())

# TODO(Phase-1/2): LakehouseConnector (FR-A1), RDBMS/CDC, REST, email (FR-A3),
# streaming (FR-A4), Drive/SharePoint (OAuth apps required).
