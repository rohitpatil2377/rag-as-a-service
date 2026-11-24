"""Azure Blob Storage as an EXTERNAL document source (implements ExternalObjectFetcher).

Same seam and same contract as :class:`MinioExternalFetcher`: the caller has already enforced
sovereignty on the endpoint, credentials arrive per call, and a missing object raises ``KeyError``.
Under this seam ``endpoint`` is the account host (``<account>.blob.core.windows.net``), ``bucket``
is the CONTAINER, ``access_key`` the account name and ``secret_key`` an account key or SAS token
(see :func:`rag_clients.external_creds.external_credentials`).

The Azure SDK is synchronous; calls run in a worker thread so a slow blob download cannot stall
the event loop for every other request.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from rag_common import ConfigError, assert_sovereign, get_logger

_log = get_logger(__name__)


def _default_client(account_url: str, credential: str) -> Any:
    try:
        from azure.storage.blob import BlobServiceClient  # noqa: PLC0415 -- optional dep, resolved on use
    except ImportError as exc:  # the extra is not installed on this deployment
        raise ConfigError(
            "Azure Blob support is not installed on this deployment — install the 'azure' extra "
            "(uv sync --extra azure) to pull documents from an Azure container.",
            details={"package": "azure-storage-blob"},
        ) from exc

    return BlobServiceClient(account_url=account_url, credential=credential)


class AzureBlobExternalFetcher:
    """``client_factory`` is injectable so the fetcher is testable without an Azure account."""

    def __init__(self, client_factory: Callable[[str, str], Any] | None = None) -> None:
        self._factory = client_factory or _default_client

    @staticmethod
    def _account_url(endpoint: str, secure: bool) -> str:
        return f"{'https' if secure else 'http'}://{endpoint}"

    def _credential(self, access_key: str, secret_key: str) -> Any:
        # An account key needs the account name alongside it; a SAS token is used as-is. The SDK
        # accepts a dict for the former and a string for the latter.
        if secret_key.startswith("?") or "sig=" in secret_key:
            return secret_key
        return {"account_name": access_key, "account_key": secret_key}

    async def fetch(
        self, *, endpoint: str, bucket: str, key: str, access_key: str, secret_key: str, secure: bool
    ) -> bytes:
        url = self._account_url(endpoint, secure)
        assert_sovereign(url)  # belt and braces: the connector checks too, this fetcher never trusts its caller

        def _run() -> bytes:
            client = self._factory(url, self._credential(access_key, secret_key))
            blob = client.get_blob_client(container=bucket, blob=key)
            try:
                return bytes(blob.download_blob().readall())
            except Exception as exc:
                if type(exc).__name__ == "ResourceNotFoundError":
                    raise KeyError(key) from exc
                raise

        data = await asyncio.to_thread(_run)
        _log.info("azure_blob_fetched", endpoint=endpoint, container=bucket, key=key, size=len(data))
        return data

    async def list_keys(
        self,
        *,
        endpoint: str,
        bucket: str,
        prefix: str,
        access_key: str,
        secret_key: str,
        secure: bool,
        limit: int = 200,
        start_after: str = "",
    ) -> list[str]:
        url = self._account_url(endpoint, secure)
        assert_sovereign(url)

        def _run() -> list[str]:
            client = self._factory(url, self._credential(access_key, secret_key))
            container = client.get_container_client(bucket)
            out: list[str] = []
            for blob in container.list_blobs(name_starts_with=prefix or None):
                name = getattr(blob, "name", None) or (blob.get("name") if isinstance(blob, dict) else None)
                # Azure lists lexicographically but has no start_after, so the already-seen span is
                # skipped here. Cheap next to one extract+embed per object, and it keeps the
                # continuation contract identical across providers.
                if name and not name.endswith("/") and (not start_after or str(name) > start_after):
                    out.append(str(name))
                if len(out) >= limit:
                    break
            return out

        keys = await asyncio.to_thread(_run)
        _log.info("azure_blob_listed", endpoint=endpoint, container=bucket, prefix=prefix, keys=len(keys))
        return keys
