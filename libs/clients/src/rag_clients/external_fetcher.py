"""External object-store fetcher (implements :class:`ExternalObjectFetcher`) — FR-A2b.

Pulls a single object from an EXTERNAL, tenant-provided S3-compatible bucket, with the
endpoint + credentials supplied per call (not from platform config). Sovereignty is
enforced by the connector via ``assert_sovereign`` BEFORE this runs; this adapter also
re-asserts the endpoint as defence-in-depth, so the guard holds even if a future caller
forgets. Credentials arrive already resolved from the ``SecretsProvider`` — this layer
never reads env/secret stores itself and never logs the credential values.
"""

from __future__ import annotations

import asyncio

from minio import Minio
from minio.error import S3Error

from rag_common import assert_sovereign, get_logger

_log = get_logger(__name__)


class MinioExternalFetcher:
    """Fetch one object from an arbitrary (allowlisted) S3-compatible endpoint."""

    async def fetch(
        self, *, endpoint: str, bucket: str, key: str, access_key: str, secret_key: str, secure: bool
    ) -> bytes:
        # Defence-in-depth: refuse a non-allowlisted endpoint even if the caller didn't check.
        assert_sovereign(f"{'https' if secure else 'http'}://{endpoint}")
        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

        def _get() -> bytes:
            try:
                resp = client.get_object(bucket, key)
            except S3Error as exc:
                if exc.code in {"NoSuchKey", "NoSuchBucket"}:
                    raise KeyError(f"{bucket}/{key}") from exc
                raise
            try:
                return resp.read()
            finally:
                resp.close()
                resp.release_conn()

        data = await asyncio.to_thread(_get)
        _log.info("external_object_fetched", endpoint=endpoint, bucket=bucket, key=key, size=len(data))
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
        assert_sovereign(f"{'https' if secure else 'http'}://{endpoint}")
        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

        def _list() -> list[str]:
            keys: list[str] = []
            # S3 lists lexicographically and takes start_after natively, so the server skips the
            # already-seen prefix rather than us paying to stream and discard it.
            for obj in client.list_objects(bucket, prefix=prefix, recursive=True, start_after=start_after or None):
                if obj.object_name and not obj.is_dir:
                    keys.append(obj.object_name)
                    if len(keys) >= limit:
                        break
            return keys

        keys = await asyncio.to_thread(_list)
        _log.info("external_keys_listed", endpoint=endpoint, bucket=bucket, prefix=prefix, count=len(keys))
        return keys
