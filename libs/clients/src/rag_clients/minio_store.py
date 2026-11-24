"""MinIO object store adapter (implements :class:`ObjectStore`) — FR-A2/G4.

Isolation: keys are prefixed per tenant via
:func:`rag_common.tenancy.object_prefix` within a shared bucket. (Phase-1 upgrades
to bucket-per-tenant + Object-Lock/WORM for the audit tier, TDD §6/§9.)

The MinIO SDK is synchronous; calls are offloaded to a thread so the async services
are not blocked. Sovereign: endpoint validated against the egress allowlist.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
from typing import IO, BinaryIO, cast

from minio import Minio
from minio.error import S3Error

from rag_common import assert_sovereign, get_logger, get_settings, object_prefix

_log = get_logger(__name__)
_DEFAULT_BUCKET = "rag"


class MinioObjectStore:
    """S3-compatible blob store for raw docs, page images, and audit artifacts."""

    def __init__(self, bucket: str = _DEFAULT_BUCKET) -> None:
        settings = get_settings()
        assert_sovereign(f"http://{settings.minio_endpoint}")
        self._bucket = bucket
        self._client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )

    async def _ensure_bucket(self) -> None:
        def _mk() -> None:
            if not self._client.bucket_exists(self._bucket):
                self._client.make_bucket(self._bucket)

        await asyncio.to_thread(_mk)

    async def put(self, tenant_id: str, key: str, data: bytes, content_type: str) -> str:
        await self._ensure_bucket()
        full_key = f"{object_prefix(tenant_id)}{key}"

        def _put() -> None:
            self._client.put_object(
                self._bucket, full_key, io.BytesIO(data), length=len(data), content_type=content_type
            )

        await asyncio.to_thread(_put)
        _log.info("object_put", bucket=self._bucket, key=full_key, size=len(data))
        return f"s3://{self._bucket}/{full_key}"

    async def put_stream(self, tenant_id: str, key: str, stream: IO[bytes], length: int, content_type: str) -> str:
        await self._ensure_bucket()
        full_key = f"{object_prefix(tenant_id)}{key}"

        def _put() -> None:
            # put_object streams from the file-like object in parts (MinIO chunks it), so a
            # multi-GB upload never has to be held whole in memory.
            self._client.put_object(
                self._bucket, full_key, cast(BinaryIO, stream), length=length, content_type=content_type
            )

        await asyncio.to_thread(_put)
        _log.info("object_put_stream", bucket=self._bucket, key=full_key, size=length)
        return f"s3://{self._bucket}/{full_key}"

    async def get(self, tenant_id: str, key: str) -> bytes:
        full_key = f"{object_prefix(tenant_id)}{key}"

        def _get() -> bytes:
            try:
                resp = self._client.get_object(self._bucket, full_key)
            except S3Error as exc:
                # Normalise "not found" (missing object OR missing bucket, e.g. a tenant
                # that never ingested anything) to KeyError — matching the in-memory
                # adapter's contract so callers catch one exception type regardless of
                # backend, per ObjectStore.get.
                if exc.code in {"NoSuchKey", "NoSuchBucket"}:
                    raise KeyError(full_key) from exc
                raise
            try:
                return resp.read()
            finally:
                resp.close()
                resp.release_conn()

        return await asyncio.to_thread(_get)

    async def delete(self, tenant_id: str, key: str) -> None:
        full_key = f"{object_prefix(tenant_id)}{key}"

        def _rm() -> None:
            # already gone / bucket missing — delete is idempotent.
            with contextlib.suppress(S3Error):
                self._client.remove_object(self._bucket, full_key)

        await asyncio.to_thread(_rm)
