"""DocumentEnvelope — the normalised internal document contract (TDD §4.1).

Every connector emits documents in this shape regardless of source, so downstream
layers (doc-intelligence, indexing) are source-agnostic. Source ACLs are captured
here at ingestion and later stamped onto chunks (permission-aware retrieval seam,
FR-A6/D8).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from rag_schemas.common import SourceKind


class ExternalObjectSource(BaseModel):
    """Pointer to an object in an EXTERNAL, tenant-provided S3-compatible bucket (FR-A2b).

    Credentials are supplied by REFERENCE only (``credentials_ref``, a name resolved via
    the platform ``SecretsProvider``) — never as raw keys — so secrets never travel in the
    request, logs, or the persisted catalog. The ``endpoint`` is enforced against the
    sovereignty egress allowlist at fetch time; a non-allowlisted host is refused."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    endpoint: str = Field(..., description="S3-compatible endpoint host[:port] (must be egress-allowlisted).")
    bucket: str = Field(..., description="Source bucket name.")
    secure: bool = Field(default=True, description="Use TLS (https) to reach the endpoint.")
    # Which object-storage API the endpoint speaks. "s3" (default) is any S3-compatible service
    # (AWS, MinIO, Ceph, ...). "azure" is Azure Blob Storage, which has no S3 API: endpoint is the
    # account host (``<account>.blob.core.windows.net``), bucket is the CONTAINER, and the
    # credential reference resolves to ``{ref}_account_key`` or ``{ref}_sas_token``.
    provider: str = Field(default="s3", pattern="^(s3|azure)$", description="Object-storage API: s3 | azure.")
    credentials_ref: str = Field(
        ..., description="Secret NAME resolved via SecretsProvider to {ref}_access_key/{ref}_secret_key."
    )


class DocumentEnvelope(BaseModel):
    """A single document as it enters the platform."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(..., description="Owning tenant; drives namespace isolation.")
    doc_id: str = Field(..., description="Stable per-tenant document id.")
    source_id: str = Field(..., description="Connector/source instance id.")
    source_kind: SourceKind
    mime: str = Field(..., description="MIME type, e.g. application/pdf.")

    # Exactly one of `uri` (by-reference, preferred for lakehouse/object) or `content`
    # (inline bytes for small uploads) is populated.
    uri: str | None = Field(default=None, description="Pointer to bytes (lineage-preserving).")
    content_b64: str | None = Field(default=None, description="Inline base64 bytes for small docs.")
    # Set (with source_kind=object_store) when `uri` is a key in an EXTERNAL tenant bucket
    # rather than the platform's own object store (FR-A2b). None => platform store (A2a).
    external_source: ExternalObjectSource | None = Field(
        default=None, description="External S3 source for by-reference pull (FR-A2b)."
    )

    source_acl: list[str] = Field(default_factory=list, description="ACL principals inherited from the source (FR-A6).")
    classification: int = Field(
        default=0, ge=0, le=3, description="Data classification level 0=Public..3=Restricted (banking pack)."
    )
    lineage_id: str | None = Field(default=None, description="Lakehouse/source lineage pointer (FR-G7).")
    metadata: dict[str, str] = Field(default_factory=dict, description="Source-provided tags (FR-A7).")
    content_hash: str | None = Field(
        default=None, description="SHA-256 of the raw bytes — drives delta sync + near-dup detection (FR-A8/C9)."
    )
    ingested_at: datetime = Field(..., description="Ingestion timestamp (UTC).")
