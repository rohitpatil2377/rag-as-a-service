"""Federation schemas — cross-tenant P2P RAG (Federated RAG)."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field


class Grant(BaseModel):
    """A publisher-approved authorisation for one department to query another's KBs.

    Signed by the issuer's Ed25519 key. The SovereignNode validates the signature
    before executing any local search — an unsigned or invalid grant returns empty
    spans, never an error.
    """

    grant_id: str
    issuer_dept_id: str
    subject_dept_id: str
    topics: Annotated[list[str], Field(min_length=1)]
    kb_ids: Annotated[list[str], Field(min_length=1)]
    max_results_per_query: int = Field(default=5, ge=1, le=20)
    not_before: datetime
    expires_at: datetime
    signed_token: str
    revoked_at: datetime | None = None
    created_at: datetime | None = None
    # Transitive delegation (A->B->C): set when this grant was RE-DELEGATED from another. The
    # issuer here is the parent grant's SUBJECT re-sharing a NARROWED subset of what it
    # received; the parent must stay active for this one to be usable (transitive revocation).
    # None = a root grant issued directly by a KB owner. See svc_control_plane.grants.attenuate.
    parent_grant_id: str | None = None

    @property
    def is_active(self) -> bool:
        from datetime import UTC  # noqa: PLC0415

        now = datetime.now(UTC)
        return self.revoked_at is None and now >= self.not_before and now < self.expires_at


class FederatedSpan(BaseModel):
    """A retrieved span returned by a remote peer node."""

    chunk_id: str
    text: str
    score: float
    doc_id: str
    kb_id: str
    source_dept_id: str


class FederatedQueryRequest(BaseModel):
    """Sent from an initiating node to a peer over the P2P channel."""

    grant_token: str
    query_text: str
    query_vector: list[float]
    top_k: int = Field(default=5, ge=1, le=20)
    topics: list[str] = Field(default_factory=list)


class FederatedQueryResponse(BaseModel):
    """Returned by a peer node after local retrieval."""

    spans: list[FederatedSpan] = Field(default_factory=list)
    peer_dept_id: str
    grant_valid: bool
    error: str | None = None
