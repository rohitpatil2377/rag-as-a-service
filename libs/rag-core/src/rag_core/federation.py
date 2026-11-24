"""Federation Protocols — cross-tenant P2P RAG (Federated RAG).

These are the seams for the publisher-approved access model: a department (issuer)
can grant another department (subject) read access to specific topics from its KBs.
All federation is opt-in and cryptographically governed — no department can query
another without a valid, unexpired, non-revoked grant signed by the publisher.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_schemas.federation import Grant


@runtime_checkable
class GrantStore(Protocol):
    """Persists publisher-approved cross-tenant federation grants."""

    async def create(self, grant: Grant) -> None:
        """Persist a new grant. Raises ValueError if grant_id already exists."""
        ...

    async def get(self, grant_id: str) -> Grant | None:
        """Return the grant or None if not found."""
        ...

    async def list_issued(self, issuer_dept_id: str) -> list[Grant]:
        """All grants published BY this department (including revoked)."""
        ...

    async def list_received(self, subject_dept_id: str) -> list[Grant]:
        """All active (non-revoked, non-expired) grants issued TO this department."""
        ...

    async def revoke(self, grant_id: str, issuer_dept_id: str) -> bool:
        """Revoke a grant. Returns True if found and revoked, False if not found.
        Raises PermissionError if issuer_dept_id does not match the grant's issuer."""
        ...

    async def get_active(self, issuer_dept_id: str, subject_dept_id: str) -> list[Grant]:
        """Active grants from issuer → subject (non-revoked, non-expired).
        Used by the federated query orchestrator to check permissions before scatter."""
        ...
