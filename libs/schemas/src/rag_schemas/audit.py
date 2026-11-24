"""AuditEvent — tamper-evident audit record (TDD §4.7/§9, FR-G4).

Records prompt + retrieved context + output + reviewer action for every request and
agent action. In production these are HSM-signed and written append-only to
WORM/Object-Lock. Phase-0 computes a hash chain (``prev_hash`` -> ``event_hash``) so
tamper-evidence is provable now; HSM signing is the Phase-1 drop-in.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rag_schemas.common import AuditAction


class AuditEvent(BaseModel):
    """One append-only audit record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    tenant_id: str
    trace_id: str
    action: AuditAction
    actor: str = Field(..., description="Principal subject (user/agent/service).")
    at: datetime = Field(..., description="Event time (UTC).")

    # Reproducibility payload (FR-G4/G7): what went in and what came out.
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Prompt, retrieved chunk_ids, model, output digest, reviewer action.",
    )

    prev_hash: str | None = Field(default=None, description="Hash of the previous event in the chain.")
    event_hash: str | None = Field(default=None, description="SHA-256 over canonical content + prev_hash.")

    def compute_hash(self) -> str:
        """Deterministic content hash chaining on ``prev_hash`` (tamper-evidence)."""
        content = {
            "event_id": self.event_id,
            "tenant_id": self.tenant_id,
            "trace_id": self.trace_id,
            "action": self.action.value,
            "actor": self.actor,
            "at": self.at.isoformat(),
            "payload": self.payload,
            "prev_hash": self.prev_hash,
        }
        canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def sealed(self) -> AuditEvent:
        """Return a copy with ``event_hash`` populated. Immutable by construction."""
        return self.model_copy(update={"event_hash": self.compute_hash()})
