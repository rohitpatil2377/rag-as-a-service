"""Error taxonomy for RAG as a Service.

Every error carries a stable, machine-readable ``code`` and an HTTP status so the
API layer can render consistent problem responses and clients can branch on codes
without string-matching. Handlers are registered on the FastAPI app in
:mod:`rag_common.middleware`.
"""

from __future__ import annotations

from typing import Any


class PlatformError(Exception):
    """Base class for all RAG as a Service domain errors."""

    code: str = "rag_error"
    http_status: int = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_problem(self) -> dict[str, Any]:
        """Render as an RFC-7807-ish problem dict (no internals leaked)."""
        return {"code": self.code, "message": self.message, "details": self.details}


class ConfigError(PlatformError):
    code = "config_error"
    http_status = 500


class NotFoundError(PlatformError):
    code = "not_found"
    http_status = 404


class ApprovalRequiredError(PlatformError):
    """A dual-control (maker-checker) gate staged the action for a second admin's approval.
    202 Accepted — not a fault; ``details`` carries the pending request id."""

    code = "approval_required"
    http_status = 202


class ValidationError(PlatformError):
    code = "validation_error"
    http_status = 422


class ConflictError(PlatformError):
    """The request is valid but the resource's current state refuses it (e.g. applying an
    embedder switch with nothing pending, deleting the embedder a live index depends on)."""

    code = "conflict"
    http_status = 409


class UnauthorizedError(PlatformError):
    code = "unauthorized"
    http_status = 401


class ForbiddenError(PlatformError):
    code = "forbidden"
    http_status = 403


class TenantIsolationError(ForbiddenError):
    """Raised when an operation would cross a tenant boundary. Always audited."""

    code = "tenant_isolation_violation"
    http_status = 403


class SovereigntyViolationError(ForbiddenError):
    """Raised when a call targets a non-allowlisted (e.g. foreign) endpoint."""

    code = "sovereignty_violation"
    http_status = 403


class RateLimitError(PlatformError):
    """Raised when a tenant exceeds its request-rate limit (FR-H1 quotas). Retryable."""

    code = "rate_limited"
    http_status = 429


class QuotaExceededError(PlatformError):
    """Raised when a tenant hits a hard resource cap (e.g. documents per KB)."""

    code = "quota_exceeded"
    http_status = 429


class RetrievalError(PlatformError):
    code = "retrieval_error"
    http_status = 502


class ConnectorError(PlatformError):
    """A connector could not reach or read an external source (network/upstream failure).

    Distinct from a sovereignty block (403) or a bad request (422): the host was permitted but
    unreachable/erroring, so the caller gets a clear, actionable 502 instead of an opaque 500."""

    code = "connector_error"
    http_status = 502


class ModelGatewayError(PlatformError):
    """The configured LLM endpoint failed (unreachable, 4xx/5xx, or bad response).

    Distinct from an internal error: it usually means the tenant's model connection is
    misconfigured (wrong URL, model name, or key), so the message is written to be
    actionable in the console (point the user at Models)."""

    code = "model_gateway_error"
    http_status = 502


class GroundingAbstain(PlatformError):  # noqa: N818 - a normal outcome (abstain), not an error condition
    """Not a failure: the grounded-or-abstain gate chose to abstain.

    Surfaced as ``200`` with an explicit abstain payload by the generation layer,
    but modelled as an exception so the gate can short-circuit generation.
    """

    code = "grounding_abstain"
    http_status = 200


class ServiceUnavailableError(PlatformError):
    """A backing store or dependency is temporarily unable to serve the request — e.g. the search
    index store is read-only / out of disk and refuses to create new indexes. Distinct from an
    internal bug: it is an operational capacity condition the caller should retry once it's
    resolved, so it surfaces as an actionable 503 rather than an opaque 500."""

    code = "service_unavailable"
    http_status = 503
