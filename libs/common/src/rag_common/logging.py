"""Structured logging via structlog.

Every log line is JSON in non-dev environments and carries the bound ``tenant_id``
and ``request_id`` when present, so logs are queryable per-tenant and correlate with
OTel traces. Secret values must never be logged (see :mod:`rag_common.secrets`).
"""

from __future__ import annotations

import logging

import structlog
from structlog.typing import EventDict, Processor, WrappedLogger

from rag_common.config import get_settings
from rag_common.tenancy import _tenant_var

_configured = False


def _add_tenant(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
    tid = _tenant_var.get()
    if tid is not None:
        event_dict.setdefault("tenant_id", tid)
    return event_dict


def configure_logging() -> None:
    """Idempotently configure process-wide structured logging."""
    global _configured  # noqa: PLW0603 - idempotent process-wide init singleton
    if _configured:
        return
    settings = get_settings()

    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        _add_tenant,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]
    renderer: Processor = structlog.processors.JSONRenderer() if settings.log_json else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[*shared, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(settings.log_level)),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structured logger (configuring on first use)."""
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)  # type: ignore[no-any-return]
