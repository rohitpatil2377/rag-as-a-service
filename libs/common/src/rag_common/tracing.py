"""OpenTelemetry tracing setup (open standard, no lock-in).

Traces export via OTLP to the collector, which fans out to Tempo/Prometheus and is
bridged into the AIOps platform (TDD §4.8). Instrumentation is a no-op when disabled so
unit tests and the extractive dev path need no collector running.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from rag_common.config import get_settings

if TYPE_CHECKING:
    from fastapi import FastAPI

_initialized = False


def init_tracing(service_name: str | None = None) -> None:
    """Idempotently install the global tracer provider + OTLP exporter."""
    global _initialized  # noqa: PLW0603 - idempotent process-wide init singleton
    if _initialized:
        return
    settings = get_settings()
    if not settings.otel_enabled:
        _initialized = True
        return

    # Imported lazily so environments without the exporter extra still import cleanly.
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # noqa: PLC0415
        OTLPSpanExporter,
    )

    resource = Resource.create(
        {"service.name": service_name or settings.otel_service_name, "deployment.environment": settings.env}
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True))
    )
    trace.set_tracer_provider(provider)
    _initialized = True


def instrument_app(app: FastAPI) -> None:
    """Attach FastAPI auto-instrumentation (spans per request)."""
    if not get_settings().otel_enabled:
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # noqa: PLC0415

    FastAPIInstrumentor.instrument_app(app)


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)


def current_traceparent() -> str | None:
    """The active span as a W3C ``traceparent`` header value, or None when no span is active.

    Used to propagate the trace across the federation hop so a peer department's retrieval
    joins the same distributed trace."""
    from opentelemetry.propagate import inject  # noqa: PLC0415 — optional, only on the federated path

    carrier: dict[str, str] = {}
    inject(carrier)
    return carrier.get("traceparent")
