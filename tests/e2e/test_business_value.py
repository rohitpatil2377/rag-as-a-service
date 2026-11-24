"""Business-value benchmark, end-to-end through the real pipeline (in-memory twins).

Runs a small workload against an ingested KB and produces a MEASURED report: latency per query,
grounded/hit quality, cost from the FinOps meters, and — with a baseline — the reduction/speedup.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime

import pytest

pytestmark = pytest.mark.e2e

from rag_schemas import DocumentEnvelope  # noqa: E402
from rag_schemas.common import SourceKind  # noqa: E402

_TENANT = "bizval"


async def _ingest(doc_id: str, text: str) -> None:
    from svc_control_plane import pipeline

    await pipeline.ingest_document(
        DocumentEnvelope(
            tenant_id=_TENANT,
            doc_id=doc_id,
            source_id="s",
            source_kind=SourceKind.UPLOAD,
            mime="text/plain",
            content_b64=base64.b64encode(text.encode()).decode(),
            ingested_at=datetime(2026, 8, 5, tzinfo=UTC),
        ),
        actor="loader",
    )


async def test_business_value_report_is_measured_end_to_end(inmemory: None, lexical_gate: None) -> None:
    from svc_control_plane.business_value import BizBaseline, render_markdown, run_business_value
    from svc_control_plane.quality import EvalQuestion

    await _ingest("sla", "The platform's uptime target is 99.5 percent baseline and 99.9 percent at peak.")

    workload = [
        EvalQuestion(question="What is the platform's uptime target?", expected="99.5"),
        EvalQuestion(question="Who won the 1998 world cup?"),  # unanswerable -> abstain
    ]
    baseline = BizBaseline(name="hosted-API RAG", cost_per_answer=5.0, latency_p50_ms=8000.0, grounded_rate=0.4)

    report = await run_business_value(_TENANT, workload, baseline, actor="bench")

    assert report.workload_size == 2
    assert report.latency.count == 2 and report.latency.p50 >= 0.0
    # Cost is priced from the run's own audit meters (queries + tokens), per answer.
    assert report.cost.cost_per_answer >= 0.0 and report.cost.currency
    assert report.quality.grounded_rate is not None
    assert report.quality.hit_rate is not None  # one question carried an expected fact
    # Baseline present -> a comparison is computed (cost + latency + grounded delta).
    assert report.comparison is not None
    assert report.comparison.baseline_name == "hosted-API RAG"
    assert report.comparison.cost_reduction_pct is not None
    assert report.comparison.productivity_speedup is not None
    # Honest: caveats always present, and the report renders to shareable markdown.
    assert report.caveats
    assert "Business-value benchmark" in render_markdown(report)
