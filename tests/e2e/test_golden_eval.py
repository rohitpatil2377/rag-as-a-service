"""Golden-set eval gate (FR-G5) — CI release gate.

Ingests the sample brief, runs each golden question through the full query pipeline,
and asserts the Phase-0 metrics hold: 100% citation coverage on grounded answers and
correct abstain behaviour. Fails the build (via ``EvalReport.passes``) on regression.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from svc_guardrails_eval import GoldenItem, evaluate

from rag_schemas import DocumentEnvelope, RetrievalQuery
from rag_schemas.common import ChunkStrategy, SourceKind

pytestmark = pytest.mark.eval

_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN = _ROOT / "sample-data" / "golden" / "golden.json"
_BRIEF = _ROOT / "sample-data" / "platform_brief.md"
TENANT = "cag"


async def test_golden_eval_gate(inmemory: None) -> None:
    from svc_control_plane import pipeline

    from rag_clients.factory import get_reranker
    from rag_clients.reranker import LexicalReranker
    from rag_common import tenant_scope

    # This is the RELEASE gate, so it has to run against a scorer whose number means something.
    # Measured 2026-07-31 on 20 questions against a real 40-page RFE, the lexical twin does not
    # separate answerable from unanswerable AT ALL (answerable 0.5714-0.8750, unanswerable
    # 0.3333-0.6250) — every threshold either answers unanswerable questions or abstains on
    # answerable ones. A pass or a fail from it would be a coincidence, not a gate. Bring the
    # stack up (`make up`) and run this against the real cross-encoder.
    if isinstance(get_reranker(), LexicalReranker):
        pytest.skip(
            "release gate needs a real cross-encoder; the lexical twin cannot separate "
            "answerable from unanswerable (see abstain_threshold_lexical in config.py)"
        )

    golden = json.loads(_GOLDEN.read_text())
    items = [
        GoldenItem(
            question=i["question"],
            expected_answer=i["expected_answer"],
            expected_doc_ids=i["expected_doc_ids"],
            expect_abstain=i["expect_abstain"],
        )
        for i in golden["items"]
    ]

    with tenant_scope(TENANT):
        doc = DocumentEnvelope(
            tenant_id=TENANT,
            doc_id=golden["doc_id"],
            source_id="golden",
            source_kind=SourceKind.UPLOAD,
            mime="text/plain",
            content_b64=base64.b64encode(_BRIEF.read_text().encode()).decode(),
            ingested_at=datetime(2026, 7, 2, tzinfo=UTC),
        )
        await pipeline.ingest_document(doc, actor="eval", strategy=ChunkStrategy.CLAUSE)

        answers = []
        for item in items:
            rq = RetrievalQuery(tenant_id=TENANT, text=item.question, top_k=3)
            answers.append(await pipeline.answer_query(rq, actor="eval"))

    report = evaluate(items, answers)
    assert report.citation_coverage == 1.0, f"citation coverage regressed: {report}"
    assert report.abstain_correctness == 1.0, f"abstain behaviour regressed: {report}"
    assert report.passes(), report
