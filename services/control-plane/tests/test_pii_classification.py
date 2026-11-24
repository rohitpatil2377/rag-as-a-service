"""PII-driven classification floor: the pure floor/trigger mapping + the ingest enforcement
(raise / block-by-mode / block-over-clearance / off)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from svc_control_plane import pipeline
from svc_control_plane.knowledge_base import SHARED_KB_ID
from svc_control_plane.pii_classification import pii_classification_floor, pii_triggers
from svc_control_plane.pipeline import IngestOutcome

from rag_schemas import DocumentEnvelope
from rag_schemas.common import AuditAction, SourceKind


def test_floor_is_max_over_detected_types() -> None:
    assert pii_classification_floor({}) == 0
    # Contact PII (email/phone/IP) is masked but does NOT raise the floor — it's ubiquitous and
    # would block ordinary uploads. So a doc with only an email stays Public.
    assert pii_classification_floor({"EMAIL": 3}) == 0
    assert pii_classification_floor({"PHONE": 2, "IP": 1}) == 0
    assert pii_classification_floor({"AADHAAR": 1}) == 2  # identity -> Confidential
    assert pii_classification_floor({"CREDIT_CARD": 1}) == 3  # financial secret -> Restricted
    # the MAX floor over all present types wins; contact PII contributes nothing
    assert pii_classification_floor({"EMAIL": 9, "PAN": 1}) == 2
    assert pii_classification_floor({"PHONE": 2, "AADHAAR": 1, "CVV": 1}) == 3
    # a zero count contributes nothing
    assert pii_classification_floor({"AADHAAR": 0, "PAN": 1}) == 2


def test_triggers_are_the_types_holding_the_floor() -> None:
    assert pii_triggers({"EMAIL": 1, "AADHAAR": 2}, 2) == ["AADHAAR"]
    assert set(pii_triggers({"AADHAAR": 1, "CVV": 1}, 3)) == {"CVV"}
    # contact PII never holds a floor (it maps to 0)
    assert pii_triggers({"EMAIL": 1}, 1) == []


@dataclass
class _Extracted:
    pii_counts: dict[str, int]
    pii_masked: int = 0


@dataclass
class _Audit:
    events: list[dict[str, Any]] = field(default_factory=list)

    async def record(self, **kw: object) -> None:
        self.events.append(dict(kw))


@dataclass
class _Catalog:
    registered: list[DocumentEnvelope] = field(default_factory=list)

    async def register_document(self, doc: DocumentEnvelope) -> None:
        self.registered.append(doc)


def _doc(classification: int = 0, tenant: str = "k") -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id=tenant,
        doc_id="d1",
        source_id="s",
        source_kind=SourceKind.UPLOAD,
        mime="text/plain",
        classification=classification,
        ingested_at=datetime.now(UTC),
    )


async def _apply(
    monkeypatch: pytest.MonkeyPatch,
    *,
    counts: dict[str, int],
    requested: int,
    clearance: int,
    policy: str,
    tenant: str = "k",
) -> tuple[DocumentEnvelope, IngestOutcome | None, _Audit, _Catalog]:
    audit, catalog = _Audit(), _Catalog()
    monkeypatch.setattr(pipeline, "get_catalog", lambda: catalog)
    doc, outcome = await pipeline._apply_pii_classification(
        _doc(requested, tenant),
        _Extracted(counts, pii_masked=sum(counts.values())),
        actor="u",
        trace_id="t",
        audit=cast(Any, audit),
        object_uri="obj://x",
        clearance=clearance,
        pii_policy=policy,
    )
    return doc, outcome, audit, catalog


async def test_raise_bumps_classification_when_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    # Aadhaar (floor 2) uploaded as Public by a Restricted-cleared ingestor -> raised to 2, indexed.
    doc, outcome, audit, _catalog = await _apply(
        monkeypatch, counts={"AADHAAR": 3}, requested=0, clearance=3, policy="raise"
    )
    assert outcome is None  # not blocked -> ingest continues
    assert doc.classification == 2
    assert any(e["action"] is AuditAction.CLASSIFICATION and e["payload"]["result"] == "raised" for e in audit.events)


async def test_raise_still_blocks_when_floor_exceeds_clearance(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same doc, but the ingestor is only Internal(1) -> cannot store Confidential(2) content -> blocked.
    doc, outcome, _audit, catalog = await _apply(
        monkeypatch, counts={"AADHAAR": 3}, requested=0, clearance=1, policy="raise"
    )
    assert outcome is not None and outcome.outcome == "blocked_pii" and outcome.classification == 2
    assert "AADHAAR" in (outcome.reason or "") and "clearance" in (outcome.reason or "")
    assert doc.classification == 0  # unchanged (not indexed)
    assert catalog.registered and catalog.registered[0].metadata.get("pii_blocked") == "true"


async def test_block_mode_rejects_even_when_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    # policy=block: any under-classified PII doc is rejected so the user reclassifies explicitly.
    _doc_out, outcome, _audit, _catalog = await _apply(
        monkeypatch, counts={"PAN": 1}, requested=0, clearance=3, policy="block"
    )
    assert outcome is not None and outcome.outcome == "blocked_pii" and outcome.classification == 2


async def test_shared_kb_always_blocks_pii_even_when_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    # The company-wide shared KB is Public-only; a PII doc there must BLOCK (never raise), even with
    # full clearance in raise mode — raising would silently broadcast PII company-wide.
    doc, outcome, _a, _c = await _apply(
        monkeypatch, counts={"AADHAAR": 1}, requested=0, clearance=3, policy="raise", tenant=SHARED_KB_ID
    )
    assert outcome is not None and outcome.outcome == "blocked_pii"
    assert "shared KB" in (outcome.reason or "") and "Public-only" in (outcome.reason or "")
    assert doc.classification == 0  # never raised


async def test_off_and_already_high_enough_are_noops(monkeypatch: pytest.MonkeyPatch) -> None:
    # policy=off -> never touches classification.
    doc, outcome, audit, _ = await _apply(monkeypatch, counts={"AADHAAR": 1}, requested=0, clearance=3, policy="off")
    assert outcome is None and doc.classification == 0 and not audit.events
    # requested already at/above the floor -> no change, no audit noise.
    doc2, outcome2, audit2, _ = await _apply(monkeypatch, counts={"EMAIL": 1}, requested=2, clearance=3, policy="raise")
    assert outcome2 is None and doc2.classification == 2 and not audit2.events
