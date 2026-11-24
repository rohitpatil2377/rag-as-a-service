"""Typed response models (mirror the ``/v1`` contracts; standalone — no platform imports)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class KnowledgeBase(BaseModel):
    kb_id: str
    name: str
    is_default: bool = False


class Document(BaseModel):
    doc_id: str
    title: str | None = None
    mime: str | None = None
    chunk_count: int | None = None


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    quote: str = ""
    page: int | None = None
    score: float | None = None


class GroundedAnswer(BaseModel):
    """A grounded-or-abstain answer with its citations (and FR-I8 follow-ups when enabled)."""

    tenant_id: str
    query: str
    status: str  # grounded | abstained | flagged_for_review
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    abstain_reason: str | None = None
    faithfulness_score: float | None = None
    trace_id: str | None = None
    model: str | None = None
    suggested_followups: list[str] = Field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        return self.status == "grounded"

    @property
    def abstained(self) -> bool:
        return self.status == "abstained"
