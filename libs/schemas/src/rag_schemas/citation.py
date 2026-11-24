"""Citation + Answer contracts (TDD §4.5, FR-E2/E3/E6).

Grounded-or-abstain is encoded in the type: a ``GroundedAnswer`` with status
GROUNDED MUST have >=1 citation; an ABSTAINED answer has none and states why.
Structured output (FR-E6) is carried in ``structured`` for agent consumption.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_schemas.chunk import Span
from rag_schemas.common import AnswerStatus, Modality


class Citation(BaseModel):
    """A clickable link from an answer span to a source span (FR-E3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    doc_id: str
    page: int | None = None
    span: Span | None = None
    source_uri: str | None = None
    quote: str = Field(..., description="The supporting text span from the source.")
    score: float = Field(..., ge=0.0, le=1.0)
    # Source modality — 'visual' marks a citation whose source is an image (multimodal),
    # so the UI can badge it. Default 'text' for every existing path.
    modality: Modality = Modality.TEXT


class AgentStep(BaseModel):
    """One step the agentic pipeline took — a transparent, auditable decision trace (FR-F).

    NON-AUTHORITATIVE: this explains *how* the answer was produced (plan → decompose →
    retrieve → retry → graph → verify); it never changes the answer or its grounding.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: str = Field(..., description="Stage id: plan|decompose|retrieve|retry|graph|themes|recency|verify.")
    detail: str = Field(..., description="Human-readable one-line summary of what the step did.")


class FederationCoverage(BaseModel):
    """Which departments\' evidence is (and is NOT) in a federated answer.

    Honesty under partial failure (T1): when a granted peer is down/slow, its evidence is
    MISSING and the answer must say so rather than imply completeness. ``reached`` lists the
    sources that contributed; ``unreachable`` names granted peers we could not reach; a
    consumer renders ``note`` as the caveat."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reached: list[str] = Field(
        default_factory=list,
        description="Departments whose evidence is in this answer (own + reachable granted peers).",
    )
    unreachable: list[str] = Field(
        default_factory=list, description="Granted peers that were down/slow — their evidence is MISSING here."
    )
    complete: bool = Field(default=True, description="True iff every granted peer was reached.")
    note: str = Field(default="", description="Human-readable completeness caveat.")


class Completeness(BaseModel):
    """NON-AUTHORITATIVE completeness signal for a grounded answer (Slice 1 — strict completeness guard).

    A partial answer must never be passed off as whole: the guard either FIXES it (bounded, adaptive
    continuation of a truncated completion + a bounded coverage loop for omitted relevant points) or
    MARKS it here with ``complete=False`` and a human-readable ``reason`` so the console/caller can
    surface "answer may be incomplete". This is NOT a citation and does NOT touch grounded-or-abstain:
    it accompanies the answer, it never gates it. ``covered``/``expected`` are the adaptive coverage
    tally (spans above the per-query relevance bar), ``truncated`` whether the completion was cut off
    by the token budget after the bounded continuation attempts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    complete: bool = Field(default=True, description="True iff the answer is whole after the guard's bounded attempts.")
    reason: str | None = Field(default=None, description="Why the answer is not complete (None when complete).")
    covered: int = Field(default=0, ge=0, description="Relevant retrieved points reflected in the answer.")
    expected: int = Field(default=0, ge=0, description="Relevant retrieved points above the per-query relevance bar.")
    truncated: bool = Field(default=False, description="True if still cut off by the token budget after continuation.")


class ProcessingReport(BaseModel):
    """Whether the SOURCE MATERIAL behind an answer was successfully processed at ingest.

    NOT THE SAME DIMENSION AS :class:`Completeness`, and the names are close enough that the
    difference has to be stated. ``Completeness`` asks whether the ANSWER is whole after bounded
    continuation and coverage attempts -- a generation-time question. This asks whether the
    DOCUMENT was readable in the first place. An answer can be complete by that measure, grounded
    in every citation it carries, and still be missing three sections whose enrichment failed a
    day earlier at ingest.

    INDEPENDENT OF GROUNDING AND VERIFICATION, in both directions. It never moves
    ``AnswerStatus``, ``faithfulness_score``, a verification method or a degradation flag, and
    none of those may move it.

    ``status`` is a scalar and the per-document detail is deliberately absent: a per-document
    composite on the answer contract would be the shape verification was careful not to take.
    ``scope_basis`` says which documents were assessed, so ``COMPLETE`` can be read for what it
    is -- complete for the ASSESSED SOURCES, never a claim that nothing in the knowledge base was
    missed before retrieval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["COMPLETE", "PARTIAL", "UNKNOWN"] = Field(
        default="UNKNOWN",
        description="COMPLETE: every assessed document has a current, complete manifest. "
        "PARTIAL: at least one has a known processing failure or was capped. "
        "UNKNOWN: a manifest is missing, stale or unreadable, or nothing could be assessed.",
    )
    scope_basis: Literal["resolved_doc_ids", "cited_documents", "mixed", "none"] = Field(
        default="none", description="Which documents were assessed, so the claim can be read correctly."
    )
    documents_assessed: int = Field(default=0, ge=0)
    documents_partial: int = Field(default=0, ge=0)
    documents_unknown: int = Field(
        default=0, ge=0, description="Kept even when status is PARTIAL: a mixture is not the same as a clean rest."
    )
    sections_offered: int = Field(default=0, ge=0)
    sections_failed: int = Field(default=0, ge=0)
    note: str = Field(
        default="",
        description="The count-only disclosure. Names no section and no document: the platform "
        "knows which section RECORDS failed and has no evidence of what they CONTAIN.",
    )


class GroundedAnswer(BaseModel):
    """Final answer object. Enforces the grounded-or-abstain invariant."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    query: str
    status: AnswerStatus
    answer: str = Field(..., description="Answer text, or the abstain message.")
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0, description="Max retrieval confidence.")
    abstain_reason: str | None = Field(default=None)
    structured: dict[str, Any] | None = Field(
        default=None, description="Optional JSON-schema output for agents (FR-E6)."
    )
    trace_id: str | None = Field(default=None, description="Correlates to the audit trail (FR-G4).")
    model: str | None = Field(default=None, description="Model/deployment that produced the answer (null on abstain).")

    # Faithfulness verification (FR-E4). Set when the faithfulness pass ran.
    faithfulness_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Fraction of the answer supported by cited spans."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Answer sentences a check did not find supported. Reachable on a GROUNDED "
        "answer too, not only the FLAGGED path: the numeric guard can report a finding on an "
        "answer that routes GROUNDED on its score, and a reader who sees nothing cannot tell "
        "'nothing was flagged' from 'the field only populates when the status changes'.",
    )
    # VERIFICATION PROVENANCE, so GROUNDED cannot be read as "every check succeeded".
    #
    # `faithfulness_score` says how well the answer verified. It does not say WHICH checker
    # produced that number, or whether the intended one was even reachable -- so an answer scored
    # by a lexical fallback after an NLI read error was indistinguishable from an NLI-verified one
    # at every surface a reader has. The only trace was a server log line.
    #
    # NO AGGREGATE SCORE IS MANUFACTURED HERE. Where several batches were verified by DIFFERENT
    # methods, an NLI score and a lexical score are not the same measurement, and averaging them
    # produces a number that means nothing and looks authoritative. The distribution and the
    # per-batch records are reported instead, and a reader who wants one number has to decide
    # which batches they trust.
    #
    # SHAPE, and `outcome` is a DISCRIMINANT a consumer must branch on before reading `score`:
    #
    #   {"outcome": "scored" | "mixed" | "unavailable",
    #    "degraded": bool,
    #    "methods": [{"method": str, "count": int}, ...],   # ordered distribution, NOT a map
    #    "batches": [{"method", "degraded", "score", "reason_code"}, ...],
    #    "findings": [{"method", "reason_code", "detail"}, ...],   # only when something degraded
    #    "score": float}                                     # ONLY when outcome == "scored"
    #
    # `scored` means one method produced every score, so one number is meaningful. `mixed` means
    # several did, and no mean is reported because an NLI score and a lexical score are not the
    # same measurement. `unavailable` means at least one batch was never checked.
    #
    # OMITTING THE KEY IS NOT A SAFEGUARD, which is why the discriminant exists: in JavaScript an
    # absent property is `undefined`, and `v.score ?? 0` yields 0 for `undefined` exactly as for
    # `null`, so a careless consumer turns "not comparable" into a failing grade either way.
    #
    # `degraded` is orthogonal to `outcome`: a run where EVERY batch fell back to the lexical
    # checker is `scored` (its scores are comparable) AND degraded (the intended mechanism never
    # ran). `lexical` is the configured checker and is not degraded; `lexical_fallback` is the one
    # that ran because NLI failed, and is.
    verification: dict[str, Any] | None = Field(
        default=None,
        description="How this answer was verified: methods used, how many batches used each, "
        "whether any verification was degraded or unavailable, and which batches were affected. "
        "None when no verification ran at all.",
    )
    review_id: str | None = Field(default=None, description="HITL review id when FLAGGED_FOR_REVIEW.")
    federation_coverage: FederationCoverage | None = Field(
        default=None,
        description="Set on a federated answer: which departments were reached vs unreachable, "
        "so a down peer never silently implies completeness (T1). None for a non-federated answer.",
    )

    # Suggested follow-up questions (FR-I8). NON-AUTHORITATIVE hints generated ONLY from the
    # retrieved context (never the model's parametric knowledge) — offered, never answers.
    suggested_followups: list[str] = Field(
        default_factory=list, description="Non-authoritative follow-up questions from retrieved context (FR-I8)."
    )

    # Grounded query disambiguation (Layer-2 item 4). NON-AUTHORITATIVE "Did you mean …?" options
    # derived ONLY from the retrieved spans (never invented) — offered when the query is ambiguous.
    # NOT a citation: it does not touch the grounded-or-abstain invariant, so it may accompany a
    # GROUNDED answer OR an ABSTAINED one (which still carries 0 citations + the abstain message).
    clarification_options: list[str] = Field(
        default_factory=list,
        description="Non-authoritative grounded 'Did you mean …?' options from retrieved spans (Layer-2 item 4).",
    )

    # Architecture-diagram affordance (auto+confirm). NON-AUTHORITATIVE: set when the question shows
    # POTENTIAL (but not strong) architecture intent, so the console OFFERS an on-demand "generate
    # diagram" chip. Strong intent renders the diagram into ``structured`` instead. Never an answer.
    diagram_available: bool = Field(
        default=False, description="True when the console should offer an on-demand architecture diagram."
    )

    # Agentic decision trace (FR-F). NON-AUTHORITATIVE — the ordered steps the pipeline took
    # (plan/decompose/retrieve/retry/graph/verify), so an agentic answer is explainable & auditable.
    agent_trace: list[AgentStep] = Field(
        default_factory=list, description="Ordered steps the agentic pipeline took for this answer (FR-F)."
    )

    # Strict completeness guard (Slice 1). NON-AUTHORITATIVE: set when the guard ran, it says whether
    # the answer is whole after bounded continuation/coverage attempts (or why not). Never a citation
    # and never gates the answer, so it does not touch the grounded-or-abstain invariant. None when
    # the guard is disabled (flag-OFF is byte-identical) or the answer was not GROUNDED.
    completeness: Completeness | None = Field(
        default=None, description="Non-authoritative completeness signal for a grounded answer (Slice 1)."
    )

    # DOCUMENT-PROCESSING completeness — a fourth governance dimension, not a variant of the field
    # above. See `ProcessingReport`: that one asks whether the ANSWER is whole, this asks whether
    # the DOCUMENT was readable. None when nothing assessed it (the flag is off, or no document
    # scope and no citation), which a consumer must not render as a clean result.
    processing: ProcessingReport | None = Field(
        default=None,
        description="Whether the source documents behind this answer were fully processed at ingest.",
    )

    @model_validator(mode="after")
    def _check_grounding_invariant(self) -> GroundedAnswer:
        # GROUNDED and FLAGGED_FOR_REVIEW are both cited, generated answers.
        if self.status in (AnswerStatus.GROUNDED, AnswerStatus.FLAGGED_FOR_REVIEW) and not self.citations:
            raise ValueError(f"{self.status} answer must carry at least one citation")
        if self.status is AnswerStatus.ABSTAINED and self.citations:
            raise ValueError("abstained answer must not carry citations")
        return self
