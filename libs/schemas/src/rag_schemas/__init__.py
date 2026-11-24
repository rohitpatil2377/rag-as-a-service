"""rag_schemas — shared pydantic data contracts for RAG as a Service."""

from rag_schemas.audit import AuditEvent
from rag_schemas.chunk import Chunk, Span
from rag_schemas.citation import (
    AgentStep,
    Citation,
    Completeness,
    FederationCoverage,
    GroundedAnswer,
    ProcessingReport,
)
from rag_schemas.common import (
    AnswerStatus,
    AuditAction,
    ChunkLevel,
    ChunkStrategy,
    Modality,
    RetrievalMode,
    SourceKind,
)
from rag_schemas.envelope import DocumentEnvelope, ExternalObjectSource
from rag_schemas.federation import (
    FederatedQueryRequest,
    FederatedQueryResponse,
    FederatedSpan,
    Grant,
)
from rag_schemas.retrieval import RetrievalQuery, RetrievalResult, RetrievedSpan

__all__ = [
    "AgentStep",
    "AnswerStatus",
    "AuditAction",
    "AuditEvent",
    "Chunk",
    "ChunkLevel",
    "ChunkStrategy",
    "Citation",
    "Completeness",
    "DocumentEnvelope",
    "ExternalObjectSource",
    "FederatedQueryRequest",
    "FederatedQueryResponse",
    "FederatedSpan",
    "FederationCoverage",
    "Grant",
    "GroundedAnswer",
    "Modality",
    "ProcessingReport",
    "RetrievalMode",
    "RetrievalQuery",
    "RetrievalResult",
    "RetrievedSpan",
    "SourceKind",
    "Span",
]
