"""rag_clients — concrete adapters implementing the rag-core Protocols.

Import backends via :mod:`rag_clients.factory`, not directly, so backend selection
stays in one place. Direct imports are for tests and the Alembic env only.
"""

from rag_clients.contextual import LLMContextualizer, build_contextualizer
from rag_clients.faithfulness import (
    LexicalFaithfulnessChecker,
    LLMFaithfulnessChecker,
    build_faithfulness_checker,
)
from rag_clients.guardrails import LLMGuard, build_input_guard, build_output_guard
from rag_clients.llm_gateway import (
    ExtractiveDevGateway,
    OpenAICompatGateway,
    VLLMGateway,
    build_llm_gateway,
)
from rag_clients.minio_store import MinioObjectStore
from rag_clients.opensearch_index import OpenSearchSparseIndex
from rag_clients.postgres_audit import PostgresAuditSink
from rag_clients.postgres_catalog import PostgresCatalog
from rag_clients.qdrant_store import QdrantVectorStore
from rag_clients.reranker import CrossEncoderReranker, LexicalReranker, build_reranker
from rag_clients.routing import LLMRouter, build_query_router
from rag_clients.tei_embedder import TeiEmbedder
from rag_clients.transform import HyDETransformer, RewriteTransformer, build_query_transformer

__all__ = [
    "CrossEncoderReranker",
    "ExtractiveDevGateway",
    "HyDETransformer",
    "LLMContextualizer",
    "LLMFaithfulnessChecker",
    "LLMGuard",
    "LLMRouter",
    "LexicalFaithfulnessChecker",
    "LexicalReranker",
    "MinioObjectStore",
    "OpenAICompatGateway",
    "OpenSearchSparseIndex",
    "PostgresAuditSink",
    "PostgresCatalog",
    "QdrantVectorStore",
    "RewriteTransformer",
    "TeiEmbedder",
    "VLLMGateway",
    "build_contextualizer",
    "build_faithfulness_checker",
    "build_input_guard",
    "build_llm_gateway",
    "build_output_guard",
    "build_query_router",
    "build_query_transformer",
    "build_reranker",
]
