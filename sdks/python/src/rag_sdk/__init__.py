"""RAG as a Service — official Python SDK.

A thin, typed, sovereign client over the ``/v1`` API. No dependency on the platform's
internal packages — install it standalone (``pip install rag-sdk``) and point it at your
RAG as a Service deployment.

    from rag_sdk import RagClient

    client = RagClient("https://rag.internal", api_key="ek_...")
    client.ingest("policies", doc_id="leave", text="Annual leave is 24 days.")
    answer = client.query("policies", "How many annual leave days?")
    print(answer.answer, [c.doc_id for c in answer.citations])

Versioning follows SemVer (MAJOR.MINOR.PATCH); the public surface and backward-compatibility
guarantees are documented in ``docs/sdk-versioning.md``. ``__version__`` is the single source
of truth and is advertised to the server on every request (see ``client.py``).
"""

# ``__version__`` is defined before the package imports so ``rag_sdk.client`` can import it
# at module load without a circular import. (Ruff's E402 exempts dunder assignments.)
__version__ = "0.1.0"

from rag_sdk.client import RagClient
from rag_sdk.errors import RagError
from rag_sdk.models import Citation, Document, GroundedAnswer, KnowledgeBase

__all__ = [
    "Citation",
    "Document",
    "GroundedAnswer",
    "KnowledgeBase",
    "RagClient",
    "RagError",
    "__version__",
]
