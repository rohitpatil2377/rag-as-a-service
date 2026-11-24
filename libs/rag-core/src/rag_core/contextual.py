"""Contextual Retrieval (FR-C2) — interfaces + deterministic contextualizer.

Prepend a short document-level context to each chunk BEFORE embedding / BM25 indexing,
so each chunk is self-contained (Anthropic "Contextual Retrieval"). Critically, the
contextualized text is EMBEDDED/INDEXED only — the ORIGINAL chunk text is what the
catalog stores and what answers cite, so the context prefix never leaks into answers.

Config-gated (``contextual_retrieval_enabled``, default OFF), new-ingest-only: turning
it on affects future ingestions; existing indexes adopt it by re-ingesting. See ADR-0007.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_core.chunking import META_SECTION
from rag_schemas import Chunk, DocumentEnvelope


@runtime_checkable
class Contextualizer(Protocol):
    """Produces a short context string situating a chunk within its document (FR-C2)."""

    async def context_for(self, doc: DocumentEnvelope, chunk: Chunk, full_text: str) -> str: ...


class MetadataContextualizer:
    """Deterministic, no-LLM contextualizer: document identity + SECTION + position + tags.

    Cheap and sovereign — the dev/default backend. Good enough to disambiguate chunks
    across documents without an LLM call per chunk.

    The section path is the part that changes answers rather than just disambiguating
    documents. A chunk reading "written sign-off is required before publication" retrieves
    very differently depending on whether it sat under "6.1.1 Mandatory Rehearsal Policy" or
    "6.4.1 Central Content Repository", and the chunk itself carries no trace of which — the
    heading is a line of text that ended up in some other chunk. Restoring it costs nothing at
    query time and no model call.

    The section is READ from the chunk rather than recomputed here: the chunker already
    recorded which heading governs each chunk, and detecting it twice would let the value the
    embedder sees drift from the value the generator is shown."""

    async def context_for(self, doc: DocumentEnvelope, chunk: Chunk, full_text: str) -> str:
        title = doc.metadata.get("title") or doc.doc_id
        parts = [f"Document: {title}"]
        section = chunk.metadata.get(META_SECTION)
        if section:
            parts.append(f"section {section}")
        if chunk.page is not None:
            parts.append(f"page {chunk.page}")
        tags = "; ".join(f"{k}={v}" for k, v in doc.metadata.items() if k not in {"title", META_SECTION})
        if tags:
            parts.append(tags)
        return "[context] " + ", ".join(parts)


def build_context_prompt(doc: DocumentEnvelope, chunk: Chunk, full_text: str) -> str:
    """Prompt for the LLM contextualizer (canonical technique). Sources-only, one sentence."""
    snippet = full_text[:2000]
    return (
        f"<document>\n{snippet}\n</document>\n\n"
        f"Here is a chunk from that document:\n<chunk>\n{chunk.text.strip()}\n</chunk>\n\n"
        "Write ONE short sentence that situates this chunk within the document to improve "
        "search retrieval (what section/topic it belongs to). Answer with ONLY the sentence."
    )


def contextualize_text(context: str, chunk_text: str) -> str:
    """Compose the text that gets embedded/indexed: context prefix + original chunk."""
    return f"{context}\n\n{chunk_text}" if context else chunk_text
