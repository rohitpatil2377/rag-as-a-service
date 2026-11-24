"""Graph bridge — put a unified KB's structured-source facts INTO the knowledge graph.

Answer-time fusion (:mod:`unified`) makes documents and connected sources meet in the ANSWER.
This goes one step further: it extracts entities from each structured evidence span (a SQL / NoSQL
/ REST / SharePoint result) and upserts them into the SAME per-KB knowledge graph the documents
populate. Because the graph keys every node by ``normalize_entity(name)``, a database entity and a
document entity with the same name become the SAME node — automatic cross-source entity
resolution. The structured fact is then literally "in contact" with the documents in GraphRAG:
reachable by local graph search (``linked_chunks``) and visible in the graph view, so a
cross-source connection can be SHOWN, not merely asserted.

Additive + fail-safe, exactly like document graph indexing (:func:`pipeline._index_graph`): a span
that fails extraction is logged and skipped and never breaks the graph or the query.
"""

from __future__ import annotations

from rag_common import get_logger
from rag_core.graph import GraphExtractor, cooccurrence_edges
from rag_core.interfaces import GraphStore
from rag_schemas import RetrievedSpan

_log = get_logger(__name__)


async def sync_entities_to_graph(kb_id: str, chunk_id: str, entities: list[str], *, store: GraphStore) -> int:
    """Upsert a source's DECLARED structural entities (SQL tables, NoSQL collections, an API/site
    host …) into the KB graph under a stable ``chunk_id`` — the connection-time half of the bridge.

    Where :func:`bridge_spans_to_graph` links a source's live ANSWER entities, this seeds the
    source's SCHEMA up front, so a table named ``orders`` shares a node with any document that
    mentions orders BEFORE any query runs. Idempotent per chunk_id; returns the entity count."""
    names = [e.strip() for e in entities if e and e.strip()]
    if not names:
        return 0
    await store.add_chunk(kb_id, chunk_id, names, cooccurrence_edges(names))
    _log.info("graph_sync_entities", kb_id=kb_id, chunk_id=chunk_id, entities=len(names))
    return len(names)


async def bridge_spans_to_graph(
    kb_id: str, spans: list[RetrievedSpan], *, store: GraphStore, extractor: GraphExtractor
) -> int:
    """Extract and upsert each structured span's entities into the KB knowledge graph.

    Returns the number of entity mentions linked. Each span is bridged under its own stable chunk
    id (e.g. ``sql::<kb>``, ``sharepoint://<host>``), so its entities join the shared node space and
    connect — through equally-named nodes — to whatever documents already mention them. Fail-safe
    per span: extraction/upsert errors are logged and skipped."""
    linked = 0
    for span in spans:
        try:
            triple = await extractor.extract(span.text)
            if triple.entities:
                await store.add_chunk(kb_id, span.chunk_id, triple.entities, triple.edges)
                linked += len(triple.entities)
                _log.info("graph_bridge_linked", kb_id=kb_id, chunk_id=span.chunk_id, entities=len(triple.entities))
        except Exception as exc:
            _log.warning("graph_bridge_failed", kb_id=kb_id, chunk_id=span.chunk_id, error=str(exc)[:160])
    return linked
