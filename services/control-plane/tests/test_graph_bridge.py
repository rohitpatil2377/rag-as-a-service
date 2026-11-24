"""Graph bridge — structured-source entities join the KB graph and share nodes with documents."""

from __future__ import annotations

from svc_control_plane import graph_bridge

from rag_clients.graph_store import InMemoryGraphStore
from rag_core.graph import GraphTriple, cooccurrence_edges
from rag_schemas import RetrievedSpan


class _Extractor:
    """Deterministic fake extractor — returns fixed entities (extractor quality is not under test;
    the bridge + graph entity-resolution are)."""

    def __init__(self, entities: list[str]) -> None:
        self._entities = entities

    async def extract(self, text: str) -> GraphTriple:
        return GraphTriple(entities=self._entities, edges=cooccurrence_edges(self._entities))


def _db_span(text: str) -> RetrievedSpan:
    return RetrievedSpan(chunk_id="sql::bank__deals", doc_id="db://bank__deals", text=text, score=1.0, scorer="exact")


async def test_bridge_puts_db_fact_in_contact_with_documents() -> None:
    store = InMemoryGraphStore()
    await store.ensure("bank__deals")
    # A document already mentions ACME Corp.
    doc_entities = ["ACME Corp", "Contract"]
    await store.add_chunk("bank__deals", "doc::1", doc_entities, cooccurrence_edges(doc_entities))
    # Bridge a DB result that also mentions ACME Corp (+ an invoice).
    linked = await graph_bridge.bridge_spans_to_graph(
        "bank__deals",
        [_db_span("Customer ACME Corp has invoice 42 outstanding")],
        store=store,
        extractor=_Extractor(["ACME Corp", "Invoice 42"]),
    )
    assert linked == 2
    # PROOF of contact: from the ACME Corp node, local graph search reaches BOTH the document
    # chunk and the DB span — they share the node, so the DB fact is in contact with the docs.
    reached = await store.linked_chunks("bank__deals", ["ACME Corp"], hops=1)
    assert "doc::1" in reached
    assert "sql::bank__deals" in reached


async def test_bridge_failsafe_on_extractor_error() -> None:
    class _Boom:
        async def extract(self, text: str) -> GraphTriple:
            raise RuntimeError("extractor down")

    store = InMemoryGraphStore()
    await store.ensure("k")
    linked = await graph_bridge.bridge_spans_to_graph("k", [_db_span("ACME Corp")], store=store, extractor=_Boom())
    assert linked == 0  # skipped, no raise


async def test_bridge_skips_empty_entities() -> None:
    store = InMemoryGraphStore()
    await store.ensure("k")
    linked = await graph_bridge.bridge_spans_to_graph("k", [_db_span("1 2 3")], store=store, extractor=_Extractor([]))
    assert linked == 0
