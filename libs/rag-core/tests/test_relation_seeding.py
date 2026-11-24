"""Resolving a document's DECLARED relations into graph edges (RAG v2 Phase 6).

Phase 3 writes what a document says about itself (`doc_relations`: amends / supersedes / attaches /
references) and Phase 4 writes the same at section level (`sections.amends`). Each row carries a
verbatim evidence quote, which is what makes this better evidence than anything an extractor infers
from prose — and what each edge is anchored to, so a graph walk that crosses an amendment lands on
citable text rather than on a document-shaped node with nothing behind it.

The resolution rule is pure, so it is tested without a store: that is where the decisions live —
which chunk proves a relation, and what to do with a relation that nothing proves.
"""

from __future__ import annotations

from rag_core.graph import RELATION_TYPES, DeclaredRelation, relation_edges
from rag_schemas import Chunk

_QUOTE = "The MAAT requirement is revised from Rs. 35 Crores to Rs. 19.44 Crores."


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="corrigendum-v", tenant_id="aegcl", text=text)


def _relation(**overrides: str) -> DeclaredRelation:
    base = {
        "rel_type": "amends",
        "source_ref": "Corrigendum-V",
        "target_ref": "Corrigendum-II",
        "evidence_quote": _QUOTE,
    }
    return DeclaredRelation(**{**base, **overrides})


def test_an_edge_is_anchored_to_the_chunk_that_proves_it() -> None:
    chunks = [
        _chunk("c1", "1. Scope. This corrigendum applies to the SOC/NOC tender."),
        _chunk("c2", f"7. Financial qualification. {_QUOTE} Bidders may revise their bids."),
    ]
    seed = relation_edges([_relation()], chunks)

    assert seed.edges == [("c2", "Corrigendum-V", "Corrigendum-II", "amends")]
    assert sorted(seed.entities) == ["Corrigendum-II", "Corrigendum-V"]
    assert not seed.unresolved and not seed.unknown_types


def test_a_relation_nothing_proves_is_dropped_and_reported() -> None:
    """Anchoring it to an arbitrary chunk would be worse than no edge: an answer that followed the
    edge would cite text that does not say what the edge claims."""
    seed = relation_edges([_relation()], [_chunk("c1", "Unrelated prose about delivery schedules.")])

    assert seed.edges == []
    assert seed.unresolved and _QUOTE[:40] in seed.unresolved[0]


def test_an_unrecognised_relation_type_is_refused() -> None:
    """`linked_chunks` walks edges whatever their label, so an unrecognised one would become a real
    graph edge that nothing can interpret. The four the plan's own prompt can emit are the set."""
    chunks = [_chunk("c1", _QUOTE)]
    assert relation_edges([_relation(rel_type="rescinds")], chunks).edges == []
    assert relation_edges([_relation(rel_type="rescinds")], chunks).unknown_types == ["rescinds"]
    for known in sorted(RELATION_TYPES):
        assert relation_edges([_relation(rel_type=known)], chunks).edges, known
    # Case and padding come from a model's JSON, not from a schema.
    assert relation_edges([_relation(rel_type=" Amends ")], chunks).edges[0][3] == "amends"


def test_a_quote_matches_across_whitespace_and_case() -> None:
    """A quote's line breaks never survive extraction identically — the plan's own Phase-4 rule is
    "a substring after whitespace normalisation" — and a model asked for verbatim text routinely
    returns it recapitalised. Refusing those would drop real evidence over cosmetics."""
    chunks = [
        _chunk(
            "c1",
            "7. Financial qualification. The MAAT requirement is revised\n  from Rs. 35\tCrores to Rs. 19.44 Crores.",
        )
    ]
    quoted = _relation(evidence_quote="the maat requirement is revised from Rs. 35 Crores to Rs. 19.44 Crores.")
    assert relation_edges([quoted], chunks).edges == [("c1", "Corrigendum-V", "Corrigendum-II", "amends")]


def test_the_first_chunk_in_document_order_wins() -> None:
    """Boilerplate repeats, so a quote can match more than once. Whichever it picks must not depend
    on dict ordering or on which store returned the chunks."""
    repeated = [_chunk("c1", _QUOTE), _chunk("c2", _QUOTE), _chunk("c3", _QUOTE)]
    assert relation_edges([_relation()], repeated).edges[0][0] == "c1"
    assert relation_edges([_relation()], list(reversed(repeated))).edges[0][0] == "c3"


def test_a_relation_missing_an_end_is_not_an_edge() -> None:
    chunks = [_chunk("c1", _QUOTE)]
    assert relation_edges([_relation(target_ref="")], chunks).edges == []
    assert relation_edges([_relation(source_ref="   ")], chunks).edges == []


def test_an_empty_quote_anchors_nothing() -> None:
    """Phase 3 requires a quote for every relation. One that arrives without it must not match the
    first chunk by vacuous substring — `"" in text` is True for every text."""
    seed = relation_edges([_relation(evidence_quote="")], [_chunk("c1", _QUOTE)])
    assert seed.edges == []
    assert len(seed.unresolved) == 1


def test_a_section_level_amendment_is_the_same_shape() -> None:
    """Phase 4's `sections.amends` arrives with the section's breadcrumb as the source, so the edge
    links two SECTIONS rather than two documents — same resolution, finer grain."""
    chunks = [_chunk("c1", f"Clause 7.2. {_QUOTE}")]
    section = _relation(source_ref="Attachment-1 > 7. Financial qualification", target_ref="Clause 7.2 of NIT")
    [edge] = relation_edges([section], chunks).edges
    assert edge == ("c1", "Attachment-1 > 7. Financial qualification", "Clause 7.2 of NIT", "amends")


def test_no_relations_means_no_work() -> None:
    """The state of every document in this build: Phases 3-4 have not landed, so nothing declares
    anything and the seeding writes nothing."""
    seed = relation_edges([], [_chunk("c1", "anything")])
    assert seed.edges == [] and seed.entities == [] and seed.unresolved == []
