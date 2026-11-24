"""ParentChildChunker: small atomic-aware children + a per-section parent context bank (FR-C2).

Pins the storage-model invariants Slice 3 depends on: atomic units never split, an over-cap table
is ONE child, children carry parent_id, parents are marked chunk_role=parent, and — when there is
no atomic unit — the children are byte-identical to the clause chunker at the same cap.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rag_core.chunking import ClauseChunker, get_chunker
from rag_core.parent_child import (
    META_CHUNK_ROLE,
    META_PARENT_ID,
    PARENT_ROLE,
    ParentChildChunker,
    is_parent,
    not_parent,
)
from rag_schemas import DocumentEnvelope
from rag_schemas.common import ChunkStrategy, SourceKind


def _doc() -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id="t",
        doc_id="d",
        source_id="s",
        source_kind=SourceKind.UPLOAD,
        mime="text/plain",
        ingested_at=datetime(2026, 9, 18, tzinfo=UTC),
    )


_PROSE = (
    "1.0 Objective. This document lays down the procedure for numbering standard operating "
    "procedures across the site. It applies to every department without exception. "
    "2.0 Scope. The scope covers drafting, review, approval and issue of each SOP. "
    "Every controlled copy is tracked in the register maintained by Quality Assurance. "
    "3.0 Responsibility. The user department prepares the draft and marks it as a draft copy."
)


def test_children_are_byte_identical_to_clause_when_no_atomic_unit() -> None:
    """The 'never worse than clause' guarantee: with no table/code/formula the atomic-aware path
    is bypassed and children match ClauseChunker exactly at the same cap."""
    pc = ParentChildChunker(child_max_chars=200, overlap=40)
    children = not_parent(pc.chunk(_doc(), _PROSE))
    clause = ClauseChunker(max_chars=200, overlap=40).chunk(_doc(), _PROSE)
    assert [c.text for c in children] == [c.text for c in clause]
    assert [c.span for c in children] == [c.span for c in clause]
    assert [c.chunk_id for c in children] == [c.chunk_id for c in clause]


def test_no_parents_when_no_atomic_and_short_prose_stays_one_child() -> None:
    """A single short section that fits one child produces no parent (a lone child is its own
    context), so nothing is marked chunk_role=parent."""
    text = "1.0 Objective. Keep it short and singular."
    chunks = ParentChildChunker(child_max_chars=350).chunk(_doc(), text)
    assert all(not is_parent(c) for c in chunks)


def test_a_table_is_never_split_and_stays_one_child() -> None:
    header = "| Department | Code |"
    rows = "\n".join(f"| Dept {i} | {chr(65 + i)} |" for i in range(6))
    text = f"5.7 Department codes follow.\n{header}\n| --- | --- |\n{rows}\nEnd of section."
    children = not_parent(ParentChildChunker(child_max_chars=120).chunk(_doc(), text))
    holding = [c for c in children if "Code" in c.text or "Dept 5" in c.text]
    assert holding, "the table vanished"
    # The whole table (header + separator + every row) lives in ONE oversized child, over the cap.
    table_child = next(c for c in children if header in c.text)
    assert len(table_child.text) > 120, "the table was split at the cap"
    for i in range(6):
        assert f"| Dept {i} | {chr(65 + i)} |" in table_child.text


def test_a_code_fence_is_never_split() -> None:
    text = "Run:\n```python\nfor x in range(10):\n    print(x)\n    do_more(x)\n```\nDone."
    children = not_parent(ParentChildChunker(child_max_chars=30).chunk(_doc(), text))
    fenced = next(c for c in children if "```python" in c.text)
    assert "for x in range(10):" in fenced.text and "do_more(x)" in fenced.text and "```" in fenced.text


def test_children_carry_parent_id_and_parents_are_marked() -> None:
    """A multi-child section produces a parent; every child in it points at that parent's id, and
    the parent id equals a real parent record's chunk_id."""
    header = "| Metric | Value |"
    rows = "\n".join(f"| Metric {i} | {i * 100} |" for i in range(4))
    # Enough prose + a table so the section yields >=2 children and a parent groups them.
    text = f"2.0 Metrics section.\n{_PROSE}\n{header}\n| --- | --- |\n{rows}\n"
    all_chunks = ParentChildChunker(child_max_chars=180, parent_max_chars=4000).chunk(_doc(), text)
    parents = [c for c in all_chunks if is_parent(c)]
    children = not_parent(all_chunks)
    assert parents, "no parent was built for a multi-child section"
    parent_ids = {p.chunk_id for p in parents}
    with_parent = [c for c in children if c.metadata.get(META_PARENT_ID)]
    assert with_parent, "no child was stamped with a parent_id"
    for c in with_parent:
        assert c.metadata[META_PARENT_ID] in parent_ids
    for p in parents:
        assert p.metadata[META_CHUNK_ROLE] == PARENT_ROLE


def test_parent_text_contains_its_children_and_parent_id_is_content_derived() -> None:
    header = "| Metric | Value |"
    rows = "\n".join(f"| Metric {i} | {i * 100} |" for i in range(4))
    text = f"2.0 Metrics section.\n{_PROSE}\n{header}\n| --- | --- |\n{rows}\n"
    all_chunks = ParentChildChunker(child_max_chars=180, parent_max_chars=4000).chunk(_doc(), text)
    parents = [c for c in all_chunks if is_parent(c)]
    children = not_parent(all_chunks)
    for p in parents:
        member_children = [c for c in children if c.metadata.get(META_PARENT_ID) == p.chunk_id]
        for c in member_children:
            assert c.text in p.text, "parent text must contain each of its children"
    # Content-derived + stable: re-chunking the same text yields the same parent id.
    again = ParentChildChunker(child_max_chars=180, parent_max_chars=4000).chunk(_doc(), text)
    assert {p.chunk_id for p in parents} == {c.chunk_id for c in again if is_parent(c)}


def test_no_parent_child_id_collision() -> None:
    """A parent id must never equal a child id (they share the catalog keyspace)."""
    text = f"2.0 Metrics.\n{_PROSE}\n| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |\n"
    all_chunks = ParentChildChunker(child_max_chars=150, parent_max_chars=4000).chunk(_doc(), text)
    child_ids = {c.chunk_id for c in all_chunks if not is_parent(c)}
    parent_ids = {c.chunk_id for c in all_chunks if is_parent(c)}
    assert not (child_ids & parent_ids)


def test_get_chunker_returns_parent_child() -> None:
    chunker = get_chunker(ChunkStrategy.PARENT_CHILD, child_max_chars=300, parent_max_chars=1500)
    assert isinstance(chunker, ParentChildChunker)
    assert chunker.child_max_chars == 300
    assert chunker.parent_max_chars == 1500
