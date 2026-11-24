"""RAG v2 Phase 2, slice B2: a parsed layout tree, flattened into ``sections`` / ``blocks`` rows.

These are the ROW-SHAPE tests — what the nesting becomes once it is flat, what stays NULL, and
that the write path REPLACES a document's structure rather than appending to it. They run against
the in-memory twin, which holds the same rows the Postgres adapter inserts (both go through
:func:`rag_clients.layout_rows.flatten_layout_tree`, so there is one row shape, not two). The
live round trip against a real Postgres — including that ``delete_document`` empties both tables —
is ``tests/integration/test_v2_structure_roundtrip.py``.
"""

from __future__ import annotations

from rag_clients.layout_rows import BlockRow, SectionRow, flatten_layout_tree
from rag_clients.memory import InMemoryCatalog
from rag_core.layout import SUPPORTED_SCHEMA_VERSIONS, LayoutTree, parse_layout

_DOC = "leave-policy"
_TENANT = "structkb"

# The overlay version this build's PARSER accepts, read off the parser instead of hardcoded: these
# tests are about the ROWS, not about the version gate (which is pinned where it belongs, in
# `services/control-plane/tests/test_layout_structure_persist.py`), so they must not start failing
# the day that number is bumped.
_VERSION = max(SUPPORTED_SCHEMA_VERSIONS)

# Two pages of a real-shaped policy document: a heading hierarchy (h1 > h2, h2), a table with a
# declared header, and that table continuing across the page break with no header of its own.
_PAGE_1 = (
    "<h1>Leave policy</h1><p>Applies to all staff.</p>"
    "<h2>Earned leave</h2><p>Eighteen days a year.</p>"
    "<table><tr><th>Grade</th><th>Days</th></tr><tr><td>M1</td><td>18</td></tr></table>"
)
_PAGE_2 = (
    "<table><tr><td>M2</td><td>21</td></tr><tr><td>M3</td><td>24</td></tr></table>"
    "<h2>Sick leave</h2><p>Twelve days a year.</p>"
)
# Two-level numbering, because `sections.detect_sections` refuses single-level numbering outright
# ("1." is far more often a list item than a heading) — see `rag_core.sections`.
_TEXT = "1.1 Earned leave\nEighteen days a year.\n1.1.1 Accrual\nOne and a half days a month.\n"


def _overlay(*pages: str) -> dict[str, object]:
    return {
        "schema_version": _VERSION,
        "extractor": "chandra",
        "pages": [
            {"page_no": i, "markup": markup, "markup_format": "html", "degradations": []}
            for i, markup in enumerate(pages, start=1)
        ],
    }


def _tree(*pages: str, text: str = "") -> LayoutTree:
    return parse_layout(_overlay(*pages), _DOC, text=text)


def _rows(tree: LayoutTree) -> tuple[list[SectionRow], list[BlockRow]]:
    return flatten_layout_tree(tree)


def _by_id(sections: list[SectionRow]) -> dict[str, SectionRow]:
    return {s.section_id: s for s in sections}


# --- the nesting becomes parent links ------------------------------------------------------


def test_the_section_hierarchy_becomes_parent_section_id() -> None:
    """`LayoutSection.children` is the only place the tree records nesting; the table has one
    column for it. A lost parent link is a lost breadcrumb for every block beneath it."""
    sections, _ = _rows(_tree(_PAGE_1, _PAGE_2))

    root = next(s for s in sections if s.depth == 0)
    assert root.parent_section_id is None, "the top of the tree has no parent"
    assert root.heading == "Leave policy"
    children = [s for s in sections if s.parent_section_id == root.section_id]
    assert [s.heading for s in children] == ["Earned leave", "Sick leave"]
    assert {s.depth for s in children} == {1}
    # Every parent link resolves to a row in the same batch — a dangling one would be a section
    # whose breadcrumb cannot be rebuilt.
    known = _by_id(sections)
    assert all(s.parent_section_id in known for s in sections if s.parent_section_id is not None)


def test_the_rows_are_in_pre_order_so_a_parent_always_precedes_its_children() -> None:
    """There are no FKs (see `postgres_catalog._v2_table_ddl`), so insertion order is not enforced
    by the database — which is exactly why a reader rebuilding the tree in one pass needs it."""
    sections, _ = _rows(_tree(_PAGE_1, _PAGE_2))
    seen: set[str] = set()
    for row in sections:
        if row.parent_section_id is not None:
            assert row.parent_section_id in seen
        seen.add(row.section_id)


def test_a_table_row_keeps_the_table_it_came_from() -> None:
    """`blocks.parent_block_id` for the table -> table_row relation. Without it a row is
    addressable but orphaned, and adjacency cannot recover it once a merged table interleaves
    rows from two pages."""
    _, blocks = _rows(_tree(_PAGE_1, _PAGE_2))

    table = next(b for b in blocks if b.block_type == "table")
    rows = [b for b in blocks if b.block_type == "table_row"]
    assert rows, "a table with data rows must produce row blocks"
    assert {r.parent_block_id for r in rows} == {table.block_id}
    assert table.parent_block_id is None, "the table itself has no owning block"
    # Every block hangs off a section that is in the same batch.
    sections, _ = _rows(_tree(_PAGE_1, _PAGE_2))
    assert {b.section_id for b in blocks} <= set(_by_id(sections))


def test_a_blocks_section_is_the_section_that_owns_it_not_the_document_root() -> None:
    sections, blocks = _rows(_tree(_PAGE_1, _PAGE_2))
    owner = {s.section_id: s.heading for s in sections}
    assert owner[next(b for b in blocks if b.text == "Twelve days a year.").section_id] == "Sick leave"
    assert owner[next(b for b in blocks if b.block_type == "table").section_id] == "Earned leave"


# --- pages ---------------------------------------------------------------------------------


def test_a_merged_cross_page_tables_span_is_not_lost_to_the_single_page_column() -> None:
    """`blocks.page` is ONE column and a merged table covers two pages. The table row holds where
    the table STARTS; the span survives in the per-row blocks (each keeps the page it sat on) and
    in the owning section's page_start/page_end."""
    sections, blocks = _rows(_tree(_PAGE_1, _PAGE_2))

    table = next(b for b in blocks if b.block_type == "table")
    assert table.page == 1, "the merged table is stamped with the page it starts on"
    rows = [b for b in blocks if b.parent_block_id == table.block_id]
    assert {r.page for r in rows} == {1, 2}, "the merge must not collapse its rows onto one page"
    assert "M3 | 24" in (table.text or ""), "the continuation's rows are part of the merged table"

    earned = next(s for s in sections if s.heading == "Earned leave")
    assert (earned.page_start, earned.page_end) == (1, 2), "the section spans the table it owns"


def test_a_sections_page_span_covers_its_own_blocks_and_its_heading() -> None:
    sections, _ = _rows(_tree(_PAGE_1, _PAGE_2))
    sick = next(s for s in sections if s.heading == "Sick leave")
    assert (sick.page_start, sick.page_end) == (2, 2)


# --- what stays NULL -----------------------------------------------------------------------


def test_the_later_phase_columns_are_absent_from_the_row_so_they_insert_as_null() -> None:
    """`summary`, `key_values`, `facet_ids` and `amends` are Phases 3-4; `bbox` is permanently
    absent (a chat-shaped VLM returns no coordinates); `token_count` is NULL rather than estimated.
    Asserted on the row FIELDS, so adding one by accident fails here rather than silently writing
    a placeholder a later phase would read as derived."""
    deferred = {"summary", "key_values", "facet_ids", "amends", "token_count"}
    assert deferred.isdisjoint(SectionRow.__dataclass_fields__)
    assert "bbox" not in BlockRow.__dataclass_fields__


# --- the plain-text fallback ----------------------------------------------------------------


def test_a_document_with_no_markup_still_produces_sections_and_no_blocks() -> None:
    """The parser falls back to `sections.detect_sections` over the text. The tree is persisted —
    only the Chandra engines emit markup, so markup-only structure would leave most of the corpus
    out — and "zero blocks" is how a reader tells a text-derived tree from a markup one without a
    column of its own."""
    tree = _tree(text=_TEXT)
    assert tree.source == "text"
    sections, blocks = _rows(tree)

    assert blocks == []
    assert [s.heading for s in sections] == ["1.1 Earned leave", "1.1.1 Accrual"]
    assert [s.depth for s in sections] == [0, 1]
    assert sections[0].parent_section_id is None
    assert sections[1].parent_section_id == sections[0].section_id, "nesting survives on this path too"
    assert all(s.page_start is None and s.page_end is None for s in sections), "the text layer has no pages"


def test_a_document_with_neither_markup_nor_headings_still_gets_one_section() -> None:
    sections, blocks = _rows(_tree(text="Just prose, no headings at all."))
    assert len(sections) == 1
    assert sections[0].heading is None, "an implicit root has no heading, and '' would claim an empty one"
    assert sections[0].breadcrumb == ""
    assert blocks == []


# --- the twin's write path: REPLACE, never append -------------------------------------------


async def test_the_write_path_replaces_rather_than_appends() -> None:
    """The contract the reindex path depends on. An append would double a document's tree on every
    replay, and the doubled half would describe text that is no longer being served."""
    catalog = InMemoryCatalog()
    tree = _tree(_PAGE_1, _PAGE_2)
    expected = _rows(tree)

    first = await catalog.replace_document_structure(_TENANT, tree)
    second = await catalog.replace_document_structure(_TENANT, tree)

    assert first == second == (len(expected[0]), len(expected[1]))
    assert catalog.document_structure(_TENANT, _DOC) == expected


async def test_a_re_extraction_that_produced_a_different_tree_leaves_none_of_the_old_one() -> None:
    catalog = InMemoryCatalog()
    await catalog.replace_document_structure(_TENANT, _tree(_PAGE_1, _PAGE_2))
    revised = _tree("<h1>Leave policy (revised)</h1><p>Now twenty days a year.</p>")

    await catalog.replace_document_structure(_TENANT, revised)

    sections, blocks = catalog.document_structure(_TENANT, _DOC)
    assert [s.heading for s in sections] == ["Leave policy (revised)"]
    assert [b.text for b in blocks] == ["Leave policy (revised)", "Now twenty days a year."]


async def test_an_empty_tree_purges_the_stored_structure() -> None:
    """A re-extraction that produced no structure at all must be able to say so — otherwise the
    previous tree outlives the text it described."""
    catalog = InMemoryCatalog()
    await catalog.replace_document_structure(_TENANT, _tree(_PAGE_1))

    await catalog.replace_document_structure(_TENANT, LayoutTree(doc_id=_DOC, sections=(), source="markup"))

    assert catalog.document_structure(_TENANT, _DOC) == ([], [])


async def test_the_write_path_is_safe_on_a_document_with_no_stored_structure_yet() -> None:
    catalog = InMemoryCatalog()
    assert catalog.document_structure(_TENANT, "never-written") == ([], [])
    assert await catalog.replace_document_structure(_TENANT, _tree(_PAGE_1)) == (2, 6)


async def test_one_documents_structure_is_not_touched_by_another_documents_replace() -> None:
    catalog = InMemoryCatalog()
    other = parse_layout(_overlay(_PAGE_1), "other-doc")
    await catalog.replace_document_structure(_TENANT, _tree(_PAGE_1))
    await catalog.replace_document_structure(_TENANT, other)

    await catalog.replace_document_structure(_TENANT, LayoutTree(doc_id=_DOC, sections=(), source="markup"))

    assert catalog.document_structure(_TENANT, _DOC) == ([], [])
    assert catalog.document_structure(_TENANT, "other-doc")[0], "the other document still has its tree"
