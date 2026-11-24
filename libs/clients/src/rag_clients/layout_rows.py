"""The ``sections`` / ``blocks`` row shape, and the flattening of a parsed layout tree into it.

RAG v2 Phase 2, slice B2. :func:`rag_core.layout.parse_layout` returns a NESTED
``LayoutTree``; the Phase-1 tables are FLAT, with the nesting carried by
``sections.parent_section_id`` / ``blocks.section_id`` / ``blocks.parent_block_id``. This module
is the one place that translation lives, so the Postgres adapter and the in-memory twin cannot
drift into two different row shapes — which would make a unit test on the twin evidence about
nothing.

It is in ``rag_clients`` and not ``rag_core`` because the row shape is the Postgres table's
shape: rag-core knows nothing about catalog columns, and the tree it produces is the contract
between the two.

WHAT IS DELIBERATELY NOT MAPPED (the columns stay NULL):

* ``summary``, ``key_values``, ``facet_ids``, ``amends`` — Phases 3-4 derive these from a model.
  Nothing on this path has them, so they are absent from the row rather than written as empty.
* ``bbox`` — permanently NULL on this path. The deployed extractor is a chat-shaped VLM and
  returns no coordinates (see :mod:`rag_core.layout`'s header), so there is nothing to map.
* ``token_count`` — NULL, not estimated. A real count needs the embedder's tokenizer, which only
  the indexing service holds (``Chunker.chunk``'s ``token_offsets``); a characters/4 guess would
  be indistinguishable in the column from a measured one, and Phase 5's chunk budgeting is
  exactly the consumer that must not silently trust a guess. NULL says "not counted", which is
  true.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_core.layout import LayoutSection, LayoutTree


@dataclass(frozen=True, slots=True)
class SectionRow:
    """One ``sections`` row. Field names ARE the column names."""

    section_id: str
    doc_id: str
    parent_section_id: str | None
    depth: int
    ordinal: int
    # The section's own heading text, or None for an implicit root (a document whose markup opened
    # with content before any <hN>). None rather than "" because the column is nullable and the
    # section genuinely has no heading — "" would read as a heading that was empty.
    heading: str | None
    # Stored in the NORMALISED form that was hashed into `section_id`, so the column and the id can
    # never disagree (`LayoutSection.breadcrumb` already is that form).
    breadcrumb: str
    page_start: int | None
    page_end: int | None
    text_hash: str


@dataclass(frozen=True, slots=True)
class BlockRow:
    """One ``blocks`` row. Field names ARE the column names."""

    block_id: str
    section_id: str
    doc_id: str
    block_type: str
    page: int | None
    ordinal: int
    text: str
    html: str | None
    parent_block_id: str | None


def flatten_layout_tree(tree: LayoutTree) -> tuple[list[SectionRow], list[BlockRow]]:
    """A parsed tree as ``(sections, blocks)`` rows, in pre-order.

    Pre-order so a parent row always precedes its children: nothing depends on insertion order
    today (there are no FKs — see ``postgres_catalog._v2_table_ddl``), but a reader that rebuilds
    the tree by walking the list in order can then do it in one pass.

    A document with no markup yields sections and NO blocks — the plain-text fallback has no block
    stream (see :func:`rag_core.layout.parse_layout`). That absence is also how a reader tells
    the two apart from the rows alone: a markup-derived tree always has blocks."""
    sections: list[SectionRow] = []
    blocks: list[BlockRow] = []
    _walk(tree.doc_id, tree.sections, None, sections, blocks)
    return sections, blocks


def _walk(
    doc_id: str,
    nodes: tuple[LayoutSection, ...],
    parent: str | None,
    sections: list[SectionRow],
    blocks: list[BlockRow],
) -> None:
    for node in nodes:
        start, end = _page_span(node)
        sections.append(
            SectionRow(
                section_id=node.section_id,
                doc_id=doc_id,
                parent_section_id=parent,
                depth=node.depth,
                ordinal=node.ordinal,
                heading=node.title or None,
                breadcrumb=node.breadcrumb,
                page_start=start,
                page_end=end,
                text_hash=node.text_hash,
            )
        )
        for block in node.blocks:
            blocks.append(
                BlockRow(
                    block_id=block.block_id,
                    section_id=node.section_id,
                    doc_id=doc_id,
                    block_type=block.block_type,
                    # A MERGED cross-page table is one block with a single `page` column: it holds
                    # where the table STARTS, matching `LayoutBlock.page`'s own meaning. The span is
                    # not lost — every data row of the table is its own `table_row` block carrying
                    # the page IT sat on (`_absorb` keeps each row's page), and this section's
                    # `page_end` below is taken from the union of its blocks' `pages`, so the
                    # table's last page is inside its section's span. Storing the first page in a
                    # column named `page` and recovering the rest from the rows beats inventing a
                    # `page_end` on `blocks` that only one block type would ever use.
                    page=block.page,
                    ordinal=block.ordinal,
                    text=block.text,
                    html=block.html,
                    parent_block_id=block.parent_block_id,
                )
            )
        _walk(doc_id, node.children, node.section_id, sections, blocks)


def _page_span(section: LayoutSection) -> tuple[int | None, int | None]:
    """``(page_start, page_end)`` from the pages this section's OWN blocks occupy, plus the page its
    heading sat on.

    Own blocks only — NOT the subtree. A parent's span is therefore the extent of the content it
    directly holds, and a consumer that wants the subtree's extent takes the min/max over the
    descendants it already reaches through ``parent_section_id``. The other way round is lossy:
    folding children in would make a parent's span indistinguishable from its own content's, and an
    h1 that owns nothing but sub-headings would claim pages it has no text on.

    Both NULL on the plain-text fallback path, which has no pages at all."""
    pages = {p for block in section.blocks for p in (block.pages or (block.page,))}
    if section.page is not None:
        pages.add(section.page)
    if not pages:
        return None, None
    return min(pages), max(pages)
