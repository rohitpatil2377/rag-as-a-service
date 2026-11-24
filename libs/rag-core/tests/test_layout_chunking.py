"""Layout-aware chunking: a structure tree -> the v2 levels (RAG v2 Phase 5).

The plan's acceptance criteria for this phase are "no chunk crosses a section boundary (test)" and
"row chunks contain header names". Both are pinned below, along with the parent links, the masking
seam, the atomic-unit guard and the byte-identity of the blockless path. The tree always comes
through the public `parse_layout` over a literal persisted overlay, so these tests exercise the
real Phase-2 output rather than a hand-built tree that could drift from it.

The shapes mirror `test_layout.py`'s: the AEGCL corrigendum the plan names is not in this
repository (discovery §7), so the fixtures encode the same shapes — a nested numbered breadcrumb, a
parameter table, a page-split table — at the smallest size that still exercises them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from rag_core.atomic_units import atomic_spans
from rag_core.chunking import (
    ClauseChunker,
    FixedChunker,
    block_id,
    get_chunker,
    section_id,
    structured_chunk_id,
    text_hash,
)
from rag_core.layout import (
    BLOCK_PARAGRAPH,
    BLOCK_TABLE,
    BREADCRUMB_SEP,
    LayoutBlock,
    LayoutSection,
    LayoutTree,
    parse_layout,
)
from rag_core.layout_chunking import (
    _DEFAULT_LAYOUT_MAX_CHARS,
    LayoutAwareChunker,
    _units,
    embed_prefix,
)
from rag_core.parent_child import ParentChildChunker
from rag_core.section_enrich import Proposition, SectionEnrichment
from rag_core.sections import PATH_SEP
from rag_schemas import Chunk, DocumentEnvelope
from rag_schemas.common import ChunkLevel, ChunkStrategy, SourceKind

_DOC_ID = "fixture-phase5-shapes.pdf"

# Three sections, each carrying a SENTINEL token that occurs nowhere else in the document. That is
# what makes "no chunk crosses a section boundary" testable as a fact rather than an impression: a
# chunk holding two sentinels has text from two sections, whatever its section_id claims.
_PAGE_1 = """
<h1>Attachment-1</h1>
<h2>13. Video Wall Technical Requirements</h2>
<h3>Standards and Compliance</h3>
<p>ALPHA. The controller shall comply with IEC 62443-4-2.</p>
<p>ALPHA. Certificates shall be furnished with the bid.</p>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Brightness</td><td>700 nits</td></tr>
  <tr><td>Contrast ratio</td><td>4000:1</td></tr>
</table>
<h3>Warranty and Support</h3>
<p>BETA. Five years comprehensive on-site warranty.</p>
<h3>Payment Terms</h3>
<p>GAMMA. Payment shall be released within 30 days.</p>
"""


def _overlay(*pages: tuple[int, str], markup_format: str = "html") -> dict[str, Any]:
    """The overlay exactly as it is persisted (schema v2) — that mapping is the contract."""
    return {
        "schema_version": 2,
        "extractor": "chandra",
        "content_hash": "",
        "pages": [
            {"page_no": no, "markup": markup, "markup_format": markup_format, "degradations": []}
            for no, markup in pages
        ],
    }


def _doc(doc_id: str = _DOC_ID, **metadata: str) -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id="phase5",
        doc_id=doc_id,
        source_id="s1",
        source_kind=SourceKind.UPLOAD,
        mime="application/pdf",
        uri=f"minio://raw/{doc_id}",
        ingested_at=datetime(2026, 10, 19, tzinfo=UTC),
        metadata=metadata,
    )


def _identity(text: str) -> str:
    """A no-op masker, for the tests that are about structure rather than masking. Production has
    no such thing — `LayoutAwareChunker` requires a masker and `get_chunker` refuses without one."""
    return text


def _chunker(
    tree: LayoutTree,
    *,
    mask: Any = _identity,
    max_chars: int = _DEFAULT_LAYOUT_MAX_CHARS,
    fallback: ClauseChunker | None = None,
    card: str = "",
    enrichment: dict[str, SectionEnrichment] | None = None,
) -> LayoutAwareChunker:
    return LayoutAwareChunker(
        tree=tree,
        mask=mask,
        fallback=fallback if fallback is not None else ClauseChunker(),
        enrichment=enrichment,
        max_chars=max_chars,
        card=card,
    )


def _tree(markup: str = _PAGE_1, *, doc_id: str = _DOC_ID) -> LayoutTree:
    return parse_layout(_overlay((1, markup)), doc_id)


def _tree_with_block(text: str) -> LayoutTree:
    """A one-section tree holding one paragraph block with ``text`` VERBATIM.

    Built directly rather than through ``parse_layout`` for one reason, pinned by its own test
    below: the HTML path collapses whitespace, so it cannot produce a block whose text contains a
    newline, and the atomic-unit guard reads lines."""
    sid = section_id("blocky", "Mixed", 0)
    block = LayoutBlock(
        block_id=block_id(sid, BLOCK_PARAGRAPH, 0, text_hash(text)),
        block_type=BLOCK_PARAGRAPH,
        text=text,
        page=1,
        ordinal=0,
        pages=(1,),
    )
    section = LayoutSection(
        section_id=sid,
        breadcrumb="Mixed",
        title="Mixed",
        depth=0,
        ordinal=0,
        text=text,
        text_hash=text_hash(text),
        heading_level=2,
        page=1,
        blocks=(block,),
    )
    return LayoutTree(doc_id="blocky", sections=(section,), source="markup")


def _walk(sections: tuple[LayoutSection, ...]) -> list[LayoutSection]:
    out: list[LayoutSection] = []
    for section in sections:
        out.append(section)
        out.extend(_walk(section.children))
    return out


def _of_level(chunks: list[Chunk], level: ChunkLevel) -> list[Chunk]:
    return [c for c in chunks if c.level is level]


# --- the plan's acceptance: no chunk crosses a section boundary ----------------------------------

_SENTINELS = ("ALPHA", "BETA", "GAMMA")


def test_no_chunk_crosses_a_section_boundary() -> None:
    """The plan's stated acceptance criterion, and the gap discovery found: today's chunkers CAN
    cross a section (FIXED freely, the clause path via its carry-over tail) and nothing forbade it.

    The sections here are small enough that any character-budget chunker would pack all three into
    one chunk — the 1600-char cap is far larger than the whole document — so this fails loudly if
    the implementation ever packs across sections."""
    chunks = _chunker(_tree()).chunk(_doc(), "")

    assert chunks, "the fixture must produce chunks or this asserts nothing"
    for chunk in chunks:
        present = [s for s in _SENTINELS if s in chunk.text]
        assert len(present) <= 1, f"{chunk.level} chunk holds text from {present}"


def test_every_chunks_text_comes_from_the_section_it_names() -> None:
    """The complement of the test above: not merely "one section's worth of text", but that
    section's. A chunk could hold one sentinel and still be labelled with the wrong section_id."""
    tree = _tree()
    by_id = {s.section_id: s for s in _walk(tree.sections)}
    chunks = _chunker(tree).chunk(_doc(), "")

    for chunk in chunks:
        assert chunk.section_id in by_id
        own = by_id[chunk.section_id]
        for sentinel in _SENTINELS:
            if sentinel in chunk.text:
                assert sentinel in own.text, f"{sentinel} is not in section {own.breadcrumb!r}"


def test_a_section_boundary_beats_the_size_budget() -> None:
    """Two adjacent one-line sections stay two chunks under a cap that would hold both ten times
    over. The budget may only ever make a chunk SMALLER than its section, never larger."""
    markup = "<h2>First</h2><p>One short line.</p><h2>Second</h2><p>Another short line.</p>"
    chunks = _of_level(_chunker(_tree(markup), max_chars=4000).chunk(_doc(), ""), ChunkLevel.CHUNK)

    assert len(chunks) == 2
    assert len({c.section_id for c in chunks}) == 2


# --- the plan's acceptance: row chunks contain header names --------------------------------------


def test_row_chunks_carry_their_header_names_and_their_breadcrumb() -> None:
    """Phase 2 linearises a row with its column names; this asserts the row LEVEL preserves that,
    because a bare "4000:1" supports no statement and the embedder cannot recover the column from
    anywhere else."""
    rows = _of_level(_chunker(_tree()).chunk(_doc(), ""), ChunkLevel.ROW)

    assert [r.text for r in rows] == [
        "Attachment-1"
        + BREADCRUMB_SEP
        + "13. Video Wall Technical Requirements"
        + BREADCRUMB_SEP
        + "Standards and Compliance"
        + BREADCRUMB_SEP
        + "Table: Parameter=Brightness | Value=700 nits",
        "Attachment-1"
        + BREADCRUMB_SEP
        + "13. Video Wall Technical Requirements"
        + BREADCRUMB_SEP
        + "Standards and Compliance"
        + BREADCRUMB_SEP
        + "Table: Parameter=Contrast ratio | Value=4000:1",
    ]


def test_a_row_is_never_split_however_wide_the_table() -> None:
    """A row is one block and is emitted whole even past the cap — half a row is unusable."""
    cells = "".join(f"<td>value-{i}</td>" for i in range(40))
    markup = f"<h2>Wide</h2><table><tr>{''.join(f'<th>col-{i}</th>' for i in range(40))}</tr><tr>{cells}</tr></table>"
    rows = _of_level(_chunker(_tree(markup), max_chars=200).chunk(_doc(), ""), ChunkLevel.ROW)

    assert len(rows) == 1
    assert len(rows[0].text) > 200
    assert "col-39=value-39" in rows[0].text


def test_a_row_that_inherited_its_header_across_a_page_break_keeps_it() -> None:
    """The cross-page merge is Phase 2's; this is the Phase-5 consequence — a continuation row on
    page 2 carries the page-1 header names into its own chunk."""
    head = (
        "<h2>Parameters</h2><table><tr><th>Parameter</th><th>Value</th></tr>"
        "<tr><td>Brightness</td><td>700</td></tr></table>"
    )
    tail = "<table><tr><td>Contrast</td><td>4000:1</td></tr></table>"
    rows = _of_level(_chunker(parse_layout(_overlay((1, head), (2, tail)), _DOC_ID)).chunk(_doc(), ""), ChunkLevel.ROW)

    assert any("Parameter=Contrast | Value=4000:1" in r.text for r in rows)
    assert [r.page for r in rows] == [1, 2]


# --- levels, parents, and whether the parent links resolve ---------------------------------------


def test_only_the_two_buildable_levels_are_emitted() -> None:
    """`doc` and `proposition` are DEFERRED, not stubbed: nothing may emit them until Phase 3/4
    gives them an input. `section` joined them — it is emittable, but every unit it produced was a
    verbatim duplicate of the `chunk`/`row` units beside it, so it waits for the Phase-4 summary
    (`LayoutAwareChunker._section_level` carries the measurement and the argument)."""
    chunks = _chunker(_tree()).chunk(_doc(), "")

    assert {c.level for c in chunks} == {ChunkLevel.CHUNK, ChunkLevel.ROW}
    assert all(c.level is not None for c in chunks)


def test_a_body_chunks_parent_is_its_section() -> None:
    tree = _tree()
    ids = {s.section_id for s in _walk(tree.sections)}
    for chunk in _of_level(_chunker(tree).chunk(_doc(), ""), ChunkLevel.CHUNK):
        assert chunk.parent_id == chunk.section_id
        assert chunk.parent_id in ids


def test_a_row_chunks_parent_is_the_table_block_it_came_from() -> None:
    """Not the section — the TABLE, so a retrieved row can be expanded to its whole table. The
    link resolves to a real block of the row's OWN section."""
    tree = _tree()
    chunks = _chunker(tree).chunk(_doc(), "")
    tables = {s.section_id: {b.block_id for b in s.blocks if b.block_type == BLOCK_TABLE} for s in _walk(tree.sections)}

    rows = _of_level(chunks, ChunkLevel.ROW)
    assert rows
    for row in rows:
        assert row.parent_id is not None
        assert row.parent_id in tables[row.section_id or ""]
        assert row.parent_id != row.section_id


def test_section_ids_are_the_trees_own_so_they_resolve_to_a_stored_section_row() -> None:
    """A chunk's section_id must be the id the structure tables stored, not one re-derived here —
    otherwise the join that the whole phase exists for silently matches nothing."""
    tree = _tree()
    expected = {s.section_id for s in _walk(tree.sections)}
    produced = {c.section_id for c in _chunker(tree).chunk(_doc(), "")}

    assert produced <= expected
    assert section_id(_DOC_ID, "Attachment-1", 0) in expected


def test_chunk_ids_are_derived_from_the_structure_and_are_stable_across_a_reparse() -> None:
    first = _chunker(_tree()).chunk(_doc(), "")
    second = _chunker(_tree()).chunk(_doc(), "")

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first), "ids must be unique within a document"
    sample = _of_level(first, ChunkLevel.ROW)[0]
    assert sample.chunk_id == structured_chunk_id(sample.section_id or "", "row", 0, text_hash(sample.text))


def test_the_level_is_part_of_the_id_so_two_levels_can_never_collide() -> None:
    """Identical text at two levels must not derive one id — it would overwrite itself on upsert in
    every store. Asserted on the id function directly, because the chunker no longer EMITS the same
    text at two levels (see the next test) and so cannot demonstrate the property."""
    sid, digest = "doc::s0123456789ab", text_hash("Only one paragraph here.")

    ids = {structured_chunk_id(sid, level, 0, digest) for level in ("section", "chunk", "row")}
    assert len(ids) == 3


def test_no_two_units_of_a_section_carry_the_same_text() -> None:
    """The section level must not re-emit a single body chunk's own bytes.

    This module's rule for keeping table text out of the body stream is that indexing the same
    evidence twice inflates the index and double-counts it at fusion time; the same rule has to bind
    this module. Before the guard, a one-paragraph section emitted `section` and `chunk` units with
    byte-identical `text` AND `embed_text` — two vectors and two BM25 docs for one sentence, so a
    `top_k=8` retrieval could be half-filled by duplicates and an answer could cite the same
    sentence twice. Measured on `_PAGE_1`: 12 units, 8 distinct texts, 4 duplicated pairs."""
    chunks = _chunker(_tree("<h2>Single</h2><p>Only one paragraph here.</p>")).chunk(_doc(), "")

    assert _of_level(chunks, ChunkLevel.SECTION) == [], "one body chunk => the section adds nothing"
    assert [c.text for c in chunks] == ["Single\nOnly one paragraph here."]

    whole = _chunker(_tree()).chunk(_doc(), "")
    texts = [c.text for c in whole]
    assert len(texts) == len(set(texts)), "no unit repeats another unit's text"
    embeds = [c.embed_text for c in whole]
    assert len(embeds) == len(set(embeds)), "...nor another unit's embedded string"


# --- the section level + propositions: Phase 4 restores both ------------------------------------


def test_the_section_level_is_absent_without_an_enrichment() -> None:
    """Phase 5 withdrew this level in review, and the argument still governs every section that
    has no Phase-4 summary. `LayoutSection.text` is the concatenation of the section's own blocks,
    so a section unit built from it either repeated a single body chunk's exact bytes, or — for
    the only shape that can hold two units while its own text still fits the cap, a table —
    contained every `row` unit's linearised text verbatim. Either way it indexed the same evidence
    twice. Content is not lost: `chunk` + `row` cover the section in full."""
    for markup in (
        "<h2>Single</h2><p>Only one paragraph here.</p>",
        "<h2>T</h2><p>Prose.</p><table><tr><th>A</th></tr><tr><td>1</td></tr></table>",
        "<h2>Long</h2><p>" + " ".join(f"Clause {i} in full." for i in range(200)) + "</p>",
    ):
        chunks = _chunker(_tree(markup)).chunk(_doc(), "")
        assert _of_level(chunks, ChunkLevel.SECTION) == [], markup[:40]
        assert chunks, "the section's content still reaches the index through the other levels"


def _enrich(tree: LayoutTree, breadcrumb: str, **kwargs: Any) -> dict[str, SectionEnrichment]:
    """An enrichment for the one section of ``tree`` whose breadcrumb ends with ``breadcrumb``."""
    from rag_core.layout_chunking import _walk

    for section in _walk(tree.sections):
        if section.breadcrumb.endswith(breadcrumb):
            return {section.section_id: SectionEnrichment(section_id=section.section_id, **kwargs)}
    raise AssertionError(f"no section ends with {breadcrumb!r}")


def test_an_enriched_section_emits_its_summary_as_the_section_unit() -> None:
    """Phase 4 restores the level the plan asks for: ONE retrievable unit per section, holding the
    SUMMARY rather than the section's own text. The summary is not a substring of anything else
    in the index, which is exactly what the Phase-5 stand-in could never say."""
    tree = _tree()
    summary = "This clause fixes the display parameters the controller must meet."
    chunks = _chunker(tree, enrichment=_enrich(tree, "Standards and Compliance", summary=summary)).chunk(_doc(), "")

    units = _of_level(chunks, ChunkLevel.SECTION)
    assert [c.text for c in units] == [summary]
    assert units[0].parent_id == units[0].section_id, "the plan's `parent_id = section_id`"
    # ...and ONLY that section. A per-section map must not leak onto its neighbours.
    assert len({c.section_id for c in units}) == 1


def test_an_enriched_section_emits_one_unit_per_proposition() -> None:
    """Each proposition is its own retrievable unit, self-contained and never split."""
    tree = _tree()
    props = (
        Proposition(text="The controller shall comply with IEC 62443-4-2.", quote="comply with IEC 62443-4-2"),
        Proposition(text="Certificates shall be furnished with the bid.", quote="Certificates shall be furnished"),
    )
    enrichment = _enrich(tree, "Standards and Compliance", summary="Standards.", propositions=props)
    chunks = _chunker(tree, enrichment=enrichment).chunk(_doc(), "")

    units = _of_level(chunks, ChunkLevel.PROPOSITION)
    assert [c.text for c in units] == [p.text for p in props]
    assert all(c.parent_id == c.section_id for c in units)
    # The QUOTE is not indexed beside the proposition — it is already in the section's own units.
    assert all("quote" not in c.text for c in units)


def test_the_new_levels_are_additive_and_ordered_after_the_section_summary() -> None:
    """With an enrichment the other levels are untouched, and `sequence` puts the aggregate first
    and the derived statements last within the section it came from."""
    tree = _tree()
    plain = _chunker(tree).chunk(_doc(), "")
    enrichment = _enrich(
        tree,
        "Standards and Compliance",
        summary="Standards.",
        propositions=(Proposition(text="A certificate is required.", quote="Certificates shall be furnished"),),
    )
    enriched = _chunker(tree, enrichment=enrichment).chunk(_doc(), "")

    before = [(c.level, c.text) for c in plain]
    after = [(c.level, c.text) for c in enriched if c.level in (ChunkLevel.CHUNK, ChunkLevel.ROW)]
    assert after == before, "chunk/row units are byte-identical to the unenriched run"

    sid = next(iter(enrichment))
    levels = [c.level for c in enriched if c.section_id == sid]
    assert levels[0] is ChunkLevel.SECTION
    assert levels[-1] is ChunkLevel.PROPOSITION
    assert [c.sequence for c in enriched] == list(range(len(enriched))), "document order is restamped"


def test_an_enrichment_naming_an_unknown_section_emits_nothing() -> None:
    """The map is keyed by `section_id` and looked up per section, so an entry for a section this
    tree does not have cannot attach itself to some other section."""
    tree = _tree()
    stray = {"not-a-section-id": SectionEnrichment(section_id="not-a-section-id", summary="Nowhere.")}
    assert _chunker(tree, enrichment=stray).chunk(_doc(), "") == _chunker(tree).chunk(_doc(), "")


def test_the_summary_and_propositions_are_masked_like_every_other_unit() -> None:
    """`_unit` is the single masking place and the new levels go through it too — a summary the
    model wrote from masked text is masked again rather than trusted to be clean."""
    tree = _tree()
    enrichment = _enrich(
        tree,
        "Standards and Compliance",
        summary="Contact a@b.com for certificates.",
        propositions=(Proposition(text="Write to a@b.com.", quote="Certificates shall be furnished"),),
    )
    chunks = _chunker(tree, mask=lambda t: t.replace("a@b.com", "[REDACTED:EMAIL]"), enrichment=enrichment).chunk(
        _doc(), ""
    )

    new = _of_level(chunks, ChunkLevel.SECTION) + _of_level(chunks, ChunkLevel.PROPOSITION)
    assert new, "the enriched levels were emitted"
    assert all("a@b.com" not in c.text and "a@b.com" not in (c.embed_text or "") for c in new)


# --- contextualize -------------------------------------------------------------------------------


def test_embed_text_carries_the_breadcrumb_and_text_does_not() -> None:
    """The plan's `embed_text` prefix, minus the Phase-3 profile slot. `text` stays what the
    catalog stores and a citation shows, so the prefix can never leak into an answer."""
    chunks = _chunker(_tree()).chunk(_doc(), "")

    for chunk in _of_level(chunks, ChunkLevel.CHUNK) + _of_level(chunks, ChunkLevel.SECTION):
        # The crumb comes from the TREE, not from `metadata["section"]`: that key holds the v1
        # separator (see the separator test below), so reading it back here made the assertion
        # self-consistent with `_unit` and blind to what it was supposed to catch.
        crumb = chunk.metadata["section"].replace(" > ", BREADCRUMB_SEP)
        assert chunk.embed_text == f"[{crumb}] {chunk.text}"
        assert not chunk.text.startswith("[")


def test_a_rows_breadcrumb_is_not_embedded_twice() -> None:
    """Phase 2's `_row_text` already opens a row with its breadcrumb, so prefixing it again gave
    "[A > B] A > B > Table: ...". On a wide table in a deep hierarchy that duplicate is a material
    share of the embedded string: it dilutes the cells that distinguish one row from the next and
    double-weights the crumb in both the vector and BM25 term frequency."""
    rows = _of_level(_chunker(_tree()).chunk(_doc(), ""), ChunkLevel.ROW)
    assert rows, "fixture must carry a table"

    for row in rows:
        assert row.embed_text == row.text
        assert row.text.count("Table:") == 1
        crumb = row.text.split(BREADCRUMB_SEP + "Table:")[0]
        assert row.text.count(crumb) == 1, "the breadcrumb appears once, not twice"


def test_the_profile_slot_is_empty_when_there_is_no_card() -> None:
    """Phase 5 left this slot deliberately empty rather than filling it with a placeholder, which
    would have been embedded into every vector in the corpus and then had to be re-embedded out of
    it. Phase 3 supplies the real card — and "" is still what a document without one gets, which
    is what keeps those documents byte-identical to Phase 5."""
    assert embed_prefix("A › B") == "[A › B] "  # noqa: RUF001 - the plan's separator
    assert embed_prefix("") == ""
    assert embed_prefix("A › B", "") == "[A › B] "  # noqa: RUF001


# --- the Phase-3 profile card in front of the breadcrumb ------------------------------------------

_CARD = "[corrigendum: Corrigendum-V — Amends the bidding document.] "


def test_the_card_precedes_the_breadcrumb_in_the_plans_order() -> None:
    """The plan's full prefix is "[{doc_type}: {title} — {purpose}] [{breadcrumb}] "."""
    assert embed_prefix("A › B", _CARD) == f"{_CARD}[A › B] "  # noqa: RUF001


def test_a_card_prefixes_every_body_chunk_and_never_its_citable_text() -> None:
    chunks = _of_level(_chunker(_tree(), card=_CARD).chunk(_doc(), ""), ChunkLevel.CHUNK)
    assert chunks, "fixture must produce body chunks"

    for chunk in chunks:
        crumb = chunk.metadata["section"].replace(" > ", BREADCRUMB_SEP)
        assert chunk.embed_text == f"{_CARD}[{crumb}] {chunk.text}"
        assert not chunk.text.startswith("[corrigendum")


def test_a_row_gets_the_card_even_though_it_does_not_get_the_breadcrumb() -> None:
    """The breadcrumb is withheld from a row because the row's own text already opens with it.
    The card is not in the row text and is not duplicated by it, so that argument does not reach
    it — and a row is exactly the unit whose bare cells most need to say which document they are
    from."""
    rows = _of_level(_chunker(_tree(), card=_CARD).chunk(_doc(), ""), ChunkLevel.ROW)
    assert rows, "fixture must carry a table"

    for row in rows:
        assert row.embed_text == _CARD + row.text
        assert row.text.count("Table:") == 1


def test_no_card_leaves_the_chunks_byte_identical_to_phase_five() -> None:
    without = _chunker(_tree()).chunk(_doc(), "")
    with_empty = _chunker(_tree(), card="").chunk(_doc(), "")
    assert [c.model_dump() for c in with_empty] == [c.model_dump() for c in without]


def test_the_card_is_subtracted_from_the_token_derived_budget() -> None:
    """The card rides in front of every unit's `embed_text`, so it spends part of the same token
    budget the character cap is derived from. Not subtracting it is how a document with a long
    card quietly pushes its largest chunks past an embedder that truncates SILENTLY (§8.10)."""
    tree = _tree()
    text = "x" * 4000
    offsets = list(range(0, 4000, 4))  # 4 chars/token, so the budget is token-derived, not max_chars

    plain = _chunker(tree)._effective_cap(text, offsets, max_tokens=100)
    carded = _chunker(tree, card=_CARD)._effective_cap(text, offsets, max_tokens=100)

    assert carded == plain - len(_CARD)


def test_the_card_does_not_shrink_a_pure_character_cap() -> None:
    """`max_chars` is a character target for the chunk's OWN text, which the card is not part of.
    A document with no token budget therefore chunks exactly as Phase 5 chunked it."""
    tree = _tree()
    assert _chunker(tree, card=_CARD)._effective_cap("abc", None, None) == _chunker(tree)._effective_cap(
        "abc", None, None
    )


def test_the_section_metadata_key_is_the_one_v1_already_uses() -> None:
    """So the metadata contextualizer, the D8 section filter and every saved query keep working on
    a v2 chunk without being taught anything."""
    chunk = _chunker(_tree()).chunk(_doc(category="tender"), "")[0]

    assert chunk.metadata["section"].startswith("Attachment-1")
    assert chunk.metadata["category"] == "tender", "document tags still flow onto chunks (FR-A7/B7)"


def test_the_section_metadata_value_uses_the_v1_separator_not_the_trees() -> None:
    """The v1 KEY is not enough — `filters` is an EXACT-match predicate, so the v1 SEPARATOR is
    part of the same contract. A stored filter `section = "Attachment-1 > 13. ..."` matched zero v2
    chunks while this key held the tree's " > " variant, and nothing failed: the query just
    returned nothing. The tree's own separator stays in `embed_text` and in the persisted
    `sections` row, which are the layout path's own namespaces."""
    chunks = _chunker(_tree()).chunk(_doc(), "")
    values = [c.metadata["section"] for c in chunks if "section" in c.metadata]
    assert values, "fixture is nested, so every unit carries a breadcrumb"

    assert all(PATH_SEP in v or " " not in v.strip() or BREADCRUMB_SEP not in v for v in values)
    assert not any(BREADCRUMB_SEP in v for v in values), "the tree separator must not reach this key"
    assert any(PATH_SEP in v for v in values), "a nested breadcrumb uses the v1 separator"


def test_a_table_with_no_row_units_still_reaches_the_index() -> None:
    """A `table` normally stays out of the body stream because every data row is its own unit. A
    table that yielded NO rows is the exception: in an OVERSIZED section (which gets no `section`
    unit either) its text reached no level at all and vanished silently. Reproduced on a
    header-only table — "Penalty Clause" appeared in zero chunks while ingest reported success."""
    body = " ".join(f"Clause {i} states a requirement in full." for i in range(200))
    markup = f"<h2>Long</h2><p>{body}</p><table><tr><th>Serial</th><th>Penalty Clause</th></tr></table>"
    chunks = _chunker(_tree(markup)).chunk(_doc(), "")

    assert _of_level(chunks, ChunkLevel.SECTION) == [], "section is oversized, so no section unit"
    assert _of_level(chunks, ChunkLevel.ROW) == [], "header-only table yields no data rows"
    assert any("Penalty Clause" in c.text for c in chunks)


def test_a_table_whose_rows_are_units_is_not_also_in_the_body_stream() -> None:
    """The exception above must stay an exception, or every cell is indexed twice."""
    chunks = _chunker(_tree()).chunk(_doc(), "")
    rows = _of_level(chunks, ChunkLevel.ROW)
    assert rows, "fixture must carry a table with data rows"

    for body in _of_level(chunks, ChunkLevel.CHUNK):
        assert "Table:" not in body.text


def test_a_unit_emptied_by_the_masker_is_dropped_not_indexed() -> None:
    """`Chunk.text` has no `min_length`, so an empty unit would be embedded (an empty vector) and
    BM25-indexed, matching nothing and diluting fusion. `RegexPIIMasker` cannot produce this today,
    but the masker is an injected seam and a Presidio/NER backend or a drop-strategy can."""
    chunks = _chunker(_tree(), mask=lambda _: "   ").chunk(_doc(), "")

    assert chunks == []


def test_a_chunker_cannot_be_constructed_without_a_callable_masker() -> None:
    """The class and module docstrings both claim this. Until the check existed only `get_chunker`
    enforced it, so a direct `mask=None` failed later inside `_unit` — after the caller believed it
    held a chunker. This is the FR-G2 masking invariant, so it fails where it still can safely."""
    with pytest.raises(TypeError, match="callable"):
        LayoutAwareChunker(tree=_tree(), mask=None, fallback=ClauseChunker())  # type: ignore[arg-type]


# --- the masking invariant -----------------------------------------------------------------------

_EMAIL = "blake@acme.example"


def _redact(text: str) -> str:
    """Stands in for `RegexPIIMasker` — the real masker is a service type and rag-core must not
    import one. The pipeline test uses the real one end to end."""
    return text.replace(_EMAIL, "[REDACTED:EMAIL]")


def test_every_string_that_leaves_this_chunker_is_masked() -> None:
    """`LayoutBlock.text` is captured PRE-masking (the overlay comes from the unmasked engine
    output), so text derived from it must go through the masker before it becomes a Chunk —
    `Chunk.text`'s own field description promises "PII already masked, FR-G2"."""
    markup = f"<h2>Contact {_EMAIL}</h2><p>Escalate to {_EMAIL} within 24 hours.</p>"
    markup += f"<table><tr><th>Owner</th></tr><tr><td>{_EMAIL}</td></tr></table>"
    chunks = _chunker(_tree(markup), mask=_redact).chunk(_doc(), "")

    assert chunks
    for chunk in chunks:
        assert _EMAIL not in chunk.text
        assert _EMAIL not in (chunk.embed_text or "")
        assert _EMAIL not in chunk.metadata.get("section", "")
    assert any(c.level is ChunkLevel.ROW for c in chunks), "the row level must be covered too"


def test_masking_does_not_move_the_structure_ids() -> None:
    """The ids are IDENTITY and must equal the rows the structure tables stored, which are derived
    from the unmasked text by the Phase-1 helpers. Masking them would break the join."""
    markup = f"<h2>Contact {_EMAIL}</h2><p>Escalate to {_EMAIL}.</p>"
    tree = _tree(markup)
    chunks = _chunker(tree, mask=_redact).chunk(_doc(), "")

    stored = {s.section_id for s in _walk(tree.sections)}
    assert {c.section_id for c in chunks} <= stored
    assert any(_EMAIL in s.breadcrumb for s in _walk(tree.sections)), "the tree itself is unmasked"


def test_a_layout_aware_chunker_cannot_be_built_without_a_masker_or_a_tree() -> None:
    """Fail closed. A missing masker would put unmasked text into the index and a missing tree
    would silently produce nothing — neither may degrade quietly."""
    with pytest.raises(ValueError, match="layout tree and a PII masker"):
        get_chunker(ChunkStrategy.LAYOUT_AWARE, layout=_tree(), mask=None)
    with pytest.raises(ValueError, match="layout tree and a PII masker"):
        get_chunker(ChunkStrategy.LAYOUT_AWARE, layout=None, mask=_identity)


# --- atomic units: never split mid-row, mid-fence or mid-formula ---------------------------------


def test_the_splitter_keeps_a_table_run_and_a_code_fence_whole() -> None:
    """The guard itself, on the one function that does the cutting. `atomic_units` is REUSED rather
    than reimplemented, and these are its signals: a run of pipe lines and a matched fence pair
    each come out as ONE piece even though the text is far over the cap."""
    rows = "\n".join(f"| Row {i} | Value {i} |" for i in range(6))
    fence = "```\n" + "\n".join(f"line_{i} = {i}" for i in range(12)) + "\n```"
    prose = " ".join(f"Sentence {i} of the lead-in." for i in range(10))

    units = _units(f"{prose}\n{rows}\n{fence}\n{prose}", 120)

    holding_rows = [u for u in units if "| Row 0 |" in u]
    assert len(holding_rows) == 1
    assert all(f"| Row {i} | Value {i} |" in holding_rows[0] for i in range(6))
    holding_fence = [u for u in units if "line_0 = 0" in u]
    assert len(holding_fence) == 1
    assert "line_11 = 11" in holding_fence[0]
    assert any(len(u) > 120 for u in units), "an atomic unit is emitted whole even when oversized"


def test_a_block_that_carries_newlines_has_its_atomic_units_protected_end_to_end() -> None:
    """The same guard through the chunker, on a hand-built tree. A tree is built directly here
    because `parse_layout`'s HTML path cannot produce such a block — see the next test."""
    rows = "\n".join(f"| Row {i} | Value {i} |" for i in range(6))
    prose = " ".join(f"Sentence {i} of the lead-in." for i in range(10))
    tree = _tree_with_block(f"{prose}\n{rows}\n{prose}")

    pieces = _of_level(_chunker(tree, max_chars=200).chunk(_doc(), ""), ChunkLevel.CHUNK)

    assert len(pieces) > 1, "the block had to be split or this asserts nothing"
    holding = [p for p in pieces if "| Row 0 |" in p.text]
    assert len(holding) == 1
    assert all(f"| Row {i} | Value {i} |" in holding[0].text for i in range(6))


def test_the_atomic_guard_is_latent_on_the_markup_path_because_phase_2_collapses_whitespace() -> None:
    """MEASURED, and recorded here rather than left as a surprise. `atomic_units` reads LINES, and
    Phase 2's HTML parser collapses whitespace runs to single spaces, so a markup-derived block has
    no newlines and no atomic span. A table is protected on this path by STRUCTURE instead — the
    `table` block is out of the body stream and each row is its own whole unit (see the row tests
    above). The guard stays wired for blocks that do carry newlines; it is simply not what is
    holding tables together today."""
    rows = "\n".join(f"| Row {i} | Value {i} |" for i in range(6))
    fence = "```\nline_0 = 0\nline_1 = 1\n```"
    block = next(b for b in _tree(f"<h2>Mixed</h2><p>Lead.\n{rows}\n{fence}</p>").sections[0].blocks if b.ordinal == 1)

    assert "\n" not in block.text
    assert atomic_spans(block.text) == []


def test_nothing_is_dropped_when_a_block_is_split() -> None:
    """A chunker that loses text loses it from the index silently."""
    prose = " ".join(f"Clause {i} states a requirement." for i in range(80))
    chunks = _chunker(_tree(f"<h2>Long</h2><p>{prose}</p>"), max_chars=300).chunk(_doc(), "")
    pieces = _of_level(chunks, ChunkLevel.CHUNK)

    for i in range(80):
        assert any(f"Clause {i} states" in p.text for p in pieces), f"clause {i} was dropped"


# --- pages ---------------------------------------------------------------------------------------


def test_a_chunk_takes_the_page_of_the_content_it_owns() -> None:
    first = "<h2>First</h2><p>Page one body.</p>"
    second = "<h2>Second</h2><p>Page two body.</p>"
    chunks = _of_level(
        _chunker(parse_layout(_overlay((1, first), (2, second)), _DOC_ID)).chunk(_doc(), ""), ChunkLevel.CHUNK
    )

    assert [(c.page, "Page one" in c.text) for c in chunks] == [(1, True), (2, False)]


def test_a_body_chunk_never_owns_content_from_two_pages() -> None:
    """`Chunk.page` is ONE number, so a chunk spanning a page break would report its first page for
    all of it and a citation would name the wrong page — the same rule the clause path enforces.
    Both halves here sit in the SAME section (the heading is on page 1), so only the page boundary
    can separate them, and the cap is far too large to."""
    first = "<h2>Spanning</h2><p>Body on page one.</p>"
    second = "<p>Body on page two.</p>"
    tree = parse_layout(_overlay((1, first), (2, second)), _DOC_ID)
    assert len(_walk(tree.sections)) == 1, "one section, so a section boundary cannot be what splits"

    pieces = _of_level(_chunker(tree, max_chars=4000).chunk(_doc(), ""), ChunkLevel.CHUNK)

    assert [(p.page, p.text) for p in pieces] == [
        (1, "Spanning\nBody on page one."),
        (2, "Body on page two."),
    ]


def test_a_layout_chunk_has_no_span() -> None:
    """`Chunk.span` is documented as offsets into the EXTRACTED TEXT, and markup-derived text does
    not occur in that string at any offset. Inventing offsets would point citation resolution at
    the wrong characters."""
    assert all(c.span is None for c in _chunker(_tree()).chunk(_doc(), ""))


# --- the blockless tree: byte-identical to today -------------------------------------------------

_PLAIN = (
    "6.1.1 Mandatory Rehearsal Policy\n"
    "Written sign-off is required before publication. The reviewer records the outcome.\n"
    "6.4.1 Central Content Repository\n"
    "All assets are stored centrally and versioned.\n"
)


def test_a_tree_with_no_blocks_produces_exactly_the_clause_chunkers_output() -> None:
    """Only the Chandra engines emit markup; every other extractor's document parses into a tree
    with sections and NO blocks, so layout-aware chunking has nothing to chunk. The fallback is a
    fail-safe — a chunker returning nothing would drop the document out of the index silently —
    and it must be byte-identical, not merely similar."""
    from rag_core.sections import detect_sections

    tree = parse_layout({"schema_version": 2, "extractor": "", "pages": []}, _DOC_ID, text=_PLAIN)
    assert tree.source == "text"

    expected = ClauseChunker(max_chars=700, overlap=150).chunk(
        _doc(), _PLAIN, page_map=[len(_PLAIN)], sections=detect_sections(_PLAIN)
    )
    produced = _chunker(tree, fallback=ClauseChunker(max_chars=700, overlap=150)).chunk(
        _doc(), _PLAIN, page_map=[len(_PLAIN)], sections=detect_sections(_PLAIN)
    )

    assert [c.model_dump() for c in produced] == [c.model_dump() for c in expected]
    assert all(c.level is None and c.section_id is None and c.embed_text is None for c in produced)


# --- the other strategies are untouched -----------------------------------------------------------


def test_the_existing_strategies_are_unchanged_by_the_new_arguments() -> None:
    """FIXED / CLAUSE / PARENT_CHILD are tuned on real corpora and the suite depends on them. The
    new `layout`/`mask`/`layout_max_chars` arguments must not reach them."""
    fixed = get_chunker(ChunkStrategy.FIXED, size=900, overlap=150, layout=_tree(), mask=_identity)
    clause = get_chunker(ChunkStrategy.CLAUSE, max_chars=700, overlap=150, layout=_tree(), mask=_identity)
    pc = get_chunker(ChunkStrategy.PARENT_CHILD, child_max_chars=350, parent_max_chars=1800, mask=_identity)
    assert isinstance(fixed, FixedChunker)
    assert isinstance(clause, ClauseChunker)
    assert isinstance(pc, ParentChildChunker)

    assert (fixed.size, fixed.overlap) == (900, 150)
    assert (clause.max_chars, clause.overlap) == (700, 150)
    assert (pc.child_max_chars, pc.parent_max_chars) == (350, 1800)
    assert [c.model_dump() for c in clause.chunk(_doc(), _PLAIN)] == [
        c.model_dump() for c in ClauseChunker(max_chars=700, overlap=150).chunk(_doc(), _PLAIN)
    ]


def test_contextual_is_still_an_unimplemented_stub() -> None:
    with pytest.raises(NotImplementedError):
        get_chunker(ChunkStrategy.CONTEXTUAL)


# --- the token budget, and the minimum a unit has to be worth ------------------------------------


def test_the_token_budget_lowers_the_character_cap_on_a_dense_script() -> None:
    """A character cap cannot bound a token budget: 1600 characters are ~310 tokens of English and
    ~1000 of Devanagari, and TEI truncates over-long input SILENTLY — HTTP 200, a normal-looking
    vector, no header — so everything past the limit stays in the stored text and in BM25 while
    being invisible to dense search, corpus-wide and undetectably. This path cannot count a block's
    tokens (`token_offsets` index the plain-text layer, and a unit is assembled from markup blocks),
    so it converts the budget using the DOCUMENT's own measured chars-per-token ratio."""
    body = " ".join(f"Clause {i} states a requirement in full." for i in range(60))
    tree = _tree(f"<h2>Dense</h2><p>{body}</p>")
    text = "x" * 1000

    # ~1.6 chars/token, the measured Devanagari ratio: 512 tokens * 1.6 * 0.9 safety ~= 737 chars.
    dense = _chunker(tree).chunk(_doc(), text, token_offsets=list(range(625)), max_tokens=512)
    assert dense, "the document must still be chunked"
    assert max(len(c.text) for c in dense) <= 760

    # ~5.1 chars/token, English prose: the budget is far above the configured cap, so the cap binds.
    prose = _chunker(tree).chunk(_doc(), text, token_offsets=list(range(196)), max_tokens=512)
    assert max(len(c.text) for c in prose) <= _DEFAULT_LAYOUT_MAX_CHARS
    assert max(len(c.text) for c in prose) > 760, "an English document is not penalised"


def test_no_token_budget_leaves_the_configured_cap_alone() -> None:
    """An embedder that cannot tokenize returns None, and that must not change chunking."""
    body = " ".join(f"Clause {i} states a requirement in full." for i in range(60))
    tree = _tree(f"<h2>Dense</h2><p>{body}</p>")

    assert [c.text for c in _chunker(tree).chunk(_doc(), "", token_offsets=None, max_tokens=None)] == [
        c.text for c in _chunker(tree).chunk(_doc(), "")
    ]


def test_a_container_headings_own_text_is_not_a_chunk_of_its_own() -> None:
    """A section whose only block is its heading, and which HAS subsections, emitted a 12-to-40
    character chunk ("Attachment-1") that matches almost nothing and dilutes the index — the exact
    population `chunking._MIN_CHUNK_CHARS` merges away on the text path, measured there at 12-15% of
    pieces once a heading became a hard boundary. That merge cannot be reused here: it joins across
    heading boundaries, and this module guarantees no unit spans two sections.

    Nothing is lost, which is what makes dropping it right rather than a trade: the heading of a
    section with children is carried by every descendant, in the `embed_text` breadcrumb that is
    what gets embedded and BM25-indexed, and in `metadata["section"]`."""
    chunks = _chunker(_tree()).chunk(_doc(), "")

    assert "Attachment-1" not in [c.text for c in chunks]
    assert "13. Video Wall Technical Requirements" not in [c.text for c in chunks]
    # ...and it is still findable, because it is in what the embedder and BM25 actually receive.
    assert any("Attachment-1" in (c.embed_text or "") for c in chunks)
    assert all(len(c.text) >= 40 for c in chunks), "no lone-heading fragments left"


def test_a_leaf_sections_heading_is_kept_because_nothing_else_carries_it() -> None:
    """The rule is about CONTAINER headings. A leaf with no prose has its heading as its only
    trace — dropping that would delete the section from the index entirely."""
    chunks = _chunker(_tree("<h2>Annexure C</h2>")).chunk(_doc(), "")

    assert [c.text for c in chunks] == ["Annexure C"]
