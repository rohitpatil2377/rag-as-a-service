"""Layout-aware chunking + the v2 contextualize prefix (RAG v2 Phase 5). Pure logic, no I/O.

Chunks a document from the Phase-2 STRUCTURE TREE (:mod:`rag_core.layout`) instead of from a
flat character stream. That is what makes the plan's acceptance criterion reachable at all: every
unit this chunker emits is built from the blocks of exactly ONE :class:`~rag_core.layout.
LayoutSection`, so a chunk cannot cross a section boundary — not because a guard rejects it, but
because no code path assembles text from two sections. Discovery §5 measured the opposite for the
existing chunkers (FIXED crosses freely; the clause path's carry-over tail can pull the previous
section's text across a heading) and found no test forbidding it.

LEVELS. The plan names five; FOUR are emitted as of Phase 4 and ``doc`` is DEFERRED, not stubbed
(see :class:`rag_schemas.common.ChunkLevel` for why it is absent):

* ``chunk`` — body prose, packed from the section's own heading/paragraph/list blocks.
  ``parent_id`` is the section.
* ``row``   — one unit per ``table_row`` block. Phase 2 already linearises a row WITH its header
  names (``"<breadcrumb>" + BREADCRUMB_SEP + "Table: <col>=<v> | ..."``), including rows that inherited
  a ``<th>`` from the previous page through the cross-page merge, so there is nothing to re-derive.
  ``parent_id`` is the owning ``table`` block (``LayoutBlock.parent_block_id``, populated by
  Phase 2 for exactly this).
* ``section`` — the section as ONE retrievable unit, holding its Phase-4 SUMMARY. BUILT, MEASURED
  AND WITHDRAWN in Phase 5 and RESTORED here: the plan wants the summary, Phase 5 had only the
  section's own text as a stand-in, and that stand-in was a verbatim duplicate of the units
  beside it in every reachable case. :meth:`LayoutAwareChunker._section_level` carries the
  argument and the measurements. Emitted only for a section that has an enrichment.
* ``proposition`` — one unit per Phase-4 proposition: a standalone requirement or fact, each
  already validated against a verbatim quote from this section
  (:mod:`rag_core.section_enrich`). ``parent_id`` is the section. Emitted only for a section
  that has an enrichment.

ENRICHMENT IS OPTIONAL EVERYWHERE. ``enrichment`` defaults to empty, and with it empty this
chunker emits exactly the ``chunk``/``row`` units Phase 5 emitted, byte for byte. That is the
whole reason the two new levels are keyed per section rather than switched on per document: a
document whose enrichment covered nine of its forty sections gets summaries and propositions for
those nine and is unchanged everywhere else.

WHAT IS NOT INCLUDED IN THE BODY STREAM, and why: ``table`` and ``table_row`` blocks. Every data
row is already its own ``row`` unit, so packing the table's linearised text into a body chunk too
would index every cell twice — once per row and once inside a body chunk — inflating the index and
double-counting the same evidence at fusion time. Every cell still reaches the index: each data row
is its own ``row`` unit, and a table that produced NO rows is admitted to the body stream for
exactly that reason (see :meth:`LayoutAwareChunker._body_level`).

THE MASKING INVARIANT (the one thing in this module that is a contract, not a judgement call).
``Chunk.text``'s own field description is "Chunk text (PII already masked, FR-G2)" and the
platform's hard constraint is that only de-identified text is chunked/embedded/indexed. But
``LayoutBlock.text`` and ``LayoutSection.text`` are UNMASKED — ``DocIntelligenceService.extract``
captures the layout overlay from the pre-masking engine output, and
``svc_control_plane.pipeline._persist_layout_structure`` says so at the site. So this chunker takes
a ``mask`` callable and there is NO DEFAULT: a :class:`LayoutAwareChunker` cannot be constructed
without one, and every string that becomes ``Chunk.text`` / ``Chunk.embed_text`` /
``metadata["section"]`` goes through it in :meth:`LayoutAwareChunker._unit`, the single place a
``Chunk`` is built here.

The masker runs THERE and not on the tree up front, for two reasons:

* The tree is also the source of the PERSISTED ``sections``/``blocks`` rows, whose ids are derived
  from the unmasked text by the Phase-1 helpers. Masking it in place would make the stored
  ``text_hash``/``block_id`` disagree with the stored text, and a v2 chunk's ``section_id`` would
  no longer resolve to any row.
* A pre-masked COPY of the tree would be a second tree to keep in step, and the one place a later
  edit could reintroduce exactly this bug by reading from the wrong one.

Consequently ``section_id`` / ``parent_id`` are the UNMASKED tree's ids (they are identity — they
must equal the persisted rows), while every stored or embedded STRING is masked.

HONEST LIMITS, so a later phase does not rediscover them as bugs:

* **A blockless tree falls back to clause chunking.** Only the Chandra engines emit markup; every
  native-PDF / pypdf / Tesseract / Paddle / Office document parses through ``parse_layout``'s
  plain-text fallback, which yields sections and NO blocks (the text layer has no block types). A
  tree with no blocks therefore has no text to chunk, and this chunker delegates to
  :class:`~rag_core.chunking.ClauseChunker`, producing byte-identical output to today and NO
  v2 level fields. The ingest path does not even pass such a tree (see
  ``svc_control_plane.pipeline._layout_tree``), so the delegation is a fail-safe: a chunker that
  returned nothing would drop the document out of the index silently, which is the only failure
  mode here that nothing would report.
* **No ``span``.** ``Chunk.span`` is documented as char offsets into the extracted document text.
  Markup-derived text does not occur in that text layer at any offset (the engine's HTML and the
  rendered text are different strings), so inventing offsets would make citation resolution point
  at the wrong characters. ``span`` is None, which every reader already handles.
* **No carry-over overlap.** The clause path carries a tail into the next chunk so a rule keeps the
  exception that qualifies it; here that tail is the one mechanism that could cross a section
  boundary, so there is none. What replaces it is the ``section``-level unit — which since Phase 4
  holds the section's SUMMARY rather than its whole text, so the aggregate grain is a condensation
  of the section and no longer a copy of it.
* **The atomic-unit guard is LATENT on the markup path, and that is not a bug here.** Phase 5
  reuses :mod:`rag_core.atomic_units` rather than writing a second guard — but every signal
  that module reads is LINE-based (a run of consecutive lines containing ``" | "``, a pair of
  ``` fence lines, a short dense formula line), and Phase 2's HTML block parser collapses
  whitespace runs to single spaces (``layout._collapse``), so a markup-derived block's text
  contains no newlines and :func:`atomic_spans` returns nothing for it. MEASURED, not assumed:
  ``<pre>``'s fences and newlines are already gone by the time this module sees the block.
  What actually protects a table here is the STRUCTURE — a ``table`` block is excluded from the
  body stream and each ``table_row`` is emitted whole as its own unit, so "never split mid-row" is
  guaranteed by construction rather than by a text heuristic. The guard is kept because it is the
  right call for any block that does carry newlines (the Markdown page parser
  ``layout.parse_layout`` currently skips, a future ``<pre>`` that preserves them) and because
  :func:`_units` is also the splitter the fallback-free path uses; it is wired and tested, just not
  load-bearing today. If Phase 6 preserves ``<pre>`` whitespace, this becomes live with no change.
* **The budget is CHARACTERS, calibrated to the plan's token band** — see
  :data:`_DEFAULT_LAYOUT_MAX_CHARS`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_core.atomic_units import atomic_spans
from rag_core.chunking import (
    META_SECTION,
    ClauseChunker,
    _clauses,
    _hard_wrap,
    structured_chunk_id,
    text_hash,
)
from rag_core.layout import (
    BLOCK_HEADING,
    BLOCK_LIST_ITEM,
    BLOCK_PARAGRAPH,
    BLOCK_TABLE,
    BLOCK_TABLE_ROW,
    BREADCRUMB_SEP,
    LayoutSection,
    LayoutTree,
)
from rag_core.section_enrich import SectionEnrichment
from rag_core.sections import PATH_SEP
from rag_schemas import Chunk, DocumentEnvelope
from rag_schemas.common import ChunkLevel, ChunkStrategy

if TYPE_CHECKING:
    from rag_core.sections import Section

# The plan's target is "300-500 tokens (configurable per doc_type)". This repo's chunkers are
# CHARACTER-based (CLAUSE 700, FIXED 900/150, PARENT_CHILD 350/1800) and a real token count needs
# the EMBEDDER's tokenizer, which only the TEI adapter exposes (`token_budget` / `token_offsets`);
# the deployed `openai_compat` adapter exposes neither, so in production the character cap is the
# only cap there has ever been.
#
# THE MAPPING, AND ITS IMPRECISION. Measured on this corpus and recorded in `chunking.py`: English
# prose runs ~5.1 chars/token, serialized table rows ~4.3, Devanagari ~1.6 and ToC dot leaders
# ~1.25. So 1600 characters is ~310 tokens of prose and ~370 of serialized rows — inside the plan's
# 300-500 band for the scripts that band was written for — and ~1000 tokens of Devanagari or ~1280
# of dot leaders, which is OUTSIDE it and, on a 512-token embedder, past the input limit (TEI
# truncates silently: HTTP 200, a normal-looking vector, no header). A character count cannot bound
# a token budget; only the embedder's own `token_offsets` can, and on this path they are
# unavailable, because the offsets the indexing service fetches are for the PLAIN TEXT layer and
# markup-derived block text does not appear in it at those positions. Fetching per-unit offsets
# means one tokenizer round trip per block, which belongs with the embed fan-out in Phase 6, not
# here. Until then: calibrated characters, stated rather than dressed up as tokens.
#
# PER-DOC_TYPE configuration arrives from the service layer: the Phase-3 profile card supplies a
# `doc_type` and `RAG_LAYOUT_CHUNK_MAX_CHARS_BY_DOC_TYPE` maps it to a size, resolved in
# `svc_indexing.service._layout_max_chars` and passed in as `max_chars`. Nothing in rag-core reads
# config (see `get_chunker`), so this stays the single library default.
_DEFAULT_LAYOUT_MAX_CHARS = 1600

# Blocks whose text is body prose. A ``heading`` is in here because the heading line is real
# document text and belongs with the text it introduces — the same reasoning `_merge_fragments`
# applies when it folds a lone heading forward. ``table`` / ``table_row`` are deliberately out
# (see the module docstring).
_BODY_BLOCK_TYPES = frozenset({BLOCK_HEADING, BLOCK_PARAGRAPH, BLOCK_LIST_ITEM})
# A document's OWN measured chars-per-token ratio is an average, and a chunk can be denser than the
# document it came from (a run of ToC dot leaders, a table of codes). The budget is shaved so an
# averagely-dense document has headroom rather than sitting exactly on the embedder's limit.
_TOKEN_BUDGET_SAFETY = 0.9
# Below this a character cap stops expressing "a retrievable passage" and starts shredding prose, so
# a pathological ratio clamps here and the honest outcome is a logged risk of silent truncation
# rather than a corpus of fragments. 300 chars is ~60 tokens of the densest script measured.
_MIN_EFFECTIVE_CAP = 300

# Separator when two whole BLOCKS land in one chunk: the blank line that already separates them in
# `LayoutSection.text`, so a chunk reads exactly as that part of the section does.
# A SINGLE newline, matching how the text layer represents a line wrap — not a blank line. The
# masker's patterns span one newline and not a blank run, so joining blocks with "\n\n" put a gap
# through the middle of any identifier that wrapped across two blocks and the whole value went
# unmasked. On the `layout-json` path every LINE is its own block, so that is the common case, not
# an edge one. Paragraph structure is not lost by this: it is in the persisted `blocks` rows.
_BLOCK_SEP = "\n"
# Separator when pieces of ONE over-long block are regrouped. A newline, not a space: the pieces
# are distinct clauses, lines, list items or table lines, and a space would fuse two lines into one
# the document never had — which matters most for the atomic units (a pipe table, a code fence)
# this split exists to keep whole.
_UNIT_SEP = "\n"


def _present(*chunks: Chunk | None) -> list[Chunk]:
    """Drop the units :meth:`LayoutAwareChunker._unit` declined to build (empty after masking)."""
    return [c for c in chunks if c is not None]


def embed_prefix(breadcrumb: str, card: str = "") -> str:
    """The plan's contextualize prefix for one unit: ``"{card}[{breadcrumb}] "``.

    The plan's full prefix is ``"[{doc_type}: {title} — {purpose}] [{breadcrumb}] "``. The
    breadcrumb half is Phase 2's, and it is the whole reason that phase was the plan's quick win:
    "Controller" on its own retrieves very differently from the same words under a breadcrumb
    reading "LED Video Wall controller", and the chunk's own words carry no trace of which one it
    is. ``card`` is the Phase-3 half — :func:`rag_core.doc_profile.profile_card` builds it —
    and it is EMPTY whenever the document has no profile, which is what keeps this byte-identical
    to Phase 5's output on every document the profile call did not produce a card for.

    BOTH ARGUMENTS MUST ALREADY BE MASKED — a heading can carry an identifier, and this string is
    embedded and BM25-indexed. :meth:`LayoutAwareChunker._unit` guarantees it for the breadcrumb;
    the chunker's constructor takes the card already built, so its caller guarantees that half.
    """
    crumb = f"[{breadcrumb}] " if breadcrumb else ""
    return card + crumb


@dataclass(frozen=True, slots=True)
class _Piece:
    """A candidate span of body text and the page it came from."""

    text: str
    page: int | None


class LayoutAwareChunker:
    """Chunks a document from its Phase-2 structure tree, emitting the v2 levels (FR-C1, Phase 5).

    ``tree`` is the SAME :class:`~rag_core.layout.LayoutTree` the caller persists into
    ``sections``/``blocks``, so a chunk's ``section_id`` resolves to a stored row.

    ``mask`` de-identifies every string that becomes part of a ``Chunk``. It is required and has no
    default — see the module docstring; the tree's text is captured pre-masking.

    ``fallback`` is used verbatim when the tree carries no blocks (the plain-text path). The caller
    builds it with the same sizing a plain ``CLAUSE`` run would get, so that output is
    byte-identical to today's.
    """

    strategy = ChunkStrategy.LAYOUT_AWARE

    def __init__(
        self,
        *,
        tree: LayoutTree,
        mask: Callable[[str], str],
        fallback: ClauseChunker,
        max_chars: int = _DEFAULT_LAYOUT_MAX_CHARS,
        card: str = "",
        enrichment: Mapping[str, SectionEnrichment] | None = None,
    ) -> None:
        if not callable(mask):
            # The class and module docstrings both state a chunker cannot exist without a masker.
            # Until now only `get_chunker` enforced it, so the claim held by convention rather than
            # by construction, and a direct `mask=None` failed later as a TypeError inside `_unit`
            # — after the caller believed it had a chunker. This is the masking INVARIANT, so it
            # fails at the one place that can still fail safely.
            raise TypeError("LayoutAwareChunker requires a callable `mask` (FR-G2): see the module docstring")
        self._tree = tree
        self._mask = mask
        self._fallback = fallback
        self.max_chars = max_chars
        # The Phase-3 profile card, already built and already masked (see `embed_prefix`). "" for a
        # document with no profile, which is every document when `v2_ingest` is off and any
        # document whose profile call failed — the chunker is unchanged in both cases.
        self._card = card
        # The Phase-4 enrichment for this document, keyed by `section_id`. Empty for every
        # document when `v2_ingest` is off, when section enrichment is off, and for any
        # section whose enrichment call did not produce one — all three leave this chunker
        # byte-identical to Phase 5 + Phase 3, which is what makes the level additive.
        self._enrichment: Mapping[str, SectionEnrichment] = enrichment or {}
        self._breadcrumbs: dict[str, str] = {}

    def chunk(
        self,
        doc: DocumentEnvelope,
        text: str,
        *,
        page_map: list[int] | None = None,
        token_offsets: list[int] | None = None,
        max_tokens: int | None = None,
        sections: list[Section] | None = None,
    ) -> list[Chunk]:
        """The tree's units, in document order; the fallback's chunks when the tree has no blocks.

        ``page_map`` and ``sections`` are only used by the fallback — the tree-driven path takes its
        text from the blocks and its pages from the blocks' own page numbers, which is the whole
        point of chunking the structure rather than the stream. ``max_tokens``/``token_offsets`` DO
        apply here, via :meth:`_effective_cap`.
        """
        if not _has_blocks(self._tree):
            # Fail-safe, not a feature — see the module docstring.
            return self._fallback.chunk(
                doc,
                text,
                page_map=page_map,
                token_offsets=token_offsets,
                max_tokens=max_tokens,
                sections=sections,
            )
        cap = self._effective_cap(text, token_offsets, max_tokens)
        out: list[Chunk] = []
        for section in _walk(self._tree.sections):
            out.extend(self._section_units(doc, section, cap))
        # DOCUMENT ORDER, stamped here because here is the only place that has it. Every v2 chunk
        # has `span=None` (markup-derived text has no offset into the extracted text layer), which
        # is the v1 ordering key — so without this a catalog returns a document's chunks in
        # whatever order the store happens to yield, and the citation viewer, whole-document
        # answering and neighbour expansion all read a scrambled document. `_walk` is depth-first
        # in reading order and `_section_units` emits body-then-rows within a section, so the
        # enumeration IS the document order. Applied as a copy so no level has to thread a counter.
        return [c.model_copy(update={"sequence": i}) for i, c in enumerate(out)]

    def _effective_cap(self, text: str, token_offsets: list[int] | None, max_tokens: int | None) -> int:
        """``max_chars``, lowered when the embedder's TOKEN limit binds first.

        The size target here is CHARACTERS, and a character cap cannot bound a token budget: the
        same 1600 characters are ~310 tokens of English prose and ~1000 of Devanagari. TEI truncates
        over-long input SILENTLY — HTTP 200, a normal-looking vector, no header — so the content past
        the limit stays in the stored text and in BM25 but is invisible to dense search, corpus-wide
        and with nothing to see it by. `FixedChunker`'s docstring records this already biting this
        corpus: 19 chunks truncated at embed DESPITE its 900-character cap.

        This path cannot count a block's tokens: `token_offsets` are offsets into the plain-TEXT
        layer and a layout unit is assembled from markup blocks, so there is no offset to look up.
        What the offsets DO give is this document's own measured chars-per-token ratio, and that
        ratio is a property of the SCRIPT, which both layers share. So the token budget converts to
        a character budget using the document's own measurement rather than a constant — a
        Devanagari tender lowers its own cap, an English one leaves the configured 1600 binding.
        Averages, hence `_TOKEN_BUDGET_SAFETY`; honest but not exact, which is why it shaves the
        budget rather than claiming to enforce it.

        A `row` is exempt: it is never split at all (half a row supports no statement), so a very
        wide row can still exceed the budget. That is a deliberate, stated limit of the row level.
        """
        if not max_tokens or not token_offsets or not text:
            return self.max_chars
        chars_per_token = len(text) / len(token_offsets)
        budget = int(max_tokens * chars_per_token * _TOKEN_BUDGET_SAFETY)
        # The Phase-3 CARD rides in front of every unit's `embed_text`, so it spends part of the
        # same token budget this converts — subtract it, or a document with a long card quietly
        # pushes its largest chunks past the embedder's limit and is truncated silently, which is
        # the failure this whole conversion exists to prevent. Subtracted from the TOKEN-derived
        # budget only: `max_chars` is a character target for the chunk's own text and the card is
        # not part of it, so a document with no token budget chunks exactly as Phase 5 chunked it.
        # The breadcrumb is NOT subtracted here, as it was not before: it varies per section while
        # this cap is per document, and that pre-existing gap is a measurement of its own.
        budget -= len(self._card)
        return max(_MIN_EFFECTIVE_CAP, min(self.max_chars, budget))

    # --- per-section emission -------------------------------------------------------------------

    def _section_units(self, doc: DocumentEnvelope, section: LayoutSection, cap: int) -> list[Chunk]:
        """One section's units. Reads nothing outside it, which is why no chunk can cross a
        section boundary.

        ORDER IS DOCUMENT ORDER for the `sequence` stamped in :meth:`chunk`, so the aggregate
        comes first and the derived statements last: `section` (the summary, which is about all of
        what follows) -> `chunk` -> `row` -> `proposition`. Only the middle two exist without
        Phase-4 enrichment."""
        return (
            self._section_level(doc, section)
            + self._body_level(doc, section, cap)
            + self._row_level(doc, section)
            + self._proposition_level(doc, section)
        )

    def _section_level(self, doc: DocumentEnvelope, section: LayoutSection) -> list[Chunk]:
        """The section's Phase-4 SUMMARY as one retrievable unit. Empty without an enrichment.

        This level is the plan's answer to "what does clause 6.2 require" — one retrievable unit
        per section, at the aggregate grain. Phase 5 BUILT it, measured it and withdrew it,
        because without a summary its text could only be `LayoutSection.text`, the concatenation
        of the section's own blocks, and that is always a duplicate of the units beside it:

        * more than one body chunk means the body text exceeded `max_chars`, which means
          `section.text` exceeds it too — and an oversized section got no section unit anyway;
        * so the only way a section could hold two or more other units while its own text still
          fit the cap was a TABLE, whose rows are separate `row` units — and `section.text`
          contains each row's linearised text VERBATIM (measured: `"Clause 1 \u203a Table: Item=Cable |
          Qty=10"` is a substring of its own section's text);
        * and a section holding exactly one other unit re-emitted that unit's bytes, with the same
          `embed_text` (measured on the Phase-5 fixture: 12 units, 8 distinct texts, 4
          byte-identical `section`/`chunk` pairs).

        Every case was the same evidence indexed twice — two vectors, two BM25 documents and two
        RRF candidates for one passage, which is exactly the objection this module raises to keep
        table text out of the body stream.

        THE SUMMARY IS WHAT CLOSES THAT ARGUMENT, and it is the only thing that does. A summary is
        not a substring of the section: it is 2-4 sentences naming what the whole section is
        about, which is the one text in this document that no `chunk` or `row` unit carries. So
        the duplication objection does not reach it, and the aggregate grain the plan asks for
        finally has a unit that is worth retrieving on its own.

        NO CAP IS APPLIED, on the same rule `_row_level` states: an atomic unit is emitted whole
        or not at all, because half a summary summarises nothing. `section_enrich` bounds the
        summary at 1500 characters, below the 1600 this chunker defaults to.

        HONEST LIMIT, since that bound is CHARACTERS and the embedder's is TOKENS: 1500 characters
        is ~290 tokens of English prose but ~940 of Devanagari, so a model that writes a maximal
        summary in a dense script can exceed a 512-token embedder, which truncates silently (§8.10
        — HTTP 200, a normal-looking vector, no header). That is the same exposure an over-wide
        `row` already carries and is accepted for the same reason; what bounds it in practice is
        that the prompt asks for 2-4 sentences, which measures in the low hundreds of characters.

        A section with no enrichment — the model declined, the call failed, the flag is off, the
        section was too short to enrich — emits nothing here, exactly as in Phase 5.
        """
        enrichment = self._enrichment.get(section.section_id)
        if enrichment is None or not enrichment.summary.strip():
            return []
        return _present(
            self._unit(
                doc,
                section,
                level=ChunkLevel.SECTION,
                text=enrichment.summary,
                # The heading's page. A summary spans the whole section and has no page of its
                # own; the heading is where a reader would turn to check it, and `None` would
                # drop the citation's page entirely.
                page=section.page,
                parent_id=section.section_id,
                ordinal=0,
            )
        )

    def _proposition_level(self, doc: DocumentEnvelope, section: LayoutSection) -> list[Chunk]:
        """One unit per Phase-4 PROPOSITION. Empty without an enrichment.

        A proposition is a standalone requirement or fact the model extracted from this section,
        each one carrying a verbatim quote that `section_enrich.parse_section_enrichment` has
        already checked against the section's own text — an unvalidated one never reaches here.

        NEVER SPLIT, for the reason a `row` is never split: half a requirement supports no
        statement. `section_enrich` bounds a proposition at 800 characters, so the length is
        bounded before this module sees it rather than by cutting one in half here — with the
        same characters-versus-tokens caveat :meth:`_section_level` records, at roughly half the
        exposure.

        THE QUOTE IS NOT INDEXED BESIDE THE PROPOSITION. It is already in the section's `chunk` or
        `row` units verbatim — that is what validating it against the section text means — so
        emitting "proposition + quote" would re-index a span this document already has at another
        level, which is the duplication this module refuses everywhere else. The quote is kept in
        `sections.key_values`/the enrichment row, where it is auditable; what is indexed is the
        self-contained statement, which is the thing a question matches and the thing no other
        level carries.
        """
        enrichment = self._enrichment.get(section.section_id)
        if enrichment is None:
            return []
        return _present(
            *(
                self._unit(
                    doc,
                    section,
                    level=ChunkLevel.PROPOSITION,
                    text=proposition.text,
                    page=section.page,
                    parent_id=section.section_id,
                    ordinal=ordinal,
                )
                for ordinal, proposition in enumerate(enrichment.propositions)
            )
        )

    def _body_level(self, doc: DocumentEnvelope, section: LayoutSection, cap: int) -> list[Chunk]:
        """Body prose, packed to the cap: whole blocks where they fit, split only where one block
        is already over it."""
        # A `table` block normally stays OUT of the body stream (module docstring: every data row
        # is already its own `row` unit). A table that produced NO row units is the exception: with
        # no rows and — for an oversized section — no `section` unit either, its text would reach
        # no level at all and the table would be silently absent from the index. Reproduced on a
        # header-only table (`<tr><th>Serial</th><th>Penalty Clause</th></tr>`) inside an oversized
        # section: "Penalty Clause" appeared in zero chunks. Keyed on the row blocks' own
        # `parent_block_id` rather than on a row count, so a section holding two tables where only
        # one yielded rows still indexes the other.
        parented = {b.parent_block_id for b in section.blocks if b.block_type == BLOCK_TABLE_ROW}
        pieces: list[_Piece] = []
        for block in section.blocks:
            rowless_table = block.block_type == BLOCK_TABLE and block.block_id not in parented
            if block.block_type in _BODY_BLOCK_TYPES or rowless_table:
                pieces.extend(_split_block(block.text, block.page, cap))
        packed = [p for p in _pack(pieces, cap, _BLOCK_SEP) if p.text.strip()]
        if not packed:
            return []
        # A CONTAINER heading is not a passage. A section whose own blocks are nothing but its
        # heading, and which has subsections, would otherwise emit a 12-to-40-character chunk
        # ("Attachment-1", "13. Video Wall Technical Requirements") that matches almost nothing and
        # dilutes the index — the exact population `chunking._MIN_CHUNK_CHARS` exists to merge away
        # on the text path, measured there at 12-15% of pieces once a heading became a hard
        # boundary. That merge cannot be reused here: it joins across heading boundaries, and this
        # module's whole guarantee is that no unit spans two sections.
        #
        # Dropping it loses nothing, which is what makes this the right answer rather than a
        # compromise: the heading of a section WITH children is already carried by every descendant,
        # in the `embed_text` breadcrumb that is what gets embedded and BM25-indexed, and in
        # `metadata["section"]`. A LEAF section is not dropped even if its only block is a heading —
        # there the heading is the section's sole trace, and nothing else would carry it.
        if section.children and all(b.block_type == BLOCK_HEADING for b in section.blocks):
            return []
        return _present(
            *(
                self._unit(
                    doc,
                    section,
                    level=ChunkLevel.CHUNK,
                    text=piece.text,
                    page=piece.page,
                    parent_id=section.section_id,
                    ordinal=ordinal,
                )
                for ordinal, piece in enumerate(packed)
            )
        )

    def _row_level(self, doc: DocumentEnvelope, section: LayoutSection) -> list[Chunk]:
        """One unit per table row. NEVER split — a row is one block and is emitted whole however
        wide the table is, because half a row supports no statement (the same rule
        :mod:`rag_core.atomic_units` enforces for the text path)."""
        rows = [b for b in section.blocks if b.block_type == BLOCK_TABLE_ROW and b.text.strip()]
        return _present(
            *(
                self._unit(
                    doc,
                    section,
                    level=ChunkLevel.ROW,
                    text=row.text,
                    page=row.page,
                    # Phase 2 populates `parent_block_id` for exactly this link. None would mean
                    # the overlay gave a row with no table, which `_blocks` cannot produce.
                    parent_id=row.parent_block_id,
                    ordinal=ordinal,
                )
                for ordinal, row in enumerate(rows)
            )
        )

    # --- the single place a Chunk is built ------------------------------------------------------

    def _unit(
        self,
        doc: DocumentEnvelope,
        section: LayoutSection,
        *,
        level: ChunkLevel,
        text: str,
        page: int | None,
        parent_id: str | None,
        ordinal: int,
    ) -> Chunk | None:
        """Build one v2 chunk. THE MASKER RUNS HERE, on every string that leaves this module.

        The id is :func:`~rag_core.chunking.structured_chunk_id` over the MASKED text: the id
        names the text that is stored, so a chunk whose only change is a new redaction strategy
        gets a new id, which is true. The ``section_id`` fed into it is the tree's own, UNMASKED —
        it is identity and must equal the persisted ``sections`` row.
        """
        masked = self._mask(text)
        if not masked.strip():
            # A unit whose text survives the tree but not the masker. `Chunk.text` has no
            # `min_length`, so an empty chunk would be embedded (an empty vector) and BM25-indexed,
            # matching nothing and diluting fusion. `RegexPIIMasker` cannot produce this today (all
            # three strategies emit a non-empty replacement), but a Presidio/NER backend or a future
            # drop-strategy can, and the masker is an injected seam.
            return None
        breadcrumb = self._breadcrumb(section)
        metadata = dict(doc.metadata)  # document tags (FR-A7/B7) flow to chunks -> D8 filterable
        if breadcrumb:
            # The same key the v1 chunkers stamp, so the metadata contextualizer, the section
            # filter and every saved query keep working on a v2 chunk unchanged — which requires
            # the v1 SEPARATOR too, not just the v1 key. `filters` is an EXACT-match predicate, so
            # a stored filter `section = "13. Video Wall > Standards"` matches zero v2 chunks if
            # this holds the tree's " > " variant. The breadcrumb keeps `BREADCRUMB_SEP` everywhere
            # it is the layout path's own (`embed_text`, the persisted `sections` row) and is
            # converted ONLY here, where the namespace is shared with v1.
            metadata[META_SECTION] = breadcrumb.replace(BREADCRUMB_SEP, PATH_SEP)
        return Chunk(
            chunk_id=structured_chunk_id(section.section_id, level.value, ordinal, text_hash(masked)),
            doc_id=doc.doc_id,
            tenant_id=doc.tenant_id,
            text=masked,
            page=_page(page),
            span=None,  # markup-derived text has no offset in the extracted text layer
            source_uri=doc.uri,
            acl=list(doc.source_acl),
            classification=doc.classification,
            lineage_id=doc.lineage_id,
            level=level,
            section_id=section.section_id,
            parent_id=parent_id,
            # A `row`'s text ALREADY opens with its breadcrumb — Phase 2's `layout._row_text`
            # builds it as "<breadcrumb> > Table: col=v | ...", so prefixing it again embedded the
            # crumb twice ("[A > B] A > B > Table: ..."), which on a wide table in a deep hierarchy
            # is a material share of the embedded string: it dilutes the row's own terms and costs
            # BM25 term-frequency weight on the cells that distinguish one row from the next.
            # A `row` gets the CARD but not the breadcrumb: its text already opens with the crumb
            # (Phase 2's `layout._row_text` builds it as "<breadcrumb> > Table: col=v | ..."), so
            # prefixing that again embedded it twice, which on a wide table in a deep hierarchy
            # dilutes the row's own terms and costs BM25 term-frequency weight on the cells that
            # distinguish one row from the next. The card is not in the row text and is not
            # duplicated by it, so that argument does not reach it.
            embed_text=embed_prefix("" if level is ChunkLevel.ROW else breadcrumb, self._card) + masked,
            metadata=metadata,
        )

    def _breadcrumb(self, section: LayoutSection) -> str:
        """The section's breadcrumb, MASKED, memoised per section (it is reused by every unit of
        that section and masking is a regex sweep)."""
        cached = self._breadcrumbs.get(section.section_id)
        if cached is None:
            cached = self._mask(section.breadcrumb)
            self._breadcrumbs[section.section_id] = cached
        return cached


# --- tree helpers -------------------------------------------------------------------------------


def _walk(sections: Sequence[LayoutSection]) -> Iterator[LayoutSection]:
    """Pre-order, so units come out in document order."""
    for section in sections:
        yield section
        yield from _walk(section.children)


def _has_blocks(tree: LayoutTree) -> bool:
    """Whether this tree carries a block stream at all. ``LayoutTree.source`` says the same thing,
    but the absence of blocks is what actually decides whether there is anything to chunk."""
    return any(section.blocks for section in _walk(tree.sections))


def _page(page: int | None) -> int | None:
    """``Chunk.page`` is validated ``ge=1``. A page number of 0 or less can only come from a
    malformed overlay, and a 422 at ingest is a worse answer than a chunk with no page."""
    return page if page is not None and page >= 1 else None


# --- splitting and packing ----------------------------------------------------------------------


def _split_block(text: str, page: int | None, cap: int) -> list[_Piece]:
    """One block as pieces within ``cap``. Whole when it fits, which is the common case."""
    body = text.strip()
    if not body:
        return []
    if len(body) <= cap:
        return [_Piece(body, page)]
    return _pack([_Piece(unit, page) for unit in _units(body, cap)], cap, _UNIT_SEP)


def _units(text: str, cap: int) -> list[str]:
    """The smallest pieces a split may ever separate.

    Each ATOMIC unit (table run, code fence, formula line — :func:`atomic_spans`) is one piece and
    is emitted whole even when it alone exceeds ``cap``: a table cut between its header and a row,
    or half a code fence, is worse than an oversized chunk. Everything between atomic units is cut
    at CLAUSE boundaries, and only a single clause longer than the cap is hard-wrapped — on a word
    boundary — which is the one place a cut can land mid-sentence and is the same rule the clause
    path already follows.

    Reuses the clause engine and the atomic-unit detector rather than adding a second guard.
    """
    out: list[str] = []
    at = 0
    for start, end in atomic_spans(text):
        if start > at:
            out.extend(_clause_units(text[at:start], cap))
        out.append(text[start:end])
        at = end
    out.extend(_clause_units(text[at:], cap))
    return [stripped for piece in out if (stripped := piece.strip())]


def _clause_units(text: str, cap: int) -> list[str]:
    """Clause-boundary pieces of ``text``, hard-wrapped only where one clause exceeds ``cap``."""
    return [piece for start, _end, clause in _clauses(text) for _offset, piece in _hard_wrap(start, clause, cap)]


def _pack(pieces: Sequence[_Piece], cap: int, sep: str) -> list[_Piece]:
    """Greedily group consecutive pieces into units of at most ``cap`` characters.

    Flushes BEFORE appending, so the cap is a cap and not a floor — the same correction
    `ClauseChunker` carries, where testing the buffer after appending made 97.6% of chunks overshoot
    on the real corpus.

    ...and flushes at a PAGE BOUNDARY too, for the reason the clause path already states: a
    ``Chunk`` carries ONE page number, so a group holding blocks from two pages would report its
    first page for all of it and a citation would name the wrong page. Unlike the clause path there
    is no carry-over to exempt here, so the rule is absolute: no body chunk owns content from two
    pages. A ``section``-level unit legitimately spans pages — that is what a section is — and its
    real extent is in its ``sections.page_start``/``page_end`` row.
    """
    out: list[_Piece] = []
    buffer: list[str] = []
    page: int | None = None
    size = 0
    for piece in pieces:
        extra = len(piece.text) + (len(sep) if buffer else 0)
        if buffer and (size + extra > cap or piece.page != page):
            out.append(_Piece(sep.join(buffer), page))
            buffer, size, page, extra = [], 0, None, len(piece.text)
        if not buffer:
            page = piece.page
        buffer.append(piece.text)
        size += extra
    if buffer:
        out.append(_Piece(sep.join(buffer), page))
    return out
