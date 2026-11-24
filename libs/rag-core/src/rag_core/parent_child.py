"""Parent/child chunking (FR-C2, Option B) — structure-aware small children + a context bank.

A precision/context trade-off that a single chunk size cannot win: SMALL children (~350 chars)
so a narrow lookup matches a tight span and the abstain gate scores something specific, plus a
per-section PARENT that carries the surrounding context the small child lost. Children are the
only search/rank/citation unit; parents are a NON-CITABLE context bank fetched at query time.

This chunker REUSES the clause engine's primitives (``_clauses``, ``_hard_wrap``,
``_merge_fragments``, ``_mk_chunk`` …) rather than reimplementing them — the only merge surface
with the owner's ``chunking.py`` is the import list below, not chunking logic. Two rules distinguish
it from plain clause chunking:

1. Atomic units (tables, code fences, formula lines — see :mod:`rag_core.atomic_units`) are
   NEVER split. A child boundary snaps to an atomic edge, and a single atomic unit over the child
   cap is emitted as ONE oversized child rather than hard-wrapped.
2. When the text contains NO atomic unit, output is BYTE-IDENTICAL to :class:`ClauseChunker` at
   the same cap — the chunker is "never worse than clause".

``chunk`` returns the children (each stamped ``metadata["parent_id"]``) FOLLOWED by the parent
records (each marked ``metadata["chunk_role"]="parent"``). The indexing service partitions on that
role: children are embedded + BM25-indexed + cataloged; parents are cataloged ONLY (never embedded
or sparse-indexed), so they cannot be double-retrieved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_core.atomic_units import atomic_spans
from rag_core.chunking import (
    _UNSET,
    META_SECTION,
    ClauseChunker,
    _clauses,
    _hard_wrap,
    _merge_fragments,
    _mk_chunk,
    _page_for_offset,
    _piece_tokens,
    _tail,
    _tokens_between,
    chunk_id,
)
from rag_schemas import Chunk, DocumentEnvelope
from rag_schemas.common import ChunkStrategy

if TYPE_CHECKING:
    from rag_core.sections import Section

# Child/parent metadata keys — a STABLE query/storage vocabulary (like META_SECTION), so a module
# constant rather than a knob. ``parent_id`` links a child to its context bank; ``chunk_role`` marks
# a record as a parent so every ``document_chunks`` consumer can exclude it from search/eval/viewer.
META_PARENT_ID = "parent_id"
META_CHUNK_ROLE = "chunk_role"
PARENT_ROLE = "parent"

# Library defaults (see rag_common.config for the operator-facing values the service injects).
# Children are tight for precision; parents align with sections._MAX_SECTION_SPAN (2000).
_DEFAULT_CHILD_MAX_CHARS = 350
_DEFAULT_PARENT_MAX_CHARS = 1800
_DEFAULT_OVERLAP = 100

# A parent must group at least this many children to exist: a lone child IS its own context, and a
# single-child "parent" would carry byte-identical text and therefore a COLLIDING content-derived
# id (chunk_id(doc, text) is the same for both), overwriting the child on catalog upsert.
_MIN_PARENT_CHILDREN = 2


def is_parent(chunk: Chunk) -> bool:
    """True if ``chunk`` is a parent context record (never a search/citation unit)."""
    return chunk.metadata.get(META_CHUNK_ROLE) == PARENT_ROLE


def not_parent(chunks: list[Chunk]) -> list[Chunk]:
    """Drop parent context records — the shared filter every ``document_chunks`` consumer applies so
    a parent never leaks into a neighbour window, the citation viewer, or an eval sample."""
    return [c for c in chunks if not is_parent(c)]


class ParentChildChunker:
    """Small clause/atomic-aware children plus per-section parent context records (FR-C2)."""

    strategy = ChunkStrategy.PARENT_CHILD

    def __init__(
        self,
        child_max_chars: int = _DEFAULT_CHILD_MAX_CHARS,
        parent_max_chars: int = _DEFAULT_PARENT_MAX_CHARS,
        overlap: int = _DEFAULT_OVERLAP,
    ) -> None:
        self.child_max_chars = child_max_chars
        self.parent_max_chars = parent_max_chars
        # Same clamp as ClauseChunker: an overlap at/above the cap would re-emit the whole previous
        # child, so no child could ever shrink.
        self.overlap = max(0, min(overlap, child_max_chars // 2))
        self._piece_limit = max(1, child_max_chars - self.overlap - 1)

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
        """Children (each with ``metadata["parent_id"]``) followed by parent records.

        When ``text`` has no atomic unit the children are byte-identical to ClauseChunker's at the
        same cap (the atomic-aware path is bypassed entirely)."""
        children = self._children(doc, text, page_map, token_offsets, max_tokens, sections)
        if not children:
            return []
        children, parents = build_parents(doc, children, sections, page_map, self.parent_max_chars)
        return children + parents

    def _children(
        self,
        doc: DocumentEnvelope,
        text: str,
        page_map: list[int] | None,
        token_offsets: list[int] | None,
        max_tokens: int | None,
        sections: list[Section] | None,
    ) -> list[Chunk]:
        spans = atomic_spans(text)
        if not spans:
            # No atomic unit — defer to the clause engine verbatim so output is byte-identical.
            return ClauseChunker(max_chars=self.child_max_chars, overlap=self.overlap).chunk(
                doc, text, page_map=page_map, token_offsets=token_offsets, max_tokens=max_tokens, sections=sections
            )
        return self._atomic_aware_children(doc, text, spans, page_map, token_offsets, max_tokens, sections)

    def _atomic_aware_children(
        self,
        doc: DocumentEnvelope,
        text: str,
        spans: list[tuple[int, int]],
        page_map: list[int] | None,
        token_offsets: list[int] | None,
        max_tokens: int | None,
        sections: list[Section] | None,
    ) -> list[Chunk]:
        units = self._units(text, spans, token_offsets, max_tokens)
        chunks: list[Chunk] = []
        seen: dict[str, int] = {}
        buffer = ""
        buf_start = 0
        buf_end = 0
        body_start = _UNSET

        def emit(chunk_text: str, start: int, end: int, page_at: int) -> None:
            chunks.append(_mk_chunk(doc, seen, chunk_text, start, end, _page_for_offset(page_at, page_map), sections))

        for start, piece, is_atomic in units:
            if is_atomic:
                # Close whatever prose preceded WITHOUT carrying a tail into the atomic child — the
                # child must be exactly the atomic unit. Then emit the unit as its own child, even
                # if it exceeds the cap (bypassing _hard_wrap: a table/code/formula is indivisible).
                if buffer.strip():
                    emit(buffer, buf_start, buf_end, body_start)
                buffer, body_start = "", _UNSET
                emit(piece, start, start + len(piece), start)
                continue

            crosses = body_start != _UNSET and _page_for_offset(start, page_map) != _page_for_offset(
                body_start, page_map
            )
            opens_section = body_start != _UNSET and any(
                body_start < sec.offset <= start for sec in sections or [] if sec.structural
            )
            over_tokens = _tokens_between(token_offsets, buf_start, start + len(piece)) > (max_tokens or 0)
            over_chars = len(buffer) + 1 + len(piece) > self.child_max_chars
            if buffer and (crosses or opens_section or over_tokens or over_chars):
                emit(buffer, buf_start, buf_end, body_start)
                buf_start, buffer = _tail(buffer, buf_end, self.overlap)
                body_start = _UNSET
            if not buffer:
                buf_start = start
            if body_start == _UNSET:
                body_start = start
            buffer = f"{buffer} {piece}".strip() if buffer else piece
            buf_end = start + len(piece)

        if buffer.strip():
            emit(buffer, buf_start, buf_end, body_start)
        return _merge_fragments(doc, chunks, sections, page_map, self.child_max_chars, token_offsets, max_tokens)

    def _units(
        self,
        text: str,
        spans: list[tuple[int, int]],
        token_offsets: list[int] | None,
        max_tokens: int | None,
    ) -> list[tuple[int, str, bool]]:
        """The document as an ordered stream of ``(abs_start, text, is_atomic)`` — non-atomic gaps
        hard-wrapped into clause pieces exactly as ClauseChunker would, each atomic span whole."""
        piece_tokens = _piece_tokens(max_tokens, self.overlap)
        units: list[tuple[int, str, bool]] = []

        def emit_gap(gap_start: int, gap_end: int) -> None:
            gap = text[gap_start:gap_end]
            for rel_s, _rel_e, clause in _clauses(gap):
                for ps, piece in _hard_wrap(gap_start + rel_s, clause, self._piece_limit, token_offsets, piece_tokens):
                    if piece.strip():
                        units.append((ps, piece, False))

        cursor = 0
        for a_start, a_end in spans:
            if a_start > cursor:
                emit_gap(cursor, a_start)
            units.append((a_start, text[a_start:a_end], True))
            cursor = a_end
        if cursor < len(text):
            emit_gap(cursor, len(text))
        return units


def build_parents(
    doc: DocumentEnvelope,
    children: list[Chunk],
    sections: list[Section] | None,
    page_map: list[int] | None,
    parent_max_chars: int,
) -> tuple[list[Chunk], list[Chunk]]:
    """Group consecutive same-section children into bounded parents; stamp each child's parent_id.

    Returns ``(children, parents)``. A parent is a real ``Chunk`` whose text is its children joined
    in reading order, id content-derived (``chunk_id(doc_id, parent_text)``) so it is stable across
    a re-chunk when the text is unchanged, marked ``metadata["chunk_role"]="parent"``. A group of a
    single child gets NO parent (that child is its own context, and a single-child parent would
    collide with the child's id). Each atomic unit lies wholly within one parent because children
    never split an atomic unit and a parent never splits a child."""
    seen: dict[str, int] = {}
    out_children: list[Chunk] = []
    parents: list[Chunk] = []
    for group in _group_children(children, parent_max_chars):
        if len(group) < _MIN_PARENT_CHILDREN:
            out_children.extend(group)  # lone child: no parent, no parent_id
            continue
        parent_text = "\n".join(c.text for c in group)
        start = group[0].span.start if group[0].span else 0
        end = group[-1].span.end if group[-1].span else 0
        parent = _mk_chunk(doc, seen, parent_text, start, end, _page_for_offset(start, page_map), sections)
        parent = parent.model_copy(update={"metadata": {**parent.metadata, META_CHUNK_ROLE: PARENT_ROLE}})
        parents.append(parent)
        for c in group:
            out_children.append(c.model_copy(update={"metadata": {**c.metadata, META_PARENT_ID: parent.chunk_id}}))
    # Re-derive child ids: they are content-derived and the text is unchanged, but the duplicate
    # suffix is scoped to the document, so rebuild them together (as _merge_fragments does).
    id_seen: dict[str, int] = {}
    out_children = [c.model_copy(update={"chunk_id": chunk_id(doc.doc_id, c.text, id_seen)}) for c in out_children]
    return out_children, parents


def _group_children(children: list[Chunk], parent_max_chars: int) -> list[list[Chunk]]:
    """Consecutive children in the SAME section, up to ``parent_max_chars`` of joined text.

    A section change always starts a new parent (a parent must not span two headings, mirroring the
    chunker's own section boundary); within a section, a larger span becomes multiple bounded
    parents and a smaller one becomes a single parent = the section."""
    groups: list[list[Chunk]] = []
    current: list[Chunk] = []
    current_len = 0
    current_section: str | None = None
    for c in children:
        section = c.metadata.get(META_SECTION, "")
        addition = len(c.text) + (1 if current else 0)  # +1 for the join newline
        if current and (section != current_section or current_len + addition > parent_max_chars):
            groups.append(current)
            current, current_len = [], 0
        if not current:
            current_section = section
        current.append(c)
        current_len += len(c.text) + (1 if len(current) > 1 else 0)
    if current:
        groups.append(current)
    return groups
