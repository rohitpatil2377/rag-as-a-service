"""Chunkers (FR-C1). Phase-0 implements FIXED (token-ish, char-based) and CLAUSE.

PARENT_CHILD (FR-C2) lives in :mod:`rag_core.parent_child` and LAYOUT_AWARE (RAG v2 Phase 5)
in :mod:`rag_core.layout_chunking`; both are built by :func:`get_chunker`, which is the only
place a strategy is resolved. CONTEXTUAL is still a stub and raises ``NotImplementedError``.
"""

from __future__ import annotations

import bisect
import hashlib
import re
from collections.abc import Iterator
from typing import TYPE_CHECKING

from rag_core.sections import Section, section_path_for
from rag_schemas import Chunk, DocumentEnvelope
from rag_schemas.chunk import Span
from rag_schemas.common import ChunkStrategy

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from rag_core.layout import LayoutTree
    from rag_core.layout_chunking import LayoutAwareChunker
    from rag_core.parent_child import ParentChildChunker
    from rag_core.section_enrich import SectionEnrichment


def _mk_chunk(
    doc: DocumentEnvelope,
    seen: dict[str, int],
    text: str,
    start: int,
    end: int,
    page: int | None,
    sections: list[Section] | None = None,
) -> Chunk:
    """Build a chunk. ``sections`` are the document's headings, used to record which one
    GOVERNS this chunk — a rule under "6.2.3 Social Media Content" does not apply to a sales
    deck, and without the heading nothing in the chunk says so."""
    metadata = dict(doc.metadata)  # document tags (FR-A7/B7) flow to chunks -> D8 filterable
    path = section_path_for(sections, start, end) if sections else ""
    if path:
        # Absent rather than empty when there is none: no section is a real answer, and an
        # empty string would look like a scope that was checked and found to be nothing.
        metadata[META_SECTION] = path
    return Chunk(
        chunk_id=chunk_id(doc.doc_id, text, seen),
        doc_id=doc.doc_id,
        tenant_id=doc.tenant_id,
        text=text,
        page=page,
        span=Span(start=start, end=end),
        source_uri=doc.uri,
        acl=list(doc.source_acl),
        classification=doc.classification,
        lineage_id=doc.lineage_id,
        metadata=metadata,
    )


# Library defaults, used when a caller injects nothing. The operator-facing values live in
# `rag_common.config` (RAG_CHUNK_SIZE / _OVERLAP / _MAX_CHARS) and are passed in by the
# indexing service — these exist so rag-core is usable on its own, not as a second source.
_DEFAULT_SIZE = 900
_DEFAULT_OVERLAP = 150
_DEFAULT_MAX_CHARS = 700
# Characters of the previous chunk carried into the next. ~10-15% of the cap is the usual
# guidance and is what FixedChunker already used (150 of 900); the clause path carried NONE,
# which is how a rule and the exception qualifying it ended up in different chunks.
_DEFAULT_CLAUSE_OVERLAP = 100
# Below this a chunk is not worth embedding on its own — a lone heading or a two-line tail
# matches nothing and dilutes the index. Measured as 12-15% of pieces once a heading became a
# hard boundary, which is the cost this merge exists to pay back.
_MIN_CHUNK_CHARS = 150

# "this chunk owns no content yet" — distinct from offset 0, which is a real position.
_UNSET = -1

# The heading a chunk sits under. A stable query vocabulary like the extraction provenance
# names — operators filter on it — so it is a module constant, not a knob.
META_SECTION = "section"
META_CATEGORY = "category"  # primary taxonomy tag (FR-B7); federation topic filter reads it
# The document's own effective date (YYYY-MM-DD) from its release block. Stamped on every
# chunk at ingest so version conflict resolution needs no extra lookup per query.
META_DOC_DATE = "doc_date"

# Chunk ids are derived from the chunk's own TEXT, not from its position (decision D2 / P44).
# A positional id ({doc}::{idx}) re-points on every re-chunk, so a stored citation, a HITL
# review row or a graph node silently comes to name different text than when it was written —
# and the audit chain is immutable, so that untruth cannot be corrected. A content id moves only
# when the text it names moves.
#
# 12 hex characters is 48 bits: over a corpus of ~5 400 chunks the chance of any collision is
# around one in a hundred million, and a collision inside one document is handled explicitly
# below rather than left to chance.
_ID_HEX = 12


def chunk_id(doc_id: str, text: str, seen: dict[str, int] | None = None) -> str:
    """``{doc_id}::{sha256(text)[:12]}`` — stable across re-chunking, scoped to the document.

    Scoped because two documents may legitimately hold the same clause: the Code of Conduct is
    reprinted verbatim inside the HR handbook, and those are different chunks that must not
    share an id.

    ``seen`` counts identical texts WITHIN one document so repeated boilerplate does not collide
    — identical ids would overwrite each other on upsert in every store, losing chunks in
    silence. The suffix is positional only among exact duplicates, which is the smallest place
    it can be."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:_ID_HEX]
    if seen is None:
        return f"{doc_id}::{digest}"
    n = seen.get(digest, 0)
    seen[digest] = n + 1
    return f"{doc_id}::{digest}" if n == 0 else f"{doc_id}::{digest}.{n}"


# --- RAG v2 structure ids (upgrade plan §1.1 "Stable IDs") --------------------------------------
# The v2 ingest path carries a section TREE, so a unit's id has to name its position in that tree
# as well as its text: `section_id = hash(doc_id + breadcrumb + ordinal)`,
# `chunk_id = hash(section_id + level + ordinal + text_hash)`, and a block id in the same spirit.
# They live beside `chunk_id` above because they share its one load-bearing property — derived, so
# an id moves only when what it names moves, and a stored citation keeps meaning the same content
# across a re-run.
#
# RELATIONSHIP TO `chunk_id` ABOVE (the plan's "keep compatibility with the existing stable-chunk-ID
# logic"): `chunk_id` is UNCHANGED and keeps producing `{doc_id}::{sha256(text)[:12]}`, so NO
# existing id moves and nothing stored has to be rewritten. The v2 derivations reuse its vocabulary
# — the same sha256, the same `_ID_HEX` width, the same `::` separator, the same scope-prefix idiom
# — and add a one-letter KIND TAG (`s` section / `c` chunk / `b` block) in front of the digest.
#
# That makes a v2 id's trailing field exactly 13 characters. A v1 id's is exactly 12, or 14+ when the
# duplicate-text counter fires (`doc.pdf::e6b2777ab9e5.1`) — never 13. So a v2 id can never EQUAL a
# v1 one and the two generations coexist safely in the same column. The guarantee is the WIDTH, not
# the tag's charset: `s` and `c` happen not to be hex digits, but `b` is.
#
# The parts are joined on ASCII unit separator rather than concatenated, because `+` would make
# (`"1"`, `23`) and (`"12"`, `3`) both `"123"` — the same id for two different sections.
_ID_PART_SEP = "\x1f"

# Characters that must not reach a hash input: C0/C1 controls and DEL, which includes `_ID_PART_SEP`
# itself. See `normalise_breadcrumb`.
_ID_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_ID_WHITESPACE_RUN = re.compile(r"\s+")


def normalise_breadcrumb(breadcrumb: str) -> str:
    """Canonical form of a breadcrumb BEFORE it is hashed (and the form to store in
    ``sections.breadcrumb``, so the column and the id agree).

    Two reasons, both about an id moving when the thing it names has not:

    * Breadcrumbs are built from heading text that comes out of OCR, where a re-run can legitimately
      differ by a double space, a non-breaking space or a stray newline. Unnormalised, that drift
      re-points the section id AND every chunk and block id beneath it — a whole subtree of stored
      citations, for no real change. Whitespace is therefore collapsed to single spaces and trimmed.
    * A control character in heading text would otherwise be hashed verbatim, and a stray
      ``_ID_PART_SEP`` in particular would re-introduce exactly the ambiguity the separator exists to
      prevent. Controls are stripped.

    The other hashed parts need no normalisation and deliberately get none: ``level`` and
    ``block_type`` come from closed vocabularies in our own code, and ``doc_id`` / ``section_id`` are
    ids we minted, so none of them carries document text."""
    return _ID_WHITESPACE_RUN.sub(" ", _ID_CONTROL_CHARS.sub(" ", breadcrumb)).strip()


def _derive_id(kind: str, parts: tuple[str | int, ...]) -> str:
    """``{parts[0]}::{kind}{sha256(parts)[:12]}`` — the first part is both the scope prefix and a
    hash input, so the id is human-traceable to its parent AND collision-scoped to it."""
    joined = _ID_PART_SEP.join(str(p) for p in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:_ID_HEX]
    return f"{parts[0]}::{kind}{digest}"


def text_hash(text: str) -> str:
    """Full sha256 of a unit's text — the stored ``sections.text_hash`` / ``chunks.chunk_hash`` and
    the text component of :func:`structured_chunk_id` / :func:`block_id`.

    Full width, not ``_ID_HEX``: this one is a change DETECTOR compared for equality, never a name
    embedded in a string, so truncating it would buy nothing and lose collision margin.

    Written by ``layout.parse_layout`` into ``sections.text_hash``, and by
    ``layout_chunking`` into a chunk's hash component (Phase 2/5, landed)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def section_id(doc_id: str, breadcrumb: str, ordinal: int) -> str:
    """``{doc_id}::s{12 hex}`` from ``(doc_id, breadcrumb, ordinal)``.

    The breadcrumb (the joined heading path) is in the hash rather than the heading alone, so two
    identically-titled subsections under different parents get different ids; the ordinal separates
    genuine repeats of the same breadcrumb within one document. It is passed through
    :func:`normalise_breadcrumb` first — see there for why.

    Called by ``rag_core.layout.parse_layout`` (Phase 2, landed), whose tree is persisted
    to ``sections`` and joined by a chunk's ``section_id`` (Phase 5)."""
    return _derive_id("s", (doc_id, normalise_breadcrumb(breadcrumb), ordinal))


def structured_chunk_id(section: str, level: str, ordinal: int, text_digest: str) -> str:
    """``{section_id}::c{12 hex}`` from ``(section_id, level, ordinal, text_hash)``.

    ``level`` is in the hash because the same text is legitimately emitted at more than one level
    (a one-paragraph section is its own ``section``-level and ``chunk``-level unit), and those are
    different records that must not overwrite each other. ``text_digest`` comes from
    :func:`text_hash`.

    The v2 chunker (:class:`rag_core.layout_chunking.LayoutAwareChunker`) is the caller. The
    v1 `chunk_id` above still names every chunk the plain-text ingest path produces, and this did
    not change that: the two generations coexist, distinguished by the id width documented above."""
    return _derive_id("c", (section, level, ordinal, text_digest))


def block_id(section: str, block_type: str, ordinal: int, text_digest: str) -> str:
    """``{section_id}::b{12 hex}`` from ``(section_id, block_type, ordinal, text_hash)`` — the same
    spirit as :func:`structured_chunk_id`, with the layout block's type in place of the level, so a
    table and its first row do not collide when the table holds a single row.

    Called by ``rag_core.layout.parse_layout`` (Phase 2, landed); a ``table_row`` block's
    id becomes the ``parent_id`` of its row-level chunk (Phase 5)."""
    return _derive_id("b", (section, block_type, ordinal, text_digest))


class FixedChunker:
    """Fixed-size character windows with overlap. Simple, deterministic, fast."""

    strategy = ChunkStrategy.FIXED

    def __init__(self, size: int = _DEFAULT_SIZE, overlap: int = _DEFAULT_OVERLAP) -> None:
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size = size
        self.overlap = overlap

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
        """See :meth:`ClauseChunker.chunk` for what the token arguments are and why.

        A character window needs them just as much: measured on the real corpus, this chunker
        truncated at embed on 19 chunks DESPITE its 900-character cap, because ToC dot leaders
        run ~1.25 chars/token so 900 characters is ~720 tokens."""
        chunks: list[Chunk] = []
        seen: dict[str, int] = {}
        start = 0
        while start < max(len(text), 1):
            # Whichever limit binds first decides the window: characters for prose, tokens for
            # dense scripts and punctuation runs.
            width = _chars_within_tokens(token_offsets, start, max_tokens, self.size)
            window = text[start : start + width].strip()
            if window:
                page = _page_for_offset(start, page_map)
                chunks.append(_mk_chunk(doc, seen, window, start, start + len(window), page, sections))
            if start + width >= len(text):
                break
            start += max(1, width - self.overlap)
        return chunks


_CLAUSE_RE = re.compile(r"(?<=[.;:\n])\s+")


def _clauses(text: str) -> Iterator[tuple[int, int, str]]:
    """Split into clauses, yielding ``(start, end, clause)`` with REAL offsets into ``text``.

    The offsets come from the same match objects that decided the split, so they cannot
    disagree with it. The previous version split the text and then reconstructed positions with
    ``cursor += len(part) + 1`` — assuming every separator was exactly one character. It is a
    whitespace RUN: two characters at a page boundary (``PAGE_SEPARATOR`` is "\\n\\n"), and more
    wherever a document indents or double-spaces. Every such separator shifted all following
    offsets left by the difference, and the error accumulated down the document, so a citation's
    page number drifted further from the truth the deeper into the text it sat.

    Same failure and same fix as the PII redaction offsets: derive both values from one source
    in one pass, never rebuild one by assuming the other's shape."""
    start = 0
    for match in _CLAUSE_RE.finditer(text):
        yield start, match.start(), text[start : match.start()]
        start = match.end()
    yield start, len(text), text[start:]


def _piece_tokens(max_tokens: int | None, overlap: int) -> int | None:
    """The token budget a single piece may use, reserving room for the carry that will precede
    it. The carry is bounded in CHARACTERS, so the worst case is one token per character —
    conservative for prose, exact for the dense scripts this cap exists to protect."""
    return None if not max_tokens else max(1, max_tokens - overlap)


def _tokens_between(offsets: list[int] | None, start: int, end: int) -> int:
    """How many of the embedder's tokens fall in ``[start, end)``. 0 when no tokenizer is
    available, so the caller's ``or 0`` comparison can never bind."""
    if not offsets:
        return 0
    return bisect.bisect_left(offsets, end) - bisect.bisect_left(offsets, start)


def _chars_within_tokens(offsets: list[int] | None, start: int, max_tokens: int | None, char_limit: int) -> int:
    """How many characters from ``start`` fit inside both limits — the tighter one wins."""
    if not offsets or not max_tokens:
        return char_limit
    first = bisect.bisect_left(offsets, start)
    last = first + max_tokens
    if last >= len(offsets):
        return char_limit  # the whole remaining text is inside the token budget
    return max(1, min(char_limit, offsets[last] - start))


def _tail(chunk: str, chunk_end: int, overlap: int) -> tuple[int, str]:
    """The last ``overlap`` characters of a chunk, on a word boundary, with their real offset.

    Carried into the next chunk so a fact sitting at a boundary keeps the context it qualifies.
    In governance corpora the exception is almost always at the tail of its paragraph — "...
    unless the content references IPO-sensitive information" — and with no carry, a retriever
    that matched the RULE got a chunk ending just before the carve-out.

    Returns the ABSOLUTE start offset of the carried text, so the next chunk's span still
    points at real source. Empty when overlap is off."""
    if overlap <= 0 or not chunk:
        return chunk_end, ""
    tail = chunk[-overlap:]
    cut = tail.find(" ")  # start at a word boundary, never mid-token
    tail = tail[cut + 1 :] if cut != -1 else tail
    return chunk_end - len(tail), tail


def _hard_wrap(
    offset: int,
    text: str,
    limit: int,
    token_offsets: list[int] | None = None,
    max_tokens: int | None = None,
) -> Iterator[tuple[int, str]]:
    """Break one over-long clause into pieces within BOTH the character and token limits.

    Flushing before appending is not enough on its own: a single piece can already exceed the
    budget, exactly as a single clause could already exceed the character cap. Whichever limit
    binds first decides the cut — for prose that is the character cap (~5.1 chars/token), for
    dot leaders the token cap (~1.25).

    Prefers a word boundary and falls back to a hard cut only when a single token is longer
    than the whole limit — a URL, a hash, a long unbroken identifier. Yields absolute offsets so
    a piece can still be located in the source text.

    Without this, a clause longer than the cap is emitted whole and the cap means nothing: the
    document with a 5052-character chunk had no clause boundary to flush on. Nothing is
    discarded — every character is carried into some piece, because a chunker that drops text
    loses it from the index silently."""
    if len(text) <= limit and _tokens_between(token_offsets, offset, offset + len(text)) <= (max_tokens or 0):
        yield offset, text
        return
    at = 0
    while at < len(text):
        room = _chars_within_tokens(token_offsets, offset + at, max_tokens, limit)
        if len(text) - at <= room:
            yield offset + at, text[at:]
            return
        window = text[at : at + room]
        cut = window.rfind(" ")
        if cut <= 0:  # one word longer than the whole window — cut it rather than overshoot
            cut = room
        yield offset + at, text[at : at + cut]
        at += cut
        while at < len(text) and text[at] == " ":  # the separator belongs to neither piece
            at += 1


class ClauseChunker:
    """Clause/sentence-boundary chunker for legal/IT-Act style text (FR-C1)."""

    strategy = ChunkStrategy.CLAUSE

    def __init__(self, max_chars: int = _DEFAULT_MAX_CHARS, overlap: int = _DEFAULT_CLAUSE_OVERLAP) -> None:
        self.max_chars = max_chars
        # Clamped rather than trusted: an overlap at or above the cap would re-emit the whole
        # previous chunk, so every chunk would duplicate its predecessor and the carry could
        # never shrink. Leaves room for at least one new clause in every chunk.
        self.overlap = max(0, min(overlap, max_chars // 2))
        # A piece must leave room for the carry that will precede it (plus the joining space),
        # or overlap would push every chunk past the cap the moment a clause is broken up.
        self._piece_limit = max(1, max_chars - self.overlap - 1)

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
        """``token_offsets`` are the character positions at which the EMBEDDER's tokenizer
        starts each token, and ``max_tokens`` its input limit — both supplied by the caller,
        because the only tokenizer that matters is the one the running model uses and this
        module does no I/O.

        A character cap cannot bound a token budget. Measured: English prose runs ~5.1
        chars/token, serialized table rows ~4.3, ToC dot leaders ~1.25 and Devanagari ~1.6 — so
        one 700-character cap is ~140 tokens of prose and ~560 of dot leaders. The embedder
        truncates at its limit silently (HTTP 200, a normal-looking vector, no header), leaving
        the tail of a chunk in the stored text and in BM25 but invisible to dense search.

        Without them the character cap applies alone, which is all an offline or non-tokenizing
        deployment can promise."""
        chunks: list[Chunk] = []
        seen: dict[str, int] = {}
        buffer = ""
        buf_start = 0
        buf_end = 0
        # Where this chunk's OWN content begins, ignoring any context carried in from the
        # previous chunk. The span must cover the carry (it is really in the text), but the
        # PAGE must not: a chunk whose first characters are borrowed from the page before
        # would otherwise be attributed to that page, and every citation after a page break
        # would name the wrong one.
        body_start: int = _UNSET
        for start, _end, clause in _clauses(text):
            # A clause that is ALREADY longer than the cap has to be broken, or no amount of
            # care about when to flush can keep a chunk inside it.
            for piece_start, piece in _hard_wrap(
                start, clause, self._piece_limit, token_offsets, _piece_tokens(max_tokens, self.overlap)
            ):
                # Flush BEFORE appending, not after. Testing the buffer once the piece was
                # already in it made `max_chars` a floor: a chunk grew to whatever the last
                # clause pushed it to, and on the real corpus 97.6% overshot.
                #
                # ...and flush at a PAGE BOUNDARY too, so no chunk ever OWNS content from two
                # pages. `Chunk.page` is one number, so a chunk spanning a break reports its
                # start page for everything in it and a citation names the wrong page. The
                # carry is deliberately exempt: context may cross a page (that is what keeps a
                # rule with the exception that follows it overleaf), but the citable content
                # may not.
                crosses = body_start != _UNSET and _page_for_offset(piece_start, page_map) != _page_for_offset(
                    body_start, page_map
                )
                # ...and at a HEADING, for the same reason: `Chunk.metadata["section"]` is one
                # value, so a chunk spanning a heading belongs wholly to neither and is left
                # unlabelled. Measured before this change: 28.9% of SOP chunks and 27.2% of RFP
                # chunks straddled one, which is why the scope the generator needs was missing
                # from about a third of them.
                # Only a STRUCTURAL heading may do this. A single-level number ("1.") is a
                # guess — sections.py admits it document-wide once a few genuine ones ascend —
                # and on the HR handbook that guess turned every enumerated list item into a
                # boundary, leaving one ~250-character chunk per item under a 700 cap. No chunk
                # then held a whole list, so "name all six core values" was unanswerable however
                # deep retrieval went. Such a heading still LABELS a chunk that sits cleanly
                # under it; it just no longer decides where chunks end.
                opens_section = body_start != _UNSET and any(
                    body_start < sec.offset <= piece_start for sec in sections or [] if sec.structural
                )
                over_tokens = _tokens_between(token_offsets, buf_start, piece_start + len(piece)) > (max_tokens or 0)
                over_chars = len(buffer) + 1 + len(piece) > self.max_chars
                if buffer and (crosses or opens_section or over_tokens or over_chars):
                    page = _page_for_offset(body_start, page_map)
                    chunks.append(_mk_chunk(doc, seen, buffer, buf_start, buf_end, page, sections))
                    # Carry the tail of the chunk we just closed into the next one, so a fact
                    # at the boundary is not severed from what it qualifies. The carried text
                    # is real source text and its offset is real, so the next chunk's span
                    # still locates it — but the next chunk does not OWN it yet.
                    buf_start, buffer = _tail(buffer, buf_end, self.overlap)
                    body_start = _UNSET
                if not buffer:
                    buf_start = piece_start
                if body_start == _UNSET:
                    body_start = piece_start  # the first piece this chunk actually owns
                buffer = f"{buffer} {piece}".strip() if buffer else piece
                buf_end = piece_start + len(piece)
        if buffer.strip():
            page = _page_for_offset(body_start, page_map)
            chunks.append(_mk_chunk(doc, seen, buffer, buf_start, buf_end, page, sections))
        return _merge_fragments(doc, chunks, sections, page_map, self.max_chars, token_offsets, max_tokens)


def _merge_fragments(
    doc: DocumentEnvelope,
    chunks: list[Chunk],
    sections: list[Section] | None,
    page_map: list[int] | None,
    max_chars: int,
    token_offsets: list[int] | None = None,
    max_tokens: int | None = None,
) -> list[Chunk]:
    """Fold a too-small chunk into the one that FOLLOWS it, within the same section.

    The cost of making a heading a hard boundary, measured on the real corpora before the
    change: +20-42% chunks and 12-15% of pieces under 150 characters. A lone heading, or the
    two-line tail of a section, is embedding-hostile — it matches nothing on its own and
    dilutes the index — and a heading belongs with the text it introduces anyway.

    FORWARD, never backward: a heading merged into what precedes it would land under the
    previous section, recreating the straddling chunk the boundary exists to prevent. A
    fragment with no following chunk in its own section is left alone rather than forced.

    The join must still fit BOTH caps. Characters bound the prompt; TOKENS bound what the
    embedder will actually accept, and a character count is not a safe proxy for them — ToC dot
    leaders run ~1.2 chars/token, so a join that looks small can still be truncated at embed,
    silently. A fragment whose neighbour is already full stays a fragment. The floor scales with
    the cap for the same reason: a 150-character floor under an 80-character cap would merge
    every chunk into the next one, forever.
    """
    if not chunks:
        return chunks
    floor = min(_MIN_CHUNK_CHARS, max_chars // 2)
    merged: list[Chunk] = []
    carry: Chunk | None = None
    for original in chunks:
        chunk = original
        if carry is not None:
            if _same_section(carry, chunk) and _fits(carry, chunk, max_chars, token_offsets, max_tokens):
                chunk = _join(doc, carry, chunk, sections, page_map)
            else:
                merged.append(carry)  # no room, or a different section: keep it as it is
            carry = None
        if len(chunk.text) < floor:
            carry = chunk
            continue
        merged.append(chunk)
    if carry is not None:
        merged.append(carry)  # a trailing fragment has nothing to merge forward into
    # Ids are content-derived, so a merged chunk's id follows its new text automatically; they
    # are only rebuilt here because a duplicate suffix is scoped to the document.
    seen: dict[str, int] = {}
    return [c.model_copy(update={"chunk_id": chunk_id(doc.doc_id, c.text, seen)}) for c in merged]


def _fits(
    first: Chunk,
    second: Chunk,
    max_chars: int,
    token_offsets: list[int] | None,
    max_tokens: int | None,
) -> bool:
    """Would joining these two stay inside both the character cap and the token budget?"""
    if len(first.text) + 1 + len(second.text) > max_chars:
        return False
    if max_tokens is None or first.span is None or second.span is None:
        return True
    return _tokens_between(token_offsets, first.span.start, second.span.end) <= max_tokens


def _same_section(a: Chunk, b: Chunk) -> bool:
    return a.metadata.get(META_SECTION, "") == b.metadata.get(META_SECTION, "")


def _join(
    doc: DocumentEnvelope,
    first: Chunk,
    second: Chunk,
    sections: list[Section] | None,
    page_map: list[int] | None,
) -> Chunk:
    start = first.span.start if first.span else 0
    end = second.span.end if second.span else 0
    page = _page_for_offset(start, page_map)
    return _mk_chunk(doc, {}, f"{first.text}\n{second.text}", start, end, page, sections)


def _page_for_offset(offset: int, page_map: list[int] | None) -> int | None:
    """Map a char offset to a 1-based page using cumulative page lengths.

    CONTRACT: ``page_map[i]`` is the number of characters page ``i+1`` occupies in the joined
    text INCLUDING the separator emitted after it (see ``svc_doc_intelligence.ocr.RenderedText``,
    which guarantees ``sum(page_map) == len(text)``). That is what lets this accumulate without
    knowing the separator — do NOT "correct" it to raw page-text lengths, which would re-introduce
    a per-page offset drift."""
    if not page_map:
        return None
    total = 0
    for page_no, length in enumerate(page_map, start=1):
        total += length
        if offset < total:
            return page_no
    return len(page_map)


def get_chunker(
    strategy: ChunkStrategy | str,
    *,
    size: int | None = None,
    overlap: int | None = None,
    max_chars: int | None = None,
    child_max_chars: int | None = None,
    parent_max_chars: int | None = None,
    layout: LayoutTree | None = None,
    mask: Callable[[str], str] | None = None,
    layout_max_chars: int | None = None,
    layout_card: str = "",
    layout_enrichment: Mapping[str, SectionEnrichment] | None = None,
) -> FixedChunker | ClauseChunker | ParentChildChunker | LayoutAwareChunker:
    """Return a chunker for the requested strategy (FIXED, CLAUSE, PARENT_CHILD, LAYOUT_AWARE).

    Sizing is INJECTED, never read from config here: rag-core stays pure and readable in
    isolation, and the service layer owns the operator's settings — the same arrangement the
    doc-intelligence service uses for its own thresholds. ``None`` keeps the library default,
    so a caller that does not care about tuning does not have to know these exist.

    ``layout`` + ``mask`` are LAYOUT_AWARE's two required inputs (RAG v2 Phase 5): the parsed
    structure tree to chunk, and the PII masker its text must be de-identified through, since the
    tree is captured pre-masking. Both are keyword-only and neither has a default — a layout-aware
    chunker without a tree has nothing to chunk, and one without a masker would put unmasked text
    into the index, so this raises rather than degrading quietly in either direction.

    ``layout_card`` is the Phase-3 profile card prefix (``doc_profile.profile_card``), already
    built and already masked. It DOES have a default, unlike the two above, because a document
    with no profile is an ordinary outcome — the profile call is fail-safe — and "" makes the
    chunker byte-identical to Phase 5.

    ``layout_enrichment`` is the Phase-4 per-section enrichment keyed by ``section_id``, and it
    defaults for the same reason: section enrichment is fail-safe per section, so a partial map
    and an empty one are both ordinary. Empty means no ``section`` and no ``proposition`` units,
    which is Phase 5's output exactly."""
    strategy = ChunkStrategy(strategy)  # coerce str -> enum member (accepts either)
    if strategy is ChunkStrategy.FIXED:
        return FixedChunker(
            size=size if size is not None else _DEFAULT_SIZE,
            overlap=overlap if overlap is not None else _DEFAULT_OVERLAP,
        )
    if strategy is ChunkStrategy.CLAUSE:
        return ClauseChunker(
            max_chars=max_chars if max_chars is not None else _DEFAULT_MAX_CHARS,
            overlap=overlap if overlap is not None else _DEFAULT_CLAUSE_OVERLAP,
        )
    if strategy is ChunkStrategy.PARENT_CHILD:
        # Local import by necessity: parent_child.py reuses this module's primitives (_clauses,
        # _hard_wrap, _merge_fragments, …), so a top-level import here would be circular.
        from rag_core.parent_child import (  # noqa: PLC0415
            _DEFAULT_CHILD_MAX_CHARS,
            _DEFAULT_PARENT_MAX_CHARS,
            ParentChildChunker,
        )

        return ParentChildChunker(
            child_max_chars=child_max_chars if child_max_chars is not None else _DEFAULT_CHILD_MAX_CHARS,
            parent_max_chars=parent_max_chars if parent_max_chars is not None else _DEFAULT_PARENT_MAX_CHARS,
            overlap=overlap if overlap is not None else _DEFAULT_CLAUSE_OVERLAP,
        )
    if strategy is ChunkStrategy.LAYOUT_AWARE:
        # Local import for the same reason PARENT_CHILD's is local: layout_chunking reuses this
        # module's primitives (_clauses, _hard_wrap, structured_chunk_id, ClauseChunker …), so a
        # top-level import here would be circular.
        from rag_core.layout_chunking import (  # noqa: PLC0415
            _DEFAULT_LAYOUT_MAX_CHARS,
            LayoutAwareChunker,
        )

        if layout is None or mask is None:
            raise ValueError("LAYOUT_AWARE chunking requires both a parsed layout tree and a PII masker")
        return LayoutAwareChunker(
            tree=layout,
            mask=mask,
            # Built with the sizing a plain CLAUSE run would get, so the blockless-tree fallback is
            # byte-identical to today's output rather than merely similar.
            fallback=ClauseChunker(
                max_chars=max_chars if max_chars is not None else _DEFAULT_MAX_CHARS,
                overlap=overlap if overlap is not None else _DEFAULT_CLAUSE_OVERLAP,
            ),
            max_chars=layout_max_chars if layout_max_chars is not None else _DEFAULT_LAYOUT_MAX_CHARS,
            card=layout_card,
            enrichment=layout_enrichment,
        )
    # TODO(Phase-1): CONTEXTUAL (FR-C2).
    raise NotImplementedError(f"chunk strategy {strategy} is a Phase-1 stub")
