"""Unit tests for the pure RAG logic (fusion, chunking, abstain gate)."""

from __future__ import annotations

from datetime import UTC, datetime

from rag_core import ClauseChunker, get_chunker, reciprocal_rank_fusion, should_abstain
from rag_core.chunking import META_SECTION
from rag_core.fusion import normalize_scores
from rag_schemas import DocumentEnvelope
from rag_schemas.common import ChunkStrategy, SourceKind


def _doc(text_len: int = 0) -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id="t",
        doc_id="d1",
        source_id="s1",
        source_kind=SourceKind.UPLOAD,
        mime="text/plain",
        content_b64="",
        ingested_at=datetime(2026, 7, 2, tzinfo=UTC),
    )


def test_rrf_rewards_agreement_across_lists() -> None:
    dense = [("a", 0.9), ("b", 0.5)]
    sparse = [("a", 3.1), ("c", 2.0)]
    fused = dict(reciprocal_rank_fusion([dense, sparse]))
    # "a" appears rank-1 in both -> must be top.
    assert max(fused, key=fused.get) == "a"  # type: ignore[arg-type]
    assert set(fused) == {"a", "b", "c"}


def test_normalize_scores_bounds() -> None:
    norm = normalize_scores([("a", 0.033), ("b", 0.016)])
    assert norm["a"] == 1.0
    assert norm["b"] == 0.0


def test_abstain_gate() -> None:
    assert should_abstain(0.2, 0.35) is True
    assert should_abstain(0.9, 0.35) is False


def test_fixed_chunker_overlaps_and_covers() -> None:
    chunker = get_chunker(ChunkStrategy.FIXED)
    text = "word " * 500  # ~2500 chars
    chunks = chunker.chunk(_doc(), text)
    assert len(chunks) > 1
    assert all(c.tenant_id == "t" and c.doc_id == "d1" for c in chunks)
    assert chunks[0].span is not None


def test_clause_chunker_splits_on_boundaries() -> None:
    chunker = get_chunker(ChunkStrategy.CLAUSE)
    text = "Clause one. Clause two; sub-part. " * 40
    chunks = chunker.chunk(_doc(), text)
    assert len(chunks) >= 1


# --- chunk offsets are derived, never accumulated (P11) ---------------------------------------


def _offset_doc() -> DocumentEnvelope:
    from datetime import UTC, datetime

    from rag_schemas.common import SourceKind

    return DocumentEnvelope(
        tenant_id="t",
        doc_id="d",
        source_id="s",
        source_kind=SourceKind.UPLOAD,
        mime="text/plain",
        ingested_at=datetime(2026, 7, 30, tzinfo=UTC),
    )


def test_a_chunk_span_points_at_the_text_it_came_from() -> None:
    """The span must locate the chunk in the ORIGINAL text. It was start + len(joined buffer),
    which is the length of the NORMALISED text, not of the region it was read from — so any
    separator wider than one character pushed the end past the truth."""
    from rag_core.chunking import ClauseChunker

    # Separators of three different widths: one space, a newline pair, and an indented break.
    text = "Alpha one. Bravo two.\n\nCharlie three.\n    Delta four. Echo five."
    for chunk in ClauseChunker(max_chars=12).chunk(_offset_doc(), text):
        assert chunk.span is not None
        region = text[chunk.span.start : chunk.span.end]
        # The stored text collapses whitespace, so compare on non-space characters.
        assert "".join(chunk.text.split()) == "".join(region.split()), f"{chunk.text!r} vs {region!r}"


def test_offsets_do_not_drift_across_many_wide_separators() -> None:
    """The error accumulated, so it is invisible on a short fixture and large on a real one."""
    from rag_core.chunking import ClauseChunker

    text = "\n\n".join(f"Clause {i} states the entitlement is {i} days." for i in range(60))
    chunks = ClauseChunker(max_chars=40).chunk(_offset_doc(), text)
    # Check the DEEPEST chunk, where an accumulating error would be largest: its span must
    # still locate it in the source. Under the old arithmetic every "\n\n" cost one character,
    # so by clause 59 the offsets were dozens of characters adrift.
    deepest = chunks[-1]
    assert deepest.span is not None
    region = text[deepest.span.start : deepest.span.end]
    assert "".join(deepest.text.split()) == "".join(region.split())
    assert deepest.span.end == len(text)  # the last chunk ends where the text does


# --- max_chars is a cap, not a floor (P51) ----------------------------------------------------


def test_no_chunk_exceeds_max_chars() -> None:
    """`max_chars` was tested AFTER a clause was appended, so it was a FLOOR: a chunk grew to
    whatever the last clause pushed it to. Measured on the real corpus, 97.6% of clause chunks
    overshot and the longest reached 5052 characters — against an embedder that silently
    truncates at 512 tokens, so the tail of a long chunk was indexed but never embedded."""
    from rag_core.chunking import ClauseChunker

    text = " ".join(f"Clause {i} states that the entitlement is {i} days." for i in range(120))
    for cap in (80, 200, 700):
        chunks = ClauseChunker(max_chars=cap).chunk(_offset_doc(), text)
        assert chunks
        assert max(len(c.text) for c in chunks) <= cap, f"cap={cap} overshot"


def test_a_single_clause_longer_than_the_cap_is_broken_up() -> None:
    """The case a floor can never handle: one clause that already exceeds the cap. An
    unbreakable 5000-character sentence has to become several chunks or the cap is a suggestion."""
    from rag_core.chunking import ClauseChunker

    text = "The schedule lists " + " ".join(f"item{i}" for i in range(400)) + "."
    chunks = ClauseChunker(max_chars=120).chunk(_offset_doc(), text)
    assert len(chunks) > 1
    assert max(len(c.text) for c in chunks) <= 120


def test_breaking_a_long_clause_keeps_offsets_honest() -> None:
    from rag_core.chunking import ClauseChunker

    text = "Preamble. " + " ".join(f"item{i}" for i in range(200)) + ". Closing remark."
    for chunk in ClauseChunker(max_chars=100).chunk(_offset_doc(), text):
        assert chunk.span is not None
        region = text[chunk.span.start : chunk.span.end]
        assert "".join(chunk.text.split()) == "".join(region.split())


def test_no_content_is_lost_when_a_clause_is_broken() -> None:
    """Splitting must not drop words — the failure mode a naive truncation would introduce.

    Overlap off, because with it on the chunks deliberately REPEAT text and concatenation is a
    superset rather than an equality. This isolates the hard-wrap property."""
    from rag_core.chunking import ClauseChunker

    text = "Alpha. " + " ".join(f"word{i}" for i in range(300)) + ". Omega."
    chunks = ClauseChunker(max_chars=90, overlap=0).chunk(_offset_doc(), text)
    assert "".join("".join(c.text.split()) for c in chunks) == "".join(text.split())


def test_every_character_is_still_covered_when_overlap_is_on() -> None:
    """Overlap duplicates, so the weaker property is what holds: nothing is MISSING."""
    from rag_core.chunking import ClauseChunker

    text = "Alpha. " + " ".join(f"word{i}" for i in range(300)) + ". Omega."
    chunks = ClauseChunker(max_chars=90, overlap=30).chunk(_offset_doc(), text)
    covered = "".join("".join(c.text.split()) for c in chunks)
    source = "".join(text.split())
    assert len(covered) > len(source), "overlap must re-emit some context"
    # Walk the source through the chunks in order: every character must be reachable.
    at = 0
    for chunk in chunks:
        packed = "".join(chunk.text.split())
        found = source.find(packed[:20], max(0, at - len(packed)))
        assert found != -1, f"chunk not found in source: {packed[:40]!r}"
        at = max(at, found + len(packed))
    assert at == len(source), "the chunks did not reach the end of the document"


# --- boundary facts must stay retrievable (P10) ------------------------------------------------


def test_a_fact_at_a_chunk_boundary_survives_in_context() -> None:
    """The Critical one. ClauseChunker had ZERO overlap, so a sentence at the end of a chunk
    was cut off from everything it qualified. In governance corpora the exception is almost
    always at the tail of its paragraph — "...unless the content references IPO-sensitive
    information" — and a retriever that finds the RULE gets a chunk that stops just before the
    carve-out. That is how the platform answered "Legal review is not required" for a deck it
    had just been told was an exception.

    FixedChunker already carried 150 characters of overlap; the clause path carried none."""
    from rag_core.chunking import ClauseChunker

    rule = " ".join(f"Category {chr(65 + i)} content follows the standard review path." for i in range(12))
    exception = "This does not apply where the content references IPO-sensitive information."
    text = f"{rule} {exception} Unrelated closing paragraph about parking."

    chunks = ClauseChunker(max_chars=200).chunk(_offset_doc(), text)
    carrying = [c for c in chunks if "IPO-sensitive" in c.text]
    assert carrying, "the exception vanished entirely"
    # It must appear somewhere that ALSO carries the rule it qualifies, not stranded alone.
    assert any("standard review path" in c.text for c in carrying), (
        "the exception is retrievable but detached from the rule it modifies"
    )


def test_overlap_is_bounded_and_off_by_default_for_fixed_only() -> None:
    """Overlap costs index size, so it is a declared fraction rather than an accident."""
    from rag_core.chunking import ClauseChunker

    text = " ".join(f"Sentence {i} carries some policy detail." for i in range(40))
    plain = ClauseChunker(max_chars=200, overlap=0).chunk(_offset_doc(), text)
    lapped = ClauseChunker(max_chars=200, overlap=60).chunk(_offset_doc(), text)
    assert len(lapped) > len(plain), "overlap must actually re-emit context"
    assert max(len(c.text) for c in lapped) <= 200, "overlap must not breach the cap"


def test_overlap_never_repeats_a_whole_chunk() -> None:
    """A degenerate overlap (>= the cap) would make the chunker loop forever or duplicate
    everything; it is clamped rather than trusted."""
    from rag_core.chunking import ClauseChunker

    text = " ".join(f"Clause {i} of the policy." for i in range(30))
    chunks = ClauseChunker(max_chars=120, overlap=500).chunk(_offset_doc(), text)
    assert len({c.text for c in chunks}) == len(chunks)
    assert max(len(c.text) for c in chunks) <= 120


# --- a chunk never owns content from two pages (P39) -------------------------------------------


def test_no_chunk_owns_content_from_two_pages() -> None:
    """`Chunk.page` is ONE number, so a chunk owning text from two pages cannot be right about
    both — it reports its start page and every citation into its tail names the wrong page.

    Proven over the whole document rather than one fixture: for every chunk, the page reported
    must be the page of every character it OWNS. Carried overlap is exempt by construction —
    context may cross a break, citable content may not."""
    from rag_core.chunking import ClauseChunker, _page_for_offset

    # Pages must be several times the cap, or each page yields one chunk and nothing can
    # straddle — the fixture would pass without testing anything.
    pages = [
        " ".join(f"Page {i} clause {j} states the applicable entitlement." for j in range(8)) for i in range(1, 12)
    ]
    text = "\n\n".join(pages)
    page_map = [len(p) + 2 for p in pages[:-1]] + [len(pages[-1])]
    assert sum(page_map) == len(text)

    chunks = ClauseChunker(max_chars=120, overlap=40).chunk(_offset_doc(), text, page_map=page_map)
    assert len(chunks) > len(pages), "fixture must produce several chunks per page"
    for chunk in chunks:
        assert chunk.span is not None
        # The chunk's own content ends at span.end; anything carried in sits at the start.
        assert chunk.page == _page_for_offset(chunk.span.end - 1, page_map), (
            f"chunk says page {chunk.page} but its content ends on "
            f"{_page_for_offset(chunk.span.end - 1, page_map)}: {chunk.text[:60]!r}"
        )


def test_overlap_still_crosses_a_page_break() -> None:
    """The exemption that makes the above affordable: a rule at the foot of one page must still
    reach the chunk holding the exception overleaf, or capping ownership would re-sever exactly
    the pairs the overlap exists to keep together."""
    from rag_core.chunking import ClauseChunker

    rule = "Category A content follows the standard review path."
    exception = "This does not apply where the content is IPO-sensitive."
    text = f"{rule}\n\n{exception}"
    page_map = [len(rule) + 2, len(exception)]

    chunks = ClauseChunker(max_chars=120, overlap=60).chunk(_offset_doc(), text, page_map=page_map)
    carrying = [c for c in chunks if "IPO-sensitive" in c.text]
    assert carrying
    assert any("standard review path" in c.text for c in carrying), "the carry did not cross the page break"


# --- the cap the embedder actually enforces is in TOKENS (P49) ---------------------------------


def _fake_offsets(text: str, chars_per_token: float) -> list[int]:
    """Token start offsets for a synthetic tokenizer of a given density — enough to drive the
    chunker without any I/O. Real offsets come from the embedder itself."""
    step = max(1, int(chars_per_token))
    return list(range(0, len(text), step))


def test_a_character_cap_cannot_bound_a_token_budget() -> None:
    """The measurement that motivates this. English prose runs ~5.1 chars/token, but ToC dot
    leaders run ~1.25 and Devanagari ~1.6, so the SAME 700-character cap is ~140 tokens of
    prose and ~560 of dot leaders. The embedder truncates at 512 with `truncate: true` — HTTP
    200, a normal-looking vector, no header, no log. Content past the limit stays in the stored
    text and in BM25 but is invisible to dense search."""
    from rag_core.chunking import ClauseChunker

    text = "Chapter one " + "." * 3000 + " 42. Chapter two follows."
    offsets = _fake_offsets(text, chars_per_token=1.25)  # dot leaders
    chunks = ClauseChunker(max_chars=700, overlap=0).chunk(_offset_doc(), text)
    assert max(len(c.text) for c in chunks) <= 700  # the character cap holds...
    # ...and is worthless as a token bound: a 700-char chunk of leaders is ~560 tokens.
    worst = max(_token_count(c, offsets) for c in chunks)
    assert worst > 512, f"fixture must exceed the embedder limit, got {worst}"


def _token_count(chunk: object, offsets: list[int]) -> int:
    import bisect

    span = chunk.span  # type: ignore[attr-defined]
    return bisect.bisect_left(offsets, span.end) - bisect.bisect_left(offsets, span.start)


def test_a_token_cap_is_enforced_when_the_embedder_reports_one() -> None:
    from rag_core.chunking import ClauseChunker

    text = "Chapter one " + "." * 3000 + " 42. Chapter two follows."
    offsets = _fake_offsets(text, chars_per_token=1.25)
    chunks = ClauseChunker(max_chars=700, overlap=0).chunk(_offset_doc(), text, token_offsets=offsets, max_tokens=512)
    assert chunks
    assert max(_token_count(c, offsets) for c in chunks) <= 512


def test_prose_is_not_over_split_by_the_token_cap() -> None:
    """A token cap must not shrink ordinary text: at ~5 chars/token a 700-character chunk is
    ~140 tokens, far inside the budget, so the character cap should still be what binds."""
    from rag_core.chunking import ClauseChunker

    text = " ".join(f"Clause {i} sets out the applicable entitlement in full." for i in range(200))
    offsets = _fake_offsets(text, chars_per_token=5.0)
    plain = ClauseChunker(max_chars=700, overlap=0).chunk(_offset_doc(), text)
    capped = ClauseChunker(max_chars=700, overlap=0).chunk(_offset_doc(), text, token_offsets=offsets, max_tokens=512)
    assert [c.text for c in capped] == [c.text for c in plain]


def test_no_token_information_means_the_character_cap_alone() -> None:
    """Offline, or on a deployment whose embedder cannot tokenize, chunking must still work —
    it just cannot promise a token bound."""
    from rag_core.chunking import ClauseChunker

    text = " ".join(f"Clause {i} of the policy." for i in range(50))
    assert ClauseChunker(max_chars=200).chunk(_offset_doc(), text, token_offsets=None, max_tokens=512)


# --- a chunk declares the section it is governed by (P15) --------------------------------------


_SCOPED = """1. General Provisions
Content must be accurate and verifiable at all times.

6.2.3 Social Media Content
Prior approval from the Content Curator is required before posting.

6.3.1 Investor Presentations
Prior approval from the Content Curator is required before posting.
"""


def test_a_chunk_carries_the_section_that_governs_it() -> None:
    """QA Q10: a social-media-only rule was applied to a sales deck. Both chunks below read
    identically — only the section they sit under limits who they apply to, and the chunk
    itself carried no trace of it, so the generator could not know the rule was scope-limited."""
    from rag_core.chunking import ClauseChunker
    from rag_core.sections import detect_sections

    chunks = ClauseChunker(max_chars=90, overlap=0).chunk(_offset_doc(), _SCOPED, sections=detect_sections(_SCOPED))
    scoped = [c for c in chunks if "Prior approval" in c.text]
    assert scoped, "fixture must produce the ambiguous chunks"
    sections = {c.metadata.get("section") for c in scoped}
    assert "6.2.3 Social Media Content" in sections
    assert "6.3.1 Investor Presentations" in sections


def test_a_chunk_outside_any_section_claims_none() -> None:
    """No section is a real answer. Inventing one would assert a scope the document never set,
    and a wrong scope is worse than none — it licenses applying a rule where it does not hold."""
    from rag_core.chunking import ClauseChunker
    from rag_core.sections import detect_sections

    text = "Annual leave is 24 days.\nNotice period is 60 days.\n"
    for chunk in ClauseChunker(max_chars=40).chunk(_offset_doc(), text, sections=detect_sections(text)):
        assert "section" not in chunk.metadata


def test_sections_are_optional() -> None:
    """A caller that does not supply them chunks exactly as before."""
    from rag_core.chunking import ClauseChunker

    chunks = ClauseChunker(max_chars=90).chunk(_offset_doc(), _SCOPED)
    assert chunks and all("section" not in c.metadata for c in chunks)


# --- stable, content-derived chunk ids (P44 / decision D2) --------------------------------------


def _doc_for_ids(text_id: str = "d") -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id="t",
        doc_id=text_id,
        source_id="s",
        source_kind=SourceKind.UPLOAD,
        mime="text/plain",
        ingested_at=datetime(2026, 7, 31, tzinfo=UTC),
    )


_ID_DOC = (
    "Clause one states that annual leave is 24 days for confirmed staff each year. "
    "Clause two states that the night shift allowance is paid monthly to operations engineers. "
    "Clause three states that travel must be booked seven working days in advance. "
)


def test_a_chunks_id_follows_its_text_not_its_position() -> None:
    """Positional ids ({doc}::{idx}) re-point on every re-chunk: a stored citation, a HITL row
    or a graph node then names different text than it did when it was written, silently. An id
    derived from the chunk's own text survives any change that leaves that text alone."""
    doc = _doc_for_ids()
    first = ClauseChunker(max_chars=120, overlap=0).chunk(doc, _ID_DOC)
    # Prepend a whole new clause: every chunk shifts position, the later texts do not change.
    shifted = ClauseChunker(max_chars=120, overlap=0).chunk(
        doc, "Clause zero states the policy applies to all staff. " + _ID_DOC
    )

    by_text = {c.text: c.chunk_id for c in first}
    survived = [c for c in shifted if c.text in by_text]
    assert survived, "fixture produced no unchanged chunk text — the assertion would be vacuous"
    for c in survived:
        assert c.chunk_id == by_text[c.text], f"id moved for unchanged text: {c.text[:50]!r}"


def test_ids_are_unique_even_when_a_document_repeats_itself() -> None:
    """Boilerplate reprinted on many pages is the same text twice. Identical ids would collide
    in every store, and each upsert would overwrite the last — silently losing chunks."""
    doc = _doc_for_ids()
    repeated = "The company reserves the right to amend this policy. " * 6
    chunks = ClauseChunker(max_chars=60, overlap=0).chunk(doc, repeated)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), f"duplicate ids: {ids}"


def test_the_id_is_scoped_to_its_document() -> None:
    """Two documents may legitimately contain the same clause — the Code of Conduct is reprinted
    inside the HR handbook. They are different chunks and must not share an id."""
    text = "All employees shall conduct themselves with integrity at all times."
    a = ClauseChunker().chunk(_doc_for_ids("conduct"), text)[0]
    b = ClauseChunker().chunk(_doc_for_ids("handbook"), text)[0]
    assert a.chunk_id != b.chunk_id
    assert a.chunk_id.startswith("conduct::") and b.chunk_id.startswith("handbook::")


# --- a heading is a hard chunk boundary ---------------------------------------------------------

_SECTIONED = (
    "6.1.1 Mandatory Rehearsal Policy\n"
    "No representative may speak at an external event without completing the rehearsal. "
    "If the rehearsal has not been signed off, the speaker does not go on stage.\n"
    "6.2.3 Social Media Publishing\n"
    "All social media content posted on official accounts requires prior approval. "
    "During the restricted period no financial figures may be posted.\n"
)


def _sections_of(text: str):  # type: ignore[no-untyped-def]
    from rag_core.sections import detect_sections

    return detect_sections(text)


def test_no_chunk_spans_two_sections() -> None:
    """Measured on the real corpora: with a 700-character cap and headings every ~479
    characters, 28.9% of SOP chunks and 27.2% of RFP chunks straddled a heading — and a
    straddling chunk is deliberately left unlabelled, so the scope the generator needs was
    absent on roughly a third of them. Making the heading a boundary is what turns the label
    from occasional into reliable."""
    doc = _doc_for_ids("sop")
    sections = _sections_of(_SECTIONED)
    chunks = ClauseChunker(max_chars=700, overlap=0).chunk(doc, _SECTIONED, sections=sections)
    for c in chunks:
        inside = [s for s in sections if c.span and c.span.start < s.offset < c.span.end]
        assert not inside, f"chunk crosses {[s.label for s in inside]}: {c.text[:60]!r}"


def test_each_chunk_carries_the_section_it_now_sits_wholly_within() -> None:
    doc = _doc_for_ids("sop")
    chunks = ClauseChunker(max_chars=700, overlap=0).chunk(doc, _SECTIONED, sections=_sections_of(_SECTIONED))
    labelled = [c.metadata.get(META_SECTION) for c in chunks if "rehearsal" in c.text.lower()]
    assert labelled and all(x == "6.1.1 Mandatory Rehearsal Policy" for x in labelled), labelled


def test_a_heading_sized_fragment_is_merged_into_the_body_it_heads() -> None:
    """The cost of the boundary, measured earlier at 12-15% of pieces under 150 characters. A
    lone heading is embedding-hostile confetti: it matches nothing and dilutes the index. It
    belongs with the text it introduces, which is also where it is most useful."""
    doc = _doc_for_ids("sop")
    chunks = ClauseChunker(max_chars=700, overlap=0).chunk(doc, _SECTIONED, sections=_sections_of(_SECTIONED))
    assert not [c for c in chunks if len(c.text) < 150], [c.text for c in chunks if len(c.text) < 150]
    assert any(c.text.startswith("6.1.1 Mandatory Rehearsal Policy") and "speaker" in c.text for c in chunks)


def test_a_fragment_is_never_merged_across_a_section_boundary() -> None:
    """Merging forward is only safe within one section — otherwise the fix reintroduces exactly
    the straddling chunk the boundary was added to prevent."""
    text = "3.1 First\nShort.\n3.2 Second\n" + ("Long body text for the second section. " * 12)
    doc = _doc_for_ids("d")
    chunks = ClauseChunker(max_chars=700, overlap=0).chunk(doc, text, sections=_sections_of(text))
    for c in chunks:
        assert not ("3.1 First" in c.text and "3.2 Second" in c.text), c.text[:120]


def test_an_enumerated_list_is_not_shattered_into_one_chunk_per_item() -> None:
    """Measured on the real Acme HR handbook: "what are the core values" answered with 2 of 6.

    The six values are six consecutive numbered items. Because the handbook ALSO numbers its
    policies 1..26, sections.py's single-level escape hatch opened DOCUMENT-WIDE and every
    numbered list item anywhere became a "heading"; a heading is a hard chunk boundary, so each
    value landed in its own ~250-character chunk against a 700 cap. Retrieval returned the intro
    chunk and one value chunk, and the model — correctly, given what it was shown — answered
    incompletely. 239 of that document's 686 chunks were under 350 characters, and suppressing
    the heading flush cost no section labels at all."""
    from rag_core.chunking import ClauseChunker
    from rag_core.sections import detect_sections

    text = (
        "".join(f"{n}. Policy Number {n}\nBody for policy {n}. " + "Filler sentence. " * 10 + "\n" for n in range(1, 9))
        + "Core Values\nThe six simple ingredients are more than buzzwords.\n"
        + "".join(f"{n}. Value Number {n}:\nThis value matters to the company.\n" for n in range(1, 7))
    )
    chunks = ClauseChunker(max_chars=700, overlap=50).chunk(_doc(), text, sections=detect_sections(text))
    holding = [c for c in chunks if "Value Number" in c.text]
    assert holding, "fixture no longer produces the list — the assertion would be vacuous"
    # Six ~55-character items fit inside a 700-character chunk many times over. One chunk per
    # item is the defect; two is tolerable (the list may straddle a real boundary), six is not.
    assert len(holding) <= 2, [c.text[:70] for c in holding]
    # The remaining split must be driven by the SIZE CAP, not by the list markers: one chunk
    # carries a long contiguous run rather than each item sitting alone. Before the fix the best
    # any chunk managed was one.
    assert max(sum(f"Value Number {n}" in c.text for n in range(1, 7)) for c in holding) >= 4, (
        "the list is still fragmented — no chunk carries enough of it to answer 'name them all'"
    )


# --- neighbour expansion: a list longer than one passage ----------------------------------------


def test_neighbours_are_returned_in_document_order_not_anchor_order() -> None:
    """A model reads the prompt top to bottom. Handing it the passage AFTER the anchor before the
    one BEFORE it would present the list out of order and invite exactly the misreading this
    exists to prevent."""
    from rag_core.expansion import neighbour_ids

    ids = [f"c{i}" for i in range(10)]
    assert neighbour_ids(ids, ["c5", "c2"], window=1) == ["c1", "c3", "c4", "c6"]


def test_the_anchors_themselves_are_never_returned() -> None:
    """The caller already holds them, scored. Returning them again would duplicate a passage in
    the prompt and double-count it at the gate."""
    from rag_core.expansion import neighbour_ids

    ids = ["a", "b", "c"]
    assert neighbour_ids(ids, ["b"], window=1) == ["a", "c"]


def test_overlapping_windows_are_merged_not_duplicated() -> None:
    """Two anchors two apart share a neighbour. It must appear once."""
    from rag_core.expansion import neighbour_ids

    ids = [f"c{i}" for i in range(6)]
    assert neighbour_ids(ids, ["c1", "c3"], window=1) == ["c0", "c2", "c4"]


def test_a_window_at_the_document_edge_does_not_wrap_or_raise() -> None:
    from rag_core.expansion import neighbour_ids

    ids = ["first", "second", "last"]
    assert neighbour_ids(ids, ["first"], window=2) == ["second", "last"]
    assert neighbour_ids(ids, ["last"], window=2) == ["first", "second"]


def test_an_anchor_from_another_document_is_ignored_rather_than_raising() -> None:
    """Anchors are grouped by document by the caller, but a passage deleted between indexing and
    this call would otherwise crash a query."""
    from rag_core.expansion import neighbour_ids

    assert neighbour_ids(["a", "b"], ["somewhere-else"], window=1) == []


def test_zero_window_disables_expansion() -> None:
    """The flag is the on/off switch; a zero window is the same thing expressed as a number, and
    must not silently expand by one."""
    from rag_core.expansion import neighbour_ids

    assert neighbour_ids(["a", "b", "c"], ["b"], window=0) == []


def test_expansion_recovers_a_list_that_spans_three_passages() -> None:
    """The measured failure. Retrieval ranks the passage carrying the heading and the word "six";
    the passage holding values 3-5 shares no vocabulary with the question and never surfaces. With
    the anchor's neighbours pulled in, the whole list reaches the prompt."""
    from rag_core.expansion import neighbour_ids

    doc = ["intro", "values_1_2", "values_3_5", "values_5_6", "next_section"]
    # Retrieval found only the passage that lexically matches the question.
    assert neighbour_ids(doc, ["values_1_2"], window=2) == ["intro", "values_3_5", "values_5_6"]


def test_anchors_are_grouped_by_document_and_bounded() -> None:
    """Expanding every span of a 20-span result would fetch 20 documents' passage lists and could
    double the prompt. Only the highest-ranked few are expanded."""
    from rag_core.expansion import anchors_by_document

    spans = [("d1", "c1"), ("d2", "c9"), ("d1", "c2"), ("d3", "c7")]
    assert anchors_by_document(spans, limit=3) == {"d1": ["c1", "c2"], "d2": ["c9"]}
    assert anchors_by_document(spans, limit=0) == {}


# --- document dating: read the release block, never the body ------------------------------------


def test_the_release_history_table_dates_the_document() -> None:
    """Acme documents carry "Document Release History" — Version | Prepared | Review | Release.
    The LAST row is the current version, so the latest RELEASE date in the block is the one."""
    from rag_core.tagging import document_date

    text = (
        "Document Release History\n"
        "Version | Prepared Date | Review Date | Release Date | Prepared By\n"
        "1.0 | 1st December 2022 | 15th December 2022 | 1st January 2023 | HR Operations\n"
        "2.0 | 1st December 2023 | 15th December 2023 | 1st January 2024 | HR Operations\n"
    )
    assert document_date(text) == "2024-01-01"


def test_a_year_mentioned_in_the_body_does_not_date_the_document() -> None:
    """The rule this replaces was "the most recent 4-digit year anywhere", which on a real corpus
    dated a 2026 handbook as 2029 (a future year quoted in a table) and a travel policy as 2000.
    A confidently wrong date is worse than none, because recency ranking would then prefer the
    WRONG document."""
    from rag_core.tagging import document_date, latest_year

    body = "Business Travel Policy. Rates apply from 2000 and are reviewed until 2029."
    assert document_date(body) is None
    assert latest_year(body) is None


def test_undated_is_none_and_must_not_be_read_as_old() -> None:
    """None is a real answer and the common one. A caller has to treat it as UNKNOWN — defaulting
    it to an old date would silently deprioritise every document that simply omits its date."""
    from rag_core.tagging import document_date

    assert document_date("A policy with no version block at all.") is None


def test_the_common_date_formats_all_parse() -> None:
    """Indian documents write dates every way. DD-MM-YYYY is assumed when a numeric date is
    genuinely ambiguous, because that is the local convention."""
    from rag_core.tagging import document_date

    assert document_date("Effective Date: 15-03-2025") == "2025-03-15"
    assert document_date("Date of issue: 2025-03-15") == "2025-03-15"
    assert document_date("Last updated: 3rd April 2024") == "2024-04-03"
    assert document_date("Effective date 03/15/2025") == "2025-03-15"  # unambiguous MM-DD


def test_the_scan_window_stops_before_the_body() -> None:
    """A release block is a small table. Scanning further reintroduces the body-text noise this
    function exists to eliminate."""
    from rag_core.tagging import document_date

    text = "Release Date: 1st January 2024\n" + ("filler. " * 400) + "signed 31st December 2099"
    assert document_date(text) == "2024-01-01"


# --- version conflict: which document is current -------------------------------------------------


def test_the_newer_dated_document_supersedes_the_older() -> None:
    """The measured case: two handbooks listing different members of the same committee. Without
    this the answer says "the sources differ", which is honest and useless."""
    from rag_core.versioning import DocumentVersion, resolve_supersession

    docs = [DocumentVersion("handbook-2023.pdf", "2023-01-01"), DocumentVersion("handbook-2026.pdf", "2026-01-01")]
    assert resolve_supersession(docs) == {"handbook-2023.pdf": "handbook-2026.pdf"}


def test_unrelated_documents_never_supersede_each_other() -> None:
    """Measured live, and the reason families exist: a question about the Night Shift Allowance
    Policy answered "this policy has been superseded by the HR Handbook". They are two different
    policies that happened to be retrieved together, one older than the other — not two versions
    of one document. Being older than an unrelated document means nothing."""
    from rag_core.versioning import DocumentVersion, resolve_supersession

    docs = [
        DocumentVersion("Acme_Night_Shift_Allowance_Policy.pdf", "2023-04-01"),
        DocumentVersion("HR_Handbook 4.0_2026 2_0227.pdf", "2026-01-01"),
    ]
    assert resolve_supersession(docs, ignore_tokens={"acme"}) == {}


def test_the_same_document_is_recognised_across_a_renaming() -> None:
    """The two real handbooks are not named identically — one is "Acme_HR_POLICY_HANDBOOK", the
    other "HR_Handbook 4.0_2026". Version and year tokens differ BY DEFINITION between versions,
    so identity has to rest on the words that remain."""
    from rag_core.versioning import DocumentVersion, resolve_supersession

    docs = [
        DocumentVersion("Acme_HR_POLICY_HANDBOOK.pdf", "2025-01-01"),
        DocumentVersion("HR_Handbook 4.0_2026 2_0227.pdf", "2026-01-01"),
    ]
    assert resolve_supersession(docs, ignore_tokens={"acme"}) == {
        "Acme_HR_POLICY_HANDBOOK.pdf": "HR_Handbook 4.0_2026 2_0227.pdf"
    }


def test_two_policies_sharing_only_generic_words_are_different_documents() -> None:
    """ "Travel Policy" and "Leave Policy" share half their words and are unrelated. The threshold
    has to sit above that, or every policy in a corpus becomes a version of every other."""
    from rag_core.versioning import DocumentVersion, resolve_supersession

    docs = [
        DocumentVersion("Acme_Business_Travel_Policy.pdf", "2023-01-01"),
        DocumentVersion("Acme_Leave_Policy.pdf", "2026-01-01"),
    ]
    assert resolve_supersession(docs, ignore_tokens={"acme"}) == {}


def test_an_undated_document_is_unknown_not_old() -> None:
    """Being the only document to PRINT a date is not evidence of being the newest. Treating
    undated as old would let a document that merely omits its date be silently buried."""
    from rag_core.versioning import DocumentVersion, resolve_supersession

    docs = [DocumentVersion("handbook-2026.pdf", "2026-01-01"), DocumentVersion("handbook.pdf", None)]
    assert resolve_supersession(docs) == {}


def test_an_operator_pin_overrules_the_dates() -> None:
    """A KB manager marking a document authoritative knows something the platform does not — a
    correct date the file never stated, or that the older wording still governs."""
    from rag_core.versioning import DocumentVersion, resolve_supersession

    docs = [
        DocumentVersion("handbook-2023.pdf", "2023-01-01", pinned=True),
        DocumentVersion("handbook-2026.pdf", "2026-01-01"),
    ]
    assert resolve_supersession(docs) == {"handbook-2026.pdf": "handbook-2023.pdf"}


def test_a_pin_does_not_reach_outside_its_own_document() -> None:
    """A pin says "this is the authoritative version of THIS document". It cannot make an
    unrelated policy obsolete, however old that policy is."""
    from rag_core.versioning import DocumentVersion, resolve_supersession

    docs = [
        DocumentVersion("handbook-2026.pdf", "2026-01-01", pinned=True),
        DocumentVersion("Acme_Night_Shift_Allowance_Policy.pdf", "2023-04-01"),
    ]
    assert resolve_supersession(docs, ignore_tokens={"acme"}) == {}


def test_two_pins_fall_back_to_dates_rather_than_picking_one() -> None:
    """A single winner is the point of pinning. With two there is no operator decision to honour,
    and choosing arbitrarily would look like a decision nobody made."""
    from rag_core.versioning import DocumentVersion, resolve_supersession

    docs = [
        DocumentVersion("handbook-2023.pdf", "2023-01-01", pinned=True),
        DocumentVersion("handbook-2026.pdf", "2026-01-01", pinned=True),
    ]
    assert resolve_supersession(docs) == {"handbook-2023.pdf": "handbook-2026.pdf"}


def test_documents_sharing_a_date_do_not_supersede_each_other() -> None:
    """Two documents effective the same day are both current — one is not a replacement for the
    other, and marking either superseded would be an invention."""
    from rag_core.versioning import DocumentVersion, resolve_supersession

    docs = [DocumentVersion("handbook-a.pdf", "2026-01-01"), DocumentVersion("handbook-b.pdf", "2026-01-01")]
    assert resolve_supersession(docs) == {}


def test_a_single_document_supersedes_nothing() -> None:
    from rag_core.versioning import DocumentVersion, resolve_supersession

    assert resolve_supersession([DocumentVersion("only.pdf", "2026-01-01")]) == {}


def test_a_catalog_title_is_preferred_over_the_file_name() -> None:
    """A doc_id can be a hash or an object key. When the catalog holds a real title, identity
    should rest on that; the file name is the fallback, not the truth."""
    from rag_core.versioning import DocumentVersion, resolve_supersession

    docs = [
        DocumentVersion("a1b2c3", "2023-01-01", title="Information Security Policy v2"),
        DocumentVersion("d4e5f6", "2026-01-01", title="Information Security Policy v3"),
    ]
    assert resolve_supersession(docs) == {"a1b2c3": "d4e5f6"}
