"""RAG v2 structure-id determinism (upgrade plan §1.1 "Stable IDs", Phase-1 acceptance).

The plan's acceptance for Phase 1 is "unit tests for ID determinism", and the property that
matters operationally is the same one `chunk_id` already has: an id names a unit of content, so it
must be reproducible across calls and across a re-run of the whole derivation, and it must separate
units that are genuinely different (different breadcrumb / ordinal / level). These tests also pin
the compatibility contract with the existing v1 `chunk_id`, which must keep producing exactly what
it produces today.
"""

from __future__ import annotations

import re

from rag_core.chunking import (
    block_id,
    chunk_id,
    normalise_breadcrumb,
    section_id,
    structured_chunk_id,
    text_hash,
)

_DOC = "AEGCL-corrigendum-01.pdf"
_BREADCRUMB = "Attachment-1 > 13. Video Wall Technical Requirements > Standards and Compliance"
_TEXT = "The controller shall comply with IEC 62443-4-2 and IEC 61000-4-2."


def _section() -> str:
    return section_id(_DOC, _BREADCRUMB, 3)


# --- same inputs -> same id, across calls and across a full re-run ------------------------------


def test_the_same_inputs_give_the_same_section_id_on_every_call() -> None:
    assert section_id(_DOC, _BREADCRUMB, 3) == section_id(_DOC, _BREADCRUMB, 3)


def test_the_same_inputs_give_the_same_chunk_and_block_id_on_every_call() -> None:
    digest = text_hash(_TEXT)
    assert structured_chunk_id(_section(), "chunk", 0, digest) == structured_chunk_id(_section(), "chunk", 0, digest)
    assert block_id(_section(), "table_row", 7, digest) == block_id(_section(), "table_row", 7, digest)


def test_every_id_is_stable_across_a_re_run_of_the_whole_derivation() -> None:
    """The acceptance criterion: re-deriving a document's ids from the same structure reproduces
    them exactly — nothing in the chain carries hidden state (no counter, no clock, no iteration
    order), so a re-ingest of unchanged content re-points at the same rows."""

    def derive() -> list[str]:
        out: list[str] = []
        for ordinal, (crumb, body) in enumerate(
            [
                ("Corrigendum > Sl. 1", "Clause 4.2 is amended as follows."),
                ("Corrigendum > Sl. 2", "The bid submission date is extended."),
                (_BREADCRUMB, _TEXT),
            ]
        ):
            sec = section_id(_DOC, crumb, ordinal)
            digest = text_hash(body)
            out.append(sec)
            out.append(structured_chunk_id(sec, "chunk", 0, digest))
            out.append(block_id(sec, "paragraph", 0, digest))
        return out

    assert derive() == derive()


def test_text_hash_is_deterministic_and_changes_with_the_text() -> None:
    assert text_hash(_TEXT) == text_hash(_TEXT)
    assert text_hash(_TEXT) != text_hash(_TEXT + " ")
    assert len(text_hash(_TEXT)) == 64  # full sha256: a change detector, not a name


# --- different inputs -> different ids ----------------------------------------------------------


def test_a_different_breadcrumb_gives_a_different_section_id() -> None:
    other = "Attachment-1 > 14. Control Room Requirements > Standards and Compliance"
    assert section_id(_DOC, other, 3) != section_id(_DOC, _BREADCRUMB, 3)


def test_a_different_ordinal_gives_a_different_section_id() -> None:
    assert section_id(_DOC, _BREADCRUMB, 4) != section_id(_DOC, _BREADCRUMB, 3)


def test_a_different_document_gives_a_different_section_id_for_the_same_breadcrumb() -> None:
    """Two tenders can carry the identical heading path; those are different sections."""
    assert section_id("other.pdf", _BREADCRUMB, 3) != section_id(_DOC, _BREADCRUMB, 3)


def test_a_different_level_gives_a_different_chunk_id_for_the_same_text() -> None:
    """A one-paragraph section is emitted at both `section` and `chunk` level — same text, same
    ordinal, two records that must not overwrite each other."""
    digest = text_hash(_TEXT)
    assert structured_chunk_id(_section(), "section", 0, digest) != structured_chunk_id(_section(), "chunk", 0, digest)


def test_a_different_ordinal_or_text_gives_a_different_chunk_id() -> None:
    sec = _section()
    base = structured_chunk_id(sec, "chunk", 0, text_hash(_TEXT))
    assert structured_chunk_id(sec, "chunk", 1, text_hash(_TEXT)) != base
    assert structured_chunk_id(sec, "chunk", 0, text_hash(_TEXT + " amended")) != base


def test_a_different_block_type_gives_a_different_block_id() -> None:
    """A single-row table emits a `table` block and a `table_row` block over the same text."""
    digest = text_hash(_TEXT)
    assert block_id(_section(), "table", 0, digest) != block_id(_section(), "table_row", 0, digest)


def test_the_part_separator_stops_adjacent_parts_being_conflated() -> None:
    """Plain `+` concatenation would make ("1", 23) and ("12", 3) BOTH "123" — one id for two
    different sections. This pair is the real counterexample, so it fails without `_ID_PART_SEP`
    (the earlier ("1.2", 3) vs ("1", 23) pair does not: "1.23" != "123" either way)."""
    assert section_id(_DOC, "1", 23) != section_id(_DOC, "12", 3)


# --- compatibility with the existing v1 chunk id ------------------------------------------------


def test_the_existing_v1_chunk_id_is_unchanged() -> None:
    """Pinned verbatim: the plan requires the new derivations to keep compatibility with the
    existing stable-chunk-id logic, and existing ids must not move."""
    import hashlib

    expected = f"{_DOC}::{hashlib.sha256(_TEXT.encode('utf-8')).hexdigest()[:12]}"
    assert chunk_id(_DOC, _TEXT) == expected


def test_a_v2_id_can_never_collide_with_a_v1_chunk_id() -> None:
    """The two generations share one column, so a v2 id must never EQUAL a v1 one. The kind tag
    makes a v2 id's trailing field exactly 13 characters; a v1 id's is exactly 12, or 14+ once the
    duplicate-text counter fires — never 13. Width, not the tag's charset, carries the guarantee
    (`b` is itself a hex digit)."""
    v1 = chunk_id(_DOC, _TEXT)
    assert re.fullmatch(r".+::[0-9a-f]{12}", v1)

    # The deduplicated form too: `{digest}.{n}` is 14+, so it is not 13 either.
    seen: dict[str, int] = {}
    chunk_id(_DOC, _TEXT, seen)
    v1_dup = chunk_id(_DOC, _TEXT, seen)
    assert re.fullmatch(r".+::[0-9a-f]{12}\.\d+", v1_dup), v1_dup
    assert len(v1_dup.rsplit("::", 1)[1]) >= 14

    sec = _section()
    digest = text_hash(_TEXT)
    for kind, new_id in (
        ("s", sec),
        ("c", structured_chunk_id(sec, "chunk", 0, digest)),
        ("b", block_id(sec, "paragraph", 0, digest)),
    ):
        assert re.fullmatch(rf".+::{kind}[0-9a-f]{{12}}", new_id)
        assert len(new_id.rsplit("::", 1)[1]) == 13
        assert not re.fullmatch(r".+::[0-9a-f]{12}", new_id)
        assert new_id not in {v1, v1_dup}


def test_a_v2_id_is_prefixed_with_its_parent_so_it_stays_traceable() -> None:
    sec = _section()
    assert sec.startswith(f"{_DOC}::")
    assert structured_chunk_id(sec, "chunk", 0, text_hash(_TEXT)).startswith(f"{sec}::")
    assert block_id(sec, "paragraph", 0, text_hash(_TEXT)).startswith(f"{sec}::")


# --- golden literals: the derivation itself is pinned, not just its properties ------------------
# Without these, reordering `section_id`'s hashed parts (or swapping the separator, or changing the
# digest width) moves every id in the corpus while every property test above stays green. These ids
# will be referenced by stored citations across releases, so the derivation is a compatibility
# surface and gets pinned the same way the v1 `chunk_id` format is.
#
# If one of these fails, that is not a test to update — it means previously stored v2 ids no longer
# name the same content, and the change needs a migration or a version bump, not a new literal.
_GOLDEN_SECTION = "AEGCL-corrigendum-01.pdf::s7af60740eb2c"
_GOLDEN_CHUNK = "AEGCL-corrigendum-01.pdf::s7af60740eb2c::cd63f4f90d396"
_GOLDEN_BLOCK = "AEGCL-corrigendum-01.pdf::s7af60740eb2c::b7e83408ac378"
_GOLDEN_TEXT_HASH = "8832788d07da6d873e94739af88122763d8ee0f5044cd886a7a332be6677a559"


def test_the_section_id_derivation_is_pinned_to_a_golden_literal() -> None:
    assert section_id(_DOC, _BREADCRUMB, 3) == _GOLDEN_SECTION


def test_the_chunk_id_derivation_is_pinned_to_a_golden_literal() -> None:
    assert structured_chunk_id(_GOLDEN_SECTION, "chunk", 0, _GOLDEN_TEXT_HASH) == _GOLDEN_CHUNK


def test_the_block_id_derivation_is_pinned_to_a_golden_literal() -> None:
    assert block_id(_GOLDEN_SECTION, "paragraph", 0, _GOLDEN_TEXT_HASH) == _GOLDEN_BLOCK


def test_the_text_hash_is_pinned_to_a_golden_literal() -> None:
    assert text_hash(_TEXT) == _GOLDEN_TEXT_HASH


# --- breadcrumb normalisation -------------------------------------------------------------------


def test_whitespace_drift_in_a_breadcrumb_does_not_move_the_section_id() -> None:
    """OCR re-runs legitimately differ by a double space or a stray newline. Unnormalised, that
    re-points the section AND every chunk/block beneath it — a subtree of stored citations invalidated
    for no real change."""
    for drifted in (
        f"  {_BREADCRUMB}  ",
        _BREADCRUMB.replace(" > ", "  >  "),
        _BREADCRUMB.replace(" > ", " >\n"),
        _BREADCRUMB.replace(" > ", " >\t"),
    ):
        assert section_id(_DOC, drifted, 3) == _GOLDEN_SECTION, drifted


def test_a_stray_separator_character_cannot_degrade_the_separator_property() -> None:
    """A control character in heading text would otherwise be hashed verbatim, and `_ID_PART_SEP`
    in particular would re-introduce the ambiguity the separator exists to prevent."""
    assert section_id(_DOC, "1\x1f23", 4) == section_id(_DOC, "1 23", 4)
    assert section_id(_DOC, "1\x1f23", 4) != section_id(_DOC, "1", 234)


def test_normalise_breadcrumb_collapses_whitespace_and_strips_controls() -> None:
    assert normalise_breadcrumb("  Attachment-1 >\t13.  Video\x1fWall  ") == "Attachment-1 > 13. Video Wall"
    assert normalise_breadcrumb(_BREADCRUMB) == _BREADCRUMB  # already canonical: unchanged


def test_a_real_heading_difference_still_moves_the_section_id() -> None:
    """Normalisation must not be so aggressive that genuinely different headings collide."""
    assert section_id(_DOC, "13. Video Wall", 3) != section_id(_DOC, "13 Video Wall", 3)
