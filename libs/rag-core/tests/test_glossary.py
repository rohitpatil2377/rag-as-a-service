"""Unit tests for deterministic glossary / synonym query expansion (pre-retrieval)."""

from __future__ import annotations

from pathlib import Path

from rag_core.glossary import Glossary, expand_glossary, load_glossary


def _default() -> Glossary:
    return load_glossary()


def test_casual_trigger_appends_canonical_terms() -> None:
    out = expand_glossary("what happens when the test fails?", _default())
    assert out.startswith("what happens when the test fails?")  # original preserved, first
    assert "out-of-specification" in out and "OOS" in out  # canonical terms appended after


def test_no_match_returns_identity() -> None:
    text = "what is the office wifi password?"
    assert expand_glossary(text, _default()) == text  # unchanged (identity) when nothing matches


def test_idempotent_expand_twice_equals_once() -> None:
    once = expand_glossary("who signs off when a product test fails?", _default())
    twice = expand_glossary(once, _default())
    assert twice == once  # re-expanding an expanded query appends nothing new


def test_acronym_is_bidirectional() -> None:
    # Long form present -> short appended.
    long_form = expand_glossary("there was an out-of-specification event", _default())
    assert "OOS" in long_form.split()
    # Short form present -> long appended.
    short_form = expand_glossary("an OOS was raised", _default())
    assert "out-of-specification" in short_form.split()


def test_validated_case_gains_both_oos_and_approval() -> None:
    """The empirically-validated query: it must pick up BOTH the OOS vocabulary (from the
    failing-test trigger) AND the approval vocabulary (from the sign-off trigger)."""
    out = expand_glossary("who signs off when a product test fails?", _default())
    assert "out-of-specification" in out and "OOS" in out
    assert "approve" in out and "approval" in out


def test_no_duplicate_tokens_appended() -> None:
    # A query already containing a canonical token must not get that token appended again.
    out = expand_glossary("the out-of-specification OOS result when a test fails", _default())
    assert out.lower().split().count("out-of-specification") == 1
    assert out.split().count("OOS") == 1


def test_append_is_the_default_mode() -> None:
    # Default keeps the user's casual words in place (recall-safe) and adds canonical terms after.
    out = expand_glossary("who signs off when a product test fails?", _default())
    assert "test fails" in out.lower()  # original casual phrase retained
    assert "out-of-specification" in out  # canonical appended


def test_replace_mode_swaps_trigger_for_canonical() -> None:
    # replace mode removes the matched casual phrase and substitutes the canonical terms in place,
    # so a lexical lookalike that matches the casual wording can no longer out-rank the right doc.
    out = expand_glossary("who signs off when a product test fails?", _default(), mode="replace")
    assert "signs off" not in out.lower()  # casual trigger swapped out
    assert "test fails" not in out.lower()  # casual trigger swapped out
    assert "out-of-specification" in out and "approve" in out  # canonical substituted in


def test_replace_mode_is_idempotent() -> None:
    once = expand_glossary("who signs off when a product test fails?", _default(), mode="replace")
    twice = expand_glossary(once, _default(), mode="replace")
    assert twice == once


def test_replace_mode_no_match_returns_identity() -> None:
    text = "what is the office wifi password?"
    assert expand_glossary(text, _default(), mode="replace") == text


def test_file_overlay_merges_over_default(tmp_path: Path) -> None:
    import json

    path = tmp_path / "glossary.json"
    path.write_text(json.dumps({"terms": [{"triggers": ["widget broke"], "canonical": "widget failure mode"}]}))
    glossary = load_glossary(str(path))
    out = expand_glossary("the widget broke overnight", glossary)
    assert "widget" in out and "failure" in out and "mode" in out
    # The default rules still apply on top of the overlay.
    assert "out-of-specification" in expand_glossary("the test fails", glossary)
