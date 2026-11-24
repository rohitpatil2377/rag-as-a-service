"""A vision model that cannot read an image does not fail — it loops.

Measured live: a handwritten chemistry worksheet produced ~350 characters of real transcription
and then "ce ce ce ce …" until the token budget ran out. Two things follow, and both are tested
here: the loop must not be indexed as if it were text the document contained, and trimming it
must not touch the legitimate repetition real documents are full of (table columns, "N/A N/A").
"""

from __future__ import annotations

from svc_doc_intelligence.ocr_check import trim_degenerate_tail


def test_the_looping_tail_is_dropped_and_the_real_transcription_is_kept() -> None:
    real = "H2 + 2Ca + H2SO4 + MgCl2 CH2=C-CH2-CH3 3CH3 O3 2Mg Fe3O4 NaOH 5MgO"
    text, trimmed = trim_degenerate_tail(f"{real} {'ce ' * 400}")
    assert trimmed is True
    assert text == real


def test_a_multi_word_loop_is_dropped_too() -> None:
    real = "Invoice 4471 dated 02/03/2026. Total payable INR 18,400."
    text, trimmed = trim_degenerate_tail(f"{real} {'of the of the ' * 60}")
    assert trimmed is True
    assert text == real


def test_a_table_column_of_repeated_values_is_left_alone() -> None:
    """The repetition a real document contains: short, bounded, and followed by more document."""
    table = "| Region | Q1 | Q2 |\n" + "| North | N/A | N/A |\n" * 6 + "\nTotal revenue 4.2 crore."
    text, trimmed = trim_degenerate_tail(table)
    assert trimmed is False
    assert text == table


def test_clean_text_is_returned_unchanged() -> None:
    clean = "The night shift allowance is Rs.200 per day, paid with the monthly salary."
    assert trim_degenerate_tail(clean) == (clean, False)


def test_a_page_that_is_nothing_but_a_loop_yields_no_text() -> None:
    """Nothing was read. Empty is the honest answer — better than indexing 'ce' 400 times."""
    text, trimmed = trim_degenerate_tail("ce " * 400)
    assert trimmed is True
    assert text == ""


def test_short_text_is_never_trimmed() -> None:
    """Too little to judge: a caption may legitimately be 'Fig 1 Fig 1'."""
    assert trim_degenerate_tail("Fig 1 Fig 1")[1] is False
