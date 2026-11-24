"""Atomic units are the ranges a structure-aware chunker must never split (FR-C2).

Tables, fenced code and formula lines are detected from the extracted text alone; prose is
never captured, so a document with none of them yields no atomic spans (and the chunker then
behaves exactly like the clause chunker)."""

from __future__ import annotations

from itertools import pairwise

from rag_core.atomic_units import atomic_spans


def _covers(spans: list[tuple[int, int]], text: str, needle: str) -> bool:
    """Is ``needle`` wholly inside one atomic span?"""
    at = text.index(needle)
    return any(start <= at and at + len(needle) <= end for start, end in spans)


def test_a_table_run_is_one_atomic_span() -> None:
    text = "Intro line.\n| Department | Code |\n| --- | --- |\n| Quality Control | Q |\nAfter."
    spans = atomic_spans(text)
    assert len(spans) == 1
    assert _covers(spans, text, "| Department | Code |")
    assert _covers(spans, text, "| Quality Control | Q |")
    # The prose before and after is NOT captured.
    assert not _covers(spans, text, "Intro line.")
    assert not _covers(spans, text, "After.")


def test_a_single_pipe_line_is_not_a_table() -> None:
    text = "The value is A | B in legacy notation.\nOrdinary prose continues here."
    assert atomic_spans(text) == []


def test_a_fenced_code_block_is_atomic() -> None:
    text = "Run this:\n```python\nx = 1\nif x:\n    print(x)\n```\nDone."
    spans = atomic_spans(text)
    assert len(spans) == 1
    assert _covers(spans, text, "x = 1\nif x:\n    print(x)")
    assert not _covers(spans, text, "Done.")


def test_an_unmatched_fence_is_ignored() -> None:
    text = "Prose.\n```\nnot really closed\nmore prose that reads normally here"
    assert atomic_spans(text) == []


def test_a_formula_line_is_atomic() -> None:
    text = "The identity is well known.\na\u00b2 - b\u00b2 = (a + b)(a - b)\nQ.E.D."
    spans = atomic_spans(text)
    assert _covers(spans, text, "a\u00b2 - b\u00b2 = (a + b)(a - b)")
    assert not _covers(spans, text, "The identity is well known.")


def test_a_chemical_reaction_line_is_atomic() -> None:
    reaction = "H\u2082SO\u2084 + 2 NaOH \u2192 Na\u2082SO\u2084 + 2 H\u2082O"
    text = f"Consider the reaction below.\n{reaction}\nIt is exothermic."
    assert _covers(atomic_spans(text), text, reaction)


def test_prose_is_never_a_formula() -> None:
    text = "It costs $5 today and section 3.2 applies to all of the relevant cases here.\n"
    assert atomic_spans(text) == []


def test_no_atomic_units_in_plain_prose() -> None:
    text = (
        "This is an ordinary paragraph of policy text. It has several sentences. "
        "None of them are tables, code, or formulas.\nA second paragraph follows."
    )
    assert atomic_spans(text) == []


def test_spans_are_sorted_and_non_overlapping() -> None:
    text = (
        "| a | b |\n| c | d |\n"  # table run
        "prose in between\n"
        "```\ncode\nblock\n```\n"  # code fence
        "E = mc\u00b2\n"  # formula
    )
    spans = atomic_spans(text)
    assert spans == sorted(spans)
    for (_s1, e1), (s2, _e2) in pairwise(spans):
        assert e1 <= s2
    assert len(spans) == 3
