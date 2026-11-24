"""Parent context in the grounding prompt + faithfulness input (FR-C2, Option B).

Pins the flag-off byte-identical guarantee, the NON-CITABLE CONTEXT block above the numbered
SOURCES, sibling dedup, that `[n]` still resolves to the precise child after expansion, and that
`with_parent_context` feeds parent text to faithfulness without adding a citation.
"""

from __future__ import annotations

from rag_core.grounding import build_prompt, ground, with_parent_context
from rag_schemas import RetrievedSpan
from rag_schemas.chunk import Span


def _child(cid: str, text: str, *, parent_id: str | None = None, parent_text: str | None = None) -> RetrievedSpan:
    return RetrievedSpan(
        chunk_id=cid,
        doc_id="d",
        text=text,
        page=3,
        span=Span(start=0, end=len(text)),
        parent_id=parent_id,
        parent_text=parent_text,
    )


def test_prompt_is_byte_identical_when_no_parent_text() -> None:
    """Flag-off / legacy guarantee: a span carrying no parent_text produces the exact prompt as
    before the feature — no CONTEXT block, byte-for-byte."""
    spans = [_child("c1", "The uptime target is 99.5 percent.")]
    prompt = build_prompt("what is the uptime target?", spans)
    assert "CONTEXT (" not in prompt
    # A parent_id WITHOUT parent_text (child not yet expanded) is still byte-identical.
    spans_id_only = [_child("c1", "The uptime target is 99.5 percent.", parent_id="d::parent")]
    assert build_prompt("what is the uptime target?", spans_id_only) == prompt


def test_context_block_is_rendered_above_sources_and_is_non_citable() -> None:
    parent = "Section 2 covers service metrics: uptime, latency and recovery objectives."
    spans = [_child("c1", "The uptime target is 99.5 percent.", parent_id="d::p", parent_text=parent)]
    prompt = build_prompt("uptime target?", spans)
    assert "CONTEXT (" in prompt
    assert parent in prompt
    # CONTEXT appears BEFORE the numbered SOURCES, and is explicitly non-citable.
    assert prompt.index("CONTEXT (") < prompt.index("SOURCES:")
    assert "do NOT cite" in prompt
    # SOURCES stay one-per-child, numbered [1..], so [n] still maps to the child.
    assert "[1] (" in prompt


def test_sibling_parents_collapse_to_one_context_block() -> None:
    """Two children sharing the same parent produce ONE fenced parent block (dedup)."""
    parent = "The shared parent section text for both children."
    spans = [
        _child("c1", "First child fact.", parent_id="d::p", parent_text=parent),
        _child("c2", "Second child fact.", parent_id="d::p", parent_text=parent),
    ]
    prompt = build_prompt("q?", spans)
    assert prompt.count(parent) == 1


def test_citation_resolves_to_child_after_expansion() -> None:
    """`ground` builds citations from the child spans only; a [1] answer cites the child, never the
    parent — parent text in the prompt does not become a citation."""
    parent = "Background: the platform's metrics live in section 2."
    spans = [_child("c1", "The uptime target is 99.5 percent.", parent_id="d::p", parent_text=parent)]
    answer = ground("t", "uptime?", "The uptime target is 99.5 percent. [1]", spans, 0.9)
    assert len(answer.citations) == 1
    assert answer.citations[0].chunk_id == "c1"
    assert answer.citations[0].page == 3


def test_with_parent_context_adds_non_citation_spans_deduped() -> None:
    parent = "The shared parent context."
    spans = [
        _child("c1", "First.", parent_id="d::p", parent_text=parent),
        _child("c2", "Second.", parent_id="d::p", parent_text=parent),
        _child("c3", "Third, different parent.", parent_id="d::p2", parent_text="Other parent."),
    ]
    augmented = with_parent_context(spans)
    # 3 children + 2 distinct parents.
    assert len(augmented) == len(spans) + 2
    parent_span_ids = {s.chunk_id for s in augmented[len(spans) :]}
    assert parent_span_ids == {"d::p", "d::p2"}


def test_with_parent_context_is_noop_without_parent_text() -> None:
    spans = [_child("c1", "Only a child."), _child("c2", "Another child.", parent_id="d::p")]
    assert with_parent_context(spans) is spans
