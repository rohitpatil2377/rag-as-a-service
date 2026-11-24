"""What the generator is actually shown (FR-E1/E2).

``build_prompt`` had no test at all before P15, which is how a source header could silently
omit the one field that says a rule is scope-limited.
"""

from __future__ import annotations

from typing import Literal

from rag_core.grounding import build_prompt, gate_for
from rag_schemas import RetrievedSpan

Scorer = Literal["cross_encoder", "lexical", "unscored"]


def _span(
    text: str,
    section: str | None = None,
    page: int | None = None,
    *,
    score: float = 0.9,
    scorer: Scorer = "cross_encoder",
) -> RetrievedSpan:
    return RetrievedSpan(
        chunk_id="handbook::0",
        doc_id="handbook",
        text=text,
        section=section,
        page=page,
        score=score,
        scorer=scorer,
    )


def test_a_scope_limited_rule_reaches_the_model_with_its_scope() -> None:
    """QA Q10: a rule from "6.2.3 Social Media Content" was applied to a sales deck. Read on its
    own the passage IS a general rule — the limit lives in the heading, which is a different
    chunk. Retrieval knew the section; the prompt threw it away."""
    prompt = build_prompt(
        "Do I need approval to send a sales deck to a customer?",
        [
            _span(
                "All content requires approval from the Communications team before release.",
                section="6.2.3 Social Media Content",
            )
        ],
    )
    assert "6.2.3 Social Media Content" in prompt
    assert "doc=handbook" in prompt


def test_a_span_with_no_section_says_nothing_about_sections() -> None:
    """Most documents are not numbered. An empty ``section=`` label would read as "this rule is
    unscoped" — a claim the extractor never made."""
    prompt = build_prompt("q", [_span("Annual leave is 24 days.")])
    assert "section=" not in prompt


def test_the_page_is_still_carried() -> None:
    """The header gained a field; it must not have lost one — the page is what makes a citation
    verifiable against the source document."""
    prompt = build_prompt("q", [_span("Annual leave is 24 days.", page=7)])
    assert "page=7" in prompt


def test_sources_are_numbered_from_one() -> None:
    """The model is asked to cite [1], [2] and the citation assembler resolves those indices
    back to spans. An off-by-one here mislabels every citation in the product."""
    prompt = build_prompt("q", [_span("first"), _span("second")])
    assert "[1] (" in prompt and "[2] (" in prompt and "[0] (" not in prompt


# --- the gate is scoped to the scorer that produced the number ----------------------------------


def test_the_gate_reads_the_threshold_for_the_scorer_that_scored() -> None:
    """A cross-encoder probability and a term-overlap fraction are numbers on different scales.
    One threshold applied to both is the P50 bug class: measured on 20 questions against a real
    RFE, the calibrated cross-encoder floor of 0.15 blocks 0 of 10 unanswerable questions when
    the lexical twin produced the score, because "what is the of on" overlaps every chunk."""
    scored = [_span("x", scorer="cross_encoder", score=0.5)]
    assert gate_for(scored, cross_encoder=0.15, lexical=0.60) == 0.15

    overlap = [_span("x", scorer="lexical", score=0.5)]
    assert gate_for(overlap, cross_encoder=0.15, lexical=0.60) == 0.60


def test_the_top_scoring_span_decides_the_scale() -> None:
    """max_score is the maximum, so the threshold it is compared against must belong to the
    span that produced it — not to whichever span happens to be first in the list."""
    spans = [_span("a", scorer="lexical", score=0.2), _span("b", scorer="cross_encoder", score=0.9)]
    assert gate_for(spans, cross_encoder=0.15, lexical=0.60) == 0.15


def test_an_unscored_result_can_never_open_the_gate() -> None:
    """Nothing judged these spans. A gate of 1.0 is unreachable because only a reranker writes
    `score`, and it writes `scorer` in the same breath — so a bypassed reranker abstains
    instead of grounding on a number that means nothing."""
    assert gate_for([_span("x", scorer="unscored", score=0.0)], cross_encoder=0.15, lexical=0.60) == 1.0
    assert gate_for([], cross_encoder=0.15, lexical=0.60) == 1.0


def test_a_score_cannot_be_set_without_naming_its_scorer() -> None:
    """I11 made structural: the contradiction is unconstructible, not merely handled downstream.
    A bare number belongs to no scale, and the gate has to pick a threshold per scale."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RetrievedSpan(chunk_id="c", doc_id="d", text="x", score=0.9)


# --- adaptive answer shape -------------------------------------------------------------------
# The default shape used to be fixed: one sentence, then a flat "- " bullet list, for every
# question that did not explicitly ask for a format. Reported live on a pharmacy SOP corpus —
# "Describe the full lifecycle of a new SOP, from draft to issue" is a SEQUENCE and came back as
# unordered bullets, and a single-fact question came back as a sentence plus bullets. The shape
# must follow what the answer IS, not a template.


def test_the_prompt_asks_the_model_to_choose_a_shape_that_fits_the_answer() -> None:
    prompt = build_prompt("Describe the lifecycle of an SOP", [_span("Draft, review, approve, issue.")])
    lowered = prompt.lower()
    # a sequence gets numbers, not bullets
    assert "numbered" in lowered
    # discrete parallel facts get bullets
    assert "bullet" in lowered
    # one fact gets a sentence and nothing else
    assert "single sentence" in lowered or "one sentence" in lowered
    # sub-points are allowed, and must be indented to nest
    assert "indent" in lowered


def test_the_shape_rules_do_not_lose_what_was_already_earned() -> None:
    """Citations, bold entities and no-restating are not formatting preferences — each was added
    for a measured reason (a restated sentence scored 0.33 where one sentence scored 1.00)."""
    prompt = build_prompt("q", [_span("Annual leave is 24 days.")])
    assert "[n]" in prompt
    assert "**bold**" in prompt
    assert "restate" in prompt.lower()


def test_an_explicit_format_request_still_overrides_everything() -> None:
    prompt = build_prompt("Give me the leave rules in table format", [_span("Annual leave is 24 days.")])
    assert "OUTPUT FORMAT OVERRIDE" in prompt
