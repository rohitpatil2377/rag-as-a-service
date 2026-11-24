"""A number in an answer that is not in the sources is fabricated (P29).

Faithfulness verification scores token overlap and explicitly drops short tokens —
`_MIN_TOKEN_LEN = 3`, commented "ignore 1-2 char tokens (numbers/particles)". So a figure the
model computed rather than read was invisible to it by construction. Measured on the real QA
run, the platform answered:

    "Total: 48 + 72 + 24 = **144 hours (18 working days)**"
    "Category B content requires worst-case total time of **18 working days**"

Neither 144 nor 18 appears anywhere in the SOP. The model summed three durations and converted
them on an unstated 8-hour working day — and the second sentence reads as though the SOP says
it. Token overlap rated that sentence supported, because "working" and "days" are both in the
sources and "18" was thrown away before scoring.

Prose is fuzzy; a number is not. If a figure is not in the sources it was invented, and for a
compliance corpus the numbers ARE the content — the thresholds, deadlines, amounts and notice
periods a reader acts on.
"""

from __future__ import annotations

from rag_core.numeracy import numbers_in, unsupported_numbers
from rag_schemas import RetrievedSpan


def _spans(*texts: str) -> list[RetrievedSpan]:
    return [RetrievedSpan(chunk_id=f"c{i}", doc_id="d", text=t) for i, t in enumerate(texts)]


# --- what counts as a number ------------------------------------------------------------------


def test_thousands_separators_and_trailing_zeros_are_the_same_number() -> None:
    """ "Rs. 50,000" in a source and "50000" in an answer are the same figure. Treating them as
    different would flag correct answers, which is how a guard gets switched off."""
    assert numbers_in("Rs. 50,000 payable") == numbers_in("an amount of 50000")
    assert numbers_in("24.0 days") == numbers_in("24 days")


def test_citation_markers_are_not_claims() -> None:
    """ "[1]" is inserted by the citation assembler, not asserted by the model."""
    assert numbers_in("The allowance is 400 rupees. [1][2]") == {"400"}


def test_list_numbering_is_formatting_not_a_claim() -> None:
    """A numbered list is layout. Flagging "1." would fire on almost every structured answer."""
    assert numbers_in("1. First point\n2. Second point\n3. Third point") == set()
    assert numbers_in("1. The notice period is 90 days") == {"90"}


# --- the check --------------------------------------------------------------------------------


def test_the_real_failure_is_caught() -> None:
    """Q7 from the QA run, with the SOP's actual figures as the sources."""
    sources = _spans(
        "Step 5 Content Curator Full Review: 48 hours.",
        "Step 6 Legal and Compliance Review for Category B: 72 hours.",
        "Step 7 Final Sign-Off: 24 hours.",
    )
    answer = "Total: 48 + 72 + 24 = 144 hours (18 working days), which exceeds the 5 working day limit."
    # 144 and 18 are the invented ones — the sum and the unit conversion. Both are caught.
    assert sorted(unsupported_numbers(answer, sources)) == ["144", "18"]


def test_the_rule_is_loose_on_purpose_and_this_is_where_it_shows() -> None:
    """ "5 working day limit" is NOT flagged, because 5 appears in the sources — as "Step 5".
    The check asks only whether a figure was retrieved, not whether it is used for the same
    thing it meant there. Judging that needs to understand the claim, which is the LLM judge's
    job; this one has to stay deterministic so it still holds when the judge is a small model.
    Pinned so the limit is known rather than discovered later."""
    sources = _spans("Step 5 Content Curator Full Review: 48 hours.")
    assert unsupported_numbers("The limit is 5 working days.", sources) == []


def test_a_figure_read_from_the_sources_is_not_flagged() -> None:
    sources = _spans("The night shift allowance is 400 rupees per shift.")
    assert unsupported_numbers("The allowance is 400 rupees per shift. [1]", sources) == []


def test_a_number_anywhere_in_the_sources_counts_as_supported() -> None:
    """The check asks "was this figure in what we retrieved", not "is it in the right place" —
    a stricter rule would need to understand the claim, which is the LLM judge's job."""
    sources = _spans("Annual leave is 24 days.", "Notice period is 90 days.")
    assert unsupported_numbers("Leave is 24 days and notice is 90 days.", sources) == []


def test_years_and_dates_are_checked_like_any_other_figure() -> None:
    sources = _spans("This policy is effective from 28 July 2026.")
    assert unsupported_numbers("Effective 28 July 2026.", sources) == []
    assert unsupported_numbers("Effective 28 July 2025.", sources) == ["2025"]


def test_an_answer_with_no_numbers_is_never_flagged() -> None:
    assert unsupported_numbers("The Content Curator approves all external material.", _spans("anything")) == []


def test_percentages_match_their_bare_figure() -> None:
    """ "99.5%" and "99.5 percent" are the same number written two ways."""
    assert unsupported_numbers("Uptime is 99.5%.", _spans("The uptime target is 99.5 percent.")) == []
