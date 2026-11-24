"""A turn that only changes the PRESENTATION is a follow-up, not a new question.

Reported live: after a table of ten SOPs, "not in table formmat but in bullet point" abstained.
Standing alone it deserves to -- it names no subject, and retrieval scored it 0.0013 against the
corpus. But it was never meant to stand alone: it asks for the PREVIOUS answer, shaped
differently. Recognising that folds the earlier question back in, so retrieval sees the subject
again and the model re-answers in the requested shape.

The rule is deliberately all-or-nothing: EVERY significant word must be presentational. A turn
carrying any subject of its own ("bullet points for the leave policy") is a real question.
"""

from __future__ import annotations

import pytest

from rag_core.conversation import heuristic_condense, is_follow_up

_PREV = "explain PBSOP001, PBSOP002 and PBSOP003"


@pytest.mark.parametrize(
    "turn",
    [
        "not in table formmat but in bullet point",
        "in bullet points please",
        "as a numbered list",
        "make it shorter",
        "in a table instead",
        "not as prose",
    ],
)
def test_a_presentation_only_turn_is_a_follow_up(turn: str) -> None:
    assert is_follow_up(turn)


@pytest.mark.parametrize(
    "turn",
    [
        "bullet points for the leave policy",
        "what is the table of department codes",
        "list the SOPs",
        "explain the escalation matrix",
    ],
)
def test_a_turn_carrying_its_own_subject_is_not_a_formatting_follow_up(turn: str) -> None:
    """These name something to retrieve, so they must stand alone rather than be fused with
    whatever happened to be asked before."""
    assert not is_follow_up(turn)


def test_the_previous_question_is_folded_back_in() -> None:
    out = heuristic_condense([_PREV], "not in table formmat but in bullet point")
    # The subject has to come back, or retrieval has nothing to match on.
    assert "PBSOP001" in out
    # ...and the instruction has to survive, or the answer comes back as a table again.
    assert "bullet" in out.lower()


def test_without_history_it_is_left_alone() -> None:
    """A first turn has nothing to fold in; it must not be mangled."""
    assert heuristic_condense([], "in bullet points please") == "in bullet points please"


def test_chained_formatting_turns_reach_back_to_the_real_question() -> None:
    """Live: "in bullet points" then "make it shorter" abstained. The second turn folded in the
    FIRST one -- itself a presentation request -- so the pair still carried no subject. A
    formatting turn is never the thing to fold in; the last real question is."""
    history = [_PREV, "give me the above in bullet point"]
    out = heuristic_condense(history, "make it shorter")
    assert "PBSOP001" in out
    assert "shorter" in out


def test_a_formatting_turn_after_a_real_question_still_uses_that_question() -> None:
    out = heuristic_condense([_PREV], "make it shorter")
    assert "PBSOP001" in out


def test_all_history_being_formatting_leaves_the_turn_alone() -> None:
    """Nothing real to fold in -- fusing two presentation requests helps nobody."""
    out = heuristic_condense(["in bullet points", "make it shorter"], "shorter still")
    assert out == "shorter still"
