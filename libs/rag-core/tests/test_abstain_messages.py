"""What an abstain SAYS to the reader (FR-E2).

The gate itself was right; the explanation was not. Every abstain -- nothing retrieved, a
retrieval that could not be written up, a guard block -- was shown one "out-of-scope" line, and
those need different actions. The reader whose question was blocked by the injection guard was
told their question was out of scope for a corpus that answered it perfectly well.
"""

from __future__ import annotations

from rag_core.grounding import abstain
from rag_schemas import AnswerStatus


def _answer(reason: str) -> str:
    a = abstain("t", "q", 0.0, reason=reason)
    assert a.status is AnswerStatus.ABSTAINED
    assert a.citations == []  # the invariant is never traded for a nicer message
    assert a.abstain_reason == reason
    return a.answer


def test_every_reason_says_something_different() -> None:
    reasons = [
        "retrieval_confidence_below_threshold",
        "faithfulness_below_threshold",
        "empty_generation",
        "blocked_prompt_injection",
        "blocked_toxic_output",
        "no_architecture_grounded",
        "no_process_grounded",
    ]
    messages = {_answer(r) for r in reasons}
    assert len(messages) == len(reasons), "two reasons share a message -- the reader cannot tell them apart"


def test_nothing_retrieved_says_the_corpus_does_not_cover_it() -> None:
    msg = _answer("retrieval_confidence_below_threshold")
    assert "knowledge base" in msg.lower()


def test_retrieved_but_unsupported_does_not_claim_the_question_was_out_of_scope() -> None:
    """The passages WERE found. Telling this reader their question is out of scope is wrong, and
    it is what sent someone to file a bug against a gate that was working correctly."""
    msg = _answer("faithfulness_below_threshold").lower()
    assert "out-of-scope" not in msg and "out of scope" not in msg
    assert "found" in msg or "passages" in msg


def test_a_guard_block_names_the_guard_rather_than_blaming_the_corpus() -> None:
    """A false positive has to be recognisable as one, or the reader rephrases a question their
    documents answer."""
    msg = _answer("blocked_prompt_injection").lower()
    assert "out-of-scope" not in msg and "out of scope" not in msg
    assert "guard" in msg or "injection" in msg

    toxic = _answer("blocked_toxic_output").lower()
    assert "filter" in toxic


def test_an_unknown_reason_still_says_something() -> None:
    """A caller that invents a reason code must never leave the reader with a blank answer."""
    msg = _answer("some_reason_invented_later")
    assert msg.strip()


def test_the_default_reason_is_used_when_none_is_given() -> None:
    a = abstain("t", "q", 0.0)
    assert a.abstain_reason == "retrieval_confidence_below_threshold"
    assert a.answer == _answer("retrieval_confidence_below_threshold")
