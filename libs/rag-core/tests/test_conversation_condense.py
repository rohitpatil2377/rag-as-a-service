"""Unit tests for conversational follow-up condensation (FR-I1) — pure logic."""

from __future__ import annotations

from rag_core import default_title, heuristic_condense, is_follow_up


def test_is_follow_up_detects_dependent_turns() -> None:
    assert is_follow_up("And the latency one?")
    assert is_follow_up("what about premium tenants?")
    assert is_follow_up("why not?")
    assert is_follow_up("it")  # dangling reference
    assert is_follow_up("tell me more")


def test_is_follow_up_leaves_standalone_questions() -> None:
    assert not is_follow_up("What is the platform uptime commitment for the year?")
    assert not is_follow_up("List every knowledge base in the finance organisation")
    assert not is_follow_up("")


def test_heuristic_condense_folds_in_previous_subject() -> None:
    history = ["What is the uptime target?"]
    out = heuristic_condense(history, "And the latency one?")
    assert "uptime target" in out.lower()
    assert "latency" in out.lower()


def test_short_question_with_own_subject_is_not_a_follow_up() -> None:
    # Regression: a 5-word question that carries its own subject must NOT be treated as a
    # follow-up (else it gets polluted with the previous topic and the model answers "I don't know").
    assert not is_follow_up("eligibility criteria for torch bearer")
    assert not is_follow_up("night shift allowance amount")
    assert not is_follow_up("maternity leave duration policy")
    # …but a short turn WITHOUT its own subject still is a follow-up.
    assert is_follow_up("the amount?")
    assert is_follow_up("why?")


def test_heuristic_condense_does_not_merge_unrelated_topics() -> None:
    # Regression: switching topics mid-thread keeps the new question standalone.
    history = ["What is the monthly leave policy?"]
    out = heuristic_condense(history, "eligibility criteria for torch bearer")
    assert out == "eligibility criteria for torch bearer"
    assert "monthly leave" not in out.lower()


def test_heuristic_condense_noop_without_history_or_for_standalone() -> None:
    assert heuristic_condense([], "And the latency one?") == "And the latency one?"
    standalone = "What is the platform uptime commitment for premium tenants?"
    assert heuristic_condense(["earlier question"], standalone) == standalone


def test_heuristic_condense_avoids_duplicating_restated_subject() -> None:
    history = ["What is the uptime target?"]
    # The user already restated the subject — don't fold it in twice.
    out = heuristic_condense(history, "uptime target?")
    assert out.lower().count("uptime target") == 1


def test_default_title_truncates() -> None:
    assert default_title("short question") == "short question"
    long = "a " * 60
    assert len(default_title(long)) <= 60
    assert default_title(long).endswith("…")
