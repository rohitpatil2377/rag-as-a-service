"""Unit tests for FR-I8 suggested follow-ups — pure logic."""

from __future__ import annotations

from rag_core.followups import (
    HeuristicFollowupGenerator,
    build_followups_prompt,
    heuristic_followups,
    parse_followups,
)


def test_parse_followups_cleans_and_dedupes() -> None:
    raw = "1. What is the SLA?\n- What is the SLA?\n* How is uptime measured?\nnot a question\nWho owns it?"
    out = parse_followups(raw, 3)
    assert out == ["What is the SLA?", "How is uptime measured?", "Who owns it?"]


def test_parse_followups_respects_count() -> None:
    raw = "What is one thing?\nWhat is two here?\nWhat is three now?\nWhat is four then?"
    assert len(parse_followups(raw, 2)) == 2


def test_heuristic_followups_anchor_on_new_entities() -> None:
    query = "Tell me about Acme."
    context = "Acme runs the Nashik Data Centre. The Torch Bearers Policy rewards service."
    out = heuristic_followups(query, context, 3)
    # Acme was already asked about → excluded; new entities surface.
    assert any("Nashik Data Centre" in q for q in out)
    assert all(q != "Acme." for q in out)
    assert all(q.endswith("?") for q in out)


def test_build_prompt_constrains_to_context() -> None:
    p = build_followups_prompt("q", "some context", 3)
    assert "ONLY" in p and "CONTEXT:" in p


async def test_heuristic_generator_matches_function() -> None:
    gen = HeuristicFollowupGenerator()
    ctx = "The Acme Cloud Platform uses AES-256 encryption."
    assert await gen.suggest("hi", ctx, 2) == heuristic_followups("hi", ctx, 2)
