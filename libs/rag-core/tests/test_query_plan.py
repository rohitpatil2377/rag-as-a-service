"""Agentic query-plan pure logic: heuristic mapping, prompt builder, and parse robustness.

No LLM, no IO — the deterministic backend and the parser that guards the LLM path. The parser is
the safety net for the whole feature: ANY malformed field must degrade to the caller's fallback, so
a model error can never mis-route or crash a query.
"""

from __future__ import annotations

import pytest

from rag_core.query_plan import (
    HeuristicQueryPlanner,
    PlanContext,
    QueryPlan,
    build_query_plan_prompt,
    parse_query_plan,
    plan_from_intent,
)

_CTX = PlanContext(doc_count=2, doc_titles=("Employee Handbook", "Leave Policy"), has_history=False)
_FALLBACK = QueryPlan(
    strategy="passages",
    coverage="targeted",
    target_doc=None,
    answer_shape="direct",
    rationale="fallback",
    confidence=0.5,
)


@pytest.mark.parametrize(
    ("intent", "strategy", "coverage", "shape"),
    [
        ("catalog", "catalog", "targeted", "list"),
        ("count", "count", "targeted", "direct"),
        ("filter", "filter", "targeted", "list"),
        ("compare", "compare", "targeted", "comparison"),
        ("doc_summary", "whole_kb", "exhaustive", "summary"),
        ("summarize", "passages", "targeted", "direct"),
        ("rag", "passages", "targeted", "direct"),
    ],
)
def test_plan_from_intent_maps_each_label(intent: str, strategy: str, coverage: str, shape: str) -> None:
    plan = plan_from_intent(intent)  # type: ignore[arg-type]
    assert (plan.strategy, plan.coverage, plan.answer_shape) == (strategy, coverage, shape)
    assert plan.target_doc is None
    assert intent in plan.rationale


async def test_heuristic_planner_wraps_the_keyword_classifier() -> None:
    planner = HeuristicQueryPlanner()
    # "what documents do we have" → catalog; a plain factual question → passages.
    cat = await planner.plan("what documents do we have?", _CTX)
    assert cat.strategy == "catalog"
    rag = await planner.plan("what is the notice period?", _CTX)
    assert rag.strategy == "passages"
    # "summarize each document" → doc_summary → whole_kb/exhaustive (the KB-wide summary shape).
    kb = await planner.plan("summarize each document in detail", _CTX)
    assert (kb.strategy, kb.coverage) == ("whole_kb", "exhaustive")


def test_prompt_includes_the_query_and_doc_context() -> None:
    prompt = build_query_plan_prompt("Summarise the key obligations in this document", _CTX)
    assert "Summarise the key obligations in this document" in prompt
    assert "Employee Handbook" in prompt and "Leave Policy" in prompt
    assert "2 document" in prompt
    # It instructs the whole-document / passages / whole-kb scope calls the pipeline routes on.
    assert "whole_document" in prompt and "passages" in prompt and "whole_kb" in prompt


def test_parse_good_json() -> None:
    reply = (
        '{"strategy":"whole_document","coverage":"exhaustive","target_doc":"Employee Handbook",'
        '"answer_shape":"summary","rationale":"comprehensive coverage of one document","confidence":0.9}'
    )
    plan = parse_query_plan(reply, fallback=_FALLBACK)
    assert plan.strategy == "whole_document"
    assert plan.coverage == "exhaustive"
    assert plan.target_doc == "Employee Handbook"
    assert plan.answer_shape == "summary"
    assert plan.confidence == pytest.approx(0.9)


def test_parse_fenced_json() -> None:
    reply = '```json\n{"strategy":"count","coverage":"targeted","answer_shape":"direct","rationale":"how many"}\n```'
    plan = parse_query_plan(reply, fallback=_FALLBACK)
    assert plan.strategy == "count"
    # confidence defaulted from the fallback when the model omits it.
    assert plan.confidence == _FALLBACK.confidence


def test_parse_json_with_surrounding_prose() -> None:
    reply = 'Here is the plan: {"strategy":"passages","coverage":"targeted","answer_shape":"direct"} — done.'
    assert parse_query_plan(reply, fallback=_FALLBACK).strategy == "passages"


@pytest.mark.parametrize(
    "reply",
    [
        "not json at all",
        "{ this is not valid json",
        '{"strategy":"bogus","coverage":"targeted","answer_shape":"direct"}',  # invalid strategy enum
        '{"strategy":"passages","coverage":"whenever","answer_shape":"direct"}',  # invalid coverage
        '{"strategy":"passages","coverage":"targeted","answer_shape":"paragraph"}',  # invalid shape
        '{"coverage":"targeted","answer_shape":"direct"}',  # missing strategy
        '["a","list","not","an","object"]',
        "",
    ],
)
def test_parse_malformed_returns_fallback(reply: str) -> None:
    assert parse_query_plan(reply, fallback=_FALLBACK) is _FALLBACK


def test_parse_blank_target_and_bad_confidence_are_tolerated() -> None:
    reply = (
        '{"strategy":"passages","coverage":"targeted","answer_shape":"direct",'
        '"target_doc":"   ","confidence":"not-a-number"}'
    )
    plan = parse_query_plan(reply, fallback=_FALLBACK)
    assert plan.strategy == "passages"
    assert plan.target_doc is None  # blank target → None, never an empty-string doc id
    assert plan.confidence == _FALLBACK.confidence  # unparseable confidence → fallback's
