"""Unit tests for FR-G8 synthetic golden-set generation — pure helpers."""

from __future__ import annotations

from svc_control_plane.golden_gen import build_golden_prompt, heuristic_golden, parse_golden

from rag_schemas import Chunk


def test_parse_golden_valid_json() -> None:
    q = parse_golden('noise {"question": "What is the SLA response time?", "expected": "4 hours"} tail')
    assert q is not None
    assert q.question == "What is the SLA response time?"
    assert q.expected == "4 hours"


def test_parse_golden_rejects_non_question() -> None:
    assert parse_golden('{"question": "SLA", "expected": "x"}') is None  # too short / no '?'
    assert parse_golden("not json at all") is None
    assert parse_golden('{"question": "This is a statement.", "expected": "x"}') is None


def test_parse_golden_expected_optional() -> None:
    q = parse_golden('{"question": "What is covered here?", "expected": ""}')
    assert q is not None and q.expected is None


def test_heuristic_golden_anchors_on_entity() -> None:
    c = Chunk(chunk_id="c1", doc_id="d1", tenant_id="t", text="The Acme Torch Bearers Policy rewards long service.")
    q = heuristic_golden(c)
    assert q is not None
    assert "Acme Torch Bearers Policy" in q.question
    assert q.expected == "Acme Torch Bearers Policy"


def test_heuristic_golden_none_without_entities() -> None:
    c = Chunk(chunk_id="c1", doc_id="d1", tenant_id="t", text="the policy applies to everyone here always.")
    assert heuristic_golden(c) is None


def test_build_prompt_includes_passage() -> None:
    assert "PASSAGE:" in build_golden_prompt("hello world") and "hello world" in build_golden_prompt("hello world")
