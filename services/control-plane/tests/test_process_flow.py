"""Workflow understanding + optimization — parse, gap detection, renderers, extractor, query."""

from __future__ import annotations

import pytest
from svc_control_plane.process_flow import (
    ProcessStep,
    ProcessTransition,
    WorkflowProcess,
    answer_process,
    detect_gaps,
    extract_process,
    is_process_query,
    parse_process,
    process_grounded_answer,
    to_mermaid,
    to_plantuml,
)

from rag_schemas import RetrievalMode, RetrievalQuery, RetrievalResult, RetrievedSpan
from rag_schemas.common import AnswerStatus


def _proc() -> WorkflowProcess:
    return WorkflowProcess(
        title="Loan Approval",
        steps=[
            ProcessStep(id="start", name="Application received", kind="start"),
            ProcessStep(id="review", name="Review", kind="task", actor="Analyst"),
            ProcessStep(id="decide", name="Approve?", kind="decision"),
            ProcessStep(id="done", name="Disburse", kind="end"),
        ],
        transitions=[
            ProcessTransition(source="start", target="review"),
            ProcessTransition(source="review", target="decide"),
            ProcessTransition(source="decide", target="done", condition="approved"),
            ProcessTransition(source="decide", target="review", condition="rejected"),
        ],
    )


class _Gateway:
    def __init__(self, out: str) -> None:
        self._out = out

    async def complete(self, prompt: str, *, max_tokens: int | None = None) -> str:
        return self._out


_PROC_JSON = (
    '{"title":"Onboarding","roles":["HR"],"steps":['
    '{"name":"Start","kind":"start"},{"name":"Collect Docs","kind":"task","actor":"HR"},'
    '{"name":"Verify","kind":"decision"},{"name":"Done","kind":"end"}],'
    '"transitions":[{"source":"Start","target":"Collect Docs"},{"source":"Collect Docs","target":"Verify"},'
    '{"source":"Verify","target":"Done","condition":"ok"},'
    '{"source":"Verify","target":"Collect Docs","condition":"missing"}]}'
)


def test_parse_maps_unknown_kind_dedupes_and_drops_dangling() -> None:
    data = {
        "steps": [
            {"name": "Start", "kind": "start"},
            {"name": "Do Work", "kind": "activity"},  # unknown -> task
            {"name": "Do Work"},  # dup -> deduped
        ],
        "transitions": [
            {"source": "Start", "target": "Do Work"},  # by name
            {"source": "Do Work", "target": "Ghost"},  # dangling -> dropped
        ],
    }
    p = parse_process(data)
    assert {s.id: s.kind for s in p.steps} == {"start": "start", "do_work": "task"}
    assert len(p.transitions) == 1 and p.transitions[0].target == "do_work"


def test_detect_gaps_finds_structural_problems() -> None:
    p = WorkflowProcess(
        steps=[
            ProcessStep(id="s", name="S", kind="start"),
            ProcessStep(id="a", name="A", kind="task"),  # reachable task, no actor
            ProcessStep(id="b", name="B", kind="decision"),  # one-armed (single branch)
            ProcessStep(id="orphan", name="Orphan", kind="task", actor="X"),  # unreachable + dead end
            ProcessStep(id="e", name="E", kind="end"),
        ],
        transitions=[
            ProcessTransition(source="s", target="a"),
            ProcessTransition(source="a", target="b"),
            ProcessTransition(source="b", target="e", condition="ok"),
        ],
    )
    kinds = {g.kind for g in detect_gaps(p)}
    assert "one_armed_decision" in kinds  # B has a single branch
    assert "no_actor" in kinds  # A is a task with no owner
    assert "unreachable" in kinds  # Orphan not reachable from start
    assert "dead_end" in kinds  # Orphan has no outgoing


def test_detect_gaps_no_start_no_end() -> None:
    p = WorkflowProcess(
        steps=[ProcessStep(id="a", name="A", kind="task", actor="X")],
        transitions=[ProcessTransition(source="a", target="a")],
    )
    kinds = {g.kind for g in detect_gaps(p)}
    assert "no_start" in kinds
    assert "no_end" in kinds


def test_clean_process_has_no_gaps() -> None:
    assert detect_gaps(_proc()) == []


def test_to_mermaid_shapes_and_conditions() -> None:
    out = to_mermaid(_proc())
    assert out.startswith("flowchart TD")
    assert 'decide{"Approve?"}' in out  # decision -> diamond
    assert 'start(["Application received"])' in out  # start -> stadium
    assert "decide -->|approved| done" in out
    assert to_mermaid(_proc()) == out  # deterministic


def test_to_plantuml_states_and_terminals() -> None:
    out = to_plantuml(_proc())
    assert out.startswith("@startuml") and out.rstrip().endswith("@enduml")
    assert "[*] --> start" in out
    assert "done --> [*]" in out
    assert "decide --> done : approved" in out


async def test_extract_process_parses_and_failsafe() -> None:
    p = await extract_process("some SOP", gateway=_Gateway(_PROC_JSON))
    assert p.title == "Onboarding"
    assert {s.id for s in p.steps} == {"start", "collect_docs", "verify", "done"}
    empty = await extract_process("x", gateway=_Gateway("not json"))
    assert empty.steps == []


def test_is_process_query() -> None:
    assert is_process_query("Explain the onboarding process")
    assert is_process_query("optimize this workflow")
    assert is_process_query("walk me through the SOP")
    assert not is_process_query("what is the refund amount")


def _svc_with(spans: list[RetrievedSpan]) -> object:
    class _Svc:
        async def retrieve(self, query: object) -> RetrievalResult:
            return RetrievalResult(
                query=query,
                spans=spans,
                max_score=max((s.score for s in spans), default=0.0),
                mode_used=RetrievalMode.HYBRID,
            )

    return _Svc()


async def test_answer_process_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    from svc_control_plane import wiring

    span = RetrievedSpan(chunk_id="c1", doc_id="d1", text="Onboarding SOP.", score=0.8, scorer="cross_encoder")
    monkeypatch.setattr(wiring, "retrieval_service_for", lambda _kb: _mk_async(_svc_with([span])))
    monkeypatch.setattr(wiring, "resolve_tenant_gateway", lambda _kb: _mk_async(_Gateway(_PROC_JSON)))
    ans = await answer_process("bank__ops", "onboarding", audience="business", actor="u")
    assert {s.id for s in ans.process.steps} == {"start", "collect_docs", "verify", "done"}
    assert "flowchart TD" in ans.mermaid and ans.plantuml.startswith("@startuml")
    assert ans.explanation  # non-empty
    assert len(ans.citations) == 1


async def test_process_grounded_answer_converts_and_abstains(monkeypatch: pytest.MonkeyPatch) -> None:
    from svc_control_plane import wiring

    span = RetrievedSpan(chunk_id="c1", doc_id="d1", text="SOP.", score=0.7, scorer="cross_encoder")
    monkeypatch.setattr(wiring, "retrieval_service_for", lambda _kb: _mk_async(_svc_with([span])))
    monkeypatch.setattr(wiring, "resolve_tenant_gateway", lambda _kb: _mk_async(_Gateway(_PROC_JSON)))
    ga = await process_grounded_answer(
        RetrievalQuery(tenant_id="k", text="explain the process", top_k=12, acl=[]), actor="u"
    )
    assert ga.status is AnswerStatus.GROUNDED
    assert ga.structured is not None and "mermaid" in ga.structured and "gaps" in ga.structured

    monkeypatch.setattr(wiring, "retrieval_service_for", lambda _kb: _mk_async(_svc_with([])))
    ga2 = await process_grounded_answer(
        RetrievalQuery(tenant_id="k", text="explain the process", top_k=12, acl=[]), actor="u"
    )
    assert ga2.status is AnswerStatus.ABSTAINED


async def _mk_async(value: object) -> object:
    return value
