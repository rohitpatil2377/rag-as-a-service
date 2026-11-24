"""RAGAS governance contract — the maths of the quality gate, verified with no stack.

Proves PASS / FAIL / INCONCLUSIVE precedence, the min-sample rule, the structural
grounded-but-uncited invariant, floor overrides, and the fleet roll-up.
"""

from __future__ import annotations

from svc_control_plane.ragas_governance import (
    CONTRACT_VERSION,
    GateStatus,
    RagasFloors,
    ScoredRun,
    evaluate_fleet,
    evaluate_kb,
    governance_statement,
    prometheus_text,
)


def _clean_run(kb_id: str = "org__kb", n: int = 10) -> ScoredRun:
    """A run that clears every default floor with a sufficient sample."""
    return ScoredRun(
        kb_id=kb_id,
        ragas_scored=n,
        total=n,
        grounded=n - 2,
        abstained=2,
        flagged=0,
        faithfulness_avg=0.92,
        answer_relevancy_avg=0.81,
        context_precision_avg=0.78,
        context_recall_avg=0.75,
    )


def test_clean_run_passes() -> None:
    res = evaluate_kb(_clean_run(), RagasFloors())
    assert res.status is GateStatus.PASS
    assert res.breaches == []
    assert all(v.passed for v in res.verdicts if v.value is not None)


def test_below_faithfulness_floor_fails() -> None:
    run = _clean_run().model_copy(update={"faithfulness_avg": 0.62})
    res = evaluate_kb(run, RagasFloors())
    assert res.status is GateStatus.FAIL
    assert any("faithfulness" in b for b in res.breaches)


def test_small_sample_is_inconclusive_not_pass() -> None:
    # A perfect score on ONE question must NOT certify.
    run = _clean_run(n=1)
    res = evaluate_kb(run, RagasFloors())
    assert res.status is GateStatus.INCONCLUSIVE
    assert res.sample == 1
    assert any("min" in n for n in res.notes)


def test_defect_on_small_sample_still_fails() -> None:
    # A real breach outranks the small-sample rule — a defect is a defect at any n.
    run = _clean_run(n=2).model_copy(update={"context_recall_avg": 0.30})
    res = evaluate_kb(run, RagasFloors())
    assert res.status is GateStatus.FAIL


def test_grounded_but_uncited_is_structural_fail() -> None:
    run = _clean_run().model_copy(update={"integrity_violations": 1})
    res = evaluate_kb(run, RagasFloors())
    assert res.status is GateStatus.FAIL
    assert any("uncited" in b or "invariant" in b for b in res.breaches)


def test_unscored_metric_is_excluded_not_failed() -> None:
    # A metric that was never scored (None) is a note, not a breach.
    run = _clean_run().model_copy(update={"context_recall_avg": None})
    res = evaluate_kb(run, RagasFloors())
    assert res.status is GateStatus.PASS
    recall = next(v for v in res.verdicts if v.metric == "context_recall")
    assert recall.passed is None
    assert any("context_recall" in n for n in res.notes)


def test_flag_rate_ceiling_optional() -> None:
    run = _clean_run(n=10).model_copy(update={"flagged": 6})
    # No ceiling by default → still passes.
    assert evaluate_kb(run, RagasFloors()).status is GateStatus.PASS
    # With a ceiling it fails.
    res = evaluate_kb(run, RagasFloors(max_flag_rate=0.2))
    assert res.status is GateStatus.FAIL
    assert any("flag_rate" in b for b in res.breaches)


def test_floor_override_raises_the_bar() -> None:
    run = _clean_run().model_copy(update={"faithfulness_avg": 0.85})
    assert evaluate_kb(run, RagasFloors()).status is GateStatus.PASS
    assert evaluate_kb(run, RagasFloors(faithfulness_min=0.90)).status is GateStatus.FAIL


def test_env_floors(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RAG_RAGAS_FAITHFULNESS_MIN", "0.95")
    monkeypatch.setenv("RAG_RAGAS_MIN_SAMPLE", "3")
    floors = RagasFloors.from_env()
    assert floors.faithfulness_min == 0.95
    assert floors.min_sample == 3


def test_fleet_pass_requires_all_pass() -> None:
    runs = [_clean_run("a__1"), _clean_run("a__2")]
    rep = evaluate_fleet(runs, RagasFloors())
    assert rep.status is GateStatus.PASS
    assert rep.passed == 2


def test_fleet_any_fail_fails_fleet() -> None:
    runs = [_clean_run("a__1"), _clean_run("a__2").model_copy(update={"faithfulness_avg": 0.1})]
    rep = evaluate_fleet(runs, RagasFloors())
    assert rep.status is GateStatus.FAIL
    assert rep.failed == 1


def test_fleet_inconclusive_when_some_thin_none_failing() -> None:
    runs = [_clean_run("a__1"), _clean_run("a__2", n=1)]
    rep = evaluate_fleet(runs, RagasFloors())
    assert rep.status is GateStatus.INCONCLUSIVE
    assert rep.inconclusive == 1


def test_empty_fleet_fails() -> None:
    assert evaluate_fleet([], RagasFloors()).status is GateStatus.FAIL


def test_prometheus_text_shape() -> None:
    rep = evaluate_fleet([_clean_run("org__kb")], RagasFloors())
    text = prometheus_text(rep)
    assert "rag_ragas_gate_pass{" in text
    assert 'kb="org__kb"' in text
    assert "rag_ragas_metric{" in text
    assert "rag_ragas_fleet_status{" in text


def test_governance_statement_mentions_floors_and_version() -> None:
    s = governance_statement(RagasFloors())
    assert "faithfulness" in s
    assert CONTRACT_VERSION in s
