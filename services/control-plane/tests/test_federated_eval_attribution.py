"""Federated-eval attribution — the pure per-peer marginal logic (no I/O).

``attribute`` turns baseline vs leave-one-out answer metrics into a per-peer verdict, tied to
the grant(s) that authorised the peer. A peer DEGRADES the joint answer when removing it
improves a metric; it HELPS when removing it worsens one; else neutral (or unused if never
cited).
"""

from __future__ import annotations

from svc_control_plane.federated_eval import (
    AnswerMetrics,
    QuestionEval,
    attribute,
    peer_of_doc,
)


def _m(
    *, grounded: bool = True, faith: float | None = None, hit: bool | None = None, cited: list[str] | None = None
) -> AnswerMetrics:
    return AnswerMetrics(grounded=grounded, faithfulness=faith, hit=hit, cited_peers=frozenset(cited or []))


def test_peer_of_doc_only_matches_federated_departments() -> None:
    peers = frozenset({"hr", "finance"})
    assert peer_of_doc("hr/leave", peers) == "hr"  # a peer's labelled span
    assert peer_of_doc("budget/doc1", peers) is None  # an OWN kb label, not a peer
    assert peer_of_doc("finance/x", peers) == "finance"


def test_a_peer_that_makes_the_answer_wrong_is_flagged_degrading() -> None:
    # Baseline (all peers) gets it WRONG; removing `bad` makes it right -> bad degrades.
    evals = [
        QuestionEval(
            "q",
            baseline=_m(hit=False, cited=["bad", "good"]),
            ablations={
                "bad": _m(hit=True),  # without bad -> correct
                "good": _m(hit=False),  # without good -> still wrong
            },
        )
    ]
    peers = {p.peer_dept: p for p in attribute(evals, {"bad": ["g_bad"], "good": ["g_good"]})}
    assert peers["bad"].degrades and peers["bad"].verdict == "degrades"
    assert peers["bad"].hit_marginal == -1.0
    assert peers["bad"].grant_ids == ["g_bad"]  # attributed to its grant
    assert not peers["good"].degrades


def test_a_peer_that_supplies_the_answer_helps() -> None:
    # Baseline right; removing `good` makes it wrong -> good helps (positive marginal).
    evals = [
        QuestionEval(
            "q",
            baseline=_m(hit=True, faith=0.9, cited=["good"]),
            ablations={"good": _m(hit=False, faith=0.4)},
        )
    ]
    [good] = attribute(evals, {"good": ["g"]})
    assert good.verdict == "helps"
    assert good.hit_marginal == 1.0
    assert good.faithfulness_marginal is not None and good.faithfulness_marginal > 0
    assert not good.degrades


def test_an_uncited_peer_is_unused_not_degrading() -> None:
    evals = [
        QuestionEval("q", baseline=_m(hit=True, cited=["good"]), ablations={"good": _m(hit=True), "idle": _m(hit=True)})
    ]
    peers = {p.peer_dept: p for p in attribute(evals, {"good": ["g1"], "idle": ["g2"]})}
    assert peers["idle"].questions_cited == 0
    assert peers["idle"].verdict == "unused" and not peers["idle"].degrades
    assert peers["good"].verdict == "neutral"  # cited but removing it changed nothing here


def test_faithfulness_degradation_alone_flags_a_peer() -> None:
    # Removing the peer RAISES faithfulness beyond the noise floor -> degrades on faithfulness.
    evals = [QuestionEval("q", baseline=_m(faith=0.5, cited=["noisy"]), ablations={"noisy": _m(faith=0.9)})]
    [noisy] = attribute(evals, {"noisy": ["g"]})
    assert noisy.degrades and noisy.faithfulness_marginal == -0.4
