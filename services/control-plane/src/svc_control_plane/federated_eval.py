"""Federated evaluation (FR-G9 mesh edge) — per-grant quality attribution.

Continuous eval already scores a KB's OWN answers (:mod:`continuous_eval`). Once a KB
federates, the joint answer also depends on PEERS, and a peer can quietly DEGRADE it —
grounded-but-wrong content that outranks the truth, or noise that drags faithfulness down.
This attributes the joint answer's quality to each contributing peer/grant, by counterfactual
ablation:

    baseline      = answer with ALL granted peers
    ablation(p)   = answer with peer p REMOVED
    marginal(p)   = baseline_metric - ablation(p)_metric

A NEGATIVE marginal means removing the peer IMPROVED the answer — i.e. the peer degrades it.
The attribution is per DEPARTMENT, tied back to the grant(s) that authorised it, so a
degrading peer is both caught and named. Pure attribution (:func:`attribute`) is I/O-free and
unit-tested; :func:`run_federated_eval` drives the pipeline and records the audit + alerts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from rag_schemas.common import AnswerStatus, AuditAction
from svc_control_plane import pipeline, wiring
from svc_control_plane.quality import MAX_EVAL_QUESTIONS, EvalQuestion

# Removing a peer must change a metric by more than this to count as help/harm (not noise).
DEGRADE_EPS = 0.05


@dataclass(frozen=True, slots=True)
class AnswerMetrics:
    """The quality of ONE answer, plus which peers its citations came from."""

    grounded: bool
    faithfulness: float | None
    hit: bool | None
    cited_peers: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class QuestionEval:
    """One golden question: the all-peers baseline + each peer's leave-one-out ablation."""

    question: str
    baseline: AnswerMetrics
    ablations: dict[str, AnswerMetrics]  # peer_dept -> metrics with THAT peer removed


class PeerQuality(BaseModel):
    """One peer's attributed contribution to the joint answer quality."""

    peer_dept: str
    grant_ids: list[str] = Field(default_factory=list)
    questions_total: int
    questions_cited: int = Field(description="Golden questions whose baseline answer cited this peer.")
    faithfulness_marginal: float | None = Field(
        default=None, description="Mean (baseline - without-peer) faithfulness; <0 means the peer lowers it."
    )
    grounded_marginal: float = Field(description="Mean (baseline - without-peer) grounded rate; <0 means it degrades.")
    hit_marginal: float | None = Field(
        default=None, description="Mean (baseline - without-peer) expected-fact hit rate; <0 means it degrades."
    )
    degrades: bool = Field(description="Removing the peer IMPROVED a metric beyond the noise floor.")
    verdict: str = Field(description="degrades | helps | neutral | unused")


class FederatedEvalReport(BaseModel):
    """Per-grant mesh-quality attribution for a federated golden run."""

    caller_dept: str
    generated_at: datetime
    federate_from: list[str]
    questions: int
    peers: list[PeerQuality] = Field(default_factory=list)
    degrading_peers: list[str] = Field(default_factory=list)


def peer_of_doc(doc_id: str, peers: frozenset[str]) -> str | None:
    """The federated PEER a citation came from, or None for own content. Federated spans are
    labelled ``<peer_dept>/<doc_id>`` (svc_control_plane.federation), so the prefix is the dept —
    but only when it is one of the departments we actually federated from (an own KB's label
    like ``budget`` must never be mistaken for a peer)."""
    head = doc_id.split("/", 1)[0]
    return head if head in peers else None


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def attribute(
    evals: list[QuestionEval],
    peer_grants: dict[str, list[str]],
    *,
    eps: float = DEGRADE_EPS,
) -> list[PeerQuality]:
    """Per-peer marginal quality from baseline vs leave-one-out ablations (pure).

    ``peer_grants`` maps a peer dept to the grant ids that authorised it (for attribution).
    A peer degrades the joint answer when removing it IMPROVES any metric by more than ``eps``.
    """
    out: list[PeerQuality] = []
    for peer, grant_ids in peer_grants.items():
        g_marg: list[float] = []
        f_marg: list[float] = []
        h_marg: list[float] = []
        cited = 0
        for ev in evals:
            without = ev.ablations.get(peer)
            if without is None:
                continue
            if peer in ev.baseline.cited_peers:
                cited += 1
            g_marg.append(float(ev.baseline.grounded) - float(without.grounded))
            if ev.baseline.faithfulness is not None and without.faithfulness is not None:
                f_marg.append(ev.baseline.faithfulness - without.faithfulness)
            if ev.baseline.hit is not None and without.hit is not None:
                h_marg.append(float(ev.baseline.hit) - float(without.hit))
        grounded_marginal = _mean(g_marg) or 0.0
        faith_marginal = _mean(f_marg)
        hit_marginal = _mean(h_marg)
        degrades = (
            grounded_marginal <= -eps
            or (faith_marginal is not None and faith_marginal <= -eps)
            or (hit_marginal is not None and hit_marginal <= -eps)
        )
        helps = (
            grounded_marginal >= eps
            or (faith_marginal is not None and faith_marginal >= eps)
            or (hit_marginal is not None and hit_marginal >= eps)
        )
        verdict = "degrades" if degrades else ("helps" if helps else ("unused" if cited == 0 else "neutral"))
        out.append(
            PeerQuality(
                peer_dept=peer,
                grant_ids=grant_ids,
                questions_total=len(evals),
                questions_cited=cited,
                faithfulness_marginal=faith_marginal,
                grounded_marginal=grounded_marginal,
                hit_marginal=hit_marginal,
                degrades=degrades,
                verdict=verdict,
            )
        )
    return out


async def _answer_metrics(
    caller_dept: str,
    own_kbs: list[str],
    federate_from: list[str],
    q: EvalQuestion,
    peers_universe: frozenset[str],
    *,
    actor: str,
) -> AnswerMetrics:
    answer = await pipeline.answer_federated(
        caller_dept, own_kbs, federate_from, q.question, top_k=5, acl=[], actor=f"fedeval:{actor}"
    )
    cited = frozenset(p for c in answer.citations if (p := peer_of_doc(c.doc_id, peers_universe)) is not None)
    hit = q.expected.lower() in answer.answer.lower() if q.expected else None
    return AnswerMetrics(
        grounded=answer.status is AnswerStatus.GROUNDED,
        faithfulness=answer.faithfulness_score,
        hit=hit,
        cited_peers=cited,
    )


async def run_federated_eval(
    caller_dept: str,
    own_kbs: list[str],
    federate_from: list[str],
    questions: list[EvalQuestion],
    *,
    actor: str = "scheduler",
    now: datetime | None = None,
) -> FederatedEvalReport:
    """Score a federated golden with per-peer counterfactual ablation, attribute each peer's
    contribution to its grant(s), and record a FEDERATED_EVAL audit event (+ a
    FEDERATION_QUALITY_ALERT naming any peer that DEGRADES the joint answer)."""
    universe = frozenset(federate_from)
    # Which grant(s) authorise each peer, for attribution (empty = reachable but no active grant).
    grant_store = wiring.grant_store()
    peer_grants: dict[str, list[str]] = {}
    for peer in federate_from:
        active = await grant_store.get_active(issuer_dept_id=peer, subject_dept_id=caller_dept)
        peer_grants[peer] = [g.grant_id for g in active]

    evals: list[QuestionEval] = []
    for q in questions[:MAX_EVAL_QUESTIONS]:
        baseline = await _answer_metrics(caller_dept, own_kbs, federate_from, q, universe, actor=actor)
        ablations: dict[str, AnswerMetrics] = {}
        for peer in federate_from:
            subset = [p for p in federate_from if p != peer]
            ablations[peer] = await _answer_metrics(caller_dept, own_kbs, subset, q, universe, actor=actor)
        evals.append(QuestionEval(question=q.question, baseline=baseline, ablations=ablations))

    peers = attribute(evals, peer_grants)
    degrading = [p.peer_dept for p in peers if p.degrades]
    report = FederatedEvalReport(
        caller_dept=caller_dept,
        generated_at=now or datetime.now(UTC),
        federate_from=federate_from,
        questions=len(evals),
        peers=peers,
        degrading_peers=degrading,
    )

    audit = wiring.audit_service()
    trace_id = report.generated_at.isoformat()
    await audit.record(
        tenant_id=caller_dept,
        trace_id=trace_id,
        action=AuditAction.FEDERATED_EVAL,
        actor=actor,
        payload=report.model_dump(mode="json"),
    )
    # A degrading peer is a governance signal: name it, its grants, and the marginal, so a
    # reviewer can revoke the grant. Its own action so the SOC/SIEM can alert directly.
    for p in peers:
        if p.degrades:
            await audit.record(
                tenant_id=caller_dept,
                trace_id=trace_id,
                action=AuditAction.FEDERATION_QUALITY_ALERT,
                actor=actor,
                payload={
                    "peer_dept": p.peer_dept,
                    "grant_ids": p.grant_ids,
                    "grounded_marginal": p.grounded_marginal,
                    "faithfulness_marginal": p.faithfulness_marginal,
                    "hit_marginal": p.hit_marginal,
                },
            )
    return report
