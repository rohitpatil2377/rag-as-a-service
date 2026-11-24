"""svc_guardrails_eval — audit + evaluation (TDD §4.7)."""

from svc_guardrails_eval.audit import AuditService
from svc_guardrails_eval.evaluation import EvalReport, GoldenItem, evaluate
from svc_guardrails_eval.suite import (
    GoldQuery,
    GoldSuite,
    QueryScore,
    SuiteFloors,
    SuiteReport,
    load_suite,
    score_query,
    score_suite,
)

__all__ = [
    "AuditService",
    "EvalReport",
    "GoldQuery",
    "GoldSuite",
    "GoldenItem",
    "QueryScore",
    "SuiteFloors",
    "SuiteReport",
    "evaluate",
    "load_suite",
    "score_query",
    "score_suite",
]
