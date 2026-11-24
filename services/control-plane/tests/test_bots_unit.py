"""Unit tests for FR-I7 bot formatting helpers (pure)."""

from __future__ import annotations

from svc_control_plane.bots import format_answer, strip_trigger

from rag_schemas import Citation, GroundedAnswer
from rag_schemas.common import AnswerStatus


def test_strip_trigger_removes_leading_word() -> None:
    assert strip_trigger("@rag what is the sla?", "@rag") == "what is the sla?"
    assert strip_trigger("plain question", "@rag") == "plain question"
    assert strip_trigger("  spaced  ", "") == "spaced"


def _answer(status: AnswerStatus, cites: list[str], followups: list[str] | None = None) -> GroundedAnswer:
    return GroundedAnswer(
        tenant_id="t",
        query="q",
        status=status,
        answer="The uptime is 99.9%.",
        citations=[Citation(chunk_id=f"c{i}", doc_id=d, quote="x", score=0.9) for i, d in enumerate(cites)],
        confidence=0.9,
        suggested_followups=followups or [],
    )


def test_format_answer_renders_sources_and_followups() -> None:
    out = format_answer(_answer(AnswerStatus.GROUNDED, ["sla", "sla", "sec"], ["What about security?"]))
    assert "99.9%" in out
    assert "`sla`" in out and "`sec`" in out and out.count("`sla`") == 1  # deduped
    assert "You could also ask" in out


def test_format_answer_abstain_is_flagged() -> None:
    a = GroundedAnswer(
        tenant_id="t",
        query="q",
        status=AnswerStatus.ABSTAINED,
        answer="I don't have enough information.",
        citations=[],
        confidence=0.0,
    )
    assert format_answer(a).startswith(":warning:")
