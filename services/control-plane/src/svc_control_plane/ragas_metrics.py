"""Sovereign RAG-quality metrics (Ragas-style), computed on OUR OWN models.

Ragas made a set of RAG metrics standard — answer relevancy, context precision, context
recall — that measure what our existing faithfulness gate cannot: whether RETRIEVAL pulled
the right evidence (an answer can be perfectly faithful to the WRONG context and still be
wrong). This module implements those metric DEFINITIONS natively so they run on the
platform's own judge (self-hosted Sarvam) and embedder (BGE-M3) — no ``ragas``/LangChain
dependency (ADR-0002: no RAG-framework lock-in) and no external LLM call (sovereignty:
``assert_sovereign`` would fail-closed on one anyway).

The metric maths are pure; the LLM judge and the embedder are INJECTED (``Judge`` / ``Embed``
callables), so the definitions are unit-testable with deterministic fakes and, in
production, bound to the tenant's real gateway/embedder.
"""

from __future__ import annotations

import math
import re
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

# A judge maps a prompt to the model's completion; an embedder maps texts to vectors. In
# production these wrap ``LLMGateway.complete`` and ``Embedder.embed``; tests inject fakes.
Judge = Callable[[str], Awaitable[str]]
Embed = Callable[[list[str]], Awaitable[list[list[float]]]]

_FIRST_TOKEN = re.compile(r"[A-Za-z01]+")


def _verdict(text: str) -> bool:
    """Parse a judge reply to a boolean, fail-closed on anything that is not an explicit YES.

    The judge is always prompted to reply with EXACTLY YES or NO, so the reliable signal is
    the FIRST token — "YES"/"NO"/"YES." parse cleanly, while a hedged "not relevant …" reads
    as its first token ("NOT") and fails closed to False. Fail-closed matters here: an
    ambiguous verdict must never inflate a quality score in an audit-grade metric."""
    m = _FIRST_TOKEN.search(text)
    if m is None:
        return False
    return m.group(0).upper() in {"YES", "TRUE", "1"}


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def _split_claims(text: str) -> list[str]:
    """Split a ground-truth answer into atomic claims (sentence-ish), dropping empties."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


async def answer_relevancy(question: str, answer: str, *, judge: Judge, embed: Embed, n: int = 3) -> float:
    """How well the answer ADDRESSES the question (Ragas definition). The judge reverse-
    engineers ``n`` questions the answer would answer; we embed them + the real question and
    take the mean cosine similarity. An on-topic answer yields questions close to the
    original; an evasive or padded one drifts away. Reference-free. 0.0 for an empty answer.
    """
    if not answer.strip():
        return 0.0
    prompt = (
        "Given ONLY the answer below, write "
        f"{n} distinct questions that this answer would correctly and fully answer. "
        "One question per line, no numbering.\n\nAnswer:\n" + answer
    )
    reply = await judge(prompt)
    gen = [q.strip(" -*\t") for q in reply.splitlines() if q.strip()][:n]
    if not gen:
        return 0.0
    vecs = await embed([question, *gen])
    q_vec, gen_vecs = vecs[0], vecs[1:]
    sims = [_cosine(q_vec, g) for g in gen_vecs]
    return round(max(0.0, sum(sims) / len(sims)), 4)


async def context_precision(question: str, contexts: list[str], *, judge: Judge, reference: str | None = None) -> float:
    """Are the RETRIEVED contexts that are useful ranked near the TOP (Ragas definition)?
    Each context is judged useful/not for answering the question, then scored as
    average-precision over the ranking: sum_k(precision@k * useful_k) / (#useful). A retriever
    that surfaces the key passage first scores ~1.0; one that buries it scores low. Empty or
    all-useless contexts score 0.0.

    ``reference`` (the grounded answer / ground truth) anchors the usefulness judgement — the
    Ragas LLM formulation asks "did this context help arrive at the ANSWER", not the vaguer "is
    it useful", which a small judge (Sarvam) answers too conservatively (marking passages that
    demonstrably grounded the answer as not-useful, and flipping across runs). Anchoring on the
    answer makes the verdict stable and correct; without a reference we fall back to the
    question-only prompt so callers/tests that have no answer still work."""
    if not contexts:
        return 0.0
    verdicts: list[int] = []
    for ctx in contexts:
        if reference:
            prompt = (
                "Given a QUESTION, the correct ANSWER, and a retrieved CONTEXT, decide whether the "
                "CONTEXT contains information that supports or contributes to the ANSWER (even "
                "partially). Reply with exactly YES or NO.\n\n"
                f"QUESTION: {question}\n\nANSWER: {reference}\n\nCONTEXT:\n{ctx}"
            )
        else:
            prompt = (
                "Is the CONTEXT useful for answering the QUESTION? Reply with exactly YES or NO.\n\n"
                f"QUESTION: {question}\n\nCONTEXT:\n{ctx}"
            )
        verdicts.append(1 if _verdict(await judge(prompt)) else 0)
    total_useful = sum(verdicts)
    if total_useful == 0:
        return 0.0
    running = 0
    weighted = 0.0
    for k, v in enumerate(verdicts, start=1):
        if v:
            running += 1
            weighted += running / k  # precision@k at each useful hit
    return round(weighted / total_useful, 4)


async def context_recall(ground_truth: str, contexts: list[str], *, judge: Judge) -> float:
    """Did retrieval bring back the evidence the GROUND TRUTH needs (Ragas definition)?
    The ground-truth answer is split into claims; each claim is judged attributable/not to the
    retrieved contexts; recall = attributable / total. This is the metric that catches
    "confidently grounded but retrieved the wrong thing" — our current blind spot. 0.0 when
    there is no ground truth to check against."""
    claims = _split_claims(ground_truth)
    if not claims or not contexts:
        return 0.0
    joined = "\n\n".join(contexts)
    supported = 0
    for claim in claims:
        prompt = (
            "Can the CLAIM be attributed to (supported by) the CONTEXT below? "
            "Reply with exactly YES or NO.\n\n"
            f"CLAIM: {claim}\n\nCONTEXT:\n{joined}"
        )
        if _verdict(await judge(prompt)):
            supported += 1
    return round(supported / len(claims), 4)


class RagasScores(BaseModel):
    """One question's sovereign RAG-quality scores (each in [0,1]; None when not computable)."""

    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    faithfulness: float | None = Field(default=None, description="Carried from the pipeline's own gate.")


class RagasAggregate(BaseModel):
    """Mean scores across an evaluation set + the count each mean is over."""

    n: int
    answer_relevancy_avg: float | None = None
    context_precision_avg: float | None = None
    context_recall_avg: float | None = None
    faithfulness_avg: float | None = None


def aggregate(scores: list[RagasScores]) -> RagasAggregate:
    """Mean each metric over the rows that have it (a None row is excluded from that mean,
    not counted as zero — a metric we couldn't compute must not drag the average down)."""

    def _avg(vals: list[float]) -> float | None:
        return round(sum(vals) / len(vals), 4) if vals else None

    return RagasAggregate(
        n=len(scores),
        answer_relevancy_avg=_avg([s.answer_relevancy for s in scores if s.answer_relevancy is not None]),
        context_precision_avg=_avg([s.context_precision for s in scores if s.context_precision is not None]),
        context_recall_avg=_avg([s.context_recall for s in scores if s.context_recall is not None]),
        faithfulness_avg=_avg([s.faithfulness for s in scores if s.faithfulness is not None]),
    )
