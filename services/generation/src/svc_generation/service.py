"""Generation & Grounding service (TDD §4.5 · FR-E1/E2/E3/E6 + E4 faithfulness).

Enforces "grounded or silent", then adds the FR-E4 faithfulness pass with three-way
routing:

* retrieval confidence < abstain_threshold        -> ABSTAINED (no model call)
* otherwise generate, then faithfulness-check the answer vs its cited spans:
    score >= faithful_threshold                    -> GROUNDED
    flag_threshold <= score < faithful_threshold   -> FLAGGED_FOR_REVIEW (HITL, not final)
    score < flag_threshold                         -> ABSTAINED (faithfulness too low)

Every path is audited (FR-G4): the retrieval abstain via ABSTAIN, and the faithfulness
verdict via a distinct, independently-queryable FAITHFULNESS event carrying the score +
unsupported_claims. Citations are always structural (from retrieved spans), never
scraped from model text.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable

from svc_guardrails_eval import AuditService

from rag_common import get_logger, get_settings
from rag_core import (
    FaithfulnessChecker,
    FaithfulnessVerdict,
    GuardBudget,
    Guardrail,
    abstain,
    as_explicit_request,
    build_completeness,
    build_continuation_prompt,
    build_coverage_reprompt,
    build_prompt,
    continuation_bound,
    escalated_budget,
    flag_for_review,
    gate_for,
    ground,
    is_refusal_text,
    is_truncated,
    should_abstain,
    stitch_continuation,
    uncovered_spans,
    with_parent_context,
)
from rag_core.faithfulness import improves_on, route_unless_unjudged
from rag_core.interfaces import LLMGateway
from rag_schemas import AgentStep, Completeness, GroundedAnswer, RetrievalResult, RetrievedSpan
from rag_schemas.common import AnswerStatus, AuditAction

_log = get_logger(__name__)

# How many times to re-call the model when it returns no final answer (empty / null content).
# Sarvam-105b misses ~1-in-10; 2 retries (3 attempts total) drives that to ~0.1%.
_EMPTY_GEN_RETRIES = 2


def _prompt_spans(spans: list[RetrievedSpan], top_k: int) -> list[RetrievedSpan]:
    """The passages the model actually sees: ``top_k`` RANKED ones, plus their contiguity
    neighbours.

    ``top_k`` answers "how many passages did retrieval judge relevant". A contiguity neighbour was
    never judged — it is present because it sits beside one that was, and it carries no score for
    exactly that reason. Counting it against ``top_k`` would let expansion evict the very ranked
    passage it was fetched to complete, which is not a trade-off anyone asked for: measured on a
    real handbook, expansion recovered a missing list item and pushed a DIFFERENT one out, leaving
    the answer still incomplete at five of six.

    Neighbours are bounded upstream (expand_top x window x 2 sides), so this cannot run away.
    Document order within the whole set is preserved by the caller's merge, so the model reads a
    split list in the order it was written."""
    ranked: list[RetrievedSpan] = []
    neighbours: list[RetrievedSpan] = []
    for span in spans:
        (neighbours if span.scorer == "unscored" else ranked).append(span)
    keep = {s.chunk_id for s in ranked[:top_k]} | {s.chunk_id for s in neighbours}
    return [s for s in spans if s.chunk_id in keep]


class GenerationService:
    def __init__(
        self,
        llm: LLMGateway,
        audit: AuditService,
        faithfulness: FaithfulnessChecker | None = None,
        *,
        output_guard: Guardrail | None = None,
        gateway_resolver: Callable[[str, str | None], Awaitable[LLMGateway]] | None = None,
        faithfulness_resolver: Callable[[str], Awaitable[FaithfulnessChecker | None]] | None = None,
        notation_resolver: Callable[[FaithfulnessChecker, str], Awaitable[FaithfulnessChecker]] | None = None,
        abstain_threshold: float | None = None,
        faithful_threshold: float | None = None,
        flag_threshold: float | None = None,
        faithfulness_enabled: bool | None = None,
        guardrails_enabled: bool | None = None,
        coverage_check_enabled: bool | None = None,
        coverage_min_score: float | None = None,
        coverage_ratio: float | None = None,
        coverage_max_spans: int | None = None,
        refusal_retry_min_score: float | None = None,
        completeness_guard_enabled: bool | None = None,
        completeness_max_continuations: int | None = None,
        completeness_continuation_budget_factor: float | None = None,
        completeness_coverage_max_loops: int | None = None,
        completeness_max_guard_calls: int | None = None,
        completeness_faithfulness_tolerance: float | None = None,
    ) -> None:
        settings = get_settings()
        self._llm = llm
        # Optional per-tenant resolvers (FR-E1 BYO-model): a tenant's own model both
        # generates AND judges faithfulness; both fall back to the injected defaults.
        self._resolve_gateway = gateway_resolver
        self._resolve_faithfulness = faithfulness_resolver
        # Per-KB notation-grounding toggle (Slice 4): given the resolved checker + kb_id, returns a
        # notation-aware view of it. None ⇒ notation is left at whatever the checker was built with.
        self._apply_notation = notation_resolver
        self._audit = audit
        self._faithfulness = faithfulness
        self._output_guard = output_guard
        self._guardrails_enabled = guardrails_enabled if guardrails_enabled is not None else settings.guardrails_enabled
        self._abstain_threshold = abstain_threshold if abstain_threshold is not None else settings.abstain_threshold
        # The same gate on the scale a lexical reranker produces. An explicit per-KB override
        # (`abstain_threshold`) still wins outright: an operator who set a number for their KB
        # meant it, whichever scorer is serving.
        self._abstain_threshold_lexical = (
            abstain_threshold if abstain_threshold is not None else settings.abstain_threshold_lexical
        )
        self._faithful_threshold = faithful_threshold if faithful_threshold is not None else settings.faithful_threshold
        self._flag_threshold = flag_threshold if flag_threshold is not None else settings.flag_threshold
        self._faith_enabled = (
            faithfulness_enabled if faithfulness_enabled is not None else settings.faithfulness_enabled
        )
        # One bounded re-attempt when the ANSWER fails verification (never loops: a retry is
        # scored once and kept only if it beats the original).
        self._self_correct = settings.self_correction_enabled
        # Coverage-aware completeness re-prompt (flag-gated, default OFF). When OFF, answer() is
        # byte-identical: none of these are read on any path.
        self._coverage_enabled = (
            coverage_check_enabled if coverage_check_enabled is not None else settings.coverage_check_enabled
        )
        self._coverage_min_score = coverage_min_score if coverage_min_score is not None else settings.coverage_min_score
        self._coverage_ratio = coverage_ratio if coverage_ratio is not None else settings.coverage_ratio
        self._coverage_max_spans = coverage_max_spans if coverage_max_spans is not None else settings.coverage_max_spans
        # Re-framed retry on a model refusal (default OFF -- see the setting for why). 0 disables.
        self._refusal_retry_min_score = (
            refusal_retry_min_score if refusal_retry_min_score is not None else settings.refusal_retry_min_score
        )
        # Strict, adaptive completeness guard (Slice 1). When ON it SUPERSEDES the single coverage
        # re-prompt above: a truncated GROUNDED answer is continued in bounded steps, an incomplete
        # one runs a bounded coverage LOOP, and if neither resolves within the bounds the answer is
        # MARKED (completeness.complete=false) rather than shipped whole. Default OFF ⇒ byte-identical.
        # The ceilings are per-KB overridable (answer() kwargs, resolved from kb_threshold_overrides).
        self._completeness_enabled = (
            completeness_guard_enabled
            if completeness_guard_enabled is not None
            else settings.completeness_guard_enabled
        )
        self._completeness_max_continuations = (
            completeness_max_continuations
            if completeness_max_continuations is not None
            else settings.completeness_max_continuations
        )
        self._completeness_budget_factor = (
            completeness_continuation_budget_factor
            if completeness_continuation_budget_factor is not None
            else settings.completeness_continuation_budget_factor
        )
        self._completeness_coverage_max_loops = (
            completeness_coverage_max_loops
            if completeness_coverage_max_loops is not None
            else settings.completeness_coverage_max_loops
        )
        self._completeness_max_guard_calls = (
            completeness_max_guard_calls
            if completeness_max_guard_calls is not None
            else settings.completeness_max_guard_calls
        )
        self._completeness_faithfulness_tolerance = (
            completeness_faithfulness_tolerance
            if completeness_faithfulness_tolerance is not None
            else settings.completeness_faithfulness_tolerance
        )
        self._llm_max_tokens = settings.llm_max_tokens
        self._heavy_max_tokens = settings.heavy_max_tokens

    async def answer(  # noqa: PLR0911, PLR0912, PLR0915 - a linear pipeline of guarded stages; splitting it hides the order
        self,
        retrieval: RetrievalResult,
        *,
        actor: str,
        trace_id: str,
        threshold: float | None = None,
        faithful_threshold: float | None = None,
        flag_threshold: float | None = None,
        model: str | None = None,
        persona: str | None = None,
        depth: str = "brief",
        max_tokens: int | None = None,
        faithfulness_enabled: bool | None = None,
        output_guard_enabled: bool | None = None,
        kb_id: str | None = None,
        max_continuations: int | None = None,
        continuation_budget_factor: float | None = None,
        coverage_max_loops: int | None = None,
        max_guard_calls: int | None = None,
        faithfulness_tolerance: float | None = None,
        completeness_coverage: bool = True,
        guard_budget: GuardBudget | None = None,
    ) -> GroundedAnswer:
        query = retrieval.query
        tenant_id = query.tenant_id
        # The user's ORIGINAL question for the USER-FACING answer (GroundedAnswer.query) + its
        # citations, while retrieval/generation run on `query.text` (possibly glossary-expanded).
        # None original_text ⇒ this is just `query.text`, so the un-expanded path is unchanged.
        display_query = query.original_text or query.text
        # Self-attribute this answer's audit trail to its KB (``payload.kb_id``) so a KB-scoped
        # union read recovers it even if it lands in the org schema. Passed ONLY for a single-KB
        # ask; a genuine cross-KB merge has no single owning KB and leaves it None.
        # Per-request guardrail toggles (from the hierarchical Guardrail policy, resolved by the
        # caller). None -> fall back to the instance/platform setting, so behaviour is unchanged
        # unless a KB/org sets the toggle.
        faith_on = self._faith_enabled if faithfulness_enabled is None else faithfulness_enabled
        guard_on = self._guardrails_enabled if output_guard_enabled is None else output_guard_enabled
        # The threshold must belong to the scale `max_score` is on — a cross-encoder
        # probability and a term-overlap fraction are different numbers (grounding.gate_for).
        gate = (
            threshold
            if threshold is not None
            else gate_for(
                retrieval.spans,
                cross_encoder=self._abstain_threshold,
                lexical=self._abstain_threshold_lexical,
            )
        )
        faithful_gate = faithful_threshold if faithful_threshold is not None else self._faithful_threshold
        flag_gate = flag_threshold if flag_threshold is not None else self._flag_threshold

        # --- Retrieval-confidence abstain (audited) ---
        # No spans at all (e.g. everything filtered out by a recency window, FR-D9) => abstain
        # regardless of the score gate; there is literally nothing to ground an answer on.
        if not retrieval.spans or should_abstain(retrieval.max_score, gate):
            await self._audit.record(
                tenant_id=tenant_id,
                kb_id=kb_id,
                trace_id=trace_id,
                action=AuditAction.ABSTAIN,
                actor=actor,
                payload={
                    "query": query.text,
                    "max_score": retrieval.max_score,
                    "threshold": gate,
                    "reason": "retrieval_confidence_below_threshold",
                    "retrieved_chunk_ids": [s.chunk_id for s in retrieval.spans],
                },
            )
            _log.info("abstained_retrieval", tenant_id=tenant_id, max_score=retrieval.max_score)
            return abstain(tenant_id, display_query, retrieval.max_score, trace_id=trace_id)

        # --- Generate (grounded, cited) ---
        spans = _prompt_spans(retrieval.spans, query.top_k)
        prompt = build_prompt(query.text, spans, persona=persona, depth=depth)
        # Use the tenant's own model if they've configured one (FR-E1), else the default.
        llm = await self._resolve_gateway(tenant_id, model) if self._resolve_gateway else self._llm
        model_name = llm.model_name
        # Token metering (FR-H4): use the usage-reporting path when the gateway has one
        # (server-reported tokens; estimated only when the server omits usage). Plain
        # stubs without it still work via complete().
        with_usage = getattr(llm, "complete_with_usage", None)

        async def _generate() -> tuple[str, dict[str, int | bool | str]]:
            if with_usage is not None:
                result: tuple[str, dict[str, int | bool | str]] = await with_usage(prompt, max_tokens=max_tokens)
                return result
            return await llm.complete(prompt, max_tokens=max_tokens), {}

        answer_text, usage = await _generate()
        # A REASONING model (e.g. Sarvam-105b) intermittently returns NO final answer — either an
        # empty completion (whole token budget spent in <think>) or content=null (answer left in
        # reasoning_content; the gateway maps that to empty). Measured live: Sarvam returns null
        # content ~1-in-10 calls. That is a generation-layer transient, not an unfaithful answer,
        # so RETRY (generation is non-deterministic → a retry almost always lands a real answer):
        # a few attempts drive a ~10% miss to ~0.1%. If every attempt is empty, abstain with an
        # HONEST reason rather than shipping "None" or faking a faithfulness failure.
        for _attempt in range(_EMPTY_GEN_RETRIES):
            if answer_text.strip():
                break
            _log.warning("empty_generation_retry", tenant_id=tenant_id, model=model_name, attempt=_attempt + 1)
            answer_text, usage = await _generate()
        if not answer_text.strip():
            await self._audit.record(
                tenant_id=tenant_id,
                kb_id=kb_id,
                trace_id=trace_id,
                action=AuditAction.GENERATE,
                actor=actor,
                payload={"query": query.text, "model": model_name, "outcome": "empty_generation"},
            )
            _log.warning("abstained_empty_generation", tenant_id=tenant_id, model=model_name)
            return abstain(tenant_id, display_query, retrieval.max_score, reason="empty_generation", trace_id=trace_id)
        # The model emitted its designed "I don't know" refusal (no source was relevant) — that is
        # an ABSTAIN, not a grounded answer. Route it honestly so refusals aren't counted as grounded.
        if is_refusal_text(answer_text):
            # A refusal on STRONG retrieval is usually about the phrasing, not the evidence: a
            # bare noun phrase does not read as something to act on. Re-frame it as an explicit
            # instruction and try once more against the SAME sources. Measured live at confidence
            # 0.92 -- refused as written, answered with real citations once re-framed.
            #
            # Bounded hard: one extra attempt, only above refusal_retry_min_score (well clear of
            # the abstain gate, so a refusal that AGREES with weak retrieval still stands), and
            # only when the query was not already explicit. The sources never change, so the
            # grounded-or-abstain contract is untouched -- a second refusal abstains as before.
            retry_floor = self._refusal_retry_min_score
            # `> 0` is the OFF switch: with a floor of 0 the score comparison alone would be
            # true for every query, turning a disabled feature into an always-on one.
            reframed = (
                as_explicit_request(query.text) if retry_floor > 0 and retrieval.max_score >= retry_floor else None
            )
            if reframed is not None:
                _log.info(
                    "model_refusal_retry_reframed",
                    tenant_id=tenant_id,
                    model=model_name,
                    confidence=retrieval.max_score,
                )
                prompt = build_prompt(reframed, spans, persona=persona, depth=depth)
                answer_text, usage = await _generate()
            if is_refusal_text(answer_text) or not answer_text.strip():
                _log.info("abstained_model_refusal", tenant_id=tenant_id, model=model_name)
                return abstain(tenant_id, display_query, retrieval.max_score, reason="out_of_scope", trace_id=trace_id)
        await self._audit.record(
            tenant_id=tenant_id,
            kb_id=kb_id,
            trace_id=trace_id,
            action=AuditAction.GENERATE,
            actor=actor,
            payload={
                "query": query.text,
                "model": model_name,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "output_sha256": hashlib.sha256(answer_text.encode()).hexdigest(),
                "citations": [c.chunk_id for c in spans],
                "confidence": retrieval.max_score,
                **usage,
            },
        )

        # --- Output guardrail (FR-G3 toxicity) — block toxic output, surface as abstain ---
        if guard_on and self._output_guard is not None:
            guard = await self._output_guard.check(answer_text)
            if not guard.allowed:
                await self._audit.record(
                    tenant_id=tenant_id,
                    kb_id=kb_id,
                    trace_id=trace_id,
                    action=AuditAction.GUARDRAIL,
                    actor=actor,
                    payload={"stage": "output", "category": guard.category, "reason": guard.reason},
                )
                _log.warning("guardrail_blocked_output", tenant_id=tenant_id, category=guard.category)
                return abstain(
                    tenant_id, display_query, retrieval.max_score, reason="blocked_toxic_output", trace_id=trace_id
                )

        # --- Faithfulness verification (FR-E4) + three-way routing ---
        # A tenant's own model judges its own answers when configured (FR-E1), so a real
        # generative answer is verified semantically, not by the lexical fallback.
        checker = self._faithfulness
        if self._resolve_faithfulness is not None:
            resolved = await self._resolve_faithfulness(tenant_id)
            if resolved is not None:
                checker = resolved
        # Per-KB notation grounding (Slice 4): toggle the notation twin for THIS KB. No-op when
        # the flag is off (returns the same checker), so behaviour is unchanged unless a KB/the
        # platform enabled it.
        if checker is not None and self._apply_notation is not None:
            checker = await self._apply_notation(checker, kb_id or tenant_id)
        if not (faith_on and checker is not None):
            _log.info("generated", tenant_id=tenant_id, faithfulness="disabled")
            answer = ground(tenant_id, display_query, answer_text, spans, retrieval.max_score, trace_id=trace_id)
            return answer.model_copy(update={"model": model_name})

        # FR-C2: parent CONTEXT as extra non-citation spans so parent-only support isn't false-flagged.
        verdict = await checker.check(answer_text, with_parent_context(spans), query.text)

        # VERIFICATION UNAVAILABLE TAKES THE SAME EXIT AS VERIFICATION DISABLED. A `None` score
        # means no judgement was made -- the scorer was unreachable and so was its fallback, or
        # none was configured -- and the one thing that must not happen is a number being invented
        # for it: routing on 1.0 would ship the answer as verified having checked nothing, and 0.0
        # would abstain on non-evidence. The citations are intact either way, which is what
        # grounded-or-abstain is about, so the answer stands with its score left unset exactly as
        # it is when the gate is off. The REASON is logged because `degraded` with no cause cannot
        # be triaged: a one-off read error and an hour of downtime need different responses.
        if verdict.score is None:
            _log.warning(
                "generated_verification_unavailable",
                tenant_id=tenant_id,
                method=verdict.method,
                reason=verdict.degraded_reason,
                detail=verdict.degraded_detail,
            )
            answer = ground(tenant_id, display_query, answer_text, spans, retrieval.max_score, trace_id=trace_id)
            return answer.model_copy(update={"model": model_name})

        # The authoritative "cut off by the token budget" signal (finish_reason == "length"),
        # threaded out of the gateway usage dict — the completeness guard's truncation detection
        # keys on it, and falls back to a structural heuristic when the server omits it. The
        # self-correction below replaces the text via complete() (which carries no usage), so it
        # clears this and the structural fallback takes over for a self-corrected answer.
        raw_finish = usage.get("finish_reason")
        finish_reason: str | None = raw_finish if isinstance(raw_finish, str) else None

        # --- Self-correction (FR-F2 extended) ---------------------------------------------
        # The existing agentic retry fires when RETRIEVAL is too weak. Nothing fired when the
        # retrieval looked fine but the ANSWER failed verification — the loop had no path back,
        # so an unfaithful answer went straight to a human or to refusal. One bounded re-attempt
        # closes it: tell the model which of its own sentences the sources did not support and
        # have it rewrite using only what IS supported. Keep the retry ONLY if it verifies
        # better, so this can never make an answer worse.
        retried = False
        if self._self_correct and verdict.unsupported_claims and verdict.score < faithful_gate and answer_text.strip():
            retry_prompt = build_prompt(query.text, spans, persona=persona, depth=depth) + (
                "\n\nA previous attempt asserted the following, which the SOURCES above do NOT "
                "support:\n"
                + "\n".join(f"- {c}" for c in verdict.unsupported_claims[:5])
                + "\n\nWrite the answer again using ONLY what the SOURCES support. Drop every "
                "unsupported claim entirely rather than rephrasing it. If that leaves nothing to "
                "say, reply exactly: I don't know."
            )
            _log.info(
                "self_correction_attempt",
                tenant_id=tenant_id,
                score=verdict.score,
                claims=len(verdict.unsupported_claims),
            )
            retry_text = (await llm.complete(retry_prompt, max_tokens=max_tokens or 512)).strip()
            if retry_text:
                retry_verdict = await checker.check(retry_text, with_parent_context(spans), query.text)
                _log.info(
                    "self_correction_result",
                    tenant_id=tenant_id,
                    before=verdict.score,
                    after=retry_verdict.score,
                )
                if improves_on(retry_verdict, verdict, tolerance=0.0) and retry_verdict.score != verdict.score:
                    _log.info(
                        "self_correction_improved",
                        tenant_id=tenant_id,
                        before=verdict.score,
                        after=retry_verdict.score,
                    )
                    answer_text, verdict, retried, finish_reason = retry_text, retry_verdict, True, None

        status = route_unless_unjudged(verdict.score, faithful_gate, flag_gate) or AnswerStatus.GROUNDED

        # --- Coverage-aware completeness re-prompt (flag-gated, default OFF) -----------------
        # The self-correction above closes the case where the answer said something the sources
        # do NOT support. This closes the opposite one: the answer is GROUNDED but INCOMPLETE —
        # retrieval surfaced a distinct, clearly-relevant point the model simply left out (seen
        # live: all six points retrieved, four listed). ONE targeted re-prompt over the SAME
        # spans + SAME citation rules asks the model to fold the omitted points back in. It is
        # kept ONLY when it (a) reduces the uncovered-span count AND (b) does not regress the
        # faithfulness gate (still GROUNDED, score no lower) — so grounded-or-abstain is never
        # loosened and an answer can never be made worse. Runs only when a drop is actually
        # detected, so the extra generation is opt-in and bounded to a single call.
        coverage_payload: dict[str, object] = {}
        guard_steps: list[AgentStep] = []
        completeness_signal: Completeness | None = None
        # Strict, adaptive completeness guard (Slice 1) — SUPERSEDES the single coverage re-prompt
        # below when ON. It (a) CONTINUES a truncated completion in bounded, budget-escalating steps,
        # (b) runs a bounded coverage LOOP for still-omitted relevant points, and (c) attaches a
        # non-authoritative completeness signal so a partial answer is either fixed or explicitly
        # marked — never shipped as whole. Every attempt is accepted only if it stays GROUNDED and
        # does not lower the faithfulness score (grounded-or-abstain untouched; the better answer is
        # always kept). Default OFF ⇒ this branch is skipped and behaviour is byte-identical.
        if self._completeness_enabled and status is AnswerStatus.GROUNDED and checker is not None:
            (
                answer_text,
                verdict,
                status,
                completeness_signal,
                coverage_payload,
                guard_steps,
            ) = await self._enforce_completeness(
                llm=llm,
                checker=checker,
                query_text=query.text,
                spans=spans,
                answer_text=answer_text,
                verdict=verdict,
                status=status,
                persona=persona,
                depth=depth,
                faithful_gate=faithful_gate,
                flag_gate=flag_gate,
                base_max_tokens=max_tokens,
                finish_reason=finish_reason,
                tenant_id=tenant_id,
                max_continuations=(
                    int(max_continuations) if max_continuations is not None else self._completeness_max_continuations
                ),
                budget_factor=(
                    continuation_budget_factor
                    if continuation_budget_factor is not None
                    else self._completeness_budget_factor
                ),
                coverage_max_loops=(
                    int(coverage_max_loops) if coverage_max_loops is not None else self._completeness_coverage_max_loops
                ),
                coverage_enabled=completeness_coverage,
                tolerance=(
                    faithfulness_tolerance
                    if faithfulness_tolerance is not None
                    else self._completeness_faithfulness_tolerance
                ),
                budget=guard_budget
                or GuardBudget(
                    int(max_guard_calls) if max_guard_calls is not None else self._completeness_max_guard_calls
                ),
            )
        elif self._coverage_enabled and status is AnswerStatus.GROUNDED and checker is not None:
            uncovered = uncovered_spans(
                answer_text,
                spans,
                min_score=self._coverage_min_score,
                coverage_ratio=self._coverage_ratio,
                max_spans=self._coverage_max_spans,
            )
            if uncovered:
                cov_prompt = build_coverage_reprompt(
                    query.text, answer_text, spans, uncovered, persona=persona, depth=depth
                )
                cov_text = (await llm.complete(cov_prompt, max_tokens=max_tokens or 512)).strip()
                before, after, accepted = len(uncovered), len(uncovered), False
                if cov_text and not is_refusal_text(cov_text):
                    cov_verdict = await checker.check(cov_text, with_parent_context(spans), query.text)
                    cov_status = route_unless_unjudged(cov_verdict.score, faithful_gate, flag_gate)
                    after = len(
                        uncovered_spans(
                            cov_text,
                            spans,
                            min_score=self._coverage_min_score,
                            coverage_ratio=self._coverage_ratio,
                            max_spans=self._coverage_max_spans,
                        )
                    )
                    if after < before and cov_status is AnswerStatus.GROUNDED and improves_on(cov_verdict, verdict):
                        answer_text, verdict, status = cov_text, cov_verdict, cov_status
                        accepted = True
                coverage_payload = {
                    "coverage_reprompt": accepted,
                    "uncovered_before": before,
                    "uncovered_after": after,
                }
                guard_steps.append(
                    AgentStep(
                        step="coverage",
                        detail=(
                            f"Coverage → added {before - after} omitted point(s)"
                            if accepted
                            else f"Coverage → re-prompt kept original ({before} omitted point(s) unresolved)"
                        ),
                    )
                )
                _log.info(
                    "coverage_reprompt",
                    tenant_id=tenant_id,
                    accepted=accepted,
                    uncovered_before=before,
                    uncovered_after=after,
                )

        await self._audit.record(
            tenant_id=tenant_id,
            kb_id=kb_id,
            trace_id=trace_id,
            action=AuditAction.FAITHFULNESS,
            actor=actor,
            payload={
                "query": query.text,
                "score": verdict.score,
                "outcome": status.value,
                "unsupported_claims": verdict.unsupported_claims,
                "faithful_threshold": faithful_gate,
                "flag_threshold": flag_gate,
                "self_corrected": retried,
                **coverage_payload,
            },
        )
        _log.info(
            "faithfulness", tenant_id=tenant_id, score=verdict.score, outcome=status.value, self_corrected=retried
        )

        if status is AnswerStatus.GROUNDED:
            answer = ground(
                tenant_id,
                display_query,
                answer_text,
                spans,
                retrieval.max_score,
                trace_id=trace_id,
                faithfulness_score=verdict.score,
            )
            update: dict[str, object] = {"model": model_name}
            if guard_steps:
                update["agent_trace"] = guard_steps
            if completeness_signal is not None:
                update["completeness"] = completeness_signal
            return answer.model_copy(update=update)
        if status is AnswerStatus.FLAGGED_FOR_REVIEW:
            # review_id is attached by the control-plane pipeline when it opens the HITL review.
            flagged = flag_for_review(
                tenant_id,
                display_query,
                answer_text,
                spans,
                retrieval.max_score,
                faithfulness_score=verdict.score if verdict.score is not None else 0.0,
                unsupported_claims=verdict.unsupported_claims,
                trace_id=trace_id,
            )
            return flagged.model_copy(update={"model": model_name})
        return abstain(
            tenant_id,
            display_query,
            retrieval.max_score,
            reason="faithfulness_below_threshold",
            trace_id=trace_id,
            faithfulness_score=verdict.score,
        )

    async def _meta_complete(self, llm: LLMGateway, prompt: str, max_tokens: int) -> tuple[str, str | None]:
        """One extra generation for the completeness guard, surfacing ``finish_reason`` when the
        gateway reports it (so a continuation can itself be detected as still-truncated). Falls back
        to :meth:`~LLMGateway.complete` (no finish_reason → structural detection) for a plain stub."""
        with_usage = getattr(llm, "complete_with_usage", None)
        if with_usage is not None:
            text, usage = await with_usage(prompt, max_tokens=max_tokens)
            fr = usage.get("finish_reason")
            return text, fr if isinstance(fr, str) else None
        return await llm.complete(prompt, max_tokens=max_tokens), None

    async def _enforce_completeness(  # noqa: PLR0912, PLR0915 - two bounded guarded loops; splitting hides the flow
        # A guard needs its full context threaded in; PLR0913 is globally ignored for such seams.
        self,
        *,
        llm: LLMGateway,
        checker: FaithfulnessChecker,
        query_text: str,
        spans: list[RetrievedSpan],
        answer_text: str,
        verdict: FaithfulnessVerdict,
        status: AnswerStatus,
        persona: str | None,
        depth: str,
        faithful_gate: float,
        flag_gate: float,
        base_max_tokens: int | None,
        finish_reason: str | None,
        tenant_id: str,
        max_continuations: int,
        budget_factor: float,
        coverage_max_loops: int,
        coverage_enabled: bool,
        tolerance: float,
        budget: GuardBudget,
    ) -> tuple[str, FaithfulnessVerdict, AnswerStatus, Completeness, dict[str, object], list[AgentStep]]:
        """Fix a truncated / incomplete GROUNDED answer within bounded, adaptive attempts, or MARK it.

        Two bounded loops, both accept an attempt ONLY if it stays GROUNDED and does not lower the
        faithfulness score by more than ``tolerance`` (never meaningfully regress; keep the better
        answer), so grounded-or-abstain is never loosened and an answer is never made worse:

        1. **Truncation continuation** — while the answer is truncated (``finish_reason == length`` or
           a structural dangling-enumeration cutoff), CONTINUE from where it stopped with an escalating
           token budget; the pieces are stitched (overlap de-duped, no mid-token seam) and re-scored
           over the SAME spans.
        2. **Coverage loop** — while a clearly-relevant retrieved point is still omitted, re-prompt
           targeting the STILL-missing points. SKIPPED when ``coverage_enabled`` is False (the
           doc-summary path, where omission is intentional — a truncation-only guard applies there).

        Every extra LLM call is drawn from the shared per-request ``budget``; when it is exhausted the
        loops stop and the answer is marked incomplete rather than fanning out. Returns the (possibly
        improved) text, verdict, status, the completeness signal, the audit payload, and trace steps.
        """
        steps: list[AgentStep] = []
        base = base_max_tokens if base_max_tokens else self._llm_max_tokens
        exhausted = False

        # --- 1. Bounded, adaptive truncation continuation ---
        continued = 0
        if is_truncated(answer_text, finish_reason):
            bound = continuation_bound(query_text, cap=max_continuations)
            for attempt in range(bound):
                if not budget.take():
                    exhausted = True
                    break  # per-request guard budget spent — stop, do not fan out
                guard_budget_tokens = escalated_budget(base, attempt, factor=budget_factor, cap=self._heavy_max_tokens)
                cont_prompt = build_continuation_prompt(query_text, answer_text, spans, persona=persona, depth=depth)
                cont_text, cont_finish = await self._meta_complete(llm, cont_prompt, guard_budget_tokens)
                cont_text = cont_text.strip()
                if not cont_text or is_refusal_text(cont_text):
                    break  # nothing usable to append — keep what we have
                combined = stitch_continuation(answer_text, cont_text)
                combined_verdict = await checker.check(combined, with_parent_context(spans), query_text)
                combined_status = route_unless_unjudged(combined_verdict.score, faithful_gate, flag_gate)
                # Accept only if the continuation keeps it GROUNDED without a meaningful faithfulness drop.
                if combined_status is not AnswerStatus.GROUNDED or not improves_on(
                    combined_verdict, verdict, tolerance=tolerance
                ):
                    break  # regressed beyond tolerance → keep the shorter-but-clean prior answer, stop
                answer_text, verdict, status, finish_reason = combined, combined_verdict, combined_status, cont_finish
                continued += 1
                if not is_truncated(answer_text, finish_reason):
                    break  # no longer cut off — done
            if continued:
                steps.append(
                    AgentStep(step="completeness", detail=f"Continuation → extended a truncated answer x{continued}")
                )
            _log.info("completeness_continuation", tenant_id=tenant_id, continued=continued, bound=bound)

        # --- 2. Bounded, adaptive coverage loop (skipped for an intentionally-lossy summary) ---
        cov_before = 0
        cov_accepted = 0
        if coverage_enabled:
            cov_before = len(
                uncovered_spans(
                    answer_text,
                    spans,
                    min_score=self._coverage_min_score,
                    coverage_ratio=self._coverage_ratio,
                    max_spans=self._coverage_max_spans,
                )
            )
            for _loop in range(coverage_max_loops):
                uncovered = uncovered_spans(
                    answer_text,
                    spans,
                    min_score=self._coverage_min_score,
                    coverage_ratio=self._coverage_ratio,
                    max_spans=self._coverage_max_spans,
                )
                if not uncovered:
                    break  # every clearly-relevant point is covered
                if not budget.take():
                    exhausted = True
                    break  # per-request guard budget spent — stop
                cov_prompt = build_coverage_reprompt(
                    query_text, answer_text, spans, uncovered, persona=persona, depth=depth
                )
                cov_text, cov_finish = await self._meta_complete(llm, cov_prompt, self._heavy_max_tokens)
                cov_text = cov_text.strip()
                if not cov_text or is_refusal_text(cov_text):
                    break
                cov_verdict = await checker.check(cov_text, with_parent_context(spans), query_text)
                cov_status = route_unless_unjudged(cov_verdict.score, faithful_gate, flag_gate)
                after = len(
                    uncovered_spans(
                        cov_text,
                        spans,
                        min_score=self._coverage_min_score,
                        coverage_ratio=self._coverage_ratio,
                        max_spans=self._coverage_max_spans,
                    )
                )
                # Accept only if it covers MORE without a meaningful faithfulness regression.
                if (
                    after >= len(uncovered)
                    or cov_status is not AnswerStatus.GROUNDED
                    or not improves_on(cov_verdict, verdict, tolerance=tolerance)
                ):
                    break  # no improvement or a regression → keep the better prior answer, stop
                # A coverage rewrite REPLACES the text, so its own finish_reason is authoritative for
                # the new answer — refresh it (a stale "length" from the initial gen must not mark a
                # complete rewrite as truncated).
                answer_text, verdict, status, finish_reason = cov_text, cov_verdict, cov_status, cov_finish
                cov_accepted += 1
            cov_after = len(
                uncovered_spans(
                    answer_text,
                    spans,
                    min_score=self._coverage_min_score,
                    coverage_ratio=self._coverage_ratio,
                    max_spans=self._coverage_max_spans,
                )
            )
            if cov_before:
                steps.append(
                    AgentStep(
                        step="coverage",
                        detail=(
                            f"Coverage loop → added {cov_before - cov_after} omitted point(s) over "
                            f"{cov_accepted} pass(es)"
                            if cov_accepted
                            else f"Coverage loop → {cov_before} omitted point(s) unresolved"
                        ),
                    )
                )
                _log.info(
                    "completeness_coverage_loop",
                    tenant_id=tenant_id,
                    accepted=cov_accepted,
                    uncovered_before=cov_before,
                    uncovered_after=cov_after,
                )

        # --- 3. Non-authoritative completeness signal ---
        completeness = build_completeness(
            answer_text,
            spans,
            min_score=self._coverage_min_score,
            coverage_ratio=self._coverage_ratio,
            max_spans=self._coverage_max_spans,
            truncated=is_truncated(answer_text, finish_reason),
            measure_coverage=coverage_enabled,
            exhausted=exhausted,
        )
        payload: dict[str, object] = {
            "completeness_complete": completeness.complete,
            "completeness_covered": completeness.covered,
            "completeness_expected": completeness.expected,
            "completeness_truncated": completeness.truncated,
            "continuations": continued,
            "coverage_passes": cov_accepted,
            "guard_budget_exhausted": exhausted,
        }
        if not completeness.complete:
            _log.warning(
                "answer_marked_incomplete",
                tenant_id=tenant_id,
                reason=completeness.reason,
                covered=completeness.covered,
                expected=completeness.expected,
                truncated=completeness.truncated,
                exhausted=exhausted,
            )
        return answer_text, verdict, status, completeness, payload, steps
