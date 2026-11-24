# ADR-0010 — Guardrails: prompt-injection (G1) + toxicity (G3)

**Status:** Accepted (Phase-1) · 2026-07-02

## Context
A sovereign-government RAG platform must defend against **prompt injection** (users trying
to override instructions / exfiltrate the system prompt) and must not emit **toxic** output
(FR-G1/G3). These are trust/safety controls, not a feature toggle to forget in prod.

## Decision
- A `Guardrail` Protocol (rag-core) + **deterministic heuristic defaults** with **no new
  dependency**: `HeuristicInjectionGuard` (pattern-based) on the **input** query,
  `HeuristicToxicityGuard` (wordlist) on the **generated output**. `LLMGuard` is an opt-in
  (gateway classification); NeMo Guardrails / LlamaGuard / Detoxify is the production upgrade.
- **Hooks:** the input guard runs in `pipeline.answer_query` **before the router / retrieval
  / any model call** — a block never reaches the LLM. The output guard runs in
  `GenerationService.answer` after generation — toxic output is dropped.
- **A block reuses `ABSTAINED`** with `abstain_reason` `blocked_prompt_injection` /
  `blocked_toxic_output` (no new `AnswerStatus`; a block is a refusal-to-answer, and the
  console already renders abstain + reason). Each block emits a distinct, independently
  queryable **`AuditAction.GUARDRAIL`** event (`stage`, `category`, `reason`).
- **Default ON** (`guardrails_enabled=True`, `guardrails_backend=heuristic`): safety-first,
  and deterministic so it does not break CI (normal queries don't trip it).

## Known limitation
The heuristic guards are a **floor, not a ceiling** — regex/wordlist detection is weak and
evadable. Production must swap in a real classifier (LlamaGuard/NeMo/Detoxify) via the `llm`
backend. Do NOT tune the wordlists to look thorough; the real model is the answer. Retrieved
context is not yet scanned for indirect injection — a follow-up.

## Consequences
- New `AuditAction.GUARDRAIL`; new `get_input_guard`/`get_output_guard` factories +
  `wiring.input_guard` and an `output_guard` on `GenerationService`. Retrieval/grounding/
  faithfulness logic untouched; grounded-or-abstain invariant unchanged.
