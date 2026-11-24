# ADR-0005 — Faithfulness verification with three-way routing (FR-E4)

**Status:** Accepted (Phase-1) · 2026-07-02

## Context
Grounded-or-abstain (FR-E2) binds answers to retrieved sources, but a generative model
can still produce a *cited-yet-unsupported* sentence. FR-E4 adds a post-generation
faithfulness check. A binary keep/abstain is too blunt for regulated buyers: a partially
supported answer is often worth a human's eyes, not silent discard.

## Decision
After generation, score the answer against its cited spans and route three ways on two
per-tenant thresholds:

| Condition | Outcome |
|---|---|
| `score ≥ faithful_threshold` | **GROUNDED** (answer stands) |
| `flag_threshold ≤ score < faithful_threshold` | **FLAGGED_FOR_REVIEW** → HITL, not final to user, with `unsupported_claims` + `review_id` |
| `score < flag_threshold` | **ABSTAINED** (`abstain_reason="faithfulness_below_threshold"`) |

- Thresholds are per-tenant config (`faithful_threshold` default 0.7, `flag_threshold`
  0.4), **TODO: calibrate on the golden set**.
- Two checker backends behind `FaithfulnessChecker` (rag-core Protocol):
  `LexicalFaithfulnessChecker` (deterministic, zero-model, CI default) and
  `LLMFaithfulnessChecker` (LLM-as-judge, structured JSON). Same factory pattern as the
  gateway/reranker.
- The verdict (score + `unsupported_claims`) is audited on **every** path via the
  distinct, independently-queryable `AuditAction.FAITHFULNESS` event.
- **FLAGGED reuses the existing HITL flow** (`control-plane/hitl.py`): a PENDING review
  with `reason="faithfulness_warning"` and the unsupported claims in `warnings`. No new
  HITL status, no parallel path.
- Citations remain **structural** (built from retrieved spans), never scraped from model
  text. The grounding prompt now instructs inline `[n]` markers, but they are a display
  nicety — correctness never depends on them.

## Known limitations (revisit, do not over-invest now)
- **Weak judge on a small model.** LLM-as-judge on the current 1.5B/3B CPU dev model is a
  weak faithfulness signal. The judge should be a **distinct, stronger** model than the
  generator once real vLLM/GPU lands (see ADR-0006). CI uses the lexical checker, which is
  a coarse lexical-overlap proxy, not semantic entailment.
- **Per-tenant ALTER caveat.** The new `reason`/`warnings` columns are added to
  `t_<slug>.hitl_reviews` via `CREATE TABLE IF NOT EXISTS`, so they apply to *fresh*
  tenants only. Existing per-tenant tables in a live DB need an `ALTER TABLE`; a real
  per-tenant migration mechanism is Phase-1 catalog work, not part of E4.

## Consequences
- New `AnswerStatus.FLAGGED_FOR_REVIEW`; `GroundedAnswer` gains `faithfulness_score`,
  `unsupported_claims`, `review_id`; invariant relaxed so GROUNDED **or** FLAGGED require
  ≥1 citation, ABSTAINED forbids citations.
- HITL persistence split behind a `HitlStore` Protocol (Postgres + in-memory twins) so the
  three-outcome e2e runs in the zero-infra profile.
