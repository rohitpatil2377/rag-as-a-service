# Measured on the SOP golden set

Method: 24 questions (20 with a required fact, 4 that must be ABSTAINED on) x 3 runs each,
semantic cache OFF, against the live stack. `scripts/eval_gate.py`.

## top_k — raised 10 -> 16 (ADOPTED)

| top_k | score | answerable | refusals |
|-------|-------|-----------|----------|
| 10    | 91.7% | 90%       | 100%     |
| 16    | 95.8% | 95%       | 100%     |
| 24    | 95.8% | 95%       | 100%     |

16 is where the gain stops. The number that had to hold is the refusal column: more context is
only safe if it does not start manufacturing answers to unanswerable questions, and it did not.
Now `Settings.query_top_k`.

## chunk_max_chars — 1400 REJECTED, stays at the 700 default

| chunk_max_chars | chunks | avg chars | score |
|-----------------|--------|-----------|-------|
| 700 (default)   | 217    | 528       | 95.8% |
| 1400            | 140    | 881       | 87.5% |

-8.3 points. Three questions regressed outright and two became flaky; nothing improved. A
partially-migrated corpus (one document at 1400, the rest at 700) measured 91.7%, in between —
so the effect is real and monotonic, not noise.

This is the change the branch carried and the one we deferred until it could be measured. It
would have degraded this corpus. That deferral was right, and the same caution applies to
re-testing it on Alembic's corpus: a value that helps one document set can hurt another, so it
is a per-corpus measurement, not a platform default to be inherited.

## Still open

* Doc-scoping WORKS -- an earlier note here said it did not, and that was wrong. `scope_for_query`
  resolves "the SOP Writing Guidance Pharma document" at coverage 1.0 / margin 0.8, and the
  citations come from that file. The failure was the QUESTION: "who authorizes an SOP?" has two
  defensible answers inside that one document -- section 3.3 names a designation (Asst. General
  Manager and above) while the procedure section describes the Location Head signing. A golden
  question with two correct answers measures nothing, so it now names the section.

  Worth stating the rule that came out of it: three of the questions in this set were rewritten
  because they were ambiguous, and NONE were rewritten because the platform answered them badly.
  Rewriting a question the platform gets wrong is how a golden set stops being evidence.
* One question ("the objective stated in ...") still fails, because that document is stuck at
  chunk_max_chars=1400 -- it cannot be rebuilt while chandra-ocr is down. It passed at 700, which
  is consistent with the chunk finding above rather than a separate defect.
* Contextual retrieval (metadata -> llm) is not yet measured: it needs a re-index, and
  `chandra-ocr` on poc-aiops was returning 500s while every other model on the same host stayed
  healthy, so two OCR documents could not be rebuilt.
