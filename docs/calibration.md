# Calibration — faithfulness thresholds (FR-E4) + router triggers (FR-D5)

**Harness:** `scripts/calibrate_faithfulness.py` (reusable) · **Run:** 2026-07-02 (CPU, in-memory, lexical checker)

## What we ran
The harness ingests the golden set (`sample-data/golden/`) and, for each question, records
the heuristic router path and the faithfulness score of the retrieved evidence.

## Honest finding — do NOT tune on this
```
route      faith  expect_abstain  question
standard   1.000          False   What is the platform's uptime target?
standard   1.000          False   What is the P95 query latency target?
standard   1.000          False   How does the platform combine search methods?
standard   1.000           True   What is the boiling point of helium on Jupiter?
observed faithful-score range: [1.000, 1.000]  n=4
```

Two reasons this set **cannot** calibrate the thresholds:
1. **No borderline samples.** Under the deterministic CPU/lexical CI setup, generated
   answers are extractive (the answer *is* the source), so faithfulness clusters at 1.0.
   The FLAGGED band (`flag_threshold ≤ score < faithful_threshold`) is never populated —
   there is nothing between "fully supported" and "fabricated" to fit a threshold to.
2. **All-simple queries route `standard`.** The four golden questions are short and
   single-part, so the D5 heuristic (correctly) routes them all standard. Heavy-path
   triggers are exercised instead by the D5 unit tests
   (`libs/rag-core/tests/test_routing.py`: comparative / multi-question → heavy).

## Decision
- **Keep the defaults `faithful_threshold=0.7`, `flag_threshold=0.4`.** They are a
  conservative FLAGGED band that sits inside the observed separating gap. We deliberately
  do **not** replace them with numbers fabricated from inadequate, non-borderline data —
  that would be tuning to mask the weak CPU judge, which the project explicitly forbids.
- **Router triggers unchanged** — validated by the D5 unit tests; the golden set is all
  simple by construction so provides no counter-signal.

## When to actually calibrate (deferred to the vLLM/GPU step)
Re-run the harness once (a) a real generative model (vLLM) produces answers that can
partially diverge from sources, and (b) the golden set is expanded with **labelled
borderline cases** (partially-supported answers). Then the FLAGGED band can be fit to
maximise agreement with the labels, and the LLM-judge thresholds (a different score
distribution from the lexical checker) can be set from data. The harness is built for
exactly that run — only the inputs (model + labelled data) are missing today.

---

# Calibration — abstain threshold (FR-E2)

**Harness:** `scripts/calibrate_abstain.py` · **Run:** 2026-07-30 against the live stack ·
**Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (CPU tier) · **KB:** `acme__hr_policies`
(9 documents, 662 chunks) · **Questions:** two plain-text lists, one answerable and one unanswerable, one question per line (the corpus-specific pair used for this run is not in the repo — supply your own and pass them to the harness)

Unlike the faithfulness run above, this one **does** have separable data — because for the
first time the score being measured is real. Before ADR-0015 the gate read a min-max
normalised rank-fusion value (1.0 for the best candidate of every query, by construction) or,
on GraphRAG-enabled KBs, a hardcoded 0.5. `0.35` was originally set against a lexical overlap
fraction, which is a different scale again.

## Observed

| | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| **Answerable** (n=10) | 0.4059 | 0.7242 | 0.9863 | 0.9962 | 0.9994 |
| **Unanswerable** (n=10) | 0.0000 | 0.0000 | 0.0001 | 0.0007 | 0.0575 |

Worst answerable 0.4059 vs best unanswerable 0.0575 — **gap +0.3484, no overlap.** Strongly
bimodal: relevant passages pile near 1.0, irrelevant ones collapse to ~1e-4. Every threshold
from **0.10 to 0.40** scores 10/10 answerable kept and 10/10 unanswerable blocked. At 0.05 one
unanswerable slips through; at 0.50 one answerable is lost.

## Reading it

The current `0.35` is inside the safe band, but only by 0.05 — it sits just under the weakest
answerable question rather than in the middle of the gap. A floor nearer **0.15** is ~7×
above the best unanswerable and ~2.7× below the weakest answerable, which is the widest
margin on both sides this sample supports.

The three weakest answerable retrievals are worth reading, because two of them are retrieval
problems rather than scoring problems and a floor cannot fix either:

- `0.4059` "What behaviour does the code of conduct prohibit?" — top span comes from the **HR
  handbook**, not the Code of Conduct document that is in the corpus.
- `0.7360` "What is the annual leave entitlement for employees?" — top span is a
  *Professional Development Reimbursement* table row.

Both are cases where the reranker scored the passage it was given honestly; the passage was
simply the wrong one. That is the chunking/representation wave's problem, not the gate's.

## Caveats — do not treat this as settled

1. **20 questions, one KB.** Enough to show the populations separate; not enough to fit a
   number precisely.
2. **MiniLM-L-6 is the CPU tier, not necessarily the shipping model.** `bge-reranker-v2-m3`
   on GPU will produce a different distribution — likely the same shape, both being
   sigmoid-squashed relevance, but not identical. The threshold is scoped to a named model
   (`RAG_RERANK_MODEL`) and the adapter refuses a mismatch precisely so this cannot be
   forgotten. **Re-run this harness after any reranker change.**
3. **Measured on plain hybrid retrieval.** The adaptive router, query transformation, agentic
   retry, graph augmentation and recency all add spans the gate also sees. Re-check per-KB
   with the features that KB actually has enabled.

## Decision

**Set to `0.15`** (2026-07-30). Chosen for MARGIN, not for the score: 0.35 also scored 10/10
and 10/10, but sat only 0.05 under the weakest real question in the sample, so one slightly
harder question would have fallen off. 0.15 is ~7x above the best unanswerable and ~2.7x below
the weakest answerable — the widest separation this sample supports.

### Note on the in-memory test profile

The in-memory profile uses `LexicalReranker`, whose score is a term-overlap fraction on a
different scale. Measured: an unrelated question scores ~0.17 there purely from function words
("Who won the 1998 world cup?" against an SLA document = 0.1667), while a genuine match scores
0.80. The cross-encoder default therefore falls *inside* that scorer's noise band.

Tests asserting "an unanswerable question abstains" now request the `lexical_gate` fixture
(`conftest.py`), which pins the gate to that scorer's scale rather than inheriting a number
calibrated for a different model. This is the same hazard as P50 and P55 — a value whose
meaning depends on which component is running — and it is why 0.35 was the right number before
the cross-encoder existed.



## Unified calibration harness (`make calibrate`)

`scripts/calibrate.py` sweeps BOTH gates in one run against a labelled golden set and prints a
before/after report — where each gate's two score populations sit, what the current threshold
costs, and what the swept optimum costs. It writes nothing to config: evidence in, a human sets
the number.

```bash
# Stack up (stores + the REAL cross-encoder). Abstain + platform-faithfulness:
make calibrate KB=acme__hr_policies

# With a stronger OpenAI-compatible faithfulness judge (the honest E4 calibration):
make calibrate KB=acme__hr_policies \
  ARGS="--judge-url http://judge:8000/v1 --judge-model Qwen2.5-32B-Instruct --judge-key $JUDGE_KEY --report calibration_report.md"
```

- **Abstain gate** — answerable golden items should pass, `expect_abstain` items should be
  stopped; the harness reads each question's `max_score` from `/v1/kb/{kb}/retrieve` and sweeps
  the threshold, reporting the current value (default 0.15) vs the Youden-optimal split.
- **Faithfulness gate** — supported answers are the positive population. Without a labelled
  UNFAITHFUL set there is no negative population, so the harness reports the supported
  distribution and confirms the current threshold sits below it, deferring a real optimum to a
  borderline set (do NOT fabricate a "tuned" value from inadequate data). Provide a stronger
  judge with `--judge-url` to re-score answers against their cited spans — that is the
  distribution worth calibrating against, since the lexical CI checker clusters near 1.0.

The sweep math is `rag_core.calibration` (pure, unit-tested); the script is only the HTTP
glue, so a threshold is always a claim about the number the gate actually reads. Re-run whenever
the reranker or the judge changes — a threshold is model-specific (ADR-0015).

# Calibration — abstain threshold for bge-reranker-v2-m3 (2026-09-11)

## Why this was re-run
The deployment's primary reranker was `jina-reranker-v3.5` on **api.jina.ai** — a foreign hosted
API that received every question plus up to 50 candidate passages of the tenant's documents (256
calls counted in one API container's log). Its downloadable weights are CC-BY-NC, so it could not
be self-hosted commercially either. It was replaced by the sovereign `bge-reranker-v2-m3` already
serving on the AIOps platform. A threshold is a claim about ONE model's score scale (ADR-0015), so the
Jina-era `0.50` had to be re-derived rather than kept.

## Method
- 40 labelled questions on `acme__test`: **25 answerable** (the 21 verified questions from
  `docs/TEST_QUESTION_SET.md` plus 4 Hindi questions over the English corpus) and **15
  unanswerable** (13 English, 2 Hindi).
- Every unanswerable question was checked against the KB's stored chunks: its distinctive terms
  appear in **none** of them. Six first-draft "unanswerable" questions (gym, adoption, blogging,
  overtime, company car, dental) were dropped because the KB does mention them.
- Scores were taken from the platform's own retrieval service in an **isolated process** with
  the reranker overridden by environment for that process only — the same retriever, stores and
  adapter the gate reads, with no traffic to the foreign API and no change to the live service.
- Sweep: `rag_core.calibration` (Youden's J over a 0.01 grid).

## Observed
| | min | median | max |
|---|---|---|---|
| answerable (25) | 0.253 | 0.916 | 0.9999 |
| unanswerable (15) | 0.0003 | 0.022 | 0.844 |

| threshold | answered (of 25) | refused (of 15) |
|---|---|---|
| 0.15 | 25 | 12 |
| 0.50 (Jina-era value) | **21 — 4 answerable wrongly refused** | 14 |
| 0.21–0.25 (Youden-optimal interval) | 25 | 14 |

The one unanswerable question above the interval — "What is Acme's policy on cryptocurrency trading
by employees?" at 0.844 — matched a Code of Conduct clause on conflicts of interest and personal
business opportunities. It is adjacent rather than absurd, and it is refused at the ANSWER stage
(`out_of_scope`), verified live after the switch.

## Decision
`RAG_ABSTAIN_THRESHOLD=0.23` — the midpoint of the optimal interval, leaving equal margin to the
highest unanswerable score below it (0.206, "stock options") and the lowest answerable score above
it (0.253, "which documents describe a reward scheme"). No fallback reranker: the only alternative
was the foreign API, and a second model on a different scale would make the gate read the wrong
threshold during a fallback window.

## Caveats — do not treat this as settled
1. **40 questions, one KB, and a thin margin.** 0.206 vs 0.253 is a 0.047 gap. Re-run on each new
   corpus; the answer-stage refusal checks remain the second line of defence.
2. **Ranking changed, not only the scale.** Live, the compound question "who approves a business
   travel expense claim, and who approves a facility staff child education reimbursement?" now
   answers only the travel half: BGE's top five did not include the child-education approval
   passage that Jina's did. Compare rerankers on recall, not only on the gate (see P2 in the
   reranker review).
