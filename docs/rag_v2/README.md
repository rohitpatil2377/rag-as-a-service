# RAG v2 — discovery and phase notes

This directory is the working record behind the `RAG_V2_*` pipeline: what was measured, what was
built, what was tried and abandoned, and which acceptances were never scored. It is kept because
the *reasoning* is the valuable part — several phases here record a measured null result or an
honest "not verified", and that is worth more than a clean-looking summary.

Read [`discovery.md`](discovery.md) first; the `_frag_*` files are the per-area deep dives and
[`goal_scorecard.md`](goal_scorecard.md) is the running assessment against the plan's goals.

## A caveat about the corpora

These notes were written against three real document corpora — an internal SOP set, an HR policy
set, and a public-tender RFP set — none of which ship with this repository, along with the golden
sets and measurement snapshots derived from them. So:

- References to `sample-data/golden/corpus_golden.json`, `measured_*_reindex.json`,
  `sample-data/calibration/*`, `scripts/run_corpus_golden.py` and `scripts/resolve_golden_quote.py`
  are **historical**. Those files are not in the repo.
- Specific numbers (facet coverage 0.54, 34/37 sections, recall@k figures) were measured against
  that corpus on that deployment. They are evidence for the *shape* of a problem, not benchmarks
  you can reproduce here.
- `eval/gold/*.yaml` and `eval/golden_sop.json` **are** present, and `scripts/eval_gate.py
  --suite rag_v2` runs against them. That is the live evaluation path.

Anything describing an unverified acceptance is still unverified; nothing here was quietly
upgraded to "done" when the corpus was removed.
