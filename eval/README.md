# Golden sets and baselines

`golden_sop.json` is the SOP corpus's golden set: 20 questions with a fact the answer must
contain, and 4 that are UNANSWERABLE from the corpus and must be abstained on. The refusal half
is not padding — a gate that only scores answerable questions rewards a model that answers
everything, which is the one behaviour this platform exists to prevent.

`baseline_sop.json` is the current reference report. Compare against it:

    scripts/eval_gate.py --kb acme__research --repeat 3 --compare eval/baseline_sop.json

Two things to know before trusting a number:

* **Turn the semantic cache off** (`RAG_SEMANTIC_CACHE_ENABLED=false`). It is invalidated on
  ingest but NOT on a configuration change, so with it on the repeats replay the first answer and
  an A/B of `top_k` or the chunker reports no difference.
* **`--repeat 1` is not evidence.** The generation model is non-deterministic; a question that
  passes 2 times in 3 looks identical to one that always passes if you only run it once.

## RAG v2 gold suites (`gold/`, plan §11)

`gold/*.yaml` is the Phase-11 format: per query an `intent`, and either an `expected_answer`
(lookup) or a `checklist` of `{id, facet, must_contain, tier}` items (aggregate), plus optional
`supersessions: [{field, old, new}]`. Run it through the same live-pipeline runner:

    scripts/eval_gate.py --kb <kb> --suite rag_v2 --repeat 3
    scripts/eval_gate.py --kb <kb> --suite eval/gold/sop.yaml --floor min_abstain_correctness=1.0

`--suite rag_v2` runs every file in this directory, which is the plan's `eval run --suite rag_v2`.
Both caveats above still apply — turn the semantic cache off, and `--repeat 1` is not evidence.

**What the three new assertions are for.** `checklist` scores an aggregate answer on what it
COVERS, tier by tier, because "give me the compliance details" has no single expected answer and a
containment check against one string measures nothing. `supersessions` is the only NEGATIVE
assertion in the format and the only one that catches an answer which is grounded, cited, faithful
and quoting the figure that was replaced. `intent` labels a query for the router's own accuracy
metric, which has no producer until Phase 7.

**No floor is set by default, deliberately.** The plan's §0 targets are owner-confirmable numbers
(the latency ones are still open — discovery §9.2), and a floor invented here would either gate
nothing or fail every build until somebody lowers it, which teaches a team to edit the threshold
instead of the code. Pass the floors you have agreed.

**A metric a suite does not assert is reported as `n/a`, never as a perfect score,** and can never
fail a floor. `sop.yaml` has no supersession items because that document is not an amendment, so
its `supersession_correctness` is null rather than 1.0.

**Corpus diversity is one document class, not four.** The plan asks for this SOP plus the AEGCL
corrigendum, an HR policy handbook and a technical paper. The other three are an open ask — see
discovery §9.3.
