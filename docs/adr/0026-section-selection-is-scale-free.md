# ADR-0026 — The section pass selects on a relative floor, not the plan's absolute threshold

Status: accepted · Date: 2026-10-22 · Phase: RAG v2 Phase 8

## Context

The upgrade plan's §8 `aggregate` strategy selects sections by cross-encoder score:

> 3. Select sections with score ≥ `section_threshold` (start low, ~0.15 — recall over precision).

`section_threshold` was implemented as `RAG_V2_SECTION_THRESHOLD`, default 0.15, and then
measured against the real document the plan's acceptance criterion names — the AEGCL SOC/NOC
Corrigendum-V of 24-09-2026, 80 pages, 78 sections after the platform's own extraction and
structure parse — scored by this deployment's own BGE cross-encoder (`reranker` container,
TEI `/rerank`).

**It selected nothing. Not at 0.15, and not at 0.05, 0.01 or 0.001.**

The reason is visible in the scores. For the acceptance query *"give me the compliance details
from this document"*, every one of the 78 sections scored between `1.2e-5` and `7.2e-5` — a 6×
spread with no relationship to relevance. The best-scoring section was the letterhead's
neighbour, not the amendment table.

Two separate effects produce that, and both were measured:

**A vague whole-document query is not "about" any passage.** The same reranker, given the
facet-specific probe *"what are the switch certifications?"*, scores the switches section
`0.65` and the canteen policy `1.2e-5`. Given *"give me the compliance details from this
document"*, it scores the switches section `1.7e-5` and the canteen policy `1.2e-5`. The model
is not failing; it is correctly reporting that a request for "the compliance details" does not
pick out one passage. This is why the plan reranks against the sub-queries as well, and it is
why Phase 8 also probes each matched facet directly (see `pipeline._section_probes`).

**A score is absolute only for a fixed candidate SHAPE.** 0.15 is a sensible floor for the
candidate the reranker was calibrated on — a retrieved passage of a few hundred words. A
section CARD is a breadcrumb, a heading and a one-line summary. Measured on the same document:

| candidate shape | probe | best score |
|---|---|---|
| heading only | vague aggregate query | 2.0e-5 |
| heading only | facet probe | 4.6e-4 |
| heading + Phase-4 summary | vague aggregate query | 3.4e-4 |
| heading + Phase-4 summary | facet probe | **0.63** |

Four orders of magnitude between the corners, for the same model and the same document.

What survives the shape change is the **ratio**. With Phase-4 summaries and a facet probe, the
three certification-bearing sections scored `1.0 / 0.994 / 0.413` of the best and every
irrelevant section scored `0.000`. The separation is clean; only the magnitude moved.

## Decision

`select_sections` applies **two floors, by OR**:

- the plan's absolute `threshold` (`RAG_V2_SECTION_THRESHOLD`, default 0.15), unchanged; and
- a scale-free `relative` floor (`RAG_V2_SECTION_RELATIVE`, default 0.1) — a section is also
  selected when it scores within that fraction of the best-scoring section.

`relative=0.0` turns the addition off and restores the plan's behaviour exactly. The floor that
was actually applied is what the selection records, because on a real document the floor applied
and the floor requested differ by three orders of magnitude and a stored selection has to say
which number produced it.

## Consequences

**The plan's knob is kept, not replaced.** An operator who calibrates an absolute threshold
against their own reranker and their own card shape gets exactly what they asked for. This is
the hazard `CrossEncoderReranker._verify_model` already names in its own docstring — *"a
threshold is a claim about one model's score distribution"* — and the same caution ADR-0005
records for the uncalibrated faithfulness thresholds. A fixed number was the wrong instrument
here for the same reason it is a known risk there.

**The degenerate case is accepted deliberately.** When nothing in the document is relevant, the
best score is itself noise and the ratios among noise are flat (measured: 0.68–1.0), so the
relative floor selects broadly. For a *scoped aggregate* that is the honest answer rather than a
defect: the caller named this document and asked for everything in it, the fan-out is bounded by
`max_sections`, and Phase 9's reduce states the coverage to the reader. It is also why the
relative floor is reachable only from a strategy with `section_pass` — a lookup is a question
that genuinely can have no answer, and must keep being able to say so.

**Phase-4 enrichment is a hard prerequisite for the section pass, not an enhancement.** The
table above is the evidence: without summaries the candidate is a bare heading and the score
scale collapses by three orders of magnitude. The pass still RUNS on unenriched cards — the
catalog read joins outer precisely so a partially-enriched document is not silently truncated —
but its selection is much weaker, and an operator enabling `RAG_V2_RETRIEVAL` on a corpus
ingested without `RAG_V2_INGEST` should expect that.

## Alternatives rejected

**Lower the absolute default until it fires.** It would have to go below `1e-5`, which is
indistinguishable from "select everything", and it would be wrong again for the next reranker,
the next card shape, or the same card shape once summaries get longer. A number that has to be
re-derived whenever anything upstream changes is not a threshold.

**Normalise the reranker's scores.** Min-max normalising a result set is what
`fusion_score` already does and what `RetrievedSpan.score`'s own comment forbids mixing with an
absolute score: it would make the gate's input depend on what else happened to be retrieved. The
relative floor reads the ratio without writing it into the score, so the stored number stays the
model's own.

**Select a fixed top-N.** Scale-free, but it cannot express "this document has four relevant
sections and that one has forty", which is exactly the difference an aggregate over a tender
versus over a memo has to carry.
