# Phase 8 acceptance — AEGCL SOC/NOC Corrigendum-V

The plan's §8 acceptance criterion:

> On the AEGCL query *"give me the compliance details from this document"*, selected sections
> include Corrigendum Sl.1–9 and DRS sections for video wall, workstation, switches, firewalls,
> routers, data diode, rack, desktop, NG-SIEM, DAM, NMS/ITSM, HCI-1/2. Log selected sections to
> `query_log`.

This file records what was measured against the real document, including the parts that failed.

## The document

`Corrigendum-5.pdf`, 80 pages, received 2026-10-22. Confirmed to BE Corrigendum-V: the header
reads *"Corrigendum-V"*, dated 24-09-2026, against NIT AEGCL/CGM/SLDC/T-161/Pt-23/26-27. Sl.1–9
are present, as is Attachment-1 (the amended Appendix D Data Requirement Sheet) carrying all 23
DRS sections.

The plan's Appendix A gold fixture was spot-checked against it and is accurate: MAAT
₹35 Cr → ₹19.44 Cr (Sl.5), SAN ≥ 75 TB usable, 6000 sustained EPS, data diode IEC 62443-4-2,
desktop console EPEAT Gold, ITSM PeopleCert ≥ 7 ITIL v4 practices, ≥ 3000 OOTB SIEM rules.

The platform's own extraction and structure parse find **78 sections**, including every DRS
section the acceptance names, by name.

## Measurement 1 — the plan's absolute threshold selects nothing

Scored by this deployment's own BGE cross-encoder (`reranker` container, TEI `/rerank`), over
the 78 real sections, for the acceptance query:

| threshold | selected / 78 | acceptance |
|---|---|---|
| 0.15 (the plan's default) | 0 | 0/14 |
| 0.05 | 0 | 0/14 |
| 0.01 | 0 | 0/14 |
| 0.001 | 0 | 0/14 |

Every section scored between `1.2e-5` and `7.2e-5` — a 6× spread with no relationship to
relevance. **The acceptance fails outright on the plan as written.**

## Measurement 2 — why

Three causes, separated by direct probes of the same reranker.

**A vague whole-document query is not "about" any passage.** The same model, same corpus:

| probe | switches section | canteen policy |
|---|---|---|
| "give me the compliance details from this document" | 1.7e-5 | 1.2e-5 |
| "what are the switch certifications?" | **0.65** | 1.2e-5 |

The model is not failing. It is correctly reporting that a request for "the compliance details"
does not pick out one passage. This is why the plan reranks against sub-queries too.

**The plan's sub-query shape made it worse, not better.** `heuristic_decision` built each
sub-query as `"<query> <facet words>"`. Measured over enriched section cards:

| sub-query shape | best score | separation (relevant vs noise) |
|---|---|---|
| `"<query> certifications"` (the shipped shape) | 4.8e-4 | 4.6× |
| `"What are the certifications requirements?"` | 0.356 | ∞ (noise scored 0.000) |
| `"certifications"` (the facet's own words) | **0.631** | ∞ |

Appending keeps every vague word and adds one. **Fixed**: `heuristic_decision` now emits
`"What are the <facet> requirements?"`, and the section pass additionally probes each facet's
own words directly (`pipeline._section_probes`) — the facet's own words keep the third relevant
section at 0.413 of the best where the question form drops it to 0.047, which is the difference
between selecting it and not.

**A score is absolute only for a fixed candidate SHAPE.** 0.15 is calibrated for a retrieved
passage. A section card is a breadcrumb, a heading and a one-line summary:

| candidate shape | probe | best score |
|---|---|---|
| heading only | vague aggregate query | 2.0e-5 |
| heading only | facet probe | 4.6e-4 |
| heading + Phase-4 summary | vague aggregate query | 3.4e-4 |
| heading + Phase-4 summary | facet probe | **0.63** |

Four orders of magnitude between the corners. What survives is the **ratio** — with summaries
and a facet probe the three relevant sections scored `1.0 / 0.994 / 0.413` of the best and every
irrelevant one scored `0.000`. See **ADR-0026** for the decision this forced.

## Measurement 3 — the end-to-end run

Phase 3 and Phase 4 run against **gemma4-31b** (`models-uat.example.internal`, the model this box's
own API container is configured for), then the shipped Phase-7 router, `pipeline._section_probes`
and `select_sections` with the real BGE cross-encoder. 39 of 72 sections enriched, 0 failures.

| condition | selected | acceptance |
|---|---|---|
| the plan as written (absolute threshold 0.15) | 0 / 78 | **0 / 14** |
| + relative floor (ADR-0026) | 8 / 78 | **6 / 14** |
| + whole-taxonomy probes (`StrategyPlan.exhaustive`) | 39 / 78 | **11 / 14** |
| + the Phase-2 heading fix (`_nameable`) | 38 / 72 | **13 / 14** |

Every step came from a defect found by the measurement, not from tuning a number — the shipped
defaults (`threshold=0.15`, `relative=0.1`, `max_sections=40`) are the ones the table was
produced with, and none was fitted to this document.

**Why the whole-taxonomy rule matters, measured twice.** `heuristic_decision` matches facets by
word overlap with the query. On the first gemma card it matched exactly one of the document's
eight facets for the acceptance query; on the second (gemma is not deterministic, and it named
the facets differently) it matched **none**. Both runs reached the same place once the
exhaustive rule applied, because an aggregate deliberately names no facet — the rule is doing
the work, not a lucky match.

**The one remaining miss is upstream and identified.** `16. NMS and ITSM` holds 16 characters:
its content sits under the following heading, `16.1 HCI-2 Application Scope Clarification`, so
no probe can name it. Same class as the Video Wall failure that the Phase-2 heading fix solved,
but the cause is different — see the follow-ups.

### What is established, and what is not

- **Established**: the selector, the catalog read, the logging, the cap and the fail-safes all
  behave as specified against a real 80-page tender; 13 of the 14 section groups the plan's
  acceptance names are selected, from 38 of 72 sections.
- **NOT established**: that the ANSWER built from those sections is correct. Phase 8 selects and
  logs; nothing reads `selected_sections` yet. That is Phase 9.
- **Caveat worth keeping**: gemma4-31b is not deterministic, and the two runs produced different
  facet vocabularies for the same document. The acceptance was stable across both, but a single
  run of this measurement is one sample, not a guarantee.

## Follow-ups this measurement opened

1. **Table column headers are still promoted to section headings** on the born-digital
   (`layout-json`) path. After the `_nameable` fix removed the single-glyph case, `"Offered by"`
   and `"the bidder."` — the two halves of a DRS table's "Offered by the bidder." column header —
   still become sections, and they still absorb body text from the real heading above them.
   Catching these needs table awareness in that path, which `_nameable` deliberately does not
   attempt: it is a length rule, and these are ordinary-looking words. **This is the direct cause
   of the one remaining acceptance miss.** Phase-2 work, recorded here with evidence.
2. **`LLMDocProfiler` does not retry a gateway error** — only a parse failure. That is right for
   the Phase-7 router, which is on the latency path; the profiler is on the INGEST path, where
   one transient timeout costs the document its card, and with it the facet vocabulary that
   Phase 7 and Phase 8 both depend on. It cost this measurement its first two attempts.
3. **The dev tier cannot run this measurement.** Qwen2.5-3B-Q4 on CPU managed ~4.7 tokens/s of
   prompt throughput under concurrency and failed the profile call on timeout twice; gemma4-31b
   answered in 0.23s. Any re-run of this file should use a hosted model, with
   `RAG_LLM_REASONING_EFFORT=off` — gemma emits a planning preamble as ordinary content
   otherwise, and the profile parse fails with `bare_reasoning_dump_without_answer`.
