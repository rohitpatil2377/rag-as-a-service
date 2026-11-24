# Phase 9 acceptance — AEGCL SOC/NOC Corrigendum-V

The plan's §9 acceptance criterion:

> AEGCL compliance query produces an answer covering ≥ 85% of Appendix A items with correct
> values and correct old→new for MAAT.

**Result: 80-85% across three runs (mean 83%), with MAAT correct every time.** The bar is
straddled, not cleared: two runs of three reached it and one did not. Every miss is attributed
below, and the four that recur in all three runs are the real ones.

A SINGLE RUN WOULD HAVE READ 85% AND "ACCEPTANCE MET". It is recorded as a range because this
model is not deterministic and the first run landed exactly on the threshold — the failure mode
`scripts/eval_gate.py` was built to prevent ("a single live run is not evidence"), reached from
the other direction.

## How it was run

Real document (`Corrigendum-5.pdf`, 80 pages, 72 sections after the Phase-2 fix), real
extraction, real Phase-3 card and Phase-4 enrichment from **gemma4-31b**, real BGE
cross-encoder for Phase 8's selection, and the shipped Phase-9 code — `LLMSectionMapper`,
`build_reduce_prompt`, `assemble_answer`. Scoring is against the plan's own Appendix A table,
all 40 items.

Shipped defaults throughout: `map_section_chars=12000`, `map_max_windows=3`,
`map_max_tokens=2400`, `reduce_max_tokens=4000`, `section_threshold=0.15`,
`section_relative=0.1`, `max_sections=40`.

## What each fix was worth

Four rounds, each from a defect the measurement exposed rather than a number that was tuned:

| | map result | acceptance |
|---|---|---|
| as first built | 20/38 mapped, **18 failed** | **30%** |
| + truncated-array salvage, `map_max_tokens` 1200→2400 | **38/38, 0 failed**, 322 items | **40%** |
| + reduce per facet, assembled by code | 38/38, 316 items | **55%** |
| + window long sections instead of truncating | 38/38, **523 items**, 35 contributed | **82%** |
| + context-budget batching of the reduce | 38/38, 515-540 items | **80-85%** (3 runs) |

The last row is the Phase-12 budget work, and it was the same defect one layer further out:
`soc_noc_infrastructure` holds ~280 of the ~520 items and was being crammed into ONE reduce call
whose output budget could not render it. Batching it across two calls (9-10 calls in total,
where the single-facet reduce made 8) recovered the items that were falling off the end.

**The array salvage.** 18 of 38 map calls returned valid JSON cut off mid-object at the token
budget. `extract_json_array` discarded all of it — including the dozen complete items before
the cut. It now closes the array after the last element whose braces balanced. Conservative: a
half-written object is dropped whole, and a reply with nothing complete still returns None.

**The per-facet reduce.** The plan's single reduce over 322 items produced a 107,000-character
prompt, and the answer it wrote was *cut off mid-word with the coverage statement — the last
line the plan's own rules ask for — never reached*. Half the requirements were absent from an
answer that read complete: the exact failure map-reduce exists to prevent, reintroduced at the
reduce step. Per facet each prompt is bounded, the calls run in parallel, and the frame (what
the document is, the coverage line) is now **printed by the assembler rather than requested
from the model** — those are facts the caller already holds.

**Windowing.** This one reverses a decision I made deliberately and documented as a trade:
truncate rather than window, because this runs on the latency path. The measurement showed it
was the single largest cause of a missed requirement. A tender section puts its specification
table first and its certification and bill-of-material clauses **last**, so every `UL 62368`,
`CSA 22.2`, `FCC Part 15`, `HBOM` and `SBOM` in the document fell off the end of a truncated
prompt and appeared **zero** times in the answer.

## The three runs

| run | items | contributed | reduce calls | covered | "must" tier |
|---|---|---|---|---|---|
| 1 | 515 | 34/38 | 9 | 34/40 (85%) | 17/20 |
| 2 | 540 | 35/38 | 10 | 32/40 (80%) | 17/20 |
| 3 | 527 | 35/38 | 9 | 34/40 (85%) | 18/20 |

The MAP pass is stable — 38 of 38 sections mapped, 0 failures, 3 still past the window bound, in
every run. The variance is downstream of it, in which items the reduce chooses to render.

**Missed in all three runs: C08, C23, C29, C36.** Those are the real gaps and they are the four
analysed below.

**Missed in some runs only: C13, C16, C19, C27.** These are the difference between an 80% run and
an 85% one, and they are not a property of the pipeline — the same code, over the same items,
rendered them in two runs of three. Tuning anything against them would be fitting to noise.

## The misses, attributed

| cause | items | note |
|---|---|---|
| section still longer than 3 windows | C29, C36 (+C27 sometimes) | `16.1 HCI-2…` is 85,744 chars; `1. NG-SIEM…` is 69,689 |
| the model read the text and did not extract it | C23 (+C16, C19 sometimes) | their sections are fully covered — 10,950 / 26,643 / 26,782 chars |
| Phase-2: the body sits under a page-header heading | C08 | the Sl.7–9 clauses, including the D.3 withdrawal, are under a section named `CIN: U40101AS2003SGC007238` |

The first group is the operator's knob (`RAG_V2_MAP_MAX_WINDOWS`) and is now **stated in
the answer** rather than silent — see below. The third is the same defect class as the Video
Wall failure the Phase-2 heading fix solved, and is the second instance of it: a page header
promoted to a section heading, so the real content below it is named by something meaningless.

**The second group is Phase 9's own, and is the honest limit of this result.** Those three
sections were selected, fully read, and the model extracted their specification rows instead of
their certification clauses. No amount of windowing fixes that; it is a map-prompt and
model-capability question.

## A defect the measurement found in the measurement

`sections_truncated` was computed by comparing the summed length of the windows to the section.
`section_windows` splits on the block separator and drops it, so the windows always sum to
slightly less than the section **even when they cover all of it** — every multi-window section
was reported as truncated. On this document that read **12** when the true figure is **3**. It
now asks the splitter for one more window than the bound and checks whether it wanted it.

That number is the one an operator is told to watch before widening the bound, and it was wrong
by a factor of four. It also misled this measurement for one round.

## What is established, and what is not

- **Established**: the map pass reads every selected section (38/38, 0 failures, in all three
  runs); every item carries a verbatim quote checked against the text it came from; citations
  resolve to real chunk ids; the cache serves the second identical question from one reduce
  call; MAAT's ₹35 Cr → ₹19.44 Cr is correct in every run.
- **NOT established**: that the 85% bar is reliably met. Two runs of three reached it. A result
  that depends on which run you look at has not been demonstrated, and the honest figure is the
  range.
- **The scorer is a keyword check, not a judgement.** It can credit an answer that names the
  right terms in the wrong relation and miss a correct paraphrase. It is a floor.
- **A taxonomy mismatch caps this measurement.** Appendix A groups by compliance *dimension*
  (certifications, supply_chain, lifecycle); the card gemma derived groups by *subsystem*
  (`soc_noc_infrastructure` alone absorbed 278 of 523 items). That is a Phase-3 property and it
  limits how well any answer built on this card can line up with that gold set.
- **Three runs is three samples.** Enough to show the bar is straddled; not enough to put a
  confidence interval on it. gemma is not deterministic in the map step, in the reduce step, and
  in the Phase-3 card — the two cards it produced for this document named different facets.

## Re-measured after the Phase-2 heading fix and the exhaustive map prompt

Two of the three named follow-ups were done. The result is a MIXED one and is recorded as such.

| | items | contributed | drop rate | covered (3 runs) | "must" |
|---|---|---|---|---|---|
| before | 515-540 | 34-35 / 38 | 2.1% | 32-34 / 40 — **mean 83.3%** | 17/20 |
| + both fixes | 994-1007 | 38 / 39 | 6.4% | 33-34 / 40 — **mean 84.2%** | **18/20** |

**What improved, and it is structural rather than cosmetic.** `_demote_repeated_candidates`
took the document from 72 parsed sections to 56 and gave `13. Video Wall Technical
Requirements` its body back — 41 characters to 19,726. The exhaustive map prompt roughly
doubled the extracted items and every selected section now contributes (38 of 39, against 34 of
38). `C16`, `C19` and `C27` went from intermittent to consistently covered, and the **"must"
tier rose to 18 of 20 in every run**.

**What did not.** The headline moved 83.3% → 84.2%, which is inside the run-to-run spread and
is NOT a demonstration that the bar is met. The gains traded against losses: `C30` and `C31`
were covered before and are missed in all three runs now, most likely because ~1,000 items need
12 reduce calls and each one gets proportionally less room to render.

**A regression the exhaustive prompt introduced, and the fix for it.** Telling the model to
split a list into separate items made it PARAPHRASE while splitting, and the quote rule — doing
its job — discarded them: the drop rate went from 2.1% to **12.0%**, almost six times. One more
prompt line ("if the text reads `UL (UL 62368)`, the quote is `UL (UL 62368)`, not `UL 62368`")
brought it back to 6.4% while item counts rose further. It is still three times the old rate,
which is the honest cost of asking for enumeration, and `dropped_unquoted` is the number to
watch if that prompt is edited again.

**Missed in all three runs now: C08, C23, C29, C30, C31, C36.** `C08`'s section is no longer
named after a CIN number — the withdrawal clause now sits under a heading that contains the
words "is withdrawn" — but the keyword scorer still does not credit it, which is as likely to be
the scorer as the answer.

## Follow-ups

1. **The map prompt favours specification rows over certification clauses** on a table-dense
   section. Three "must" items were lost this way with the text fully in front of the model.
   This is the first thing to try before widening any bound.
2. **Page headers are still promoted to section headings** (`CIN: U40101AS2003SGC007238`),
   alongside the table column headers already recorded in `phase8_aegcl.md`. Same Phase-2 path.
3. ~~FR-E4 faithfulness does not run on this path.~~ **CLOSED in Phase 12.** Both the
   streaming and non-streaming map-reduce answers now go through
   `_route_map_reduce_faithfulness` — one function, so the two cannot reach different verdicts —
   with the per-KB thresholds, the distinct `FAITHFULNESS` audit event and the existing HITL
   hook. The generation service's self-correction and completeness re-prompting are still not
   copied, deliberately: both rewrite the answer by re-prompting over the same spans, and this
   answer was not written from spans.

## Scored against the SHIPPED harness, and timed (2026-10-23)

Everything above was scored by a throwaway keyword counter living in a scratch script. That is
a floor, it was never reviewable, and it measures only ONE of the plan's criteria. The gold set
is now a repository artifact — `eval/gold/aegcl_corrigendum.yaml`, the plan's own Appendix A —
and the same three saved answers were rescored by the scorer that actually ships.

| run | must recall | should recall | value accuracy | supersession |
|---|---|---|---|---|
| 1 | 0.90 | 0.50 | 0.82 | pass |
| 2 | 0.90 | 0.50 | 0.82 | pass |
| 3 | 0.90 | 0.55 | 0.82 | fail (BDS pair absent) |

**The must tier CLEARS the plan's 0.85 bar at 0.90, in every run, with no variance.** That is a
firmer result than the keyword count suggested, and it is the tier the plan weights.

**The should tier is the content gap: 0.50–0.55.** Same causes already attributed above — the
three sections past the window bound, and the map prompt preferring specification rows to the
certification clauses at a section's end.

**Value accuracy is 0.82 and is not a separate defect.** All three misses are `IEC 61000-4-5`,
`IEC 62443-4-2` and `ISO 27001`, which are the numeric shadow of recall misses C16, C23 and C33.
In these runs the system never states a figure WRONGLY; what it misses, it omits. Worth stating
plainly because the opposite would be far more serious and the quote rule would not catch it —
a span can be quoted verbatim and still be summarised into a wrong number at the reduce.

That number first read 0.65–0.71, and the difference was a defect in the GOLD, not the answer:
five entries were transcribed as bare standards numbers (`"62368"`, `"27001"`). `extract_values`
canonicalises `UL 62368` to `std:ul 62368` and a bare `62368` to `num:62368` — on purpose, so a
designation is not read as three loose figures — so those entries could never match a correct
answer. The gold now uses the designations the document itself writes.

### Latency, measured end to end for the first time

A full cold run — select, map, reduce, assemble — against gemma4-31b:

```
map: mapped=39 failed=0 contributed=38 items=990 dropped_unquoted=76 truncated=3
     12 calls after window batching
WALL_SECONDS = 665.72
```

**666 seconds against the plan's ≤60 s aggregate target — an 11× miss**, and the first time the
criterion was tested at all. 39 map calls at concurrency 6 plus 12 reduce calls is what this
design costs on a 31B model; no single step is pathological. The Phase-12 map cache makes the
SECOND identical question cheap, but the criterion measures the first.

Nothing here is tuned in response. The three levers are visible — map concurrency, the window
budget, and batching several short sections into one call — and each trades against a number
above, so picking one is a decision to take with the measurement in hand rather than a knob to
turn now. `docs/rag_v2/goal_scorecard.md` carries the full §0 table.

## The should-tier gap closed, and the explanation above was wrong (2026-10-23)

**should-tier recall 0.50 -> 0.85-0.90 over three runs; value accuracy 0.82 -> 0.94;
supersession 2 of 3 -> 3 of 3; must tier 0.90 -> 0.95.** The §9 acceptance bar is met.

It was not the map prompt. Everything above attributes the recurring misses to "the map prompt
favours specification rows over certification clauses on a table-dense section", and the
follow-up list opens by calling that "the first thing to try before widening any bound". That
explanation is retired. It was plausible, it was repeated in two documents, and the measurement
that falsifies it is three model calls long: hold the prompt fixed and vary only the completion
budget.

| `v2_map_max_tokens` | one dense 12,000-char window | items kept |
|---|---|---|
| 2400 (shipped) | 47.6 s | **21** |
| 4000 | 79.2 s | 35 |
| 6000 | 120.0 s | 40 |

The model was not preferring specification rows. It was running out of output budget part-way
through the window, and a tender section puts its certification and bill-of-material clauses
LAST — so those are what fell off. The same geometry that made truncation so costly (recorded
above) made the completion budget costly in exactly the same way, one layer in.

Three defects, one shape: **a stage sized by its input while its output budget silently
truncated the work.**

1. `v2_map_max_tokens` 2400 -> 4000 (the large one, above).
2. `_reduce_batches` packed by prompt size only, so `infrastructure_hardware` arrived as ONE
   call with 446 items needing ~13,000 output tokens against a 4,000-token reply budget. Two
   gold items (C20, C40) were extracted correctly, passed the quote check, and never appeared
   in the answer. `batch_to_budget` now takes a renderable-items ceiling derived from
   `reduce_max_tokens`.
3. `v2_map_max_windows` 3 -> 6. The old comment named `sections_truncated` as "the number
   to watch before widening this"; watched, it was 3, and two requirements sat just outside the
   36,000-character cap — PeopleCert at offset 36,178, ISO 27001 at 65,487.

**None of the first two leaves a trace, which is why they outlasted four rounds of measurement.**
A budget-limited map reply is not truncated JSON — the model closes the array cleanly and returns
a complete, well-formed, SHORT list, so `dropped_unquoted` is 0 and `sections_truncated` is
silent because the section WAS fully shown. The reduce behaves the same way: the call succeeds
and the un-rendered items are simply absent. Only the window bound had a counter, and it is the
only one anyone had found.

### The three runs

| run | items | must | should | value | supersession | wall |
|---|---|---|---|---|---|---|
| 1 | 1744 | 0.95 | 0.90 | 0.94 | pass | 763 s |
| 2 | 1742 | 0.95 | 0.85 | 0.94 | pass | 766 s |
| 3 | 1726 | 0.95 | 0.85 | 0.94 | pass | 973 s |

**Two of three sit exactly ON the 0.85 bar.** Met, not comfortably cleared, and recorded as a
range for the same reason as every other number in this file.

**What it cost: 485 s -> 766 s.** Items per answer nearly doubled and those tokens have to be
decoded. A deliberate trade — the aggregate latency target is already unreachable on this serving
tier (see `goal_scorecard.md`) while recall was a live criterion — and a one-line revert if the
priority changes.

**Still missed in all three runs: C23 (IEC 62443-4-2, must), C29 (PeopleCert), C31 (EPEAT);
C32 (MITRE/3000) in two of three.** These are now genuinely "read it and did not extract it":
the text is inside the window and inside the budget. That is the prompt-or-capability question
the retired explanation claimed to be, and it should be measured the same way this one was
rather than reasoned about.

## CORRECTION to the section below: it measured the wrong text (2026-10-26)

The window numbers in "The last four misses" were taken against `LayoutSection.text`. That is
not what the map reads. `_map_inputs` builds each `MapJob` from the section's stored CHUNKS,
joined — `"\n\n".join(text for _cid, text in chunks)` — so the map's input is the indexed,
masked, overlapping chunk text, which is a third thing again from both `LayoutSection.text`
(tables counted twice) and `section_prompt_text` (tables de-duplicated). The same section
measures 85,744, 29,755 or something else depending which one you take.

What that invalidates, specifically:

* **"the largest section needs 29 windows at 3,000 chars"** — 29 is the raw-text figure. By
  prompt text the largest section is 29,755 characters and needs 10. The real map input has not
  been measured, so neither number should be quoted as the cost.
* **"roughly four times the map calls"** — derived from the 29, so it carries the same error.
* **the window sweep itself** (37/37/37 at 12,000 against 40/40/40 and EPEAT recovered at 3,000)
  was run on raw text. The DIRECTION is probably sound, since all three conditions used the same
  string and only the window changed. The absolute sizes are not production's.

What survives unchanged, because it does not depend on which text was used: `_MAX_ITEMS_PER_WINDOW`
truncates silently and now does not — that is measured from the reply, not the input, and the
40/40/40 ceiling is a property of the cap. The observability change this branch shipped stands.

Also unresolved and worth stating rather than leaving implied: the full run reports
`sections_truncated=2`, while NO section exceeds the 6x12,000 cap by prompt text (63 map calls,
0 over). Chunk overlap plausibly explains it — joined chunks can be longer than the text they
came from — but that is a hypothesis, not a measurement.

**So the four misses are NOT diagnosed, and the section below should be read as a record of what
was tried rather than a conclusion.** Settling it needs the map's real input, which means a
throwaway KB with the document ingested, not a layout parse in a scratch script. The reason to
stop here instead is that the `should` tier already clears its bar and these are three items;
the reason to write this down is that a wrong cost estimate in a commit message is how the next
person picks the wrong lever.

## The last four misses, diagnosed and NOT fixed (2026-10-26)

C23 (IEC 62443-4-2, must), C29 (PeopleCert), C31 (EPEAT) and C32 (MITRE/3000) survive every
round of the above. They are now fully attributed, and the attribution says the remaining fix
costs more than it is worth on this model — so this records the measurement and stops, rather
than tuning toward four items.

**They are not one problem.** C29 is EXTRACTED and lost at the reduce, which writes prose over
~1,700 items and is lossy by construction; the other three never reach the item list at all.

**For those three, three plausible causes are ruled out by measurement, not by argument.** The
40-item parse cap is not binding at shipped settings (32-37 kept). The quote rule is not binding
(`dropped_unquoted` 0-1 on these sections, against the 78-120 the whole-document counts show
elsewhere). The completion budget is not binding (~2,870 tokens used of 4,000). EPEAT sits at
offset 3,512 of an 11,688-character section, inside the window, in a table row the extractor
mangled to `33 | Environment al | ENERGY STAR®, EPEAT Gold, RoHS, CE, FCC Certified |`, and the
model's 11,478-character reply never mentions it.

**The map has a throughput ceiling in ITEMS, not characters.** Same section, same prompt, same
budget, three runs per window:

| window | items per run | EPEAT recovered |
|---|---|---|
| 12,000 | 37, 37, 37 | 0 of 3 |
| 6,000 | 37, 36, 29 | 0 of 3 |
| 3,000 | 40, 40, 40 | **3 of 3** |

Roughly 37 items come back whatever the window holds, so a 12,000-character window of a
60-requirement table is answered about half way and the rest is invisible. Shrink the window and
the same model finds the item every time.

**And at 3,000 the parse cap starts truncating.** 40, 40, 40 exactly is not a model deciding it
is finished; it is `_MAX_ITEMS_PER_WINDOW` being hit. So the fix is not one knob but two — and
3,000-character windows put the largest section at 29 windows against a bound of 6, taking map
calls from ~81 to roughly four times that on a path already at 766 s, to recover three items on
a tier whose bar is already met. That is the trade, stated so it can be taken deliberately later.

**A prompt fix was tried first and did nothing.** Telling the model to walk a table row by row,
with the failure named ("if the last row you return is numbered 34 and the table runs to 61..."),
left the counts where they were: 37, 37, 37. It was reverted. The lesson is the one this file
already learned once in the other direction — the output-budget fixes worked because they removed
a hard constraint, and arguing a model out of a soft behaviour did not.

**What did ship from this: the cap is no longer silent.** `MapParse.dropped_over_cap` counts what
`_MAX_ITEMS_PER_WINDOW` discards, `MapReport` carries it, and the pipeline raises its own
`rag_v2_map_items_over_cap` warning naming the knob to turn. The bound was fine; its silence was
not, and it is the seventh truncation of that shape found on this branch.

## C31 traced through the real pipeline (2026-10-26)

`scripts/trace_gold_item.py` follows one gold item through the shipped stages — it calls
`_route_v2`, `_select_sections`, `_map_inputs`, `_mapped_items` and `_reduce_batches` rather
than rebuilding them, which is the whole reason it exists. Against a throwaway KB with the
corrigendum ingested through the normal path:

```
OK    router          intent=aggregate facets=0
OK    selection       32 of 56 sections
OK    map input       section d00fe2ef066e text=7903 chars -> 1 window of ~1 needed
OK    map window      cues in window 0 of 1
OK    map             1102 items from 32 sections (unquoted 107, over cap 7)
LOST  map extraction  the model was shown the text and returned no item carrying it
```

**The map's real input for that section is 7,903 characters.** Not the 11,688 of
`LayoutSection.text` and not the 3,825 of `section_prompt_text` — a third number, because the
map reads the section's stored CHUNKS joined. It fits in ONE window, so the entire window
analysis above is irrelevant to this item: there was never a window boundary to fall past.

So C31 is settled, and it is the dull answer: shown a single 7,903-character window containing
`33 | Environment al | ENERGY STAR®, EPEAT Gold, RoHS, CE, FCC Certified |`, the model returns
no item for it. Not a budget, not a bound, not the quote rule for this item specifically.

Two things the same run surfaced that the aggregate counters had been hiding:

* **`over cap 7`** — `_MAX_ITEMS_PER_WINDOW` is discarding items at SHIPPED settings, not only
  in the small-window experiment. The counter added for exactly this caught it on its first
  production run.
* **`unquoted 107` of ~1,200 items**, about 9%, which is the quote rule rejecting what the model
  wrote. Worth its own look: it is seven times the 1.3% that the four-round tuning above was
  chasing, and nobody has asked which items they are.

### The v2 section pass needs the document NAMED

Traced by accident and worth recording, because it changes what a measurement means.
`_select_sections` computes `scope = decision.doc_ids or query.doc_ids` and returns None when
both are empty — deliberately, so a question about "this document" is never answered from
documents the asker never mentioned. A query carrying no `doc_ids` therefore skips Phases 8 and
9 entirely.

Asked unscoped through the live API, "give me the compliance details from this document" is
served instead by the whole-document read, which answers from **400 of 1,793 passages** and says
so only inside the answer text. No `rag_v2_*` event fires at all. The v2 numbers in this file
are all for the SCOPED case, which is the intended one; the unscoped case is a different
algorithm with its own truncation, and from outside the two are indistinguishable.
