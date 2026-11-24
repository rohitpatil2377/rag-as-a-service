# Did we hit the goal? — the plan's §0 criteria, measured

Every phase of the upgrade plan has its own acceptance criterion and those were reported as the
work landed. This file answers the different question the plan actually opens with: **are the
§0 success criteria met?** They are not the same question, and reporting only the former let
the latter drift out of sight for the whole build.

Measured 2026-10-23 against the real AEGCL SOC/NOC Corrigendum-V, with Phase 3 and Phase 4 run
by **gemma4-31b** and the selection scored by this deployment's BGE cross-encoder. Scoring is by
the SHIPPED harness (`svc_guardrails_eval.suite`) against `eval/gold/aegcl_corrigendum.yaml`,
which is the plan's own Appendix A transcribed — not the ad-hoc keyword scorer the earlier
Phase-9 records used. Three runs of the aggregate query.

## The configuration a number was measured under is part of the number

A measurement is only valid for the configuration and workload that produced it, and a REAL
measurement can mislead just as badly as a synthetic one once the thing it measured no longer
ships. This branch produced three examples in two days: a synthetic 173 s map stage quoted as if
it predicted production (the real figure is 587 s); a genuine 302 s map stage quoted after the
settings that produced it had been replaced; and a concurrency-24 gain predicted from a per-call
sweep that the batch contradicted.

So: **every headline below carries the configuration it was taken under, and a material change
to extraction depth, token budget, window count, model, concurrency or serving tier invalidates
the affected rows until they are remeasured.** A row whose configuration has moved is marked
STALE rather than quietly carried forward — being out of date is a fact about a number, not a
reason to hide it.

CONFIGURATION OF RECORD for the rows below, unless a row says otherwise:

    model gemma4-31b @ uat-bodhi (SHARED)   RAG_LLM_REASONING_EFFORT=off
    v2_map_concurrency   12             v2_map_max_tokens      4000
    v2_map_max_windows    6             v2_map_section_chars  12000
    v2_reduce_max_tokens 4000           doc_profile_head_chars    12000
    corpus: AEGCL Corrigendum-V, one document, one class

| § 0 metric | Target | Measured | Verdict |
|---|---|---|---|
| Aggregate item recall — **must** tier | ≥ 0.85 | **0.95** in all three runs | **MET** |
| Aggregate item recall — *should* tier | ≥ 0.85 | **0.85 – 0.90** | **MET, on the bar** |
| Value accuracy | ≥ 0.95 | **0.94** in all three runs | **MISSED by 0.01** |
| Claim faithfulness (NLI rate) | ≥ 0.90 | **0.9875** on the current answer (~1,744 items, 19 unsupported of ~500 claims) · **1.00** lookups | **MET** |
| Supersession correctness | 100% | **3 of 3 runs**, after a metric correction | **MET** |
| Lookup accuracy — no regression | no regression | **0.905** over 3 repeats, 1 flaky. Was 0.714 — two of the three "failures" were GRADER bugs | **MET** |
| p95 latency — lookup | ≤ 8 s | **11.9 s** latest (14.2 s earlier; v1 baseline 13.2 s) | **MISSED (~1.5×)** |
| p95 latency — aggregate | ≤ 60 s | **766 s** median of 3 | **MISSED (~13×)** |

Every row is now measured; nothing is left at UNKNOWN. The lookup rows come from
`scripts/eval_gate.py` over the shipped suite's seven §A.2 lookups, 3 repeats each, against a
throwaway KB holding the real corrigendum — run twice, once on the v2 path and once with the v2
flags off on the SAME code, which is what makes the regression row a comparison rather than an
assertion. Faithfulness is the shipped NLI checker: 1.00 over the lookups, 0.9947 over the
aggregate answer (the 9 claims it flags there are sentence-splitter artifacts — bare `[m27]`
citation markers and headings — not unsupported statements).

## The five target behaviours are all built

The plan's §0 also lists five behaviours rather than numbers, and those are done: the system
knows what each document is (Phase 3) and how it is structured (Phase 2), detects the question
kind and picks a strategy (Phases 7–8), grounds every claim in a verbatim quote and verifies it
(the quote rule, Phase 10's value check, FR-E4 on both answer paths), and organises the answer
by the document's own facets with clause references, old → new for amendments and a closing
coverage statement (Phase 9).

The hard constraint held too: it is still **true RAG**. Nothing is answered from model
knowledge, and the map step reads only the sections Phase 8 selected — there is no
full-document stuffing on the primary path.

## What the numbers say

**Latency is the one order-of-magnitude miss, and the target is not reachable on this serving
tier.** The aggregate is now 485 s, down from 666 s — `v2_map_concurrency` was 6, chosen to
avoid overloading a tenant's endpoint before anything had been timed, and 12 is where the
measured curve is still nearly free. The remaining gap is not a tuning problem, and the
measurement says so plainly:

| stage | seconds | share |
|---|---|---|
| map | 302 | 62% |
| reduce | 99 | 20% |
| section selection (local cross-encoder) + overhead | ~83 | 17% |

**The cost is DECODE, not prefill.** A 5,346-character prompt and a 15,815-character one both
take 48 s, because both emit ~2,000 tokens; prompt size barely registers. The map stage emits
roughly 115,000 tokens in total, so finishing it inside 60 s needs about 1,900 tok/s sustained
before the reduce writes anything. Measured at full 39-way concurrency this endpoint peaks at
**771 tok/s**, and it was still returning more throughput at 39 than at 20 — it is not
mis-scheduled, it is out of capacity.

Two things follow. Raising concurrency further keeps paying but shallowly: at ~60 map calls,
conc 20 buys a ~230 s map stage and conc 39 about 155 s, against 302 s today — a best case near
340 s all-in, still 5-6x the target. And the only lever that would actually reach 60 s is
emitting fewer tokens, which means extracting fewer items or dropping the verbatim quotes —
trading directly against recall, which already misses its own bar, and against the grounding
rule. So the target as written is incompatible with this design on this hardware. That is a
scope or a hardware decision, and it belongs with the owner rather than in a config file.

Worth noting what did NOT change: 1035 items extracted (up from 990), 39/39 sections mapped,
0 failures, must tier 17-18/20, MAAT correct. Concurrency is scheduling — it moved the clock
and nothing else.

**The must tier clears its bar and is stable.** 0.90 in every run, no variance — the twenty
items the plan weights most are answered.

**Value accuracy is not an independent finding.** It reads 0.82, and all three misses are
`IEC 61000-4-5`, `IEC 62443-4-2` and `ISO 27001` — the same standards behind recall misses C16,
C23 and C33. There is no case in these runs of the system stating a figure *wrongly*; every
value it missed, it omitted. That is worth knowing precisely because the opposite would have
been far more serious, and the quote rule would not have caught it.

**So the should tier is the real content gap**, at 0.50–0.55 against 0.85. Two causes, already
attributed in `phase9_aegcl.md`: three sections still exceed `RAG_V2_MAP_MAX_WINDOWS=3` and
are read in part, and on table-dense sections the map prompt favours specification rows over the
certification clauses that sit at the end.

## The quality gaps closed, and the one root cause behind all of them

should-tier recall went **0.50 -> 0.85-0.90** and value accuracy **0.82 -> 0.94**, and the work
was not prompt engineering. Every miss that mattered traced to ONE defect shape, in three places:
a stage sized by its INPUT while its OUTPUT budget silently truncated the work.

**1. The map gave up half of a dense section.** `v2_map_max_tokens` was 2400, set where a
typical section finishes. A tender's dense sections are not typical. Same 12,000-character window
of the real corrigendum, same prompt, three budgets: 2400 -> 21 items, 4000 -> 35, 6000 -> 40. So
the shipped setting was emitting barely half of what the window holds, and this was the single
largest cause. Raised to 4000, the knee.

**2. The reduce was handed batches it could not answer.** `_reduce_batches` packed by prompt size
alone, so `infrastructure_hardware` arrived as ONE call carrying 446 items — comfortable inside
the 32k window, needing ~13,000 output tokens against a 4,000-token reply budget. `batch_to_budget`
now takes a second ceiling on renderable items, derived from `reduce_max_tokens` so the two cannot
drift apart.

**3. Two requirements sat just past the window bound.** `v2_map_max_windows` was 3, a 36,000
character cap; the PeopleCert clause is at offset 36,178 — 178 characters outside — and ISO 27001
at 65,487. Raised to 6 for +9.5% map calls on this document.

**WHY THIS HID FOR SO LONG, and the part worth remembering.** The first two leave no trace at all.
A budget-limited map reply is not truncated JSON: the model closes the array cleanly and returns a
complete, well-formed, SHORT list, so `dropped_unquoted` is 0 and `sections_truncated` says nothing
— the section WAS fully shown. The reduce is the same: the call succeeds and the un-rendered items
are simply absent. Only the third had a signal, which is exactly why it was the only one previously
known.

This retires an explanation this repo carried since Phase 9: "the map prompt favours specification
rows over certification clauses on a table-dense section". It never did. Certifications sit at the
END of a tender section and the model was running out of budget before reaching them. A plausible
story, repeated in two documents, and wrong — the measurement that could have falsified it (vary
the budget, hold the prompt) is three calls long and had never been run.

**What it cost.** Items per answer went 984 -> ~1740, and those tokens have to be decoded: the
aggregate went 485 s -> 766 s. That is a deliberate trade, taken because the latency target is
already out of reach on this hardware (see below) while recall was a live criterion. Reverting is
one line: `v2_map_max_tokens` back to 2400.

**What is still missed, and it is now a short list.** C23 (must, IEC 62443-4-2), C29 (PeopleCert)
and C31 (EPEAT) fail in all three runs; C32 (MITRE/3000) in two of three. All four are now
genuinely "the model read it and did not extract it" — the text is inside the window and inside the
budget. That is a prompt or model-capability question and the next thing to measure, not guess at.

**Two of three runs sit exactly ON the 0.85 bar**, so the criterion is met rather than comfortably
cleared. Recorded as a range for the same reason every other number here is.

**Lookup accuracy does not regress, and the one real failure is not v2's.** The same seven
lookups score 0.714 on both paths, and both miss the same two. One of those is not the system's
fault at all — see below. The other, the sustained-EPS figure, fails identically with v2 off, so
it is a retrieval limit the v2 work neither caused nor fixed.

**One gold item is unanswerable from the corpus, and that is a corpus gap, not a system gap.**
`How long must the bidder have operated a SOC or NOC?` expects `6 (six) months`, and that phrase
appears nowhere in Corrigendum-V in any wording — there is no SOC/NOC operating-period clause in
it. Appendix A was transcribed for the tender AS A WHOLE, but only the corrigendum is ingested,
so one lookup and one `must` checklist item (C03) can never be satisfied by this KB. Scored
against what the corpus can actually support the lookups are 5 of 6 (0.833), and the must tier
is 18 of 19 (0.947). BOTH numbers are kept in this file: the raw one is what the suite says, and
the adjusted one is what the system is responsible for. The item stays in the gold rather than
being deleted, because a missing source document is a real coverage gap worth seeing.

**Lookup latency misses its target on both paths.** 14.2 s against ≤ 8 s, with the v1 baseline at
13.2 s — so v2 adds about a second (7%) and the bar was already out of reach before any of this
work. An earlier run of the same suite read 162 s p95 and was discarded rather than reported: two
other jobs were sharing the box, and a contended number is not a measurement.

## Two defects this measurement found in the measurement

Both were in the scoring, not the pipeline, and both would have been invisible without wiring
the shipped scorer to a real artifact.

**The plan contradicts itself on supersession.** §9.2 reduce rule 3 says *"for amendments, show
'old → new' explicitly and state what was withdrawn"*; §11 and §0 say *"amended values shown as
new"*, implemented as "states the new value and does **not** state the old". Phase 9 obeys the
first, so every run failed the metric for doing what the plan asked — it scored **0 of 3** on
first contact. §9.2 is the one worth keeping: a compliance reader needs to know a figure
CHANGED, not merely what it is now, and an answer that silently states ₹19.44 Cr gives a bidder
no way to tell their ₹35 Cr assumption is stale. `_supersession_scores` now counts an old value
as stale only when no occurrence of it is followed closely by its replacement; the failure it
exists to catch — the superseded figure asserted as current — is still caught, because the new
value must appear at all. 2 of 3 runs then pass, and the third genuinely does not pair the BDS
amendment.

**Value accuracy first read 0.65–0.71, and a third of that was the gold file's fault.** Five
checklist entries were transcribed as bare standards numbers (`"62368"`, `"27001"`). The scorer
canonicalises `UL 62368` in an answer to the token `std:ul 62368` and a bare `62368` to
`num:62368` — deliberately, so one designation is not read as three loose figures — so those
entries could never match a correctly-written answer, however right it was. The gold now carries
the designations the document itself uses, and the number moved to 0.82. The lesson is about
gold sets, not about this one: an entry that the scorer cannot score looks exactly like a
failing system.

## Phases 3 and 4, scored at last

Both had carried "acceptance STILL UNVERIFIED" since 2026-10-21, and the reason was never that
nobody looked: the plan states these two acceptances as prose about an ARTIFACT (a profile card),
while the shipped harness scores ANSWERS. There was nothing to run them with.
`eval/gold/aegcl_doc_card.yaml` is that prose turned into assertions and
`scripts/score_doc_card.py` runs it, against the card and enrichment the real pipeline emits.

**Phase 4 — MET, 3 of 3.** On the real corrigendum the MAAT section yields
`amends {old: "Rs. 35 Crores", new: "Rs. 19.44 Crores"}`, the router section's propositions carry
both HBOM and SBOM, and quote validation is **398 of 398 = 1.000** against the plan's >= 0.95 bar.
Every quote the enricher attached is verbatim in its own section; 38 of 39 enrichable sections
enriched, 1 failed.

**Phase 3 — 3 of 4, and the fourth is IMPROVED BUT STILL FAILING.** `doc_type` is `corrigendum`;
the `amends` relations are there and on target (the NIT by number, and Corrigendum-II by date),
each with its verbatim evidence quote, which is §3's hard rule. What fails is FACET COVERAGE —
now **0.50-0.60, mean 0.54 over five runs**, against a 0.80 bar. It was **0.38** before this
branch touched it.

The card is subsystem-shaped — `ng_siem`, `hci_infrastructure`, `video_wall_systems`,
`noc_infrastructure` — where §3 asks for compliance dimensions. So supply-chain security, OEM
credentials, deployment/sovereignty, lifecycle/support and training get no facet at all, and the
document covers every one of them substantially: `training` appears 70 times, `AMC period` 33,
`no internet` 11, SBOM and HBOM 4 each. Two independent profile runs produced the same shape with
different labels, so it is the prompt's framing rather than a bad roll.

THIS IS THE TAXONOMY MISMATCH `phase9_aegcl.md` ALREADY BLAMED for capping that measurement —
"Appendix A groups by compliance dimension; the card gemma derived groups by subsystem" — now
measured at its source instead of inferred from downstream recall. It is also the most likely
reason `soc_noc_infrastructure` absorbed 278 of 523 items in the earlier Phase-9 runs: a facet
that names a subsystem collects everything that subsystem mentions, which is not a grouping an
answer can be organised by.

### What moved it, and the idea that lost

Two changes, one of which did not survive its own measurement.

**The evidence window was the larger half.** `_profile_head` implemented the plan's "first ~2
pages of text" literally, which on this document is 4,119 of 193,432 characters — **2.1%**. The
model was being asked to name the content areas of a document it had seen one fiftieth of, and
the only whole-document signal it had was the 400-heading outline, which is why the card came
back shaped like the table of contents. Spending the whole `doc_profile_head_chars` budget it
was already provisioned — same tokens, same call — took coverage from 0.38 to 0.54.

**Sampling the rest of the document scored WORSE and was reverted.** The obvious next step is to
reach past the opening, so the same budget was spent on the head plus six evenly spaced samples
of the remaining 94%. That reads the whole document and it measured **0.44**, below the 0.52 the
same budget scores read contiguously. Six fragments of mid-document table rows are poorer
evidence than one coherent passage. The change is gone; the finding is kept, because it is the
reason not to try it again.

**The prompt change fixed the SHAPE, not the coverage.** Facets no longer carry product or
subsystem names at all — `bidder_technical_experience` and `compliance_documentation` where there
used to be `ng_siem` and `video_wall_systems` — which is the property that matters downstream,
since a facet named after a subsystem collects everything that subsystem is mentioned in. On the
theme metric it was worth about +0.02, inside the run-to-run spread.

**What is still missing splits in two, and only one half is ours.** `training` first appears at
character 61,842 and the standards clauses at 33,817: no front-loaded window reaches them, and
the one technique that would — sampling — measured worse. Supply-chain security (3,637), OEM
credentials (5,466) and lifecycle/support (9,393) are all *inside* the window the model now sees
and are still not faceted, which is a prompt or capability question and not a budget one.

**A second, separate finding: the profiler gave up after one retry, and a cluster of failures
beat it.** `LLMDocProfiler` allowed exactly one retry and then returned None. One dump run hit two
consecutive `no JSON object in the reply` replies and lost the card for that document entirely;
in the pipeline that is fail-safe — the document ingests without a card — so Phase 4's prompt,
Phase 7's routing and Phase 8's facets all quietly get less, with nothing reporting a fault.

The rate is NOT the story, and an early claim here that it "fails about half the time" was wrong:
it came from two adjacent observations. Eight consecutive calls on the identical prompt all
returned a usable card, so the steady-state rate is high. Failures arrive in CLUSTERS, and a
cluster is exactly what a one-retry budget cannot survive.

The one-retry cap was reasoned, in the adapter's own docstring, from "retrying the identical
prompt would be pointless at temperature=0.0: a greedy decode re-reads the same prompt and writes
the same bytes". That premise does not hold here. `OpenAICompatGateway._eff_temp` raises a
requested 0.0 to 0.6 whenever reasoning is suppressed — which `RAG_LLM_REASONING_EFFORT=off`
does for every model, deliberately, to stop a no-reasoning model looping. The attempts were
always independent samples; the profiler simply declined to take them. Now bounded by
`doc_profile_retries` (default 4), early-exiting on the first usable card so the ordinary case
still costs one call, and still breaking out immediately on a GATEWAY error, where there is
nothing different to send.

## Three production bugs the measuring found, none of which a test would have

Taking these numbers needed the real document through the real API, and that is the only reason
these surfaced. Each one is silent by construction — a warning log and a plausible-looking
result — so nothing in 2705 unit tests, 380 e2e tests or any phase acceptance was ever going to
catch them.

**1. FR-E4 was not running on v2 aggregate answers.** `NLIFaithfulnessChecker.check` scored every
sentence concurrently with no bound. A 377-claim map-reduce answer opened more requests than
httpx's 100-connection pool, the overflow waited for a slot, the wait counted against the
per-call timeout, and one `PoolTimeout` sent the WHOLE answer down the fail-soft path to the
lexical checker. So the faithfulness control Phase 12 added to this path specifically was
reporting a lexical score while looking exactly like NLI. The bound now lives on `TEINLIScorer`,
which owns the pool and is an `lru_cache` singleton — the first attempt bounded `check()`
instead, and that was not enough: four concurrent checks each honoured their own 16 and together
held 65 sockets, reproducing the same exhaustion. A caller-side bound can always be multiplied by
another caller.

**2. The prompt-injection guard quarantined a real government tender.** The signature was the
bare substring `exfiltrat`, and it fired on the ordinary SIEM requirement `System detects data
exfiltration and unusual internal data transfers`. A quarantine indexes nothing and tells the
user nothing. Since this platform is sold into exactly the security-document market where "data
exfiltration" is everyday vocabulary, the guard was set to refuse its own corpus. Replaced with
two patterns built on the grammatical split that actually separates the cases: an attack uses the
verb as an imperative ("then exfiltrate the salary data") or names a destination, while a
security document uses the noun ("prevent data exfiltration") or the third person ("malware
exfiltrates credentials"). Verified in both directions.

**3. RAG v2 ingest could not ingest a table-dense document at all.** `QdrantVectorStore.upsert`
sent every point of a document in ONE request. Qdrant caps a request at 32 MB and rejects an
oversized one WHILE THE CLIENT IS STILL UPLOADING, so httpx raises a bare `ReadError` and ingest
500s with nothing written and no mention of size in any error. The corrigendum did not come close
to fitting: v2 ingest turned 68 text chunks into **1801 points** — 1425 of them table ROWS — and
each point carries the full document-level provenance (`extraction_*`, `ocr_*`, `sanitization_*`,
`entities`) at ~20 KB apiece, for **36.9 MB in a single call**. Now batched at 256 points
(`qdrant_upsert_batch`), after which the same document ingests as 1800 chunks. This is v2-only in
practice: the v1 path produces 379 chunks (~5 MB) and has always fitted, which is why it was
never seen.

Worth separating from the bug: that ~8 KB of identical per-document provenance on all 1801 points
is roughly half the index payload and buys nothing at retrieval time. Not touched here — it is a
storage-design question, not a defect.

## Retrieval, measured for the first time (2026-10-26)

Every number in this file until now has been about the ANSWER, which makes three different
failures look identical: the evidence is not in the corpus, the evidence is not retrieved, or
the evidence reaches the model and is not used. `scripts/retrieval_metrics.py` separates them,
against a real KB through `wiring.retrieval_service()`.

On the seven §A.2 lookups — the fair case, where passage retrieval really is the mechanism that
has to find the answer:

| metric | value |
|---|---|
| Recall@20 | **1.000** |
| Recall@5 | **1.000** |
| MRR | **0.929** |
| nDCG@10 | **0.947** |
| corpus gaps | **0** |
| retrieval misses | **0** |

Every lookup's evidence is at rank 1 or 2. Lookup ACCURACY is 0.714, so **both failures are
generation failures** — the sustained-EPS figure is at rank 1 and the answer still does not
state it. Retrieval is not this system's problem at the lookup grain, and the two known misses
have been attributed to the wrong stage until now.

**This corrects a claim this file carried.** `C03` was recorded as unanswerable, a corpus gap:
the SOC/NOC operating-period clause was said to appear nowhere in the corrigendum. It does —
"a continuous period of minimum 6 (six) months" — and it retrieves at rank 2. The original
check grepped a scratch extraction artifact instead of the ingested chunks, which is the sixth
time on this branch that a conclusion came from the wrong text. The adjusted figures derived
from it (lookups "5 of 6 = 0.833", must tier "18 of 19") were wrong and are withdrawn; the raw
0.714 and 0.95 stand.

**The checklist items are deliberately NOT scored here**, and the exclusion matters. Forty of
them share one aggregate question, so scoring them against its top 20 passages asks whether
twenty chunks can cover forty requirements spread across 1,808 — they cannot, and the Recall
that falls out (0.386, measured) is arithmetic rather than quality. The aggregate path does not
use passage top-k at all; it uses the Phase-8 section pass, which `trace_gold_item.py` reports
on. `--include-checklist` will still produce the number for anyone who wants to see it fail.

Two caveats on the method, both in the script's own docstring: gold evidence is DERIVED from
cue containment rather than annotated, so Recall reads optimistic; and a multi-cue item whose
words land in different chunks is reported as "split across chunks", not as a corpus gap,
because that is a chunking property and not a missing document.

## Where the 766 s actually goes: decode, not queueing (2026-10-26)

"The endpoint gives 771 tok/s" said the capacity was the problem but not WHICH part, and the
three candidates need different remedies: a scheduler backlog is an ops fix, prompt-processing
cost is a prompt fix, and decode is a hardware or a token-count decision. Streaming the real
map prompt through the production client separates them — time to first token is queue plus
prefill, everything after it is decode.

| concurrency | TTFT | decode | decode rate | wall | TTFT share |
|---|---|---|---|---|---|
| 1 | 1.62 s | 36.4 s | 169 ch/s | 38.0 s | 4% |
| 6 | 2.98 s | 42.5 s | 152 ch/s | 45.5 s | 7% |
| 12 | 3.31 s | 48.4 s | 130 ch/s | 51.7 s | **6%** |
| 24 | 5.98 s | 57.0 s | 111 ch/s | 63.1 s | 9% |

**Decode is 94% of a map call at the shipped concurrency of 12.** Queueing is real and it is
small: TTFT grows 3.7x from concurrency 1 to 24 while decode grows 1.6x, and TTFT never exceeds
9% of the call. The shape is continuous batching, not a backlog — per-request decode rate falls
(169 -> 111 ch/s) exactly as aggregate throughput rises (169 -> ~2,660 ch/s, about 666 tok/s).
The GPU is time-slicing across sequences; it is not holding requests in a line.

**Prefill is ruled out twice.** TTFT is 1.62 s at concurrency 1 for a ~15,000-character prompt,
and separately a 5,346-character prompt and a 15,815-character one both took 48 s end to end —
three times the input, identical wall time.

So of the three remedies only one reaches the TARGET: serving optimisation, a faster tier,
quantisation or speculative decoding.

CONCURRENCY 24 WAS PREDICTED TO HELP AND MEASURED SLOWER. The per-call sweep above shows
aggregate throughput still climbing — 912, 1,560, 2,664 ch/s at concurrency 6, 12 and 24 — and
since the map is a BATCH the makespan, not the per-call cost, is what matters. That argued 24
should beat 12 by roughly 71%. Run on the full workload it does not:

| concurrency | runs | wall | quality |
|---|---|---|---|
| 12 (shipped) | 3 | 763, 766, 973 s — median **766** | 37/40, must 18/20 |
| 24 | 2 | 917, 954 s — median **936** | 37/40, must 18/20 |

Same answer, 22% more wall time, zero failures either way. The extrapolation failed, and the
sweep's own setup explains why: it ran ONE fixed prompt at `max_tokens=2000`, while the real
stage runs 25 calls at a 4,000-token budget over 12,000-character prompts. Twenty-four of those
in flight is far more KV-cache pressure than the sweep ever created, and a server that starts
preempting or recomputing sequences gives back more than batching wins.

So the practical answer is: KEEP 12. Not because concurrency is exhausted in principle — per-call
numbers say it is not — but because the only measurement of the actual workload says 24 is worse
on this endpoint at these budgets. The general lesson is the one this branch keeps relearning:
a per-call curve is not a batch result, and the batch is what ships.

**And it settles the 60 s target — but use the MEASURED number, not the extrapolated one.**
Three figures get confused here and only one belongs in an owner brief:

| figure | what it is |
|---|---|
| 173 s | arithmetic from the synthetic sweep: 115,000 tokens / 666 tok/s |
| 302 s | a real map stage, at SUPERSEDED settings (`max_windows` 3, `map_max_tokens` 2400) |
| **587 s** | **the real map stage at the shipped settings today, concurrency 12** |
| 766 s | the real median end-to-end, concurrency 12 |

The synthetic figure understates the production map stage by **3.4x**. That is the same gap
that made concurrency 24 look attractive and measure slower, from the same cause: a sweep of one
short-budget prompt does not predict a batch of 25 long ones. 302 s is also no longer the
baseline — it was measured before the window and completion-budget fixes that took should-tier
recall from 0.50 to 0.85-0.90, so quoting it compares today's quality against yesterday's clock.

**587 s is the number to decide against.** At 60 s the map stage alone would need roughly ten
times the throughput it currently achieves, before selection or the reduce. The 60 s objective
is a capacity decision, not an algorithm one, and it is still unconfirmed with the owner.

CAVEAT, and it bounds how hard this can be pushed: `uat-bodhi` is a SHARED endpoint. Another
tenant's traffic is indistinguishable from our own queueing in these numbers. The direction is
solid — decode dominates by an order of magnitude over TTFT — but the absolute figures are one
sweep on a machine we do not control.

## Two grader bugs were costing 19 points of lookup accuracy (2026-10-26)

Lookup accuracy sat at 0.714 and both "failures" were traced, through the real retriever and
the real generation service, to answers that were CORRECT:

    "The SIEM must be able to handle a sustained 6,000 EPS ..."      gold: 6000
    "... for a continuous period of minimum **6** months."           gold: 6 (six) months

Neither is a model failure. `fold_text` folded case, whitespace and dashes, and compared
everything else literally — so a thousands separator, a spelled-out gloss and a pair of markdown
bold markers each hid a correct value. The last one is the most general: the answers ARE
markdown, the model bolds its figures, and the grader was reading the raw source, so
`**19.44**` or `**75 TB**` would have failed the same way on any question.

`fold_text` now folds all three. **Lookup accuracy 0.714 -> 0.905** over three repeats, with
faithfulness, context precision and abstain correctness all at 1.000.

The remaining gap is ONE FLAKY ITEM, not a bug: "minimum combined value of qualifying projects"
passes 33% of the time, because the model sometimes writes "₹15 crore" where the gold says
"15 crores". That is model variation, and the fix is NOT to relax the cue to "crore" — that
would be fitting the gold to one run's phrasing, which is how three of this branch's gold
entries went wrong in the first place.

**And the substring limit is now closed too.** Folding away separators and markdown left the
matcher able to find `600` inside `6000`, so an answer stating the WRONG capacity scored as
stating the right one. `contains_cue` adds boundary guards, but only to a cue that carries a
figure and only at whichever end of it is itself a digit — `75 TB` needs protecting at the front
("175 TB") and `UL 62368` at the back ("UL 623689"). Text cues stay plain substring matches,
because "Video Wall" and "PeopleCert" are assertions that some wording appears and making them
boundary-sensitive would change the meaning of every checklist item to fix a problem none of
them has.

A figure at the end of a sentence still matches — `(?!\.\d)` rather than `(?!\.)`, so "6000." is
the value and "6000.5" is not. Re-scoring all three aggregate runs with the stricter rule left
every number identical, which is the result worth having: no item had been passing on a spurious
substring.

What it deliberately does NOT do is equate "15 crores" with "₹15 crore". That would be a semantic
equivalence rule written to make one observed phrasing pass, and three gold entries on this branch
are already wrong from exactly that reasoning. Recurrence across document classes would be
evidence for such a rule; one flaky item on one corpus is not.

## What would move each number

| Gap | Lever |
|---|---|
| aggregate latency 766 s → 60 s | **the arithmetic below is now known to be the WRONG model of where the time goes** — the map stage is gated by one section's serial window chain, not by aggregate throughput (see the critical-path row), and the concurrency bound of 12 runs at 6.9-8.4 in flight. Scheduling windows through the existing semaphore takes the measured 534 s to a 348 s floor at the shipped window size with no quality change. That is still not 60 s, and the original objection stands for the remainder: **not reachable by tuning alone** — needs ~1,900 tok/s and the endpoint gives 771. Either a faster serving tier, or a revised target. Note recall was bought WITH latency here: `v2_map_max_tokens` 2400 trades ~280 s for ~0.3 of should-tier recall |
| ~~aggregate latency via concurrency~~ | **CLOSED — measured, 24 is SLOWER.** Median 936 s against 766 s at 12, same quality, no failures. The per-call curve predicted +71% throughput; the full workload disagreed, most likely KV-cache pressure the sweep's 2,000-token budget never created. Keep 12 |
| lookup latency 14.2 s → 8 s | present on the v1 path too (13.2 s), so this is not v2 work — start where v1 spends it |
| ~~should-tier recall 0.50~~ | **CLOSED** — 0.85-0.90; it was three output-budget defects, not the prompt |
| value accuracy 0.94 (target 0.95) | the same four items as the recall gap — C23, C29, C31, C32 — no separate work |
| ~~supersession 2 of 3~~ | **CLOSED** — 3 of 3 once the reduce stopped dropping items |
| the sustained-EPS lookup | **not retrieval** — the evidence is at rank 1 and the answer still omits the figure. A generation failure, as is the SOC/NOC lookup at rank 2 |
| C29 | **not the reduce.** Marker retention measured 100% on the three largest batches, and the PeopleCert item survives a reduce both alone and diluted in a 100-item batch. Where it is lost in a full run is still unknown |
| C31 | **settled by `scripts/trace_gold_item.py`**: the map's real input for that section is 7,903 chars in ONE window, the model is shown the EPEAT row and returns no item for it. Not a window, budget or cap problem — the window analysis was moot for it |
| C23, C32 | trace them the same way; the tool exists now and the answer takes one run |
| ~~map `unquoted` ~9%, kind unknown~~ | **MEASURED** — 7.9% (77 of 979) on the real `_mapped_items` path with the map cache off, and the kind is now known: of the 44 rejections judged against a whole section, 44 are text that section really holds (24 elided with `...`, 20 a list stem welded to its bullet) and NONE is a paraphrase. Of the other 33 the document holds 31, but in a different window, so the rule was right there and the fix is overlap, not a looser rule. 2 of 77 are unrecoverable. The model is reformatting, not inventing — see `scripts/quote_rejections.py` |
| map windows truncating at the token cap | **MEASURED, reproducible, and SILENT** — 30 of 47 windows in one run and 29 of 47 in the next stopped at exactly 4,000 completion tokens (`finish_reason: "length"`), across 22 sections each run with 21 in common. Every one of them PARSED, so nothing is counted: the parser salvages the complete items and the rest of the reply never existed as far as the report is concerned. `SectionMapper._attempt` calls `gateway.complete()`, which returns a bare string and discards `finish_reason`, so the mapper cannot see this even in principle — `complete_with_usage()` on the same class carries it. Bears directly on facet coverage: evidence past the cut cannot be extracted |
| ~~map `sections_failed` 9, then 5~~ | **NOT REPRODUCED** — two instrumented runs over the same corpus 45 minutes later mapped 47 of 47 windows with 0 failures, 0 unusable parses and 0 raised calls. The earlier failures were GATEWAY EXCEPTIONS, not unparseable replies (47 windows minus 36 and 38 recorded parses = 11 and 9 calls that never reached the parser), and the first of them ran minutes after a ~70-minute endpoint outage. A content- or section-specific cause is contradicted: the same 35 sections map cleanly twice. HONEST LIMIT: the root cause is not identified and cannot now be — both runs were piped through `grep -v '"event":'`, which destroyed the `map_section_call_failed` records naming the sections and the exception types. Transient, unexplained, and not currently occurring |
| the client read timeout at the shipped 4,000-token cap | **MEASURED, production risk, NOT a test artifact** — the gateway's read timeout was a hardcoded 120 s (`OpenAICompatGateway.__init__`), with no setting and no caller overriding it. Measured generation throughput on the UAT endpoint at concurrency 12 is ~38 output tokens/s, flat across token caps, so a 4,000-token map call needs ~105 s and clears the timeout by roughly 15 s. That margin is not reliable: one of two identical runs at the SHIPPED cap lost **8 of 54 map calls** to the timeout, and a lost call means its window never reaches the parser and its section quietly contributes nothing to the aggregate answer. Now operator-tunable via `RAG_LLM_TIMEOUT_SECONDS`, and the shipped default was **raised 120 s → 240 s on 2026-10-28 by owner decision** after a four-arm rerun completed 47 of 47 calls per run with zero failures at 240 s. A healthy call is unaffected — it returns long before either ceiling — so what grows is how long a genuinely stalled request takes to give up; watch `elapsed_s` and the timeout count after a deploy. **CORRECTION 2026-10-28: 240 s is not generous headroom, only sufficient headroom.** A scheduler benchmark run caught a ~20-minute endpoint degradation in which 40 of 64 calls died at exactly 240.0 s and 11 sections failed, at the SHIPPED window size and cap, while that run's surviving calls averaged 113 s against 85-88 s either side of it. The raise was still right — it is what let every other run complete — but the earlier 'all four arms completed 47/47' must not be read as margin to spare. The same arithmetic is why a 6,000-token cap cannot be evaluated at 120 s: ~158 s of generation against a 120 s limit |
| `ModelGatewayError` conflated read timeouts with an unreachable endpoint | **FIXED** — every non-HTTP-status `httpx` failure produced the same string, "Could not reach the model endpoint (…). Check the URL and that it is reachable." A client-side read timeout therefore read as endpoint instability, and did: 102 timeouts across two runs were first diagnosed as the endpoint flapping, when the endpoint was healthy and the limit was ours. `_failure_kind` now names the flavour (`read_timeout`/`connect_timeout`/`pool_timeout`/`unreachable`/`http_status`/`transport_error`), the timeout message states the elapsed time and the configured limit and points at `RAG_LLM_TIMEOUT_SECONDS`, and `details` carries `failure_kind`/`timeout_s`/`elapsed_s`/`exc_type`. `SectionMapper._attempt` logs those fields on `map_section_call_failed`, so a future `sections_failed` count is attributable without re-instrumenting |
| map output cap 4,000 vs 6,000 | **MEASURED under a timeout that let both arms finish; 4,000 RETAINED.** Two runs per arm, interleaved, same corpus (1,814 chunks), 240 s timeout, concurrency 12, 40-item window limit and everything else held; only `RAG_V2_MAP_MAX_TOKENS` varied, per-process. All four arms completed 47 of 47 calls with zero timeouts and zero `sections_failed`, which by itself settles that the earlier arm's 102 failures were the client limit and not the endpoint. 6,000 is a REAL quality gain and still not worth shipping: unique source spans 1248/1225 at 4,000 against 1379/1417 at 6,000 (the arms' ranges do not overlap, which is stronger evidence than any Jaccard), stable spans 937 → 1157 (+23%), truncation 68.1% → 40.4/44.7%. The cost is +29% map wall time (603 s → 778 s mean) and +31% completion tokens; throughput held at ~38 tok/s in both arms, so the cost is the requests that generate more, not a uniform slowdown. What kills it: the per-window 40-item limit then discarded **373 distinct spans at 6,000 against 13 at 4,000**, 325 of them recovered by no other window — more evidence lost to the second ceiling than the larger cap gained. Raising the cap moves the bottleneck instead of resolving it, and 8,000 would move more into the same ceiling. Owner decision 2026-10-28: retain 4,000, reject 6,000 for now, measure windowing next |
| subdividing the windows that hit a ceiling | **MEASURED; a REAL and LARGE recall gain at 2.2x the cost, and one level of splitting does not clear the ceiling.** Two runs per arm, interleaved, corpus `zzgoal-b7a0b1f0` (1,792 chunks), and NEITHER ceiling moved: cap 4,000, 40-item window bound, 240 s timeout, concurrency 12, same model, prompt, query, first-pass windowing and retry policy, cache off, both arms on the same instrumented map loop. The one variable is whether a ceiling-bound window is re-split at half the window size on the document's own block boundaries and its halves mapped in place of it (REPLACED, never merged, so nothing is double-counted). TRIGGER DEVIATION, deliberate: the brief said to subdivide the windows that hit the 40-item limit, but at 4,000 tokens that bound is nearly idle — 1 to 2 windows per run hit it WITHOUT also hitting the token cap, and every one of them was too small to split — while 61-66% of windows stop at the TOKEN cap, the same failure through the other ceiling. Triggering on the item bound alone would have measured a no-op by construction, so the trigger is either ceiling and the item-cap-only population is reported separately. RESULT: stable source spans 977 -> 1,436 (+47%), and the gain survives the check that matters — of 666 spans the subdivide arm adds, 587 cover section text no baseline span overlaps by half or more, so this is new evidence and not the same passages re-cut into shorter quotes. Union of quoted characters 50,696 -> 84,232 (1.66x). COSTS, stated separately from the evidence: 44 -> 95 calls (2.16x), 147.8k -> 303.3k completion tokens (2.05x), map wall time 526 s -> 1,160 s (2.20x). Throughput held at 35-40 tok/s in all four runs, so the cost is the extra requests, not a slowdown. The wall-time multiple EXCEEDS the call multiple because `_map_one` runs a section's windows in SEQUENCE while only sections run in parallel, so sub-windows lengthen the critical path inside a section rather than filling the concurrency bound — a scheduling artifact that is fixable under the same bound, not an inherent cost of the idea. TWO LIMITS: (1) one level of halving leaves 49-57% of sub-windows STILL at the token cap, down from 61-66%, so splitting attenuates truncation by about a third rather than resolving it; (2) 6 ceiling-bound windows per run could not be split at all — they are 1,211-5,733 chars yet return 28-40 items, below the 6,000-char half-size, so `section_windows` returns one window and the densest material in the corpus is exactly what subdivision cannot reach. REGRESSION, not noise: 119 spans stable in both baseline runs appear in NEITHER subdivide run, concentrated in 18 sections (worst 25, 23, 18) — re-splitting changes what the model considers salient within a half, and some baseline evidence stops being reported. Within-arm noise floor is 362 spans (baseline) and 538 (subdivide), which is why only the stable-in-both-runs sets are compared. No arm had a failed call, a failed section or a retry; `unquoted` drops 76-96 baseline against 73-77 subdivide; exact duplicate items zero in every run. NOT a shipping recommendation — the recall gain is large enough to pursue, but the right next lever is the first-pass window size, which costs nothing in scheduling and reaches the dense windows subdivision cannot |
| ~~**the map stage is gated by ONE section's serial window chain**~~ | **FIXED 2026-10-28 (see the row below for the measured result).** Was: MEASURED, and the binding constraint — not window size. In all six runs of the window-size sweep, map wall time equalled the longest section's sequential chain to within 0.1 s: 475.3 s vs 475.2 s, 592.7/592.7, 791.7/791.7, 816.9/816.8, 1129.3/1129.2, 1086.5/1086.4. Every one of those chains is the SAME 58,484-char section, and the other 33 sections finish inside its shadow. `LLMSectionMapper._map_one` runs a section's windows in sequence while only sections run in parallel, so the concurrency bound of 12 is never reached: mean calls in flight is 6.9-8.4, measured as total call-seconds over wall time. The floor if a section's windows were scheduled under the same global semaphore instead of serially is `max(work/12, slowest call)` — 348 s against 534 s measured at 12,000 chars, 479 s against 804 s at 6,000, and **643 s against 1,108 s at 3,000**. So the entire latency cost of a smaller window is a scheduling artifact, recoverable without touching windowing, the cap or the concurrency setting. The sequential loop is DELIBERATE (fanning out naively would breach the bound), so the fix is to schedule windows through the existing semaphore rather than to remove the loop. This also reframes the earlier subdivision result: its 2.2x wall-time cost was mostly this, not the idea |
| the map semaphore now bounds CALLS, not sections | **SHIPPED and MEASURED: utilisation 64% → 89-93%, wall time 56-57% above the achievable floor → 8-12%.** `map_sections` builds one `asyncio.Semaphore(concurrency)` for the whole stage and passes it DOWN to `_attempt`, which holds it around the gateway request and nothing else; `_map_one` offers a section's windows concurrently via `gather` and holds no permit itself (it awaits its windows, each of which acquires the gate, so a permit held across that await would wait on itself). Parsing and quote-validation run outside the permit — local CPU work that loads nothing at the endpoint. The retry stays sequential WITHIN a window: it has nothing to retry until the first call returns. Ordering is positional on both levels (`gather` returns in argument order), so section order, window order and the item ids assigned from them do not depend on which call came back first — the property that makes this invisible downstream. BENCHMARK, three clean runs per arm, interleaved, on one corpus (`zzgoal-73dacfc7`, 1,780 chunks), the `serial` arm restoring the PREVIOUS permit-per-section loop verbatim and the `parallel` arm running the SHIPPED code untouched — measurement at the gateway edge for both, because a scheduler change measured through a reimplemented scheduler measures the reimplementation. Endpoint speed varied 36% between runs (busy 3,526-5,006 s), so each run is scored against ITS OWN floor (busy/12): serial 1.56/1.57/1.57 at 64% utilisation, parallel 1.08/1.09/1.12 at 89-93%. Peak in flight was exactly 12 and never 13 in all six runs. Raw wall 466 s → 374 s (-20%), or 461-471 s → 332/336 s (-28%) on the two parallel runs that met a healthy endpoint. Zero failed calls, sections, retries or unusable replies throughout. EXTRACTION held on every measure but one: items 1094-1152 vs 1056-1115 and `unquoted` 78-138 vs 123-149 both OVERLAP, and generated volume is flat (452,683-457,555 vs 448,473-460,622 reply chars, 0.4% apart) — so higher sustained concurrency is not shortening replies. Unique spans per run are ~3% lower with non-overlapping ranges (1065-1098 vs 1027-1056), but the persistent regression is negligible and SYMMETRIC: of 771 spans stable across all three serial runs, 8 are missing from all three parallel runs and 4 of those are genuinely no longer reported (0.5%, 799 chars), against 2 in the reverse direction; pooled union across runs is HIGHER for parallel (1,453 vs 1,401), which reads as more run-to-run variance rather than less capability. HONEST LIMIT: three runs per arm against a stage that reproduces only 64-72% of its own spans CANNOT resolve a 3% per-run effect. It is one-directional and so worth another three runs per arm before any later change leans on span counts from this stage. Tests: `test_one_sections_windows_do_not_wait_for_each_other` and `test_a_sections_windows_still_queue_when_they_exceed_the_bound` both FAIL against the old scheduler and pass against the new one (verified by stashing the source); the bound and the ordering guarantee have their own tests |
| first-pass window size 12,000 / 6,000 / 3,000 | **SWEPT, two runs per arm, interleaved, on one corpus (`zzgoal-29590211`, 1,790 chunks). Smaller windows roughly DOUBLE reproducible evidence at a FLAT cost per unit of evidence.** Held: cap 4,000, 40-item bound, 240 s timeout, concurrency 12, model, prompt, query, extraction rules, retry policy, cache off, and subdivision OFF in every arm. Coverage was held constant by raising `max_windows` per arm until no arm truncated — verified, 0 truncated sections and 297,926-298,080 covered chars in all six runs — because leaving it at the shipped 6 would have cut the 58,484-char section to 18,000 at a 3,000-char window and charged the loss to window size. Stable spans (found in BOTH runs of an arm) 951 -> 1,412 -> 2,065; of the 1,410 spans 3,000 adds, 1,216 cover text no baseline span overlaps by half or more, so the gain is new evidence and not re-cut quotes. Quoted-char union 55,793 -> 77,784 -> 99,507. Truncation COLLAPSES: 63.0% of windows at the token cap at 12,000, 46.5/52.1% at 6,000, **13.0-13.8% at 3,000** — the output cap stops being the binding ceiling. Reproducibility improves slightly too (79.3% -> 78.0% -> 82.6% of a run's spans survive into the other run). Cost per unit of evidence is FLAT: 165 / 164 / 149 completion tokens per stable span and 0.56 / 0.57 / 0.54 s per stable span; throughput 38-41 tok/s throughout. The raw wall time (534 / 804 / 1,108 s) is the serial-chain artifact above, not a cost of the window size. NOT YET SHIPPABLE, two reasons: (1) `v2_map_max_windows` was `Field(..., ge=1, le=10)` and the longest section took 22 windows at 3,000 chars and 11 at 6,000, so neither size was expressible in the shipped config — **the ceiling is now 32 (2026-10-29), default still 6**, which clears the measured 22; the sweep itself was never blocked, because it built `LLMSectionMapper` directly and never went through the field; (2) the regression below. Fix the scheduler first — a window-size change evaluated on today's serial map would be priced wrongly |
| smaller windows drop evidence the baseline reproducibly found | **MEASURED and SMALLER than it first appears, but real.** Against a regression set of the 951 spans found in BOTH 12,000-char runs, the 6,000 arm misses 274 in run 1, 230 in run 2 and 162 in both; the 3,000 arm misses 244, 255 and 203. The per-run figures are mostly the stage's own variability — the baseline itself only reproduces 79% of its spans between two runs — so only the both-runs figure is a persistent regression. Those were then put through the SAME containment test the gains face, because counting a loss whose text the arm quotes under different offsets would hold losses to a looser standard than gains: of the 162 at 6,000, 93 are re-cut and **69 are text genuinely no longer reported** (7.3%, 7,081 chars, 10 sections); of the 203 at 3,000, 144 are re-cut and **59 are genuine** (6.2%, 4,877 chars, 9 sections). The loss is NOT monotone in window size — 3,000 loses fewer than 6,000 in both absolute and relative terms — so this is not a window-size trend but per-window salience noise, and it is concentrated: the worst section accounts for 23 of 69 and 17 of 59. Reported apart from the gains and never netted against them. A windowing change must still answer for these before it ships. **2026-10-29: the mechanism is now classified and the answer is reassuring — zero of these losses are boundary or coverage effects; 44% are generation omissions and 16% the output cap, against 85% output-cap for the 932 spans going the other way. See the mechanism row below.** |
| item-level recall was being measured with an instrument that could not measure it | **CORRECTED.** The first comparison keyed an item as `section_id\|quote_key(statement)`. The model rewords a statement freely between runs, so two runs at the SAME cap agreed on item count to within 0.8% while sharing only 52% of their keys — the key tracked phrasing, and every cross-arm set difference it produced was noise. Identity is now the SOURCE SPAN: the quote's character offsets inside its section's text, taken in the same `quote_key`-normalised space the shipped quote rule uses and computed against the whole section rather than one window, so it survives window boundaries. The normalised statement is kept alongside, separately. Within-arm Jaccard rose to 0.61/0.71 on the same stage, and the arms separated cleanly. A measured number is only as good as the identity it counts |
| throwaway-tenant teardown misses MinIO | **LIFECYCLE DEFECT, open** — the teardown path drops the Qdrant collections, the OpenSearch index and the Postgres schema, but not the tenant's MinIO prefix (`rag/t_<slug>/`), so object bytes outlive a "torn down" throwaway tenant. Verified twice by hand (2 objects each time, removed under a `t_zzgoal` prefix assertion, all four namespaces then confirmed absent). The manual cleanup is NOT automation coverage — the fourth namespace is still missing from the product teardown path. The throwaway harness used for these experiments now purges it itself (prefix-asserted), verified on `zzgoal-b7a0b1f0`, but that is scratchpad scaffolding and closes nothing here |
| ~~the ~3% per-run span gap between the schedulers~~ | **DOES NOT REPLICATE, and the question was malformed.** Three more clean runs per arm, interleaved, on `zzgoal-64f256f6` (1,807 chunks, 35 sections, 46 windows), zero failed calls and peak in flight exactly 12 in all six. Mean unique spans per run: serial **1120.0**, parallel **1121.0** — 0.1% apart, sign reversed from the earlier 3%. Then the statistic itself was tested: pool the six runs, relabel into all ten possible 3/3 splits, recompute the identical measure (same stability rule, same containment test, same direction). The real arm split gives 6 serial-only spans and 0 the other way, the largest spread of the ten — but a MIXED split gives 5, and with ten splits the smallest attainable p-value is 0.10. Applying the same test to the earlier six runs settles it: there the real arm boundary produced a spread of 2 while a mixed split produced **17**. The statistic is dominated by run-to-run variation. The earlier row's "4 genuine spans lost vs 2 in reverse" should be read as noise — under one consistent rule those same six runs give 0 and 2. Latency on this corpus: serial 445 s mean (78% utilisation), parallel 372 s (87%), a smaller gap than the earlier corpus's -28% because 35 sections over 46 windows leaves the permit-per-section scheduler much less to waste |
| **the map stage SAMPLES at temperature 0.6, and that is the noise floor under every span measurement on this branch** | **MEASURED, and it is the dominant source of the irreproducibility above.** `RAG_LLM_REASONING_EFFORT=off` is set on this deployment, which makes `_eff_temp` rewrite a requested temperature of 0.0 to 0.6 — confirmed on the live resolved gateway: the map stage asks for 0.0, the gateway sends 0.6. Across the six runs only **35.6%** of the 1,725 distinct spans appear in all six and **18.4% appear in exactly one of six**; within one arm only ~51% are stable across its three runs. No span-counting comparison on this stage can see an effect smaller than that without many more runs than any of these experiments ran. CONTROLLED TEST, one variable: the same map prompts, five repeats each, on two windows that always stop at the output cap and two that never do, with the greedy arm patching `_eff_temp` on the gateway INSTANCE so only the temperature field of the request changes. Repeat-to-repeat Jaccard **0.24-0.94 nudged against 0.97-1.00 greedy**; item counts 22.2±7.8 / 6.2±4.4 / 12.0±5.5 nudged against 22.0±0.0 / 3.0±0.0 / 16.0±0.0 greedy. The nudge's own failure mode did NOT appear: all 20 greedy replies parsed and were usable, and the finish-reason split was identical in both arms (10 `stop` / 10 `length` each). THE TRADE-OFF IS REAL AND NOT ONE-SIDED: greedy's output is very nearly a SUBSET of what sampling finds across repeats (nudged union 43/31/4/17 against greedy 22/30/3/7, with greedy-only spans 0/0/0/1), so sampling explores more of a section across repeats while greedy returns one deterministic slice at comparable per-run yield. For a grounded-citation product that answers a question once, determinism is probably worth more than a union no single run ever sees — but that is a product decision and the nudge was added for a real observed failure, so it is NOT being changed here. It does mean every later span experiment should either run greedy or budget for a 64% noise floor |
| the ~60 spans smaller windows "lose" — what mechanism | **CLASSIFIED: generation omissions, NOT window boundaries and NOT coverage.** Each genuinely-lost span was placed against the losing arm's recomputed window boundaries in the same normalised space the span keys use, and sorted into coverage (past `window_chars * max_windows`), boundary (straddling a split), cap (its window's reply stopped at a ceiling), unusable reply, or omission (shown in full, finished cleanly, under both ceilings, simply not enumerated). Of the 45 spans stable at 12,000 and genuinely absent at 3,000: **20 omission (44%), 7 cap (16%), 18 unclassifiable** (their sections are not in the current corpus) — and **zero boundary, zero coverage**. The direction nobody asked about is far larger and has a different cause: of the 932 spans stable at 3,000 and genuinely absent at 12,000, **792 are cap (85%)**, 120 unclassifiable, 20 omission. So smaller windows do not lose evidence to their own boundaries; they gain evidence because the big window's reply runs out of output budget before it finishes enumerating. Read with the temperature row above, the 45 "losses" are mostly the stage's own sampling noise measured at two runs per arm, while the 932 gains are 20x larger than noise could explain and have a named mechanism |
| map failures were counted but not CLASSIFIED | **FIXED.** `MapReport` had `sections_failed` and nothing else, so a dashboard could see that six sections produced nothing and not whether the endpoint had timed out, gone away, rejected the request, or answered fine with JSON the parser refused — and those ask for opposite responses. The gateway has classified failures since the `ModelGatewayError` fix, but the mapper logged the classification per call and discarded it. `_attempt` now returns the kind alongside the parse, `_map_one` carries each window's FINAL kind (a window whose first call timed out and whose retry parsed is a success that cost an extra call, not a timeout), and `MapReport.failures` is the tally, ordered by count then name so it diffs between runs. A non-empty tally logs its own WARNING and rides the `rag_v2_map_reduce` payload as `rag_v2_map_failure_kinds`. Six tests, one per kind plus the recovered-on-retry case |
| a failed section was indistinguishable from an empty one in the answer | **FIXED.** `Coverage` is built from `sections_contributed`, so a section whose call died and a section that was read and held nothing relevant subtract from `used` IDENTICALLY — the reader could not tell "this document has nothing more to say" from "we did not manage to ask". During the scheduler benchmark a ~20-minute endpoint degradation lost 28 of 63 calls across 11 sections and the answer that came back read exactly like an answer over a thinner document. `Coverage.failed` is now threaded from `report.sections_failed` on both the answer and streaming paths, and `Coverage.line` — the sentence the reduce prompt is required to end with — says *"This answer is INCOMPLETE: N of the selected sections could not be read at all"*. An addition, not a replacement: the counts and the partial-read qualifier survive alongside it |
| bounded retries for failed map windows | **EVALUATED, NOT BUILT — and the evidence argues against it.** Every map failure observed on this branch has been endpoint-wide rather than per-call: 102 timeouts across two runs at the old 120 s limit, 40 of 64 calls in one degradation, 28 of 63 in another. In all three the endpoint was saturated or slow for everything at once, which is precisely the regime where retrying each failed window multiplies load on the thing that is already failing — and the stage would retry at concurrency 12, so a degradation would be met with a second full wave. The existing rule (retry an unusable REPLY once, never retry a raised call) is the right default for that failure mode: there is nothing to re-parse, and a second call to a failing endpoint is latency for a section the answer survives without. What would change the recommendation is evidence of a failure mode that is per-call rather than endpoint-wide — `MapReport.failures` is now the instrument that would show it, by distinguishing a scattering of `unusable_reply` from a wall of `read_timeout`. Revisit when that tally shows the former |
| first-pass window size, re-measured GREEDY on the fixed scheduler | **CONFIRMED under a deterministic generation policy, and the margins are LARGER than the sampled sweep found. 3,000 wins on evidence; it costs 1.81x the wall time.** Two runs per arm, interleaved 12,000 -> 6,000 -> 3,000 -> 12,000 -> 6,000 -> 3,000, on one corpus (`zzgoal-03fe3f41`, 1,804 chunks, 33 sections); section text byte-identical in all six runs (same `chars`/`norm_chars`/sha digest per section). Held: cap 4,000, 40-item bound, concurrency 12, 240 s timeout, shipped scheduler, model, prompt, query, extraction rules, retry policy, cache off, subdivision off. `max_windows` was raised per arm (6 / 12 / 24) so coverage never varied: **0 truncated sections, 0 failed calls, 0 retries, 100% parse success and 99.94-99.99% of section characters covered in every run**, which is the control the whole comparison rests on. GREEDY, measurement only: `_eff_temp` patched on the resolved gateway INSTANCE in-process so the request's temperature field is the only change; the shipped 0.6 nudge and the production config are untouched. RESULT, stable spans (found in BOTH runs of an arm) **1,149 -> 1,714 -> 2,394**, stable quoted characters 62,168 -> 102,452 -> 119,588. Against the 12,000 baseline and put through the same symmetric containment test the earlier sweep used, 6,000 adds **631** spans covering text no baseline span overlaps by half or more (759 before containment) and 3,000 adds **1,173** (1,441 raw). The marginal step 6,000 -> 3,000 is still real: **518 genuine gains** (983 raw) for 30% more tokens. Truncation is the mechanism and it collapses on cue: **67.0% -> 47.8% -> 15.5%** of windows stopping at the 4,000-token cap. COST, stated apart from the gain: completion tokens 152.6k -> 228.1k -> 296.4k (1.94x) and map wall time **363 s -> 547 s -> 656 s (1.81x)**; cost per unit of evidence is flat to slightly better (133 / 133 / 124 tokens and 0.32 / 0.32 / 0.27 s per stable span) and throughput held at 36-40 tok/s in all six runs, so the extra wall time is extra requests, not a slower endpoint. The scheduler fix is visible here and settles the old artifact: every run finished within **7-11% of its own concurrency floor** (busy/12) against 56-57% before, and at 3,000 the longest single section's sequential chain is 1,167 s against a 650 s wall -- the chain no longer bounds the stage, so the smaller window's latency is now a real cost rather than a scheduling artifact. REGRESSION, reported apart and never netted: **37** baseline-stable spans (1,856 chars, 8 sections) are genuinely absent from both 6,000 runs and **46** (3,384 chars, 12 sections) from both 3,000 runs, worst section 8 and 10 of them -- the 58,484-char `16.1 HCI-2` section in both. Greedy shrinks this to roughly half what the sampled sweep reported (69 and 59) and makes it harder to dismiss: these losses survive at 96.6-99.0% within-arm reproducibility, so they are not sampling noise and a windowing change still has to answer for them. HONEST LIMITS: (1) greedy did NOT make the stage deterministic -- 97.1% / 96.6% / 99.0% of a run's spans survive into its pair, against 78-83% sampled, so ~3% residual variation remains at 12,000 and any effect smaller than that is still invisible at two runs; (2) the arms do not contradict each other and no arm needed more runs -- the between-arm gaps (max 1,197 at 12,000 against min 1,772 at 6,000) are an order of magnitude wider than the within-arm spread; (3) **this measures which windowing strategy is better under a deterministic policy, nothing more.** It does NOT establish that these margins, this completeness or this latency hold at the shipped temperature 0.6. That is a separate evaluation and it has to run before the production default moves. Production `v2_map_section_chars` stays at 12,000 and `v2_map_max_windows` at 6 pending it |
| the same window-size question at the SHIPPED temperature | **ANSWERED, and it does not overturn the greedy result -- it shrinks the margins and recommends a different winner. 3,000 still finds the most evidence; 6,000 is the candidate, because it buys 60% of the gain for a quarter of the latency.** Eighteen runs, six per arm, interleaved as six full cycles 12,000 -> 6,000 -> 3,000 so a drifting endpoint lands on all three arms equally. Six runs and not two because the shipped temperature is the independent variable here and only ~66% of a run's spans survive into another run of the same arm. CONFIGURATION VERIFIED ON THE WIRE, not inferred from config: `_client.post` was tapped and the request bodies tallied by stage -- `map` 1,251 requests and `reduce` 411, **every one of them `temperature=0.6, max_tokens=4000`**, reconciled exactly against the gateway's own call records per stage (map 1,251/1,251, reduce 411/411, untagged 0). Nothing was patched; this is what production sends. CORPUS VERIFIED BEFORE AND AFTER: digest `05706e56de48754b`, 28 sections, 264,277 characters, identical at both ends and byte-identical across all 18 runs. RESULT, spans reproducible in >=4 of 6 runs: **976 -> 1,484 -> 1,864** (quoted characters 56,806 -> 87,329 -> 107,272); found in all six: 611 -> 996 -> 1,425. Against the 12,000 baseline through the same symmetric containment test, 6,000 adds **506** genuine spans (34,607 chars) and 3,000 adds **838** (49,882); the marginal 6,000 -> 3,000 step is still real at **324** (15,720 chars). Truncation is still the mechanism: windows stopping at the 4,000-token cap **63.3% -> 49.5% -> 8.2%**. THE SIX RUNS ARE ENOUGH, and this was tested rather than assumed: the unions were still climbing at run six (+2.4% / +4.4% / +3.1%), so the margin was re-derived from the first k runs of every arm with the stability threshold held at the same proportion of k. The gain is **flat in k with no trend** -- 6,000 scores 552 / 517 / 488 / 506 at k = 3 / 4 / 5 / 6 and 3,000 scores 895 / 824 / 752 / 838, a spread of under 10% with the ranking never changing. The union is not exhausted and the DECISION does not depend on it. RELIABILITY, four measures kept apart: 0 sections with no usable window, 0 coverage-truncated sections and 168/168 sections contributing in every arm; 3 of 237 map calls failed at 12,000 (all three in one run, three distinct sections, each retried successfully, **0 terminal failures**, `MapReport.failures` empty) and 0 of 366 and 0 of 648 elsewhere. Items dropped for an unquotable quote 7.3% / 5.3% / 5.1%; over the 40-item window bound 0.7% / 0.5% / 0.7%. **The 63.3% output-cap rate is not a completeness score** -- those windows all parsed and contributed; what they cost is answered only by the span sets above. COST, and the old multiplier is RETIRED: map wall 417 s -> 466 s -> 578 s (**1.12x / 1.38x**, not the greedy sweep's 1.81x) and completion tokens 134.2k -> 200.4k -> 250.1k (1.49x / 1.86x). The ratio fell mostly because the 12,000 BASELINE got slower and much noisier at 0.6 (329-582 s against a greedy 363 s), not because 3,000 got faster, so quote the absolute numbers rather than any multiplier off that denominator. **The reduce stage is NOT window-size-invariant** and the earlier reports implied it was: reduce requests 18.0 -> 22.8 -> 27.7, reduce wall 67.9 s -> 101.6 s -> 114.9 s (**1.50x / 1.69x**, steeper than the map), and the answer itself grows 75.7k -> 119.4k -> 140.3k characters. Route..reduce totals 488 s -> 570 s -> 695 s (1.17x / 1.43x). Cost per unit of evidence is flat (138 / 135 / 134 tokens per stable span) and the scheduler is still holding -- every arm finished within 8-13% of its own concurrency floor. TEMPERATURE, isolated on the 27 sections both corpora hold byte-identically (246,161 chars, 85.5% of the greedy corpus and 93.1% of this one; unions and cores are NOT compared across experiments because the arms differ 2 runs against 6 and both grow with depth): sampling does not change HOW MUCH evidence one run finds (-0.3% / +1.2% / -3.9% spans per run) -- **it changes WHICH**. Mean pairwise agreement collapses from 0.943 / 0.931 / 0.978 to 0.665 / 0.704 / 0.754, about 0.25 of Jaccard, while the average 2-run union RISES 8.5-16.5%. One production request is far less reproducible than the greedy measurement suggested, and the 2-run arms of that sweep were reading a much narrower slice of the variation than their agreement implied. HONEST LIMITS: (1) the retrieval-driven selection shifted between the two experiments -- this corpus is 28 sections, the greedy one was 33, 27 shared and byte-identical, and the **6 dropped sections (41,875 chars, including the 17,804-char `21. Switches`) are UNMEASURED at the shipped temperature**, however decisive the result is on the 27; (2) this measures RETRIEVAL EVIDENCE and the route..reduce wall, **not production answer quality and not request latency** -- faithfulness, guardrails, citation rendering and audit are outside it and are not window-size-invariant either; (3) the 3,000 answer is 1.85x longer, which is a DIFFERENT answer and not merely a better-evidenced one -- more evidence into the reduce is not automatically a better answer out of it, and nothing here measured that. So 6,000/12 is nominated, not shipped: **production `v2_map_section_chars` stays at 12,000 and `v2_map_max_windows` at 6** until the complete production path is measured on the candidate. |
| **Phases 8 and 9 were UNREACHABLE from the query API**, under every flag and every phrasing | **FOUND AND FIXED (the reachability, not the enablement).** The scorecard has said for weeks that "v2 needs `doc_ids`" and treated that as a caller's responsibility. It was not one: `QueryRequest` had **no `doc_ids` field**, so no API caller could express a document scope at all. `filters={"doc_id": ...}` is refused 422 with the hint *"Use `doc_ids` for a document scope"* — naming a field that did not exist. And `_route_v2` cannot infer a scope, for the reason its own docstring gives: the plan wants "top docs by doc-level retrieval" and **the `doc` chunk level has no producer (§8.17)**. So `_select_sections` always logged `rag_v2_section_pass_skipped: no document in scope`, Phase 9 never ran, and **switching `v2_generate` on would have changed nothing**. Verified against four phrasings including one naming the file exactly (`give me the compliance details from Corrigendum-5.pdf`) — `doc_ids=[]` every time. Every earlier measurement on this branch reached Phase 9 only by constructing `RetrievalQuery(..., doc_ids=(...))` IN-PROCESS, which is not a thing any caller can do; that is why a bespoke harness saw a stage the product could not. **A SECOND interceptor sits above it**: with `RAG_QUERY_PLANNER_ENABLED=true` (set in the live deployment) `_execute_query_plan` resolves an aggregate question to `whole_document`, answers it and RETURNS at `pipeline.py:6096` — above `_route_v2` (6143), `_select_sections` (6153), `_map_reduce_answer` (6332) **and `_verify_answer` (6370)**. Four phases skipped, so a planner-routed answer is never Phase-10 verified either. FIX: `doc_ids` on `QueryRequest`, optional and default-empty so every existing caller is byte-identical, threaded through all three handlers that share it (query, query/stream, retrieve). It is NOT an authorisation channel — it narrows a search already bound to the KB namespace and already ACL- and clearance-filtered, and can never widen one — but `_assert_doc_scope_visible` adds a clean 422 instead of a silently empty answer that reads exactly like the document not saying it, and **no existence oracle**: the check is against documents the caller may SEE, so an unknown id and a Restricted one they cannot read fail identically, because telling them apart would let an unprivileged caller enumerate classified titles. Five e2e tests. **This is reachability, NOT a decision to enable Phase 9**, which stays off by default |
| the whole-document read DISCARDS flagged evidence silently | **GOVERNANCE DEFECT, reproduced and recorded as `xfail(strict=True)`; fix approved, not yet built.** The incumbent answers an aggregate question by batching the document, running each batch through the unchanged grounded path and assembling the GROUNDED ones. Per-batch faithfulness works correctly — 19 evaluations, 19 `FAITHFULNESS` audit events on a real run. The COMPOSITE is the problem. Four `continue` branches (`pipeline.py:3843, 4079, 4293, 4486`) drop any batch that is not GROUNDED, and `flag_for_review` (`grounding.py:828`) only CONSTRUCTS a flagged answer — persistence lives at four `hitl_service().create()` sites, all on the normal `answer_query` path AFTER generation returns, which the whole-document read never reaches. Forced-flag result, configured to match the live container: **4 of 8 batches `flagged_for_review` at 0.6511, zero review entries queued, composite delivered as plainly `grounded` with `review_id=None` and `faithfulness_score=None`.** So the delivered answer is the SURVIVING SUBSET of grounded batches, presented as a grounded answer: the evidence that failed grounding is excluded, the reader is not told, and no reviewer will ever see it. The coverage loop — ON in production — makes it worse rather than better, re-reading the uncited chunks and re-flagging the same content each pass (1 flagged batch became 4). Exclusion and partial-coverage accounting DO work and are asserted positively, so the two defects are isolated. GETTING THE TEST RIGHT TOOK THREE TRIES AND THE FAILURES ARE INSTRUCTIVE: a stub whose extract shares no wording with its chunks scores near zero and ABSTAINS — a different outcome with different and CORRECT handling, since there is nothing to review — so the flag has to be earned by echoing real sentences and appending one unsupported claim; `Settings` is frozen, so overrides go through the env and `get_settings.cache_clear()`; and `completeness_guard_enabled` defaults OFF in code but is `true` in the live container, so left unset `completeness is None` reads as a finding when it is only the harness not reproducing production. APPROVED FIX, in order: persist every flagged batch through the real `hitl_service().create()` before discarding it (linked by the composite's `trace_id` — no new table needed) with three new nullable `HitlReview` fields (`faithfulness_score`, `faithfulness_threshold`, `model_version`); carry `FLAGGED_FOR_REVIEW` onto the composite rather than adding a fourth `AnswerStatus` (the console keys an EXHAUSTIVE `Record<AnswerStatus, ...>` at `ui.tsx:59` and both SDKs type the three values, so a new value is a coordinated release for an outcome the existing status already expresses); a DETERMINISTIC withholding disclosure, never model-generated, that says "pending review" only when a review really persisted; and flagged evidence made TERMINAL for the coverage loop so it cannot re-enter the uncited set. Abstained batches must never create a review — that is a correct outcome with nothing to review |
| v2 needs `doc_ids` | an unscoped question skips Phases 8-9 and is served by the whole-document read, itself capped at 400 of 1,793 passages. Now LOGGED (`rag_v2_section_pass_skipped`) rather than silent. Making it automatic for a single-document KB was tried and REVERTED: `test_an_unscoped_aggregate_runs_no_section_pass` encodes the opposite contract on latency grounds, and at 766 s that trade is worse now than when it was written — a product decision, not a bug |
| Phase 3 facet coverage 0.54 (bar 0.80) | 0.38 -> 0.54 by giving the profiler its whole evidence budget instead of 2 pages. The rest needs the themes whose evidence sits past any front-loaded window (`training` at char 61,842) — and SAMPLING measured worse (0.44), so that route is closed |
| ~~`LLMDocProfiler` gives up after 1 retry~~ | **CLOSED** — `doc_profile_retries` (default 4), early-exit on success. Failures are clustered, not ~50%; the old cap rested on a temperature premise `_eff_temp` invalidates |
| C03 / the SOC-NOC lookup | ingest the main tender NIT alongside the corrigendum — the clause is not in the corpus |
| the two latency targets | still **[ASK: confirm]** in the plan and never confirmed with the owner |

## Three measurement defects found while preparing the articulation comparison (2026-10-31)

None of these changed an answer. All three changed what we believed about answers, and two of
them had been silently shaping conclusions already recorded above.

### The whole-aggregate retry was a second load, not a retry

`prod_run._ask` allowed four attempts at the aggregate question with a 900 s client read
timeout. A candidate map-reduce pass on this corpus takes **1,036-1,173 s**. So every attempt
was abandoned client-side — and a client that stops listening does not cancel the handler. The
server ran each pass to completion while the next attempt started on top of it:

    attempt 1   07:36:24 -> 07:53:40   1,036 s   finished and verified; client left at ~+900 s
    attempt 2   07:53:39 -> 08:13:12   1,173 s   finished and verified; client left at ~+900 s
    attempt 3   08:06:34 -> (killed)             started while attempt 2 still had 6.6 min to run
    attempt 4   08:21:43 -> (killed)

**6.6 minutes of two full map-reduce passes running concurrently**, at map concurrency 12 each:
24 in-flight calls against an endpoint configured for 12. No 5xx, no exception, no server fault
anywhere in the log — the server answered correctly all four times and nobody was listening.

This is very likely the mechanism behind Gate 1's **219 map read timeouts, all on the candidate
arm**, which were recorded as a reliability property of Phase 9. Gate 1's candidate averaged
870 s, just inside the 900 s budget, so it usually escaped; on this corpus it never does. The
arm-correlation that made it look like a Phase-9 characteristic follows from the arm being the
slow one, not from the architecture. **Gate 1's reliability figures should be re-read with that
in mind, and the 2/6 failed candidate runs are not safe to attribute to Phase 9.**

Fixed in the harness, not the platform: the aggregate gets ONE attempt and its outcome is
preserved as whatever it was, lookups keep their bounded retry (seconds each, where a transient
503 is noise rather than a result), and the client's patience is raised past the observed pass
time. Nothing about the system under test changed — only how long the client waits and how many
times it asks. No new retry mechanism was introduced; the per-window behaviour inside the map is
untouched and stays separately instrumented.

### Verification, not the map, is where the candidate's latency goes

From the same two completed passes, the split had never been measured:

    map phase        390 s / 445 s     38% of wall
    verification     646 s / 728 s     62% of wall

So the headline "map latency 466 s -> 870 s (1.87x)" is mostly **Phase-10 verification of the
synthesised answer**, which the incumbent does not do at all — it verifies its batches and never
the aggregate it assembles from them. Comparing the two arms' wall times compares one arm that
pays for aggregate verification against one that does not. The latency blocker is real and it is
a different blocker from the one it was recorded as.

### Terminal citation-marker coverage was understated by a punctuation assumption

`articulation`'s cited-line test anchors on `\[\d+\]\s*$`, requiring the marker to be the last
character on the line. The model writes `[99].` — the full stop AFTER the bracket, where
`build_prompt`'s own example puts it before. On the incumbent's own answer, 409 extractor lines:

    marker flush at line end     252   61.6%   <- what was recorded
    punctuation-aware            360   88.0%   <- corrected
    marker absent                 49   12.0%   (32 organiser lines ending ':', 17 preamble/other)

**26.4 points recovered by correcting the matching rule**, and the two causes that would have
been worse are both absent: no marker was lost in markdown processing (the stored text is the raw
answer), and none was stranded mid-line (tolerating trailing punctuation takes 252 to exactly the
360 that carry a marker at all). Third time this corpus has produced this defect class, after
`fold_text` folding markdown bold and `contains_cue`'s substring boundaries.

Corrected matcher: `\[\d+\]\s*[.!?…]?\s*$` — ONE optional sentence-final mark, so `[99].` counts
and `[99], and see below` still does not. Applied in the REPORTER, not the harness: the harness
computes shape at run time, so changing it mid-run would score early cycles by one rule and later
cycles by another, and deriving it in the reporter applies one rule to every arm and every stored
cycle. The strict figure is kept beside the corrected one.

**Gate 1's 69.7% cannot be corrected and is not comparable to the 88.0%.** Those artifacts
predate the harness storing answer text, so there is nothing to recompute from. It stands as the
as-run measurement it was. The harness now keeps the answer text, because a derived metric cannot
answer a question thought of afterwards — this correction was only possible at all because the
question was asked on a corpus where the text had been kept.

It says nothing about citation CORRECTNESS. Whether `[99]` supports the claim it sits after is
faithfulness, and for the incumbent aggregate that remains unknown.

## Origin attribution must precede quality scoring (2026-10-31)

Six defects found in one afternoon, all the same shape: a check or a metric read an answer as one
undifferentiated text stream, when an answer actually contains three kinds of statement with
three different provenances.

    model_assertion              the model wrote it from the evidence
    document_provenance          a reference the DOCUMENT supplies (a clause number)
    platform_generated_metadata  the PLATFORM wrote it (a coverage line, a disclosure footnote)

Scoring the second and third as if they were the first invents defects that are not there, and it
does so ASYMMETRICALLY — Phase 9 emits more clause references and a longer disclosure footnote
than the incumbent, so every one of these biased the comparison against the arm being evaluated.

### The register

| Defect | Class | Evidence | Correction |
|---|---|---|---|
| Numeric guard flags clause references and the coverage line as invented figures | verification scope | 40 occurrences of 18 flagged figures classified: **32 clause refs, 6 coverage line, 2 model assertions, ZERO hallucinated numbers** | validate platform metadata against pipeline records, document references against their source mapping |
| Amendment old-value flagged although the document states it | evidence lineage | `Rs. 35 Crores → Rs. 19.44 Crores`; Phase 9 passes `item.quote` into numeric verification, not the supporting chunk, so the evidence base is narrower than the task | pass sufficient source context; do NOT exempt amendment numbers |
| Disclosure footnote counted as model prose | evaluation attribution | 111 of 763 lines (14.5%) are Phase 10's `_Could not be verified…_` footnote, 0 citations by design, dragging the body's **79.8% to a reported 68.2%** | split on the emitter's own constant, report body / disclosure / rendered separately |
| `extract_values` reads raw text while `contains_cue` reads folded text | grader, formatting-blind | `'15 crores'` → `qty:15 crore`; `'₹**15** crores'` → nothing. The entire 58.8% → 52.9% value-accuracy "regression" is this one entry | fold before extracting, then re-score every stored artifact |
| `GROUNDED` cannot carry what a check found | finding propagation | `unsupported_claims` is set only inside `flag_for_review`; `ground()` has no such parameter, so a grounded answer's list is empty by construction | structured verification findings on the answer, independent of `AnswerStatus` |
| `faithful=False` has no routing effect | decision consistency | `route_faithfulness` reads the SCORE; the numeric guard's `faithful=False` is documented as a "latent no-teeth property" and fires on real answers | a failed check must not be silently outranked by a passing aggregate score |

### Why `extract_values` is the one worth dwelling on

`fold_text` was given markdown-folding precisely so a bolded figure would still match — that fix
is recorded earlier in this document. Its sibling on the SAME LINE of the SAME function kept
reading raw text:

```python
covered       = all(contains_cue(folded, entry) for entry in item.must_contain)   # folded
answer_values = extract_values(raw_answer)                                        # raw
```

So the fix landed in one of two call sites and the other kept the bug. And because the
presentation contract this branch just shipped says "put EVERY number in **bold**", the defect
only became visible once an arm started OBEYING that rule — **any** arm that obeys it is
penalised, the incumbent's 58.8% included, with its 622 bold spans.

### A sixth category the classifier needed

Classifying value misses required one category beyond formatting / scale / rounding / absent:
an IDENTIFIER. `UL 62368` is a name, not 62,368 of anything, so numeric tolerance is meaningless
on it — applied anyway, the first version of the classifier called `IEC 61000-4-5` a "rounding
difference". Identifier misses are a CONTENT finding, and on the first paired cycle they run
**7 for the incumbent against 6 for both Phase 9 arms**: the certification omission Gate 1 found,
reproducing. `TPM 2.0` and `IEC 62443-4-2` are missing from all three arms and are not a Phase 9
win.

### What this costs to ignore

Improving the platform's observability made its answers look LESS trustworthy, by flagging its
own metadata as hallucination and its own disclosure as uncited prose. Every factual-looking
statement in an answer needs a known origin before anything scores it.

## The shared presentation contract, measured over three paired cycles (2026-10-31)

Nine runs: three arms interleaved in rotating order, three cycles, the aggregate question only,
one fresh corpus (`zzgoal-9bdbfe1e`, PRESERVED as the reference dataset). The three arms are the
deployed whole-document route, the Phase-9 route at `86c6cf6` (before the contract) and the same
route at `ae85208` (after). One variable between the last two: `mapreduce.py` and
`prompts/reduce.md`, restored from git per arm with both files' md5 recorded on each run line.
`pipeline.py` sat at HEAD for every arm, so the citation-marker fix could not become a second
treatment.

**THE CONTROL HELD, TWICE OVER.** All six candidate runs selected the IDENTICAL 27 sections —
byte-identical id sets from `query_log.selected_sections`, matching the post-run corpus digest
`effe9090db5c7ee0`. And the incumbent scored 0.45 must / 0.50 should in all three cycles. So
retrieval was constant and the control did not move: every difference below is GENERATION.

### Per cycle, nothing averaged

| | incumbent | before (`86c6cf6`) | after (`ae85208`) |
|---|---|---|---|
| must-tier | 0.45 · 0.45 · 0.45 | 0.60 · 0.65 · 0.65 | 0.65 · 0.60 · 0.70 |
| should-tier | 0.50 · 0.50 · 0.50 | 0.70 · 0.70 · 0.70 | 0.70 · 0.70 · 0.70 |
| **bold spans** (body) | 622 · 631 · 634 | **0 · 0 · 0** | **885 · 899 · 837** |
| bold per 100 body lines | 144 · 147 · 153 | 0 | 90 · 94 · 88 |
| terminal markers (body) | 82.9% · 84.1% · 85.8% | 79.8% · 79.8% · 81.3% | 76.7% · 80.3% · 80.3% |
| duplicate lines (body) | 0 · 0 · 0 | 3 · 4 · 2 | 6 · 4 · 3 |
| faithfulness | — · — · — | 0.954 · 0.953 · 0.951 | 0.968 · 0.975 · 0.972 |
| citations | 336 · 342 · 338 | 660 · 709 · 715 | 780 · 753 · 777 |
| latency s | 502 · 480 · 417 | 1,042 · 1,229 · 1,118 | 1,385 · 1,172 · 1,093 |

### What the contract did

**It restored the missing presentation behaviour.** Bold was 0 in every before-run and 837-899 in
every after-run; mean sentence length 15.8 -> 10.7 words; prose share 0.3% -> 0.2%. Terminal
citation coverage essentially flat (80.2% -> 79.1% on the body), so placement cost nothing.

**It did NOT improve completeness, and that claim is withdrawn.** must-tier 63.3% -> 65.0% is
inside the before-arm's own 0.60-0.65 spread, and the arms SWAPPED between cycles 1 and 2 (0.60/
0.65 then 0.65/0.60). Missed gold items are 13 on both. A single-cycle reading of this said
"+5 points" and was wrong.

**PARITY IS NOT DEMONSTRATED.** Bold density is 90 per 100 lines against the incumbent's 148, and
sentence length 10.7 against 8.9. The missing behaviour is restored, not matched. Chasing the
incumbent's density with more emphasis instructions is explicitly NOT the follow-up — excessive
emphasis is harder to scan, and nothing here says 148 is the right number.

**LATENCY IS NOT PROVEN IDENTICAL.** The +33% claim from cycle 1 is withdrawn: 1,385 · 1,172 ·
1,093 overlaps the before-arm's 1,042-1,229. But the first after-run EXCEEDS the before-arm's
observed maximum, and three observations do not establish zero cost. The absolute figures are in
the table for exactly that reason.

**The redundancy regression is withdrawn too.** 6 · 4 · 3 after against 3 · 4 · 2 before.

### The value-accuracy "regression" was the grader

58.8% -> 51.0% is not a content loss. Classified over three cycles:

| | matched | formatting-only | identifier absent | value absent |
|---|---|---|---|---|
| incumbent | 30 | 0 | 21 | 0 |
| before | 30 | 0 | 18 | 3 |
| after | 26 | **3** | 19 | 3 |

**RE-SCORED UNIFORMLY AFTER THE FIX, every arm scores 0.5882 in every cycle.** The regression was
entirely the grader. The recovered entries are C04 in all three cycles (`₹**15** crores`, written
correctly and per the contract) plus C40 in cycle 2 — and C40 is the more interesting one. It was
first classified as an absent identifier and written off as generation variance; it was the same
defect hitting an IDENTIFIER rather than a quantity. The answer wrote `TPM **2.0**`, bolding the
version number because the contract says to bold every number, and that splits the `std:` key:

    'TPM 2.0'      -> ['std:tpm 2.0']
    'TPM **2.0**'  -> ['num:2']          the standard is gone, its digits loose
    stripped       -> ['std:tpm 2.0']

So "bold every number" damages typed identifier extraction in a third way, beyond separating a
number from its unit and beyond `ISO **27001**`. The deltas: shared c1 +0.0588, c2 +0.1176, c3
+0.0588; incumbent and before-arm +0.0000 in all three, because neither hit a case the fix
touches. The gold was not relaxed.

### Two conclusions that must stay separate

**Architecture:** Phase 9 is +18-20 points must-tier and +20 points should-tier against an
incumbent that reproduced at 0.45/0.50 three times out of three, with 13 missed items against 21
and roughly double the citations. **This is the more important product finding and it owes nothing
to the presentation contract.**

**Presentation:** the shared contract restores articulation with no demonstrated penalty to
content, citations, redundancy or latency. **It is not evidence of better retrieval or greater
factual completeness**, and must never be reported as though it were.

### The validation baseline this was measured on

`make ci` (ruff over the whole repo, mypy strict 524 files, 2,799 unit) · 394 e2e · 107
integration against live stores. Two tests SKIPPED and not executed, named so they are not
mistaken for passes: `test_phase1_composition` (wants `RAG_TEST_REAL_STORES=true`) and
`test_reranker`'s remote probe (no reranker serving at the configured host).

A green gate is repository quality, NOT production readiness. Outstanding before Phase 9 is
exposed to users: verification-provenance wiring (9 strict xfail), provenance-aware numeric
verification, supersession correctness (0% on both arms and untouched by any of this), aggregate
retry and incomplete-result handling, and full production-path reliability and latency validation.
(Supersession is no longer "untouched" and has now been re-scored end-to-end: see the two
sections below. What remains is that `supersession_ok` measures the 3 GOLD amendments, while the
document supports at least 5.)

## Supersession: one symptom, four defects, three of them outside generation (2026-11-01)

The line above said supersession was "0% on both arms and untouched by any of this". That
reading was itself one of the defects. Traced stage by stage on the corrigendum's three
amendments — MAAT (a numeric replacement), D.3 (a withdrawal) and BDS (a ~200-character
requirement replacing another ~200-character requirement) — the single 0% turned out to be four
independent faults, and **only one of them was the model omitting anything**.

| defect | layer | status |
|---|---|---|
| the metric could not recognise a correct answer | measurement | corrected |
| the PII masker destroyed clause identifiers | ingestion | fixed, corpus-validated |
| the heading heuristic promoted prose | ingestion | fixed, calibrated on 18 documents / 3 families |
| D.3 never reached the model | extraction | fixed, verified end-to-end |
| BDS reached the model and was dropped in reduction | generation | fixed deterministically |

### The metric was measuring the wrong thing twice

`_supersession_scores` matched on OLD values only and compared raw strings. So an answer that
correctly reported a change scored zero if it named the change by its NEW value, and
`**GTP**/Data Sheet` did not match `GTP/Data Sheet` because of the emphasis the presentation
contract asks for. **The tracker was bold-blind and keyed only on superseded values — the exact
defect fixed in `_checklist_scores` that morning, one layer up, found in my own measurement
code after I had already published numbers from it.** Rewritten to value-identity matching with
pair proximity, it reports what was actually true.

**THE FIRST READING OF THAT CORRECTED METRIC WAS ITSELF WRONG, and this paragraph said so for a
day.** It recorded "the incumbent leaves all three amendments unmarked, Phase 9 reports MAAT".
That was measured on the corpus ingested BEFORE the heading fix, where the sentence carrying
D.3's withdrawal had been split in two. On a corpus ingested after it, the incumbent reports
D.3 in both cycles, in a cited single statement. So D.3's incumbent failure was an INGESTION
defect recorded as a route difference -- the exact misattribution this section exists to
correct, committed inside the correction. See the end-to-end run below for the measured
per-amendment outcomes.

### Two ingestion defects destroyed evidence before any model saw it

The PII masker's IP-address detector consumed dotted clause numbers (`4.3.1`), so the clause
identifier an amendment is cited BY was gone from the chunk text. Narrowly scoped with a
context validator rather than by weakening IP redaction.

The heading classifier promoted numbered PROSE to a heading, which split the sentence carrying
D.3's withdrawal in two. The unconditional sentence-terminator rule that would have fixed it
rejects a genuine heading (`4. NG-SIEM ... License Details.`), which a hand-written classifier
structurally cannot see — so the rule was adopted only as a combined discriminator, after a
counterfactual probe over every corpus in `eval/gold/`: **18 documents, 3 families, 1,359
candidates, 190 rejections, 190/190 adjudicated prose, exemption population 1.**

### BDS: the deterministic fix, and the evidence that it is deterministic

The reduce call is handed `AMENDMENT: <old> -> <new>` verbatim and rule 3 tells it to show the
change. For the two amendments whose sides are SHORT it does. For the one whose sides are two
~200-character competing requirements it reports the replacement and silently drops what it
replaced — **in both attempts**.

So `amendment_lines` builds the change log from the items, in code, and `reduce.md` was NOT
changed. The regression evidence is the comparison that isolates it:

| | attempts 1-2 (before) | attempt 3 (after) |
|---|---|---|
| model's reply | 787 chars | **787 chars, byte-identical** |
| BDS superseded side present in that reply | no | **no** |
| MAAT relationship in the assembled answer | yes | yes |
| D.3 relationship in the assembled answer | yes | yes |
| BDS relationship in the assembled answer | **no** | **yes** |

The model's output did not change. The answer did. That is the whole claim, and it is why this
is not a prompt result: nothing here depends on sampling, on rule 3, or on the model handling
long amendments. Preserve this comparison — it is the only evidence that distinguishes a
deterministic fix from a lucky generation.

### What the renderer may and may not say

An item with no `old_value` is reported as **"the previous wording is not in the extracted
evidence"**, never "previously unstated". Those are different claims: the second is about the
SOURCE document, and an empty field does not establish it — the map records a missing side both
when the document never stated one and when extraction missed it, and the renderer cannot tell
those apart. Asserting the stronger one would be the deterministic path inventing history while
preventing the model from inventing it. Neither replacement nor disposition is still not
reported at all, since "amended" alone tells a reader a change exists while withholding what it
was.

### Measured end to end: the renderer rescues ONE amendment, not supersession (2026-11-02)

`prod_run` through the complete production path -- retrieval, map, reduce, grounding, the output
guardrail, FR-E4 routing, Phase-10 verification, citation assembly -- scored by the corrected
`svc_guardrails_eval.suite`, two arms, two cycles each.

**CONFIGURATION AND PROVENANCE.** Corpus `zzgoal-cf3b136d`, source bytes `3dc59d958833024d`,
ingested 2026-11-01 14:42 IST, 53 sections / 1,771 chunks / 362,293 chunk characters. Code at
`d093855`, clean tree, renderer / reduce prompt / scorer md5 `6c30f36b` / `12778eb1` /
`a26a1436`, unchanged across all four cycles. Aggregate intent only, caches off, map concurrency
12, 4,000 output tokens, 12,000-character windows, 6 windows -- production defaults. Candidate
arm: `RAG_V2_ROUTER/RETRIEVAL/GENERATE=true`, planner and intent routing off, document
scope `corrigendum-5.pdf`. Incumbent arm: planner and intent routing on, no scope, v2 unset.

**NOT THE PRESERVED REFERENCE CORPUS, deliberately.** `zzgoal-9bdbfe1e` predates the PII fix
(`11cfe4e`, 13:30) and the heading fix (`e948b41`, 14:39) and still carries one masked clause
identifier. All three corpora share IDENTICAL source bytes, so they differ only by the ingestion
code that processed them; absolute scores are NOT comparable across them and the reference corpus
remains the control for the articulation comparison it was preserved for.

| per amendment | Phase 9 c1 | Phase 9 c2 | incumbent c1 | incumbent c2 |
|---|---|---|---|---|
| MAAT | reported | reported | unmarked | unmarked |
| BDS ITB 1.3.9.1 item 5 | reported | reported | unmarked | unmarked |
| D.3 | reported | reported | **reported** | **reported** |
| `supersession_ok` | **True** | **True** | False | False |

Every `reported` verdict is backed by a STATEMENT-LEVEL check, not a character distance: both
values in one line of the answer, old side first, that line carrying its own citation markers.
An earlier version of this measurement printed `gap=0` for MAAT, which was an artifact -- the
scorer's +/-40-character window can satisfy both values at one digit position -- and a
precise-looking zero is worse than no measurement.

**THE ABLATION IS THE RESULT.** The candidate answer re-scored with only the deterministic
`### amendments in this document` block removed, everything else byte-identical, no second
generation:

| | MAAT | BDS | D.3 | `supersession_ok` |
|---|---|---|---|---|
| Phase 9, full | reported | reported | reported | True |
| Phase 9, block ablated (c1, -2,117 chars) | reported | **unmarked** | reported | **False** |
| Phase 9, block ablated (c2, -1,772 chars) | reported | **unmarked** | reported | **False** |

**The model-authored prose independently reports MAAT and D.3. The deterministic renderer
rescues exactly BDS -- the long-form case -- and nothing else.** So the defensible claim is
narrow: MAAT is a Phase-9 model-authored advantage, BDS is the renderer's contribution, and D.3
is an INGESTION correction that benefits both architectures equally. "Phase 9 has better
supersession" overstates all three.

**NEGATIVE CONTROL, permanent.** A synthetic answer stating only the three current values, in
real answer shape with citations, is refused: MAAT and BDS `unmarked`, D.3 `absent`,
`supersession_ok=False`. D.3 reads absent rather than unmarked because the scorer no longer
credits the stray word "withdrawn" landing anywhere in 47,000 characters, which is how the old
metric passed an amendment that was never reported.

**THE BLOCK'S ENTRIES, AUDITED AGAINST THE SOURCE.** 9 entries in c1 and 11 in c2; all 20
supported. c1: 4 confirmed pairs, 1 confirmed withdrawal, 4 evidence-limited. c2: 4, 1, 6. ZERO
unsupported as a change. The renderer also surfaces two amendments the gold set does NOT list --
"the former standalone NBAD server table is superseded by the common HCI-1 node specification",
and the same construction for DAM. **So the gold set has 3 amendments and the document supports
at least 5, which is a CEILING ON THE METRIC: an answer can score `supersession_ok=True` while
silently omitting the two the gold omits.** The audit found them, not the metric -- and only
after its own cue list was corrected to include the word `supersede`.

| this corpus, aggregate | Phase 9 c1 | Phase 9 c2 | incumbent c1 | incumbent c2 |
|---|---|---|---|---|
| must-tier recall | 0.95 | 0.90 | 0.45 | 0.45 |
| should-tier recall | 0.75 | 0.75 | 0.45 | 0.50 |
| value accuracy | 0.824 | 0.824 | 0.529 | 0.588 |
| citations | 939 | 990 | 346 | 361 |
| faithfulness | 0.9831 | 0.9845 | **unjudged** | **unjudged** |
| request seconds | 1,378 | 1,528 | 372 | 431 |

**FAITHFULNESS IS NOT COMPARABLE HERE.** The incumbent returned no score in either cycle. That
is recorded as unjudged -- not zero, not assumed to have passed -- and the supersession and
faithfulness comparisons stay separate.

**HONEST LIMITS.** (1) Two cycles establish that the pattern REPEATS, not statistical
significance, and nothing here supports a general claim about the incumbent's supersession rate.
(2) Candidate c2 ran 16 hours after the other three rather than interleaved with them, because
the session running it was terminated mid-pass and the cycle had to be re-run; the order
rotation therefore holds for c1 and the incumbent pair but not for c2, which weakens it as a
control on latency and anything endpoint-sensitive. It does not weaken what it was for: the
amendment block is code, and it rendered all three relationships again. (3) The harness records
`supersession_ok` and `stale_values` but NOT `unmarked_changes`, and nothing that associates a
rendered line with the `MappedItem` it was built from; both per-amendment results above come
from offline rescoring of the preserved answers through the shipped scorer. Those are two
instrument gaps to repair, and the instrument was deliberately left frozen for the duration.

### The metric measures the GOLD SET, not the document: an adjudicated inventory (2026-11-02)

The findings above are frozen as measured. This section adds nothing to them and revises none of
them; it records a defect in the INSTRUMENT that the run exposed, and the terminology that defect
requires.

**`supersession_correctness` is gold recall, and should be read as `gold_supersession_ok`.** The
shipped field answers "were all amendments IN THE GOLD SET reported as changes?" It does not
answer "were all amendments in the SOURCE DOCUMENT reported?" The gold set lists 3; the source
supports **at least 5**. Three consequences, none of them visible in the number:

- an answer can score `True` while silently omitting two genuine amendments;
- an answer that reports those two earns no credit for it;
- the renderer can demonstrate broader coverage than the benchmark can measure.

Four measures are therefore kept apart, and only the first is currently computed by the suite:

| measure | what it means | status |
|---|---|---|
| gold amendment recall | the 3 specified gold amendments reported as changes | computed (`supersession_ok`) |
| audited amendment recall | coverage of independently adjudicated SOURCE amendments | **not computed** |
| unsupported amendment claims | renderer precision: entries the source does not present as a change | audited by hand: **0 of 20** |
| evidence-limited entries | changes where one side was not captured | audited by hand: 4 (c1), 6 (c2) |

**THE TWO ADDITIONAL AMENDMENTS, ADJUDICATED FROM SOURCE** -- quoted from the document's own
clause text, NOT from the platform-generated section summary that paraphrases it ("The
requirements clarify that..." is `platform_generated_metadata` and is not evidence of what the
document says):

| | NBAD | DAM |
|---|---|---|
| section | `corrigendum-5.pdf::s46a5546fd7b8` | `corrigendum-5.pdf::s8fa8694d957f` |
| breadcrumb | 8. NBAD Central Infrastructure — HCI-1 Cross-Reference | 12. DAM Central Infrastructure — HCI-1 Cross-Reference |
| page | 32 | 42 |
| chunk | `::c6e77711b4da6` | `::cf371c0335c4e` |
| clause | numbered item 2 | numbered item 2 |
| superseded side | "the former standalone NBAD server table" | "the former standalone DAM server table" |
| replacement | "the common HCI-1 node specification" | "the common HCI-1 node specification" |
| disposition | superseded ("is superseded by") | superseded ("is superseded by") |
| corollary in the same clause | "shall not be interpreted as requiring a separate physical NBAD server" | "...separate physical DAM server" |

**THEY ARE NOT PROMOTED INTO THE GOLD SET, and adjudicating them shows why that is not
mechanical.** The superseded side is a REFERENCE TO A TABLE, not requirement text: the document
says "the former standalone NBAD server table", and the table's figures appear separately as
"The original standalone-server values—2 × Xeon Platinum 8568Y+ class processors, ...". A gold
entry whose `old` is "standalone NBAD server table" would therefore test whether an answer NAMES
the superseded artifact, not whether it states what the artifact required -- a weaker question
than the three existing entries ask, and one an answer could satisfy while leaving a bidder with
no idea what changed. Expanding the gold set is the owner's call and needs that decision made
explicitly, per amendment.

**"AT LEAST FIVE" IS THE CLAIM, not five.** The audit covered the 20 entries the renderer
produced across two candidate cycles. It was not a systematic sweep of the whole document for
amendment language, so the inventory is a lower bound and must not be described as exhaustive.

**A CITATION MARKER IS NOT SOURCE SUPPORT.** The statement-level check above records that a
reported amendment's line carries its own markers. That establishes the line has a reference, NOT
that the cited chunk supports every part of the statement. Source support stays a separate
question, answered by Phase-10 verification and by the hand audit, and the two are never summed.

**THE LATENCY COST, stated rather than implied.** Phase 9 took **1,378 s and 1,528 s** against
the incumbent's **372 s and 431 s** -- roughly **3.5x to 3.7x**, on four observations with c2
not interleaved. The must-tier and value-accuracy advantages are real and repeated on this
corpus; so is this cost, and any production decision owes it the same weight.

#### Three categories, and what "0 unsupported" does NOT mean

A superseded document ARTIFACT is not a superseded REQUIREMENT that can be evaluated
meaningfully. That is the distinction the NBAD and DAM entries turn on, and it needs three
categories rather than a gold set and an overflow:

| category | treatment |
|---|---|
| requirement-level amendments | the existing gold benchmark (MAAT, BDS item 5, D.3) |
| artifact / table supersession | source-audited, OUTSIDE the current gold set (NBAD, DAM) |
| incomplete amendment evidence | rendered with explicit qualification, scored as neither |

A gold entry reading `former standalone NBAD server table -> common HCI-1 node specification`
would reward an answer for naming two artifacts without explaining what changed, which is a
WEAKER standard than the three existing entries hold. There is a route into the gold set later:
reconstruct the old requirement from the original table, establish its association with the
superseding HCI-1 specification, and adjudicate the actual technical differences. Until that
relationship is independently established, the artifact-level description is insufficient.

**ZERO UNSUPPORTED CLAIMS IS NOT COMPLETE EVIDENCE COVERAGE.** 0 of 20 entries were unsupported
as a change; 10 of those 20 were evidence-limited, carrying only the current side. Those two
findings do not contradict each other and must not be netted: the first is renderer PRECISION
(nothing was asserted that the source does not present as a change), the second is renderer
COMPLETENESS (one side of ten entries was never captured). An audit of 20 GENERATED entries is
also not an audit of the document.

**THREE KINDS OF SUPPORT, never summed.** A citation marker says a line carries a reference.
Phase-10 NLI verification says a claim was judged against retrieved text. A hand source audit
says a human traced the clause. They answer different questions at different strengths, and
none of them substitutes for another -- a marker in particular is not proof of semantic support.

## Experiment A — does Phase-8 selection preserve an exhaustive question's evidence? (2026-11-02)

**NOT an R@1 benchmark, for a structural reason rather than a choice of metric.** Only
`aggregate`, `compare` and `summarize` carry `section_pass`, and those are the intents that "do
not run a passage search at all -- they read every section out of the catalog". Selection there
NARROWS an already-complete set for cost and precision; it is not a finding operation. `lookup`,
`procedural` and `relational` have no section pass at all, so the 7 lookup queries with usable
derived gold never reach Phase 8, and precision@1 against a 37-of-53 evidence set would measure
nothing. The question is coverage at the production budget.

    Coverage@Budget = |selected INTERSECT derived_evidence_set| / |derived_evidence_set|

**THE VALIDITY CONTROL FIRST.** The offline reproduction selected **32 sections, identical in
SET AND IN ORDER to what production selected in both candidate cycles** (`query_log
.selected_sections`, `prod_sup9_c1` and `prod_sup9_c2`). This measures the shipped selector, not
a re-implementation of it.

| stage | selected | Coverage@Budget |
|---|---|---|
| 0 · universe | 53 | — |
| 1 · candidates reaching the cross-encoder | 53 | 37/37 gold present |
| 2 · score floor alone | 25 | **25/37 = 67.6%** |
| 3 · final, score + facet + amendment unions | 32 | **32/37 = 86.5%** |

Corpus `zzgoal-cf3b136d`, source bytes `3dc59d958833024d`; code `9e630c1`; budget
`max_sections=40`, `max_section_candidates=400`, `threshold=0.15`, `relative=0.1` — production
defaults, unchanged.

**THE FACET UNION IS THE MOST INFORMATIVE NUMBER HERE.** It added **7 sections and all 7 are in
the evidence set** — no waste at all, and it is what carries coverage from 67.6% to 86.5%. The
plan's step 4 is not a safety net on this document; it is doing a fifth of the work. The
amendment union added **0** sections beyond what score and facet had already admitted, which is
not evidence that step 5 is redundant: the three amendments' sections were already selected, so
on this document it had nothing left to contribute.

**PRECISION AGAINST THE DERIVED SET IS TOTAL: 0 of the 32 selected sections lie outside it**, and
**0 were dropped by the cap** (32 < 40), so the cap is not currently a constraint on this
document.

**THE FIVE SECTIONS MISSED** — all equipment or technical-specification sections:

    22. Firewalls · 20. Enterprise Cabinet/Rack · 16. NMS and ITSM
    13. Video Wall Technical Requirements · Log Collector in HA

**"32 OF 32 SELECTED SECTIONS CONTRIBUTED" IS A CONTRIBUTION RATE, NOT RECALL.** The answer's own
coverage line reports 100% contribution, and that statement is true and says nothing about the
five sections never offered to the map. The intersection with the derived evidence set is 86.5%,
and the two numbers must never be read as the same claim.

### Honest limits

**(1) The gold is a DERIVED set, not adjudicated relevance.** `derived_section_evidence_set` is
the union of sections holding any of the suite's 71 answer cues, every one located, with
cue-to-section provenance kept. **51 of those 71 cues span more than one section**, and an
occurrence is not automatically independently answer-bearing. Everything above is coverage
against this derived set.

**(2) Candidate recall is trivially 100% and is NOT a meaningful measurement here.** The
candidate cap is 400 and the document has 53 sections, so stage 1 never excludes anything. This
is recorded rather than presented as a result; on a document with more than 400 sections it
would become a real metric.

**(3) One aggregate query, one document, one family.** This establishes a measurement METHOD and
one observed data point. It is not a per-family result and says nothing about the other 19 KBs or
the other two document families — and the family with 16 of the 18 calibration documents has no
authored queries at all.

**(4) Structural validity is not semantic correctness.** Phase 8 cannot select a nonexistent
section, but it can select a badly segmented one. The same source bytes yielded 56 sections
before yesterday's heading fix and 53 after, so the ToC this selector narrows is DERIVED by our
own classifier, not authored by the publisher — which is the condition STAIR's own limitations
section lists as future work.

**(5) A is not a verdict on the lookup path.** Near-complete exhaustive coverage says nothing
about semantic-mismatch queries, which the lookup path serves through chunk retrieval plus parent
expansion and which Experiment B exists to test.

### Adjudicating the five, and a robustness check on the derived set

**TWO OF THE FIVE ARE GENUINE MISSES, not five.** Each missed section was traced back to the cues
that admitted it, and to how many sections each of those cues spans:

| missed section | admitted by | cue spans | verdict |
|---|---|---|---|
| 20. Enterprise Cabinet/Rack | `IEC 60297` | **1 section** | **genuine miss** |
| 22. Firewalls | `different original equipment manufacturers` | **1 section** | **genuine miss** |
| 13. Video Wall Technical Requirements | `Video Wall` (+ `HBOM`) | 4 sections | probable miss |
| 16. NMS and ITSM | `NMS`, twice | **20 sections** | derived-set artifact |
| Log Collector in HA | `NMS`, `NMS`, `DAM` | **20 / 18 sections** | derived-set artifact |

`16. NMS and ITSM` and `Log Collector in HA` are in the evidence set only because the acronym
`NMS` appears in 20 of the 53 sections -- a cross-reference token with almost no localisation
value. The two genuine misses are the opposite: each was admitted by a cue appearing in exactly
ONE section of the document, so the evidence is specific and Phase 8 did not select it.

**ADJUDICATING ONLY THE MISSES WOULD HAVE INFLATED THE RESULT, so the filter was applied
symmetrically to all 71 cues** rather than to the 5 that happened to fail:

| gold restricted to cues spanning | gold size | covered | coverage | missed |
|---|---|---|---|---|
| ≤ 53 (unfiltered) | 37 | 32 | 86.5% | 5 |
| ≤ 11 | 28 | 25 | 89.3% | 3 |
| ≤ 6 | 25 | 22 | 88.0% | 3 |
| ≤ 3 | 17 | 15 | 88.2% | 2 |
| ≤ 1 (cue unique to one section) | 10 | 8 | **80.0%** | 2 |

**The 86.5% is not an artifact of low-specificity cues.** Across every threshold the gold set
shrinks from 37 to 10 while coverage moves within 86.5-89.3%, and at the STRICTEST setting it
*falls* to 80.0% -- so the filter does not flatter the selector. What the strict end isolates is
the real failure: at cue-unique-to-one-section, exactly the two genuine misses remain.

**Pinned in code.** `test_removing_the_facet_union_costs_coverage_a_future_simplification_must
_recover` is the measurement in miniature -- 53 sections, 37 evidence-bearing, 25 reachable by
score, 7 only by facet. The 7 scored BELOW THE FLOOR, so no threshold change recovers them; only
the router's facets do. A change that makes step 4 redundant now fails a test and has to show
where the 7 came back from.

**Still a proxy.** Zero selected sections outside the derived set is 100% precision against
THAT PROXY, not demonstrated real-world relevance precision, and the adjudication above covers
the 5 misses rather than all 37 members.

### The real finding: a section without Phase-4 enrichment is UNREACHABLE by Phase 8

Crossing Experiment A's selection against enrichment availability explains every miss and
falsifies a claim written in our own code.

| | sections | selected | |
|---|---|---|---|
| enriched (has a Phase-4 summary) | 35 | **32** | 91% |
| **degraded (no summary)** | **18** | **0** | **0%** |

**All five missed sections are degraded cards. Not one of the 18 was selected, by any reason.**

**WHY, and it is structural rather than a tuning gap.** All three selection reasons read Phase-4
output, and a section whose enrichment did not run has none of it:

| | summary | `facet_ids` | `amends` | `key_values` |
|---|---|---|---|---|
| enriched (35) | 35 | 32 | 2 | 35 |
| degraded (18) | **0** | **0 of 18** | **0** | **0** |

So a degraded section scores on breadcrumb and heading alone (below the floor), **and** step 4
cannot union it in because it has no `facet_ids`, **and** step 5 cannot because it has no
`amends`. Every path into the selection is closed at once. 18 of 53 sections -- 34% of this
document -- are invisible to the section pass by construction.

**THE `SectionCard` DOCSTRING IS WRONG ON BOTH COUNTS**, and it is worth quoting because it is
the assumption this measurement overturns:

> "a section whose enrichment never ran ... is scored on its breadcrumb and heading alone, which
> is degraded but not broken: a heading like *Video Wall Technical Requirements* carries real
> signal for a facet sub-query."

`13. Video Wall Technical Requirements` is one of the five misses. The claim fails on the
mechanism too: such a card does not merely lose score quality, it loses BOTH union paths, which
the docstring does not mention. Measured, degraded is not "degraded but not broken" -- it is
unreachable.

**HONEST SEVERITY.** Of the 18, roughly nine read as table-row fragments (`16 Processor`,
`17 Memory`, `21 Serial Ports`, `22 Temperature Range` ...) that arguably should not be sections
at all -- the residual of the same heading-promotion class fixed yesterday -- and four are
document furniture (`S. Offered by`, the letterhead, the NIT number). About **five are
substantive requirement sections**, two of which Experiment A already adjudicated as genuine
misses (`22. Firewalls`, `20. Enterprise Cabinet/Rack`). So the practical cost on this document
is smaller than 34% suggests, and the architectural exposure is not: on a document where
enrichment fails over important sections, Phase 8 cannot see them and nothing in the answer says
so.

**NOT YET ESTABLISHED: why enrichment did not run on those 18.** `token_count` is NULL for every
section in this corpus, so no size rule is visible from the catalog. Skipped, failed, or
legitimately declined for table fragments are three different causes with three different fixes,
and this measurement does not distinguish them. That is the next diagnostic.

**THIS CORRECTS WHAT I SAID ABOUT EXPERIMENT B3.** I had argued that Experiment A gave an
unfavourable prior for enrichment, because the score floor alone reached only 67.6%. That reading
was wrong: the 67.6% was achieved ENTIRELY over enriched cards, and the whole miss set is cards
with no summary. Enrichment is not measured as insufficient -- its ABSENCE is the only failure
mode observed. B3's prior is favourable, not unfavourable, and the summary-availability stratum
is not one stratum among several on this corpus: it is the dominant variable.

### Why those sections were never enriched: the model returned no JSON, twice (2026-11-02)

The ingest log settles it. `section_enrich_scoped` for this corpus:

    sections_total=53  sections_offered=38  sections_enriched=35  sections_failed=3  capped=False

So the 18 degraded sections are **two different populations**, and only one of them is a defect:

| population | count | cause | correct? |
|---|---|---|---|
| below `section_enrich_min_chars` (200) | **15** | never offered — 13 to 157 prompt chars | **yes, by design** |
| offered, called, **failed** | **3** | `no JSON object in the reply`, on BOTH attempts | **no — this is the defect** |

The three failures are, exactly, the three substantive misses:

    s8a63a97488c5  13. Video Wall Technical Requirements   6,720 prompt chars   attempts 1,2 failed
    s3cd9cabe84d2  20. Enterprise Cabinet/Rack             4,397 prompt chars   attempts 1,2 failed
    s17c0dd85dc0f  22. Firewalls                           8,675 prompt chars   attempts 1,2 failed

**NONE of the documented filters explains them.** They are 20-43x over the 200-char minimum; the
cap did not bind (`capped=False`, 53 << 400); and table content is not the cause -- **23 enriched
sections carry tables, the largest at 29,733 prompt chars**, nearly four times the biggest
failure.

**THE NEXT SENTENCE OF THIS SECTION WAS WRONG AND IS CORRECTED BELOW** (see "The replies were
never prose"). It read: "The model simply returned prose instead of a JSON object, twice, for
these three." It did not. Every one of those replies is a valid JSON object truncated
mid-structure, and the conclusion was drawn from the parser's error string without looking at a
single reply.

**FOUR MORE SECTIONS HIT THE SAME FAILURE AND RECOVERED ON THE SECOND ATTEMPT** (`1. NG-SIEM`
window 2, `16.1 HCI-2`, `18. Data Diode`, `21. Switches`). So the reply is unreliable rather than
impossible, and the retry is what rescues it -- which makes the retry budget the live variable.

**THE RETRY BUDGET IS HARDCODED AT 2, AND IT IS THE ONLY LLM PHASE ON THIS PATH THAT NEVER GOT
THE TREATMENT THE OTHER TWO DID.** `LLMSectionEnricher._enrich_one` tries `attempt=1` then
`attempt=2` as literals, with no setting. Meanwhile:

    architecture_extract_retries   default 5   added because this model tier returns an EMPTY
                                               extraction ~2/3 of the time
    doc_profile_retries            default 4   config comment: "Same shape and the same reason
                                               as architecture_extract_retries"

Phase 4 is the third instance of a failure mode the repo has already diagnosed and fixed twice.

**THE SECOND DEFECT IS THE SILENCE, and it is the one that cost coverage.** `sections_failed=3`
appears in one structured log line at ingest and nowhere else. The section rows are written
without summary, `facet_ids`, `amends` or `key_values`, which closes all three Phase-8 selection
paths at once -- and the answer's coverage line counts those sections in "53 sections in scope"
while reporting "32 of 32 selected sections contributed". A reader is told the document has 53
sections and that everything selected contributed; nothing says three substantive sections were
never eligible to be selected because an ingest-time model call failed yesterday.

**Not a fail-safe working as intended, on the second count.** `_section_enrichment`'s docstring is
right that a partial map must not fail the ingest. What it does not say, and what is now measured,
is that a partial map silently removes those sections from an aggregate answer's reachable
evidence.

**Two separable fixes, neither implemented here:** (1) give enrichment the retry budget its two
siblings have, since 4 of 7 affected sections recovered on attempt 2 and nothing was tried at
attempt 3; (2) make an enrichment failure VISIBLE where it has consequences -- a section whose
enrichment failed is not the same as a section the minimum-length rule declined, and Phase 8 and
the coverage line currently cannot tell them apart.

### The replies were never prose: enrichment fails on OUTPUT TRUNCATION (2026-11-02)

**CORRECTING MY OWN DIAGNOSIS OF TWO HOURS EARLIER.** The section above concluded "the model
simply returned prose instead of a JSON object". That was inferred from
`parse_section_enrichment`'s error string -- `no JSON object in the reply` -- without ever
looking at a reply. Adding a bounded `reply_head` to the log, which was done to tell prose from
malformed JSON, showed the opposite of what the error says:

    window 0 attempt 1  reply_chars=4412  head: '```json {   "summary": "This section details…'
    window 0 attempt 2  reply_chars=4290  head: '{   "summary": "This section outlines…'
    window 0 attempt 5  reply_chars=4173  head: '{   "summary": "This section outlines…'

Every failing reply is a **valid JSON object, cut off mid-structure**, at **3,569-4,539
characters = 965-1,227 tokens against `section_enrich_max_tokens` = 1,200**. The parser reports
"no JSON object" because it cannot find a COMPLETE one; for a truncation that message is
actively misleading, and it is what sent this investigation at an instruction-following problem.

**THE FIVE-ATTEMPT EXPERIMENT THEREFORE DOES NOT SUPPORT RAISING THE RETRY DEFAULT.** One
variable (`section_enrich_max_attempts` 2 -> 5), same source bytes `3dc59d958833024d`, same
model, same prompt, throwaway corpus `zzgoal-retry5`, `sections_offered=38` in both runs:

| | 2 attempts | 5 attempts |
|---|---|---|
| enriched / failed | 35 / **3** | 36 / **2** |
| 13. Video Wall Technical Requirements | FAILED | **FAILED** |
| 20. Enterprise Cabinet/Rack | FAILED | recovered |
| 22. Firewalls | FAILED | recovered |
| **21. Switches** | enriched | **FAILED** |

**3 failures became 2 by swapping which sections they were.** Two recovered and one that had
succeeded was lost. That is churn around a ceiling, not a reliability gain -- retrying a
truncated reply re-rolls its length rather than buying it room. A single-number read of this
(`3 -> 2`, "the retry recovered one") would have been wrong in exactly the way the arms-swapped
completeness claim was wrong in the articulation comparison.

**PRODUCTION DEFAULT UNCHANGED at 2**, as instructed, and now for a second reason: the
experiment argues against raising it.

**WHAT THE SETTING IS STILL WORTH.** It is kept -- configurable, default 2, byte-identical
behaviour, and the right shape for a genuinely flaky reply, which is a real failure mode this
model tier has elsewhere (`architecture_extract_retries`, `doc_profile_retries`). Its code
comments and its test now say explicitly that it does NOT address the corpus failure, so nobody
reaches for it next time.

**THE LIKELY REAL FIX IS NOT IMPLEMENTED, because it was not the approved experiment.** Two
candidates, and they are not equivalent: raise `section_enrich_max_tokens` for the sections that
need it, or window those sections more tightly so each reply has less to carry
(`section_enrich_window_chars` is 12,000 and the three failures have 4,397-8,675 prompt chars, so
all of them are ONE window today). The second keeps the output bound and costs more calls; the
first keeps the calls and costs output budget. Both need measuring, and the parser's error string
should say `truncated` when the reply opens a JSON object it never closes -- a misleading message
cost this investigation two hours and one wrong committed conclusion.

### Video Wall recovery: the output budget is the cause, and windowing makes it worse (2026-11-02)

`13. Video Wall Technical Requirements` lost its Phase-4 enrichment in three independent ingests
of identical source bytes. Nothing else failed more than once. Driven directly on that one
section's real prompt text (6,748 chars raw, 6,603 masked through the same `prepare_for_index`
the pipeline uses), the document's own profile card and its 10-facet vocabulary, three runs per
arm, model/prompt/temperature/retries held:

| arm | window / output | parse | reply chars | quote validation |
|---|---|---|---|---|
| **control** (shipped) | 12,000 / 1,200 | **TRUNCATED 3/3** | 4,155-4,361 | — |
| **A** | 12,000 / **3,000** | **OK 3/3** | 5,662-6,368 | **27/27 · 24/24 · 25/25** |
| **A+** | 12,000 / 4,000 | OK 3/3 | 5,317-6,029 | 24/24 · **20/21** · **19/22** |
| **B** | **4,000** / 1,200 | **TRUNCATED in all 3 runs** | 3 windows | 21/25 · 24/25 · 10/10 |

**THE DIAGNOSIS IS NOW CAUSAL RATHER THAN STRUCTURAL.** The parser could only say that a JSON
object opened and never closed; `LLMGateway.complete` returns a bare string, so no
`finish_reason` was available to prove the output ceiling caused it. Raising ONLY
`max_tokens`, with the input byte-identical, turns 3 failures out of 3 into 3 successes out of 3.
That is an intervention, not an inference.

**ARM A IS A FULL RECOVERY, not merely a parse.** 5-6 facets drawn from the document's own
vocabulary, 10-11 key values, 14-17 propositions, and **100% of quotes validating against the
section text** in all three runs — against the plan's 0.95 floor.

**MORE IS WORSE: A+ DEGRADES THE QUOTE RATE.** At 4,000 tokens the model produces more claims
and some are unquotable — 0.952 and **0.864** on two of three runs, the second below the floor.
So the ceiling is not "set it high"; 3,000 is measured and 4,000 is measurably worse.

**ARM B DOES NOT WORK, AND FAILS IN THE MORE DANGEROUS DIRECTION.** Smaller windows do not shrink
the reply enough — this section is table-dense, so each window still exceeds 1,200 tokens, and the
final window truncated in all three runs. Worse, `merge_enrichments` merges whatever parsed, so a
partial truncation becomes a **stored enrichment that looks successful**: facets collapse from 5-6
to 2, key values swing 2-11, and the summary ranges 255-631 characters across runs. A clean
failure is reported and disclosed; this is not. **Windowing would convert a visible failure into
an invisible content loss**, which is the opposite of the direction the whole investigation has
been pushing.

**RAISING THE CEILING COSTS NOTHING ON SECTIONS THAT ALREADY WORK**, unlike the retry budget.
`max_tokens` is an upper bound, not a spend: a section whose reply completes in 800 tokens still
generates 800. The only risk is a model rambling to fill the room, which is exactly what A+
measured — so the recommendation is a specific number, not a generous one.

**RECOMMENDED, NOT LANDED:** `section_enrich_max_tokens` 1,200 -> 3,000. The approved experiment
was to decide between the two mechanisms, and it has; changing a production default is the
owner's call.

**HONEST LIMITS.** (1) One section, one document, one family — the number 3,000 is calibrated
against the largest known failure, not against a population. (2) The arms are three runs each,
which establishes a repeated pattern and not significance; the control's 3/3 and A's 3/3 are
clean, but A+'s quote-rate degradation is 2 of 3 and could be noise. (3) This recovers the
ENRICHMENT; whether the recovered summary improves an ANSWER is a separate measurement nobody
has made. (4) Arm B's silent-merge hazard exists TODAY for any section whose windows partially
truncate, independently of this experiment — it is not introduced by changing the budget.

**THE FIRST RUN OF THIS EXPERIMENT WAS INVALID AND IS RECORDED AS SUCH.** The section text was
reconstructed from `blocks` by splitting `psql -A -t` output on its field separator, but
`blocks.text` contains newlines, so only the FIRST LINE of each block survived: 169 characters
against the 6,720 already measured for this section. Every arm then "passed" on input a tenth the
real size. It was caught only because that 6,720 figure existed to compare against, and the
script now refuses to run on an input under 5,000 characters.

### The 3,000-token default is landed, and the whole document recovers (2026-11-02)

**APPROVED AND LANDED** (`7af7103`): `section_enrich_max_tokens` 1,200 -> 3,000. Windowing is
**REJECTED** as the recovery mechanism. The owner's conditions and their status:

| condition | status |
|---|---|
| default output budget 3,000 | **landed** (`7af7103`) |
| existing retry default stays 2 | unchanged, `section_enrich_max_attempts: int = Field(default=2, ...)` |
| tighter-windowing recovery rejected | not implemented; `section_enrich_window_chars` stays 12,000 |
| 0.95 quote-acceptance floor preserved | `QUOTE_RATE_FLOOR = 0.95` untouched; measured below |
| all five parser codes preserved | preserved, and `TRUNCATED` is now **authoritative** (`2c3ef91`) |
| actual token consumption recorded per section | `section_enrich_section_done`, one line per section, server-reported |
| rollback keeps 1,200 configurable | same setting; it is a `Field` default, not a constant |
| partial-window merging treated as a separate defect | fixed independently in `2c3ef91` |

**THE ROLLOUT VALIDATION THE OWNER ASKED FOR: THREE FULL INGESTS, NOT ONE SECTION.** The Video
Wall experiment established that 3,000 works for the largest known failure; it did not establish
that the rest of the document behaves the same. So the whole corrigendum was re-ingested THREE
times through the shipped path into three throwaway tenants (`zzgoal-bud1/2/3`, dropped in a
trap), same bytes (`sha256 3dc59d958833024d...`, 53 sections), same prompt, same windowing
(12,000/4), same attempt budget (2), reading the STORED artifacts — the manifest and the
`sections` rows — rather than the run's own log.

| | bud1 | bud2 | bud3 | at 1,200 |
|---|---|---|---|---|
| sections offered | 38 | 38 | 38 | 38 |
| **sections enriched** | **38** | **38** | **38** | **35** |
| sections failed | **0** | **0** | **0** | 3 |
| sections partial | 0 | 0 | 0 | n/a |
| policy skips | 15 | 15 | 15 | 15 |
| quotes offered | 496 | 468 | 481 | — |
| **quote rate** | **0.958** | **0.968** | **0.971** | — |
| completion tokens, document | 37,833 | 35,797 | 37,657 | — |
| **calls at the ceiling** | **0** | **0** | **0** | every failure |

**THE THREE SECTIONS 1,200 LOST ARE NOW ENRICHED IN ALL THREE RUNS**, which is what "reliably"
required — the retry experiment's three failures became two by SWAPPING which sections they were,
so a single clean run would have proved nothing:

    s8a63a97488c5  13. Video Wall Technical Requirements   1,509t / 1,604t / 1,463t
    s3cd9cabe84d2  20. Enterprise Cabinet/Rack             1,310t / 1,209t / 1,151t
    s17c0dd85dc0f  22. Firewalls                           1,519t / 1,367t / 1,559t

Those figures ARE the mechanism: every one of the nine calls spent more than 1,200 tokens, so
under the old ceiling none of them could finish. `21. Switches`, which failed all five attempts
when the retry budget was raised, also enriched in all three runs. **No section failed in any
run**, so nothing that worked at 1,200 regressed.

**WHAT IT ACTUALLY SPENT, which is the number to monitor rather than the setting.** Over 135 calls
(45 windows x 3 runs) the mean call spent **795-841 tokens** and the median section 758-864; the
largest SINGLE call in any run was **1,604 tokens — 53% of the ceiling**. Not one call in any run
stopped at the cap (`finish_reason="length"` never fired), so the 3,000 is genuine headroom on
this corpus and not a new wall. Server-reported, not estimated: this deployment's gateway returns
`usage` with `estimated: False` and a real `finish_reason`. The per-section totals above 3,000
(`1. NG-SIEM` 3,487-3,756, `16.1 HCI-2` 3,288-3,899) are MULTI-WINDOW sections summed across 3-4
calls, not single replies — the ceiling is per call.

**THE QUOTE FLOOR HOLDS ACROSS THE DOCUMENT**, which the single-section experiment could not say.
0.958 / 0.968 / 0.971 over ~480 quotes per run, all above 0.95. This is the measurement that
matters more than the 4,000-token arm's 0.864: that figure came from one section's three runs,
and what a ceiling does to a whole document's evidence rate is only visible here.

**HONEST LIMITS.** (1) ONE DOCUMENT, one model (`gemma4-31b`), one corpus family. 38 sections
across three runs is a population for this document and not for the product. (2) `calls_at_cap`
is 0 HERE; a document with a denser table than `1. NG-SIEM` can still hit 3,000, and the warning
that would show it is the per-section line, not a failure. (3) The 4,000-token degradation was
measured on one section and was 2 of 3 runs; 3,000 is chosen as the measured point, not as proof
that 4,000 is harmful at document scale. (4) `amends_found` varied 2 / 3 / 2 across three runs of
identical bytes at identical settings — this is run-to-run model variation that PREDATES this
change and is not attributable to it, but it means amendment detection is still not reproducible
and no claim here rests on it.

### A half-read section is not an enriched one (2026-11-02)

**THE INDEPENDENT DEFECT THE WINDOWING ARM EXPOSED, now fixed** (`2c3ef91`).
`merge_enrichments` merges whatever it is handed — the right contract for a merge, the wrong one
for deciding that a section is ENRICHED. Handed two of a section's three windows it returned a
perfectly valid enrichment, which was then stored, counted as a success, and used by Phase 8 as
that section's selection metadata with a third of its facets, key values and summary missing.
Nothing recorded the loss: not the report, not the manifest, not the answer.

`merge_windows` now refuses the merge unless every window parsed, retains the failing window's
parser code, and counts the section as `sections_partial` — a SUBSET of `sections_failed`, warned
separately (`section_enrich_partial_windows_refused`) because the model demonstrably could read
the section, so the fix is room and not a prompt. The INPUT cap (`section_enrich_max_windows`) is
deliberately untouched: text past it is a disclosed truncation decided before any call, and
conflating the two would mark every over-long section in a corpus incomplete.

**A PASSING TEST ASSERTED THE DEFECT.** `test_one_failed_window_does_not_lose_the_others` demanded
exactly the old behaviour — `sections_enriched == 1` on a section whose first window never
answered — and it passed for as long as it existed. It is rewritten as
`test_a_partially_parsed_section_is_refused_not_merged` with that history in its docstring,
because a test that documents a hazard as the intended behaviour is worse than no test.

**`finish_reason` NOW DECIDES `TRUNCATED`.** The earlier diagnosis of this failure rested on the
parser's inference that an object had opened and never closed. That inference is right in the
common case and is still an inference — a reply cut off BETWEEN two complete objects parses as
valid JSON of the wrong shape and would send an operator to the schema for a budget problem. Where
the gateway reports usage, the server's own "I ran out of room" now sets the code. A gateway with
only `complete()` reports nothing and loses nothing else.

**STILL OPEN:** the three preserved reference corpora (`zzgoal-9bdbfe1e`, `550c4198`, `cf3b136d`)
were ingested at 1,200 and their stored manifests record that run. They are controls for other
experiments and were deliberately NOT re-ingested; any measurement that reads their enrichment is
reading the old ceiling.

### Experiment A, replayed on the recovered corpus: coverage up, and the score stage is hostage to one facet name (2026-11-02)

The 1,200-era Experiment A measured a corpus with three substantive sections unenriched. Those
numbers describe that ingestion state, so the measurement was replayed UNCHANGED against three
independent 3,000-era corpora (`zzgoal-ea1/2/3`, same bytes, same query, same selection config).
**THE GOLD AND EVERY LABEL WERE FROZEN FIRST**: the derived evidence set, its 37 members and the
five recorded misses were fixed before any of this ran, and nothing was re-labelled afterwards.
Only the JSON output path changed, so the baseline survived to be compared against.

| | 1,200 baseline | ea1 | ea2 | ea3 |
|---|---|---|---|---|
| universe / gold | 53 / 37 | 53 / 37 | 53 / 37 | 53 / 37 |
| gold in candidates | 37/37 | 37/37 | 37/37 | 37/37 |
| **stage 2 score-only** | **25/37** | **14/37** | **18/37** | **12/37** |
| **stage 3 Coverage@Budget** | **32/37 (86.5%)** | **33/37 (89.2%)** | **35/37 (94.6%)** | **35/37 (94.6%)** |
| facet union added (of them gold) | 7 (7) | 19 (19) | 18 (17) | 23 (23) |
| amendment union added | 0 | 0 | 0 | 0 |
| dropped by the cap | 0 | 0 | 0 | 0 |
| selected NOT in gold | 0 | 0 | 1 | 0 |
| gold missed | 5 | 4 | 2 | 2 |

**THE THREE RECOVERED SECTIONS ARE NOW SELECTED IN ALL THREE RUNS.** `13. Video Wall` every time
through the FACET union, `22. Firewalls` every time through SCORE, `20. Enterprise Cabinet/Rack`
by score / facet / score. All three were misses in the baseline.

**AND THE ORIGINAL ATTRIBUTION HOLDS.** The baseline's other two misses — `16. NMS and ITSM` and
`Log Collector in HA` — are missed in all three replays. They were never enrichment failures, so
the earlier claim that 3 of the 5 misses were attributable to unavailable enrichment and 2 were
not is confirmed by the fix rather than by argument.

**THE WITHIN-CORPUS CONTROL IS EXACT, which is what makes the rest attributable.** `_route_v2` is
an LLM call and `_section_scores` is a cross-encoder, so cross-corpus variation could have been
theirs. ea1 was measured TWICE: identical in every recorded field, including the selected ORDER.
So every difference below belongs to the enrichment.

**SELECTION IS NOT STABLE ACROSS INGESTS OF IDENTICAL BYTES.** Pairwise Jaccard 0.917 / 0.943 /
0.972; 33 sections selected in all three, 36 in at least one; the order never identical. Two GOLD
sections are missed in exactly one run each (`24 TAP`, `14. Desktop/Console for LED Video Wall`) —
**neither was missed at 1,200**. So the recovered pipeline is better on average and not uniformly
better: stable-core coverage is 33/37 while the union of the three runs reaches 35/37.

**THE SCORE STAGE COLLAPSED, AND THE FLOOR IS NOT WHY.** Score-only admission fell 25 -> 12-18
even though the effective floor barely moved. `select_sections` step 3 admits a section at or
above `threshold` **OR** within `relative` of the best score — an OR, so the effective floor is
`min(0.15, 0.1 x top)`, which is 0.0948 / 0.0979 / 0.0971 / 0.0990 across the four corpora.
Admissions reconcile exactly with the measurement at that floor (25 / 14 / 18 / 12). What moved
is the SCORES: document median 0.0909 -> 0.0141 / 0.0114 / 0.0116.

**THE CAUSE IS THE FACET VOCABULARY, NOT THE SUMMARIES.** `_section_probes` builds the probe set
from the facet TAXONOMY, which is LLM output regenerated on every ingest — so a different
enrichment run rotates the probes and every section is scored against different text:

    1,200 baseline   ... hardware and infrastructure specifications ...
    ea1              ... deployment and infrastructure flexibility ...   (no hardware probe)
    ea2              ... hardware infrastructure sizing ...
    ea3              ... infrastructure and deployment ...

**SIX of the eight sections** that were score-admitted at 1,200 and in none of the three replays
were carried by that ONE probe, `hardware and infrastructure specifications`:

    s95dc70e49e40  17. HCI-2 NOC/Infra Mgmt      0.1437 -> 0.0056   (best ea1 probe: deployment and infrastructure flexibility)
    sb5df7228e5d7  17.2 HCI-2 Node/Server HW     0.3322 -> 0.0285   (best ea1 probe: vendor and oem qualifications)
    sc6eee907af80  4. NG-SIEM/SOAR/UEBA License  0.2820 -> 0.0039
    sd9410b18528f  6. Log Collector              0.1968 -> 0.0097
    sd00fe2ef066e  14. Desktop/Console LED Wall  0.1764 -> 0.0207
    sc31bcc9fc142  5.2 HCI-1 Workload Allocation 0.0985 -> 0.0013

The other two were carried by `deployment and environment constraints` (0.2588) and `high
availability and disaster recovery` (0.2585); ea1's nearest equivalents score them 0.0393 and
0.0350. **A 25-70x score drop from one facet name the model did not re-invent.** Nothing about
these sections' own cards changed in a way that explains it: `rerank_text` lengths are flat
(median 313 -> 315 / 332 / 342 characters).

**WHY COVERAGE STILL ROSE.** The facet union absorbed the loss — it added 7 sections at 1,200 and
18-23 here, and nearly all of them are in the gold (19/19, 17/18, 23/23). Phase 8 is now
facet-union-dominated rather than score-led. That is a different selection mechanism reaching a
better number, not the same mechanism working better.

**WHAT THIS MEANS FOR EXPERIMENT B, which is the reason it was worth measuring.** B was to
measure the value of enriched summaries. On this evidence a summaries-on/summaries-off comparison
is CONFOUNDED: enrichment produces both the summaries AND the facet vocabulary, the vocabulary
drives the probes, and the probes dominate the score stage. B needs the facet taxonomy held fixed
across arms, or it will attribute to summaries an effect that belongs to one facet name.

**HONEST LIMITS.** (1) One query, one intent (`aggregate`), one document. The 71-cue derived
evidence set is NOT adjudicated relevance and nothing here should be read as recall. (2) Three
corpora is enough to show instability and not enough to size it. (3) The probe attribution above
is measured per-probe argmax on two corpora (baseline and ea1); ea2/ea3 were not broken down
per-probe. (4) The 1,200 baseline's own numbers are a SINGLE run — the instability demonstrated
here applies to it too, so the 25 -> 12-18 drop is a shift between distributions sampled once and
three times respectively, not a measured delta between two stable points.

**CORRECTED IN FLIGHT:** the first version of the floor probe computed `max(threshold,
relative x top)` instead of the OR's `min`, and reported 20 admissions on a corpus whose own
measurement recorded 25. The five-section discrepancy is what caught it; the published floor
values and admission counts are from the corrected script and reconcile exactly.

### Phase-4 extraction is NOT reproducible, and the dominant cause is Phase 3's facet vocabulary (2026-11-02)

`amends_found` varied 2/3/2 across identical bytes, so three more 3,000-era corpora
(`zzgoal-rp1/2/3`) were ingested and their extraction CONTENT compared rather than their counts.
**QUOTE VALIDATION IS NOT USED AS A COMPLETENESS METRIC ANYWHERE BELOW**: it says the model's
claims are supported by the text in front of it, not that the same requirements were found. This
document proves the distinction — `rp1` has the LOWEST quote rate of the seven runs and the same
38/38 and the same amendment count.

| | rp1 | rp2 | rp3 |
|---|---|---|---|
| sections enriched / offered | 38/38 | 38/38 | 38/38 |
| quote rate | **0.948** | 0.968 | 0.951 |
| amendments found | 2 | 2 | 2 |
| key values (38 common sections) | 177 | 179 | 176 |
| propositions (same) | 293 | 301 | 279 |

Every COUNT is stable. Every IDENTITY is not:

| what | reproducibility across the three |
|---|---|
| amendments | **0 of 3** sections carry one in all three; 3 distinct amendments exist and each run finds a different TWO |
| facet vocabulary | 10 ids per run, **1** id common to all three; pairwise Jaccard 0.111 / 0.250 / **0.053** |
| facet ids per section | mean pairwise Jaccard 0.173; identical in **2 of 38** sections |
| key values (name, value, unit) | mean pairwise Jaccard 0.546 |
| key values, NAME only | mean pairwise Jaccard 0.637 |
| propositions | 109-121 exact matches of ~290; closest-match similarity 0.82 |
| summaries | median pairwise similarity 0.430, **0 identical**, worst 0.226 |

**THE AMENDMENT VARIATION IS A GENUINE OMISSION, NOT A GROUPING DIFFERENCE** — which is the
question that was asked of it. The three amendments have distinct targets and live on distinct
sections, and each run drops exactly one:

    rp1   standalone DAM server table   +  Appendix to ITB-2 ... MAAT
    rp2   standalone NBAD server table  +  Appendix to ITB-2 ... MAAT
    rp3   standalone NBAD server table  +  standalone DAM server table

So the MAAT amendment — the one the supersession evaluation turns on — is absent from one run in
three. (All three carry `old_value: None` and `new_value: None`, which is the evidence-gap the
renderer wording was made safe for.)

**THE DOMINANT INSTABILITY IS PHASE 3's, NOT PHASE 4's**, and that distinction decides where a
fix belongs. The `facets` table is written from `profile.facets` — the Phase-3 document-profile
call — and Phase 4 may only ASSIGN from that vocabulary. The three vocabularies are semantically
near-identical and lexically disjoint:

    performance_and_scalability          vs  performance_and_scalability_metrics
    data_retention_and_storage           vs  data_retention_and_storage_management
    deployment_and_infrastructure        vs  deployment_and_infrastructure_flexibility
    threat_detection_and_analytics       vs  security_analytics_and_threat_detection
    compliance_documentation             vs  technical_compliance_documentation

Identical meaning does not help, because the id is a JOIN KEY for the facet union and the LABEL
is a cross-encoder probe. This is the mechanism behind the Experiment A result one section above:
the score stage collapsed because six sections lost the probe that was carrying them.

**PHASE-4 ASSIGNMENT, HELD AGAINST A FIXED VOCABULARY, IS STABLE.** Six re-extractions of
`13. Video Wall` against rp1's stored taxonomy returned the SAME four facet ids every time. So
Phase 4 is not inventing facet drift; it is faithfully assigning whatever Phase 3 invented.

**THE ENRICHER DOES NOT RUN AT THE TEMPERATURE IT ASKS FOR — MEASURED ON THE WIRE.**
`_attempt` passes `temperature=0.0`; the request body carried **0.6**. `OpenAICompatGateway._eff_temp`
substitutes 0.6 when a caller asks for greedy AND reasoning is suppressed, and this deployment
sets `RAG_LLM_REASONING_EFFORT=off`. That nudge was added for a real, documented failure (greedy
with no reasoning made this model loop on tables and echo the prompt's own examples), but it
applies to the structured-extraction calls too, where reproducibility is the property that
matters.

**TEMPERATURE AMPLIFIES THE VARIATION; IT DOES NOT CREATE IT.** The same prompt posted directly,
three times at each setting, bypassing `_eff_temp`:

    temp 0.0   2 of 3 extractions distinct   (2,020 / 2,020 / 1,907 tokens; 11 / 11 / 13 propositions)
    temp 0.6   3 of 3 extractions distinct   (1,909 / 1,729 / 1,957 tokens; 14 / 14 / 18 propositions)

So forced greedy is MORE reproducible and still not reproducible — the endpoint itself is
nondeterministic. Any fix that only lowers the temperature will narrow this, not close it.

**A STALE PREMISE THIS FALSIFIES.** `section_enricher`'s module docstring justifies the
corrective-nudge retry with "an unmodified retry at `temperature=0.0` would re-read the same
prompt and write the same bytes." On this deployment it would not: the call runs at 0.6. The
retry's corrective line may still be worth having, but that argument for it is void — and it is
also a better explanation for the retry experiment's "membership churn" (three failures becoming
a different two at five attempts) than the ceiling alone.

**RECOMMENDED, NOT LANDED** (both are the owner's call, and the enrichment implementation is
frozen):
1. Let structured-extraction calls opt out of the `_eff_temp` nudge, measured against the looping
   failure it was introduced to prevent — that failure was observed on GENERATION, and whether it
   occurs on a schema-constrained extraction prompt has not been tested.
2. Stabilise the facet vocabulary across ingests of the same document — persist and reuse it, or
   canonicalise near-synonyms. Until then, a document's Phase-8 score stage is re-rolled on every
   re-ingest, and **Experiment B cannot attribute anything to summaries** while the vocabulary
   moves underneath it.

**HONEST LIMITS.** (1) One document, one model, three corpora; the temperature isolation is ONE
section and six calls. (2) The 0.95 quote floor is marginal rather than clear at 3,000 on this
document — 0.948 to 0.971 across seven ingests, and `section_enrich_quote_rate_low` correctly
fired on `rp1`. (3) Proposition comparison uses exact match plus closest-match similarity; a
reworded statement and an omitted one are distinguished by that similarity, not proven apart by
it. (4) Nothing here measures whether the variation changes an ANSWER — only what is stored.

### The pinned vocabulary makes a re-profile reproducible, and Phase-8 selection byte-stable (2026-11-03)

Both approved interventions landed as separately measurable changes — the sampling-task policy
(`2359572`) and the pinned facet vocabulary (`1ccab69`) — and were validated on ONE corpus in
THREE states: the first ingest, then two reindexes. A reindex is where the churn actually landed
in production: same bytes, fresh Phase-3 call, new facet ids, different Phase-8 selection for a
document nobody touched. Three fresh tenants would not test this at all — the vocabulary is per
(document, revision), and nothing is meant to be shared between separate KBs.

| | before (unpinned, nudged to 0.6) | after (pinned, greedy honoured) |
|---|---|---|
| facet vocabulary shared across runs | **1 of 10** (Jaccard 0.053-0.250) | **10 of 10** (Jaccard 1.000) |
| canonical labels drifting while the id held | n/a | **0** |
| per-section facet assignment | Jaccard 0.173, identical 2/38 | **Jaccard 0.925, identical 25/38** |
| APPLIED taxonomy (the probe set's source) | different every run | **identical** |
| key values (name, value) | Jaccard 0.546 | **0.854** |
| summaries | median 0.430, 0 identical | **median 0.887, 9/38 identical** |
| propositions | 109-121 exact of ~290 | **189 exact of ~278** |
| amendments | 2/2/2, **0** shared by all three | 4/3/3, **3** shared by all three |
| **Phase-8 selected set** | Jaccard 0.917-0.972, 33 of 36 stable | **Jaccard 1.000, identical set** |
| Coverage@Budget | 33 / 35 / 35 of 37 | **34 / 34 / 34** |

**THE PINNING DOES REAL WORK, and the second reindex is the proof.** Reindex 1's fresh profile
call proposed all ten canonical ids (`mapped: 10, unmapped: none`), so on that run the temperature
fix alone would have sufficed. Reindex 2 did NOT — it re-invented three of the ten as near
synonyms, and they were refused:

    proposed by reindex-2                       canonical, and kept
    data_retention_and_storage_management   ->  data_management_and_retention
    deployment_and_infrastructure_flexibility -> deployment_and_infrastructure_constraints
    licensing_and_environment_provisioning  ->  licensing_and_support_provisions

Unpinned, those three renames would have changed three of Phase 8's ten probes and three of its
facet-union join keys. `unmapped` records them as CANDIDATES for a controlled alias approval;
nothing was adopted automatically, and no similarity threshold was consulted.

**SELECTION IS NOW BYTE-STABLE IN SET.** The same 35 sections every state, the same 3 gold
sections missed every state, the same facet-union contribution (17) and the same score-only
admission (18). Selected ORDER still varies, which is the cross-encoder's scores on re-extracted
summaries and not the vocabulary.

**WHAT THIS DID NOT BUY: COVERAGE.** 34/37 is inside the unpinned range (33-35), so the gain is
REPRODUCIBILITY, not recall. And reproducibility cuts both ways: `14. Desktop/Console for LED
Video Wall` was missed in 1 of 3 unpinned runs and is now missed in 3 of 3. A stable pipeline
locks in its misses as firmly as its hits, which is the right trade for an auditable system and
should not be reported as an improvement in what the system can find.

**AMENDMENT STABILITY IS BETTER AND NOT SOLVED**, which is why it stays a separate dimension. All
three amendments are now found by all three states (0 before), but the first ingest found a
FOURTH that neither reindex did. So the earlier finding — each run omitting a different real
amendment — is improved, not closed, and nothing here should be read as vocabulary persistence
fixing amendment extraction.

**THE 0.95 QUOTE FLOOR IS STILL MARGINAL**, and unchanged by either fix: 0.957 / 0.950 / 0.951
across these three states, with the seven earlier 3,000-token ingests spanning 0.948-0.971. One of
those ten runs fell below and correctly warned. Kept at 0.95: that is evidence of marginal support
quality, not a reason to move the threshold.

**HONEST LIMITS.** (1) One document, one model, three states. (2) The two interventions are
separately COMMITTED but were validated TOGETHER here, so the per-section and summary improvements
cannot be attributed between them from this run alone — reindex 2's three refused renames are the
only clean attribution to pinning, and the `mapped: 10` of reindex 1 is the only clean sign that
greedy sampling alone can hold a vocabulary. (3) Phase-4 assignment stability (0.925, 25/38
identical) is still not 1.0: the model varies which sections a stable label describes, which the
vocabulary cannot fix and which still moves the facet union's membership. (4) An alias registry
exists and is honoured, with no self-service surface to populate it; today an approval is a
controlled catalog write.

### CORRECTION: two of twelve ingests fell below the quote floor, not one (2026-11-03)

The entry above says "0.957 / 0.950 / 0.951 across these three states ... One of those ten runs
fell below and correctly warned." **The third figure is wrong and the tally is wrong.** The third
state's rate is **0.949474** (451 of 475), which is BELOW the 0.95 floor. 0.951 was `rp3` from the
PREVIOUS run set — a different corpus state read out of the wrong file.

Every 3,000-token ingest, exact rather than rounded:

    bud1  475/496  0.957661      ea1  471/492  0.957317      rp1  472/498  0.947791  BELOW
    bud2  453/468  0.967949      ea2  488/510  0.956863      rp2  489/505  0.968317
    bud3  467/481  0.970894      ea3  476/498  0.955823      rp3  461/485  0.950515
    vc-s1 463/484  0.956612      vc-s2 458/482 0.950207      vc-s3 451/475 0.949474  BELOW

**12 ingests, 2 below the floor, range 0.947791-0.970894.** `section_enrich_quote_rate_low` fired
on exactly those two and nowhere else, so the gate was never wrong — the error was mine in
reading, and the claim "all three are at or above the threshold" does not hold for this corpus.

**THE LOGGED RATE NOW CARRIES SIX PLACES, not three.** The comparison always used the unrounded
rate, but the number an operator READS could not be trusted at the precision the decision turns
on: 0.949474 rounds to 0.949, and a rate of 0.9496 would round to 0.950 — reading as passing a
floor it fails. The low-rate warning also now carries `quotes_kept` and the shortfall, so the
exact fraction is on the line an operator acts on. The floor stays at 0.95: a thin margin is
evidence about support quality, not an argument for moving the threshold.

### Adjudicated: `14. Desktop/Console for LED Video Wall` is a GENUINE missed requirement (2026-11-03)

The newly reproducible miss, adjudicated against the aggregate question under the SAME specificity
standard the two accepted genuine misses were accepted under. **The frozen gold was not touched**,
and this does not change it — it confirms it.

| section | distinguishing cue | sections holding that cue | verdict |
|---|---|---|---|
| 20. Enterprise Cabinet/Rack | `IEC 60297` | **1** | genuine miss (accepted earlier) |
| 22. Firewalls | `different original equipment manufacturers` | **1** | genuine miss (accepted earlier) |
| 13. Video Wall Technical Requirements | `Video Wall`, `HBOM` | 4 | probable miss (accepted earlier) |
| **14. Desktop/Console for LED Video Wall** | **`EPEAT`** | **1** | **GENUINE MISS** |

`EPEAT` occurs in exactly ONE section of the corpus, and it is this one. An aggregate compliance
answer that omits this section cannot state that requirement at all, from anywhere. The section is
also substantive on its own terms rather than furniture: 9 key values (processor clocks, 16 GB
memory, 1 TB storage, 5-year warranty), 6 propositions (TPM 2.0 hardware security module, OEM-
licensed OS, four display outputs), a 301-character summary. Its five other cues (`Video Wall`,
`5 Years`, `PCI`, `TPM 2.0`) are each shared with 3-5 other sections and carry no weight here; the
verdict rests on `EPEAT` alone, which is the same shape as `IEC 60297` and
`different original equipment manufacturers`.

**SO THE REPRODUCIBLE MISS IS A REAL LOSS, not a harmless consequence of determinism.** Before
pinning it was missed in 1 of 3 runs; it is now missed in 3 of 3, and what is now stable is the
omission of the corpus's only EPEAT requirement.

**THE CAUSE IS DIFFERENT FROM THE OTHER THREE, which is the useful part.** Those three failed
Phase-4 ENRICHMENT outright. This section IS enriched — and Phase 4 assigned it **zero facet ids**
in all three states. With no facet label it cannot be reached by Phase 8's facet union, and its
cross-encoder score (0.0207 at the earlier measurement) is below the floor, so BOTH selection
paths are closed for a section that was read successfully. Only 3 of 38 enriched sections carry no
facet id and the other two are document furniture (`Copy to:`, the title page), so this is the one
substantive section in that state.

**NOT FIXED HERE, and deliberately.** The remedy is a Phase-4 or Phase-8 decision, not a gold
edit: either the assignment prompt should not be allowed to return an empty label set for a
section with nine key values, or Phase 8 needs a path that can reach an enriched-but-unlabelled
section. Both change selection for every document and belong to the owner.

**HONEST LIMIT.** The gold is cue-derived and unadjudicated BY CONSTRUCTION (see
`sec_gold_derive.py`), so "genuine" here means "meets the standard the two accepted misses met",
not "a human has validated the whole evidence set". The `EPEAT` cue itself comes from the answer
suite's own `must_contain` list, so it is a requirement the answer gold already demanded.

### Experiment B: the structure arms do not beat retrieval on recall, they beat it on PRECISION — and summaries reallocate rather than add (2026-11-03)

Seven runs against the 40-query authored lookup gold (`eval/gold/aegcl_section_lookup.json`), on
three frozen snapshots of one corpus sharing ONE pinned canonical vocabulary (`zzgoal-expb`,
preserved). All three snapshots reused the pinned vocabulary (`mapped: 10`, nothing unmapped),
which is a second independent confirmation of the pin on a fresh corpus.

| run | hit | hit@1 | standards (11) | lexical (12) | hierarchy (7) | mismatch (10) |
|---|---|---|---|---|---|---|
| **B1** hybrid retrieval | **35** | 26 | **11/11** | 12/12 | 5/7 | 7/10 |
| B2 ToC, snap 1 | 34 | **33** | 8/11 | 12/12 | **7/7** | 7/10 |
| B2 ToC, snap 2 | 32 | 31 | 8/11 | 12/12 | 5/7 | 7/10 |
| B2 ToC, snap 3 | 32 | 31 | 8/11 | 12/12 | 6/7 | 6/10 |
| B3 ToC+summaries, snap 1 | **36** | **34** | 10/11 | 12/12 | **7/7** | 7/10 |
| B3 ToC+summaries, snap 2 | 32 | 30 | 8/11 | 12/12 | 6/7 | 6/10 |
| B3 ToC+summaries, snap 3 | 32 | 30 | 8/11 | 12/12 | 6/7 | 6/10 |

**THE BUDGETS ARE COMPARABLE, and this was checked rather than assumed.** The structure arms may
name at most 3 sections; B1 was scored on everything it retrieved. B1's hit@3 is 35/40 and so is
its hit@8 — the median query retrieves exactly 3 distinct sections, so the cap never binds and the
comparison is at the same budget.

**THE CONTROL SETS THE RESOLUTION, and it is coarse.** B2's input is byte-identical across the
three snapshots (the ToC and the canonical ids are pinned), so its spread is the model's own
nondeterminism: **34 / 32 / 32**, and 37 of 40 queries answered identically all three times
(H-01, H-03 and M-07 flipped). A 40-query benchmark on this model cannot resolve a difference of
two records. Everything below is reported against that bar.

**WHAT SURVIVES IT.**

1. **The structure arms do NOT beat hybrid retrieval on recall.** 32-36 against 35. Inside the
   noise, in both directions.
2. **They beat it decisively on PRECISION:** hit@1 **30-34 against 26**. Every structure run beats
   B1's rank-1 accuracy by 4-8 records, far outside the spread. A ToC reader returns the right
   section FIRST; the retriever needs two or three tries.
3. **Standards lookup is retrieval's, consistently.** B1 11/11; the structure arms 8/11 in five of
   six runs. A standard id is lexically distinctive and absent from any heading, so BM25 finds it
   every time and a ToC reader must guess which section would contain it. The three it loses are
   `ISO 27001` (S-03, never found by any structure run), `MITRE` (S-05) and `TAXII` (S-11).
4. **Lexical value lookup is saturated: 12/12 for every arm in every run.** That stratum
   discriminates nothing and should be shrunk or replaced in any future gold.
5. **Nobody solves semantic mismatch: 6-7/10 everywhere**, including the arm with summaries.

**SUMMARIES REALLOCATE RATHER THAN ADD, which the aggregate hides.** B3 minus B2 is +2 / 0 / 0 —
nothing, at this resolution. But the per-record pattern is not noise:

    S-06 PeopleCert    B2 misses 3/3, B3 finds 3/3   <- summaries WIN, replicated
    S-09 MAAT 19.44    B2 misses 3/3, B3 finds 3/3   <- summaries WIN, replicated
    S-05 MITRE         B2 finds 3/3, B3 misses 2/3   <- summaries LOSE
    S-11 TAXII         B2 finds 3/3, B3 misses 2/3   <- summaries LOSE
    M-10 headroom      B2 finds 3/3, B3 misses 2/3   <- summaries LOSE

So summaries decide two records the ToC cannot (a certification scheme and a turnover figure
appear in a summary and in no heading) and cost three the ToC already had, which is consistent
with 53 added summaries acting as distraction on queries the headings alone answered. A net of
zero is the sum of a real gain and a real loss, not an absence of effect.

**TWO QUERIES NO ARM EVER ANSWERS, and they fail differently.**

* **M-06** ("what certification must the *industrial security gateway* hold?" -> `18. Data Diode`,
  IEC 62443-4-2). Every structure run answers `22. Firewalls` — a plausible reading of "security
  gateway" that is simply wrong. A genuine semantic failure.
* **M-09** ("how long after acceptance must the supplier keep issuing software fixes?" -> the
  SOAR / UEBA / TAP sections carrying "Support for Patches and Upgradation, 5 years"). **All six
  structure runs answer `71 Warranty & Support`** — unanimously, and that section holds a warranty
  term rather than the patch obligation. Six-for-six agreement against the gold is a signal about
  the GOLD, so this is recorded as an OPEN ADJUDICATION. **The gold was not changed**: relabelling
  after seeing the result is the one thing a frozen benchmark must never do.

**HONEST LIMITS.** (1) One document, one model, 40 queries; the per-stratum cells are 7-12
records, so a one-record difference inside a stratum means nothing. (2) B2/B3 are a ZERO-SHOT
prompt over a ToC — STAIR fine-tunes a reader for this task, and nothing here tests that. (3) The
gold's L, H and M strata are authored from the enrichment and are blind where it is; stratum S is
not, and S is where the arms differ most. (4) This measures FINDING the section, not answering
from it: no faithfulness, no citations, no answer quality. (5) B1 ran against the snapshot-3 index
only, which is correct (it retrieves from text, not from summaries) but means its own run-to-run
variability is unmeasured.

### Experiment C: fusion beats both components, and summaries help INSIDE the fusion (2026-11-03)

Equal-weight Reciprocal Rank Fusion of the two rankings, using the SHIPPED
`reciprocal_rank_fusion` (k=60) that Phase-0 already fuses dense and sparse with. Rank-based on
purpose: the components produce a cross-encoder score and a generation, and calibrating those
against each other is what 40 queries cannot support. **No weights were tuned.** The budget is
three sections for every arm — an unbounded union would make fusion succeed by construction.

Nine instrumented runs (B1, B2, B3 x three snapshots); the fusion is computed offline from their
ranked lists, so C3 costs exactly C1 plus C2 and no extra model call.

| arm | first-choice (P@1) | spread | recall@3 | spread | nDCG@3 | s/query | tokens |
|---|---|---|---|---|---|---|---|
| **C1** retrieval | 26.0 | **0** | 35.0 | **0** | 0.7725 | **0.24** | **0** |
| **C2** structure | 31.7 | 2 | 32.3 | 3 | 0.8022 | 5.52 | 1177 |
| **C3** B1+B2 | 30.7 | 2 | 37.3 | 1 | 0.8622 | 5.77 | 1177 |
| **C4** B1+B3 | **31.7** | 2 | **38.0** | **0** | **0.8731** | 5.88 | 1174 |

| arm | standards (11) | lexical (12) | hierarchy (7) | mismatch (10) |
|---|---|---|---|---|
| C1 | **11/11** | 12/12 | 5/7 | 7/10 |
| C2 | 8/11 | 12/12 | 5-7/7 | 6-7/10 |
| C3 | **11/11** | 12/12 | 6-7/7 | **8/10** |
| C4 | **11/11** | 12/12 | **7/7** | **8/10** |

**FUSION BEATS BOTH COMPONENTS, on every measure except cost.** C4's recall@3 is 38.0 against
C1's 35.0 and C2's 32.3; its nDCG@3 is 0.873 against 0.773 and 0.802. It keeps retrieval's
standards (11/11, which C2 loses) AND structure's hierarchy (7/7, which C1 loses), and it reaches
8/10 on semantic mismatch — one more than EITHER component, because the two miss different
records. The per-query detail is explicit: H-01, H-07 and M-02 are structure-only wins; M-01,
S-03, S-06 and S-09 are retrieval-only wins; the fusion takes all seven.

**IT IS ALSO ALMOST LOSSLESS.** Across six fusion runs, exactly ONE query that a component had at
top-3 was lost by the fusion (H-03, C3, one run). C4 lost none in any run.

**SUMMARIES HELP IN FUSION WHERE THEY DID NOT HELP ALONE, and this is the reversal worth noting.**
B3 minus B2 was +2 / 0 / 0 — nothing at this resolution. C4 minus C3 is +0.7 recall, +1.0
first-choice, +0.011 nDCG, hierarchy 7/7 against 6-7/7, and a recall spread of 0 against 1, in
EVERY run. The same enrichment that could not beat the ToC on its own improves the fused ranking
consistently. A plausible reading is that RRF needs only a USEFUL ORDERING from each component and
is indifferent to the distraction that cost B3 its wins outright; that is an interpretation, not
something these runs establish.

**C1 IS PERFECTLY REPRODUCIBLE**, which Experiment B had to list as unmeasured: three runs,
40/40 identical top-3, zero spread on every metric. So all run-to-run variation in C2/C3/C4 is the
model's. C4 is the most stable fused arm — 40/40 identical hit/miss, against C3's 38/40.

**THE COST IS THE FINDING'S OTHER HALF.** C4 is **24x C1's latency** (5.88 s against 0.24 s per
query) and spends 1,174 completion tokens where C1 spends none. For +3 recall@3 and +5.7
first-choice on 40 queries. Whether that trade is worth making is a product decision about which
queries deserve it, not a retrieval result — and nothing here measures it on a corpus where the
ToC does not fit in one prompt.

**TERMINOLOGY, CORRECTED.** Experiment B's entry above calls the hit@1 result a win on
"precision". That is wrong and is left visible: hit@1 is FIRST-CHOICE SUCCESS, and nothing in
either experiment establishes that the three-section result is more precise. The structure arms
put a relevant section first more often; what the other two slots contain was never scored.

**HONEST LIMITS.** (1) 40 queries, one document, one model; per-stratum cells are 7-12 records.
(2) The three snapshots differ only in their summaries, so C1 and C2 are each the same system run
three times — their spread is reproducibility, not sampling over corpora. (3) RRF k=60 and equal
weights were fixed in advance and not tuned; a tuned fusion might do better or might just fit 40
queries. (4) Still zero-shot: STAIR fine-tunes a reader and nothing here tests that. (5) This
measures FINDING a section, not answering from it.

### The quote floor's claims were unrecoverable, and now are not (2026-11-03)

"Investigate the below-floor claims individually" could not be done. `parse_section_enrichment`
has always returned `unquoted_key_values` and `unquoted_propositions` — the items dropped because
their quote is not in the section text — and **nothing read them**. The warning said
`quote_rate=0.948, quotes_seen=498` and there was no record anywhere of WHICH of the 26 claims the
document could not support. Three of fourteen 3,000-token ingests (21.4%) landed below the floor
and all three are now unexaminable after the fact.

Fixed: a per-section `section_enrich_quote_dropped` line naming the dropped key-value names and
bounded heads of the dropped propositions, and `EnrichReport.sections_with_unquoted` so the
document-level warning points at the sections that produced the shortfall. Masked-derived, the
same standing as `reply_head` — the model only ever saw masked text.

The floor stays at 0.95, and a passing retrieval benchmark is not evidence that support quality
has stabilised: the two measure different things, which is the distinction the whole
quote-rate-versus-completeness thread exists to keep.
