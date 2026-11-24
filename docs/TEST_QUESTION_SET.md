# Test question set — `acme__test` knowledge base

A question bank for exercising the platform against the nine real Acme documents in
`acme__test`, in the three shapes that stress different parts of the pipeline:

| Category | What it tests | Result |
|---|---|---|
| **A. Static / single-fact** | Exact retrieval of one value from one document | 9/9 answered |
| **B. Multi-document** | Evidence assembled across two or more documents | 5/6 answered |
| **C. Whole-document context** | Breadth over an entire document, not one passage | 5/6 answered **[9-11]** |
| **D. Controls** | The platform must *refuse* these | 3/3 correct |

**Every question below was run against the live stack and the result recorded.** Nothing here
is predicted. Where a question fails, it says so and why — those are the ones to know about
before a client demo, not the ones that work.

Run date: 2026-09-10 · KB `acme__test` · `top_k=5` · model `Gemma-31B` · faithfulness `nli`.

**Re-measured 2026-09-11** after two platform changes: the reranker moved from the foreign
`jina-reranker-v3.5` to the self-hosted `bge-reranker-v2-m3` (abstain threshold recalibrated
0.50 → 0.23, see `docs/calibration.md`), and the faithfulness anchor became IDF-weighted. Rows
whose result changed are marked **[9-11]**; everything else re-ran identically.

---

## How to read the results

Each question shows the **status**, the **faithfulness score**, and **how many distinct
documents** were cited.

- **grounded** — answered with citations. This is the pass state.
- **abstained** — the platform declined. Correct for a control question, a failure otherwise.
- **faithfulness** — the share of the answer's sentences an independent NLI checker could verify
  against the retrieved passages. `1.0` = every sentence verified. Below `0.4` the answer is
  withheld entirely. A score of `0.5` on a correct answer usually means the model added a
  framing sentence the sources do not literally contain — not that it invented a fact.

### Two things that will confuse you while testing

1. **The semantic cache is ON.** Ask a question, then ask a near-identical rephrasing, and you
   get the *first* answer back verbatim — the pipeline never runs. Confirmed while building this
   set: two differently-worded questions returned byte-identical answers with `cache_hit: true`
   in the audit trail, one of which had abstained minutes earlier on a cold run. **When testing
   a change, vary the question meaningfully or clear the cache** — otherwise you are measuring
   the cache, not the platform.
2. **Faithfulness is a second opinion, not the answer's correctness.** A correct answer can score
   0.5 and a wrong one could in principle score high. Read the citations.

---

## The corpus

Nine documents. The **effective date** column is what the platform derived at ingest — it drives
which version wins when two documents disagree.

| Document | Derived effective date |
|---|---|
| `HR_Handbook 4.0_2026 2_0227.pdf` | 2026-01-01 |
| `Acme_HR_POLICY_HANDBOOK_1772449486984.pdf` | 2025-01-01 |
| `Acme_Night_Shift_Allowance_Policy__1772449505662.pdf` | 2023-04-01 |
| `Abstract_1772449455561.pdf` | 2021-08-31 |
| `Code_Of_Conduct_1772449462307.pdf` | — |
| `Customer_Testimonial_Reward_Policy_1772449467807.pdf` | — |
| `Acme_Business_Travel_Policy_1772449474018.pdf` | — |
| `Acme_Torch_Bearers_Policy_1772449511513.pdf` | — |
| `Facility_Staff_Child_Education_Reimbursement_Policy_...pdf` | — |

A dash means the document states no release date **in a form the extractor reads** — it looks for
a release-history block (`Version | Prepared | Review | Release`). Known gap: the Child Education
policy *does* print `Publish Date | 1st April 2024` in a different table shape and is not picked
up. An undated document is treated as UNKNOWN, never as old, so it is never silently buried.

---

## A. Static / single-fact questions

One document, one value. These test retrieval precision and whether the number survives
end-to-end. **9/9 grounded.**

| # | Question | Result |
|---|---|---|
| A1 | What is the night shift allowance rate per day? | ✅ grounded · **1.00** · 1 doc — "₹200 per day" |
| A2 | What is the effective date of the Night Shift Allowance Policy? | ✅ grounded · **1.00** · 1 doc — "1 April 2023" |
| A3 | Within how many hours must a travel advance be settled? | ✅ grounded · **1.00** · 1 doc — "48 hours" |
| A4 | Which employee grades are permitted to take a travel advance, and for trips longer than how many days? | ✅ grounded · **1.00** · 1 doc — "M1–M7", "more than 7 days" |
| A5 | How much is reimbursed for Graduation (FY to TY/BE) under the Facility Staff Child Education Reimbursement Policy? | ✅ grounded · **1.00** · 1 doc — "₹30,000" |
| A6 | What perks do employees receive under the Torch Bearers Policy? | ✅ grounded · **1.00** · 1 doc — leadership training, annual vacation, self-growth books |
| A7 | Who is eligible for the Customer Testimonial Reward Policy? | ✅ grounded · 0.48 · 1 doc — full-time employees, consultants, trainees, interns; Sales/Presales only |
| A8 | What are the core values of Acme? | ✅ grounded · 0.71 **[9-11]** · 1 doc — all six returned |
| A9 | Who are the members of the work from home committee? | ✅ grounded · 0.42 · 2 docs — all five, current version first |

**A8 and A9 are the two worth demoing.** Both are lists that span more than one passage. Before
neighbour expansion they returned 2 of 6 and 2 of 5 — the middle of a list shares almost no
vocabulary with the question, so it ranks below passages that merely mention the topic, and
raising `top_k` does not help because the problem is ranking, not depth. Both are now complete.

**A9 also demonstrates version conflict**: two handbooks list *different* committee members. The
answer leads with the members in force under the 2026 handbook and records that the 2025 one is
superseded. Its 0.42 score reflects that the supersession sentence is a platform statement, not a
sentence lifted from any single source.

**A7's 0.48** is the same shape: the eligibility list is verified, the framing sentence around it
is not literally in one passage.

---

## B. Multi-document questions

The answer does not exist in any single document. **5 of 6 answered.**

| # | Question | Result |
|---|---|---|
| B1 | Who approves a business travel expense claim, and who approves a facility staff child education reimbursement? | ⚠️ **[9-11]** grounded · 1.00 · 1 doc — now answers the travel half and says the sources do not state the other. Asked alone, the child-education half retrieves at 0.99; the two halves compete for 5 slots. |
| B2 | What eligibility conditions does an employee have to meet for the Torch Bearers Policy and for the Customer Testimonial Reward Policy? | ✅ grounded · **1.00** · **2 docs** |
| B3 | What does the Code of Conduct say about expected employee conduct, and what does the HR Policy Handbook say about disciplinary action? | ✅ grounded · **0.97** · **3 docs** |
| B4 | Which Acme policies offer a financial benefit to employees, and what does each one pay? | ✅ grounded · 0.62 · 1 doc |
| B5 | Compare what Acme reimburses under the Business Travel Policy with what it pays under the Night Shift Allowance Policy. | ✅ grounded · 0.56 · 1 doc |
| B6 | Which Acme documents describe a reward or recognition scheme, **and what does each one recognise?** | ❌ **abstained** — see below |
| B6′ | Which Acme documents describe a reward or recognition scheme? | ✅ grounded · 0.91 **[9-11]** · **2 docs** |

**B1–B3 are the ones to demo** — genuine cross-document assembly, three documents on B3.

**B4 and B5 answer, but from one document.** The handbook happens to summarise the other policies,
so the model answered from it rather than assembling across sources. The answer is correct; the
citation breadth is narrower than the question implies. Worth knowing before you claim
"synthesised from five documents" on stage.

**B6 is a real limitation, and B6′ is the same question that works.** Adding *"and what does each
one recognise?"* makes it abstain. The cause: the platform splits a compound question into its
sub-questions and refuses to answer a half the corpus does not cover — a deliberate protection
against inventing an answer for the unanswered half of a two-part question. But this sub-question
asks about *documents* ("which documents describe…", "what does each recognise"), and words like
"documents", "describe" and "scheme" appear in no policy passage, so the sub-question reads as
uncovered even though the content is plainly there. Corpus-level "which documents…" questions are
the known-unbuilt GraphRAG whole-KB search; **do not put B6 in a client demo.**

---

## C. Whole-document context questions

Breadth over an entire document. **6/6 grounded.**

| # | Question | Result |
|---|---|---|
| C1 | Summarise the Code of Conduct. | ✅ grounded · **1.00** · 2 docs |
| C2 | What topics does the Acme HR Policy Handbook cover? | ✅ grounded · **1.00** · 1 doc |
| C3 | Give me an overview of the Business Travel Policy from start to finish. | ✅ grounded · **1.00** · 1 doc |
| C4 | What are the main sections of the Torch Bearers Policy? | ❌ **[9-11] abstained** — retrieval returns 5 passages of the right document but all are body text (0.10–0.34), so the model declines to list "sections". Same shape as B6: a question about a document's STRUCTURE, not its content. |
| C5 | What is the purpose and scope of the Facility Staff Child Education Reimbursement Policy? | ✅ grounded · **1.00** · 1 doc |
| C6 | Walk me through the entire Night Shift Allowance Policy — its objective, who is eligible, the rate, and the process. | ✅ grounded · 0.96 · 1 doc |

**C6 is the best single demo question in the whole set**: one question, four distinct facts, all
correct, all cited.

**C4 is the exception, and it is the same limitation as B6** — a question about a document's
structure rather than its content. Ask "what perks do Torch Bearers receive?" (A6) instead, which
answers at 1.00.

---

## D. Control questions — the platform must refuse

An answer here is a **failure**. These prove grounded-or-abstain is real and not a slogan.

| # | Question | Result |
|---|---|---|
| D1 | What is Acme's policy on cryptocurrency trading by employees? | ✅ **abstained** — "the answer isn't available in this knowledge base" |
| D2 | What is the sabbatical leave entitlement for employees who have completed twelve years of service? | ✅ **abstained** |
| D3 | What is the night shift allowance rate, and what is the FY2027 cloud infrastructure budget? | ✅ answered the first half ("₹200 per day"), and stated plainly that the sources do not cover the second |

**D2 is the harder control.** D1 is obviously foreign to an HR corpus; D2 sounds exactly like
something an HR handbook *would* say, and the corpus is full of leave and service-tenure language
for a model to pattern-match against. It still refuses. D1 and D2 both abstain at the retrieval
gate — before generation — which is why they carry no faithfulness score at all.

**D3 is the most persuasive question in this document for a sceptical audience.** It is a trap: a
real question welded to one the corpus cannot answer. The platform answers what it can and refuses
the rest, in the same breath, rather than inventing a budget figure to look complete.

---

## Suggested 6-question demo run

If you have ten minutes: **A1** (it just works) → **C6** (breadth, four facts, one question) →
**A8** (a list that used to come back incomplete) → **B3** (three documents) → **A9** (two
handbooks disagree; the platform knows which is in force) → **D3** (it refuses to invent).

Avoid **C4** and **B6** on stage: both ask about document structure, and both abstain.

Open on A1, close on D3.
