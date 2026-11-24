# RAG as a Service — Platform Guide

**What this is:** everything you need to explain the platform to a customer, to management, or
to an architect — in plain language and in technical detail, side by side.

**Registry ID:** REG-2026-020 · **Read time:** ~45 minutes · **Demo time:** 20 minutes

| Part | For | What it covers |
|---|---|---|
| **A** | Everyone | What the product is, in plain language |
| **B** | Presenter | The 20-minute demo — what to click, what to say |
| **C** | Everyone | All eight modules: what each does, and how |
| **D** | Technical | Deep dives — chunking, embedding, retrieval, grounding |
| **E** | Architect | Architecture and the diagram set |
| **F** | Commercial | Pricing: subscription, tokens, vector store, GPU, storage |
| **G** | Presenter | Hard questions and honest answers |

---
---

# PART A — The product, in plain language

## A1. The one-sentence version

> **A customer puts their own documents in. Their staff ask questions in plain language. The
> platform answers using only those documents, and shows you the exact page it took the answer
> from — or it says it doesn't know.**

Everything else in this guide is detail underneath that sentence.

## A2. The problem we solve

An Indian bank has 30 years of circulars. A pharmaceutical company has thousands of SOPs and
batch records. A government department has scheme guidelines going back decades. In every case
the knowledge exists — and staff cannot find it. They ask a colleague, or they guess, or the
knowledge simply goes unused.

The obvious answer is "use AI". Three things stop a regulated enterprise from doing that:

| | The blocker | What it means in practice |
|---|---|---|
| 1 | **It makes things up** | A general AI assistant will produce a fluent, confident, wrong answer. A compliance officer cannot act on an answer they cannot trace to a source. |
| 2 | **The data leaves the country** | Using a foreign hosted model means the customer's documents travel outside Indian jurisdiction. For BFSI, PSU, government and pharma, that is frequently a hard legal blocker, not a preference. |
| 3 | **No one can audit it** | No record of what was asked, what was answered, or on what basis. No respect for who is allowed to see what. |

**The key insight to communicate:** the constraint is not *intelligence*, it is **admissibility**.
A brilliant answer nobody can defend to a regulator is worth nothing here.

## A3. The three promises

Everything the platform does comes back to three commitments. Learn these three — they are the
whole pitch.

### Promise 1 — Grounded, or it stays silent

Every answer carries a citation to the exact passage it came from. If the platform cannot find
good enough evidence, **it refuses to answer** rather than guessing.

> **Say it like this:** "Most AI tools are optimised to always give you an answer. Ours is
> optimised to only give you an answer it can prove. If it doesn't know, it tells you it doesn't
> know — and in your industry, that's the feature, not the limitation."

**Why it's credible, not just a claim:** this isn't a filter we apply at the end and hope holds.
It is built into the *data structure* of an answer. An answer object with no citation attached
**cannot be created** — the code refuses to construct it. A future developer cannot accidentally
remove the check, because there is no check to remove; there is only a shape that an uncited
answer doesn't fit.

### Promise 2 — Sovereign by construction

Every part runs inside India, on Acme infrastructure. No document, no question, no answer leaves
the country.

> **Say it like this:** "Your documents never leave Indian jurisdiction. And that's not a policy
> we've written down — it's a check the software performs on every single outbound connection,
> and it fails closed. If a component ever tried to call somewhere it shouldn't, the request
> doesn't go out."

### Promise 3 — Governed by default

Every action is recorded on a tamper-evident log. Every document carries a sensitivity level.
Every person carries a clearance. A person sees a document only if they are cleared for it —
**and an administrator cannot override that.**

> **Say it like this:** "Your existing document classification carries into the AI. If someone
> isn't cleared to read a Restricted file today, they can't get its contents out of the assistant
> either — including by asking cleverly."

## A4. A real worked example — use this in the demo

This is a genuine result from the platform, and it demonstrates all three promises at once.

**The question:** *"Who are the members of the Work from Home committee?"*

**The document** contains, on one page:
```
Work from Home Committee:
Committee Members:
1. the Chief Human Resources Officer
2. Mr. Akshay Joshi (Vice-President – HR)
3. Mr. Rushikesh Aher (General Manager - HR)
4. Ms. Ashi Nair (Junior Associate - HR)
5. Mr. Lovelesh Gahlod (Manager)
```

**What the platform returned, before the fix:** the members it could see, **plus the sentence
"The sources do not list the remaining members."**

**Why this is the best thing to show a client.** A normal AI would have invented three plausible
Indian names to complete the list, and you would never have known. Ours told you where its
knowledge ended.

**After the indexing fix was deployed and the document re-indexed** (verified end-to-end against
the running platform on 2026-09-10): the same question returns **all five members, each cited to
page 91, faithfulness 1.00.**

> **The line to use:** "This is what safe looks like. It would rather be visibly incomplete than
> invisibly wrong."

---
---

# PART B — The 20-minute demo

## B0. Before you start

| Check | Why |
|---|---|
| One knowledge base with a **real, recognisable** document | A customer's own policy handbook beats a synthetic file every time |
| One question you know answers well | Rehearse it — don't discover a bad answer live |
| One question you know it **abstains** on | This is your strongest moment, not your weakest |
| Two logins at different clearance levels | For the access-control demo in B5 |

## B1. Open on the problem, not the product *(2 min)*

Don't start with the console. Ask the room:

> *"How long does it take someone in your compliance team to answer a question that's definitely
> in a policy document somewhere — but they don't know which one?"*

Let them answer. Then: *"And when they do find it, how do they prove to an auditor that's where
it came from?"* That second question is the one nobody has a good answer to. **That's your
opening.**

## B2. Ingest a document *(3 min)*

Upload a document live. While it processes, narrate what is happening — this is where you earn
technical credibility:

> "It's reading the document — text pages directly, scanned pages through OCR. It's finding
> personal data — Aadhaar numbers, PAN, card numbers — and masking them *before* anything is
> stored, so the raw identifier never enters the search index. It's noticing that this document
> contains identity data and automatically raising its classification. Then it's splitting it
> into passages, and turning each one into a mathematical fingerprint so we can search by
> meaning, not just keywords."

**If asked "how long for our 100 GB?"** — around 4–8 hours of GPU time, once. It is a one-time
cost at onboarding, not an ongoing one.

## B3. Ask the question *(4 min)*

Ask your rehearsed question. When the answer comes back, **do not read the answer aloud** — the
room can read. Point at three things instead:

1. **The citation.** Click it. The source document opens with the exact passage highlighted.
   *"That's the difference. You're not trusting the AI — you're checking its work in two seconds."*
2. **The faithfulness score.** *"A second, independent model has read the answer and checked it
   against the sources. This isn't the AI marking its own homework."*
3. **The status badge — Grounded.** *"There are three states: Grounded, Flagged for review, and
   Abstained. This one cleared both bars."*

## B4. Ask something it can't answer *(3 min)* — **the most important 3 minutes**

Ask something plausible but genuinely absent from the corpus.

> *"Watch what it does here. It won't guess."*

It abstains. **Sit in the silence for a moment.** Then:

> "Every AI tool demo you have seen this year showed you the answers. I'm showing you the
> refusal, because in your business that is the expensive case. This is the behaviour you're
> actually buying."

**If someone says "so it's less capable?"** — *"It's equally capable and more honest. It found
nothing relevant, so it told you. The alternative isn't a better answer; it's a wrong answer you
don't detect."*

## B5. Show the governance *(5 min)*

This is what closes a regulated buyer.

**Access control** — log in as a lower-clearance user, ask the same question. Different, narrower
answer, or none.
> *"Same question, same platform, different person. Your existing document classification is
> enforced inside the AI. And note — this isn't the admin choosing to be careful. An administrator
> with low clearance cannot see Restricted content either. It's a mandatory control."*

**Audit trail** — open it.
> *"Every question, every answer, every document load — on a tamper-evident chain. If any entry
> were altered, the chain breaks and we can prove it. This is what you hand your auditor."*

**Compliance view** — open the framework mapping.
> *"Live mapping against the DPDP Act, NITI Aayog's Responsible AI principles, ISO 42001 and the
> OWASP LLM Top-10 — computed from the system's actual configuration right now, not a PDF someone
> wrote last year."*

**Right to erasure** — show it if they're DPDP-sensitive.
> *"A person exercises their right to erasure. It cascades through every store, including the
> chat history that quoted them, and issues a signed certificate."*

## B6. Close on scale and sovereignty *(3 min)*

> "This runs on Acme infrastructure in India — your documents, your questions, your answers, none
> of it leaves the country. It's multi-tenant, so your departments each get their own isolated
> knowledge base, and one department cannot see another's documents unless someone explicitly and
> revocably grants access. And it scales from one department to terabytes on the same design."

**Then stop talking and ask:** *"What's the first document set you'd want to try this on?"*

## B7. Demo failure recovery

| If this happens | Say this |
|---|---|
| A bad or thin answer | "Good — let's look at why." Open the citations. Usually the document genuinely doesn't say it. **Turn it into the abstain story.** |
| It's slow | "That's a real answer being generated on a CPU tier — production runs on GPU at a fraction of this." |
| An unexpected abstention | **This is a win, not a failure.** "That's the guardrail doing its job in front of you." |
| Someone asks something you don't know | "I don't want to guess at that — let me get you the accurate answer." **Model the product's own behaviour.** It lands well. |

---
---

# PART C — The eight modules

The platform is eight layers. Layers 1–3 turn documents into something searchable. Layer 4 finds
the right passages. Layer 5 writes the answer. Layers 6–8 orchestrate, guard, and govern.

Each module below has **What it does** (say this to anyone) and **How it works** (say this to an
architect).

---

## Module 1 — Connectors & Ingestion

**What it does.** Gets the customer's documents into the platform, from wherever they live.

**How it works.**

| Route | Detail |
|---|---|
| **Direct upload** | Through the console or the API. |
| **Object storage** | Pulls by reference from S3-compatible storage — including a customer's *own* bucket, whose endpoint is checked against the sovereignty allow-list first. |
| **Web crawl** | Bounded breadth-first crawl from a seed URL, with depth and page caps, same-host scope, and robots.txt respected. Every single hop is sovereignty-checked. |
| **Scheduled sync** | Watches a folder and re-syncs on a schedule. |

**The detail worth mentioning:** every document is SHA-256 hashed at ingest. Re-ingesting
unchanged bytes **skips extraction, embedding and indexing entirely** — a re-sync of a
100,000-document folder where three files changed costs three files of work, not 100,000.

> **Plain language:** "Point it at your document store. It only does real work on what actually
> changed."

---

## Module 2 — Document Intelligence

**What it does.** Reads the document — including scans — and removes personal data before
anything is stored.

**How it works.**

- **Text extraction.** Digital pages are read directly. Scanned or image-bearing pages are routed
  to OCR and merged back **by page**, so citations still name the right page.
- **Layout, tables and entities** are extracted, not just flat text.
- **Indic scripts** are supported across multiple languages.
- **PII detection and masking**, and this is the part to emphasise:

**PII masking runs BEFORE chunking.** That ordering is the whole point. If you masked afterwards,
the raw Aadhaar number would already be sitting inside the vector index. Because masking runs
first, **only de-identified text is ever embedded or indexed.**

**Detection is checksum-validated, not pattern-matched.** Aadhaar numbers are validated by the
Verhoeff algorithm, card numbers by Luhn. A 12-digit invoice number is *not* mistaken for an
Aadhaar. This matters: a naive regex tool floods the customer with false positives and gets
switched off.

Three redaction modes the operator chooses:

| Mode | Output | When to use |
|---|---|---|
| `redact` | `[REDACTED:AADHAAR]` | Default — safest |
| `mask` | `XXXX XXXX 4521` | When staff need to confirm which record |
| `hash` | A stable token — same value always produces the same token | When you need to see that the *same* person recurs, without knowing who |

**Automatic classification.** If a document's detected PII implies a higher sensitivity than the
uploader claimed, the platform **raises the classification automatically** — and if the required
level is above the uploader's own clearance, the upload is **blocked**. You cannot store
something more secret than you are trusted to read.

> **Plain language:** "It reads scans, it understands tables, and it strips out personal data
> before that data can ever reach the search index. And it's smart enough to tell an Aadhaar
> number from an invoice number."

---

## Module 3 — Indexing

**What it does.** Splits documents into passages and turns them into searchable form.

Covered in full in **Part D** — this is the module customers ask about most.

Headlines: heading-aware splitting; content-derived passage IDs so a re-index doesn't break
stored citations; and **two** indexes built in parallel — one for meaning, one for exact words.

---

## Module 4 — Retrieval Engine

**What it does.** Given a question, finds the passages most likely to answer it — and filters out
everything the asker isn't allowed to see, before anything else happens.

**How it works** — four stages, covered in depth in **D3**:

1. **Two searches in parallel** — semantic (meaning) and keyword (exact terms).
2. **Fuse** the two result sets by rank.
3. **Re-score** the survivors with a more expensive, more accurate model.
4. **Filter by permission** — then, and only then, trim to what goes into the prompt.

**The ordering point:** access filtering happens **before** the top-k trim and **before** the
prompt is built. An unauthorised passage is never a candidate, so it cannot influence an answer
even indirectly.

---

## Module 5 — Generation & Grounding

**What it does.** Writes the answer, attaches the citations, and decides whether the answer is
good enough to show at all.

**How it works.**

- The prompt carries, for each source: **document, page, and the governing section heading.**
- The model is asked to cite by index. **The platform resolves those indices back to passages
  itself** — it does not trust the model's citation claims.
- Source blocks are fenced, and fence-like text inside a document is neutralised, so **a document
  cannot impersonate the prompt's own structure**. Retrieved content is treated as untrusted
  input, because it is.
- The confidence gate runs before anything is returned.

**Why the section heading matters — a real bug this fixed.** A rule reading *"All content must be
approved before release"* was applied to a sales deck. Read alone the rule *is* general. The
limitation lived in the heading — **6.2.3 Social Media Content** — which was in a different
passage. The platform now carries the governing heading with the passage into the prompt.

> **Plain language:** "It doesn't just find the rule — it carries the heading the rule sits under,
> so the AI knows what the rule actually applies to."

---

## Module 6 — Agentic Orchestration

**What it does.** Runs multi-step processes, and routes uncertain answers to a human.

**How it works.** A declarative workflow of steps whose branching is decided by a **transparent
rules engine** — typed conditions over named values — **not** a black-box AI agent deciding what
to do next.

> **Say this to a regulated buyer:** "When this thing takes a decision, you can read why. It's a
> recorded decision trace, not 'the model felt like it'. In your industry that difference is the
> difference between usable and not."

Every step is re-checked against the caller's role before it executes, so a workflow cannot be
used to do something the person couldn't do directly. Runs are bounded by a step limit.

**Human-in-the-loop:** answers that land between the confidence thresholds go to a reviewer
queue, where a human can accept, modify, flag or override — every action logged.

---

## Module 7 — Guardrails & Evaluation

**What it does.** Screens what goes in, checks what comes out, and measures quality continuously.

**How it works.**

| Guard | When | What |
|---|---|---|
| Prompt-injection screening | Before retrieval | A blocked question abstains with a recorded reason |
| Fence neutralisation | Prompt assembly | A document cannot forge prompt structure |
| Toxicity screening | After generation | On the output |
| **Faithfulness verification** | After generation | An independent model checks the answer against its sources |

**Faithfulness is scored as `1 − P(contradiction)`, not `P(entailment)`.** That is a deliberate
choice worth explaining: a good answer paraphrases and summarises rather than quoting. Demanding
strict entailment would fail perfectly good answers. Asking instead *"do the sources contradict
this?"* is the right question — and it works even when the answer and the source are in different
languages.

Three outcomes: **Grounded** / **Flagged for review** / **Abstained**.

**The release gate.** Before a new build reaches production it is scored against a golden set —
citation coverage, abstention correctness, whether answers came from the right documents, recall
and ranking. The metrics are **deliberately arithmetic, not AI-judged**: a quality gate that needs
an AI to decide whether the build passes inherits that AI's randomness, and a threshold against a
random number means nothing.

---

## Module 8 — Control Plane

**What it does.** Everything around the AI: who you are, what you can see, what it cost, what
happened.

**How it works.** Identity (OIDC/SSO with MFA, local accounts, scoped API keys) · role-based
access · per-knowledge-base roles and clearances · KB managers with delegated authority ·
cross-KB access requests and grants · org-wide model registry with an approval workflow ·
usage metering · budgets and caps · and the tamper-evident audit chain.

**~200 API endpoints.** Everything in the console is also an API — nothing is UI-only.

---

## Beyond the console: how else it's consumed

| Surface | What it's for |
|---|---|
| **REST API `/v1`** | Any system integration |
| **Python & TypeScript SDKs** | For the customer's developers |
| **Embeddable widget** | Drop-in chat on the customer's own intranet or portal |
| **MCP server** | AI agents consume the platform as a governed tool — inheriting every control |
| **Chat-platform bot** | Answers in a Mattermost/Slack-shaped channel |

**The widget point worth making:** the API key **never reaches the browser**. The host page's
*server* exchanges it for a short-lived token bound to one knowledge base, reader-only.

---
---

# PART D — Deep dives

This is the technical heart. If you understand D1–D4 you can hold your own with any architect.

---

## D1. Chunking — how a document becomes passages

### Why chunk at all?

Two hard reasons.

1. **A model has a limited window.** You cannot put a 170-page handbook into a prompt.
2. **Precision.** If you retrieved whole documents, the answer to "what is the notice period"
   would be a 200-page file. You want the paragraph.

So the document is split into **passages**. Get this wrong and nothing downstream can recover:
retrieval can only return what indexing created.

### The two strategies

| | **Fixed** | **Clause** *(the default)* |
|---|---|---|
| Splits on | A character count | Sentence and clause boundaries |
| Target | 900 chars, 150 overlap | 700 chars, 100 overlap |
| Problem | Cuts mid-sentence | — |
| Use | Uniform machine-generated text | Everything real |

Clause chunking accumulates whole clauses until adding the next would exceed the cap, then closes
the passage. **It never splits mid-sentence.**

### The four things that close a passage

A passage ends when **any** of these becomes true:

| Trigger | Why |
|---|---|
| **Character cap** (700) | The size budget |
| **Token budget** | If the embedding model can report its own tokenisation, that limit wins — a character cap cannot bound a token budget (prose is ~5.1 chars/token, table rows ~4.3, table-of-contents dot leaders ~1.25, Devanagari ~1.6) |
| **Page boundary** | `page` is a single number. A passage spanning two pages would report one page for both halves and **every citation after it would name the wrong page** |
| **A reliable heading** | So the passage belongs to exactly one section |

### Overlap — and the bug it prevents

Each passage carries ~100 characters of the previous one.

**Why:** a rule and the exception qualifying it must not end up in different passages with
neither making sense alone. The clause path originally carried *no* overlap, and that is exactly
what happened.

### Content-derived passage IDs — an important design decision

An ID is `{doc_id}::{first 12 hex of sha256(passage text)}`.

**Not** `{doc_id}::{position}`. Here's why that matters, and it's a good architect-level story:

> A positional ID re-points on every re-chunk. A stored citation, a human-review record, or a
> knowledge-graph node silently comes to name *different text* than when it was written. And the
> audit chain is immutable — so that untruth can never be corrected.
>
> A content-derived ID moves only when the text it names moves.

Scoped to the document, because the Code of Conduct is reprinted verbatim inside the HR handbook
and those are legitimately different passages. Identical text within one document gets a counter
suffix, so repeated boilerplate doesn't collide and silently overwrite.

### A real chunking bug — the best technical story you have

Ask "what are the six core values?" and get **two of six**. Ask "who's on the WFH committee?" and
get **two of five**.

**Root cause.** The heading detector normally *rejects* single-level numbering (`1.`, `2.`) —
those are usually list items, not headings. But it has an escape hatch: if a document's
single-level numbers ascend, they're accepted as headings. The handbook numbered its policies
1…26, so the hatch opened — **document-wide**. Every numbered list item in 174 pages was promoted
to a heading. And a heading closes a passage. So each list item became its own ~250-character
passage. **No passage held a whole list, so no depth of retrieval could answer "name them all."**

**The fix.** A heading now needs to be *reliable* to close a passage. Multi-level numbering
(`6.2.3 Social Media Content`) is unambiguous and still does. A single-level guess still *labels*
a passage but no longer shreds the document.

**Measured on the real handbook** — chunker output, then confirmed end-to-end through the live
platform after deploying and re-indexing:

| | Before | After |
|---|---|---|
| Passages | 686 | **552** |
| Passages under 350 chars | 239 | **76** |
| Median passage | 478 chars | **609** |
| The committee list | **3 garbled passages** | **1 clean passage, all 5 members** |
| Section labels retained | 184 | 164 |

> **The point to make:** "The model wasn't hallucinating. It was starved. It answered honestly
> about what it had been given — the defect was one step upstream, in how we'd split the
> document. That's why we measure the pipeline and not just the answer."

---

## D2. Embedding — how text becomes searchable by meaning

### What an embedding is — the explanation that lands

> "The model reads a passage and outputs a list of 1,024 numbers. That list is a *coordinate* —
> a position in a space with 1,024 dimensions. Passages that mean similar things land near each
> other in that space, even with no words in common.
>
> So 'What's the notice period?' lands near 'Employees must serve 60 days before separation' —
> despite sharing not one significant word. That is what people mean by semantic search."

### The specifics

| | |
|---|---|
| **Model** | `bge-m3` — multilingual, 100+ languages |
| **Dimensions** | **1,024** numbers per passage |
| **Multilingual property** | A Hindi passage and its English equivalent land near each other — **a question in English can retrieve a Hindi document** |
| **Where it runs** | Self-hosted on the GPU fleet. Nothing leaves the country |
| **Batching** | 32 passages per request |

### Two engineering details worth knowing

**Dimension is a stored contract, not a discovery.** When a knowledge base attaches an embedding
model, the platform *probes* the endpoint and records the true dimension. The vector collection is
created at that width. If the endpoint ever returns a different width, the platform **fails loudly
and refuses** rather than warning — a mismatched vector would poison the collection, not merely
score oddly.

**Changing the embedding model is a staged operation.** You cannot mix vectors from two models in
one space — the coordinates mean different things. So attaching a new embedder is *staged*: an
administrator must explicitly apply it, which recreates the collection at the new dimension and
re-embeds every passage. During that window search degrades to keyword-plus-rerank rather than
breaking.

> **If asked "can we use our own model?"** — yes, any OpenAI-compatible endpoint, attached from
> the console, subject to the same sovereignty check. But switching the *embedding* model means
> re-indexing the corpus, so it's a planned operation, not a toggle.

---

## D3. Retrieval — finding the right passages

### Why two searches, not one

| | Finds | Misses |
|---|---|---|
| **Semantic (vector)** | Meaning and paraphrase | Exact tokens — a policy number, a person's name, a product code |
| **Keyword (BM25)** | Exact terms | Anything phrased differently |

Real questions need both. **"What's our policy on POSH?"** needs the keyword index to match the
acronym; **"how do we handle harassment complaints"** needs the semantic index.

### Fusing them — and why by *rank*

The two searches return two ranked lists. They are combined by **Reciprocal Rank Fusion**:

```
score(passage) = Σ  1 / (60 + rank_in_that_list)
```

**Why rank and not score:** a cosine similarity and a BM25 score are numbers on **different
scales**. Averaging them is meaningless. A passage's *rank* is comparable across both.

The constant 60 dampens the top — being #1 in one list shouldn't dominate a passage that placed
respectably in both.

### Re-ranking

Fusion gets a good candidate set (~50). A **cross-encoder** then reads the question *and* each
candidate **together** and scores real relevance. This is more accurate and far more expensive —
which is exactly why it runs on ~50 candidates, not the whole corpus.

### Then the filter — and the ordering matters

Four checks, described in Part C. **All four run before the top-k trim and before the prompt is
built.**

### Score semantics — a subtle defect this prevents

Every passage records **which scorer** produced its score, and the abstain threshold is chosen
**per scorer**.

**Why this exists.** With one threshold across both scales, the question *"what is the boiling
point of helium on Jupiter"* — against an HR knowledge base — came back **GROUNDED at 0.33**. A
term-overlap fraction and a cross-encoder probability are simply not the same number. A passage
with no named scorer produces an unreachable gate, so a bypassed re-ranker **abstains** rather
than grounding on a meaningless value.

> **This is a great story for an architect.** It shows the team found a class of silent failure
> and fixed it structurally rather than tuning a number until the symptom went away.

---

## D4. Grounding — deciding whether to answer

### The confidence gate

The best passage's score is compared against the threshold **for that scorer**. Below it, the
platform abstains with a recorded reason. It does not generate at all.

### Building the prompt

Each source arrives labelled with document, page and governing section, inside delimiters. Any
text inside a document that looks like those delimiters is neutralised first.

### Citations are resolved by the platform

The model cites `[1]`, `[2]`. **The platform maps those back to passages itself.** It never takes
the model's word for what it cited.

### The type-level invariant — the thing to emphasise

An answer marked *Grounded* with zero citations **cannot be constructed**. The code raises an
error. Likewise *Abstained* with citations attached.

> "This isn't a validation rule someone remembered to write. It's the shape of the data. A future
> developer can't forget it, because there's nothing to forget — an uncited answer simply doesn't
> fit the mould."

### The three outcomes

| Outcome | Meaning | What happens |
|---|---|---|
| **Grounded** | Cleared retrieval confidence and faithfulness | Answer + citations shown |
| **Flagged for review** | Answered, but faithfulness sat between thresholds | Shown **and** queued for a human reviewer |
| **Abstained** | Below the bar, or blocked by a guardrail, or the model itself refused | No answer, explicit reason |

---

## D4b. The limitation that remains — a list longer than one passage

**Measured on the live platform, 2026-09-10.** Worth knowing before you demo, because a customer
may hit it.

Ask *"name each of the six Acme core values"* and the platform returns **five of six**, and says
so. That is honest, but it is not right, and the reason is structural rather than a bug:

The six values occupy about 1,900 characters — nearly three passages at the 700-character cap. So
they land in three passages:

| Passage | Holds |
|---|---|
| 1 | The heading "Core Values", the phrase "six simple ingredients", values **1–2** |
| 2 | values **3–5** |
| 3 | values **5–6** |

Retrieval ranks passage 1 highly, because it contains the words the question uses — *core values*,
*six*. Passage 2 begins *"…assets to ensure there is Luxury in Hard work. 3. Continuous Knowledge
Enhancement…"* — it **holds the answer but shares almost no vocabulary with the question**, so it
ranks below passages that merely mention values in passing.

Verified: the value *"Organization above self & team"* appears in **no retrieved passage** at
top_k=5 **or** top_k=10. The model named everything it was given. Raising top_k does not fix it,
because the problem is relevance ranking, not depth.

### The fix, and why it is not yet built

**Neighbour expansion** — when a passage is retrieved, also pull the passages either side of it in
the same document. A list is contiguous by nature, so retrieving any part of it should bring the
rest. This is a well-understood technique and the architecture supports it (passages carry their
document and position).

It is **not implemented**. It changes what reaches the prompt for every query in the system, so it
needs to be built flag-gated, default-off, and measured on the golden set before being switched on.

### What to say if it comes up in a demo

> "Good catch — and notice it told you it only had five. What's happening is that this list is
> longer than one passage, and the middle of a list doesn't look much like the question you asked.
> We know the fix — pull the neighbouring passages when we retrieve one — it's on the plan. What
> you're seeing is the system being honest about an incomplete retrieval rather than inventing the
> sixth value, which is the behaviour you actually want."

**Do not demo this question.** Use the Work from Home committee question instead — that list fits
in one passage and returns all five members cleanly.

---

## D5. The operational trap — a fix is not live until the data is re-indexed

**This caught us, so it is worth stating plainly.**

Indexing decisions — how a document is split, what an ID is, which model embedded it — are baked
into the **stored data**, not evaluated at query time. So a chunking fix has to travel through
three stages before a user sees any difference:

```
   1 · merged in git        the code is correct
   2 · deployed             the running service has the new code
   3 · RE-INDEXED           the stored passages were rebuilt with it   <-- the one that gets missed
```

Stopping after stage 1 or 2 means the demo still shows the OLD behaviour, and it looks as though
the fix did not work. It did — it just has not touched the data yet.

**Real example.** The list-shredding fix in D1 was merged, but neither deployed nor re-indexed.
The platform kept answering "two of five" for days. Deploying it and re-indexing that one document
took **2 minutes** and changed the answer to all five.

**What this means in practice:**

| Change | Needs re-index? |
|---|---|
| Chunking, section detection, passage IDs | **Yes** — every affected document |
| Embedding model or its dimension | **Yes** — and it is a staged operation (see D2) |
| PII masking rules | **Yes** — masking runs at ingest |
| Retrieval, fusion, re-ranking, thresholds | No — query-time |
| Prompt construction, guardrails, faithfulness | No — query-time |

**Before any demo, on any knowledge base built before a fix landed: re-index it.** One document
takes about two minutes. Ask a question you know the answer to and check you get it.

> **If a customer asks "so we have to rebuild every time you improve something?"** — "Only when
> the improvement changes how documents are *stored*, which is a minority of changes. It runs in
> the background using a write-then-swap: the new passages are written before the old ones are
> removed, so the knowledge base stays queryable throughout — nobody sees an outage."

---
---

# PART E — Architecture

## E1. The views worth drawing

The generated figure set is not part of this repository. These are the eight views that
explain the platform, and where in the code each one is actually defined — draw them from
the source rather than trusting a stale export.

| # | View | Where it lives in the code |
|---|---|---|
| 1 | The whole platform | `CLAUDE.md` "Architecture map" |
| 2 | Ingest flow | `services/control-plane/.../pipeline.py` (`ingest_document`) |
| 3 | Query flow | `services/control-plane/.../pipeline.py` (`answer_query`) |
| 4 | Why two indexes | `libs/rag-core/.../hybrid.py` + `fusion.py` |
| 5 | The access layers | `libs/common/.../rbac.py`, `auth.py`, `rag-core/access.py` |
| 6 | Tenant isolation | `libs/common/.../tenancy.py`; proven by `tests/integration/test_tenant_isolation.py` |
| 7 | Integration planes | `docs/CONSOLE_USER_FLOW.md`, `sdks/`, `apps/mcp/` |
| 8 | Production topology | `infra/helm/`, `infra/compose/docker-compose.prod.yml` |

## E2. Three architectural decisions worth defending

### It's one process, not microservices

All eight layers run in a single process — a **modular monolith**.

> **Why, if asked:** "At this scale the cost is GPU inference, not application CPU. Eight
> services would add network hops inside a request that must finish in three seconds and buy
> nothing. But the package boundaries mirror a microservice split exactly — no layer imports
> another's internals — so splitting one out is a deployment change, not a rewrite."

### There is no RAG framework

No LangChain. No LlamaIndex. Retrieval, fusion, chunking and grounding are self-owned code.

> **Why, if asked:** "A framework owns precisely the seams we most need to control — grounding,
> scoring, tenancy. Owning them is what lets grounded-or-abstain be a type invariant instead of a
> convention. It also keeps a large transitive dependency surface, with its own licence and CVE
> exposure, out of a product we sell into regulated environments."

### Isolation is by namespace, not by container

Each knowledge base gets its own Qdrant collection, OpenSearch index, PostgreSQL schema and
object-store prefix — inside **shared** clusters.

> **If a security reviewer pushes on this:** "A container per knowledge base adds no boundary a
> namespace lacks — both rely on the store's own access control — while multiplying operational
> cost by the number of knowledge bases. The stronger claim, and the one we make, is: namespace
> isolation **plus** an explicit tenant identifier on every single data path **plus** mandatory
> clearance **plus** role-based access — with a test that proves cross-tenant reads fail, and
> that test gates every merge."

**Do not claim container-per-KB.** It isn't true and it's a weaker claim anyway.

---
---

# PART F — The commercial model

## F1. The shape: four layers, not one price

The customer buys a **platform subscription**. On top of that they consume **Acme infrastructure**
— and that consumption is the larger and more durable revenue line.

```
   4 · SERVICES        onboarding · corpus migration · quality tuning        one-time
   3 · PLATFORM        the software subscription, per KB / per user          recurring
   2 · CONSUMPTION     tokens · vector store · object storage · egress       metered
   1 · INFRASTRUCTURE  GPU hours · compute · NVMe                            metered
```

**The point management needs to hear:** a ₹75,000/month subscription that drags ₹40,000–100,000
of infrastructure consumption behind it is worth far more to Acme than the subscription line
alone suggests. RAG is unusually good at this, because a customer must bring their **entire
document corpus** onto Acme storage before they get any value at all.

**This should be measured and reported as attributable revenue** — today it isn't, and the
platform therefore looks less valuable than it is.

## F2. Layer 1 — GPU (the biggest cost, and the one that decides profitability)

### The number the whole business turns on

A ten-user tenant asks about **200 questions a day**. Spread over a working day that is
**0.42 questions per minute**. **The GPU is idle better than 99% of the time.**

So cost is almost entirely **fixed** (a card, a VM, storage) and value is **per tenant**:

| Tenants sharing one GPU | Infra cost per tenant/month | Margin at ₹75,000 |
|---|---|---|
| 1 (dedicated) | ₹249,010 | **−232%** |
| 5 | ₹105,010 | −40% |
| **10** | **₹41,003** | **+45%** |
| 20 | ₹20,502 | +73% |
| 30 | ₹12,901 | +83% |

**Break-even is 6–7 tenants per GPU.** Below that the platform loses money at *any* defensible
price. The concurrency ceiling lands in the same place — a ten-user tenant peaks at ~2 concurrent
questions and one 48 GB card serves ~20 — so **~10 tenants per GPU is both the engineering limit
and the economic sweet spot.**

> **The rule: sell density, not deployments.** A dedicated GPU is an Enterprise-tier option
> priced accordingly — never the default. Fortunately the platform was architected for density:
> isolation is per namespace, so many tenants on shared infrastructure is the designed path.

### GPU sizing

| Phase | Models resident | VRAM | When |
|---|---|---|---|
| **Ingest** | OCR + embeddings | ~22 GB | Once, at onboarding |
| **Serving** | Generation + reranker + faithfulness NLI | ~24 GB | Continuously |

They don't overlap, so **one 48 GB card (L40S / A100) covers both.**

## F3. Layer 2 — Token consumption

Two very different token flows. Price them separately.

### Ingest tokens — one-time, per document

| | |
|---|---|
| One passage | ~503 characters ≈ **126 tokens** |
| Per MB of corpus | 166 passages ≈ **20,900 embedding tokens** |
| **100 GB corpus** | **17.0 M passages ≈ 2.14 billion embedding tokens** |
| **1 TB corpus** | 174 M passages ≈ 21.9 billion tokens |

Embedding tokens are **far cheaper** than generation tokens — a small model, one pass, no output.
This is a one-time onboarding cost. *(Indic scripts run richer — roughly 1.6 chars/token against
~4 for English — so an Indic corpus produces more tokens per MB.)*

### Query tokens — recurring, per question

| Component | Tokens |
|---|---|
| Context — 5 retrieved passages | ~629 |
| Instructions, question, citation scaffolding | ~250 |
| **Input total** | **~879** |
| Output (typical answer) | ~300 |
| **Per question** | **~1,180 tokens** |

| Users | Questions/month | Generation tokens/month |
|---|---|---|
| 10 | 4,400 | **5.2 M** |
| 100 | 44,000 | **51.9 M** |
| 500 | 220,000 | **259.3 M** |

**Two cost controls already built:** the semantic cache (a repeated question costs nothing —
invalidated on any change to the knowledge base) and the adaptive router (simple questions take
the cheaper path).

## F4. Layer 2 — Vector store

At 1,024 dimensions:

| Corpus | Vectors | float32 | **int8** | binary |
|---|---|---|---|---|
| 1 GB | 0.17 M | 0.6 GB | 0.2 GB | 0.02 GB |
| **100 GB** | **17.0 M** | 64.8 GB | **16.2 GB** | 2.0 GB |
| 1 TB | 174 M | 664 GB | **166 GB** | 20.8 GB |

**Quantisation is mandatory beyond ~20 M vectors, not optional.** int8 is a ~4× reduction with
negligible accuracy loss; binary is ~30× with a rescore step. Without it, the 100 GB case needs
122 GB of vector RAM and the economics stop working.

**Price the vector store per million vectors, not per GB of source** — a scanned corpus and a
digital corpus of the same size produce wildly different vector counts (~8× apart).

## F5. Layer 2 — Object and block storage

| Store | 100 GB corpus | Purpose |
|---|---|---|
| Object (raw documents) | 100 GB + versioning | The source of truth |
| Vector store on disk | ~122 GB | Full-precision originals |
| PostgreSQL | ~24 GB | Catalog, passage text, audit chain |
| Keyword index | ~18 GB | BM25 |
| **Total** | **~264 GB** | ≈ **2.6× the raw corpus** |

> **A rule of thumb worth memorising:** *"Expect the platform to occupy roughly two and a half to
> three times the size of the documents you give it."*

## F6. Layer 3 — Platform subscription

| | **Starter** | **Professional** | **Enterprise** | **Sovereign** |
|---|---|---|---|---|
| Infrastructure | Shared pool | Shared pool | Dedicated GPU slice | Customer's own / air-gapped |
| Knowledge bases | 1 | 10 | Unlimited | Unlimited |
| Users | 25 | 250 | Unlimited | Unlimited |
| Corpus | 50 GB | 1 TB | Negotiated | Negotiated |
| Adds | Core RAG, citations, audit | + GraphRAG, NL→SQL, workflows, SSO | + BYOK, federation, compliance pack, HA/DR | + air-gap, signed offline updates |

**Recommended structure: platform fee + included quota + metered overage.** The fee is
predictable (regulated procurement requires that); the overage protects margin against a heavy
tenant. **Do not price on pure consumption** — it's unpredictable for the buyer and, at realistic
question rates, bills far below the fixed cost base.

> **⚠ The metering defaults in the config are internal SHOWBACK figures, not prices.** At 4,400
> questions/month they bill about ₹3,500 against a ~₹250,000 cost base. Commercial rates must be
> derived from the cost model, never copied from the config.

## F7. An illustrative portfolio

| Edition | Price/month | Tenants | Revenue/month | Margin |
|---|---|---|---|---|
| Starter | ₹45,000 | 20 | ₹900,000 | **12%** |
| Professional | ₹150,000 | 10 | ₹1,500,000 | 73% |
| Enterprise | ₹600,000 | 3 | ₹1,800,000 | 70% |
| Sovereign | ₹900,000 | 1 | ₹900,000 | 100% |
| **Total** | | **34** | **₹5,100,000** | **66%** |

**≈ ₹6.1 crore annual recurring revenue at ~66% gross margin, on 34 tenants and roughly 4 GPUs.**

**Note what this says about Starter.** At ₹45,000 against a ₹39,763 shared cost it earns 12% —
it is 20 of the 34 tenants and contributes 3% of the margin. Either price it at ₹65–75,000, or
run it deliberately as a loss-leader that lands accounts which upgrade. **Not by accident.**

*All rates are placeholders pending Finance confirmation. The structure is the deliverable; the
numbers are illustrative.*

## F8. Market

**Primary — where residency is a requirement, not a preference:** BFSI (policy and circular
lookup, credit-file search, audit response) · Pharmaceutical (SOPs, batch records, validation
reports, regulatory dossiers) · Government and PSU (schemes, circulars, internal policy) ·
Healthcare (clinical protocols).

**The most credible near-term market is Acme's own hosting customers** — they have already
accepted Indian hosting and the documents are already inside the estate.

**Not the target:** SMEs and consumers. The compliance depth that justifies the price isn't
something they'll pay for.

### The competition, honestly

| Competitor | Their strength | Why we win |
|---|---|---|
| **Hyperscaler AI** (Azure OpenAI, Bedrock, Vertex) | Vastly larger ecosystems | Data leaves Indian jurisdiction — disqualifying for a material share of buyers |
| **Global RAG vendors** (Glean, Vectara) | Mature products | Same residency problem; most can't deploy air-gapped in India |
| **Indian IT services** (TCS, Infosys, Wipro) | Deep relationships, delivery scale | They build a *bespoke* RAG per client — a project, not a product. **As much a channel as a competitor** |
| **Indian AI startups** (Sarvam, Krutrim) | Strong Indic models | Model layer, not governed platform layer — we consume such models |
| **In-house build** | Free, and they control it | The realistic default. Wiring an LLM to a vector DB takes weeks; the governance takes a year — and most such pilots never reach production |

> **The differentiator in one line:** *"Anyone can retrieve. Almost nobody can answer a
> regulator's questions afterwards."*

---
---

# PART G — Hard questions, honest answers

The rule for all of these: **answer the real question, then say what you're doing about it.** A
buyer in a regulated industry has been pitched by people who claim everything works. Being the
one who names a limitation is a differentiator, not a weakness.

## G1. On accuracy and trust

**"How do you know it doesn't hallucinate?"**
> "We don't rely on knowing. Three mechanisms: an uncited answer can't be constructed — the code
> refuses. Retrieval below the confidence bar abstains instead of generating. And a second,
> independent model checks every answer against its sources. It isn't marking its own homework."

**"What's your accuracy percentage?"**
> "On our internal set we target retrieval accuracy above 0.85 and faithfulness above 0.90, with
> 100% of answers carrying a citation. But I'd rather not sell you a number from someone else's
> documents. Give us a sample of yours and we'll measure it on your corpus and publish the
> result — including if it's disappointing. That measurement is the first thing we'd do in a POC."

**"Will it work on our documents?"**
> "The honest answer is that it depends on your documents, and the way to find out is to measure
> rather than assume. Give us 1–2 GB representative of the real thing. In under a day we can tell
> you the text-extraction ratio, how much is scanned, and what quality looks like."

## G2. On languages

**"Does it work in Hindi / regional languages?"**
> "Retrieval works well multilingually — that's the embedding model, and it's genuinely good. You
> can ask in English and retrieve a Hindi document.
>
> Answer *generation* is more honest to describe carefully. We fixed a real problem where English
> questions were sometimes answered in Hindi — that went from roughly 78% failure to about 6%.
> But we then tested whether the fix generalised, and it doesn't: Tamil and Bengali still fail at
> meaningful rates. It's a limitation of the generation model's instruction-following, and it's
> scheduled work. If regional-language *output* is critical for you, let's test your specific
> languages before you commit."

**Why say this.** They will find out. Saying it first makes everything else you claim credible.

## G3. On security and access

**"Can an administrator see everything?"**
> "No — and that's deliberate. Clearance is a *mandatory* control, separate from role. An
> organisation administrator with clearance 1 cannot read Restricted content. A discretionary
> role can't override a mandatory one; that's the point of having one."

**"What if someone asks a clever question to extract restricted content?"**
> "They can't, because the filtering isn't applied to the *answer* — it's applied to the
> *passages*, before the AI ever sees them. An unauthorised passage is never a candidate, so it
> can't influence the answer even indirectly."

**"What about prompt injection — a malicious instruction inside a document?"**
> "Two defences. Questions are screened before retrieval. And retrieved documents are treated as
> untrusted input: source blocks are fenced, and fence-like text inside a document is neutralised,
> so a document can't forge the prompt's own structure."

**"Is our data used to train models?"**
> "No. We don't fine-tune, we don't train, we don't send your content anywhere. Models are
> selected from configuration and self-hosted."

## G4. On cost

**"Why is it expensive for only ten users?"**
> "Because the cost is the GPU, and the GPU is largely fixed — ten users leave it idle 99% of the
> time. That's exactly why we run a shared pool: you pay a fraction of a card, not a whole one.
> A dedicated card is available, and priced accordingly, but for your volume it would be paying
> for capacity you'd never use."

**"Can we start smaller?"**
> "Yes — a POC on one department's documents. That's also the right way to get a real quality
> number before a bigger commitment."

## G5. On the architecture

**"Why not just use ChatGPT / Copilot?"**
> "For general work, use them. For your policy documents, three things stop you: your data leaves
> Indian jurisdiction, the answers aren't traceable to a source, and there's no access control or
> audit trail. Those aren't features you add later — they're architecture."

**"You built your own instead of using LangChain — isn't that reinventing the wheel?"**
> "It's owning the three seams we most need to control: grounding, scoring and tenancy. That's
> what lets 'never answer without a citation' be a structural property rather than a convention
> someone could bypass. It also keeps a large dependency surface, with its own licence and
> vulnerability exposure, out of a product we sell into regulated environments."

**"What happens if a model vendor disappears or changes their licence?"**
> "Every model role sits behind an interface and is selected from configuration. Swapping is a
> config change plus a re-measurement of the confidence threshold. We've actually done this — our
> current re-ranker has a non-commercial licence, so the permissively-licensed replacement is
> already implemented and wired as an automatic fallback."

## G6. Where it genuinely isn't ready

Know these. If a customer's use case lands on one, say so.

| Not built | Why / when |
|---|---|
| **Visual document retrieval (ColPali)** | Needs GPU we haven't allocated. Feasibility confirmed. Post-GA |
| **Audio / video transcription** | Same GPU constraint |
| **Lakehouse / RDBMS / REST connectors** | Scoped, not built. Document ingestion works today |
| **Native mobile apps** | Console is responsive; API is open |
| **Regional-language output beyond Hindi** | Verified as still failing at meaningful rates — see G2 |
| **Independent VAPT & ISO 27001 certification** | Scheduled, not complete. **Do not claim these are done** |
| **Neighbour expansion** | A list spanning several passages retrieves incompletely — see D4b. Known, understood, not built |

---

## G7. The five things to remember

If you retain nothing else:

1. **"Grounded, or it stays silent."** The abstention is the feature.
2. **"Your documents never leave India"** — enforced in code, failing closed, not in a policy PDF.
3. **"Your existing classification carries into the AI"** — and an admin can't override it.
4. **"Anyone can retrieve. Almost nobody can answer a regulator's questions afterwards."**
5. **Density is the business.** ~10 tenants per GPU. Never sell a dedicated card at a shared price.

---

## Appendix — the numbers, one place

| | |
|---|---|
| Embedding dimensions | 1,024 (`bge-m3`) |
| Passage target / overlap | 700 chars / 100 chars (clause) |
| Measured passage size | ~503 chars median |
| Passages per MB of corpus | **166** |
| Text-extraction ratio (measured) | **8%** on a digital PDF; 10–25% typical; ~1% scanned |
| Rerank candidate pool | 50 |
| RRF constant | 60 |
| Tokens per question | ~1,180 (879 in, 300 out) |
| Embedding tokens per MB | ~20,900 |
| Vector RAM, 100 GB corpus | 16.2 GB int8 · 64.8 GB float32 |
| Platform storage overhead | **~2.6× the raw corpus** |
| GPU break-even | **6–7 tenants per card** |
| P95 latency target | ≤ 3 seconds (retrieval) |
| Automated tests | 3,341 (2,947 unit · 394 e2e) |
| API endpoints | ~200 |
| Architecture decisions recorded | 26 ADRs |

**Source of truth:** the code. Every figure above should be drawn from the files named in E1,
and the sizing numbers come from a running system rather than an estimate — see
`docs/production-sizing.md`.
