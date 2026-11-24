# Models — RAG as a Service (isolated multilingual deployment)

Every model this deployment uses, why it was chosen, how it's served, and its sovereignty /
licensing posture. The platform is **model-agnostic** — each row is selected purely by config
(`docker-compose.override.yml` env), nothing is hard-coded — so any row can be swapped for a
self-hosted equivalent without code changes.

Deployment: isolated dev VM `192.0.2.12`, CPU-only (no local GPU), stores local
(Qdrant/OpenSearch/MinIO/Postgres), models via hosted APIs.

## Model roster

| Role | Model | Served via | Wire | Sovereignty / License |
|---|---|---|---|---|
| **Generation (LLM)** | **Sarvam-105b** | `api.sarvam.ai/v1` (hosted) | OpenAI-compat chat | Hosted API. Sarvam AI (Indian). Reasoning model. |
| **Embeddings** | **bge-m3** (1024-dim, multilingual) | Acme **AIOps** `models.example.internal` (GPU) | OpenAI `/embeddings` | Sovereign (Acme platform). Apache-2.0 weights. |
| **Reranker (primary)** | **jina-reranker-v3.5** (0.6B listwise) | `api.jina.ai/v1` (hosted) | Cohere `/rerank` + sigmoid | Hosted API. Weights on HF, **CC-BY-NC** (non-commercial). |
| **Reranker (fallback)** | **bge-reranker-v2-m3** | AIOps `models.example.internal` (GPU) | Cohere `/rerank` | Sovereign (Acme platform). Apache-2.0 weights. |
| **OCR + Vision** | **Chandra** | **Datalab** `www.datalab.to` (hosted async) | `/convert` submit+poll | Hosted API. Chandra is Apache-2.0 open weights. |
| **Faithfulness (NLI)** | **mDeBERTa-v3-base-xnli** (multilingual) | self-hosted TEI container `nli-checker` | TEI `/predict` | **Sovereign** (runs in-cluster). MIT. |
| **Translation (graph)** | **Sarvam-Translate** (Mayura) | `api.sarvam.ai/translate` (hosted) | Sarvam `/translate` REST | Hosted API. Open weights `sarvamai/sarvam-translate`, **GPL-3.0** (self-hostable). |

## Why each model

- **Sarvam-105b (generation)** — Indian-language-native LLM; answers in the question's language
  across Indic scripts. It is a *reasoning* model, which drives two workarounds in code:
  `RAG_LLM_MAX_TOKENS=4000` (it spends budget in a `<think>` block) and a null-content guard +
  retry in the gateway (it intermittently returns `content: null`, which used to surface as a
  literal `"None"` answer).
- **bge-m3 (embeddings)** — multilingual dense embeddings (100+ languages) so a query in one
  language retrieves passages in another. Replaced the English-only `bge-small-en` that broke
  cross-lingual retrieval. 1024-dim.
- **jina-reranker-v3.5 (reranker)** — best cross-lingual relevance on *abstract* terms
  (e.g. English "environment" ↔ Hindi "पर्यावरण"); verified across 10 Indian languages, scoring
  relevant passages positive and off-topic **negative** (clean abstain margin). Served with
  `+sigmoid` (v3.5 emits logit-style scores) and the abstain gate calibrated to **0.50**.
  **bge-reranker** is wired as an automatic fallback (`FallbackReranker`) if jina is unreachable.
- **Chandra (OCR/vision)** — layout-aware document OCR for all Indic scripts, tables, and LaTeX.
  A PDF's digital pages are read directly (pdfplumber); scanned/image pages — and, with
  `RAG_PDF_IMAGE_PAGES=ocr`, any page carrying an embedded image — go through Chandra and merge
  back by page.
- **mDeBERTa NLI (faithfulness)** — cross-lingual entailment. Faithfulness is scored as
  **`1 − P(contradiction)`** (not entailment): an abstractive answer is "faithful" if the sources
  don't contradict it — works even when the answer and sources are in different languages.
- **Sarvam-Translate (graph translation)** — the *dedicated* Sarvam MT model, because the
  **reasoning chat model is unreliable at translation** (echoes the source or returns null),
  which left the Indic knowledge-graph with no entities. Used by translate-then-extract:
  Indic chunk → English → extract entities → map back to the source language.

## Sovereignty posture

- **Fully sovereign (in-cluster / the GPU fleet):** bge-m3, bge-reranker, mDeBERTa NLI.
- **Hosted APIs (foreign or managed):** Sarvam-105b, Sarvam-Translate (Sarvam), jina-v3.5 (Jina),
  Chandra (Datalab). All egress hosts are explicitly allow-listed
  (`RAG_ALLOWED_EGRESS_HOSTS`) and enforced at runtime by `assert_sovereign`.
- **Path to full sovereignty (prod):** every hosted model has open weights that can be
  self-hosted on GPU behind the same OpenAI-/REST-compatible wire — no code change, just point
  the URL at the internal endpoint:
  - Sarvam-105b / Sarvam-Translate → self-host on GPU (Sarvam-Translate weights are GPL-3.0).
  - jina-reranker-v3.5 → self-host (weights CC-BY-NC — needs a commercial license for prod use),
    or stay on the sovereign **bge-reranker** (already the fallback).
  - Chandra → self-host the Apache-2.0 weights on a vLLM/GPU OCR service.

## Secrets

API keys live only in `docker-compose.override.yml` on the VM (gitignored) and in
an `.env` file outside the repository. No key is ever committed to git.

## How to switch a model

Change the relevant `RAG_*` env in `docker-compose.override.yml` and recreate the `api`
service. Per-KB overrides (a KB bringing its own LLM/embedder/reranker endpoint) are configured
live in the console **Models** section — probed and attached without redeploying.
