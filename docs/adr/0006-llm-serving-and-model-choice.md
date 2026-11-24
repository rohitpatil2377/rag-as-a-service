# ADR-0006 — LLM serving tiers and (deferred) production model choice

**Status:** Accepted (dev tiers) / Deferred (prod model) · 2026-07-02

## Context
Phase-0 ran the zero-model `extractive_dev` fallback. Phase-1 quality/guardrail features
(faithfulness, later HyDE/judge) need real generation. The dev VM has **no GPU** (no
`nvidia-smi`, docker runtime is `runc` only), so the vLLM `gpu` compose profile cannot run
here.

## Decision — serving tiers (all sovereign, OpenAI-compatible)
One adapter, `OpenAICompatGateway` (chat completions), serves all tiers; only the URL and
served model differ:

| Tier | Server | Model | When |
|---|---|---|---|
| CPU dev (this VM) | **llama.cpp server** | Qwen2.5-3B-Instruct (GGUF Q4_K_M), fallback 1.5B | now — real generation, no GPU |
| GPU dev/staging | **vLLM** (`gpu` profile) | Qwen2.5-7B-Instruct | when a GPU VM exists |
| Prod | vLLM/Triton on GPUaaS | see "Deferred" below | production |

- **Weights bootstrap:** the GGUF (and any HF pull) is a **one-time, documented dev
  bootstrap** from `huggingface.co`, cached in a volume — the same rule as the TEI
  `bge-small` pull. The runtime egress guard (`assert_sovereign`) is **never loosened**;
  if a pull host is blocked, stop and request mirroring. Prod points `HF_ENDPOINT` at an
  internal model mirror (TDD NFR "Sovereignty").
- CPU 3B chosen for the dev tier: real generation with acceptable latency at 23 GB RAM;
  **do not go to 7B+ on CPU**. Fall back to 1.5B if 3B breaches the memory headroom
  target or is too slow (recorded at run time in the README / commit notes).

## Deferred decision — production model + a distinct faithfulness judge
The clean-licence production model choice is **deferred to the vLLM/GPU step** and to the
AI team (PRD open-Q2). Candidates to evaluate then, all permissively licensed:
- **Phi-3.5-mini** (MIT)
- **Llama-3.2-3B** (Llama Community licence — check govt/sovereign acceptability)
- **verified Qwen2.5** (Apache-2.0) once the AI team signs off the specific weights.

The **faithfulness judge (FR-E4/ADR-0005) must be a different / stronger model than the
generator** — a self-judging small model is a weak signal. This selection is part of the
same deferred decision.
