# RAG as a Service — Production Sizing Note (TB-scale ingestion & retrieval)

**Audience:** infra / platform engineering.
**Question this answers:** *"Can it handle TBs of data and stay fast?"* — Yes, with the
stores clustered and quantization + GPU embeddings on. This note gives the capacity math
and the concrete knobs. It is infrastructure sizing, not a code change.

---

## 1. The three meanings of "TBs" (only one is hard)

| Concern | Difficulty | Why |
|---|---|---|
| **Storing raw bytes** | Trivial | MinIO/object store scales to PB. Docs just sit there. |
| **Indexing** | The real work | RAM for vectors is the binding constraint. Solved by quantization + sharding. |
| **Retrieving** | Stays easy | ANN (HNSW) is ~O(log N). Latency barely moves with corpus size. |

**Takeaway:** retrieval latency is *not* the problem at scale. Vector RAM and
ingestion throughput are — and both are standard horizontal-scaling problems.

---

## 2. Capacity math (ballpark — validate on a real sample of your corpus)

**Raw docs → text → chunks → vectors.** Mixed enterprise docs (PDF/DOCX/scans) are
typically only **~10–25% extractable text** by bytes.

```
1 TB mixed docs
  → ~100–250 GB extractable text
  → ~50–100 M chunks         (≈500 tokens ≈ ~2 KB/chunk)
  → ~50–100 M vectors

10 TB  → ~0.5–1 B vectors
```

> Pure-text corpora (logs, DB exports) are the worst case: closer to
> ~400–500 M chunks per TB. Always measure the text-extraction ratio on a
> representative sample before committing hardware.

**Vector RAM (the binding constraint), at 1024-dim (BGE-M3):**

| Vectors | float32 (raw) | int8 (scalar quant) | binary quant (+ rescore) |
|---|---|---|---|
| 100 M | ~400 GB | ~100 GB | ~13 GB |
| 500 M | ~2 TB | ~500 GB | ~64 GB |
| 1 B | ~4 TB | ~1 TB | ~128 GB |

**Conclusion: beyond a few tens of millions of vectors, quantization is mandatory,
not optional.** Binary quantization + on-disk original vectors + HNSW rescoring keeps
RAM sane while preserving recall (rescore the top candidates against full vectors).

---

## 3. Qdrant at scale

Apply at **collection creation** (per-tenant collection):
- **Sharding:** `shard_number` ≥ number of nodes; spread across the cluster.
- **Replication:** `replication_factor` ≥ 2 for HA.
- **Quantization:** scalar (int8) as the safe default; binary for the largest tenants
  (with `always_ram` for the quantized vectors + `oversampling` on rescore).
- **On-disk:** `on_disk: true` for the original vectors and the payload index; keep only
  quantized vectors in RAM.
- **HNSW:** default `m=16` is fine; raise `ef_search` to trade latency for recall.

> **Wired into the platform:** these are driven by config on the `QdrantVectorStore`
> adapter — set `RAG_QDRANT_QUANTIZATION` (`none`|`scalar`|`binary`), `RAG_QDRANT_ON_DISK`,
> `RAG_QDRANT_SHARDS`, `RAG_QDRANT_REPLICATION` (see `infra/compose/.env.prod.example`).
> When quantization is on, search rescores the top candidates against full-precision vectors
> so recall holds. **Applied at collection creation** — existing tenant collections keep
> their config until re-indexed, so set these **before ingesting at scale**. Defaults
> (`none`/off/1/1) reproduce the original single-node full-precision collection exactly.

---

## 4. OpenSearch (BM25 / hybrid sparse side)

- Run a **multi-node cluster**, one index per tenant (already the tenancy model).
- Size primary shards to keep each ≤ ~30–50 GB; add replicas for HA.
- For exact-term metadata filters (B7 tags), map those fields as `keyword`
  (noted in CLAUDE.md) so D8 filters stay fast.
- Hot/warm tiering for older/cold tenants if the index footprint grows.

---

## 5. Ingestion throughput — the actual bottleneck

Retrieval scales for free; **ingestion is gated by embedding speed.**

| Embedder tier | ~Throughput | 100 M chunks |
|---|---|---|
| CPU TEI (bge-small) | ~hundreds/sec | **days** |
| **GPU TEI (BGE-M3)** | ~thousands/sec | **hours** |

**Levers:**
- **GPU embeddings** (`tei-gpu` service, `gpu` compose profile) — the single biggest win.
- **`RAG_BULK_INGEST_CONCURRENCY`** — raise from the dev default of 2 toward 16–32 on a
  GPU tier (bounds how many docs embed at once).
- **Bulk-ingest worker** (`RAG_BULK_INGEST_WORKER_ENABLED`, on by default) — files are
  staged and ingested by a background worker that resumes after restarts. Use it, not the
  synchronous path, for volume.
- **Delta sync + SHA-256 dedup** — re-syncs skip unchanged bytes, so the big embed cost is
  paid once. Keep scheduled sync on.
- Scale ingestion horizontally: multiple API/worker replicas + `RAG_REDIS_URL` set, so the
  coordinator handles shared rate-limiting and single-leader scheduling.

---

## 6. Model choices (sovereign, self-hosted)

- **Embedder = retrieval ceiling.** `BGE-M3` (MIT, 1024-dim, multilingual incl. Hindi/Indic).
  Matters more than the LLM for retrieval quality. **Switching embedder ⇒ full re-index**
  (dim 384→1024) — decide before loading at scale.
- **Generation LLM.** `Qwen2.5-7B-Instruct` (Apache-2.0) on vLLM/GPU as the org default;
  14B/32B with GPU headroom. Attach an Indic model (e.g. Sarvam) per-KB for
  Indic-language *answers*.
- No foreign/managed AI at runtime — enforced by `assert_sovereign`. All of the above are
  India-self-hostable.

---

## 7. Rollout checklist

1. Pick the embedder (**BGE-M3**) and set `RAG_EMBED_MODEL` / `RAG_EMBED_DIM=1024`
   **before first ingest** (see `infra/compose/.env.prod.example`).
2. Bring up the `gpu` compose profile (`vllm` + `tei-gpu`): `docker compose --profile gpu up -d`.
3. Cluster Qdrant (shards + replicas) and enable **quantization + on-disk** on collections.
4. Cluster OpenSearch; size shards; `keyword`-map metadata fields.
5. Set `RAG_REDIS_URL` and run multiple API/worker replicas.
6. Ingest a **representative sample first**, measure the text-extraction ratio and real
   chunk count, then extrapolate hardware from §2 before the full load.
7. Turn on `RAG_FAITHFULNESS_BACKEND=llm` for the stronger judge in prod.
