# `rag_core`

The retrieval and grounding logic, hand-rolled: **Protocols plus pure functions, no RAG
framework** (no LlamaIndex, no LangChain — see [ADR-0002](../../docs/adr/0002-thin-core-no-rag-framework.md)).

57 modules, 63 test files. Nothing here performs I/O: every external dependency is a Protocol that
`rag_clients` implements, which is why this package is cheap to test and why the same test bodies
run against both real adapters and their in-memory twins.

## The seams (`interfaces.py`)

`VectorStore`, `SparseIndex`, `Embedder`, `Reranker`, `LLMGateway`, `Chunker`, `ObjectStore`,
`CatalogAdapter`, `AuditSink`, `Retriever`, `FaithfulnessChecker`, `Contextualizer`, `QueryRouter`.

## The logic

- **Retrieval** — `hybrid.py` orchestrates dense + sparse; `fusion.py` is reciprocal rank fusion;
  `recency.py`, `access.py` and the metadata filters narrow it; `router.py`/`routing.py` pick the
  shallow or deep path.
- **Grounding** — `grounding.py` assembles citations and makes the grounded-or-abstain decision;
  `faithfulness.py` routes an answer to GROUNDED / FLAGGED_FOR_REVIEW / ABSTAINED;
  `verification.py` and `numeracy.py` catch claims the spans do not support.
- **Chunking** — `chunking.py`, `layout_chunking.py`, `sections.py`, `parent_child.py`,
  `atomic_units.py`, `contextual.py`.
- **Graph** — `graph.py` extracts entities/relations and detects communities for GraphRAG.
- **RAG v2** — `mapreduce.py`, `doc_profile.py`, `section_enrich.py`, `query_plan.py`, and the
  prompt files under `prompts/` (reviewable Markdown, loaded as package data so they ship in the wheel).

## If you change something here

Pure logic means a test can assert on it directly — add one. A behaviour change that only shows up
through the API is a change in the wrong layer.
