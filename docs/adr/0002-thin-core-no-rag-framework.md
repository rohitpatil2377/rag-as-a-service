# ADR-0002 — Thin, Hand-Rolled RAG Core; No Heavy RAG Framework

**Status:** Accepted · **Date:** 2026-07-02

## Context
The TDD (§12) lists **LlamaIndex** as the RAG framework. The user's explicit target
stack for this build omits it, naming only LangGraph for orchestration. RAG as a Service is
an self-owned platform with a 5-year horizon; RAG frameworks (LlamaIndex, LangChain)
move fast, carry large transitive dependency trees, and tend to impose their own
abstractions for exactly the things Acme needs full control over: chunking strategy,
retrieval fusion, grounding/citation logic, and multi-tenant isolation.

## Decision
`libs/rag-core` contains **only**:
1. `Protocol` interfaces for every swappable seam (`Embedder`, `VectorStore`,
   `SparseIndex`, `Reranker`, `LLMGateway`, `Chunker`, `ObjectStore`,
   `CatalogAdapter`, `AuditSink`, `Retriever`).
2. Pure-logic implementations Acme owns outright: RRF fusion (`fusion.py`),
   chunking strategies (`chunking.py`), the grounded-or-abstain gate and citation
   assembly (`grounding.py`), and the hybrid-retrieve orchestrator (`hybrid.py`).

Narrowly-scoped, easily-replaceable libraries are still used for **mechanical** tasks
only — e.g. `pypdf` for PDF text extraction. Orchestration via **LangGraph** is
accepted as a Phase-1 stub because it is a graph-of-steps *primitive*, not a RAG
framework that owns retrieval/chunking/grounding decisions.

## Consequences
- **Positive:** every RAG-specific decision (how we fuse dense+sparse, when we
  abstain, how citations are built) lives in code Acme controls and can evolve
  without waiting on an upstream framework release or working around its opinions.
- **Positive:** the dependency surface of `rag-core` is `pydantic` + `rag_common` +
  `rag_schemas` only — no framework version-compatibility risk.
- **Negative:** Acme forgoes ready-made framework features (e.g. LlamaIndex's large
  connector/loader ecosystem, built-in query engines) and must build or adapt them.
  Mitigated by the Protocol seams — a framework can still be adopted *behind* a
  Protocol later if a specific feature justifies it, without leaking framework types
  into service code.
- **Deviation from TDD §12 noted and intentional** — this is a conscious divergence,
  not an oversight.

## Alternatives considered
- **LlamaIndex as the retrieval/indexing backbone** (TDD default) — rejected: framework
  lock-in risk on a 5-year product; harder to enforce Acme-specific invariants (e.g.
  the grounded-or-abstain hard gate) inside a framework's own answer-synthesis loop.
- **LangChain** — rejected for the same lock-in reasons; also not in the TDD or the
  user's target stack.
