# RAG as a Service — Python SDK

A thin, typed, **sovereign** client for the RAG as a Service `/v1` API. Depends only on
`httpx` + `pydantic` — no platform internals — so you can `pip install rag-sdk` anywhere.

## Install

```bash
pip install rag-sdk   # from the internal sovereign package registry
```

## Quickstart

```python
from rag_sdk import RagClient

# Authenticate with a service-account API key (recommended) or a bearer JWT.
client = RagClient("https://rag.internal", api_key="ek_...")

# Create a knowledge base and add a document.
client.create_knowledge_base("policies")
client.ingest("policies", doc_id="leave", text="Annual leave is 24 days per year.")

# Ask — grounded, with citations, or an honest abstain.
answer = client.query("policies", "How many annual leave days?")
if answer.is_grounded:
    print(answer.answer)
    for c in answer.citations:
        print(f"  [{c.doc_id}] {c.quote!r}")
    # FR-I8 follow-ups (when enabled) are grounded in the same sources:
    for q in answer.suggested_followups:
        print("  next:", q)
else:
    print("No grounded answer:", answer.abstain_reason)

client.close()  # or use `with RagClient(...) as client:`
```

## Scopes

An API key's scope maps to what the SDK can do: `read` → query, `ingest` → query + ingest,
`admin` → everything. The server enforces this — the SDK simply carries the key.

## Errors

Any non-2xx response raises `RagError` with `.status_code`, `.code`, and `.message`.
