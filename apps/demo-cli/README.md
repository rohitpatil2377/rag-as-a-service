# `demo_cli` — the walking skeleton

One command that drives the entire vertical slice and prints what happened at each step: create a
knowledge base → ingest a document → ask a question that the corpus answers → ask one it does not.

```bash
make demo                   # in-memory: ~2 seconds, no Docker, no model
make demo ARGS=--stores     # the same path against the live stack (needs `make up`)
```

It talks to the real FastAPI app over `httpx.ASGITransport`, so there is **no network and no
server to start** — but it exercises the genuine routes, auth, pipeline and audit chain, not a
mock. That makes it the fastest honest answer to "is the platform working end to end?".

Expected output ends with:

```
status=GROUNDED  confidence=…        answer: … [1]
  [1] platform_brief p.1  "The platform's uptime target is …"
status=ABSTAINED  abstain_reason: retrieval_confidence_below_threshold
✓ walking skeleton complete — grounded, cited, and abstaining as designed.
```

Both halves matter. A run where the second question produces an answer is a **failure**, even
though it looks like success — the abstain gate is the feature.

Options: `--ask` (question), `--tenant` (knowledge-base id), `--stores` (real adapters).
