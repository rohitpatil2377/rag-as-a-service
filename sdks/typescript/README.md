# RAG as a Service — TypeScript SDK

A thin, typed, **sovereign** client for the RAG as a Service `/v1` API. Zero runtime
dependencies (uses the platform `fetch`). Runs in the browser, Node 18+, Deno, and edge.

## Install

```bash
npm install @ragaas/rag-sdk   # from the internal sovereign registry
```

## Quickstart

```ts
import { RagClient } from '@ragaas/rag-sdk'

// Authenticate with a service-account API key (recommended) or a bearer JWT.
const client = new RagClient('https://rag.internal', { apiKey: 'ek_...' })

await client.createKnowledgeBase('policies')
await client.ingest('policies', { docId: 'leave', text: 'Annual leave is 24 days per year.' })

const answer = await client.query('policies', 'How many annual leave days?')
if (answer.status === 'grounded') {
  console.log(answer.answer)
  answer.citations.forEach((c) => console.log(`  [${c.doc_id}] ${c.quote}`))
  // FR-I8 follow-ups (when enabled) are grounded in the same sources:
  answer.suggested_followups.forEach((q) => console.log('  next:', q))
} else {
  console.log('No grounded answer:', answer.abstain_reason)
}
```

## Errors

Any non-2xx response throws `RagError` with `.statusCode`, `.code`, and `.message`.

> **Browser note:** never ship an API key to the browser. Have your server hold the key
> and mint short-lived scoped tokens for the client — the same model the embeddable widget
> (FR-I6) uses.
