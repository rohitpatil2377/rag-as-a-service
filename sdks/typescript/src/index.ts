/**
 * RAG as a Service — official TypeScript SDK.
 *
 * A thin, typed, sovereign client for the `/v1` API. Zero runtime dependencies (uses the
 * platform `fetch`). Works in the browser, Node 18+, Deno, and edge runtimes.
 *
 *   import { RagClient } from '@ragaas/rag-sdk'
 *   const client = new RagClient('https://rag.internal', { apiKey: 'ek_...' })
 *   await client.ingest('policies', { docId: 'leave', text: 'Annual leave is 24 days.' })
 *   const answer = await client.query('policies', 'How many annual leave days?')
 *   if (answer.status === 'grounded') console.log(answer.answer, answer.citations)
 */

export interface Citation {
  chunk_id: string
  doc_id: string
  quote: string
  page: number | null
  score: number | null
}

export interface GroundedAnswer {
  tenant_id: string
  query: string
  status: 'grounded' | 'abstained' | 'flagged_for_review'
  answer: string
  citations: Citation[]
  confidence: number
  abstain_reason: string | null
  faithfulness_score: number | null
  trace_id: string | null
  model: string | null
  /** Non-authoritative follow-ups grounded in the retrieved context (FR-I8), when enabled. */
  suggested_followups: string[]
}

export interface KnowledgeBase {
  kb_id: string
  name: string
  is_default: boolean
}

export interface Document {
  doc_id: string
  title?: string | null
  mime?: string | null
  chunk_count?: number | null
}

export class RagError extends Error {
  constructor(public statusCode: number, public code: string, message: string) {
    super(`[${statusCode} ${code}] ${message}`)
    this.name = 'RagError'
  }
}

export interface ClientOptions {
  /** Service-account API key (recommended). */
  apiKey?: string
  /** Or a bearer JWT. */
  token?: string
  /** Injectable fetch (tests / non-global-fetch runtimes). */
  fetchImpl?: typeof fetch
}

export interface IngestOptions {
  docId: string
  text?: string
  contentB64?: string
  mime?: string
  strategy?: string
}

function toBase64(text: string): string {
  if (typeof btoa === 'function') return btoa(unescape(encodeURIComponent(text)))
  // Node fallback (no @types/node dependency — reach Buffer via globalThis).
  const buf = (globalThis as { Buffer?: { from(s: string, enc: string): { toString(enc: string): string } } }).Buffer
  if (buf) return buf.from(text, 'utf-8').toString('base64')
  throw new Error('no base64 encoder available; pass contentB64 instead of text')
}

export class RagClient {
  private readonly auth: string
  private readonly base: string
  private readonly doFetch: typeof fetch

  constructor(baseUrl: string, opts: ClientOptions) {
    const credential = opts.apiKey ?? opts.token
    if (!credential) throw new Error('provide either apiKey or token')
    this.auth = `Bearer ${credential}`
    this.base = baseUrl.replace(/\/$/, '')
    this.doFetch = opts.fetchImpl ?? fetch
  }

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const resp = await this.doFetch(`${this.base}${path}`, {
      method,
      headers: { Authorization: this.auth, ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}) },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    })
    if (!resp.ok) {
      let code = 'http_error'
      let message = await resp.text()
      try {
        const parsed = JSON.parse(message)
        code = parsed.code ?? code
        message = parsed.message ?? parsed.detail ?? message
      } catch {
        // non-JSON error body — keep the text
      }
      throw new RagError(resp.status, code, message)
    }
    if (resp.status === 204) return undefined as T
    return (await resp.json()) as T
  }

  createKnowledgeBase(displayName: string): Promise<KnowledgeBase> {
    return this.request('POST', '/v1/knowledge-bases', { display_name: displayName })
  }

  listKnowledgeBases(): Promise<KnowledgeBase[]> {
    return this.request('GET', '/v1/knowledge-bases')
  }

  ingest(kbId: string, opts: IngestOptions): Promise<Record<string, unknown>> {
    if ((opts.text === undefined) === (opts.contentB64 === undefined)) {
      throw new Error('provide exactly one of text or contentB64')
    }
    const content_b64 = opts.contentB64 ?? toBase64(opts.text as string)
    return this.request('POST', `/v1/kb/${encodeURIComponent(kbId)}/ingest`, {
      doc_id: opts.docId, mime: opts.mime ?? 'text/plain', content_b64, strategy: opts.strategy ?? 'clause',
    })
  }

  listDocuments(kbId: string): Promise<Document[]> {
    return this.request('GET', `/v1/kb/${encodeURIComponent(kbId)}/documents`)
  }

  deleteDocument(kbId: string, docId: string): Promise<void> {
    return this.request('DELETE', `/v1/kb/${encodeURIComponent(kbId)}/documents/${encodeURIComponent(docId)}`)
  }

  query(kbId: string, query: string, opts: { topK?: number; model?: string } = {}): Promise<GroundedAnswer> {
    const body: Record<string, unknown> = { query, top_k: opts.topK ?? 5 }
    if (opts.model) body.model = opts.model
    return this.request('POST', `/v1/kb/${encodeURIComponent(kbId)}/query`, body)
  }
}
