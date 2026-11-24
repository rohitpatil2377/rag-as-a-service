// Audit Trail (FR-G4) — the tamper-evident, hash-chained record of everything the platform
// did for this knowledge base.
//
// Rebuilt 2026-10. The previous page had ONE input, bound to `trace_id`, labelled "Filter by
// trace id" — so typing anything a human would actually search for ("index", an actor, a
// document) sent it as a trace id, matched nothing, and fell through to the "No audit events
// yet" empty state, which reads as "the platform recorded nothing" rather than "your filter
// matched nothing". That is the bug this file fixes, and the reason the filters below are
// deliberately explicit about WHAT they match.
//
// Filtering is CLIENT-SIDE over a loaded window (the API exposes `trace_id` + `limit` only),
// so every count on this page is phrased against that window — never as a total. `trace_id`
// is the one filter pushed to the server, because it is the one the server understands.
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useI18n } from '../lib/i18n'
import { Icon } from '../components/icons'
import { Alert, Button, Card, EmptyState, PageHeader, SegmentedControl, Skeleton } from '../components/ui'
import { ApiError, api, type AuditEvent } from '../lib/api'
import { useAuth } from '../lib/auth'
import { useKb } from '../lib/kb'

// --- Action vocabulary ----------------------------------------------------
// Every AuditAction the platform can write (libs/schemas/.../common.py). A raw action name is
// kept visible in the detail view, but the trail is scanned by humans, so the table shows a
// plain-English label grouped into the five things an auditor actually asks about.
type Category = 'knowledge' | 'answering' | 'governance' | 'federation' | 'platform'

const CATEGORY_LABEL: Record<Category, string> = {
  knowledge: 'Knowledge', answering: 'Answering', governance: 'Governance',
  federation: 'Federation', platform: 'Platform',
}

// Display order: roughly the life of a question — what entered the corpus, what was asked of it,
// what the platform checked, who it was shared with, and what was done to the platform itself.
// Not alphabetical, because the first two are what an auditor opens this page for.
const CATEGORY_ORDER: Category[] = ['knowledge', 'answering', 'governance', 'federation', 'platform']

// One line per category, so the chips explain what they select rather than relying on a noun.
const CATEGORY_HINT: Record<Category, string> = {
  knowledge: 'What entered or left the corpus — ingestion, indexing, graph extraction, deletion, erasure, classification.',
  answering: 'What was asked and answered — questions, retrieval, generation, abstentions, SQL, workflows, agents.',
  governance: 'What the platform checked and who decided — faithfulness, guardrails, human review, evaluations, approvals, roles.',
  federation: 'Cross-department access — grants issued and revoked, federated questions, denials and alerts.',
  platform: 'Changes to the platform itself — key rotation and billing statements.',
}

export const ACTION_META: Record<string, { label: string; tone: string; cat: Category }> = {
  // Knowledge — what entered or left the corpus
  ingest: { label: 'Document ingested', tone: '--text-heading', cat: 'knowledge' },
  index: { label: 'Document indexed', tone: '--text-heading', cat: 'knowledge' },
  graph: { label: 'Graph extracted', tone: '--text-heading', cat: 'knowledge' },
  delete: { label: 'Deleted', tone: '--danger', cat: 'knowledge' },
  erasure: { label: 'Erasure certificate', tone: '--danger', cat: 'knowledge' },
  classification: { label: 'Classification set', tone: '--warning', cat: 'knowledge' },
  // Answering — what was asked and answered
  query: { label: 'Question asked', tone: '--info', cat: 'answering' },
  retrieve: { label: 'Evidence retrieved', tone: '--info', cat: 'answering' },
  generate: { label: 'Answer generated', tone: '--success', cat: 'answering' },
  abstain: { label: 'Abstained', tone: '--text-muted', cat: 'answering' },
  source_query: { label: 'Source queried', tone: '--info', cat: 'answering' },
  sql_query: { label: 'SQL query', tone: '--info', cat: 'answering' },
  image_analysis: { label: 'Image read', tone: '--warning', cat: 'answering' },
  workflow_run: { label: 'Workflow run', tone: '--info', cat: 'answering' },
  agent_run: { label: 'Agent run', tone: '--info', cat: 'answering' },
  receipt: { label: 'Trust receipt', tone: '--success', cat: 'answering' },
  // Governance — what the platform checked, and who decided
  faithfulness: { label: 'Faithfulness checked', tone: '--warning', cat: 'governance' },
  guardrail: { label: 'Guardrail fired', tone: '--danger', cat: 'governance' },
  feedback: { label: 'User feedback', tone: '--accent', cat: 'governance' },
  hitl_view: { label: 'Review opened', tone: '--text-muted', cat: 'governance' },
  hitl_accept: { label: 'Review accepted', tone: '--success', cat: 'governance' },
  hitl_modify: { label: 'Review modified', tone: '--warning', cat: 'governance' },
  hitl_flag: { label: 'Review flagged', tone: '--warning', cat: 'governance' },
  hitl_override: { label: 'Review overridden', tone: '--warning', cat: 'governance' },
  eval_run: { label: 'Evaluation run', tone: '--info', cat: 'governance' },
  eval_alert: { label: 'Quality alert', tone: '--danger', cat: 'governance' },
  approval: { label: 'Dual-control decision', tone: '--success', cat: 'governance' },
  role_assign: { label: 'Role assigned', tone: '--warning', cat: 'governance' },
  model_request: { label: 'Model change request', tone: '--warning', cat: 'governance' },
  egress: { label: 'Egress decision', tone: '--warning', cat: 'governance' },
  // Federation — cross-department access
  grant_issue: { label: 'Grant issued', tone: '--success', cat: 'federation' },
  grant_revoke: { label: 'Grant revoked', tone: '--danger', cat: 'federation' },
  federated_query: { label: 'Federated question', tone: '--info', cat: 'federation' },
  federation_denied: { label: 'Federation denied', tone: '--danger', cat: 'federation' },
  federation_alert: { label: 'Federation alert', tone: '--danger', cat: 'federation' },
  federated_eval: { label: 'Federated evaluation', tone: '--info', cat: 'federation' },
  federation_quality_alert: { label: 'Peer quality alert', tone: '--danger', cat: 'federation' },
  // Platform — keys and money
  key_rotation: { label: 'Key rotated', tone: '--warning', cat: 'platform' },
  billing: { label: 'Billing statement', tone: '--text-heading', cat: 'platform' },
}

function meta(action: string) {
  return ACTION_META[action] ?? { label: action.replace(/_/g, ' '), tone: '--text-primary', cat: 'platform' as Category }
}

// Human labels for the payload keys the platform records, so the detail view reads clearly.
const FIELD_LABEL: Record<string, string> = {
  query: 'Question', top_k: 'Top-K', mode: 'Mode', model: 'Model', score: 'Faithfulness',
  outcome: 'Outcome', reason: 'Reason', category: 'Category', stage: 'Stage',
  cache_hit: 'Cache hit', agentic_retry: 'Agentic retry', confidence: 'Confidence',
  rating: 'Rating', comment: 'Comment', documents_erased: 'Documents erased',
  certificate_hash: 'Certificate', prompt_tokens: 'Prompt tokens', completion_tokens: 'Completion tokens',
  total_tokens: 'Total tokens', output_sha256: 'Output hash', citations: 'Citations',
  subject: 'Subject', chunks_purged: 'Chunks purged', conversations_purged: 'Chats scrubbed',
  kb_id: 'Knowledge base', doc_id: 'Document', chunks: 'Chunks', filename: 'File',
  classification: 'Classification', sql: 'SQL', safe_sql: 'Executed SQL', rows: 'Rows',
  total: 'Scored', breached: 'Breached floor', steps: 'Steps', peer: 'Peer', grant_id: 'Grant',
  entities: 'Entities', edges: 'Edges', duplicate_of: 'Duplicate of', reindex: 'Reindex',
  scope: 'Scope', object_uri: 'Stored object', content_hash: 'Content hash', modality: 'Modality',
  pii_masked: 'PII masked', injection: 'Injection check', replaced_chunks: 'Chunks replaced',
  intent: 'Intent', route_path: 'Route', route_reason: 'Route reason', route_top_k: 'Route top-K',
  max_score: 'Best score', threshold: 'Threshold', retrieved_chunk_ids: 'Retrieved chunks',
  unsupported_claims: 'Unsupported claims', self_corrected: 'Self-corrected',
  flag_threshold: 'Flag threshold', faithful_threshold: 'Faithful threshold',
  grounded_rate: 'Grounded rate', faithfulness_avg: 'Avg faithfulness', hit_rate: 'Hit rate',
  golden_n: 'Golden set', sampled_n: 'Live sample', breach_reasons: 'Breach reasons',
  reasons: 'Reasons', finish_reason: 'Finish reason', estimated: 'Token count estimated',
  prompt_sha256: 'Prompt hash', subqueries: 'Sub-queries', transform: 'Query transform',
  coverage: 'Coverage', plan_strategy: 'Plan', plan_rationale: 'Plan rationale',
  event: 'Event', admin_email: 'Admin', org_id: 'Organisation', plan: 'Plan',
  documents: 'Documents', stage_ms: 'Stage (ms)', graph_ms: 'Graph (ms)',
}

function fmtValue(v: unknown): string {
  if (v == null) return '—'
  if (Array.isArray(v)) return v.length ? v.map((x) => (typeof x === 'string' ? x : JSON.stringify(x))).join(', ') : '—'
  if (typeof v === 'boolean') return v ? 'yes' : 'no'
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

function str(v: unknown): string {
  return v == null ? '' : typeof v === 'string' ? v : fmtValue(v)
}

// One scannable line per event. Hand-written per action because a generic "first payload key"
// summary is what made the old table unreadable (every query row read `top_k: 8`).
export function summarise(e: AuditEvent): string {
  const p = e.payload
  // A payload field written by a future (or third-party) producer may not be a number; skip it
  // rather than rendering "score NaN" in the governance table.
  const n = (k: string) => {
    if (p[k] == null) return null
    const v = Number(p[k])
    return Number.isNaN(v) ? null : v
  }
  switch (e.action) {
    case 'ingest': {
      const name = str(p.filename) || str(p.doc_id)
      const outcome = str(p.outcome)
      return [name, outcome && outcome !== 'created' ? `· ${outcome}` : ''].filter(Boolean).join(' ')
    }
    case 'index': {
      const c = n('chunks')
      return [c != null ? `${c} chunk${c === 1 ? '' : 's'}` : '', p.reindex ? '· reindex' : '', str(p.doc_id)].filter(Boolean).join(' ')
    }
    case 'graph': {
      const ents = n('entities'); const edges = n('edges')
      return [ents != null ? `${ents} entities` : '', edges != null ? `${edges} edges` : ''].filter(Boolean).join(' · ')
    }
    case 'query':
    case 'federated_query':
      return str(p.query)
    case 'generate': {
      const tok = n('total_tokens')
      return [str(p.model), tok ? `${tok} tokens` : ''].filter(Boolean).join(' · ')
    }
    case 'faithfulness': {
      const s = n('score')
      return [s != null ? `score ${s.toFixed(2)}` : '', str(p.outcome)].filter(Boolean).join(' · ')
    }
    case 'abstain':
      return str(p.reason) || 'no grounded evidence'
    case 'guardrail':
      return [str(p.category), str(p.stage)].filter(Boolean).join(' · ') || 'blocked'
    case 'feedback':
      return [str(p.rating) === '1' || str(p.rating) === 'up' ? 'positive' : str(p.rating) ? 'negative' : '', str(p.comment)].filter(Boolean).join(' · ')
    case 'erasure': {
      const d = n('documents_erased')
      return [d != null ? `${d} document${d === 1 ? '' : 's'} erased` : 'erasure', str(p.subject)].filter(Boolean).join(' · ')
    }
    case 'delete': {
      const purged = n('chunks_purged')
      const what = str(p.doc_id) || str(p.kb_id) || str(p.target)
      const scope = str(p.scope).replace(/_/g, ' ')
      return [scope, what, purged ? `· ${purged} chunk${purged === 1 ? '' : 's'} purged` : ''].filter(Boolean).join(' ')
    }
    case 'classification':
      return [str(p.outcome), str(p.reason)].filter(Boolean).join(' · ')
    case 'eval_run': {
      const t = n('total')
      return [t != null ? `${t} scored` : '', p.breached ? '· breached floor' : ''].filter(Boolean).join(' ')
    }
    case 'eval_alert':
      return str(p.reasons) || str(p.reason) || 'quality floor breached'
    case 'sql_query': {
      const r = n('rows')
      return [str(p.query) || str(p.question), r != null ? `· ${r} rows` : ''].filter(Boolean).join(' ')
    }
    case 'workflow_run':
    case 'agent_run':
      return [str(p.workflow_id) || str(p.goal), str(p.outcome)].filter(Boolean).join(' · ')
    case 'grant_issue':
    case 'grant_revoke':
    case 'federation_denied':
      return [str(p.peer) || str(p.audience), str(p.topic) || str(p.reason)].filter(Boolean).join(' · ')
    case 'key_rotation':
      return str(p.outcome) || 'tenant key rotated'
    default: {
      // Unknown/new action: prefer the keys that carry meaning, never the tuning knobs.
      for (const k of ['query', 'reason', 'reasons', 'outcome', 'event', 'detail', 'message', 'scope', 'doc_id', 'certificate_hash']) {
        const v = str(p[k])
        if (v) return v
      }
      return ''
    }
  }
}

// Everything one event can be searched by, lower-cased. Deliberately wide: the raw action AND
// its human label (so both "index" and "Document indexed" hit), the actor, the summary line, the
// ids, the seal, and the raw payload (so a document name or a model name is findable). The old
// page matched the typed text against `trace_id` ALONE, which is why searching "index" returned
// nothing at all.
// Cached per event object: the loaded window is re-filtered on every keystroke, and
// re-serialising each payload each time is what turns a large window into a laggy search box.
// Events are immutable once fetched, so caching on identity is safe, and a WeakMap lets a
// replaced window be collected with no bookkeeping.
const haystackCache = new WeakMap<AuditEvent, string>()

export function eventHaystack(e: AuditEvent): string {
  const hit = haystackCache.get(e)
  if (hit !== undefined) return hit
  const hay = [
    e.action, meta(e.action).label, e.actor, summarise(e), e.trace_id, e.event_id,
    e.event_hash ?? '', JSON.stringify(e.payload),
  ].join(' ').toLowerCase()
  haystackCache.set(e, hay)
  return hay
}

// --- Time window ----------------------------------------------------------
type Range = '24h' | '7d' | '30d' | 'all'
const RANGE_MS: Record<Range, number | null> = { '24h': 864e5, '7d': 6048e5, '30d': 2592e6, all: null }
const RANGE_OPTIONS: { value: Range; label: string }[] = [
  { value: '24h', label: '24h' }, { value: '7d', label: '7d' }, { value: '30d', label: '30d' }, { value: 'all', label: 'All' },
]

function relative(iso: string): string {
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return iso
  const secs = Math.round((Date.now() - then) / 1000)
  if (secs < 60) return 'just now'
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`
  if (secs < 2592000) return `${Math.floor(secs / 86400)}d ago`
  return new Date(iso).toLocaleDateString()
}

function dayKey(iso: string): string {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toDateString()
}

function dayLabel(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  const today = new Date().toDateString()
  const yesterday = new Date(Date.now() - 864e5).toDateString()
  if (d.toDateString() === today) return 'Today'
  if (d.toDateString() === yesterday) return 'Yesterday'
  return d.toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' })
}

// --- Hash chain -----------------------------------------------------------
// Each event seals the hash of whatever was at the tip of the chain when it was written, so a
// page of the trail can be walked to look for events that have gone missing.
//
// It is walked as a SET, not a straight line, and that distinction matters: `append()` reads the
// current tip and then inserts, so two events written concurrently (the continuous-eval scheduler
// and a live query, say) legitimately seal the SAME parent and the chain forks. Measured on this
// deployment's real trail: 5 such forks in 250 events, every one a pair of genuine concurrent
// writes. A naive "my prev_hash == the previous row's hash" check therefore reports a healthy,
// untampered trail as broken. Do not reinstate it.
//
// WHAT THIS DOES AND DOES NOT ESTABLISH — read before changing the label it drives:
//   It checks LINK TOPOLOGY ONLY, within the loaded window. It catches an event removed from the
//   middle of the chain, because its child is left pointing at a parent that is gone.
//   It does NOT catch:
//     · a rewritten event — the console cannot recompute `compute_hash()` over the payload, so an
//       edit that leaves the hash columns untouched resolves perfectly here;
//     · deletion of a LEAF — the newest event, or either tip of a fork, is sealed by nothing, so
//       removing it orphans nothing. `unreferenced` counts exactly how many such events are in
//       the window, so the UI can say so rather than imply full coverage.
//   Hence the verdict says "Links intact", never "Verified". The real content-level verification
//   is server-side (`compliance.py:_verify_audit_chain`, surfaced as the attestation's
//   `audit_chain_verified`) — that is what a regulator should be shown, not this.
//
// A MISSING PARENT IS ONLY MEANINGFUL ON A COMPLETE TRAIL. Where the chain forks, the newest-N
// page is not closed under "is the parent of": a sibling branch's event can sit just below the
// window floor while rows above it still point at it. Sweeping every window size over this
// deployment's real trail found exactly one such n (207 of 250) where a healthy chain looked
// like it had a gap. So an unresolved parent is reported as a gap ONLY when the whole trail is
// loaded, where it is unambiguous; on a truncated window it is noted as uncovered edge instead
// of raised as a finding. `tools/audit-logic.check.ts` sweeps all window sizes to keep it so.
type ChainVerdict = 'intact' | 'gap' | 'anomaly'
type Chain = {
  resolved: number      // links whose parent event is present in this window
  unresolved: number    // parent missing and NOT the window's own boundary parent
  forks: number         // one parent sealed by >1 event: concurrent writes, not damage
  unreferenced: number  // events nothing seals on — removable without leaving a trace here
  anomalies: number     // duplicate hash, or a second "genesis"
  complete: boolean     // was the whole trail loaded, or only its newest page?
  verdict: ChainVerdict
} | null

export function verifyChain(events: AuditEvent[], opts: { complete: boolean } = { complete: false }): Chain {
  if (events.length < 2) return null
  const hashes = new Set<string>()
  let anomalies = 0
  for (const e of events) {
    if (!e.event_hash) continue
    if (hashes.has(e.event_hash)) anomalies++ // the same sealed event appearing twice
    hashes.add(e.event_hash)
  }
  // The window is a contiguous newest-first page, so the events at its oldest edge seal a parent
  // that legitimately predates it. A FORK can straddle that edge — two siblings both sealing the
  // same out-of-window parent — so exempting only the last row would report a healthy boundary
  // fork as a gap. Exempt that parent hash itself, which covers every sibling of it.
  let edgeParent: string | null = null
  for (let i = events.length - 1; i >= 0; i--) {
    if (events[i].event_hash) { edgeParent = events[i].prev_hash; break }
  }
  const parents = new Map<string, number>()
  let resolved = 0
  let unresolved = 0
  for (const e of events) {
    if (!e.event_hash) continue // unsealed row (in-memory sink) — nothing to verify
    if (e.prev_hash == null) {
      // Genesis. Legitimate only at the oldest edge of the window.
      if (e.prev_hash !== edgeParent) anomalies++
      continue
    }
    parents.set(e.prev_hash, (parents.get(e.prev_hash) ?? 0) + 1)
    if (hashes.has(e.prev_hash)) resolved++
    else if (e.prev_hash !== edgeParent) unresolved++
  }
  if (!resolved && !unresolved && !anomalies) return null
  let forks = 0
  for (const count of parents.values()) if (count > 1) forks++
  // Leaves: sealed events that nothing in this window points back to.
  let unreferenced = 0
  for (const e of events) if (e.event_hash && !parents.has(e.event_hash)) unreferenced++
  // Only a complete trail can turn a missing parent into a finding (see the note above).
  const verdict: ChainVerdict = anomalies ? 'anomaly' : (opts.complete && unresolved) ? 'gap' : 'intact'
  return { resolved, unresolved, forks, unreferenced, anomalies, verdict, complete: opts.complete }
}

// Deliberately modest wording. "Verified" would assert integrity this window check cannot
// establish (see the note above), and would also contradict the server attestation, which
// additionally re-hashes every payload.
const CHAIN_VIEW: Record<ChainVerdict, { label: string; tone: string }> = {
  intact: { label: 'Links intact', tone: '--success' },
  gap: { label: 'Gap in window', tone: '--warning' },
  anomaly: { label: 'Needs a look', tone: '--danger' },
}

function chainSub(c: NonNullable<Chain>): string {
  if (c.verdict === 'anomaly') return `${c.anomalies} event${c.anomalies === 1 ? '' : 's'} do not fit this chain`
  if (c.verdict === 'gap') {
    // Neutral: a short window and a removed event look identical from here, and only one of
    // them is routine. Do not state the reassuring cause as fact.
    return `${c.unresolved} event${c.unresolved === 1 ? '' : 's'} point at a parent that is not loaded`
  }
  const parts = [`${c.resolved} link${c.resolved === 1 ? '' : 's'} resolved`]
  if (c.forks) parts.push(`${c.forks} concurrent write${c.forks === 1 ? '' : 's'}`)
  if (!c.complete) parts.push('oldest edge not covered')
  return parts.join(' · ')
}

const CHAIN_HINT = 'Checks that every loaded event still points at an event that is present — it '
  + 'catches an event removed from the middle of the trail. It does not re-hash payloads, and an '
  + 'event that nothing seals on (the newest, or either tip of a concurrent write) is not covered. '
  + 'The full content-level check is the signed attestation on the Compliance surface.'

const LOAD_STEP = 250

export function Audit() {
  const { session } = useAuth()
  const { t } = useI18n()
  const { kbId } = useKb()

  const [events, setEvents] = useState<AuditEvent[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [limit, setLimit] = useState(LOAD_STEP)
  const [truncated, setTruncated] = useState(false)

  // Filters. `trace` is the only one the API understands, so it alone triggers a refetch.
  const [trace, setTrace] = useState('')
  const [q, setQ] = useState('')
  const [cat, setCat] = useState<Category | 'all'>('all')
  const [action, setAction] = useState('all')
  const [range, setRange] = useState<Range>('all')
  const [open, setOpen] = useState<Set<string>>(new Set())
  const searchRef = useRef<HTMLInputElement>(null)

  // Switching knowledge base resets the window and the filters: a 2,000-row window or an
  // isolated trace from the previous KB means nothing here. Done during render, not in an
  // effect, so the reset is visible to the fetch effect below on this same pass — an effect
  // would let it fire once for {new KB, OLD trace} before the reset landed, firing a second
  // request whose response can arrive first.
  const lastKb = useRef(kbId)
  if (lastKb.current !== kbId) {
    lastKb.current = kbId
    setLimit(LOAD_STEP); setTrace(''); setQ(''); setCat('all'); setAction('all'); setRange('all')
    // Clear the DATA too, not only the filters: `load` flips `loading` in a passive effect after
    // commit, so without this the committed tree still holds the previous KB's rows and can paint
    // one department's actors and questions under another department's header.
    setEvents([]); setTruncated(false); setError(null); setLoading(true)
  }

  // Only the newest request may write state. Without this, two in-flight reads (switch KB,
  // hit Refresh, click "load older") resolve in arrival order rather than request order, and
  // a slow earlier response silently replaces the window the user is actually looking at.
  const reqSeq = useRef(0)

  const load = useCallback(async (traceId: string, take: number) => {
    if (!session) return
    const seq = ++reqSeq.current
    setLoading(true); setError(null); setOpen(new Set())
    try {
      // Over-fetch by one: the API reports no total, so `rows.length === take` cannot tell
      // "exactly `take` exist" from "more exist". Asking for take+1 and showing `take` makes
      // the distinction exact, so a KB with exactly 250 events stops offering a dead pager.
      const rows = await api.listAudit(session.token, kbId, traceId || undefined, take + 1)
      if (seq !== reqSeq.current) return // superseded
      setEvents(rows.slice(0, take))
      setTruncated(rows.length > take)
    } catch (err) {
      if (seq !== reqSeq.current) return
      setError(err instanceof ApiError ? err.message : 'Failed to load audit trail')
      setEvents([])
      setTruncated(false)
    } finally {
      if (seq === reqSeq.current) setLoading(false)
    }
  }, [session, kbId])

  useEffect(() => { void load(trace, limit) }, [load, trace, limit])

  // "/" focuses search, the way every log viewer behaves.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const el = e.target as HTMLElement | null
      if (e.key === '/' && !e.metaKey && !e.ctrlKey && !e.altKey && !el?.isContentEditable
        && el?.tagName !== 'INPUT' && el?.tagName !== 'TEXTAREA' && el?.tagName !== 'SELECT') {
        e.preventDefault(); searchRef.current?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // `truncated` says there are older events than the window holds, so the chain can only be
  // spoken for as a complete trail when it is false.
  const chain = useMemo(
    () => (trace ? null : verifyChain(events, { complete: !truncated })),
    [events, trace, truncated],
  )

  // Action options, counted — only the actions actually present in the window, grouped by the
  // category they belong to, so the dropdown never offers a filter that would return nothing.
  // Per-action and per-category tallies over the loaded window. Categories are always all five,
  // including the empty ones: a chip reading "Federation 0" tells an auditor this KB has had no
  // cross-department activity, which is itself a finding. Actions are only those present, so the
  // dropdown can never offer a filter that returns nothing.
  const { catCounts, actionsByCat } = useMemo(() => {
    const counts = new Map<string, number>()
    for (const e of events) counts.set(e.action, (counts.get(e.action) ?? 0) + 1)
    const byCat = new Map<Category, { action: string; label: string; count: number }[]>()
    const cats = new Map<Category, number>(CATEGORY_ORDER.map((c) => [c, 0]))
    for (const [a, count] of counts) {
      const m = meta(a)
      const list = byCat.get(m.cat) ?? []
      list.push({ action: a, label: m.label, count })
      byCat.set(m.cat, list)
      cats.set(m.cat, (cats.get(m.cat) ?? 0) + count)
    }
    for (const list of byCat.values()) list.sort((a, b) => b.count - a.count)
    return { catCounts: cats, actionsByCat: byCat }
  }, [events])

  // The dropdown lists the chosen category's actions; with no category chosen it lists all of
  // them, still grouped, so nothing is unreachable without picking a chip first.
  const actionGroups = useMemo(
    () => CATEGORY_ORDER
      .filter((c) => (cat === 'all' || c === cat) && (actionsByCat.get(c)?.length ?? 0) > 0)
      .map((c) => [c, actionsByCat.get(c) ?? []] as const),
    [actionsByCat, cat],
  )

  // Picking a category drops an action refinement that belongs to a different one, so the two
  // controls can never contradict each other (and show zero rows while both look set).
  useEffect(() => {
    if (cat !== 'all' && action !== 'all' && meta(action).cat !== cat) setAction('all')
  }, [cat, action])

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase()
    const span = RANGE_MS[range]
    const floor = span == null ? null : Date.now() - span
    return events.filter((e) => {
      if (cat !== 'all' && meta(e.action).cat !== cat) return false
      if (action !== 'all' && e.action !== action) return false
      if (floor != null) {
        const at = new Date(e.at).getTime()
        if (!Number.isNaN(at) && at < floor) return false
      }
      if (!needle) return true
      return eventHaystack(e).includes(needle)
    })
  }, [events, q, cat, action, range])

  const activeFilters = (q.trim() ? 1 : 0) + (cat !== 'all' ? 1 : 0) + (action !== 'all' ? 1 : 0)
    + (range !== 'all' ? 1 : 0) + (trace ? 1 : 0)
  const clearAll = () => { setQ(''); setCat('all'); setAction('all'); setRange('all'); setTrace('') }

  const actors = useMemo(() => new Set(filtered.map((e) => e.actor)).size, [filtered])
  const span = useMemo(() => {
    if (!filtered.length) return null
    const times = filtered.map((e) => new Date(e.at).getTime()).filter((n) => !Number.isNaN(n))
    if (!times.length) return null
    // reduce, not spread: `limit` has no server-side ceiling and the pager keeps raising it,
    // and spreading a six-figure array throws RangeError.
    return times.reduce((acc, t) => ({ from: Math.min(acc.from, t), to: Math.max(acc.to, t) }),
      { from: times[0], to: times[0] })
  }, [filtered])

  function toggle(id: string) {
    setOpen((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n })
  }

  function exportCsv() {
    const head = ['at', 'action', 'actor', 'summary', 'trace_id', 'event_id', 'prev_hash', 'event_hash', 'payload']
    // Quote-escape for CSV, AND neutralise spreadsheet formula injection. The payload carries
    // text typed by end users (every question asked), so a value beginning =, +, - or @ would
    // be evaluated as a formula when an auditor opens the export in Excel or Sheets. Prefixing
    // a single quote keeps the value readable and inert.
    const cell = (v: unknown) => {
      const raw = String(v ?? '')
      const safe = /^[=+\-@\t\r]/.test(raw) ? `'${raw}` : raw
      return `"${safe.replace(/"/g, '""')}"`
    }
    const body = filtered.map((e) => [
      e.at, e.action, e.actor, summarise(e), e.trace_id, e.event_id, e.prev_hash ?? '', e.event_hash ?? '',
      JSON.stringify(e.payload),
    ].map(cell).join(','))
    // State the scope INSIDE the artefact. Every number on screen is scoped to the loaded
    // window, but this file is the thing that leaves the browser and reaches an auditor — it
    // must not be mistakable for the complete trail.
    const scope = `# RAG as a Service audit export · knowledge base ${kbId} · generated ${new Date().toISOString()}`
    const window_ = `# Scope: ${filtered.length} of ${events.length} loaded events`
      + `${truncated ? ' (the newest page of a longer trail — older events are NOT included)' : ' (the complete trail at time of export)'}`
      + `${trace ? ` · isolated to trace ${trace}` : ''}${q.trim() ? ` · matching "${q.trim()}"` : ''}`
      + `${action !== 'all' ? ` · action ${action}` : ''}${range !== 'all' ? ` · last ${range}` : ''}`
    // Leading BOM: Excel on Windows ignores the charset and assumes the local codepage, which
    // mojibakes Devanagari — and this console ships a Hindi locale.
    const csv = '\ufeff' + [scope, window_, head.join(','), ...body].join('\n')
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `audit-${kbId}-${filtered.length}of${events.length}-${new Date().toISOString().slice(0, 10)}.csv`
    a.click()
    // Revoking in the same tick can abort the download in some browsers; mirror Mermaid.tsx.
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  const isolate = (traceId: string) => { setAction('all'); setQ(''); setRange('all'); setTrace(traceId) }

  return (
    <div>
      <PageHeader
        eyebrow={t('eyebrow.govern')}
        title={t('page.audit.title')}
        actions={
          <>
            <Button variant="subtle" size="sm" icon={<Icon.refresh size={15} />} onClick={() => void load(trace, limit)} disabled={loading}>Refresh</Button>
            <Button variant="ghost" size="sm" icon={<Icon.doc size={15} />} onClick={exportCsv} disabled={!filtered.length}
              title="Exports the events currently loaded and filtered on this page, not the entire trail. The file states its own scope.">Export CSV</Button>
          </>
        }
      />

      {/* Window summary — every number is scoped to what is loaded and filtered, never claimed
          as a lifetime total, because the API returns a newest-first page. */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
        <MiniStat label="Events shown" value={loading ? '—' : filtered.length.toLocaleString()}
          sub={filtered.length === events.length ? `of ${events.length.toLocaleString()} loaded` : `filtered from ${events.length.toLocaleString()}`} />
        <MiniStat label="Actors" value={loading ? '—' : actors || '—'} sub={actors === 1 ? 'single actor' : 'distinct actors'} />
        <MiniStat
          label="Chain links"
          hint={CHAIN_HINT}
          value={chain ? CHAIN_VIEW[chain.verdict].label : '—'}
          tone={chain ? CHAIN_VIEW[chain.verdict].tone : '--text-muted'}
          icon={chain ? (chain.verdict === 'intact' ? <Icon.shield size={14} /> : <Icon.alert size={14} />) : undefined}
          sub={chain ? chainSub(chain) : trace ? 'not checked on a single trace' : 'needs 2+ sealed events'}
        />
        <MiniStat label="Window" value={span ? relative(new Date(span.from).toISOString()) : '—'}
          sub={span ? `oldest shown · newest ${relative(new Date(span.to).toISOString())}` : 'nothing in range'} />
      </div>

      {/* Toolbar — one row, each control labelled with what it actually matches. */}
      <Card className="!p-3 mb-4">
        {/* Category is the primary cut: an auditor asks "show me the governance decisions", not
            "show me the hitl_override events". The action dropdown below refines within it. */}
        <div className="flex flex-wrap items-center gap-1.5 mb-2.5 pb-2.5" style={{ borderBottom: '1px solid var(--border)' }}>
          <CategoryChip label="All" count={events.length} active={cat === 'all'} onClick={() => setCat('all')}
            hint="Every recorded event in the loaded window." />
          {CATEGORY_ORDER.map((c) => (
            <CategoryChip key={c} label={CATEGORY_LABEL[c]} count={catCounts.get(c) ?? 0}
              active={cat === c} onClick={() => setCat(cat === c ? 'all' : c)} hint={CATEGORY_HINT[c]} />
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative flex-grow" style={{ minWidth: 240 }}>
            <Icon.ask size={15} style={{ position: 'absolute', left: 12, top: 13, color: 'var(--text-muted)', pointerEvents: 'none' }} />
            <input
              ref={searchRef} className="field-input" value={q} onChange={(e) => setQ(e.target.value)}
              aria-label="Search the audit trail"
              placeholder="Search action, actor, question, document, hash…   ( / )"
              style={{ paddingLeft: 36, minHeight: 40, fontSize: 14 }}
            />
            {q && (
              <button type="button" onClick={() => setQ('')} aria-label="Clear search"
                className="absolute inline-flex items-center justify-center"
                style={{ right: 8, top: 8, width: 24, height: 24, color: 'var(--text-muted)', cursor: 'pointer' }}>
                <Icon.x size={15} />
              </button>
            )}
          </div>

          <select
            value={action} onChange={(e) => setAction(e.target.value)} aria-label="Filter by action"
            className="rounded-md text-sm font-semibold px-3"
            style={{ minHeight: 40, maxWidth: 260, color: 'var(--text-heading)', background: 'var(--surface-2)', border: '1px solid var(--border-strong)', cursor: 'pointer' }}
          >
            <option value="all">
              {cat === 'all' ? `All actions${events.length ? ` (${events.length})` : ''}`
                : `All ${CATEGORY_LABEL[cat].toLowerCase()} actions (${catCounts.get(cat) ?? 0})`}
            </option>
            {actionGroups.map(([cat, list]) => (
              <optgroup key={cat} label={CATEGORY_LABEL[cat]}>
                {list.map((o) => <option key={o.action} value={o.action}>{o.label} ({o.count})</option>)}
              </optgroup>
            ))}
          </select>

          <SegmentedControl options={RANGE_OPTIONS} value={range} onChange={setRange} />
        </div>

        {(activeFilters > 0 || truncated) && (
          <div className="flex flex-wrap items-center gap-2 mt-2.5 pt-2.5" style={{ borderTop: '1px solid var(--border)' }}>
            {trace && (
              <FilterChip label="One request chain" value={trace} onClear={() => setTrace('')} mono />
            )}
            {q.trim() && <FilterChip label="Matching" value={q.trim()} onClear={() => setQ('')} />}
            {cat !== 'all' && <FilterChip label="Category" value={CATEGORY_LABEL[cat]} onClear={() => setCat('all')} />}
            {action !== 'all' && <FilterChip label="Action" value={meta(action).label} onClear={() => setAction('all')} />}
            {range !== 'all' && <FilterChip label="Last" value={range} onClear={() => setRange('all')} />}
            {activeFilters > 1 && (
              <button type="button" onClick={clearAll} className="text-xs font-semibold underline"
                style={{ color: 'var(--action)', cursor: 'pointer' }}>Clear all</button>
            )}
            {truncated && (
              <span className="text-xs ml-auto" style={{ color: 'var(--text-muted)' }}>
                Showing the newest {events.length.toLocaleString()} events — older ones are outside this window.
              </span>
            )}
          </div>
        )}
      </Card>

      {error && <Alert kind="error">{error}</Alert>}
      {/* Only skeleton the FIRST load. "Load older events" and Refresh refetch the whole window,
          and swapping the table for a skeleton would collapse every expanded row and throw the
          reader back to the top of the page mid-audit. */}
      {loading && events.length === 0 && <Skeleton style={{ height: 260 }} />}

      {!loading && !error && filtered.length === 0 && (
        events.length === 0 && !trace ? (
          <EmptyState icon={<Icon.audit size={28} />} title="Nothing recorded for this knowledge base yet"
            hint="Ingest a document or ask a question — every step is written to the hash-chained trail as it happens." />
        ) : (
          <EmptyState icon={<Icon.ask size={28} />} title="No events match these filters"
            hint={trace
              ? 'That trace id returned nothing. It is an exact match on a full request id — use the search box for text, actions or actors.'
              : cat !== 'all' && (catCounts.get(cat) ?? 0) === 0
                ? `This knowledge base has recorded no ${CATEGORY_LABEL[cat].toLowerCase()} activity in the loaded window.`
                : `${events.length.toLocaleString()} event${events.length === 1 ? '' : 's'} are loaded; none of them match. Widen the time range or clear the filters.`}
            action={<Button variant="ghost" size="sm" onClick={clearAll}>Clear filters</Button>} />
        )
      )}

      {filtered.length > 0 && (
        <Card className="!p-0 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="tbl" style={{ width: '100%' }}>
              <thead>
                <tr>
                  <th style={{ width: 28 }}></th>
                  <th style={{ width: 92 }}>When</th>
                  <th style={{ width: 210 }}>Action</th>
                  <th style={{ width: 170 }}>Actor</th>
                  <th>Detail</th>
                  <th style={{ width: 130 }}>Event hash</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((e, i) => {
                  const isOpen = open.has(e.event_id)
                  const m = meta(e.action)
                  const summary = summarise(e)
                  // Day separators only make sense on a time-ordered list, which is what the
                  // API returns (newest first) — so group as we walk it.
                  const newDay = i === 0 || dayKey(e.at) !== dayKey(filtered[i - 1].at)
                  return (
                    <Fragment key={e.event_id}>
                      {newDay && (
                        <tr>
                          <td colSpan={6} className="mono uppercase font-bold tracking-wider"
                            style={{ fontSize: 10, color: 'var(--text-muted)', background: 'var(--surface-2)', padding: '6px 16px', borderBottom: '1px solid var(--border)' }}>
                            {dayLabel(e.at)}
                          </td>
                        </tr>
                      )}
                      <tr onClick={() => toggle(e.event_id)} style={{ cursor: 'pointer' }}>
                        {/* A real button, not just a clickable row: this disclosure is the only
                            route to the payload, the trace isolation and both hashes, and a row
                            click alone leaves all of it unreachable by keyboard or screen reader
                            on the console's evidence-of-record page. The row click stays as a
                            mouse convenience. */}
                        <td style={{ color: 'var(--text-muted)' }}>
                          <button type="button" aria-expanded={isOpen}
                            aria-label={`${isOpen ? 'Hide' : 'Show'} details of ${m.label} at ${new Date(e.at).toLocaleString()}`}
                            onClick={(ev) => { ev.stopPropagation(); toggle(e.event_id) }}
                            className="inline-flex items-center justify-center rounded"
                            style={{ width: 24, height: 24, margin: -3, color: 'inherit', cursor: 'pointer' }}>
                            <Icon.chevron size={13} style={{ transform: isOpen ? 'rotate(90deg)' : 'none', transition: 'transform .15s' }} />
                          </button>
                        </td>
                        <td className="mono num text-xs" style={{ color: 'var(--text-muted)', whiteSpace: 'nowrap' }} title={new Date(e.at).toLocaleString()}>
                          {relative(e.at)}
                        </td>
                        <td>
                          <span className="inline-flex items-center gap-2" style={{ whiteSpace: 'nowrap' }}>
                            <span className="inline-block rounded-full shrink-0" style={{ width: 7, height: 7, background: `var(${m.tone})` }} />
                            <span className="font-semibold text-sm" style={{ color: 'var(--text-heading)' }}>{m.label}</span>
                          </span>
                        </td>
                        <td className="text-sm" style={{ maxWidth: 170, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={e.actor}>{e.actor}</td>
                        <td className="text-sm" style={{ maxWidth: 420, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--text-muted)' }} title={summary}>
                          {summary || <span style={{ opacity: .6 }}>—</span>}
                        </td>
                        <td className="mono text-xs" style={{ color: 'var(--text-muted)', whiteSpace: 'nowrap' }} title={`this ${e.event_hash ?? '∅'}\nprev ${e.prev_hash ?? '∅'}`}>
                          {e.event_hash ? `${e.event_hash.slice(0, 12)}…` : '—'}
                        </td>
                      </tr>
                      {isOpen && (
                        <tr>
                          <td colSpan={6} style={{ background: 'var(--surface-2)', padding: '14px 20px' }}>
                            <EventDetail event={e} onTrace={isolate} />
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  )
                })}
              </tbody>
            </table>
          </div>
          <div className="flex items-center justify-between gap-3 px-4 py-3" style={{ borderTop: '1px solid var(--border)' }}>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {filtered.length.toLocaleString()} of {events.length.toLocaleString()} loaded event{events.length === 1 ? '' : 's'} · most recent first
            </span>
            {truncated && !trace && (
              <Button variant="subtle" size="sm" icon={<Icon.plus size={14} />} onClick={() => setLimit((l) => l + LOAD_STEP * 3)} loading={loading}>
                Load older events
              </Button>
            )}
          </div>
        </Card>
      )}
    </div>
  )
}

// --- Pieces ---------------------------------------------------------------
function MiniStat({ label, value, sub, tone = '--text-heading', icon, hint }:
  { label: string; value: React.ReactNode; sub?: string; tone?: string; icon?: React.ReactNode; hint?: string }) {
  return (
    <div className="surface px-4 py-3" title={hint}>
      <div className="flex items-center justify-between gap-2">
        <span className="mono uppercase" style={{ fontSize: 10, color: 'var(--text-muted)', letterSpacing: '.06em' }}>{label}</span>
        {icon && <span style={{ color: `var(${tone})` }}>{icon}</span>}
      </div>
      <div className="font-display font-bold num mt-0.5" style={{ fontSize: 20, lineHeight: 1.1, color: `var(${tone})` }}>{value}</div>
      {sub && <div className="text-xs mt-0.5 truncate" style={{ color: 'var(--text-muted)' }} title={sub}>{sub}</div>}
    </div>
  )
}

// A category selector. Empty categories stay visible but recede: "Federation 0" is information
// (this KB has had no cross-department activity), not a control to hide.
function CategoryChip({ label, count, active, onClick, hint }:
  { label: string; count: number; active: boolean; onClick: () => void; hint: string }) {
  const empty = count === 0
  return (
    <button
      type="button" onClick={onClick} title={hint} aria-pressed={active}
      className="inline-flex items-center gap-1.5 rounded-md font-semibold transition-colors"
      style={{
        minHeight: 30, padding: '0 10px', fontSize: 12.5, cursor: 'pointer',
        color: active ? 'var(--on-action)' : empty ? 'var(--text-muted)' : 'var(--text-heading)',
        background: active ? 'var(--action)' : 'var(--surface-2)',
        border: `1px solid ${active ? 'var(--action)' : 'var(--border)'}`,
        opacity: !active && empty ? 0.6 : 1,
      }}
    >
      {label}
      <span className="mono num" style={{ fontSize: 11, opacity: active ? 0.85 : 0.7 }}>{count.toLocaleString()}</span>
    </button>
  )
}

function FilterChip({ label, value, onClear, mono }: { label: string; value: string; onClear: () => void; mono?: boolean }) {
  return (
    <span className="chip" style={{ maxWidth: 420 }}>
      <span className="uppercase font-bold" style={{ fontSize: 9, letterSpacing: '.08em', color: 'var(--text-muted)' }}>{label}</span>
      <span className={mono ? 'mono' : ''} style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--text-heading)' }} title={value}>{value}</span>
      <button type="button" onClick={onClear} aria-label={`Clear ${label} filter`}
        className="inline-flex items-center justify-center"
        style={{ width: 24, height: 24, margin: '-4px -6px -4px 0', color: 'var(--text-muted)', cursor: 'pointer' }}>
        <Icon.x size={12} />
      </button>
    </span>
  )
}

function EventDetail({ event, onTrace }: { event: AuditEvent; onTrace: (t: string) => void }) {
  const [copied, setCopied] = useState<string | null>(null)
  const entries = Object.entries(event.payload).filter(([, v]) => v != null && !(Array.isArray(v) && v.length === 0))
  const wide = (k: string) => ['query', 'question', 'comment', 'sql', 'safe_sql', 'output_sha256', 'certificate_hash', 'reason'].includes(k)

  function copy(text: string, what: string) {
    void navigator.clipboard?.writeText(text).then(() => {
      setCopied(what)
      window.setTimeout(() => setCopied(null), 1400)
    }).catch(() => undefined)
  }

  return (
    <div>
      <div className="flex items-center gap-2 mb-3 flex-wrap text-xs">
        <span className="mono" style={{ color: 'var(--text-muted)' }}>{event.action}</span>
        <span style={{ color: 'var(--border-strong)' }}>·</span>
        <span style={{ color: 'var(--text-muted)' }}>{new Date(event.at).toLocaleString()}</span>
        <span style={{ color: 'var(--border-strong)' }}>·</span>
        <button type="button" onClick={() => onTrace(event.trace_id)} className="mono underline"
          style={{ color: 'var(--info)', cursor: 'pointer' }} title="Show only this request's chain of events">
          trace {event.trace_id.slice(0, 18)}{event.trace_id.length > 18 ? '…' : ''}
        </button>
        <button type="button" onClick={() => copy(event.trace_id, 'trace')} className="inline-flex items-center justify-center gap-1"
          style={{ minWidth: 24, minHeight: 24, padding: '0 4px', color: 'var(--text-muted)', cursor: 'pointer' }}
          aria-label="Copy trace id" title="Copy trace id">
          <Icon.copy size={12} />{copied === 'trace' ? 'copied' : ''}
        </button>
        <button type="button" onClick={() => copy(JSON.stringify(event, null, 2), 'json')} className="inline-flex items-center justify-center gap-1 ml-auto"
          style={{ minHeight: 24, padding: '0 4px', color: 'var(--text-muted)', cursor: 'pointer' }}
          aria-label="Copy the whole event as JSON" title="Copy the whole event as JSON">
          <Icon.copy size={12} />{copied === 'json' ? 'copied' : 'copy JSON'}
        </button>
      </div>

      {entries.length === 0
        ? <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No additional detail on this event.</p>
        : (
          <dl className="grid sm:grid-cols-2 gap-x-8 gap-y-2.5">
            {entries.map(([k, v]) => (
              <div key={k} className={wide(k) ? 'sm:col-span-2' : ''}>
                <dt className="text-xs font-semibold uppercase" style={{ color: 'var(--text-muted)', letterSpacing: '.05em' }}>{FIELD_LABEL[k] ?? k.replace(/_/g, ' ')}</dt>
                <dd className={`text-sm mt-0.5 ${k.includes('hash') || k.includes('sha') ? 'mono' : ''}`} style={{ color: 'var(--text-primary)', wordBreak: 'break-word' }}>{fmtValue(v)}</dd>
              </div>
            ))}
          </dl>
        )}

      {/* The seal. Shown last because it is provenance, not content. */}
      <dl className="mt-3 pt-3 grid sm:grid-cols-2 gap-x-8 gap-y-1.5" style={{ borderTop: '1px solid var(--border)' }}>
        <div>
          <dt className="text-xs font-semibold uppercase" style={{ color: 'var(--text-muted)', letterSpacing: '.05em' }}>Sealed on previous</dt>
          <dd className="mono text-xs mt-0.5" style={{ color: 'var(--text-muted)', wordBreak: 'break-all' }}>{event.prev_hash ?? 'genesis — first event in this chain'}</dd>
        </div>
        <div>
          <dt className="text-xs font-semibold uppercase" style={{ color: 'var(--text-muted)', letterSpacing: '.05em' }}>This event hash</dt>
          <dd className="mono text-xs mt-0.5" style={{ color: 'var(--text-primary)', wordBreak: 'break-all' }}>{event.event_hash ?? '—'}</dd>
        </div>
      </dl>
    </div>
  )
}
