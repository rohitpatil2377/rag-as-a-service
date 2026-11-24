import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { Icon } from '../components/icons'
import { Markdown } from '../components/Markdown'
import { MermaidBlock } from '../components/Mermaid'
import { useToast } from '../components/overlay'
import { AnswerPill, Button, SegmentedControl } from '../components/ui'
import { DegradedNote, MethodBreakdown, VerificationScore, VerificationStrip } from '../components/Verification'
import { describeVerification } from '../lib/verification'
import { CitationViewer } from '../components/CitationViewer'
import { ApiError, api, streamConversationTurn, type AgentStep, type AnswerStatus, type Citation, type ConversationInfo, type ConversationMessage, type GroundedAnswer } from '../lib/api'
import { useAuth } from '../lib/auth'
import { useI18n } from '../lib/i18n'
import { useKb } from '../lib/kb'

const convKey = (kbId: string) => `rag:conv:${kbId}`

// Rebuild a transcript message from a persisted conversation turn (FR-I1 restore).
function msgFromStored(m: ConversationMessage): Msg {
  if (m.role === 'user') return { id: newId(), role: 'user', text: m.content }
  const answer: GroundedAnswer = {
    tenant_id: '', query: '', status: (m.status ?? 'grounded') as AnswerStatus, answer: m.content,
    citations: m.citations, confidence: 0, abstain_reason: null, faithfulness_score: m.faithfulness,
    unsupported_claims: [], review_id: null, trace_id: m.trace_id, model: null,
  }
  return { id: newId(), role: 'assistant', text: m.content, answer }
}

const EXAMPLES = [
  'What is the platform uptime target?',
  'Summarise the key obligations in this document.',
  'What are the P95 latency requirements?',
]

const PHASE_LABEL: Record<string, string> = { 'retrieving': 'Searching your documents…', 'generating': 'Thinking…', 'searching all knowledge bases': 'Searching all knowledge bases…' }

// A GROUNDED response whose text is actually a refusal ("I don't know", "the documents
// don't cover this") is not a confident answer — surface it honestly, not as a green pill.
const NON_ANSWER_RE = /^\s*(i (do ?n'?t|do not|cannot|can'?t|am unable to) (know|answer|find|help|provide|determine|locate)|i (do ?n'?t|do not) have (enough|sufficient|any|the)|(the )?(provided |retrieved )?(documents?|context|sources?|knowledge ?base|passages?|information)( provided)? (do(es)? not|do ?n'?t) (contain|mention|include|provide|cover|have|specify|address)|there (is|'s| is) no (relevant )?(information|mention|reference|detail)|no relevant (information|content|answer)|unable to)/i
function isNonAnswer(text: string): boolean {
  const t = text.trim()
  return t.length > 0 && t.length < 320 && NON_ANSWER_RE.test(t)
}

interface Msg {
  // Stable identity for THIS turn. Updates from an in-flight request address the message by id
  // rather than by "the last one": patching m.length-1 silently retargets whenever anything is
  // appended while a request is streaming, and the answer then lands under a later question.
  id: string
  role: 'user' | 'assistant'
  text: string
  phase?: string
  condensed?: string
  answer?: GroundedAnswer
  error?: string
  feedback?: 'up' | 'down'
  stopped?: boolean
  ms?: number // client-measured time to answer (diagnostics panel)
  pendingFollowups?: boolean // answer shown; follow-ups still generating
  diagramLoading?: boolean // on-demand architecture diagram is being generated for this turn
}

function newId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
}

// What the citation viewer needs to open: which document, and which span to highlight.
export interface ViewerTarget { docId: string; chunkId: string; quote: string; page: number | null }

export function Ask() {
  const { session } = useAuth()
  const { t } = useI18n()
  const { kbId, kbs, setActiveModel } = useKb()
  const [params] = useSearchParams()
  const toast = useToast()
  const [q, setQ] = useState('')
  const [busy, setBusy] = useState(false)
  const [messages, setMessages] = useState<Msg[]>([])
  // Index of the user message currently being edited-and-reasked (or null). Resending
  // truncates the transcript back to this point instead of appending after it, so an edit
  // to an earlier turn doesn't leave the superseded turns dangling (looked like a new/
  // branched chat — the resend just tacked on at the end, disconnected from the edit).
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [scope, setScope] = useState<'kb' | 'all'>('kb')
  const [recencyDays, setRecencyDays] = useState(0)  // 0 = any time (FR-D9)
  // Federated RAG: partner departments that have granted this org access (from received
  // grants), and which of them to federate this question across.
  const [partnerDepts, setPartnerDepts] = useState<string[]>([])
  const [federateFrom, setFederateFrom] = useState<string[]>([])
  const [viewer, setViewer] = useState<ViewerTarget | null>(null)
  const convId = useRef<string | null>(null)  // active conversation thread (kb scope)
  const [answeringWith, setAnsweringWith] = useState<{ label: string; tenant: boolean } | null>(null)
  // Answer-details/sources side panel: collapsible so the answer area can go full-width. Remembered.
  const [panelOpen, setPanelOpen] = useState(() => localStorage.getItem('rag.askPanel') !== 'closed')
  const togglePanel = () => setPanelOpen((o) => { localStorage.setItem('rag.askPanel', o ? 'closed' : 'open'); return !o })
  useEffect(() => () => setActiveModel(null), [setActiveModel])   // clear when leaving Ask
  const [history, setHistory] = useState<ConversationInfo[] | null>(null)
  const [showHistory, setShowHistory] = useState(false)
  const scroller = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const abortRef = useRef<AbortController | null>(null)
  const kbName = kbs.find((k) => k.kb_id === kbId)?.name ?? kbId

  // Load a saved conversation into the transcript (persists across tab switches).
  const restoreConversation = useCallback(async (cid: string) => {
    if (!session) return
    setEditingIndex(null) // switching threads invalidates any in-progress edit
    try {
      const detail = await api.getConversation(session.token, kbId, cid)
      convId.current = cid
      localStorage.setItem(convKey(kbId), cid)
      setMessages(detail.messages.map(msgFromStored))
    } catch {
      convId.current = null
      localStorage.removeItem(convKey(kbId))
      setMessages([])
    }
  }, [session, kbId])

  const loadHistory = useCallback(async () => {
    // Always resolve to a list — an early return left history null, so the panel was stuck
    // on "Loading…" forever when no KB was selected (QA ISS-018).
    if (!session || !kbId) { setHistory([]); return }
    try { setHistory(await api.listConversations(session.token, kbId)) } catch { setHistory([]) }
  }, [session, kbId])

  // Federated RAG: which partner departments have granted us access (received grants).
  // Federation is opt-in and may be disabled (403) — treat any failure as "no partners".
  const loadPartners = useCallback(async () => {
    if (!session) return
    try {
      const grants = await api.listReceivedGrants(session.token)
      setPartnerDepts([...new Set(grants.map((g) => g.issuer_dept_id))].sort())
    } catch {
      setPartnerDepts([])
    }
  }, [session])
  useEffect(() => { void loadPartners() }, [loadPartners])

  // Show which model actually answers this KB, so there's no ambiguity between the
  // tenant's own model (FR-E1) and the platform default.
  useEffect(() => {
    if (!session || !kbId) { setAnsweringWith(null); return }
    let live = true
    ;(async () => {
      try {
        const conn = await api.getModelConnection(session.token, kbId)
        if (conn.configured && conn.active && conn.model) {
          if (live) setAnsweringWith({ label: conn.model, tenant: true })
          return
        }
      } catch { /* fall through to platform default */ }
      try {
        const sys = await api.systemInfo(session.token)
        if (live) setAnsweringWith({ label: sys.llm_model, tenant: false })
      } catch {
        if (live) setAnsweringWith(null)
      }
    })()
    return () => { live = false }
  }, [session, kbId])

  // A conversation belongs to one KB. On entering a KB (or returning to Ask), restore the
  // last active thread from storage so switching tabs never loses the chat.
  useEffect(() => {
    if (scope !== 'kb') { setMessages([]); return }
    const stored = localStorage.getItem(convKey(kbId))
    if (stored) void restoreConversation(stored)
    else { convId.current = null; setMessages([]) }
  }, [kbId, scope, restoreConversation])

  useEffect(() => { scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: 'smooth' }) }, [messages])

  useEffect(() => {
    const handed = params.get('q')
    if (handed && session) void run(handed)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  function patchById(id: string, patch: Partial<Msg>) {
    setMessages((m) => m.map((msg) => (msg.id === id ? { ...msg, ...patch } : msg)))
  }



  function newChat() {
    convId.current = null
    localStorage.removeItem(convKey(kbId))
    setMessages([])
    setShowHistory(false)
    setEditingIndex(null)
  }

  async function openConversation(cid: string) {
    setShowHistory(false)
    await restoreConversation(cid)
  }

  async function removeConversation(cid: string) {
    if (!session) return
    try {
      await api.deleteConversation(session.token, kbId, cid)
      setHistory((h) => (h ?? []).filter((c) => c.conversation_id !== cid))
      if (convId.current === cid) newChat()
      toast.push('info', 'Conversation deleted.')
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Delete failed')
    }
  }

  async function run(question: string, opts?: { appendUser?: boolean; truncateFrom?: number }) {
    const text = question.trim()
    if (!session || !text || busy) return
    const appendUser = opts?.appendUser !== false
    const truncateFrom = opts?.truncateFrom
    setQ('')
    setBusy(true)
    // Any real send (not a silent regenerate) resolves whatever edit was in flight.
    if (appendUser) setEditingIndex(null)
    const placeholderId = newId()
    const placeholder: Msg = { id: placeholderId, role: 'assistant', text: '', phase: scope === 'all' ? 'searching all knowledge bases' : 'retrieving' }
    setMessages((m) => {
      // Editing an earlier turn supersedes it and everything after it — drop that tail
      // instead of appending, so resend continues the thread instead of branching off it.
      const base = truncateFrom != null ? m.slice(0, truncateFrom) : m
      return appendUser ? [...base, { id: newId(), role: 'user', text } as Msg, placeholder] : [...base, placeholder]
    })
    const t0 = performance.now()
    try {
      if (scope === 'all') {
        // Federated ask has no per-KB thread — single-shot across knowledge bases. When
        // partner departments are selected (Federated RAG), also pull from their granted KBs.
        const a = federateFrom.length > 0
          ? await api.askFederated(session.token, text, federateFrom)
          : await api.askAcross(session.token, text, [])
        patchById(placeholderId, { text: a.answer, answer: a, phase: undefined, ms: Math.round(performance.now() - t0) })
      } else {
        // Conversational turn: create the thread lazily on the first message, then every
        // turn streams through it so follow-ups resolve against history (FR-I1).
        if (!convId.current) {
          const conv = await api.createConversation(session.token, kbId, text)
          convId.current = conv.conversation_id
          localStorage.setItem(convKey(kbId), conv.conversation_id)
        }
        const ac = new AbortController()
        abortRef.current = ac
        let acc = ''
        for await (const ev of streamConversationTurn(session.token, kbId, convId.current, text, undefined, recencyDays || undefined, ac.signal)) {
          if (ev.type === 'condensed') patchById(placeholderId, { condensed: ev.text })
          else if (ev.type === 'phase') patchById(placeholderId, { phase: ev.phase })
          else if (ev.type === 'token') { acc += ev.text; patchById(placeholderId, { text: acc, phase: undefined }) }
          else if (ev.type === 'done') patchById(placeholderId, { text: ev.answer.answer, answer: ev.answer, phase: undefined, ms: Math.round(performance.now() - t0), pendingFollowups: ev.answer.status === 'grounded' })
          // Follow-ups arrive AFTER the answer (extra model call) — patch them in when they land.
          else if (ev.type === 'followups') {
            setMessages((m) => m.map((msg) => (msg.id === placeholderId && msg.answer
              ? { ...msg, answer: { ...msg.answer, suggested_followups: ev.items }, pendingFollowups: false }
              : msg)))
          }
          else if (ev.type === 'error') patchById(placeholderId, { error: ev.message, phase: undefined })
        }
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        // User pressed Stop — keep whatever streamed so far, marked as stopped.
        patchById(placeholderId, { phase: undefined, stopped: true })
      } else {
        patchById(placeholderId, { error: err instanceof ApiError ? err.message : 'Something went wrong. Is the API running?', phase: undefined })
      }
      patchById(placeholderId, { pendingFollowups: false })
    } finally {
      abortRef.current = null
      setBusy(false)
      // The stream has ended, so no more events are coming: if a `followups` event never
      // arrived (grounded answer but the generator produced none, or follow-ups are disabled),
      // clear the pending flag so the "Preparing follow-up questions…" spinner never hangs.
      setMessages((m) => m.map((msg) => (msg.id === placeholderId && msg.pendingFollowups ? { ...msg, pendingFollowups: false } : msg)))
    }
  }

  function stop() {
    abortRef.current?.abort()
  }

  // Re-ask the last question as a fresh turn (drops the previous answer bubble).
  function regenerate() {
    if (busy) return
    let text = ''
    for (let i = messages.length - 1; i >= 0; i--) if (messages[i].role === 'user') { text = messages[i].text; break }
    if (!text) return
    setMessages((m) => (m.length && m[m.length - 1].role === 'assistant' ? m.slice(0, -1) : m))
    void run(text, { appendUser: false })
  }

  // Grow the composer with its content (up to a cap, then scroll) — with rows={1} and a
  // fixed height a wrapped question was clipped mid-line.
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 168)}px`
  }, [q])

  // Load a previously-sent question back into the composer to tweak and re-ask. Remembers
  // which message index this is so resend can truncate the superseded tail (see run()).
  function editQuestion(index: number, text: string) {
    setEditingIndex(index)
    setQ(text)
    const el = textareaRef.current
    if (el) { el.focus(); el.setSelectionRange(text.length, text.length) }
  }

  function cancelEdit() {
    setEditingIndex(null)
    setQ('')
  }

  async function copyAnswer(text: string) {
    // navigator.clipboard exists only in a secure context (HTTPS/localhost). Over plain HTTP
    // (browsing the console by IP) it is undefined, so fall back to the legacy execCommand path.
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text)
      } else {
        const ta = document.createElement('textarea')
        ta.value = text
        ta.style.position = 'fixed'
        ta.style.opacity = '0'
        document.body.appendChild(ta)
        ta.focus()
        ta.select()
        const ok = document.execCommand('copy')
        document.body.removeChild(ta)
        if (!ok) throw new Error('copy command rejected')
      }
      toast.push('success', 'Answer copied.')
    } catch {
      toast.push('error', 'Could not copy — your browser blocked clipboard access.')
    }
  }

  // On-demand architecture diagram (the "auto+confirm" chip): the question showed potential but
  // not strong intent, so we OFFER a diagram instead of auto-rendering. Clicking generates it and
  // attaches it in place (into this turn's structured), so the existing MermaidBlock renders it.
  async function generateDiagram(idx: number) {
    const m = messages[idx]
    if (!session || !m.answer || m.diagramLoading) return
    setMessages((ms) => ms.map((x, i) => (i === idx ? { ...x, diagramLoading: true } : x)))
    try {
      const arch = await api.generateArchitecture(session.token, kbId, m.answer.query, 'non-technical')
      setMessages((ms) => ms.map((x, i) => (i === idx && x.answer
        ? { ...x, diagramLoading: false, answer: { ...x.answer, diagram_available: false,
            structured: { ...(x.answer.structured ?? {}), mermaid: arch.mermaid, plantuml: arch.plantuml, architecture: arch.architecture } } }
        : x)))
    } catch (err) {
      setMessages((ms) => ms.map((x, i) => (i === idx ? { ...x, diagramLoading: false } : x)))
      toast.push('error', err instanceof ApiError ? err.message : 'Could not generate the diagram')
    }
  }

  async function sendFeedback(idx: number, rating: 'up' | 'down') {
    const m = messages[idx]
    if (!session || !m.answer?.trace_id) return
    // Toggle: clicking the active rating clears it; clicking the other switches (QA ISS-017).
    const next = m.feedback === rating ? undefined : rating
    setMessages((ms) => ms.map((x, i) => (i === idx ? { ...x, feedback: next } : x)))
    if (!next) return // cleared the rating — nothing to submit
    try {
      await api.submitFeedback(session.token, kbId, m.answer.trace_id, next)
      toast.push('success', next === 'up' ? 'Thanks — glad it helped.' : 'Thanks — logged for review.')
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Could not send feedback')
    }
  }

  // Pull the answer's signed trust receipt (by trace id) and verify its signature server-side.
  async function verifyReceiptFor(idx: number) {
    const tid = messages[idx]?.answer?.trace_id
    if (!session || !tid) return
    try {
      const receipt = await api.getReceipt(session.token, tid)
      const res = await api.verifyReceipt(session.token, receipt)
      toast.push(res.valid ? 'success' : 'error',
        res.valid ? 'Trust receipt verified — signature valid.' : 'Trust receipt INVALID — signature check failed.')
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'No trust receipt for this answer.')
    }
  }

  const empty = messages.length === 0

  return (
    <div className="flex flex-col" style={{ height: 'calc(100vh - 64px - 80px)' }}>
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <div>
          <div className="flex items-center gap-3 mb-1"><span className="w-9 rule-gold" /><span className="eyebrow">{t('page.ask.title')}</span></div>
          <h1 className="font-display font-semibold tracking-tight" style={{ fontSize: 24 }}>
            {t('page.ask.title')} <span style={{ color: 'var(--action)' }}>{scope === 'all' ? t('ask.title_all') : kbName}</span>
          </h1>
          {scope === 'kb' && answeringWith && (() => {
            // The org-wide default answers every question; there is no per-question pick to follow.
            const { label, tenant } = answeringWith
            return (
              <div className="flex items-center gap-1.5 mt-1.5 text-xs" style={{ color: 'var(--text-muted)' }}>
                <Icon.layers size={13} style={{ color: tenant ? 'var(--success)' : 'var(--text-muted)' }} />
                Answering with {tenant ? 'your model' : 'the platform default'}:
                <span className="mono font-semibold" style={{ color: 'var(--text-heading)' }}>{label}</span>
                {!tenant && (
                  <Link to="/models" className="underline" style={{ color: 'var(--action)' }}>connect your own →</Link>
                )}
              </div>
            )
          })()}
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {/* Do NOT call newChat() here — it deletes the persisted per-KB thread from
              localStorage. The restore effect above (keyed on [kbId, scope,
              restoreConversation]) already clears the visible transcript when entering
              'all' and restores the last thread from storage when returning to 'kb'. */}
          <SegmentedControl value={scope} onChange={setScope}
            options={[{ value: 'kb', label: kbName.length > 14 ? t('ask.scope_this') : kbName }, { value: 'all', label: t('ask.scope_all', { count: kbs.length }) }]} />
          {scope === 'all' && partnerDepts.length > 0 && (
            <div className="flex items-center gap-1.5 flex-wrap" title="Also pull from these partner departments — they granted you access (Federated RAG). Your own KBs are always included.">
              <Icon.shield size={14} style={{ color: 'var(--text-muted)' }} />
              <span className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>also ask:</span>
              {partnerDepts.map((dept) => {
                const on = federateFrom.includes(dept)
                return (
                  <button key={dept} type="button"
                    onClick={() => setFederateFrom((f) => on ? f.filter((d) => d !== dept) : [...f, dept])}
                    className="rounded-full text-xs font-semibold px-2.5 py-1 transition-colors"
                    style={{
                      background: on ? 'var(--action)' : 'var(--surface-2)',
                      color: on ? 'var(--action-contrast, #fff)' : 'var(--text-muted)',
                      border: `1px solid ${on ? 'var(--action)' : 'var(--border)'}`,
                    }}>
                    {on ? '✓ ' : '+ '}{dept}
                  </button>
                )
              })}
            </div>
          )}
          <select value={recencyDays} onChange={(e) => setRecencyDays(Number(e.target.value))} aria-label="Time filter"
            title="Only answer from documents within this window"
            className="rounded-md text-sm font-semibold px-3" style={{ minHeight: 36, background: 'var(--surface-2)', color: 'var(--text-heading)', border: '1px solid var(--border)', cursor: 'pointer' }}>
            <option value={0}>{t('ask.time.any')}</option>
            <option value={7}>{t('ask.time.7')}</option>
            <option value={30}>{t('ask.time.30')}</option>
            <option value={90}>{t('ask.time.90')}</option>
            <option value={365}>{t('ask.time.365')}</option>
          </select>
          {scope === 'kb' && (
            <div className="relative">
              <Button variant="subtle" size="sm" onClick={() => { const n = !showHistory; setShowHistory(n); if (n) void loadHistory() }} icon={<Icon.audit size={15} />}>{t('ask.history')}</Button>
              {showHistory && (
                <>
                  <div className="fixed inset-0" style={{ zIndex: 20 }} onClick={() => setShowHistory(false)} />
                  <div className="absolute right-0 mt-1.5 surface rounded-lg shadow-2xl overflow-hidden" style={{ zIndex: 21, width: 320, maxHeight: 420, border: '1px solid var(--border)' }}>
                    <div className="px-3 py-2 eyebrow" style={{ borderBottom: '1px solid var(--border)' }}>Recent chats · {kbName}</div>
                    <div className="overflow-y-auto" style={{ maxHeight: 372 }}>
                      {history === null && <div className="p-3 text-sm" style={{ color: 'var(--text-muted)' }}>Loading…</div>}
                      {history !== null && history.length === 0 && <div className="p-3 text-sm" style={{ color: 'var(--text-muted)' }}>No conversations yet.</div>}
                      {history?.map((c) => (
                        <div key={c.conversation_id} className="flex items-stretch" style={{ borderBottom: '1px solid var(--border)', background: c.conversation_id === convId.current ? 'color-mix(in srgb, var(--action) 8%, transparent)' : 'transparent' }}>
                          <button type="button" onClick={() => openConversation(c.conversation_id)}
                            className="flex-grow text-left px-3 py-2.5 interactive min-w-0" style={{ cursor: 'pointer', background: 'transparent', border: 'none' }}>
                            <div className="text-sm font-medium truncate" style={{ color: 'var(--text-heading)' }}>{c.title}</div>
                            <div className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>{new Date(c.updated_at).toLocaleString()}</div>
                          </button>
                          <button type="button" onClick={() => removeConversation(c.conversation_id)} aria-label="Delete conversation"
                            className="btn btn-subtle btn-icon shrink-0 mr-1 self-center" style={{ color: 'var(--danger)' }}><Icon.trash size={14} /></button>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
          {messages.length > 0 && <Button variant="subtle" size="sm" onClick={newChat} icon={<Icon.plus size={15} />}>{t('ask.new_chat')}</Button>}
        </div>
      </div>

      <div className="flex flex-grow gap-5" style={{ minHeight: 0 }}>
      <div className="flex flex-col flex-grow" style={{ minWidth: 0 }}>
      <div ref={scroller} className="flex-grow overflow-y-auto pr-1" style={{ scrollbarGutter: 'stable' }}>
        {empty && (
          <div className="h-full flex flex-col items-center justify-center text-center">
            <div className="mb-4" style={{ color: 'var(--text-muted)' }}><Icon.ask size={30} /></div>
            <h2 className="font-display font-semibold mb-1" style={{ fontSize: 20 }}>{t('ask.empty_title')}</h2>
            <p className="text-sm mb-6 max-w-md" style={{ color: 'var(--text-muted)' }}>{t('ask.empty_lead', { scope: scope === 'all' ? t('ask.empty_all') : kbName })} {scope === 'kb' && t('ask.empty_followup_hint')}</p>
            <div className="flex flex-wrap gap-2 justify-center max-w-lg">
              {EXAMPLES.map((ex) => <button key={ex} type="button" onClick={() => run(ex)} className="chip interactive" style={{ cursor: 'pointer' }}>{ex}</button>)}
            </div>
          </div>
        )}
        <div className="space-y-5 max-w-4xl mx-auto pb-4">
          {messages.map((m, i) => m.role === 'user'
            ? <UserBubble key={i} text={m.text} onEdit={busy ? undefined : () => editQuestion(i, m.text)} />
            : <AssistantBubble key={i} m={m} isLast={i === messages.length - 1} onCite={(t) => setViewer(t)}
                onFeedback={(r) => sendFeedback(i, r)} onAsk={(question) => run(question)}
                onDiagram={() => generateDiagram(i)}
                onCopy={() => copyAnswer(m.text)} onRegenerate={busy ? undefined : regenerate}
                onVerifyReceipt={m.answer?.trace_id ? () => verifyReceiptFor(i) : undefined} />)}
        </div>
      </div>

      <form onSubmit={(e) => { e.preventDefault(); run(q, editingIndex != null ? { truncateFrom: editingIndex } : undefined) }} className="pt-3 max-w-4xl mx-auto w-full">
        {messages.length > 0 && (
          <div className="flex justify-end items-center gap-2 mb-2">
            {editingIndex != null && (
              <span className="text-xs flex items-center gap-1.5" style={{ color: 'var(--text-muted)' }}>
                Editing an earlier message — resending will replace it and everything after it.
                <button type="button" onClick={cancelEdit} className="underline" style={{ cursor: 'pointer', background: 'none', border: 'none', color: 'inherit' }}>Cancel</button>
              </span>
            )}
            <Button variant="subtle" size="sm" onClick={newChat} icon={<Icon.plus size={14} />}>{t('ask.new_chat')}</Button>
          </div>
        )}
        <div className="surface flex items-end gap-2 p-2" style={{ borderRadius: 14 }}>
          <textarea ref={textareaRef} value={q} onChange={(e) => setQ(e.target.value)} rows={1} placeholder={scope === 'all' ? 'Ask across all your knowledge bases…' : (messages.length ? `Reply to ${kbName}…` : `Message ${kbName}…`)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); run(q, editingIndex != null ? { truncateFrom: editingIndex } : undefined) } }}
            className="flex-grow resize-none bg-transparent px-2 py-2 text-sm"
            style={{ outline: 'none', color: 'var(--text-primary)', maxHeight: 168, overflowY: 'auto', lineHeight: 1.5 }} />
          {busy && scope === 'kb'
            ? <button type="button" onClick={stop} className="btn btn-subtle btn-icon" aria-label="Stop generating" title="Stop"><Icon.stop size={16} /></button>
            : <button type="submit" disabled={!q.trim() || busy} className="btn btn-primary btn-icon" aria-label="Send"><Icon.ask size={18} /></button>}
        </div>
      </form>
      </div>

      {(() => {
        const last = [...messages].reverse().find((m) => m.role === 'assistant' && m.answer && m.answer.abstain_reason !== 'smalltalk')
        if (!last?.answer) return null
        return panelOpen ? (
          <aside className="hidden xl:block w-80 shrink-0 overflow-y-auto pb-2 relative" style={{ minHeight: 0 }}>
            <button type="button" onClick={togglePanel} className="btn btn-subtle btn-icon absolute z-10"
              style={{ top: 0, right: 0 }} aria-label="Hide details" title="Hide answer details & sources — widen the answer">
              <Icon.chevron size={16} />
            </button>
            <AnswerPanel answer={last.answer} ms={last.ms} onCite={(t) => setViewer(t)} />
          </aside>
        ) : (
          <div className="hidden xl:flex shrink-0 items-start" style={{ paddingTop: 2 }}>
            <button type="button" onClick={togglePanel} className="btn btn-subtle btn-icon"
              aria-label="Show details" title="Show answer details & sources">
              <Icon.chevron size={16} style={{ transform: 'rotate(180deg)' }} />
            </button>
          </div>
        )
      })()}
      </div>

      {viewer && session && (
        <CitationViewer token={session.token} kbId={kbId} target={viewer} onClose={() => setViewer(null)} />
      )}
    </div>
  )
}

function UserBubble({ text, onEdit }: { text: string; onEdit?: () => void }) {
  return (
    <div className="group flex justify-end items-center gap-1.5">
      {onEdit && (
        <button type="button" onClick={onEdit} aria-label="Edit and re-ask" title="Edit & re-ask"
          className="btn btn-subtle btn-icon opacity-0 group-hover:opacity-100 transition-opacity" style={{ color: 'var(--text-muted)' }}>
          <Icon.edit size={14} />
        </button>
      )}
      <div className="px-4 py-2.5 rounded-2xl rounded-br-sm max-w-[80%]" style={{ background: 'var(--action)', color: 'var(--on-action)' }}>{text}</div>
    </div>
  )
}

function AssistantBubble({ m, isLast, onCite, onFeedback, onAsk, onCopy, onRegenerate, onDiagram, onVerifyReceipt }: {
  m: Msg; isLast: boolean; onCite: (t: ViewerTarget) => void; onFeedback: (r: 'up' | 'down') => void;
  onAsk: (q: string) => void; onCopy: () => void; onRegenerate?: () => void; onDiagram?: () => void; onVerifyReceipt?: () => void
}) {
  const { t } = useI18n()
  const nonAnswer = m.answer?.status === 'grounded' && isNonAnswer(m.text)
  return (
    <div className="surface p-5" style={{ borderRadius: 14 }}>
      {m.error ? (
        <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--danger)' }}><Icon.alert size={16} />{m.error}</div>
      ) : m.phase ? (
        <div>
          <div className="flex items-center gap-2.5 text-sm" style={{ color: 'var(--text-muted)' }}>
            <span className="inline-block rounded-full animate-spin" style={{ width: 15, height: 15, border: '2px solid color-mix(in srgb, var(--action) 30%, transparent)', borderTopColor: 'var(--action)' }} />
            {PHASE_LABEL[m.phase] ?? m.phase}
          </div>
        </div>
      ) : (
        <>
          {/* A greeting gets a plain conversational reply — no status pill, no sources, no
              diagnostics: it was never a knowledge question. */}
          {m.answer?.abstain_reason === 'smalltalk' ? (
            <Markdown text={m.text} />
          ) : (
          <>
          {m.answer && (
            <div className="flex items-center justify-between gap-3 mb-3 flex-wrap">
              <div className="flex items-center gap-2 min-w-0">
                {nonAnswer
                  ? <span className="pill" title="The documents didn't answer this — nothing was fabricated" style={{ color: 'var(--text-muted)', borderColor: 'var(--border)', background: 'transparent' }}><Icon.minus size={12} /> No grounded answer</span>
                  : <AnswerPill status={m.answer.status} faithfulness={m.answer.faithfulness_score} />}
                {/* The machine-readable reason, next to the pill. The answer text already explains
                    the abstain in words; this is what makes a report actionable ("it said
                    faithfulness_below_threshold") without the reader having to guess which gate
                    fired. Smalltalk is not withheld information, so it carries no code. */}
                {m.answer.status === 'abstained' && m.answer.abstain_reason && m.answer.abstain_reason !== 'smalltalk' && (
                  <span className="mono truncate" title="Which gate withheld the answer" style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                    {m.answer.abstain_reason.replace(/_/g, ' ')}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-3">
                {m.answer.model && <span className="chip" title="Model that answered"><Icon.layers size={12} />{m.answer.model}</span>}
                {!nonAnswer && <VerificationStrip answer={m.answer} />}
              </div>
            </div>
          )}
          {m.answer && m.answer.citations.length > 0 && <Contributors citations={m.answer.citations} />}
          {m.answer?.agent_trace && m.answer.agent_trace.length > 0 && <AgentTrace steps={m.answer.agent_trace} />}
          {m.answer
            ? <><Markdown text={m.text} />{m.answer.structured?.mermaid && <MermaidBlock code={m.answer.structured.mermaid} />}{m.answer.structured?.diagram_note && <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>{m.answer.structured.diagram_note}</div>}</>
            : <div style={{ fontSize: 15.5 }}><Markdown text={m.text} />{!m.stopped && <Cursor show />}</div>}
          {/* Auto+confirm: the question showed POTENTIAL architecture intent (not strong enough to
              auto-draw), so OFFER a diagram on demand rather than presuming. Clicking attaches it
              in place, then the MermaidBlock above renders it. Hidden once a diagram is present. */}
          {m.answer && !m.answer.structured?.mermaid && (m.answer.diagram_available || m.diagramLoading) && onDiagram && (
            <div className="mt-3">
              <button type="button" onClick={onDiagram} disabled={m.diagramLoading} className="chip interactive"
                style={{ cursor: m.diagramLoading ? 'default' : 'pointer' }}
                title="Generate an architecture diagram grounded in this answer's sources">
                {m.diagramLoading
                  ? <><span className="inline-block rounded-full animate-spin" style={{ width: 11, height: 11, border: '2px solid color-mix(in srgb, var(--action) 25%, transparent)', borderTopColor: 'var(--action)' }} /> Diagramming…</>
                  : <><Icon.connectors size={12} style={{ opacity: 0.8 }} /> Generate architecture diagram</>}
              </button>
            </div>
          )}
          {/* Non-authoritative completeness signal (Slice 1): a small, non-alarming note when the
              guard could not fully complete the answer (truncated or missing a relevant point). */}
          {m.answer?.completeness && m.answer.completeness.complete === false && (
            <div className="mt-2 flex items-start gap-1.5 text-xs" style={{ color: 'var(--text-muted)' }}>
              <Icon.info size={13} className="mt-0.5 shrink-0" style={{ opacity: 0.7 }} />
              <span>This answer may be incomplete{m.answer.completeness.reason ? ` — ${m.answer.completeness.reason}` : ''}.</span>
            </div>
          )}
          {m.stopped && !m.answer && <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>Stopped.</p>}
          {m.answer?.status === 'flagged_for_review' && (
            <div className="mt-3 flex items-start gap-2 rounded-lg border p-2.5 text-sm" style={{ color: 'var(--warning)', borderColor: 'color-mix(in srgb, var(--warning) 35%, transparent)', background: 'color-mix(in srgb, var(--warning) 10%, transparent)' }}>
              <Icon.flag size={15} className="mt-0.5 shrink-0" />
              <span><strong>Held for human review</strong> — not shown as final. <Link className="underline font-semibold" to="/reviews">Open review queue →</Link></span>
            </div>
          )}
          {m.answer && m.answer.citations.length > 0 && <Sources citations={m.answer.citations} onCite={onCite} />}
          {m.pendingFollowups && (
            <div className="mt-3 flex items-center gap-2 text-xs" style={{ color: 'var(--text-muted)' }}>
              <span className="inline-block rounded-full animate-spin" style={{ width: 11, height: 11, border: '2px solid color-mix(in srgb, var(--action) 25%, transparent)', borderTopColor: 'var(--action)' }} />
              Preparing follow-up questions…
            </div>
          )}
          {m.answer && m.answer.suggested_followups && m.answer.suggested_followups.length > 0 && (
            <div className="mt-4 pt-4" style={{ borderTop: '1px solid var(--border)' }}>
              <div className="eyebrow mb-2">{t('ask.followups')}</div>
              <div className="flex flex-wrap gap-2">
                {m.answer.suggested_followups.map((q, i) => (
                  <button key={i} type="button" onClick={() => onAsk(q)} className="chip interactive" style={{ cursor: 'pointer' }} title="Ask this follow-up">
                    <Icon.chevron size={12} style={{ opacity: 0.7 }} /> {q}
                  </button>
                ))}
              </div>
            </div>
          )}
          {(m.answer || m.stopped) && m.text && (
            <div className="mt-4 flex items-center gap-2 flex-wrap">
              <button type="button" onClick={onCopy} className="btn btn-subtle btn-icon" aria-label="Copy answer" title="Copy answer" style={{ color: 'var(--text-muted)' }}><Icon.copy size={15} /></button>
              {isLast && onRegenerate && (
                <button type="button" onClick={onRegenerate} className="btn btn-subtle btn-icon" aria-label="Ask again" title="Ask again" style={{ color: 'var(--text-muted)' }}><Icon.refresh size={15} /></button>
              )}
              {m.answer && m.answer.trace_id && (m.answer.status === 'grounded' || m.answer.status === 'flagged_for_review') && (
                <>
                  <span className="text-xs ml-1" style={{ color: 'var(--text-muted)' }}>{t('ask.helpful')}</span>
                  <FeedbackButton dir="up" active={m.feedback === 'up'} onClick={() => onFeedback('up')} />
                  <FeedbackButton dir="down" active={m.feedback === 'down'} onClick={() => onFeedback('down')} />
                  {onVerifyReceipt && (
                    <button type="button" onClick={onVerifyReceipt} className="btn btn-subtle" title="Verify this answer's signed trust receipt" style={{ fontSize: 12, marginLeft: 4 }}>
                      <Icon.shield size={13} style={{ marginRight: 4, verticalAlign: '-2px' }} />Verify receipt
                    </button>
                  )}
                </>
              )}
            </div>
          )}
          </>
          )}
        </>
      )}
    </div>
  )
}

function citeTarget(c: Citation): ViewerTarget {
  return { docId: c.doc_id, chunkId: c.chunk_id, quote: c.quote, page: c.page }
}

// Human-friendly document label: drop the extension + the ingest-time numeric suffix.
function prettyDoc(docId: string): string {
  return docId.replace(/\.[a-z0-9]+$/i, '').replace(/_\d{6,}$/, '').replace(/_/g, ' ').trim() || docId
}

// Compact, collapsed-by-default source list: one "N sources · Doc A, Doc B +k" line that
// expands to the full cited passages. Keeps every citation clickable (grounded + auditable).
// Federated RAG attribution: when an answer draws on more than one source group — the caller's
// own KBs plus partner departments that granted access — the doc_ids are prefixed (e.g.
// "hr/leave"). Surface the distinct contributors so a federated answer visibly names which
// departments it used. Hidden for a single-source answer (no prefix / one group).
function Contributors({ citations }: { citations: Citation[] }) {
  const groups = [...new Set(citations.map((c) => (c.doc_id.includes('/') ? c.doc_id.split('/')[0] : null)).filter((x): x is string => !!x))]
  if (groups.length < 2) return null
  return (
    <div className="mb-3 flex items-center gap-1.5 flex-wrap text-xs" style={{ color: 'var(--text-muted)' }}>
      <Icon.shield size={13} />
      <span className="font-semibold" style={{ color: 'var(--text-heading)' }}>Answered using</span>
      {groups.map((g) => (
        <span key={g} className="rounded-full px-2 py-0.5 font-semibold" style={{ background: 'var(--surface-2)', border: '1px solid var(--border)', textTransform: 'uppercase', letterSpacing: '.02em' }}>{g}</span>
      ))}
    </div>
  )
}

function Sources({ citations, onCite }: { citations: Citation[]; onCite: (t: ViewerTarget) => void }) {
  const [open, setOpen] = useState(false)
  const docs = [...new Set(citations.map((c) => prettyDoc(c.doc_id)))]
  return (
    <div className="mt-4 pt-3" style={{ borderTop: '1px solid var(--border)' }}>
      <button type="button" onClick={() => setOpen(!open)} aria-expanded={open}
        className="inline-flex items-center gap-2 text-xs max-w-full" style={{ color: 'var(--text-muted)', cursor: 'pointer', background: 'none', border: 'none', padding: 0 }}>
        <Icon.book size={13} />
        <span className="font-semibold shrink-0" style={{ color: 'var(--text-heading)' }}>{citations.length} source{citations.length === 1 ? '' : 's'}</span>
        {!open && (
          <span className="truncate" style={{ opacity: 0.85 }}>· {docs.slice(0, 2).join(', ')}{docs.length > 2 ? ` +${docs.length - 2}` : ''}</span>
        )}
        <Icon.chevron size={12} className="shrink-0" style={{ transform: open ? 'rotate(90deg)' : 'none', transition: 'transform .15s' }} />
      </button>
      {open && (
        <ol className="space-y-2.5 mt-3">
          {citations.map((c, i) => (
            <li key={c.chunk_id}>
              <button type="button" onClick={() => onCite(citeTarget(c))} className="w-full text-left pl-3 interactive rounded-r-md" style={{ borderLeft: '2px solid var(--accent)', cursor: 'pointer' }} aria-label={`Open source ${i + 1} in ${c.doc_id}`}>
                <div className="mono text-xs mb-0.5 flex items-center gap-1.5" style={{ color: 'var(--text-muted)' }}><span className="font-semibold" style={{ color: 'var(--success)' }}>[{i + 1}]</span> {prettyDoc(c.doc_id)}{c.page != null ? ` · p.${c.page}` : ''} {c.modality === 'visual'
                  ? <span className="inline-flex items-center gap-1 rounded px-1.5 font-semibold" title="Source is an image (multimodal)" style={{ background: 'color-mix(in srgb, var(--action) 14%, transparent)', color: 'var(--action)' }}><Icon.eye size={11} /> visual</span>
                  : <Icon.book size={12} style={{ opacity: 0.6 }} />}</div>
                <p className="text-sm italic" style={{ color: 'var(--text-muted)' }}>"{c.quote}"</p>
              </button>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}

// FR-F: a collapsible, plain-language record of the steps the agentic pipeline took —
// plan / decompose / retrieve / retry / graph / verify — so the answer is explainable.
function AgentTrace({ steps }: { steps: AgentStep[] }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="mb-3">
      <button type="button" onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-muted)', cursor: 'pointer', background: 'none', border: 'none', padding: 0 }}>
        <Icon.activity size={13} />
        <span className="font-semibold">How this was answered</span>
        <span className="mono" style={{ opacity: 0.8 }}>· {steps.length} step{steps.length === 1 ? '' : 's'}</span>
        <Icon.chevron size={12} style={{ transform: open ? 'rotate(90deg)' : 'none', transition: 'transform .15s' }} />
      </button>
      {open && (
        <ol className="mt-2 space-y-1.5 ml-1 pl-3" style={{ borderLeft: '1px solid var(--border)' }}>
          {steps.map((s, i) => (
            <li key={i} className="flex items-start gap-2.5 text-xs">
              <span className="mono font-semibold shrink-0" style={{ color: 'var(--action)', width: 66 }}>{s.step}</span>
              <span style={{ color: 'var(--text-primary)' }}>{s.detail}</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}

function FeedbackButton({ dir, active, onClick }: { dir: 'up' | 'down'; active: boolean; onClick: () => void }) {
  const tone = dir === 'up' ? 'var(--success)' : 'var(--danger)'
  return (
    <button type="button" onClick={onClick} aria-pressed={active}
      aria-label={dir === 'up' ? 'Helpful' : 'Not helpful'} title={active ? 'Click to undo' : dir === 'up' ? 'Helpful' : 'Not helpful'}
      className="btn btn-subtle btn-icon" style={{ color: active ? tone : 'var(--text-muted)', cursor: 'pointer' }}>
      {dir === 'up' ? <Icon.thumbUp size={15} /> : <Icon.thumbDown size={15} />}
    </button>
  )
}

function Cursor({ show }: { show: boolean }) {
  if (!show) return null
  return <span className="inline-block align-middle animate-pulse" style={{ width: 8, height: 16, marginLeft: 2, background: 'var(--action)', borderRadius: 2 }} />
}

// Right-rail diagnostics for the latest answer — surfaces what the pipeline actually did
// (retrieval breadth, confidence, faithfulness, model, latency) so trust is inspectable.
function AnswerPanel({ answer, ms, onCite }: { answer: GroundedAnswer; ms?: number; onCite: (t: ViewerTarget) => void }) {
  const docs = [...new Set(answer.citations.map((c) => c.doc_id))]
  const pct = (v: number | null | undefined) => (v == null ? '—' : `${Math.round(v * 100)}%`)
  const row = (label: string, value: React.ReactNode) => (
    <div className="flex items-center justify-between py-1.5 text-xs" style={{ borderBottom: '1px solid var(--border)' }}>
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span className="font-semibold mono" style={{ color: 'var(--text-heading)', maxWidth: 170, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{value}</span>
    </div>
  )
  return (
    <div className="space-y-4">
      <div className="surface rounded-xl p-4">
        <div className="eyebrow mb-2">Answer diagnostics</div>
        {row('Status', <AnswerPill status={answer.status} />)}
        {row('Documents used', String(docs.length))}
        {row('Cited passages', String(answer.citations.length))}
        {answer.confidence > 0 && row('Retrieval confidence', pct(answer.confidence))}
        {row(
          'Verification',
          <span className="flex items-center gap-2 flex-wrap">
            <VerificationScore display={describeVerification(answer)} />
            <MethodBreakdown display={describeVerification(answer)} />
            <DegradedNote display={describeVerification(answer)} />
          </span>,
        )}
        {answer.model && row('Model', answer.model)}
        {ms != null && row('Answered in', ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms} ms`)}
        {answer.trace_id && row('Trace', answer.trace_id.slice(0, 12) + '…')}
        {answer.agent_trace && answer.agent_trace.length > 0 && (
          <div className="mt-3">
            <div className="eyebrow mb-2" style={{ fontSize: '0.62rem' }}>How this was answered · {answer.agent_trace.length} steps</div>
            <ol className="space-y-1.5 ml-1 pl-3" style={{ borderLeft: '1px solid var(--border)' }}>
              {answer.agent_trace.map((s, i) => (
                <li key={i} className="flex items-start gap-2.5 text-xs">
                  <span className="mono font-semibold shrink-0" style={{ color: 'var(--action)', width: 62 }}>{s.step}</span>
                  <span style={{ color: 'var(--text-primary)' }}>{s.detail}</span>
                </li>
              ))}
            </ol>
          </div>
        )}
        <p className="field-help mt-2" style={{ marginBottom: 0 }}>Every value is recorded on the tamper-evident audit trail.</p>
      </div>
      {answer.citations.length > 0 && (
        <div className="surface rounded-xl p-4">
          <div className="eyebrow mb-2">Sources · {docs.length} document{docs.length === 1 ? '' : 's'}</div>
          <ul className="space-y-1">
            {answer.citations.map((c, i) => (
              <li key={c.chunk_id + i}>
                <button type="button"
                  onClick={() => onCite({ docId: c.doc_id, chunkId: c.chunk_id, quote: c.quote, page: c.page })}
                  className="w-full text-left rounded-md px-2 py-1.5 interactive"
                  style={{ cursor: 'pointer', background: 'transparent', border: 'none' }}>
                  <span className="mono text-xs" style={{ color: 'var(--success)' }}>[{i + 1}]</span>{' '}
                  <span className="text-xs font-medium" style={{ color: 'var(--text-heading)' }}>{prettyDoc(c.doc_id)}</span>
                  {c.page != null && <span className="text-xs" style={{ color: 'var(--text-muted)' }}> · p.{c.page}</span>}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
