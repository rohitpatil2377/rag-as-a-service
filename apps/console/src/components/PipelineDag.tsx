import { useEffect, useMemo, useState } from 'react'
import { Icon } from './icons'
import { MermaidBlock } from './Mermaid'
import { Alert, Button, Card, Field, Table } from './ui'
import { ApiError, api, type IngestDagNode, type IngestDagResponse } from '../lib/api'
import { useAuth } from '../lib/auth'
import { useKb } from '../lib/kb'

// Backend-traceability view: the ingestion pipeline as a legible vertical timeline of four real
// phases — the phases the audit chain actually measures — animated top-to-bottom as a chain
// reaction. Each phase shows its real duration (persisted in the audit payload, so it survives for
// any document, not just a fresh upload) and the tamper-evident audit event hash that proves it.

function fmtDur(ms: number | null | undefined): string {
  if (ms == null || ms <= 0) return ''
  if (ms < 1000) return `${ms} ms`
  return `${(ms / 1000).toFixed(2)} s`
}

// The pipeline's four AUDIT-BACKED measurement boundaries. `close` is the stage whose completion
// the payload timestamp marks — its duration_ms is that phase's real elapsed time.
const PHASES: { key: string; title: string; icon: keyof typeof Icon; desc: string; stages: { id: string; label: string }[]; close: string }[] = [
  {
    key: 'ingest', title: 'Ingest', icon: 'upload', close: 'ingest',
    desc: 'Raw bytes land in the object store and the document is registered in the catalog.',
    stages: [{ id: 'ingest', label: 'Ingest' }, { id: 'object', label: 'Object · MinIO' }],
  },
  {
    key: 'extract', title: 'Extract & Scan', icon: 'doc', close: 'sanitize',
    desc: 'Doc-intelligence classifies, OCRs and masks PII; the prompt-injection scan gates the text.',
    stages: [{ id: 'doc_intel', label: 'Doc-Intel' }, { id: 'sanitize', label: 'Sanitize' }],
  },
  {
    key: 'index', title: 'Index', icon: 'layers', close: 'index',
    desc: 'Split into chunks, contextualize, embed, then write the vector, sparse and catalog stores.',
    stages: [
      { id: 'chunk', label: 'Chunk' }, { id: 'contextualize', label: 'Contextualize' },
      { id: 'embed', label: 'Embed' }, { id: 'vector', label: 'Vector · Qdrant' },
      { id: 'sparse', label: 'Sparse · OpenSearch' }, { id: 'catalog', label: 'Catalog · Postgres' },
      { id: 'index', label: 'Index checkpoint' },
    ],
  },
  {
    key: 'graph', title: 'Graph', icon: 'connectors', close: 'graphrag',
    desc: 'GraphRAG extracts entities and edges so this document connects into the knowledge graph.',
    stages: [{ id: 'graphrag', label: 'GraphRAG' }],
  },
]

type PhaseState = { running: boolean; ran: boolean; skipped: boolean; duration: number | null; at: number | null; hash: string; chunks: string | number | null }

function readPhase(byId: Record<string, IngestDagNode>, phase: (typeof PHASES)[number]): PhaseState {
  const nodes = phase.stages.map((s) => byId[s.id]).filter(Boolean)
  const close = byId[phase.close]
  const running = nodes.some((n) => n.status === 'running')
  const ran = close?.status === 'ran' || (nodes.length > 0 && nodes.every((n) => n.status === 'ran'))
  const skipped = close?.status === 'skipped'
  return {
    running, ran, skipped,
    duration: close?.duration_ms ?? null,
    at: typeof close?.at_ms === 'number' ? close.at_ms : null,
    hash: close?.event_hash || nodes.find((n) => n.event_hash)?.event_hash || '',
    chunks: byId['index']?.metrics?.chunks ?? null,
  }
}

function PipelineTimeline({ dag }: { dag: IngestDagResponse['dag'] }) {
  const byId = useMemo(() => Object.fromEntries(dag.nodes.map((n) => [n.id, n])) as Record<string, IngestDagNode>, [dag])
  const phases = useMemo(() => PHASES.map((p) => ({ ...p, state: readPhase(byId, p) })), [byId])
  const doneCount = phases.filter((p) => p.state.ran || p.state.skipped).length

  // Reveal follows the REAL ingestion state, never a replay. A document that is actively ingesting
  // (a stage is 'running') animates its phases top-to-bottom as they genuinely complete — the parent
  // polls, doneCount grows, and the reveal keeps pace. A FINISHED document snaps straight to its
  // completed state with no animation, so revisiting the page does not re-run the flow.
  const live = dag.nodes.some((n) => n.status === 'running')
  const [revealed, setRevealed] = useState(doneCount)
  useEffect(() => { setRevealed(live ? 0 : doneCount) }, [dag.doc_id]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!live) { setRevealed(doneCount); return } // done → real complete, no timer replay
    if (revealed >= doneCount) return
    const real = phases[revealed]?.state.duration ?? 0
    const delay = Math.max(450, Math.min(real, 1800))
    const t = setTimeout(() => setRevealed((r) => Math.min(r + 1, doneCount)), delay)
    return () => clearTimeout(t)
  }, [revealed, doneCount, dag.doc_id, live]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div>
      {phases.map((p, i) => {
        const shown = i < revealed
        const active = i === revealed && revealed < doneCount
        const skipped = p.state.skipped && shown
        const dotBg = skipped ? 'rgba(148,163,184,0.25)' : shown ? '#16a34a' : active ? '#f59e0b' : 'rgba(148,163,184,0.18)'
        const IconC = Icon[p.icon] ?? Icon.activity
        const lineColor = i < revealed ? '#16a34a' : 'rgba(148,163,184,0.25)'
        return (
          <div key={p.key} className="flex gap-3" style={{ opacity: shown || active ? 1 : 0.55, transition: 'opacity .4s' }}>
            {/* rail: status dot + connector */}
            <div className="flex flex-col items-center" style={{ width: 34 }}>
              <span
                className={active ? 'animate-pulse' : ''}
                style={{
                  width: 34, height: 34, borderRadius: 999, background: dotBg, color: skipped ? 'var(--text-muted)' : '#fff',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                  boxShadow: shown || active ? '0 1px 5px rgba(0,0,0,.15)' : 'none', transition: 'background .4s',
                }}
              >
                {shown && !skipped ? <Icon.check size={17} /> : <IconC size={16} />}
              </span>
              {i < PHASES.length - 1 && <span style={{ flex: 1, width: 2, minHeight: 26, background: lineColor, transition: 'background .4s' }} />}
            </div>
            {/* body */}
            <div style={{ paddingBottom: 18, flex: 1 }}>
              <div className="flex items-baseline justify-between gap-3 flex-wrap">
                <span className="font-display font-semibold" style={{ fontSize: 15 }}>
                  {p.title}
                  {p.key === 'index' && p.state.chunks != null && (
                    <span style={{ color: 'var(--text-muted)', fontWeight: 500, fontSize: 13 }}> · {p.state.chunks} chunks</span>
                  )}
                </span>
                <span className="mono" style={{ fontSize: 13, fontWeight: 700, color: shown ? '#16a34a' : 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                  {skipped ? 'skipped' : active ? 'running…' : fmtDur(p.state.duration) || (shown ? 'done' : 'pending')}
                </span>
              </div>
              <p style={{ color: 'var(--text-muted)', fontSize: 12.5, margin: '2px 0 7px' }}>{p.desc}</p>
              <div className="flex items-center gap-1.5 flex-wrap">
                {p.stages.map((s) => {
                  const st = byId[s.id]?.status
                  const on = shown && st === 'ran'
                  return (
                    <span key={s.id} className="chip" style={{ fontSize: 11, padding: '2px 8px', background: on ? '#dcfce7' : undefined, color: on ? '#166534' : 'var(--text-muted)' }}>
                      {s.label}
                    </span>
                  )
                })}
                {shown && p.state.hash && (
                  <span className="chip mono" title="Tamper-evident audit event hash" style={{ fontSize: 10.5, padding: '2px 7px', opacity: 0.75 }}>
                    ⛓ {p.state.hash.slice(0, 10)}
                  </span>
                )}
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function TrustLine({ trust, elapsedMs, quarantined }: { trust: IngestDagResponse['trust']; elapsedMs: number | null; quarantined: boolean }) {
  const ok = trust.verified
  return (
    <div className="flex items-center gap-2 flex-wrap" style={{ fontSize: 13 }}>
      {/* Explicit dark text on the fixed light backgrounds so the chip is legible in BOTH themes. */}
      <span className="chip" title={ok
        ? 'Every stage shown as run is backed by a real audit event whose hash chain verifies un-tampered.'
        : 'The stages really ran, but the tenant audit hash chain did not fully verify (a gap or an edited / out-of-order event). The run is real; its cryptographic proof is incomplete.'}
        style={{ background: ok ? '#dcfce7' : '#fef3c7', color: ok ? '#166534' : '#92400e', fontWeight: 600 }}>
        {ok ? '✓ Audit-verified · chain intact' : '⚠ Not chain-verified'}
      </span>
      {quarantined && (
        <span className="chip" title="The prompt-injection scan flagged this document, so it was quarantined and NOT indexed — a safety block, not a pipeline error."
          style={{ background: '#fee2e2', color: '#991b1b', fontWeight: 600 }}>
          ⛔ Quarantined — injection suspected
        </span>
      )}
      {elapsedMs != null && elapsedMs > 0 && <span className="chip">⏱ {fmtDur(elapsedMs)} pipeline time</span>}
    </div>
  )
}

function RanStagesTable({ dag }: { dag: IngestDagResponse['dag'] }) {
  const ran = dag.nodes.filter((n) => n.status === 'ran')
  if (ran.length === 0) return null
  return (
    <Table>
      <thead><tr><th>Stage</th><th>Metrics</th><th>Duration</th><th>Audit event hash</th></tr></thead>
      <tbody>
        {ran.map((n) => (
          <tr key={n.id}>
            <td style={{ fontWeight: 600 }}>{n.label}</td>
            <td style={{ color: 'var(--text-muted)', fontSize: 12.5 }}>
              {Object.entries(n.metrics ?? {}).map(([k, v]) => `${k}: ${v}`).join(' · ') || '—'}
            </td>
            <td className="mono" style={{ fontSize: 12.5 }}>{fmtDur(n.duration_ms) || '—'}</td>
            <td className="mono" style={{ fontSize: 12 }}>{n.event_hash ? n.event_hash.slice(0, 16) : '—'}</td>
          </tr>
        ))}
      </tbody>
    </Table>
  )
}

// Fullscreen, zoomable DAG viewer — the inline diagram is unreadably small for a 13-node graph, so
// the full view takes over the screen and renders the SVG at natural size (scrollable + zoomable).
function DagModal({ code, onClose }: { code: string; onClose: () => void }) {
  const [zoom, setZoom] = useState(1.5)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div role="dialog" aria-modal="true" onClick={onClose}
      style={{ position: 'fixed', inset: 0, zIndex: 80, background: 'rgba(2,6,23,0.74)', display: 'flex', flexDirection: 'column' }}>
      <div className="flex items-center justify-between gap-3" style={{ padding: '12px 18px' }} onClick={(e) => e.stopPropagation()}>
        <span className="font-display font-semibold" style={{ color: '#fff', fontSize: 15 }}>Ingestion pipeline — full DAG</span>
        <div className="flex items-center gap-2">
          <Button onClick={() => setZoom((z) => Math.max(0.5, +(z - 0.25).toFixed(2)))}>−</Button>
          <span className="mono" style={{ color: '#fff', minWidth: 48, textAlign: 'center', fontSize: 13 }}>{Math.round(zoom * 100)}%</span>
          <Button onClick={() => setZoom((z) => Math.min(5, +(z + 0.25).toFixed(2)))}>+</Button>
          <Button onClick={onClose} icon={<Icon.x size={16} />}>Close</Button>
        </div>
      </div>
      {/* CSS `zoom` resizes layout (unlike transform) so the scroll area tracks the enlarged SVG;
          the SVG itself already renders at true pixel size via naturalSize. */}
      <div style={{ flex: 1, overflow: 'auto', margin: '0 14px 14px', background: 'var(--surface)', borderRadius: 10 }} onClick={(e) => e.stopPropagation()}>
        <div style={{ zoom, padding: 12 }}>
          <MermaidBlock code={code} naturalSize />
        </div>
      </div>
    </div>
  )
}

// `docId` (optional) lets the parent deep-link a document (e.g. the newest upload).
export function PipelineDag({ docId: initialDocId = '', active = false }: { docId?: string; active?: boolean }) {
  const { session } = useAuth()
  const { kbId } = useKb()
  const [template, setTemplate] = useState<IngestDagResponse | null>(null)
  const [docId, setDocId] = useState(initialDocId)
  const [trace, setTrace] = useState<IngestDagResponse | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dagOpen, setDagOpen] = useState(false)

  useEffect(() => {
    if (!session) return
    api.getIngestPipelineDag(session.token).then(setTemplate).catch(() => setTemplate(null))
  }, [session])

  // Auto-trace the document the parent hands us (e.g. the newest upload). CRITICAL: clear the prior
  // trace on every KB/doc switch — trace is per-(kb,doc), so without the reset a previous KB's trace
  // would render in another KB (a false cross-KB "leak").
  useEffect(() => {
    setDocId(initialDocId)
    setTrace(null)
    setError(null)
    if (initialDocId && session && kbId) void fetchTrace(initialDocId)
  }, [initialDocId, session, kbId]) // eslint-disable-line react-hooks/exhaustive-deps

  async function fetchTrace(id: string): Promise<IngestDagResponse | null> {
    if (!session || !kbId) return null
    try {
      const r = await api.getDocumentDag(session.token, kbId, id)
      setTrace(r)
      return r
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not load the document DAG.')
      return null
    }
  }

  async function traceDoc() {
    if (!session || !kbId || !docId.trim() || busy) return
    setBusy(true); setError(null); setTrace(null)
    await fetchTrace(docId.trim())
    setBusy(false)
  }

  // Live ONLY while a stage is actively 'running' (amber). A finished document has stages that
  // simply lack their own audit action — that is NOT "in progress" and must not poll forever.
  const inProgress = !!trace && trace.dag.nodes.some((n) => n.status === 'running')
  const ranCountEarly = (trace?.dag.nodes ?? []).filter((n) => n.status === 'ran').length
  const allDoneEarly = !!trace && !inProgress && ranCountEarly > 0
  // Poll while the ingest JOB is active (parent signal) OR a stage is 'running'. Gating only on
  // 'running' froze at all-pending, because a freshly-started async ingest has no marks yet.
  useEffect(() => {
    if (!docId.trim() || allDoneEarly) return
    if (!active && !inProgress) return
    const t = setTimeout(() => { void fetchTrace(docId.trim()) }, 1200)
    return () => clearTimeout(t)
  }, [trace, inProgress, active, allDoneEarly]) // eslint-disable-line react-hooks/exhaustive-deps

  const primary = trace ?? template
  const quarantined = useMemo(() => (trace?.dag.nodes ?? []).some((n) => n.id === 'quarantine' && n.status === 'ran'), [trace])
  const ranCount = useMemo(() => (trace?.dag.nodes ?? []).filter((n) => n.status === 'ran').length, [trace])
  const elapsedMs = useMemo(() => {
    const offs = (trace?.dag.nodes ?? []).map((n) => n.at_ms).filter((v): v is number => typeof v === 'number')
    return offs.length ? Math.max(...offs) : null
  }, [trace])
  const allDone = !!trace && !inProgress && ranCount > 0

  return (
    <div className="space-y-5">
      <Card>
        <div className="flex items-center justify-between flex-wrap gap-2 mb-1">
          <h3 className="font-display font-semibold" style={{ fontSize: 16 }}>
            Ingestion pipeline{trace?.dag.doc_id ? ` · ${trace.dag.doc_id}` : ''}
          </h3>
          {trace && (inProgress
            ? <span className="chip" style={{ background: '#fef9c3', color: '#713f12', fontWeight: 600 }}>● Ingesting…</span>
            : allDone && <span className="chip" style={{ background: '#dcfce7', color: '#166534', fontWeight: 600 }}>✓ Complete</span>)}
        </div>
        <p className="field-help mb-4">
          The four phases the backend runs after a document is ingested — each fills in as it completes,
          with the real time it took and the tamper-evident audit hash that proves it.
        </p>

        {trace
          ? <PipelineTimeline dag={trace.dag} />
          : <p style={{ color: 'var(--text-muted)' }}>Upload a document, or trace one below, to watch its pipeline run phase by phase.</p>}

        {trace && <div className="mt-3"><TrustLine trust={trace.trust} elapsedMs={elapsedMs} quarantined={quarantined} /></div>}

        {(primary || trace) && (
          <div className="mt-4 flex flex-col gap-2 items-start">
            {primary && (
              <Button onClick={() => setDagOpen(true)} icon={<Icon.layers size={15} />} style={{ fontSize: 13 }}>
                View full pipeline DAG
              </Button>
            )}
            {trace && (
              <details style={{ width: '100%' }}>
                <summary style={{ cursor: 'pointer', color: 'var(--text-muted)', fontSize: 13, fontWeight: 600 }}>Per-stage audit detail</summary>
                <div className="mt-3"><RanStagesTable dag={trace.dag} /></div>
              </details>
            )}
          </div>
        )}
        {dagOpen && primary && <DagModal code={primary.mermaid} onClose={() => setDagOpen(false)} />}
      </Card>

      <Card>
        <div className="grid sm:grid-cols-[1fr_auto] gap-4 items-end">
          <Field label="Trace another document" help="Auto-traces your newest upload; enter a different id to trace another document.">
            <input className="field-input" value={docId} onChange={(e) => setDocId(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') void traceDoc() }}
              placeholder="e.g. contract-2026-04" />
          </Field>
          <Button loading={busy} disabled={!docId.trim() || !kbId} onClick={traceDoc}
            icon={!busy && <Icon.activity size={16} />}>
            Trace
          </Button>
        </div>
        {error && <div className="mt-4"><Alert kind="error">{error}</Alert></div>}
      </Card>
    </div>
  )
}
