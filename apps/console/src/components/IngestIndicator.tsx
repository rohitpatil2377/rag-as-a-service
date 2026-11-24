import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useIngestJobs, type TrackedIngest } from '../lib/ingestJobs'
import { useKb } from '../lib/kb'
import { Icon } from './icons'

/** Topbar pill shown only while documents are ingesting (or just finished). Aggregate progress
 * across every KB, visible on every page; click for per-job detail + a jump back to the KB. */
export function IngestIndicator() {
  const { jobs, active, dismiss, cancel } = useIngestJobs()
  const { setKbId } = useKb()
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function onClick(e: MouseEvent) { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false) }
    if (open) document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [open])

  if (jobs.length === 0) return null

  const running = active.length > 0
  const totalDocs = active.reduce((s, j) => s + j.total, 0)
  const doneDocs = active.reduce((s, j) => s + (j.phase === 'staging' ? j.staged : j.done + j.failed), 0)
  const pct = totalDocs ? Math.round((doneDocs / totalDocs) * 100) : running ? 0 : 100
  const anyFail = jobs.some((j) => j.failed > 0)
  const label = running ? `Ingesting ${doneDocs}/${totalDocs || '…'}` : anyFail ? 'Ingest finished' : 'Ingest complete'

  const accent = anyFail ? 'var(--danger, #d64545)' : running ? 'var(--action)' : 'var(--success, #1f9d55)'

  return (
    <div ref={ref} className="relative">
      <button onClick={() => setOpen((o) => !o)} aria-label={label} title={label}
        className="flex items-center gap-2 rounded-md px-2.5"
        style={{ height: 30, border: '1px solid var(--border)', background: 'var(--surface-2)', cursor: 'pointer' }}>
        <Icon.refresh size={13} style={{ color: accent, animation: running ? 'spin 1s linear infinite' : 'none' }} />
        <span className="text-xs font-medium hidden sm:inline" style={{ color: 'var(--text-heading)' }}>{label}</span>
        <span aria-hidden style={{ width: 46, height: 5, borderRadius: 999, background: 'var(--border)', overflow: 'hidden' }}>
          <span style={{ display: 'block', width: `${pct}%`, height: '100%', background: accent, transition: 'width .3s ease' }} />
        </span>
      </button>

      {open && (
        <div style={{ position: 'absolute', right: 0, top: 38, width: 320, maxWidth: '90vw', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 12, boxShadow: '0 16px 48px rgba(0,0,0,.32)', zIndex: 2147482000, overflow: 'hidden' }}>
          <div className="flex items-center justify-between px-4 py-2.5" style={{ borderBottom: '1px solid var(--border)' }}>
            <span className="font-display font-semibold" style={{ fontSize: 14 }}>Document ingestion</span>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>runs in the background</span>
          </div>
          <ul style={{ maxHeight: 340, overflowY: 'auto' }}>
            {jobs.map((j) => (
              <JobRow key={j.jobId} job={j}
                onOpen={() => { setKbId(j.kbId); setOpen(false); navigate('/knowledge-base') }}
                onCancel={() => cancel(j.jobId)}
                onDismiss={() => dismiss(j.jobId)} />
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

function JobRow({ job, onOpen, onCancel, onDismiss }: { job: TrackedIngest; onOpen: () => void; onCancel: () => void; onDismiss: () => void }) {
  const running = job.phase === 'staging' || job.phase === 'processing'
  const seen = job.phase === 'staging' ? job.staged : job.done + job.failed
  const pct = job.total ? Math.round((seen / job.total) * 100) : running ? 0 : 100
  const color = job.failed > 0 ? 'var(--danger, #d64545)' : running ? 'var(--action)' : 'var(--success, #1f9d55)'
  const state = job.phase === 'staging' ? 'Uploading' : job.phase === 'processing' ? 'Ingesting'
    : job.phase === 'failed' ? 'Failed' : 'Complete'

  return (
    <li className="px-4 py-2.5" style={{ borderTop: '1px solid var(--border)' }}>
      <div className="flex items-center justify-between gap-2">
        <button onClick={onOpen} className="text-sm font-medium truncate text-left" style={{ color: 'var(--text-heading)', cursor: 'pointer', background: 'none', border: 'none', padding: 0 }} title={`Open ${job.kbName}`}>{job.kbName}</button>
        <span className="text-xs mono shrink-0" style={{ color: 'var(--text-muted)' }}>{state} {job.total ? `${seen}/${job.total}` : ''}</span>
      </div>
      <div className="mt-1.5" style={{ height: 6, borderRadius: 999, background: 'var(--border)', overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, transition: 'width .3s ease' }} />
      </div>
      <div className="flex items-center gap-2 mt-1.5">
        {job.done > 0 && <span className="text-xs" style={{ color: 'var(--success, #1f9d55)' }}>{job.done} done</span>}
        {job.failed > 0 && <span className="text-xs" style={{ color: 'var(--danger, #d64545)' }}>{job.failed} failed</span>}
        <div className="flex-grow" />
        {job.phase === 'processing' && <button onClick={onCancel} className="text-xs" style={{ color: 'var(--danger, #d64545)', cursor: 'pointer', background: 'none', border: 'none' }}>Cancel</button>}
        {!running && <button onClick={onDismiss} className="text-xs" style={{ color: 'var(--text-muted)', cursor: 'pointer', background: 'none', border: 'none' }}>Dismiss</button>}
      </div>
    </li>
  )
}
