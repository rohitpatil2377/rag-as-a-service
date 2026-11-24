import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import { useAuth } from '../lib/auth'
import { useKb } from '../lib/kb'
import { Icon } from './icons'

type Activity = { action: string; label: string; detail: string; at: string }

const ICON: Record<string, keyof typeof Icon> = { ingest: 'connectors', index: 'database', eval_run: 'activity', erasure: 'scale' }

/** Topbar job-centre — recent long-running operations (ingest/index/eval), polled. */
export function JobCentre() {
  const { session } = useAuth()
  const { kbId } = useKb()
  const [open, setOpen] = useState(false)
  const [items, setItems] = useState<Activity[] | null>(null)
  const ref = useRef<HTMLDivElement>(null)

  const load = useCallback(async () => {
    if (!session || !kbId) return
    try { setItems(await api.activity(session.token, kbId)) } catch { setItems([]) }
  }, [session, kbId])

  // Poll while open; refresh once on mount so the badge reflects recent work.
  useEffect(() => { void load() }, [load])
  useEffect(() => {
    if (!open) return
    void load()
    const t = setInterval(() => void load(), 8000)
    return () => clearInterval(t)
  }, [open, load])

  // Close on outside click.
  useEffect(() => {
    function onClick(e: MouseEvent) { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false) }
    if (open) document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [open])

  const recent = (items ?? []).filter((a) => Date.now() - new Date(a.at).getTime() < 5 * 60_000).length

  return (
    <div ref={ref} className="relative">
      <button onClick={() => setOpen((o) => !o)} aria-label="Activity" title="Recent activity"
        className="flex items-center justify-center rounded-md" style={{ width: 34, height: 30, color: 'var(--text-muted)', border: '1px solid var(--border)', background: 'var(--surface-2)', cursor: 'pointer', position: 'relative' }}>
        <Icon.activity size={16} />
        {recent > 0 && <span style={{ position: 'absolute', top: -4, right: -4, minWidth: 15, height: 15, padding: '0 3px', borderRadius: 8, background: 'var(--accent)', color: '#fff', fontSize: 9, lineHeight: '15px', textAlign: 'center', fontWeight: 700 }}>{recent}</span>}
      </button>
      {open && (
        <div style={{ position: 'absolute', right: 0, top: 38, width: 340, maxWidth: '90vw', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 12, boxShadow: '0 16px 48px rgba(0,0,0,.32)', zIndex: 2147482000, overflow: 'hidden' }}>
          <div className="flex items-center justify-between px-4 py-2.5" style={{ borderBottom: '1px solid var(--border)' }}>
            <span className="font-display font-semibold" style={{ fontSize: 14 }}>Activity</span>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>ingest · index · eval</span>
          </div>
          <ul style={{ maxHeight: 340, overflowY: 'auto' }}>
            {items === null && <li className="px-4 py-3 text-sm" style={{ color: 'var(--text-muted)' }}>Loading…</li>}
            {items && items.length === 0 && <li className="px-4 py-3 text-sm" style={{ color: 'var(--text-muted)' }}>No recent activity.</li>}
            {(items ?? []).map((a, i) => {
              const Ico = Icon[ICON[a.action] ?? 'activity']
              return (
                <li key={i} className="flex items-start gap-2.5 px-4 py-2.5" style={{ borderTop: i ? '1px solid var(--border)' : 'none' }}>
                  <span style={{ marginTop: 1, color: 'var(--accent)' }}><Ico size={15} /></span>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium" style={{ color: 'var(--text-heading)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{a.label}</div>
                    <div className="text-xs" style={{ color: 'var(--text-muted)' }}>{a.action}{a.detail ? ` · ${a.detail}` : ''} · {new Date(a.at).toLocaleTimeString()}</div>
                  </div>
                </li>
              )
            })}
          </ul>
        </div>
      )}
    </div>
  )
}
