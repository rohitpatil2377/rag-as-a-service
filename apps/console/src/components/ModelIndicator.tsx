import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type ModelStatus } from '../lib/api'
import { useAuth } from '../lib/auth'
import { useKb } from '../lib/kb'

/** Topbar indicator: which model answers this KB (connected vs platform default) and whether
 * that endpoint is reachable right now. Polls the live /model-status probe. Click to re-check. */
export function ModelIndicator() {
  const { session } = useAuth()
  const { kbId, activeModel } = useKb()
  const [status, setStatus] = useState<ModelStatus | null>(null)
  const [checking, setChecking] = useState(false)
  const busy = useRef(false)

  const load = useCallback(async (refresh = false) => {
    if (!session || !kbId || busy.current) return
    busy.current = true
    if (refresh) setChecking(true)
    try { setStatus(await api.modelStatus(session.token, kbId, refresh)) } catch { setStatus(null) }
    finally { busy.current = false; setChecking(false) }
  }, [session, kbId])

  // Refresh on KB change, then poll (reachability can flap; keep it fresh but gentle).
  useEffect(() => { void load() }, [load])
  useEffect(() => {
    const t = setInterval(() => void load(), 45000)
    return () => clearInterval(t)
  }, [load])

  if (!kbId || !status) return null

  const online = status.online
  const dot = online === true ? 'var(--success, #1f9d55)' : online === false ? 'var(--danger, #d64545)' : 'var(--text-muted)'
  const connected = status.source === 'connected'
  // A per-question pick on the Ask page overrides the label (topbar mirrors the dropdown); the
  // reachability dot still reflects the KB's default endpoint probe.
  const picked = activeModel
  const label = picked ? picked.label : (connected ? status.name : 'Default')
  const own = Boolean(picked) || connected
  const stateWord = online === true ? 'online' : online === false ? 'offline' : 'unknown'
  const tip = [
    `${status.name} (${status.model})`,
    `${connected ? 'Connected model' : 'Platform default'} — ${stateWord}`,
    status.endpoint ? `endpoint: ${status.endpoint}` : 'internal endpoint',
    status.latency_ms != null ? `probe: ${status.latency_ms} ms` : '',
    status.detail,
    'Click to re-check',
  ].filter(Boolean).join('\n')

  return (
    <button onClick={() => void load(true)} title={tip} aria-label={`Model ${label}, ${stateWord}. Click to re-check.`}
      className="hidden md:flex items-center gap-1.5 rounded-md px-2.5"
      style={{ height: 30, border: '1px solid var(--border)', background: 'var(--surface-2)', cursor: 'pointer', maxWidth: 190 }}>
      <span aria-hidden style={{
        width: 8, height: 8, borderRadius: 999, background: dot, flexShrink: 0,
        boxShadow: online === true ? `0 0 0 3px color-mix(in srgb, ${dot} 22%, transparent)` : 'none',
        opacity: checking ? 0.4 : 1, transition: 'opacity .15s ease',
      }} />
      <span className="text-xs font-medium truncate" style={{ color: 'var(--text-heading)' }}>{label}</span>
      <span className="text-xs mono uppercase shrink-0" style={{
        fontSize: 9, letterSpacing: '.04em', padding: '1px 5px', borderRadius: 4,
        color: own ? 'var(--action)' : 'var(--text-muted)',
        background: own ? 'color-mix(in srgb, var(--action) 12%, transparent)' : 'var(--surface)',
        border: '1px solid var(--border)',
      }}>{picked ? 'pick' : (connected ? 'conn' : 'def')}</span>
    </button>
  )
}
