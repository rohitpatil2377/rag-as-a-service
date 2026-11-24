import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { NAV, canSee } from './Layout'
import { Icon } from './icons'
import { useI18n } from '../lib/i18n'
import { useAuth } from '../lib/auth'

interface Command {
  label: string
  hint: string
  run: (nav: ReturnType<typeof useNavigate>) => void
  icon?: React.ComponentType<{ size?: number }>
}

/** Cmd/Ctrl-K command palette — keyboard-first navigation + quick actions. */
export function CommandPalette() {
  const navigate = useNavigate()
  const { t } = useI18n()
  const { session } = useAuth()
  const [open, setOpen] = useState(false)
  const [q, setQ] = useState('')
  const [active, setActive] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const roles = session?.roles ?? []

  const commands = useMemo<Command[]>(() => {
    const pages: Command[] = NAV.filter((n) => canSee(n, roles)).map((n) => {
      const label = t(n.tkey, n.label)
      return { label, hint: `Go to ${label}`, icon: n.icon, run: (nav) => nav(n.to) }
    })
    const actions: Command[] = [
      { label: 'Ask a question', hint: 'Action', icon: Icon.ask, run: (nav) => nav('/ask') },
      { label: 'New knowledge base', hint: 'Action', icon: Icon.database, run: (nav) => nav('/knowledge-base') },
      { label: 'Crawl a website', hint: 'Action', icon: Icon.connectors, run: (nav) => nav('/connectors') },
      { label: 'Run an evaluation', hint: 'Action', icon: Icon.activity, run: (nav) => nav('/quality') },
    ]
    return [...pages, ...actions]
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [t, roles.join(',')])

  const filtered = useMemo(() => {
    const t = q.trim().toLowerCase()
    if (!t) return commands
    return commands.filter((c) => c.label.toLowerCase().includes(t) || c.hint.toLowerCase().includes(t))
  }, [q, commands])

  // Global open/close shortcut.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setOpen((o) => !o)
      } else if (e.key === 'Escape') {
        setOpen(false)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  useEffect(() => {
    if (open) { setQ(''); setActive(0); setTimeout(() => inputRef.current?.focus(), 0) }
  }, [open])

  useEffect(() => { setActive(0) }, [q])

  if (!open) return null

  function choose(i: number) {
    const cmd = filtered[i]
    if (!cmd) return
    setOpen(false)
    cmd.run(navigate)
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'ArrowDown') { e.preventDefault(); setActive((a) => Math.min(a + 1, filtered.length - 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive((a) => Math.max(a - 1, 0)) }
    else if (e.key === 'Enter') { e.preventDefault(); choose(active) }
  }

  return (
    <div role="dialog" aria-label="Command palette" onClick={() => setOpen(false)}
      style={{ position: 'fixed', inset: 0, zIndex: 2147483000, background: 'color-mix(in srgb, #0a1020 55%, transparent)', display: 'flex', alignItems: 'flex-start', justifyContent: 'center', paddingTop: '14vh' }}>
      <div onClick={(e) => e.stopPropagation()} onKeyDown={onKeyDown}
        style={{ width: 560, maxWidth: 'calc(100vw - 32px)', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 14, boxShadow: '0 24px 70px rgba(0,0,0,.4)', overflow: 'hidden' }}>
        <div className="flex items-center gap-2 px-4" style={{ borderBottom: '1px solid var(--border)' }}>
          <Icon.ask size={16} style={{ color: 'var(--text-muted)' }} />
          <input ref={inputRef} value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search pages and actions…"
            aria-label="Search commands"
            style={{ flex: 1, background: 'transparent', border: 'none', outline: 'none', padding: '14px 0', fontSize: 15, color: 'var(--text-heading)' }} />
          <kbd className="mono" style={{ fontSize: 11, color: 'var(--text-muted)', border: '1px solid var(--border)', borderRadius: 5, padding: '2px 6px' }}>esc</kbd>
        </div>
        <ul style={{ maxHeight: 360, overflowY: 'auto', padding: 6 }}>
          {filtered.length === 0 && <li className="px-3 py-4 text-sm" style={{ color: 'var(--text-muted)' }}>No matches.</li>}
          {filtered.map((c, i) => {
            const Ico = c.icon
            return (
              <li key={c.label} onMouseEnter={() => setActive(i)} onClick={() => choose(i)}
                className="flex items-center gap-3 rounded-lg cursor-pointer"
                style={{ padding: '9px 12px', background: i === active ? 'color-mix(in srgb, var(--action) 13%, transparent)' : 'transparent', color: i === active ? 'var(--action)' : 'var(--text-muted)' }}>
                {Ico ? <Ico size={16} /> : <span style={{ width: 16 }} />}
                <span className="flex-1 text-sm font-medium" style={{ color: 'var(--text-heading)' }}>{c.label}</span>
                <span className="text-xs" style={{ color: i === active ? 'var(--action)' : 'var(--text-muted)' }}>{c.hint}</span>
              </li>
            )
          })}
        </ul>
      </div>
    </div>
  )
}
