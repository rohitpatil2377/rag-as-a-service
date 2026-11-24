// Toasts (ARIA-live) + Modal. Token-driven, both themes.
import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from 'react'
import { Icon } from './icons'

type ToastKind = 'success' | 'error' | 'info'
interface Toast { id: number; kind: ToastKind; message: string }
interface ToastCtx { push: (kind: ToastKind, message: string) => void }
const Ctx = createContext<ToastCtx | null>(null)

const TONE: Record<ToastKind, string> = { success: '--success', error: '--danger', info: '--info' }
const ICON: Record<ToastKind, ReactNode> = { success: <Icon.check size={16} />, error: <Icon.alert size={16} />, info: <Icon.info size={16} /> }

// Success/info toasts auto-dismiss after this long. Errors are STICKY — they stay until the
// user dismisses them with the ✕, so a failure (e.g. a rejected upload) can't vanish before
// it's been read. See UAT feedback: transient error toasts were unreadable.
const AUTO_DISMISS_MS = 4200

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const seq = useRef(0)
  const remove = useCallback((id: number) => setToasts((t) => t.filter((x) => x.id !== id)), [])
  const push = useCallback((kind: ToastKind, message: string) => {
    const id = ++seq.current
    setToasts((t) => [...t, { id, kind, message }])
    // Errors persist until manually dismissed; everything else auto-clears.
    if (kind !== 'error') setTimeout(() => remove(id), AUTO_DISMISS_MS)
  }, [remove])
  const value = useMemo(() => ({ push }), [push])
  return (
    <Ctx.Provider value={value}>
      {children}
      <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-2" role="region" aria-live="polite" aria-label="Notifications">
        {toasts.map((t) => (
          <div key={t.id} className="surface flex items-start gap-2.5 px-4 py-3 text-sm animate-[fadeIn_.2s_ease-out]"
            role={t.kind === 'error' ? 'alert' : 'status'}
            style={{ minWidth: 260, maxWidth: 420, borderLeft: `3px solid var(${TONE[t.kind]})` }}>
            <span style={{ color: `var(${TONE[t.kind]})`, marginTop: 1 }}>{ICON[t.kind]}</span>
            <span className="flex-1" style={{ color: 'var(--text-primary)' }}>{t.message}</span>
            <button onClick={() => remove(t.id)} aria-label="Dismiss notification"
              title="Dismiss"
              style={{ color: 'var(--text-muted)', cursor: 'pointer', background: 'none', border: 'none', padding: 0, lineHeight: 0, marginTop: 1 }}>
              <Icon.x size={14} />
            </button>
          </div>
        ))}
      </div>
    </Ctx.Provider>
  )
}

export function useToast(): ToastCtx {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useToast must be used within ToastProvider')
  return ctx
}
