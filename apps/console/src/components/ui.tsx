// Reusable, token-driven primitives (institutional, both themes). Screens compose these.
import { useState, type ButtonHTMLAttributes, type CSSProperties, type InputHTMLAttributes, type ReactNode } from 'react'
import type { AnswerStatus, HitlStatus } from '../lib/api'
import { useTheme } from '../lib/theme'
import { Icon } from './icons'

// --- Button --------------------------------------------------------------
type Variant = 'primary' | 'secondary' | 'ghost' | 'subtle' | 'danger'
export function Button(
  { variant = 'primary', size, loading, icon, children, className = '', ...p }:
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: 'sm'; loading?: boolean; icon?: ReactNode },
) {
  return (
    <button
      className={`btn btn-${variant} ${size === 'sm' ? 'btn-sm' : ''} ${className}`}
      disabled={loading || p.disabled}
      {...p}
    >
      {loading ? <Spinner /> : icon}
      {children}
    </button>
  )
}

export function IconButton({ label, children, className = '', ...p }: ButtonHTMLAttributes<HTMLButtonElement> & { label: string }) {
  return (
    <button aria-label={label} title={label} className={`btn btn-ghost btn-icon ${className}`} {...p}>
      {children}
    </button>
  )
}

// --- Surfaces & headers --------------------------------------------------
export function Card({ children, className = '', interactive }: { children: ReactNode; className?: string; interactive?: boolean }) {
  return <div className={`card ${interactive ? 'interactive' : ''} ${className}`}>{children}</div>
}
export function Eyebrow({ children }: { children: ReactNode }) {
  return <span className="eyebrow">{children}</span>
}
// Compact, product-grade header: eyebrow + title only. The old `lead` paragraph (a
// didactic explainer under every title) read as marketing copy and cost vertical space —
// the prop is still accepted for compatibility but deliberately not rendered.
export function PageHeader({ eyebrow, title, actions }: { eyebrow: string; title: string; lead?: string; actions?: ReactNode }) {
  return (
    <header className="mb-3">
      <div className="flex items-center gap-3 mb-1"><span className="w-9 rule-gold" /><Eyebrow>{eyebrow}</Eyebrow></div>
      <div className="flex items-start justify-between gap-4">
        <h1 className="font-display font-semibold tracking-tight" style={{ fontSize: 26, lineHeight: 1.15, letterSpacing: '-0.01em' }}>{title}</h1>
        {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
      </div>
    </header>
  )
}

// --- Status pills (icon + label + colour, never colour alone) ------------
export function toneStyle(color: string): CSSProperties {
  return { color: `var(${color})`, borderColor: `color-mix(in srgb, var(${color}) 35%, transparent)`, background: `color-mix(in srgb, var(${color}) 12%, transparent)` }
}
const ANSWER: Record<AnswerStatus, { label: string; tone: string; icon: ReactNode }> = {
  grounded: { label: 'Grounded', tone: '--success', icon: <Icon.check size={13} /> },
  flagged_for_review: { label: 'Flagged for review', tone: '--warning', icon: <Icon.flag size={13} /> },
  abstained: { label: 'Abstained', tone: '--text-muted', icon: <Icon.minus size={13} /> },
}
// Grounded means "traceable to a cited source", NOT "verified correct". A grounded answer
// whose faithfulness sits just above the gate (QA ISS-015: a wrong answer badged green at
// 0.75) is honestly medium-confidence — show it as "Grounded · verify" (amber) rather than a
// confident green, so a reader knows to check the cited span. UI-only; the grounding gate is
// unchanged. Above CONFIDENT it reads as a plain green Grounded.
const CONFIDENT_FAITHFULNESS = 0.85
export function AnswerPill({ status, faithfulness }: { status: AnswerStatus; faithfulness?: number | null }) {
  if (status === 'grounded' && faithfulness != null && faithfulness < CONFIDENT_FAITHFULNESS) {
    return (
      <span className="pill" style={toneStyle('--warning')}
        title={`Grounded (traceable to a cited source) but medium confidence (faithfulness ${faithfulness.toFixed(2)}) — verify against the cited source before relying on it.`}>
        <Icon.alert size={13} /> Grounded · verify
      </span>
    )
  }
  const s = ANSWER[status]
  return <span className="pill" style={toneStyle(s.tone)}>{s.icon}{s.label}</span>
}
const HITL: Record<HitlStatus, { tone: string; icon: ReactNode }> = {
  pending: { tone: '--warning', icon: <Icon.flag size={13} /> },
  accepted: { tone: '--success', icon: <Icon.check size={13} /> },
  modified: { tone: '--info', icon: <Icon.doc size={13} /> },
  flagged: { tone: '--danger', icon: <Icon.alert size={13} /> },
  overridden: { tone: '--text-muted', icon: <Icon.check size={13} /> },
}
export function HitlPill({ status }: { status: HitlStatus }) {
  return <span className="pill capitalize" style={toneStyle(HITL[status].tone)}>{HITL[status].icon}{status}</span>
}

// --- Forms ---------------------------------------------------------------
export function Field({ label, help, children }: { label: string; help?: string; children: ReactNode }) {
  return <label className="block"><span className="field-label">{label}</span>{children}{help && <span className="field-help">{help}</span>}</label>
}

// --- Feedback ------------------------------------------------------------
export function Spinner({ size = 16 }: { size?: number }) {
  return <span role="status" aria-label="Loading" className="inline-block rounded-full animate-spin" style={{ width: size, height: size, border: '2px solid color-mix(in srgb, var(--action) 30%, transparent)', borderTopColor: 'var(--action)' }} />
}
export function Skeleton({ className = '', style }: { className?: string; style?: CSSProperties }) {
  return <div className={`animate-pulse rounded-md ${className}`} style={{ background: 'color-mix(in srgb, var(--text-muted) 16%, transparent)', ...style }} />
}
export function EmptyState({ icon, title, hint, action }: { icon?: ReactNode; title: string; hint?: string; action?: ReactNode }) {
  return (
    <div className="card flex flex-col items-center text-center py-14">
      {icon && <div className="mb-3" style={{ color: 'var(--text-muted)' }}>{icon}</div>}
      <h3 className="font-display font-semibold" style={{ fontSize: 17 }}>{title}</h3>
      {hint && <p className="mt-1 text-sm max-w-sm" style={{ color: 'var(--text-muted)' }}>{hint}</p>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  )
}
type AlertKind = 'error' | 'info' | 'success' | 'warning'
const ALERT: Record<AlertKind, { tone: string; icon: ReactNode }> = {
  error: { tone: '--danger', icon: <Icon.alert size={16} /> },
  info: { tone: '--info', icon: <Icon.info size={16} /> },
  success: { tone: '--success', icon: <Icon.check size={16} /> },
  warning: { tone: '--warning', icon: <Icon.flag size={16} /> },
}
export function Alert({ kind, children }: { kind: AlertKind; children: ReactNode }) {
  const a = ALERT[kind]
  return (
    <div className="flex items-start gap-2.5 rounded-lg border p-3 text-sm" style={toneStyle(a.tone)}>
      <span className="mt-0.5 shrink-0">{a.icon}</span>
      <div style={{ color: 'var(--text-primary)' }}>{children}</div>
    </div>
  )
}

// --- Meters --------------------------------------------------------------
export function FaithfulnessMeter({ score }: { score: number }) {
  // Quiet by design: muted label + value, a slim track, and only the fill carries the
  // band colour (softened) — informative without shouting.
  const tone = score >= 0.7 ? '--success' : score >= 0.4 ? '--warning' : '--danger'
  const band = score >= 0.7 ? 'faithful' : score >= 0.4 ? 'borderline' : 'unsupported'
  return (
    <div className="flex items-center gap-2" title={`faithfulness ${score.toFixed(3)} (${band})`}>
      <span className="text-xs mono" style={{ color: 'var(--text-muted)' }}>faithfulness</span>
      <span className="relative rounded-full overflow-hidden" style={{ width: 64, height: 4, background: 'color-mix(in srgb, var(--text-muted) 16%, transparent)' }}>
        <span className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${Math.max(4, score * 100)}%`, background: `color-mix(in srgb, var(${tone}) 78%, transparent)` }} />
      </span>
      <span className="text-xs mono num" style={{ color: 'var(--text-muted)' }}>{score.toFixed(2)}</span>
    </div>
  )
}

// --- Citation chip -------------------------------------------------------
export function CitationChip({ n, doc, page, score }: { n: number; doc: string; page: number | null; score: number }) {
  return (
    <span className="chip">
      <span className="mono font-semibold" style={{ color: 'var(--success)' }}>[{n}]</span>
      <span className="truncate" style={{ maxWidth: 180 }}>{doc}</span>
      {page != null && <span className="mono" style={{ color: 'var(--text-muted)' }}>p.{page}</span>}
      <span className="mono num" style={{ color: 'var(--text-muted)' }}>{score.toFixed(2)}</span>
    </span>
  )
}

// --- Segmented control ---------------------------------------------------
export function SegmentedControl<T extends string>({ options, value, onChange }: { options: { value: T; label: string }[]; value: T; onChange: (v: T) => void }) {
  return (
    <div className="inline-flex p-1 rounded-md" style={{ background: 'var(--surface-2)', border: '1px solid var(--border)' }}>
      {options.map((o) => (
        <button
          key={o.value} onClick={() => onChange(o.value)}
          className="px-3 rounded text-sm font-semibold transition-colors"
          style={{ minHeight: 34, color: value === o.value ? 'var(--on-action)' : 'var(--text-muted)', background: value === o.value ? 'var(--action)' : 'transparent' }}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

// --- Theme toggle --------------------------------------------------------
export function ThemeToggle() {
  const { theme, toggle } = useTheme()
  return (
    <IconButton label={theme === 'light' ? 'Switch to dark theme' : 'Switch to light theme'} onClick={toggle}>
      {theme === 'light' ? <Icon.moon size={18} /> : <Icon.sun size={18} />}
    </IconButton>
  )
}

// --- Table wrapper -------------------------------------------------------
export function Table({ children }: { children: ReactNode }) {
  return <div className="surface overflow-hidden"><table className="tbl">{children}</table></div>
}

// --- KPI stat tile -------------------------------------------------------
export function StatTile(
  { label, value, sub, tone = '--text-heading', icon }:
  { label: string; value: ReactNode; sub?: string; tone?: string; icon?: ReactNode },
) {
  return (
    <div className="card !p-5">
      <div className="flex items-center justify-between mb-2">
        <span className="mono uppercase" style={{ fontSize: 10, color: 'var(--text-muted)', letterSpacing: '.06em' }}>{label}</span>
        {icon && <span style={{ color: `var(${tone})` }}>{icon}</span>}
      </div>
      <div className="font-display font-bold num" style={{ fontSize: 26, lineHeight: 1.05, color: `var(${tone})` }}>{value}</div>
      {sub && <div className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>{sub}</div>}
    </div>
  )
}


// --- PasswordInput -------------------------------------------------------
// A password field with an in-field show/hide toggle. Defaults to masked; the toggle is a
// button (not a checkbox) so it never submits the form it lives in.
export function PasswordInput({ style, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  const [show, setShow] = useState(false)
  return (
    <div className="relative">
      <input {...props} type={show ? 'text' : 'password'} className="field-input"
        style={{ paddingRight: 40, ...style }} />
      <button type="button" onClick={() => setShow((v) => !v)} tabIndex={-1}
        aria-label={show ? 'Hide password' : 'Show password'} title={show ? 'Hide password' : 'Show password'}
        className="absolute flex items-center justify-center"
        style={{ right: 6, top: '50%', transform: 'translateY(-50%)', width: 30, height: 30, background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}>
        {show ? <Icon.eyeOff size={16} /> : <Icon.eye size={16} />}
      </button>
    </div>
  )
}
