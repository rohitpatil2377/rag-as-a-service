// Reusable disclosure section — a Card with a clickable header (icon + title + subtitle
// + a chevron that rotates when open). Children render below the header only when open.
// Kept out of ui.tsx deliberately so it can evolve without touching the shared file.
import { useState, type ReactNode } from 'react'
import { Icon } from './icons'
import { Card } from './ui'

export function Collapsible({
  title,
  subtitle,
  icon,
  defaultOpen,
  children,
}: {
  title: string
  subtitle?: string
  icon?: ReactNode
  defaultOpen?: boolean
  children: ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen ?? false)
  return (
    <Card>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex items-center gap-3 w-full text-left"
        style={{ background: 'transparent', border: 'none', padding: 0, cursor: 'pointer', color: 'inherit' }}
      >
        {icon && <span style={{ color: 'var(--text-muted)', flexShrink: 0, display: 'inline-flex' }}>{icon}</span>}
        <span className="min-w-0 flex-grow">
          <span className="font-display font-semibold block" style={{ fontSize: 16 }}>{title}</span>
          {subtitle && <span className="field-help block" style={{ margin: 0 }}>{subtitle}</span>}
        </span>
        <span
          style={{
            color: 'var(--text-muted)',
            flexShrink: 0,
            display: 'inline-flex',
            transition: 'transform .18s ease',
            transform: open ? 'rotate(90deg)' : 'rotate(0deg)',
          }}
        >
          <Icon.chevron size={18} />
        </span>
      </button>
      {open && <div className="mt-4">{children}</div>}
    </Card>
  )
}
