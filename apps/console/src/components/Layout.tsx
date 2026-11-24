// App shell — sidebar (saffron active rail) + topbar (breadcrumb · live status · theme · user).
import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { api, type SystemInfo } from '../lib/api'
import { useAuth } from '../lib/auth'
import { useKb } from '../lib/kb'
import { Icon } from './icons'
import { IconButton, ThemeToggle } from './ui'
import { CommandPalette } from './CommandPalette'
import { JobCentre } from './JobCentre'
import { ModelIndicator } from './ModelIndicator'
import { IngestIndicator } from './IngestIndicator'
import { BrandMark } from './BrandMark'
import { BRAND_LEAD, BRAND_MARK, TAGLINE } from '../lib/brand'
import { LANGS, useI18n } from '../lib/i18n'

function LanguageSwitcher() {
  const { lang, setLang, t } = useI18n()
  return (
    <select value={lang} onChange={(e) => setLang(e.target.value as typeof lang)} aria-label={t('common.language', 'Language')}
      title={t('common.language', 'Language')}
      className="rounded-md text-xs font-semibold" style={{ height: 30, padding: '0 6px', color: 'var(--text-muted)', border: '1px solid var(--border)', background: 'var(--surface-2)', cursor: 'pointer' }}>
      {LANGS.map((l) => <option key={l.value} value={l.value}>{l.label}</option>)}
    </select>
  )
}

// Brand mark — the trimmed logo on an ivory plate (legible on the dark sidebar), with a
// monogram mark; no raster asset needed.
function SidebarBrand() {
  return (
    <div className="flex items-center gap-2">
      <span className="logo-plate inline-flex shrink-0"><BrandMark size={30} /></span>
      <div className="leading-tight min-w-0">
        <div className="font-display font-bold" style={{ fontSize: 14.5, letterSpacing: '-0.01em', color: 'var(--text-heading)', whiteSpace: 'nowrap' }}>
          {BRAND_LEAD} <span style={{ color: 'var(--action)' }}>{BRAND_MARK}</span>
        </div>
        {/* Same lockup as the sign-in screen, and from the same source: the two halves are
            styled differently, so they must stay two strings. See lib/brand.ts. */}
        <div className="mono truncate" style={{ fontSize: 9, letterSpacing: '.14em', color: 'var(--text-muted)' }}>{TAGLINE}</div>
      </div>
    </div>
  )
}

// Knowledge-base switcher — pick which KB every screen operates on.
function KbSwitcher() {
  const { kbs, kbId, setKbId } = useKb()
  const navigate = useNavigate()
  return (
    <div className="px-4 pt-4">
      <div className="mono uppercase font-bold tracking-wider mb-1.5" style={{ fontSize: 10, color: 'var(--text-muted)' }}>Knowledge base</div>
      {kbs.length === 0 ? (
        <button type="button" onClick={() => navigate('/knowledge-base')}
          className="w-full rounded-md text-sm font-semibold px-3 py-2.5 text-left flex items-center gap-2"
          style={{ background: 'var(--surface-2)', color: 'var(--text-muted)', border: '1px dashed var(--border)', cursor: 'pointer' }}>
          <Icon.plus size={14} /> Create a knowledge base
        </button>
      ) : (
        <div className="relative">
          <select value={kbId} onChange={(e) => setKbId(e.target.value)} aria-label="Select knowledge base"
            className="w-full appearance-none rounded-md text-sm font-semibold pl-3 pr-8 py-2.5"
            style={{ background: 'var(--surface-2)', color: 'var(--text-heading)', border: '1px solid var(--border)', cursor: 'pointer' }}>
            {kbs.map((k) => <option key={k.kb_id} value={k.kb_id}>{k.name}</option>)}
          </select>
          <Icon.chevron size={14} style={{ position: 'absolute', right: 10, top: 12, transform: 'rotate(90deg)', color: 'var(--text-muted)', pointerEvents: 'none' }} />
        </div>
      )}
    </div>
  )
}

// Routes that work without any KB (org-level, or where you create the first KB).
const KB_OPTIONAL_ROUTES = new Set(['/', '/knowledge-base', '/settings', '/team'])

function NoKbGate() {
  const navigate = useNavigate()
  return (
    <div className="flex flex-col items-center justify-center text-center" style={{ minHeight: '60vh' }}>
      <Icon.database size={34} style={{ color: 'var(--text-muted)' }} />
      <h2 className="font-display font-semibold mt-4" style={{ fontSize: 20, color: 'var(--text-heading)' }}>No knowledge base yet</h2>
      <p className="mt-2 mb-5" style={{ color: 'var(--text-muted)', maxWidth: 420 }}>
        This screen works on a knowledge base. Create your first one — its own isolated space for documents,
        retrieval, model, and audit — then come back.
      </p>
      <button type="button" onClick={() => navigate('/knowledge-base')} className="btn btn-primary" style={{ gap: 8 }}>
        <Icon.plus size={16} /> Create a knowledge base
      </button>
    </div>
  )
}

// `roles` = capability roles that unlock a module. Undefined = visible to any signed-in user.
// A user sees a module if any of their roles matches (owner carries admin, so owners see all).
// INVARIANT: every entry carries a DISTINCT `icon`. The rail is scanned by glyph, so a shared
// icon (Knowledge Bases vs Data & SQL, Connectors vs Workflows vs Agents, the four shields)
// makes two different modules read as the same one — see `icons.tsx` for the per-module glyphs.
export const NAV = [
  { to: '/', label: 'Dashboard', tkey: 'nav.dashboard', section: 'Overview', icon: Icon.dashboard },
  { to: '/knowledge-base', label: 'Knowledge Bases', tkey: 'nav.knowledge_bases', section: 'Knowledge', icon: Icon.database, roles: ['ingestor', 'admin'] },
  { to: '/connectors', label: 'Connectors', tkey: 'nav.connectors', section: 'Knowledge', icon: Icon.connectors, roles: ['ingestor', 'admin'] },
  { to: '/graph', label: 'Knowledge Graph', tkey: 'nav.knowledge_graph', section: 'Knowledge', icon: Icon.graph, roles: ['reader', 'ingestor', 'reviewer', 'admin'] },
  { to: '/ask', label: 'Ask', tkey: 'nav.ask', section: 'Answer', icon: Icon.chat },
  { to: '/data', label: 'Data & SQL', tkey: 'nav.data', section: 'Answer', icon: Icon.table, roles: ['admin'] },
  { to: '/workflows', label: 'Workflows', tkey: 'nav.workflows', section: 'Answer', icon: Icon.workflow, roles: ['admin'] },
  { to: '/reviews', label: 'Review Queue', tkey: 'nav.reviews', section: 'Govern', icon: Icon.review, roles: ['reviewer', 'admin'] },
  { to: '/quality', label: 'Quality', tkey: 'nav.quality', section: 'Govern', icon: Icon.gauge, roles: ['reviewer', 'admin'] },
  { to: '/guardrails', label: 'Guardrails', tkey: 'nav.guardrails', section: 'Govern', icon: Icon.shield, roles: ['admin'] },
  { to: '/grants', label: 'Federation', tkey: 'nav.grants', section: 'Govern', icon: Icon.globe, roles: ['reader', 'admin'] },
  { to: '/access', label: 'Access', tkey: 'nav.access', section: 'Govern', icon: Icon.key },
  { to: '/audit', label: 'Audit Trail', tkey: 'nav.audit', section: 'Govern', icon: Icon.audit, roles: ['admin'] },
  { to: '/models', label: 'Models', tkey: 'nav.models', section: 'Platform', icon: Icon.layers, roles: ['admin'] },
  { to: '/agents', label: 'Agents', tkey: 'nav.agents', section: 'Platform', icon: Icon.agent, roles: ['admin'] },
  { to: '/team', label: 'Team', tkey: 'nav.team', section: 'Platform', icon: Icon.users, roles: ['admin'] },
  { to: '/settings', label: 'Settings', tkey: 'nav.settings', section: 'Platform', icon: Icon.settings, roles: ['admin'] },
] as const

export type NavItem = (typeof NAV)[number]

// How often the sidebar re-counts access requests awaiting a decision. Slow on purpose: this is
// a "somebody is waiting on you" signal, not a live feed, and it costs one call per tick.
const ACCESS_POLL_MS = 60_000

// A module is visible if it has no role requirement, or the user holds one of its roles.
export function canSee(item: NavItem, roles: string[]): boolean {
  const req = 'roles' in item ? item.roles : undefined
  return !req || req.some((r) => roles.includes(r))
}

// Live platform status: sovereignty posture + API reachability + environment.
function StatusBadge() {
  const { session } = useAuth()
  const [info, setInfo] = useState<SystemInfo | null>(null)
  const [ok, setOk] = useState<boolean | null>(null)
  useEffect(() => {
    if (!session) return
    api.systemInfo(session.token).then(setInfo).catch(() => setInfo(null))
    api.health().then(() => setOk(true)).catch(() => setOk(false))
  }, [session])
  return (
    <div className="hidden sm:flex items-center gap-2 mr-1">
      {info?.sovereign_enforce && (
        <span className="pill" title="Sovereignty egress guard enforced" style={{ color: 'var(--success)', borderColor: 'color-mix(in srgb, var(--success) 35%, transparent)', background: 'color-mix(in srgb, var(--success) 12%, transparent)' }}>
          <Icon.shield size={12} />Sovereign
        </span>
      )}
      <span className="pill" title="API status" style={{ color: 'var(--text-muted)', borderColor: 'var(--border)', background: 'transparent' }}>
        <span className="inline-block rounded-full" style={{ width: 7, height: 7, background: ok === false ? 'var(--danger)' : 'var(--success)' }} />
        {info?.env ?? (ok === false ? 'offline' : '…')}
      </span>
    </div>
  )
}

export function Layout() {
  const { session, logout } = useAuth()
  const { kbs } = useKb()
  const navigate = useNavigate()
  const loc = useLocation()
  const { t } = useI18n()
  const current = NAV.find((n) => n.to === loc.pathname) ?? NAV[0]
  const roles = session?.roles ?? []
  // A KB manager (manages ≥1 KB) also sees Team — but scoped to their KB's members (see Team.tsx).
  const [managesKb, setManagesKb] = useState(false)
  useEffect(() => {
    if (!session) return
    api.myManagedKbs(session.token).then((m) => setManagesKb(m.length > 0)).catch(() => setManagesKb(false))
  }, [session])
  // Models is visible to EVERY member: org admins and KB managers manage it; everyone else gets a
  // read-only view of the models in use (the page + API enforce what each role can actually do).
  // Access is visible to EVERY member (like Models): anyone can request access to another KB;
  // managers get the approval queue and org admins the overview — the page renders each section by role.
  // A KB manager also manages their KB's agentic + governance surface — Team, Agents (MCP
  // connection), Workflows and Guardrails — even though those are org-admin-only for ordinary
  // members. Scoped by the page/API (the KB-guardrails write is _can_manage_kb-gated server-side).
  const managerNav = new Set(['/team', '/agents', '/workflows', '/guardrails'])
  // Pending access requests waiting on THIS user. Without this the only way to learn that
  // somebody asked for access is to open the page and look -- a request could sit for days.
  // Counted for whoever can actually decide it: a KB's manager, or an org owner/admin (whose
  // queue is the whole org).
  const [pendingAccess, setPendingAccess] = useState(0)
  useEffect(() => {
    if (!session) return
    let stop = false
    const isAdmin = roles.includes('owner') || roles.includes('admin')
    const count = async () => {
      try {
        const rows = isAdmin
          ? await api.accessOverview(session.token)
          : (await Promise.all(
              (await api.myManagedKbs(session.token)).map((id) =>
                api.listAccessRequests(session.token, id).catch(() => []),
              ),
            )).flat()
        if (!stop) setPendingAccess(rows.filter((r) => r.status === 'pending').length)
      } catch { if (!stop) setPendingAccess(0) }
    }
    void count()
    const t = window.setInterval(count, ACCESS_POLL_MS)
    return () => { stop = true; window.clearInterval(t) }
  }, [session, roles])

  const canSeeNav = (item: NavItem) =>
    canSee(item, roles) || item.to === '/models' || item.to === '/access' || (managerNav.has(item.to) && managesKb)
  const needsKbButHasNone = kbs.length === 0 && !KB_OPTIONAL_ROUTES.has(loc.pathname)
  // Sidebar collapse — icons-only rail, so a long page gets the horizontal room back.
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem('rag.sidebar') === 'collapsed')
  const toggleSidebar = () => setCollapsed((c) => {
    localStorage.setItem('rag.sidebar', c ? 'open' : 'collapsed')
    return !c
  })
  let lastSection = ''

  // Route guard: a user who directly navigates to a module their role can't see is sent home
  // (defence-in-depth — the backend already 403s the underlying calls).
  useEffect(() => {
    if (current && !canSeeNav(current) && loc.pathname !== '/') navigate('/', { replace: true })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loc.pathname])

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="flex flex-col shrink-0" style={{ width: collapsed ? 62 : 256, background: 'var(--sidebar)', borderRight: '1px solid var(--border)', transition: 'width .16s ease' }}>
        <div className={`h-16 flex items-center ${collapsed ? 'justify-center px-2' : 'pl-3 pr-2 gap-1.5'}`} style={{ borderBottom: '1px solid var(--border)' }}>
          {!collapsed && <div className="min-w-0 flex-grow"><SidebarBrand /></div>}
          <button type="button" onClick={toggleSidebar}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            className="rounded-md flex items-center justify-center shrink-0 interactive"
            style={{ width: 26, height: 26, border: '1px solid var(--border)', background: 'var(--surface-2)', color: 'var(--text-muted)', cursor: 'pointer' }}>
            <Icon.sidebar size={14} />
          </button>
        </div>
        {!collapsed && <KbSwitcher />}
        <nav className="flex-grow overflow-y-auto py-3">
          {NAV.filter(canSeeNav).map((item) => {
            const header = item.section !== lastSection ? ((lastSection = item.section), item.section) : null
            const IconC = item.icon
            return (
              <div key={item.to}>
                {header && !collapsed && <div className="px-5 mt-3 mb-1 mono uppercase font-bold tracking-wider" style={{ fontSize: 10, color: 'var(--text-muted)' }}>{header}</div>}
                <NavLink to={item.to} end={item.to === '/'} title={collapsed ? t(item.tkey, item.label) : undefined}
                  className={`flex items-center gap-3 py-2 text-sm font-medium transition-colors ${collapsed ? 'justify-center px-0' : 'px-5'}`}
                  style={({ isActive }) => ({
                    color: isActive ? 'var(--action)' : 'var(--text-primary)',
                    borderLeft: `3px solid ${isActive ? 'var(--action)' : 'transparent'}`,
                    background: isActive ? 'color-mix(in srgb, var(--action) 8%, transparent)' : 'transparent',
                    fontWeight: isActive ? 600 : 500,
                  })}>
                  <IconC size={18} /> {!collapsed && t(item.tkey, item.label)}
                  {item.to === '/access' && pendingAccess > 0 && (
                    <span
                      className="ml-auto pill"
                      title={`${pendingAccess} access request(s) waiting for your decision`}
                      style={{ fontSize: 10, color: 'var(--on-action)', background: 'var(--action)', borderColor: 'var(--action)' }}
                    >
                      {pendingAccess}
                    </span>
                  )}
                </NavLink>
              </div>
            )
          })}
        </nav>
      </aside>

      {/* Main column */}
      <div className="flex-grow flex flex-col overflow-hidden" style={{ background: 'var(--canvas)' }}>
        <header className="h-16 shrink-0 flex items-center justify-between px-8" style={{ borderBottom: '1px solid var(--border)' }}>
          <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-muted)' }}>
            <span>{current.section}</span><Icon.chevron size={14} /><span className="font-semibold" style={{ color: 'var(--text-heading)' }}>{t(current.tkey, current.label)}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <button onClick={() => window.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', metaKey: true }))}
              className="hidden sm:flex items-center gap-1.5 rounded-md px-2.5 text-xs"
              style={{ height: 30, color: 'var(--text-muted)', border: '1px solid var(--border)', background: 'var(--surface-2)', cursor: 'pointer' }}
              aria-label="Open command palette" title="Command palette (⌘K / Ctrl-K)">
              <Icon.ask size={13} /> Search <kbd className="mono" style={{ fontSize: 10 }}>⌘K</kbd>
            </button>
            <IngestIndicator />
            <ModelIndicator />
            <JobCentre />
            <LanguageSwitcher />
            <StatusBadge />
            <ThemeToggle />
            {/* Identity lives here (top-right), not as a block at the foot of the sidebar. */}
            <div className="hidden md:flex items-center gap-2 pl-1.5 ml-0.5" style={{ borderLeft: '1px solid var(--border)' }}
              title={`${session?.name || session?.subject} · ${session?.orgName} · ${session?.role}`}>
              <span className="rounded-full flex items-center justify-center font-display font-semibold shrink-0"
                style={{ width: 26, height: 26, fontSize: 11, background: 'var(--surface-2)', color: 'var(--text-heading)', border: '1px solid var(--border)' }}>
                {(session?.name || session?.subject)?.[0]?.toUpperCase()}
              </span>
              <span className="leading-none truncate" style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-heading)', maxWidth: 110 }}>
                {session?.name?.split(' ')[0] || session?.subject}
              </span>
            </div>
            <IconButton label="Sign out" onClick={() => { logout(); navigate('/login') }}><Icon.logout size={18} /></IconButton>
          </div>
        </header>
        <main className="flex-grow overflow-y-auto">
          <div className="px-6 pt-3 pb-5">{needsKbButHasNone ? <NoKbGate /> : <Outlet />}</div>
        </main>
      </div>
      <CommandPalette />
    </div>
  )
}
