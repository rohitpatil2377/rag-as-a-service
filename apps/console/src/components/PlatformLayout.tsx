// Platform portal shell — the SUPER_ADMIN console, separate from the tenant app. Its own
// sidebar shows ONLY fleet operations (orgs, residency, air-gap, billing); no tenant nav.
// Guarded: a non-platform-admin is bounced back to the tenant console.
import { NavLink, Navigate, Outlet } from 'react-router-dom'
import { BRAND_NAME, TAGLINE } from '../lib/brand'
import { useAuth } from '../lib/auth'
import { Icon } from './icons'
import { ThemeToggle } from './ui'

const PLATFORM_NAV = [
  { to: '/platform/organisations', label: 'Organisations', icon: Icon.layers, ready: true },
  { to: '/platform/residency', label: 'Residency', icon: Icon.shield, ready: true },
  { to: '/platform/airgap', label: 'Air-gap Preflight', icon: Icon.scale, ready: true },
  { to: '/platform/billing', label: 'Billing', icon: Icon.activity, ready: false },
]

export function PlatformLayout() {
  const { session, isPlatformAdmin, logout } = useAuth()

  if (isPlatformAdmin === null) {
    return <div className="flex h-screen items-center justify-center" style={{ color: 'var(--text-muted)', fontSize: 13 }}>Loading platform console…</div>
  }
  if (!isPlatformAdmin) return <Navigate to="/" replace />

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Platform sidebar — deliberately distinct from the tenant shell */}
      <aside className="flex flex-col shrink-0" style={{ width: 248, background: 'var(--sidebar)', borderRight: '1px solid var(--border)' }}>
        <div className="h-16 flex items-center px-5" style={{ borderBottom: '1px solid var(--border)', gap: 10 }}>
          <div style={{ width: 30, height: 30, borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'linear-gradient(135deg,#6C86F0,#A47BE8)', color: '#fff' }}>
            <Icon.shield size={17} />
          </div>
          <div style={{ lineHeight: 1.15 }}>
            <div style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: 14, color: 'var(--text-heading)' }}>{BRAND_NAME}</div>
            {/* NOT `textTransform: uppercase` — it would render "RAGaaS" as "RAGAAS". */}
            <div className="mono" style={{ fontSize: 10, letterSpacing: '.08em', color: 'var(--action)' }}>{TAGLINE}</div>
          </div>
        </div>

        <nav className="flex-grow overflow-y-auto py-3">
          <div className="px-5 mt-1 mb-1 mono uppercase font-bold tracking-wider" style={{ fontSize: 10, color: 'var(--text-muted)' }}>Fleet</div>
          {PLATFORM_NAV.map((item) => {
            const IconC = item.icon
            if (!item.ready) {
              return (
                <div key={item.to} className="flex items-center gap-3 py-2 px-5 text-sm font-medium" style={{ color: 'var(--text-muted)', opacity: 0.55, cursor: 'not-allowed' }}>
                  <IconC size={17} /><span>{item.label}</span>
                  <span className="mono" style={{ marginLeft: 'auto', fontSize: 9, padding: '1px 5px', borderRadius: 4, border: '1px solid var(--border)' }}>soon</span>
                </div>
              )
            }
            return (
              <NavLink key={item.to} to={item.to}
                className="flex items-center gap-3 py-2 px-5 text-sm font-medium transition-colors"
                style={({ isActive }) => ({
                  color: isActive ? 'var(--text-heading)' : 'var(--text-muted)',
                  background: isActive ? 'color-mix(in srgb, var(--action) 10%, transparent)' : 'transparent',
                  borderLeft: isActive ? '2px solid var(--action)' : '2px solid transparent',
                })}>
                <IconC size={17} /><span>{item.label}</span>
              </NavLink>
            )
          })}
        </nav>

        <div style={{ borderTop: '1px solid var(--border)', padding: '10px 14px', display: 'flex', alignItems: 'center', gap: 8 }}>
          <div className="min-w-0 flex-grow">
            <div className="truncate" style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-heading)' }}>{session?.name || 'Super Admin'}</div>
            <div className="truncate mono" style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>{session?.subject}</div>
          </div>
          <ThemeToggle />
          <button type="button" onClick={logout} title="Sign out" aria-label="Sign out"
            className="rounded-md flex items-center justify-center interactive" style={{ width: 28, height: 28, border: '1px solid var(--border)', background: 'var(--surface-2)', color: 'var(--text-muted)', cursor: 'pointer' }}>
            <Icon.logout size={15} />
          </button>
        </div>
      </aside>

      {/* Content */}
      <main className="flex-grow overflow-y-auto" style={{ background: 'var(--bg)' }}>
        <div style={{ maxWidth: 1080, margin: '0 auto', padding: '28px 32px' }}>
          <Outlet />
        </div>
      </main>
    </div>
  )
}
