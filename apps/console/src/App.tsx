import { Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { PlatformLayout } from './components/PlatformLayout'
import { useAuth } from './lib/auth'
import { Agents } from './pages/Agents'
import { Access } from './pages/Access'
import { Ask } from './pages/Ask'
import { Audit } from './pages/Audit'
// Compliance page retired from the UI (2026-08) — the page component (pages/Compliance.tsx) and
// all backend compliance endpoints (attestation / erasure / retention / frameworks) are retained;
// only the navigation entry and route are removed so it is no longer surfaced in the console.
// Usage & Showback page retired from the UI (2026-10) — the nav entry, route and page
// component are removed. The metering API (`/v1/kb/{id}/usage`) is retained and still feeds
// the Dashboard query counters and the KB page's 30-day card.
import { Connectors } from './pages/Connectors'
import { Dashboard } from './pages/Dashboard'
import { Data } from './pages/Data'
import { Graph } from './pages/Graph'
import { Grants } from './pages/Grants'
import { KnowledgeBase } from './pages/KnowledgeBase'
import { Login } from './pages/Login'
import { SetPassword } from './pages/SetPassword'
import { Airgap } from './pages/Airgap'
import { Models } from './pages/Models'
import { Organisations } from './pages/Organisations'
import { Guardrails } from './pages/Guardrails'
import { Quality } from './pages/Quality'
import { Residency } from './pages/Residency'
import { Reviews } from './pages/Reviews'
import { Settings } from './pages/Settings'
import { Team } from './pages/Team'
import { Workflows } from './pages/Workflows'

export function App() {
  const { session, isPlatformAdmin } = useAuth()

  if (!session) {
    return (
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/set-password" element={<SetPassword />} />
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    )
  }

  // First-ever login for an account: brief loader while we probe whether it's a platform operator,
  // so a fresh super-admin doesn't flash the tenant shell before landing in the portal. Returning
  // users are seeded from cache (non-null) and skip this entirely.
  if (isPlatformAdmin === null) {
    return <div className="flex h-screen items-center justify-center" style={{ color: 'var(--text-muted)', fontSize: 13 }}>Loading…</div>
  }

  // SUPER_ADMIN gets a SEPARATE platform portal — never the tenant shell (separation of duties:
  // the fleet operator can't read tenant data anyway).
  if (isPlatformAdmin) {
    return (
      <Routes>
        <Route path="/platform" element={<PlatformLayout />}>
          <Route index element={<Navigate to="/platform/organisations" replace />} />
          <Route path="organisations" element={<Organisations />} />
          <Route path="residency" element={<Residency />} />
          <Route path="airgap" element={<Airgap />} />
        </Route>
        <Route path="/set-password" element={<SetPassword />} />
        <Route path="*" element={<Navigate to="/platform/organisations" replace />} />
      </Routes>
    )
  }

  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/knowledge-base" element={<KnowledgeBase />} />
        <Route path="/connectors" element={<Connectors />} />
        <Route path="/ingest" element={<Navigate to="/knowledge-base" replace />} />
        <Route path="/ask" element={<Ask />} />
        <Route path="/graph" element={<Graph />} />
        <Route path="/data" element={<Data />} />
        <Route path="/workflows" element={<Workflows />} />
        <Route path="/reviews" element={<Reviews />} />
        <Route path="/quality" element={<Quality />} />
        <Route path="/guardrails" element={<Guardrails />} />
        <Route path="/grants" element={<Grants />} />
        <Route path="/access" element={<Access />} />
        <Route path="/audit" element={<Audit />} />
        <Route path="/models" element={<Models />} />
        <Route path="/agents" element={<Agents />} />
        <Route path="/team" element={<Team />} />
        <Route path="/settings" element={<Settings />} />
      </Route>
      <Route path="/set-password" element={<SetPassword />} />
      <Route path="/login" element={<Navigate to="/" replace />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
