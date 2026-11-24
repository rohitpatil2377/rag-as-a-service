// Auth context — real accounts (email/password → JWT). Session carries the user and
// their organisation; `tenantId` is the org id (the isolation tenant). OIDC/SSO will
// swap the login/signup calls with no change to pages, which only read `useAuth()`.
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { api, type AuthResponse } from './api'

export interface Session {
  token: string
  tenantId: string      // org id
  subject: string       // email
  name: string
  orgName: string
  role: string
  roles: string[]       // fine-grained capability roles (reader/ingestor/reviewer/admin/owner)
  kbAccess: string[]    // allowed kb_ids ([] = all in org)
  groups: string[]
}

// Fine-grained capability check — mirrors the backend's role gates (ingest requires
// `ingestor`, HITL act requires `reviewer`, config/workflow/grant/member writes require
// `admin`, etc). Always prefer this over the coarse `session.role` (owner/admin/member)
// for gating UI controls: `role` is a display label, `roles` is what the API actually
// enforces, and the two are not guaranteed to line up. `owner` implicitly carries `admin`
// server-side, so callers that mean "admin-only" should pass 'admin' — owners already have
// it in `roles` — rather than re-checking `role === 'owner'` separately.
export function hasRole(session: Session | null | undefined, ...need: string[]): boolean {
  const roles = session?.roles ?? []
  return need.some((r) => roles.includes(r))
}

interface AuthCtx {
  session: Session | null
  // Whether the signed-in account is a platform operator (SUPER_ADMIN). null = still probing.
  // Detected by probing the platform-only endpoint (no extra JWT claim needed).
  isPlatformAdmin: boolean | null
  login: (email: string, password: string) => Promise<void>
  signup: (orgName: string, name: string, email: string, password: string) => Promise<void>
  logout: () => void
}

const Ctx = createContext<AuthCtx | null>(null)
const STORAGE_KEY = 'rag.session'

function load(): Session | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? (JSON.parse(raw) as Session) : null
  } catch {
    return null
  }
}

function toSession(r: AuthResponse): Session {
  return {
    token: r.access_token, tenantId: r.org_id, subject: r.email, name: r.name, orgName: r.org_name,
    role: r.role, roles: r.roles ?? [], kbAccess: r.kb_access ?? [], groups: [],
  }
}

const PLAT_KEY = (email: string) => `rag.platformAdmin.${email}`
function cachedPlatformFlag(s: Session | null): boolean | null {
  if (!s) return false
  const v = localStorage.getItem(PLAT_KEY(s.subject))
  return v === null ? null : v === 'true'
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(load)
  // Seed from cache so a returning super-admin lands in the platform portal with no flash; the
  // probe below re-verifies in the background.
  const [isPlatformAdmin, setIsPlatformAdmin] = useState<boolean | null>(() => cachedPlatformFlag(load()))

  // Probe the platform-only endpoint once per session: 2xx ⇒ SUPER_ADMIN, 401/403 ⇒ tenant user.
  useEffect(() => {
    if (!session) { setIsPlatformAdmin(false); return }
    let cancelled = false
    if (localStorage.getItem(PLAT_KEY(session.subject)) === null) setIsPlatformAdmin(null)
    const settle = (v: boolean) => { if (!cancelled) { setIsPlatformAdmin(v); localStorage.setItem(PLAT_KEY(session.subject), String(v)) } }
    api.listTenants(session.token).then(() => settle(true)).catch(() => settle(false))
    return () => { cancelled = true }
  }, [session])

  const persist = useCallback((next: Session) => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
    localStorage.removeItem('rag.kb') // reset KB selection on identity change
    setSession(next)
  }, [])

  const login = useCallback(async (email: string, password: string) => {
    persist(toSession(await api.authLogin(email, password)))
  }, [persist])

  const signup = useCallback(async (orgName: string, name: string, email: string, password: string) => {
    persist(toSession(await api.signup(orgName, name, email, password)))
  }, [persist])

  const logout = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY)
    localStorage.removeItem('rag.kb')
    setSession(null)
  }, [])

  const value = useMemo(() => ({ session, isPlatformAdmin, login, signup, logout }), [session, isPlatformAdmin, login, signup, logout])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useAuth(): AuthCtx {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
