// Shared fetch + tiny module cache for the org's CUSTOM named roles (label + capability bundle).
// Mirrors lib/roleLabels.ts. A custom role is pure sugar over the 4 fixed caps
// (reader/ingestor/reviewer/admin): assigning one sets those caps, and here we provide the
// reverse-match (a member's cap-set -> the custom role's label) used to display it.
import { useEffect, useState } from 'react'
import { api, type CustomRole } from './api'
import { useAuth } from './auth'

let cache: { tenantId: string; roles: CustomRole[] } | null = null

export function useCustomRoles(): { roles: CustomRole[]; reload: () => void } {
  const { session } = useAuth()
  const [roles, setRoles] = useState<CustomRole[]>(
    cache && cache.tenantId === session?.tenantId ? cache.roles : [],
  )
  const [nonce, setNonce] = useState(0)

  useEffect(() => {
    if (!session) return
    if (nonce === 0 && cache && cache.tenantId === session.tenantId) { setRoles(cache.roles); return }
    let cancelled = false
    api.getCustomRoles(session.token).then((r) => {
      if (cancelled) return
      cache = { tenantId: session.tenantId, roles: r.roles }
      setRoles(r.roles)
    }).catch(() => { /* keep empty — org may have none / offline */ })
    return () => { cancelled = true }
  }, [session, nonce])

  return { roles, reload: () => setNonce((n) => n + 1) }
}

export function invalidateCustomRolesCache(tenantId: string, roles: CustomRole[]): void {
  cache = { tenantId, roles }
}

// Given a member's capability set, find the custom role whose caps EXACTLY match (order-agnostic).
// Returns the matching role's label, or null when no custom role fits (caller falls back to the
// per-capability labels).
export function matchCustomRole(memberCaps: string[], custom: CustomRole[]): CustomRole | null {
  const set = new Set(memberCaps.filter((c) => c !== 'owner'))
  for (const r of custom) {
    const rc = new Set(r.caps)
    if (rc.size === set.size && [...rc].every((c) => set.has(c))) return r
  }
  return null
}
