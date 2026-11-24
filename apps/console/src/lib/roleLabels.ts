// Shared fetch for the org's custom display labels for the fixed capability roles
// (reader/ingestor/reviewer/admin) + the read-only 'owner' label. Cosmetic only — the
// permission values themselves never change, only what they're called in the UI (Settings
// lets an owner/admin customize them; see GET/PUT /v1/org/role-labels).
//
// Deliberately NOT a React context/provider: it's 4-5 strings read by a handful of pages, so
// each consumer just calls this hook and gets a fetch-on-mount + tiny module-level cache
// (shared across mounts in the same session so switching pages doesn't re-fetch every time).
import { useEffect, useState } from 'react'
import { api } from './api'
import { useAuth } from './auth'

// Same defaults as the backend's ROLE_LABEL_DEFAULTS — what GET returns before any org has
// customized anything, and what the UI shows while the real fetch is in flight / if it fails.
export const ROLE_LABEL_DEFAULTS: Record<string, string> = {
  reader: 'Associate',
  ingestor: 'Contributor',
  reviewer: 'Reviewer',
  admin: 'Manager',
  owner: 'Owner',
}

let cache: { tenantId: string; labels: Record<string, string> } | null = null

export function useRoleLabels(): Record<string, string> {
  const { session } = useAuth()
  const [labels, setLabels] = useState<Record<string, string>>(
    cache && cache.tenantId === session?.tenantId ? cache.labels : ROLE_LABEL_DEFAULTS,
  )

  useEffect(() => {
    if (!session) return
    if (cache && cache.tenantId === session.tenantId) { setLabels(cache.labels); return }
    let cancelled = false
    api.getRoleLabels(session.token).then((r) => {
      if (cancelled) return
      cache = { tenantId: session.tenantId, labels: r.labels }
      setLabels(r.labels)
    }).catch(() => { /* keep defaults — e.g. offline or not-yet-migrated org */ })
    return () => { cancelled = true }
  }, [session])

  return labels
}

// Consumers that just changed the labels (the Settings page) should call this so every other
// mounted page picks up the new values without a full reload.
export function invalidateRoleLabelsCache(tenantId: string, labels: Record<string, string>): void {
  cache = { tenantId, labels }
}
