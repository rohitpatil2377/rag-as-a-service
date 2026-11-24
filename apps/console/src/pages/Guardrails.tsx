import { useCallback, useEffect, useMemo, useState } from 'react'
import { Icon } from '../components/icons'
import { useToast } from '../components/overlay'
import { Alert, Card, PageHeader, Skeleton } from '../components/ui'
import { ApiError, api, type GuardrailEffective, type GuardrailEntry } from '../lib/api'
import { hasRole, useAuth } from '../lib/auth'
import { useKb } from '../lib/kb'

// The Guardrails page renders the RESOLVED policy for the selected KB (platform -> org -> kb),
// showing each control's value, who owns it, and whether the viewer may edit it. A control locked
// by a higher tier — or a constitutional control — is read-only. Writes go to the viewer's own
// tier: an owner/admin sets ORG-wide (and may lock downward); a KB manager sets THIS KB only.
const TIER_LABEL: Record<string, string> = { platform: 'platform', org: 'org', kb: 'this KB' }
const CATEGORY_ORDER = ['Input', 'Grounding', 'Output', 'Sovereignty', 'Governance']

export function Guardrails() {
  const { session } = useAuth()
  const { kbId, kbs } = useKb()
  const toast = useToast()
  const [rows, setRows] = useState<GuardrailEffective[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)

  const isOrgAdmin = hasRole(session, 'owner') || hasRole(session, 'admin')
  const myTier = isOrgAdmin ? 'org' : 'kb'
  const kbName = kbs.find((k) => k.kb_id === kbId)?.name ?? kbId

  const load = useCallback(async () => {
    if (!session || !kbId) return
    try {
      setRows(await api.getGuardrailsEffective(session.token, kbId))
      setError(null)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Failed to load guardrails')
    }
  }, [session, kbId])

  useEffect(() => { void load() }, [load])

  // A control is editable by THIS viewer if it is not constitutional and not locked by a tier
  // strictly above the viewer's own tier.
  const editable = useCallback((r: GuardrailEffective): boolean => {
    if (r.constitutional) return false
    if (!r.locked_by) return true
    if (r.locked_by === 'platform') return false
    if (r.locked_by === 'org') return myTier === 'org' // an org admin may change its own org lock
    return true // locked at kb -> the kb tier (and org admin) may change it
  }, [myTier])

  const write = useCallback(async (id: string, entry: GuardrailEntry) => {
    if (!session || !kbId) return
    setBusy(id)
    try {
      const updates = { [id]: entry }
      if (myTier === 'org') await api.setOrgGuardrails(session.token, updates)
      else await api.setKbGuardrails(session.token, kbId, updates)
      toast.push('success', 'Guardrail updated')
      await load()
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Update failed')
    } finally {
      setBusy(null)
    }
  }, [session, kbId, myTier, load, toast])

  const grouped = useMemo(() => {
    const by: Record<string, GuardrailEffective[]> = {}
    for (const r of rows ?? []) (by[r.category] ??= []).push(r)
    return CATEGORY_ORDER.filter((c) => by[c]?.length).map((c) => [c, by[c]] as const)
  }, [rows])

  return (
    <div>
      <PageHeader
        eyebrow="Govern"
        title="Guardrails"
        actions={<span style={{ fontSize: 13, color: 'var(--text-muted)' }}>{kbName}</span>}
      />
      <p style={{ color: 'var(--text-muted)', maxWidth: '68ch', marginTop: -4 }}>
        The safety controls in force for this knowledge base, resolved top-down (platform → organisation → KB).
        A control locked by a higher tier, or a constitutional guarantee, is enforced and read-only.
        {isOrgAdmin
          ? ' You are editing at the organisation tier — changes apply org-wide, and locking one enforces it on every KB below.'
          : ' You are a KB manager — you may tune any control not locked above, for this KB only.'}
      </p>

      {error && <Alert kind="error">{error}</Alert>}
      {!rows && <Skeleton className="mt-4" style={{ height: 240 }} />}

      {grouped.map(([category, items]) => (
        <Card key={category} className="mt-4">
          <div style={{ fontFamily: 'var(--font-mono, monospace)', fontSize: 12, letterSpacing: '.08em',
            textTransform: 'uppercase', color: 'var(--accent, #0E5A54)', fontWeight: 600, marginBottom: 10 }}>
            {category}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {items.map((r) => {
              const canEdit = editable(r)
              const on = r.value === true || (typeof r.value === 'number' && r.value > 0)
              return (
                <div key={r.id} style={{ display: 'flex', alignItems: 'center', gap: 14,
                  padding: '11px 0', borderTop: '1px solid var(--border, #e2e5e0)' }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ fontWeight: 600 }}>{r.label}</span>
                      {r.locked_by && (
                        <span title={`Locked by the ${TIER_LABEL[r.locked_by]} tier`}
                          style={{ display: 'inline-flex', alignItems: 'center', gap: 3, fontSize: 10,
                            fontFamily: 'monospace', fontWeight: 600, padding: '2px 7px', borderRadius: 5,
                            color: '#7A3E9C', background: '#F0E7F6' }}>
                          <Icon.shield width={11} height={11} /> locked · {TIER_LABEL[r.locked_by]}
                        </span>
                      )}
                      {!r.locked_by && r.owner && (
                        <span style={{ fontSize: 10, fontFamily: 'monospace', fontWeight: 600, padding: '2px 7px',
                          borderRadius: 5, color: '#173456', background: '#E5EBF2' }}>
                          set · {TIER_LABEL[r.owner]}
                        </span>
                      )}
                      {r.constitutional && (
                        <span style={{ fontSize: 10, fontFamily: 'monospace', fontWeight: 600, padding: '2px 7px',
                          borderRadius: 5, color: '#8F1D3A', background: '#F7E2E7' }}>
                          constitutional
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: 12.5, color: 'var(--text-muted)', marginTop: 2 }}>{r.help}</div>
                  </div>

                  {r.kind === 'threshold' ? (
                    <span style={{ fontFamily: 'monospace', fontSize: 13, minWidth: 44, textAlign: 'right' }}>
                      {typeof r.value === 'number' ? r.value.toFixed(2) : String(r.value)}
                    </span>
                  ) : (
                    <button
                      disabled={!canEdit || busy === r.id}
                      onClick={() => write(r.id, { value: !on, locked: myTier === 'org' && r.locked_by === 'org' })}
                      aria-pressed={on}
                      title={canEdit ? 'Toggle' : `Enforced by the ${TIER_LABEL[r.locked_by ?? 'platform']} tier`}
                      style={{
                        width: 44, height: 24, borderRadius: 999, border: 'none', position: 'relative',
                        cursor: canEdit ? 'pointer' : 'not-allowed', flex: 'none',
                        background: on ? 'var(--accent, #2C7A54)' : 'var(--border, #c8cdc6)',
                        opacity: canEdit ? 1 : 0.55, transition: 'background .15s',
                      }}>
                      <span style={{ position: 'absolute', top: 3, left: on ? 23 : 3, width: 18, height: 18,
                        borderRadius: '50%', background: '#fff', transition: 'left .15s' }} />
                    </button>
                  )}

                  {isOrgAdmin && canEdit && r.kind === 'toggle' && (
                    <button
                      onClick={() => write(r.id, { value: on, locked: !(r.locked_by === 'org') })}
                      title={r.locked_by === 'org' ? 'Unlock for KBs below' : 'Lock this value for every KB below'}
                      style={{ background: 'none', border: '1px solid var(--border,#c8cdc6)', borderRadius: 6,
                        padding: '4px 8px', fontSize: 11, cursor: 'pointer',
                        color: r.locked_by === 'org' ? '#7A3E9C' : 'var(--text-muted)',
                        display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                      <Icon.shield width={12} height={12} /> {r.locked_by === 'org' ? 'Locked' : 'Lock'}
                    </button>
                  )}
                </div>
              )
            })}
          </div>
        </Card>
      ))}

      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 16 }}>
        Constitutional controls are the sovereign floor — always on, platform-locked, and unchangeable by design.
      </p>
    </div>
  )
}
