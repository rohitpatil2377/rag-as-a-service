// Platform → Organisations — the SUPER_ADMIN fleet console. Provision, plan, suspend/resume
// and deprovision tenant orgs. Backed by the /v1/platform/tenants endpoints (platform-admin
// only; a non-super-admin gets a clean "not authorised" state instead of a broken page).
import { useCallback, useEffect, useState } from 'react'
import { useToast } from '../components/overlay'
import { Button, Card, Field, PageHeader, PasswordInput, Skeleton } from '../components/ui'
import { api, ApiError, type TenantRecord } from '../lib/api'
import { useAuth } from '../lib/auth'

const PLANS = ['trial', 'standard', 'enterprise']

export function Organisations() {
  const { session } = useAuth()
  const toast = useToast()
  const [tenants, setTenants] = useState<TenantRecord[] | null>(null)
  const [denied, setDenied] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)

  // create form
  const [orgId, setOrgId] = useState('')
  const [name, setName] = useState('')
  const [adminEmail, setAdminEmail] = useState('')
  const [password, setPassword] = useState('')
  const [plan, setPlan] = useState('standard')
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    if (!session) return
    try {
      setTenants(await api.listTenants(session.token))
      setDenied(false)
    } catch (e) {
      if (e instanceof ApiError && (e.status === 403 || e.status === 401)) setDenied(true)
      else toast.push('error', e instanceof ApiError ? e.message : 'Could not load organisations.')
      setTenants([])
    }
  }, [session, toast])

  useEffect(() => { void load() }, [load])

  async function create(e: React.FormEvent) {
    e.preventDefault()
    if (!session) return
    setError(null)
    const slug = orgId.trim().toLowerCase()
    if (!/^[a-z0-9][a-z0-9-]{1,62}$/.test(slug)) return setError('Org ID: 2–63 chars, lowercase letters/digits/hyphens.')
    if (password.length < 8) return setError('Owner password must be at least 8 characters.')
    setCreating(true)
    try {
      await api.provisionTenant(session.token, { org_id: slug, name: name.trim(), admin_email: adminEmail.trim(), password, plan })
      toast.push('success', `Organisation "${name.trim()}" provisioned.`)
      setOrgId(''); setName(''); setAdminEmail(''); setPassword(''); setPlan('standard')
      await load()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not create the organisation.')
    } finally { setCreating(false) }
  }

  async function act(orgId: string, fn: () => Promise<unknown>, ok: string) {
    if (!session) return
    setBusy(orgId)
    try { await fn(); toast.push('success', ok); await load() }
    catch (e) { toast.push('error', e instanceof ApiError ? e.message : 'Action failed.') }
    finally { setBusy(null) }
  }

  if (denied) {
    return (
      <div>
        <PageHeader eyebrow="Platform" title="Organisations" lead="Fleet-wide tenant management." />
        <Card>
          <div style={{ padding: '32px 8px' }}>
            <span className="eyebrow">Restricted</span>
            <p className="field-help" style={{ maxWidth: '58ch', marginTop: 6 }}>
              This is the platform operator (SUPER_ADMIN) console — managing organisations across the
              whole fleet. Your account isn't a platform administrator, so it's hidden. Sign in with a
              super-admin account to manage organisations.
            </p>
          </div>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <PageHeader eyebrow="Platform" title="Organisations"
        lead="Provision and govern tenant organisations across the fleet (SUPER_ADMIN)." />

      {/* Create */}
      <Card>
        <form onSubmit={create} className="space-y-3" style={{ padding: 4 }}>
          <div style={{ fontFamily: 'var(--font-heading)', fontSize: 14, fontWeight: 600, color: 'var(--text-heading)' }}>New organisation</div>
          <p className="field-help">Creates an isolated tenant: its own org, owner account, and knowledge-base namespaces.</p>
          <div className="grid" style={{ gridTemplateColumns: 'repeat(2, minmax(0,1fr))', gap: 12 }}>
            <Field label="Org ID"><input className="field-input" placeholder="acme-corp" value={orgId} onChange={(e) => setOrgId(e.target.value)} /></Field>
            <Field label="Display name"><input className="field-input" placeholder="Acme Corporation" value={name} onChange={(e) => setName(e.target.value)} /></Field>
            <Field label="Owner email"><input className="field-input" type="email" placeholder="admin@acme.com" value={adminEmail} onChange={(e) => setAdminEmail(e.target.value)} /></Field>
            <Field label="Owner password"><PasswordInput value={password} onChange={(e) => setPassword(e.target.value)} placeholder="min 8 chars" /></Field>
            <Field label="Plan">
              <select className="field-input" value={plan} onChange={(e) => setPlan(e.target.value)}>
                {PLANS.map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
            </Field>
          </div>
          {error && <div style={{ fontSize: 12.5, color: 'var(--danger, #E8556B)' }}>{error}</div>}
          <Button type="submit" loading={creating}>Provision organisation</Button>
        </form>
      </Card>

      {/* List */}
      <Card>
        {tenants === null ? (
          <Skeleton style={{ height: 120 }} />
        ) : tenants.length === 0 ? (
          <p className="field-help" style={{ padding: 16 }}>No organisations provisioned yet. Create one above.</p>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ textAlign: 'left', color: 'var(--text-muted)', fontSize: 11, textTransform: 'uppercase', letterSpacing: '.04em' }}>
                  <th style={{ padding: '8px 10px' }}>Org</th>
                  <th style={{ padding: '8px 10px' }}>Owner</th>
                  <th style={{ padding: '8px 10px' }}>Plan</th>
                  <th style={{ padding: '8px 10px' }}>Status</th>
                  <th style={{ padding: '8px 10px' }}>Doc cap / KB</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {tenants.map((tn) => (
                  <tr key={tn.org_id} style={{ borderTop: '1px solid var(--border)' }}>
                    <td style={{ padding: '10px' }}>
                      <div style={{ fontWeight: 600, color: 'var(--text-heading)' }}>{tn.name}</div>
                      <div className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>{tn.org_id}</div>
                    </td>
                    <td style={{ padding: '10px', color: 'var(--text-muted)' }}>{tn.admin_email}</td>
                    <td style={{ padding: '10px' }}>
                      {tn.managed === false ? (
                        <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>—</span>
                      ) : (
                        <select className="field-input" style={{ height: 28, padding: '0 6px', fontSize: 12 }} value={tn.plan} disabled={busy === tn.org_id}
                          onChange={(e) => act(tn.org_id, () => api.setTenantPlan(session!.token, tn.org_id, e.target.value), `Plan set to ${e.target.value}.`)}>
                          {PLANS.map((p) => <option key={p} value={p}>{p}</option>)}
                        </select>
                      )}
                    </td>
                    <td style={{ padding: '10px' }}>
                      <span style={{ fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 999,
                        background: tn.status === 'active' ? 'color-mix(in srgb, #3FD1A8 18%, transparent)' : 'color-mix(in srgb, #E8B44C 20%, transparent)',
                        color: tn.status === 'active' ? '#1a9a76' : '#a97a1e' }}>{tn.status}</span>
                      {tn.managed === false && (
                        <span title="Exists in identity but not fleet-provisioned — self-serve signup or legacy" style={{ marginLeft: 6, fontSize: 10, fontWeight: 600, padding: '2px 6px', borderRadius: 999, background: 'color-mix(in srgb, #6C86F0 16%, transparent)', color: '#4a5fc0' }}>self-serve</span>
                      )}
                    </td>
                    <td style={{ padding: '10px', color: 'var(--text-muted)' }} className="mono">{tn.max_documents_per_kb.toLocaleString()}</td>
                    <td style={{ padding: '10px', textAlign: 'right', whiteSpace: 'nowrap' }}>
                      {tn.managed === false ? (
                        <span style={{ fontSize: 11, color: 'var(--text-muted)' }} title="Exists in identity but not fleet-provisioned; lifecycle actions need a fleet record">not in fleet</span>
                      ) : (
                        <>
                          {tn.status === 'active' ? (
                            <button type="button" disabled={busy === tn.org_id} className="btn btn-subtle" style={{ fontSize: 12, padding: '2px 8px' }}
                              onClick={() => act(tn.org_id, () => api.suspendTenant(session!.token, tn.org_id), `Suspended ${tn.org_id}.`)}>Suspend</button>
                          ) : (
                            <button type="button" disabled={busy === tn.org_id} className="btn btn-subtle" style={{ fontSize: 12, padding: '2px 8px', color: 'var(--action)' }}
                              onClick={() => act(tn.org_id, () => api.resumeTenant(session!.token, tn.org_id), `Resumed ${tn.org_id}.`)}>Resume</button>
                          )}
                          <button type="button" disabled={busy === tn.org_id} className="btn btn-subtle" style={{ fontSize: 12, padding: '2px 8px', color: 'var(--danger, #E8556B)' }}
                            onClick={() => { if (window.confirm(`Permanently deprovision "${tn.name}" (${tn.org_id})? This purges its KBs.`)) void act(tn.org_id, () => api.deprovisionTenant(session!.token, tn.org_id), `Deprovisioned ${tn.org_id}.`) }}>Delete</button>
                        </>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}
