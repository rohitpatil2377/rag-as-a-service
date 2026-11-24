// Roles management for a KB (admins/owners only). Two panels:
//  (a) define/edit named roles — clearance level + ingest/review capabilities + description;
//  (b) assign members (email → role).
// Contract: GET/PUT /v1/kb/{id}/roles  and  GET/PUT /v1/kb/{id}/roles/assignments.
import { useCallback, useEffect, useState } from 'react'
import { Icon } from './icons'
import { useToast } from './overlay'
import { Alert, Button, Card, Field, Skeleton } from './ui'
import { ApiError, api, type KbRole, type KbRoleAssignments, type KbRoles } from '../lib/api'
import { useAuth } from '../lib/auth'

// Clearance 0..3 → human label. Ordered so the index IS the numeric level.
const CLEARANCES: { value: number; label: string }[] = [
  { value: 0, label: 'Public' },
  { value: 1, label: 'Internal' },
  { value: 2, label: 'Confidential' },
  { value: 3, label: 'Restricted' },
]
const clearanceLabel = (n: number) => CLEARANCES.find((c) => c.value === n)?.label ?? `Level ${n}`
// Tone shifts warmer as clearance rises, so a Restricted role reads as more sensitive.
const CLEARANCE_TONE = ['--text-muted', '--info', '--warning', '--danger']

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

function emptyRole(): KbRole {
  return { clearance: 0, can_ingest: false, can_review: false, description: '' }
}

// A small, accessible ingest/review capability toggle (checkbox styled as a pill row).
function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <label className="flex items-center gap-2 cursor-pointer select-none" style={{ fontSize: 13 }}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span style={{ color: 'var(--text-primary)' }}>{label}</span>
    </label>
  )
}

export function RolesCard({ kbId }: { kbId: string }) {
  const { session } = useAuth()
  const toast = useToast()

  // Only KB admins/owners manage roles. This mirrors the gating used across the console
  // (Settings/Team/Grants). The card renders nothing for everyone else.
  const canManage = session?.role === 'owner' || session?.role === 'admin'

  const [roles, setRoles] = useState<KbRoles | null>(null)
  const [assignments, setAssignments] = useState<KbRoleAssignments | null>(null)
  const [savingRoles, setSavingRoles] = useState(false)
  const [savingAssign, setSavingAssign] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)

  // Draft for a brand-new role.
  const [newRoleName, setNewRoleName] = useState('')
  // Draft for a new assignment.
  const [assignEmail, setAssignEmail] = useState('')
  const [assignRole, setAssignRole] = useState('')

  const load = useCallback(async () => {
    if (!session || !kbId) return
    setLoadError(null)
    setRoles(null); setAssignments(null)
    try {
      const [r, a] = await Promise.all([
        api.getKbRoles(session.token, kbId),
        api.getKbRoleAssignments(session.token, kbId),
      ])
      setRoles(r ?? {})
      setAssignments(a ?? {})
    } catch (e) {
      setLoadError(e instanceof ApiError ? e.message : 'Could not load roles.')
      setRoles({}); setAssignments({})
    }
  }, [session, kbId])

  useEffect(() => { if (canManage) void load() }, [canManage, load])

  if (!canManage) return null

  const roleNames = Object.keys(roles ?? {})

  // --- Role edits (kept local; persisted on Save) ---
  function patchRole(name: string, patch: Partial<KbRole>) {
    setRoles((r) => (r ? { ...r, [name]: { ...r[name], ...patch } } : r))
  }
  function addRole() {
    const name = newRoleName.trim()
    if (!name) return
    if (roles && name in roles) { toast.push('error', `A role named "${name}" already exists.`); return }
    setRoles((r) => ({ ...(r ?? {}), [name]: emptyRole() }))
    setNewRoleName('')
  }
  function removeRole(name: string) {
    setRoles((r) => {
      if (!r) return r
      const next = { ...r }
      delete next[name]
      return next
    })
    // Also drop any assignments that pointed at the removed role.
    setAssignments((a) => {
      if (!a) return a
      const next: KbRoleAssignments = {}
      for (const [email, role] of Object.entries(a)) if (role !== name) next[email] = role
      return next
    })
  }

  async function saveRoles() {
    if (!session || !roles) return
    setSavingRoles(true)
    try {
      const saved = await api.setKbRoles(session.token, kbId, roles)
      setRoles(saved ?? roles)
      toast.push('success', 'Roles saved.')
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not save roles.')
    } finally {
      setSavingRoles(false)
    }
  }

  // --- Assignment edits ---
  function addAssignment() {
    const email = assignEmail.trim().toLowerCase()
    if (!EMAIL_RE.test(email)) { toast.push('error', 'Enter a valid email address.'); return }
    if (!assignRole) { toast.push('error', 'Pick a role for this member.'); return }
    setAssignments((a) => ({ ...(a ?? {}), [email]: assignRole }))
    setAssignEmail(''); setAssignRole('')
  }
  function removeAssignment(email: string) {
    setAssignments((a) => {
      if (!a) return a
      const next = { ...a }
      delete next[email]
      return next
    })
  }

  async function saveAssignments() {
    if (!session || !assignments) return
    setSavingAssign(true)
    try {
      const saved = await api.setKbRoleAssignments(session.token, kbId, assignments)
      setAssignments(saved ?? assignments)
      toast.push('success', 'Member assignments saved.')
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not save assignments.')
    } finally {
      setSavingAssign(false)
    }
  }

  const loading = roles === null || assignments === null

  return (
    <div className="grid md:grid-cols-2 gap-4">
      {/* --- Panel (a): define / edit named roles --- */}
      <Card>
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2">
            <Icon.shield size={16} style={{ color: 'var(--text-muted)' }} />
            <h2 className="font-display font-semibold" style={{ fontSize: 16 }}>Roles</h2>
          </div>
          {!loading && (
            <Button size="sm" loading={savingRoles} onClick={saveRoles} icon={!savingRoles && <Icon.check size={15} />}>
              Save roles
            </Button>
          )}
        </div>
        <p className="field-help mb-3">
          Named roles for this knowledge base. Each sets a clearance level (Public → Restricted) and whether
          holders can ingest documents or review answers.
        </p>

        {loadError && <div className="mb-3"><Alert kind="error">{loadError}</Alert></div>}
        {loading && <Skeleton style={{ height: 160 }} />}

        {!loading && roleNames.length === 0 && (
          <p className="field-help mb-3">No roles yet — add one below.</p>
        )}

        {!loading && roleNames.length > 0 && (
          <div className="space-y-3">
            {roleNames.map((name) => {
              const role = roles![name]
              return (
                <div key={name} className="rounded-lg p-3" style={{ border: '1px solid var(--border)', background: 'var(--surface-2)' }}>
                  <div className="flex items-center justify-between gap-2 mb-2.5">
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="font-semibold truncate" style={{ color: 'var(--text-heading)', fontSize: 14 }}>{name}</span>
                      <span className="pill" style={{ fontSize: 10, color: `var(${CLEARANCE_TONE[role.clearance] ?? '--text-muted'})`, borderColor: 'var(--border)', background: 'transparent' }}>
                        {clearanceLabel(role.clearance)}
                      </span>
                    </div>
                    <button type="button" onClick={() => removeRole(name)} aria-label={`Remove role ${name}`}
                      title="Remove role" className="btn btn-subtle btn-icon" style={{ color: 'var(--danger)' }}>
                      <Icon.trash size={14} />
                    </button>
                  </div>
                  <div className="grid grid-cols-2 gap-2.5 mb-2.5">
                    <label className="text-xs" style={{ color: 'var(--text-muted)' }}>Clearance
                      <select className="field-input mt-1" value={role.clearance}
                        onChange={(e) => patchRole(name, { clearance: Number(e.target.value) })}
                        style={{ minHeight: 38, fontSize: 13 }}>
                        {CLEARANCES.map((c) => <option key={c.value} value={c.value}>{c.label}</option>)}
                      </select>
                    </label>
                    <div className="flex flex-col justify-center gap-2 pt-4">
                      <Toggle label="Can ingest" checked={role.can_ingest} onChange={(v) => patchRole(name, { can_ingest: v })} />
                      <Toggle label="Can review" checked={role.can_review} onChange={(v) => patchRole(name, { can_review: v })} />
                    </div>
                  </div>
                  <input className="field-input" value={role.description} placeholder="Description (optional)"
                    onChange={(e) => patchRole(name, { description: e.target.value })}
                    style={{ minHeight: 38, fontSize: 13 }} />
                </div>
              )
            })}
          </div>
        )}

        {!loading && (
          <div className="mt-3 pt-3 flex items-end gap-2" style={{ borderTop: '1px solid var(--border)' }}>
            <Field label="New role name">
              <input className="field-input" value={newRoleName} onChange={(e) => setNewRoleName(e.target.value)}
                placeholder="e.g. Analyst" onKeyDown={(e) => e.key === 'Enter' && addRole()}
                style={{ minHeight: 38, fontSize: 13 }} />
            </Field>
            <Button size="sm" variant="secondary" onClick={addRole} disabled={!newRoleName.trim()} icon={<Icon.plus size={15} />}>
              Add
            </Button>
          </div>
        )}
      </Card>

      {/* --- Panel (b): assign members --- */}
      <Card>
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2">
            <Icon.shield size={16} style={{ color: 'var(--text-muted)' }} />
            <h2 className="font-display font-semibold" style={{ fontSize: 16 }}>Members</h2>
          </div>
          {!loading && (
            <Button size="sm" loading={savingAssign} onClick={saveAssignments} icon={!savingAssign && <Icon.check size={15} />}>
              Save members
            </Button>
          )}
        </div>
        <p className="field-help mb-3">Assign a member (by email) to one of the roles defined for this knowledge base.</p>

        {loading && <Skeleton style={{ height: 160 }} />}

        {!loading && Object.keys(assignments!).length === 0 && (
          <p className="field-help mb-3">No members assigned yet.</p>
        )}

        {!loading && Object.keys(assignments!).length > 0 && (
          <div className="space-y-1.5">
            {Object.entries(assignments!).map(([email, role]) => (
              <div key={email} className="flex items-center gap-2 rounded-md px-2.5 py-2" style={{ border: '1px solid var(--border)' }}>
                <span className="flex-grow truncate" style={{ color: 'var(--text-heading)', fontSize: 13 }} title={email}>{email}</span>
                <select className="field-input" value={roleNames.includes(role) ? role : ''}
                  onChange={(e) => setAssignments((a) => ({ ...(a ?? {}), [email]: e.target.value }))}
                  style={{ minHeight: 34, maxWidth: 160, fontSize: 13 }}>
                  {!roleNames.includes(role) && <option value="" disabled>{role || 'Select role'}</option>}
                  {roleNames.map((n) => <option key={n} value={n}>{n}</option>)}
                </select>
                <button type="button" onClick={() => removeAssignment(email)} aria-label={`Remove ${email}`}
                  title="Remove member" className="btn btn-subtle btn-icon" style={{ color: 'var(--danger)' }}>
                  <Icon.x size={15} />
                </button>
              </div>
            ))}
          </div>
        )}

        {!loading && (
          <div className="mt-3 pt-3 space-y-2.5" style={{ borderTop: '1px solid var(--border)' }}>
            <Field label="Add member">
              <input className="field-input" type="email" value={assignEmail} onChange={(e) => setAssignEmail(e.target.value)}
                placeholder="name@org.in" onKeyDown={(e) => e.key === 'Enter' && addAssignment()}
                style={{ minHeight: 38, fontSize: 13 }} />
            </Field>
            <div className="flex items-end gap-2">
              <label className="text-xs flex-grow" style={{ color: 'var(--text-muted)' }}>Role
                <select className="field-input mt-1" value={assignRole} onChange={(e) => setAssignRole(e.target.value)}
                  disabled={roleNames.length === 0} style={{ minHeight: 38, fontSize: 13 }}>
                  <option value="" disabled>{roleNames.length === 0 ? 'Define a role first' : 'Select role'}</option>
                  {roleNames.map((n) => <option key={n} value={n}>{n}</option>)}
                </select>
              </label>
              <Button size="sm" variant="secondary" onClick={addAssignment}
                disabled={!assignEmail.trim() || !assignRole} icon={<Icon.plus size={15} />}>
                Add
              </Button>
            </div>
          </div>
        )}
      </Card>
    </div>
  )
}
