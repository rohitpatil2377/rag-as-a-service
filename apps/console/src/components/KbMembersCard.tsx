// KB-scoped member management (KB-level isolation). Two roles use it:
//  - org owner/admin: appoint KB MANAGERS (pick org members who manage THIS KB's users) + see/add
//    the KB's members.
//  - a KB manager: see + add + remove users scoped to THIS KB only (no managers panel).
// Backend: GET/PUT /v1/kb/{id}/managers, GET/POST/DELETE /v1/kb/{id}/members. A manager never
// gains org-wide or admin powers; the org member roster (Team page) stays owner/admin-only.
import { useCallback, useEffect, useState } from 'react'
import { useToast } from './overlay'
import { Button, Card, Field, PasswordInput, Skeleton } from './ui'
import { ApiError, api, type Member } from '../lib/api'
import { hasRole, useAuth } from '../lib/auth'
import { useRoleLabels } from '../lib/roleLabels'

const CAPS = ['reader', 'ingestor', 'reviewer'] as const  // KB managers can't grant admin
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

export function KbMembersCard({ kbId }: { kbId: string }) {
  const { session } = useAuth()
  const toast = useToast()
  const ROLE_LABELS = useRoleLabels()
  const isOrgAdmin = session?.role === 'owner' || hasRole(session, 'admin')

  const [members, setMembers] = useState<Member[] | null>(null)
  const [managers, setManagers] = useState<string[] | null>(null)
  const [denied, setDenied] = useState(false)
  // add form
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [roles, setRoles] = useState<string[]>(['reader'])
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  // managers editor (org admin)
  const [mgrEmail, setMgrEmail] = useState('')

  const load = useCallback(async () => {
    if (!session) return
    try {
      setMembers(await api.listKbMembers(session.token, kbId))
      setDenied(false)
    } catch (e) {
      if (e instanceof ApiError && (e.status === 403 || e.status === 401)) setDenied(true)
      setMembers([])
    }
    api.getKbManagers(session.token, kbId).then((r) => setManagers(r.managers)).catch(() => setManagers([]))
  }, [session, kbId])
  useEffect(() => { void load() }, [load])

  async function add(e: React.FormEvent) {
    e.preventDefault()
    if (!session) return
    setErr(null)
    if (!name.trim() || !email.trim()) return setErr('Name and email are required.')
    if (!EMAIL_RE.test(email.trim())) return setErr('Enter a valid email address (e.g. you@org.in).')
    if (password && password.length < 8) return setErr('A password must be at least 8 characters (or leave blank).')
    if (roles.length === 0) return setErr('Select at least one role.')
    setBusy(true)
    try {
      await api.addKbMember(session.token, kbId, { name: name.trim(), email: email.trim(), password: password || undefined, roles })
      toast.push('success', password ? `Added ${email.trim()} (can sign in now).` : `Invited ${email.trim()}.`)
      setName(''); setEmail(''); setPassword(''); setRoles(['reader'])
      await load()
    } catch (e2) {
      setErr(e2 instanceof ApiError ? e2.message : 'Could not add member.')
    } finally { setBusy(false) }
  }

  async function remove(m: Member) {
    if (!session) return
    if (!window.confirm(`Remove ${m.name || m.email} from this KB?`)) return
    try { await api.removeKbMember(session.token, kbId, m.email); toast.push('success', `Removed ${m.email}`); await load() }
    catch (e) { toast.push('error', e instanceof ApiError ? e.message : 'Could not remove.') }
  }

  async function saveManagers(next: string[]) {
    if (!session) return
    try { const r = await api.setKbManagers(session.token, kbId, next); setManagers(r.managers); await load() }
    catch (e) { toast.push('error', e instanceof ApiError ? e.message : 'Could not update managers.') }
  }

  if (denied) return null  // not an org admin or a manager of this KB — hide entirely

  const toggleRole = (r: string) => setRoles((rs) => rs.includes(r) ? rs.filter((x) => x !== r) : [...rs, r])

  return (
    <Card>
      <div style={{ fontFamily: 'var(--font-heading)', fontSize: 15, fontWeight: 600, color: 'var(--text-heading)', marginBottom: 4 }}>KB members</div>
      <p className="field-help mb-3">Users scoped to <strong>this knowledge base only</strong>. A KB manager can add and manage them without any org-wide access.</p>

      {/* Managers (org admin only) */}
      {isOrgAdmin && (
        <div style={{ border: '1px solid var(--border)', borderRadius: 8, padding: 12, marginBottom: 14 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-heading)', marginBottom: 6 }}>Managers of this KB</div>
          <p className="field-help mb-2">These members can add & manage this KB's users — nothing else.</p>
          <div className="flex flex-wrap gap-2 mb-2">
            {(managers ?? []).length === 0 && <span className="field-help">No managers yet.</span>}
            {(managers ?? []).map((m) => (
              <span key={m} className="pill" style={{ fontSize: 12, display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                {m}
                <button type="button" onClick={() => saveManagers((managers ?? []).filter((x) => x !== m))} title="Remove manager" style={{ background: 'none', border: 'none', color: 'var(--danger, #E8556B)', cursor: 'pointer', fontSize: 14, lineHeight: 1 }}>×</button>
              </span>
            ))}
          </div>
          <div className="flex items-center gap-2">
            <input className="field-input" placeholder="member@org.in" value={mgrEmail} onChange={(e) => setMgrEmail(e.target.value)} style={{ height: 32, fontSize: 13, flex: 1 }} />
            <Button size="sm" variant="secondary" onClick={() => { const e = mgrEmail.trim().toLowerCase(); if (e && !(managers ?? []).includes(e)) { void saveManagers([...(managers ?? []), e]); setMgrEmail('') } }}>Add manager</Button>
          </div>
        </div>
      )}

      {/* Members list */}
      {members === null ? <Skeleton style={{ height: 80 }} /> : (
        <div style={{ marginBottom: 12 }}>
          {members.length === 0 ? <p className="field-help">No members in this KB yet.</p> : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead><tr style={{ textAlign: 'left', color: 'var(--text-muted)', fontSize: 11 }}>
                <th style={{ padding: '4px 8px' }}>Name</th><th style={{ padding: '4px 8px' }}>Email</th><th style={{ padding: '4px 8px' }}>Roles</th><th style={{ padding: '4px 8px' }}>Status</th><th></th>
              </tr></thead>
              <tbody>
                {members.map((m) => (
                  <tr key={m.email} style={{ borderTop: '1px solid var(--border)' }}>
                    <td style={{ padding: '6px 8px', color: 'var(--text-heading)' }}>{m.name}</td>
                    <td style={{ padding: '6px 8px', color: 'var(--text-muted)' }} className="mono text-xs">{m.email}</td>
                    <td style={{ padding: '6px 8px', color: 'var(--text-muted)' }}>{(m.roles.length ? m.roles : ['reader']).map((r) => ROLE_LABELS[r] ?? r).join(', ')}</td>
                    <td style={{ padding: '6px 8px' }}><span style={{ fontSize: 11, color: m.status === 'active' ? '#1a9a76' : 'var(--text-muted)' }}>{m.status}</span></td>
                    <td style={{ padding: '6px 8px', textAlign: 'right' }}>
                      {m.email !== session?.subject && m.role !== 'owner' && (
                        <button type="button" onClick={() => remove(m)} className="btn btn-subtle" style={{ fontSize: 11, color: 'var(--danger, #E8556B)', padding: '2px 8px' }}>Remove</button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Add member */}
      <form onSubmit={add} style={{ borderTop: '1px solid var(--border)', paddingTop: 12 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-heading)', marginBottom: 8 }}>Add a member to this KB</div>
        <div className="grid" style={{ gridTemplateColumns: 'repeat(2, minmax(0,1fr))', gap: 10 }}>
          <Field label="Name"><input className="field-input" value={name} onChange={(e) => setName(e.target.value)} /></Field>
          <Field label="Email"><input className="field-input" type="email" value={email} onChange={(e) => setEmail(e.target.value)} /></Field>
          <Field label="Password (optional)" help="Set one to let them sign in immediately."><PasswordInput value={password} onChange={(e) => setPassword(e.target.value)} placeholder="min 8 chars" /></Field>
          <div>
            <span className="field-label">Roles</span>
            <div className="mt-1 space-y-1">
              {CAPS.map((c) => (
                <label key={c} className="flex items-center gap-2 text-sm" style={{ cursor: 'pointer' }}>
                  <input type="checkbox" checked={roles.includes(c)} onChange={() => toggleRole(c)} />
                  <span style={{ color: 'var(--text-heading)' }}>{ROLE_LABELS[c] ?? c}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
        {err && <div style={{ fontSize: 12.5, color: 'var(--danger, #E8556B)', marginTop: 8 }}>{err}</div>}
        <Button type="submit" loading={busy} className="mt-3">Add member</Button>
      </form>
    </Card>
  )
}
