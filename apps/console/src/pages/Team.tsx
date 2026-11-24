import { useCallback, useEffect, useState } from 'react'
import { useI18n } from '../lib/i18n'
import { Icon } from '../components/icons'
import { useToast } from '../components/overlay'
import { Alert, Button, Card, Field, PageHeader, SegmentedControl, Skeleton, Table , PasswordInput } from '../components/ui'
import { ApiError, SHARED_KB_ID, api, type ApiKeyInfo, type Member } from '../lib/api'
import { hasRole, useAuth } from '../lib/auth'
import { useKb } from '../lib/kb'
import { useRoleLabels } from '../lib/roleLabels'
import { matchCustomRole, useCustomRoles } from '../lib/customRoles'
import { KbMembersCard } from '../components/KbMembersCard'

// Fine-grained capability roles an admin selects/deselects per user (matches the backend).
// Labels come from the org's custom role-labels (Settings) with the platform defaults as
// fallback — see lib/roleLabels.ts. The `value`s are fixed permission keys and never change.
const CAP_ROLE_META = [
  { value: 'reader', hint: 'Ask & read answers' },
  { value: 'ingestor', hint: 'Add documents & connectors' },
  { value: 'reviewer', hint: 'Review queue & quality' },
  { value: 'admin', hint: 'Manage everything' },
]

const KEY_SCOPES = [
  { value: 'read', label: 'Read' },
  { value: 'ingest', label: 'Ingest' },
  { value: 'admin', label: 'Admin' },
]

// Data classification clearance (banking pack): 0=Public..3=Restricted. Labels match
// rag_schemas.classification.classification_label. A grantor can never assign a level above
// their own — enforced server-side (can_grant_clearance) — so this is best-effort UI; the
// authoritative check happens on save.
const CLEARANCE_LEVELS = [
  { value: '0', label: 'Public' },
  { value: '1', label: 'Internal' },
  { value: '2', label: 'Confidential' },
  { value: '3', label: 'Restricted' },
]

export function Team() {
  const { session } = useAuth()
  const { t } = useI18n()
  const toast = useToast()
  const { kbs, kbId } = useKb()
  const ROLE_LABELS = useRoleLabels()
  const { roles: customRoles } = useCustomRoles()
  const CAP_ROLES = CAP_ROLE_META.map((r) => ({ ...r, label: ROLE_LABELS[r.value] ?? r.value }))
  const [members, setMembers] = useState<Member[] | null>(null)
  const [search, setSearch] = useState('')
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [roles, setRoles] = useState<string[]>(['reader'])
  const [kbAll, setKbAll] = useState(true)
  const [kbSel, setKbSel] = useState<string[]>([])
  // '0'..'3' (see CLEARANCE_LEVELS). Only settable on edit — the add-member endpoint doesn't
  // accept clearance yet, so new members start at Public and are elevated via a follow-up edit.
  const [clearance, setClearance] = useState('0')
  const [editing, setEditing] = useState<string | null>(null)  // email being edited, or null = add mode
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [keys, setKeys] = useState<ApiKeyInfo[] | null>(null)
  const [keyName, setKeyName] = useState('')
  const [keyScope, setKeyScope] = useState('read')
  const [keyBusy, setKeyBusy] = useState(false)
  const [newKey, setNewKey] = useState<string | null>(null)
  const [inviteLink, setInviteLink] = useState<{ email: string; link: string } | null>(null)

  // Backend gates member management on `admin`. Check both the coarse `role` label
  // (owner/admin) and the fine-grained `roles` capability array — the latter is what the API
  // actually enforces, so relying on `role` alone could show a mismatched UI state.
  const canManage = session?.role === 'owner' || session?.role === 'admin' || hasRole(session, 'admin')

  const load = useCallback(async () => {
    if (!session) return
    try {
      setMembers(await api.listMembers(session.token))
    } catch {
      setMembers([])
    }
    api.listApiKeys(session.token).then(setKeys).catch(() => setKeys([]))
  }, [session])

  useEffect(() => { void load() }, [load])

  async function createKey() {
    if (!session || !keyName.trim()) return
    setKeyBusy(true); setError(null)
    try {
      const k = await api.createApiKey(session.token, keyName.trim(), keyScope as 'read' | 'ingest' | 'admin')
      setNewKey(k.key)
      setKeyName('')
      void load()
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Could not create key')
    } finally {
      setKeyBusy(false)
    }
  }

  async function revokeKey(k: ApiKeyInfo) {
    if (!session || !window.confirm(`Revoke "${k.name}"? Any integration using it stops working immediately.`)) return
    try {
      await api.revokeApiKey(session.token, k.key_id)
      setKeys((ks) => (ks ?? []).filter((x) => x.key_id !== k.key_id))
      toast.push('info', `Revoked "${k.name}".`)
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Revoke failed')
    }
  }

  async function activate(m: Member) {
    if (!session) return
    try {
      const r = await api.activateMember(session.token, m.email)
      if (r.email_sent) toast.push('success', `Activation link emailed to ${m.email}.`)
      else setInviteLink({ email: m.email, link: r.link.startsWith('http') ? r.link : window.location.origin + r.link })
      void load()
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Could not send the activation link')
    }
  }

  async function revoke(m: Member) {
    if (!session) return
    if (!window.confirm(`Remove ${m.name || m.email} (${m.email}) from the organisation? This permanently deletes their account. Their documents/KBs are unaffected. Consider Suspend if you only need to pause access.`)) return
    try {
      await api.revokeMember(session.token, m.email)
      toast.push('success', `Removed ${m.email}`)
      void load()
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Could not remove member')
    }
  }

  async function changeStatus(m: Member, status: 'active' | 'suspended') {
    if (!session) return
    try {
      await api.setMemberStatus(session.token, m.email, status)
      toast.push('success', status === 'suspended' ? `Suspended ${m.email}` : `Reactivated ${m.email}`)
      void load()
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Could not change status')
    }
  }

  const toggleRole = (r: string) => setRoles((rs) => (rs.includes(r) ? rs.filter((x) => x !== r) : [...rs, r]))
  const toggleKb = (id: string) => setKbSel((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]))
  const resetForm = () => {
    setEditing(null); setName(''); setEmail(''); setPassword('')
    setRoles(['reader']); setKbAll(true); setKbSel([]); setClearance('0'); setError(null)
  }

  function startEdit(m: Member) {
    setEditing(m.email)
    setName(m.name); setEmail(m.email); setPassword('')
    setRoles(m.roles.filter((r) => r !== 'owner'))
    setKbAll(m.kb_access.length === 0); setKbSel(m.kb_access)
    // The members list doesn't return the current clearance (backend gap), so this can't be
    // pre-filled — it starts at Public and must be set explicitly to change it.
    setClearance(String(m.clearance ?? 0))
    setError(null)
  }

  async function add() {
    if (!session || roles.length === 0) return setError('Select at least one role.')
    const kb_access = kbAll ? [] : kbSel
    setBusy(true); setError(null)
    try {
      if (editing) {
        await api.updateMember(session.token, editing, { roles, kb_access, clearance: Number(clearance) })
        toast.push('success', `Updated ${editing}`)
      } else {
        if (!name.trim() || !email.trim()) { setBusy(false); return setError('Name and email are required.') }
        // Same rule as the sign-in form — an invalid email means the member can never log in.
        if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim())) { setBusy(false); return setError('Enter a valid email address (e.g. you@org.in).') }
        if (password && password.length < 8) { setBusy(false); return setError('A temporary password must be at least 8 characters (or leave it blank).') }
        await api.addMember(session.token, { name: name.trim(), email: email.trim(), password: password || undefined, roles, kb_access })
        toast.push('success', `Invited ${email.trim()} — click Activate to send their access link.`)
      }
      resetForm()
      void load()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not save member')
    } finally {
      setBusy(false)
    }
  }

  const kbName = kbs.find((k) => k.kb_id === kbId)?.name ?? kbId
  // A KB manager (not org admin/owner) manages only THEIR KB's members — the org roster below is
  // owner/admin only. From the Team nav they get the KB-scoped members panel instead.
  if (!canManage) {
    return (
      <div>
        <PageHeader eyebrow={t('eyebrow.organisation')} title={t('page.team.title')}
          lead={`Members of ${kbName} — the knowledge base you manage.`} />
        <KbMembersCard kbId={kbId} />
      </div>
    )
  }

  return (
    <div>
      <PageHeader eyebrow={t('eyebrow.organisation')} title={t('page.team.title')}
        lead={t('page.team.lead', { name: session?.orgName ?? 'your organisation' })} />

      <div className="grid lg:grid-cols-3 gap-5">
        <Card className="lg:col-span-2">
          <div className="flex items-center justify-between gap-3 mb-4 flex-wrap">
            <h2 className="font-display font-semibold" style={{ fontSize: 16 }}>{t('team.card.members')}</h2>
            {members !== null && members.length > 0 && (
              <input className="field-input" placeholder="Search name / email / role" value={search} onChange={(e) => setSearch(e.target.value)} style={{ height: 32, fontSize: 13, minWidth: 220, maxWidth: 280 }} />
            )}
          </div>
          {members === null && <Skeleton style={{ height: 120 }} />}
          {members !== null && (() => {
            const q = search.trim().toLowerCase()
            const ROLE_ORDER = ['owner', 'admin', 'reviewer', 'ingestor', 'reader']
            const filtered = members.filter((m) => !q || m.name.toLowerCase().includes(q) || m.email.toLowerCase().includes(q) || (m.roles ?? []).some((r) => (ROLE_LABELS[r] ?? r).toLowerCase().includes(q)))
            return (
              <>
                <Table>
                  <thead><tr><th>Name</th><th>Email</th><th>Role</th><th>KB access</th><th>Status</th>{canManage && <th></th>}</tr></thead>
                  <tbody>
                    {filtered.map((m) => {
                      const roleList = m.roles.length ? m.roles : ['reader']
                      const primary = ROLE_ORDER.find((r) => roleList.includes(r)) ?? roleList[0]
                      const extra = roleList.filter((r) => r !== primary).length
                      // If the member's capabilities exactly match a defined custom role, show that name.
                      const custom = m.role === 'owner' ? null : matchCustomRole(roleList, customRoles)
                      return (
                      <tr key={m.email}>
                        <td className="font-medium" style={{ color: 'var(--text-heading)' }}>{m.name}{m.email === session?.subject && <span className="mono text-xs" style={{ color: 'var(--text-muted)' }}> · you</span>}</td>
                        <td className="mono text-xs" style={{ color: 'var(--text-muted)' }}>{m.email}</td>
                        <td>
                          {custom ? (
                            <span className="pill" title={roleList.map((r) => ROLE_LABELS[r] ?? r).join(', ')} style={{ fontSize: 11, color: 'var(--action)', borderColor: 'color-mix(in srgb, var(--action) 40%, transparent)', background: 'color-mix(in srgb, var(--action) 8%, transparent)' }}>{custom.label}</span>
                          ) : (
                            <span className="pill" title={roleList.map((r) => ROLE_LABELS[r] ?? r).join(', ')} style={{ fontSize: 11, color: primary === 'owner' || primary === 'admin' ? 'var(--action)' : 'var(--text-primary)', borderColor: 'var(--border)', background: 'var(--surface-2)' }}>{ROLE_LABELS[primary] ?? primary}{extra > 0 ? ` +${extra}` : ''}</span>
                          )}
                        </td>
                        <td className="text-xs" style={{ color: 'var(--text-muted)' }}>{m.kb_access.length === 0 ? 'All' : `${m.kb_access.length} KB${m.kb_access.length === 1 ? '' : 's'}`}</td>
                        <td><StatusBadge status={m.status} /></td>
                        {canManage && (
                          <td style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                            {m.role !== 'owner' && (
                              <div className="inline-flex items-center gap-1">
                                {(m.status === 'invited' || m.status === 'suspended') && (
                                  <button type="button" onClick={() => changeStatus(m, 'active')} title={m.status === 'invited' ? 'Activate now — no email link needed (member must already have a password)' : 'Reactivate access'} className="btn btn-subtle" style={{ fontSize: 12, color: 'var(--action)', padding: '2px 8px' }}>{m.status === 'invited' ? 'Activate' : 'Reactivate'}</button>
                                )}
                                {m.status !== 'active' && (
                                  <button type="button" onClick={() => activate(m)} title="Send an activation link instead" className="btn btn-subtle" style={{ fontSize: 12, color: 'var(--text-muted)', padding: '2px 8px' }}>{m.status === 'suspended' ? 'Re-send link' : 'Send link'}</button>
                                )}
                                {m.status === 'active' && (
                                  <button type="button" onClick={() => changeStatus(m, 'suspended')} title="Suspend access" className="btn btn-subtle" style={{ fontSize: 12, color: 'var(--text-muted)', padding: '2px 8px' }}>Suspend</button>
                                )}
                                <button type="button" onClick={() => startEdit(m)} aria-label={`Edit ${m.email}`} title="Edit roles, KB access & clearance" className="btn btn-subtle inline-flex items-center gap-1" style={{ fontSize: 12, color: 'var(--text-muted)', padding: '2px 8px' }}><Icon.edit size={13} /> Edit</button>
                                {m.email !== session?.subject && (
                                  <button type="button" onClick={() => revoke(m)} title="Remove from organisation" className="btn btn-subtle" style={{ fontSize: 12, color: 'var(--danger, #E8556B)', padding: '2px 8px' }}>Remove</button>
                                )}
                              </div>
                            )}
                          </td>
                        )}
                      </tr>
                      )
                    })}
                  </tbody>
                </Table>
                {filtered.length === 0 && <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>{q ? `No members match "${search}".` : 'No members yet.'}</p>}
                {q && filtered.length > 0 && <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>Showing {filtered.length} of {members.length}.</p>}
              </>
            )
          })()}
        </Card>

        <Card>
          <h2 className="font-display font-semibold mb-1" style={{ fontSize: 16 }}>{editing ? 'Edit access' : t('team.card.add_member')}</h2>
          {canManage ? (
            <>
              <p className="field-help mb-3">{editing ? `Roles & KB access for ${editing}.` : 'Creates an account in your organisation with the roles + KB access you choose.'}</p>
              <div className="space-y-3">
                {!editing && (
                  <>
                    <Field label="Name"><input className="field-input" value={name} onChange={(e) => setName(e.target.value)} /></Field>
                    <Field label="Email"><input className="field-input" type="email" value={email} onChange={(e) => setEmail(e.target.value)} /></Field>
                    <Field label="Temporary password" help="No email is configured here, so set a temp password (8+ chars) and share it — then click Activate on the member's row for instant sign-in. Leave blank only if you'll hand them the activation link yourself."><PasswordInput value={password} onChange={(e) => setPassword(e.target.value)} /></Field>
                  </>
                )}
                <div>
                  <span className="field-label">Roles</span>
                  {customRoles.length > 0 && (
                    <div className="mt-1 mb-2">
                      <select className="field-input" style={{ height: 32, fontSize: 13 }} value=""
                        onChange={(e) => { const r = customRoles.find((c) => c.id === e.target.value); if (r) setRoles([...r.caps]) }}>
                        <option value="">Apply a custom role…</option>
                        {customRoles.map((r) => <option key={r.id} value={r.id}>{r.label} ({r.caps.map((c) => ROLE_LABELS[c] ?? c).join(' · ')})</option>)}
                      </select>
                      <p className="field-help mt-1" style={{ fontSize: 11 }}>Picks the capabilities below — adjust them freely after.</p>
                    </div>
                  )}
                  <div className="space-y-1.5 mt-1">
                    {CAP_ROLES.map((r) => (
                      <label key={r.value} className="flex items-start gap-2 text-sm" style={{ cursor: 'pointer' }}>
                        <input type="checkbox" checked={roles.includes(r.value)} onChange={() => toggleRole(r.value)} className="mt-1" />
                        <span><span className="font-semibold" style={{ color: 'var(--text-heading)' }}>{r.label}</span> <span style={{ color: 'var(--text-muted)' }}>· {r.hint}</span></span>
                      </label>
                    ))}
                  </div>
                </div>
                <div>
                  <span className="field-label">Knowledge-base access</span>
                  <SegmentedControl value={kbAll ? 'all' : 'some'} onChange={(v) => setKbAll(v === 'all')} options={[{ value: 'all', label: 'All KBs' }, { value: 'some', label: 'Specific' }]} />
                  {!kbAll && (
                    <div className="mt-2 space-y-1 max-h-40 overflow-y-auto pr-1">
                      {/* The company-wide shared KB is accessible to everyone by design, so it's
                          NOT a grantable per-member access — exclude it from the picker. */}
                      {kbs.filter((k) => k.kb_id !== SHARED_KB_ID).length === 0 && <p className="text-xs" style={{ color: 'var(--text-muted)' }}>No knowledge bases yet.</p>}
                      {kbs.filter((k) => k.kb_id !== SHARED_KB_ID).map((k) => (
                        <label key={k.kb_id} className="flex items-center gap-2 text-sm" style={{ cursor: 'pointer' }}>
                          <input type="checkbox" checked={kbSel.includes(k.kb_id)} onChange={() => toggleKb(k.kb_id)} />
                          <span style={{ color: 'var(--text-heading)' }}>{k.name}</span>
                        </label>
                      ))}
                    </div>
                  )}
                </div>
                <div>
                  <span className="field-label">Data clearance</span>
                  <SegmentedControl value={clearance} onChange={setClearance} options={CLEARANCE_LEVELS} />
                  <p className="field-help mt-1">Highest classification this member may see in retrieval — and the highest an <strong>ingestor</strong> may classify a document at. You can only grant at or below your own clearance (enforced server-side).</p>
                </div>
              </div>
              <div className="flex gap-2 mt-4">
                <Button loading={busy} disabled={roles.length === 0} onClick={add} icon={!busy && <Icon.plus size={16} />}>{editing ? 'Save' : 'Add member'}</Button>
                {editing && <Button variant="subtle" onClick={resetForm}>Cancel</Button>}
              </div>
              {error && <div className="mt-3"><Alert kind="error">{error}</Alert></div>}
            </>
          ) : (
            <Alert kind="info">Only an owner or admin can manage members.</Alert>
          )}
        </Card>
      </div>

      {/* API keys (FR-I4) */}
      <div className="grid lg:grid-cols-3 gap-5 mt-5">
        <Card className="lg:col-span-2">
          <h2 className="font-display font-semibold mb-1" style={{ fontSize: 16 }}>{t('team.card.api_keys')}</h2>
          <p className="field-help mb-4">Service-account keys for programmatic access (SDKs, scripts, agents). Scoped: <span className="mono">read</span> answers only, <span className="mono">ingest</span> adds write, <span className="mono">admin</span> full. Sent as <span className="mono">Authorization: Bearer ek_…</span>.</p>
          {keys === null && <Skeleton style={{ height: 80 }} />}
          {keys !== null && keys.length === 0 && <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No keys yet.</p>}
          {keys !== null && keys.length > 0 && (
            <Table>
              <thead><tr><th>Name</th><th>Key</th><th>Scope</th><th>Created</th><th></th></tr></thead>
              <tbody>
                {keys.map((k) => (
                  <tr key={k.key_id}>
                    <td className="font-medium" style={{ color: 'var(--text-heading)' }}>{k.name}</td>
                    <td className="mono text-xs" style={{ color: 'var(--text-muted)' }}>ek_{k.prefix}_…</td>
                    <td><span className="pill" style={{ color: 'var(--text-primary)', borderColor: 'var(--border)', background: 'var(--surface-2)' }}>{k.scope}</span></td>
                    <td className="mono num text-xs" style={{ color: 'var(--text-muted)' }}>{new Date(k.created_at).toLocaleDateString()}</td>
                    <td style={{ textAlign: 'right' }}>
                      {canManage && <button type="button" onClick={() => revokeKey(k)} aria-label={`Revoke ${k.name}`} title="Revoke" className="btn btn-subtle btn-icon" style={{ color: 'var(--danger)' }}><Icon.trash size={15} /></button>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
        </Card>

        <Card>
          <h2 className="font-display font-semibold mb-1" style={{ fontSize: 16 }}>{t('team.card.create_key')}</h2>
          {canManage ? (
            <>
              <p className="field-help mb-3">The full key is shown once — copy it now.</p>
              <div className="space-y-3">
                <Field label="Name"><input className="field-input" value={keyName} onChange={(e) => setKeyName(e.target.value)} placeholder="ci-pipeline" /></Field>
                <div><span className="field-label">Scope</span><SegmentedControl value={keyScope} onChange={setKeyScope} options={KEY_SCOPES} /></div>
              </div>
              <Button className="mt-4" loading={keyBusy} disabled={!keyName.trim()} onClick={createKey} icon={!keyBusy && <Icon.plus size={16} />}>Create key</Button>
              {newKey && (
                <div className="mt-4 rounded-lg p-3" style={{ background: 'color-mix(in srgb, var(--success) 8%, transparent)', border: '1px solid color-mix(in srgb, var(--success) 30%, transparent)' }}>
                  <div className="flex items-center gap-2 text-sm font-semibold mb-2" style={{ color: 'var(--success)' }}><Icon.check size={15} />Copy this now — it won't be shown again</div>
                  <div className="mono text-xs p-2 rounded" style={{ background: 'var(--surface-2)', wordBreak: 'break-all', color: 'var(--text-heading)' }}>{newKey}</div>
                  <div className="flex gap-2 mt-2">
                    <Button size="sm" variant="ghost" onClick={() => { void navigator.clipboard?.writeText(newKey); toast.push('success', 'Copied.') }} icon={<Icon.check size={14} />}>Copy</Button>
                    <Button size="sm" variant="subtle" onClick={() => setNewKey(null)}>Done</Button>
                  </div>
                </div>
              )}
            </>
          ) : (
            <Alert kind="info">Only an owner or admin can create API keys.</Alert>
          )}
        </Card>
      </div>

      {kbId && (
        <div className="mt-5">
          {/* KB members & managers (moved here from KB Settings): appoint a KB manager + manage
              that KB's users, scoped to the selected knowledge base. */}
          <KbMembersCard kbId={kbId} />
        </div>
      )}

      {inviteLink && (
        <div className="fixed inset-0 flex items-center justify-center p-4" style={{ zIndex: 50, background: 'color-mix(in srgb, #0a1020 55%, transparent)' }} onClick={() => setInviteLink(null)}>
          <div onClick={(e) => e.stopPropagation()} className="surface w-full" style={{ maxWidth: 480, padding: 22 }}>
            <h3 className="font-display font-semibold mb-1" style={{ fontSize: 16 }}>Activation link for {inviteLink.email}</h3>
            <p className="field-help mb-3">No mail server is configured, so share this single-use link with them directly. They'll set a password and sign in.</p>
            <div className="mono text-xs p-2 rounded" style={{ background: 'var(--surface-2)', wordBreak: 'break-all', color: 'var(--text-heading)' }}>{inviteLink.link}</div>
            <div className="flex gap-2 mt-3">
              <Button size="sm" onClick={() => { copyText(inviteLink.link); toast.push('success', 'Link copied.') }} icon={<Icon.copy size={14} />}>Copy link</Button>
              <Button size="sm" variant="subtle" onClick={() => setInviteLink(null)}>Done</Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function StatusBadge({ status }: { status: Member['status'] }) {
  const map = {
    active: { label: 'Active', tone: 'var(--success, #1f9d55)' },
    invited: { label: 'Invited', tone: 'var(--warning, #C58118)' },
    suspended: { label: 'Suspended', tone: 'var(--danger, #C0483B)' },
  } as const
  const s = map[status] ?? map.active
  return (
    <span className="pill" style={{ fontSize: 11, color: s.tone, borderColor: `color-mix(in srgb, ${s.tone} 35%, transparent)`, background: `color-mix(in srgb, ${s.tone} 12%, transparent)` }}>{s.label}</span>
  )
}

function copyText(text: string) {
  if (navigator.clipboard) { void navigator.clipboard.writeText(text); return }
  const ta = document.createElement('textarea')
  ta.value = text; document.body.appendChild(ta); ta.select()
  try { document.execCommand('copy') } catch { /* ignore */ }
  document.body.removeChild(ta)
}
