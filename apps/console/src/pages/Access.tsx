// Cross-KB access requests. Four role-gated sections on one page:
//  1. Request access  (everyone)  — browse the KB directory + request a clearance level.
//  2. My access       (everyone)  — the caller's own requests/grants + self-revoke.
//  3. Approval queue   (KB managers) — pending requests for the KBs they manage; approve/reject/revoke.
//  4. Org overview     (org admin/owner) — read-only who-has-what across the org.
// The grant is LIVE (not baked into the JWT) so approve/revoke take effect with no re-login.
import { useCallback, useEffect, useState } from 'react'
import { useI18n } from '../lib/i18n'
import { Icon } from '../components/icons'
import { useToast } from '../components/overlay'
import { Button, Card, EmptyState, Field, PageHeader, SegmentedControl, Skeleton, Table } from '../components/ui'
import { ApiError, api, type AccessRequest, type KbDirectoryEntry, type KbInfo } from '../lib/api'
import { hasRole, useAuth } from '../lib/auth'
import { useKb } from '../lib/kb'

// Data classification clearance (matches Team.tsx CLEARANCE_LEVELS / the backend). 0=Public..3=Restricted.
const CLEARANCE_LEVELS = [
  { value: '0', label: 'Public' },
  { value: '1', label: 'Internal' },
  { value: '2', label: 'Confidential' },
  { value: '3', label: 'Restricted' },
] as const

function clearanceLabel(level: number | null | undefined): string {
  if (level == null) return '—'
  return CLEARANCE_LEVELS.find((c) => c.value === String(level))?.label ?? String(level)
}

// Status pill — mirrors the app's other status pills (colour + icon, never colour alone).
function StatusPill({ status }: { status: AccessRequest['status'] }) {
  const map = {
    pending: { label: 'Pending', tone: '--warning', icon: <Icon.flag size={12} /> },
    approved: { label: 'Approved', tone: '--success', icon: <Icon.check size={12} /> },
    rejected: { label: 'Rejected', tone: '--danger', icon: <Icon.minus size={12} /> },
    revoked: { label: 'Revoked', tone: '--text-muted', icon: <Icon.minus size={12} /> },
  } as const
  const s = map[status] ?? map.pending
  return (
    <span className="pill" style={{ fontSize: 11, color: `var(${s.tone})`, borderColor: `color-mix(in srgb, var(${s.tone}) 35%, transparent)`, background: `color-mix(in srgb, var(${s.tone}) 12%, transparent)` }}>
      {s.icon}{s.label}
    </span>
  )
}

export function Access() {
  const { session } = useAuth()
  const { t } = useI18n()
  const toast = useToast()
  const { kbs, refresh: refreshKbs } = useKb()
  const myEmail = session?.subject ?? ''
  const isOrgAdmin = session?.role === 'owner' || hasRole(session, 'admin')

  const [directory, setDirectory] = useState<KbDirectoryEntry[] | null>(null)
  const [mine, setMine] = useState<AccessRequest[] | null>(null)
  const [managedKbs, setManagedKbs] = useState<string[]>([])
  const [queue, setQueue] = useState<AccessRequest[] | null>(null)
  const [overview, setOverview] = useState<AccessRequest[] | null>(null)
  const [personalEnabled, setPersonalEnabled] = useState(false)
  // The personal KBs I own — created by me, shareable by me.
  const myPersonalKbs = kbs.filter((k) => k.personal && (k.owner ?? '').toLowerCase() === myEmail.toLowerCase())

  // Request-access form state (per-KB, keyed by kb_id).
  const [reqFor, setReqFor] = useState<KbDirectoryEntry | null>(null)
  const [reqClearance, setReqClearance] = useState('1')
  const [reqReason, setReqReason] = useState('')
  const [reqBusy, setReqBusy] = useState(false)

  const load = useCallback(async () => {
    if (!session) return
    api.kbDirectory(session.token).then(setDirectory).catch(() => setDirectory([]))
    api.myAccess(session.token).then(setMine).catch(() => setMine([]))
    // Approval queue: for each KB the caller manages, fetch its pending/decided requests.
    api.myManagedKbs(session.token).then(async (kbIds) => {
      setManagedKbs(kbIds)
      if (kbIds.length === 0) { setQueue([]); return }
      const lists = await Promise.all(
        kbIds.map((id) => api.listAccessRequests(session.token, id).catch(() => [] as AccessRequest[])),
      )
      setQueue(lists.flat())
    }).catch(() => { setManagedKbs([]); setQueue([]) })
    // An org owner/admin may decide on ANY KB (the backend's _can_manage_kb allows it), but they
    // are not an APPOINTED manager of anything, so /me/managed-kbs is empty for them and the
    // approval queue never rendered — the pending request was visible in the read-only overview
    // with no way to act on it. The org-wide list IS their queue.
    if (isOrgAdmin) {
      api.accessOverview(session.token)
        .then((rows) => { setOverview(rows); setQueue((q) => _mergeById(q ?? [], rows)) })
        .catch(() => setOverview([]))
    }
    api.systemInfo(session.token).then((s) => setPersonalEnabled(Boolean(s.features?.personal_kb))).catch(() => setPersonalEnabled(false))
  }, [session, isOrgAdmin])

  useEffect(() => { void load() }, [load])

  async function submitRequest() {
    if (!session || !reqFor) return
    if (!reqReason.trim()) { toast.push('error', 'Please give a reason for the request.'); return }
    setReqBusy(true)
    try {
      await api.requestAccess(session.token, reqFor.kb_id, Number(reqClearance), reqReason.trim())
      toast.push('success', `Access request for ${reqFor.name} submitted.`)
      setReqFor(null); setReqReason(''); setReqClearance('1')
      void load()
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Could not submit the request.')
    } finally {
      setReqBusy(false)
    }
  }

  async function selfRevoke(r: AccessRequest) {
    if (!session) return
    if (!window.confirm(`Revoke your own access to ${r.kb_name}? You can request it again later.`)) return
    try {
      await api.revokeAccess(session.token, r.kb_id, myEmail)
      toast.push('success', `Revoked your access to ${r.kb_name}.`)
      void load()
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Could not revoke access.')
    }
  }

  return (
    <div>
      <PageHeader eyebrow={t('eyebrow.governance', 'Governance')} title={t('page.access.title', 'Access')} />
      <p className="mb-5" style={{ color: 'var(--text-muted)', maxWidth: 720 }}>
        {t('page.access.lead', 'Request access to another team’s knowledge base at a clearance level. A KB manager approves (capped at their own clearance). Grants are live — no re-login — and revocable by you or the manager.')}
      </p>

      {/* 0. My knowledge bases — create a personal KB + share it (only when the feature is enabled) */}
      {personalEnabled && (
        <PersonalKbSection
          mine={myPersonalKbs}
          onCreated={async () => { await refreshKbs(); void load() }}
          onShared={load}
        />
      )}

      {/* 1. Request access — everyone */}
      <RequestAccessSection
        directory={directory}
        onRequest={(e) => { setReqFor(e); setReqClearance('1'); setReqReason('') }}
      />

      {/* 2. My access — everyone */}
      <MyAccessSection mine={mine} onRevoke={selfRevoke} />

      {/* 3. Approval queue — KB managers, and org owner/admin for every KB */}
      {(managedKbs.length > 0 || isOrgAdmin) && (
        <ApprovalQueueSection queue={queue} myEmail={myEmail} onChanged={load} />
      )}

      {/* 4. Org overview — org admin/owner only */}
      {isOrgAdmin && <OrgOverviewSection overview={overview} />}

      {/* Request modal */}
      {reqFor && (
        <div className="fixed inset-0 flex items-center justify-center p-4" style={{ zIndex: 50, background: 'color-mix(in srgb, #0a1020 55%, transparent)' }} onClick={() => setReqFor(null)}>
          <div onClick={(e) => e.stopPropagation()} className="surface w-full" style={{ maxWidth: 460, padding: 22 }}>
            <h3 className="font-display font-semibold mb-1" style={{ fontSize: 16 }}>Request access to {reqFor.name}</h3>
            <p className="field-help mb-3">A manager of this knowledge base approves your request, at or below the clearance you ask for (and below their own).</p>
            <div className="space-y-3">
              <div>
                <span className="field-label">Clearance level requested</span>
                <SegmentedControl value={reqClearance} onChange={setReqClearance} options={CLEARANCE_LEVELS.map((c) => ({ value: c.value, label: c.label }))} />
                <p className="field-help mt-1">The highest classification you’d like to be able to see in this KB’s answers.</p>
              </div>
              <Field label="Reason" help="Why you need access — shown to the KB manager.">
                <textarea className="field-input" rows={3} value={reqReason} onChange={(e) => setReqReason(e.target.value)} placeholder="e.g. Cross-team incident review" />
              </Field>
            </div>
            <div className="flex gap-2 mt-4">
              <Button loading={reqBusy} onClick={submitRequest} icon={!reqBusy && <Icon.check size={16} />}>Submit request</Button>
              <Button variant="subtle" onClick={() => setReqFor(null)}>Cancel</Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// --- 0. My knowledge bases (personal KBs: create + share) -----------------
function PersonalKbSection(
  { mine, onCreated, onShared }:
  { mine: KbInfo[]; onCreated: () => void | Promise<void>; onShared: () => void | Promise<void> },
) {
  const { session } = useAuth()
  const toast = useToast()
  const [name, setName] = useState('')
  const [creating, setCreating] = useState(false)

  async function create() {
    if (!session || !name.trim()) return
    setCreating(true)
    try {
      const kb = await api.createPersonalKb(session.token, name.trim())
      toast.push('success', `Created your knowledge base “${kb.name}”. It’s private until you share it.`)
      setName('')
      await onCreated()
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not create the knowledge base.')
    } finally {
      setCreating(false)
    }
  }

  return (
    <Card className="mb-5">
      <div className="flex items-center gap-2 mb-1">
        <span style={{ color: 'var(--action)' }}><Icon.book size={16} /></span>
        <h2 className="font-display font-semibold" style={{ fontSize: 16 }}>My knowledge bases</h2>
      </div>
      <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
        Create your own knowledge base — it’s <strong>private to you</strong> until you share it. Share with anyone
        in your organisation at a clearance level; they read it live, and either of you can revoke.
      </p>
      <div className="flex items-end gap-2 flex-wrap mb-1">
        <div className="flex-grow" style={{ minWidth: 240 }}>
          <Field label="New knowledge base name">
            <input className="field-input" value={name} onChange={(e) => setName(e.target.value)}
              placeholder="e.g. My research notes"
              onKeyDown={(e) => { if (e.key === 'Enter') void create() }} />
          </Field>
        </div>
        <Button loading={creating} disabled={!name.trim()} onClick={create} icon={<Icon.plus size={15} />}>Create</Button>
      </div>
      {mine.length > 0 && (
        <div className="mt-4">
          <div className="eyebrow mb-2">Yours — share with a colleague</div>
          <ul className="space-y-2">
            {mine.map((k) => <PersonalKbRow key={k.kb_id} kb={k} onShared={onShared} />)}
          </ul>
        </div>
      )}
    </Card>
  )
}

function PersonalKbRow({ kb, onShared }: { kb: KbInfo; onShared: () => void | Promise<void> }) {
  const { session } = useAuth()
  const toast = useToast()
  const [email, setEmail] = useState('')
  const [clearance, setClearance] = useState('1')
  const [busy, setBusy] = useState(false)

  async function share() {
    if (!session || !email.trim()) return
    setBusy(true)
    try {
      await api.shareKb(session.token, kb.kb_id, email.trim(), Number(clearance))
      toast.push('success', `Shared “${kb.name}” with ${email.trim()} (${clearanceLabel(Number(clearance))}).`)
      setEmail('')
      await onShared()
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not share the knowledge base.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <li className="rounded-lg p-3 flex flex-wrap items-end gap-2" style={{ border: '1px solid var(--border)' }}>
      <div className="min-w-0 mr-2" style={{ flexBasis: 160 }}>
        <div className="font-semibold truncate" style={{ color: 'var(--text-heading)' }}>{kb.name}</div>
        <div className="mono text-xs truncate" style={{ color: 'var(--text-muted)' }}>{kb.kb_id}</div>
      </div>
      <div style={{ minWidth: 200 }}>
        <Field label="Share with (email)">
          <input className="field-input" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="colleague@org.in" />
        </Field>
      </div>
      <Field label="Clearance">
        <SegmentedControl value={clearance} onChange={setClearance} options={CLEARANCE_LEVELS.map((c) => ({ value: c.value, label: c.label }))} />
      </Field>
      <Button variant="secondary" loading={busy} disabled={!email.trim()} onClick={share} icon={<Icon.shield size={14} />}>Share</Button>
    </li>
  )
}

// --- 1. Request access ----------------------------------------------------
function RequestAccessSection({ directory, onRequest }: { directory: KbDirectoryEntry[] | null; onRequest: (e: KbDirectoryEntry) => void }) {
  // Only KBs the caller does NOT already hold — exclude their own KB(s) and any active grant, so
  // you never see a "Request access" row for a KB you already manage/have.
  const available = directory === null ? null : directory.filter((e) => !e.has_access)
  return (
    <Card className="mb-5">
      <h2 className="font-display font-semibold mb-1" style={{ fontSize: 16 }}>Request access</h2>
      <p className="field-help mb-4">Knowledge bases in your organisation you don’t yet hold. Request access to ask across them.</p>
      {available === null && <Skeleton style={{ height: 100 }} />}
      {available !== null && available.length === 0 && (
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No other knowledge bases available to request — you already have access to everything you can see.</p>
      )}
      {available !== null && available.length > 0 && (
        <Table>
          <thead><tr><th>Knowledge base</th><th>Status</th><th></th></tr></thead>
          <tbody>
            {available.map((e) => {
              const pending = e.request_status === 'pending'
              return (
                <tr key={e.kb_id}>
                  <td className="font-medium" style={{ color: 'var(--text-heading)' }}>{e.name}</td>
                  <td>
                    {e.request_status ? <StatusPill status={e.request_status as AccessRequest['status']} /> : <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Not requested</span>}
                    {e.granted_clearance != null && <span className="text-xs ml-2" style={{ color: 'var(--text-muted)' }}>· {clearanceLabel(e.granted_clearance)}</span>}
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    <Button size="sm" variant="subtle" disabled={pending} onClick={() => onRequest(e)} icon={<Icon.shield size={14} />}>
                      {pending ? 'Requested' : 'Request access'}
                    </Button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </Table>
      )}
    </Card>
  )
}

// --- 2. My access ---------------------------------------------------------
function MyAccessSection({ mine, onRevoke }: { mine: AccessRequest[] | null; onRevoke: (r: AccessRequest) => void }) {
  return (
    <Card className="mb-5">
      <h2 className="font-display font-semibold mb-1" style={{ fontSize: 16 }}>My access</h2>
      <p className="field-help mb-4">Your cross-KB requests and grants. Revoke an active grant to give up the access.</p>
      {mine === null && <Skeleton style={{ height: 80 }} />}
      {mine !== null && mine.length === 0 && (
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>You haven’t requested access to any other knowledge base yet.</p>
      )}
      {mine !== null && mine.length > 0 && (
        <Table>
          <thead><tr><th>Knowledge base</th><th>Requested</th><th>Granted</th><th>Status</th><th></th></tr></thead>
          <tbody>
            {mine.map((r) => (
              <tr key={r.id}>
                <td className="font-medium" style={{ color: 'var(--text-heading)' }}>{r.kb_name}</td>
                <td className="text-xs" style={{ color: 'var(--text-muted)' }}>{clearanceLabel(r.requested_clearance)}</td>
                <td className="text-xs" style={{ color: 'var(--text-muted)' }}>{clearanceLabel(r.granted_clearance)}</td>
                <td><StatusPill status={r.status} /></td>
                <td style={{ textAlign: 'right' }}>
                  {r.status === 'approved' && (
                    <button type="button" onClick={() => onRevoke(r)} title="Give up this access" className="btn btn-subtle" style={{ fontSize: 12, color: 'var(--danger)', padding: '2px 8px' }}>Revoke</button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </Table>
      )}
    </Card>
  )
}

/** Union of two request lists, newest wins on id. An org admin's overview and a manager's own
 *  queue overlap whenever they are both — showing the row twice would invite two decisions. */
function _mergeById(a: AccessRequest[], b: AccessRequest[]): AccessRequest[] {
  const byId = new Map<string, AccessRequest>()
  for (const r of [...a, ...b]) byId.set(r.id, r)
  return [...byId.values()]
}

// --- 3. Approval queue (KB managers, and org owner/admin across the org) --
function ApprovalQueueSection({ queue, myEmail, onChanged }: { queue: AccessRequest[] | null; myEmail: string; onChanged: () => void | Promise<void> }) {
  const { session } = useAuth()
  const toast = useToast()
  // Approve-clearance picker per request (defaults to the requested level).
  const [approveLevel, setApproveLevel] = useState<Record<string, string>>({})
  const [busyId, setBusyId] = useState<string | null>(null)

  const pending = (queue ?? []).filter((r) => r.status === 'pending')
  const granted = (queue ?? []).filter((r) => r.status === 'approved')

  const levelFor = (r: AccessRequest) => approveLevel[r.id] ?? String(r.requested_clearance)

  async function decide(r: AccessRequest, approve: boolean) {
    if (!session) return
    setBusyId(r.id)
    try {
      // On approve, cap at the picker's level. The backend also caps at the manager's own
      // clearance (can_grant_clearance) — if the picked level exceeds it the call 4xxs and we
      // surface the error, so the UI doesn't need the manager's clearance to stay correct.
      await api.decideAccessRequest(session.token, r.kb_id, r.id, approve, Number(levelFor(r)))
      toast.push('success', approve ? `Approved ${r.requester} for ${r.kb_name}.` : `Rejected ${r.requester}’s request.`)
      void onChanged()
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Could not record the decision.')
    } finally {
      setBusyId(null)
    }
  }

  async function revokeGrant(r: AccessRequest) {
    if (!session) return
    if (!window.confirm(`Revoke ${r.requester}’s access to ${r.kb_name}?`)) return
    setBusyId(r.id)
    try {
      await api.revokeAccess(session.token, r.kb_id, r.requester)
      toast.push('success', `Revoked ${r.requester}’s access to ${r.kb_name}.`)
      void onChanged()
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Could not revoke access.')
    } finally {
      setBusyId(null)
    }
  }

  return (
    <Card className="mb-5">
      <h2 className="font-display font-semibold mb-1" style={{ fontSize: 16 }}>Approval queue</h2>
      <p className="field-help mb-4">Access requests for the knowledge bases you manage. Approve at a clearance up to your own — the level is capped server-side.</p>
      {queue === null && <Skeleton style={{ height: 100 }} />}

      {queue !== null && (
        <>
          <h3 className="font-semibold mb-2" style={{ fontSize: 13, color: 'var(--text-heading)' }}>Pending</h3>
          {pending.length === 0 && <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>No pending requests.</p>}
          {pending.length > 0 && (
            <Table>
              <thead><tr><th>Requester</th><th>Knowledge base</th><th>Requested</th><th>Reason</th><th>Approve at</th><th></th></tr></thead>
              <tbody>
                {pending.map((r) => (
                  <tr key={r.id}>
                    <td className="mono text-xs" style={{ color: 'var(--text-muted)' }}>{r.requester}</td>
                    <td className="font-medium" style={{ color: 'var(--text-heading)' }}>{r.kb_name}</td>
                    <td className="text-xs" style={{ color: 'var(--text-muted)' }}>{clearanceLabel(r.requested_clearance)}</td>
                    <td className="text-xs" style={{ color: 'var(--text-primary)', maxWidth: 220 }}>{r.reason || '—'}</td>
                    <td>
                      <select className="field-input" style={{ height: 30, fontSize: 12, minWidth: 120 }} value={levelFor(r)}
                        onChange={(e) => setApproveLevel((m) => ({ ...m, [r.id]: e.target.value }))}>
                        {CLEARANCE_LEVELS.map((c) => <option key={c.value} value={c.value}>{c.label}</option>)}
                      </select>
                    </td>
                    <td style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                      <div className="inline-flex items-center gap-1">
                        <button type="button" disabled={busyId === r.id} onClick={() => decide(r, true)} title="Approve at the selected clearance" className="btn btn-subtle" style={{ fontSize: 12, color: 'var(--success)', padding: '2px 8px' }}>Approve</button>
                        <button type="button" disabled={busyId === r.id} onClick={() => decide(r, false)} title="Reject this request" className="btn btn-subtle" style={{ fontSize: 12, color: 'var(--danger)', padding: '2px 8px' }}>Reject</button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </Table>
          )}

          <h3 className="font-semibold mt-5 mb-2" style={{ fontSize: 13, color: 'var(--text-heading)' }}>Active grants on your KBs</h3>
          {granted.length === 0 && <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No active grants.</p>}
          {granted.length > 0 && (
            <Table>
              <thead><tr><th>Grantee</th><th>Knowledge base</th><th>Clearance</th><th></th></tr></thead>
              <tbody>
                {granted.map((r) => (
                  <tr key={r.id}>
                    <td className="mono text-xs" style={{ color: 'var(--text-muted)' }}>{r.requester}{r.requester === myEmail && <span className="text-xs"> · you</span>}</td>
                    <td className="font-medium" style={{ color: 'var(--text-heading)' }}>{r.kb_name}</td>
                    <td className="text-xs" style={{ color: 'var(--text-muted)' }}>{clearanceLabel(r.granted_clearance)}</td>
                    <td style={{ textAlign: 'right' }}>
                      <button type="button" disabled={busyId === r.id} onClick={() => revokeGrant(r)} title="Revoke this grant" className="btn btn-subtle" style={{ fontSize: 12, color: 'var(--danger)', padding: '2px 8px' }}>Revoke</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
        </>
      )}
    </Card>
  )
}

// --- 4. Org overview (org admin/owner) -----------------------------------
function OrgOverviewSection({ overview }: { overview: AccessRequest[] | null }) {
  return (
    <Card className="mb-5">
      <h2 className="font-display font-semibold mb-1" style={{ fontSize: 16 }}>Organisation access overview</h2>
      <p className="field-help mb-4">Every cross-KB grant and request across the organisation. Read-only — managers approve on their own KBs.</p>
      {overview === null && <Skeleton style={{ height: 100 }} />}
      {overview !== null && overview.length === 0 && (
        <EmptyState icon={<Icon.shield size={28} />} title="No cross-KB access yet" hint="When members request access to another team’s knowledge base, every grant and its status shows here." />
      )}
      {overview !== null && overview.length > 0 && (
        <Table>
          <thead><tr><th>Requester</th><th>Knowledge base</th><th>Requested</th><th>Granted</th><th>Status</th><th>Decided by</th></tr></thead>
          <tbody>
            {overview.map((r) => (
              <tr key={r.id}>
                <td className="mono text-xs" style={{ color: 'var(--text-muted)' }}>{r.requester}</td>
                <td className="font-medium" style={{ color: 'var(--text-heading)' }}>{r.kb_name}</td>
                <td className="text-xs" style={{ color: 'var(--text-muted)' }}>{clearanceLabel(r.requested_clearance)}</td>
                <td className="text-xs" style={{ color: 'var(--text-muted)' }}>{clearanceLabel(r.granted_clearance)}</td>
                <td><StatusPill status={r.status} /></td>
                <td className="mono text-xs" style={{ color: 'var(--text-muted)' }}>{r.decided_by || '—'}</td>
              </tr>
            ))}
          </tbody>
        </Table>
      )}
    </Card>
  )
}
