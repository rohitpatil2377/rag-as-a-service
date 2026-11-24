// Federation grants (Federated RAG). Two lanes: Issued (grants this department gave other
// departments) and Received (grants other departments gave this one). An owner/admin can
// issue a new grant and revoke any active one they issued. The feature is flag-gated on the
// backend — a 403 with code=forbidden means the operator must set RAG_FEDERATION_ENABLED.
import { useCallback, useEffect, useState } from 'react'
import { useI18n } from '../lib/i18n'
import { Icon } from '../components/icons'
import { useToast } from '../components/overlay'
import { Alert, Button, Card, EmptyState, Field, PageHeader, SegmentedControl, Skeleton, Table } from '../components/ui'
import { ApiError, api, type CreateGrantBody, type DelegateGrantBody, type GrantView } from '../lib/api'
import { hasRole, useAuth } from '../lib/auth'
import { useKb } from '../lib/kb'

type Lane = 'issued' | 'received'

// A 403 whose code is `forbidden` is the federation-disabled flag response.
function isFederationDisabled(err: unknown): boolean {
  return err instanceof ApiError && err.status === 403 && err.code === 'forbidden'
}

const isActive = (g: GrantView): boolean => Boolean(g.active)

// active = green; revoked/expired = muted. Consistent with the other status pills.
function GrantStatus({ grant }: { grant: GrantView }) {
  const state = grant.active
    ? { label: 'Active', tone: '--success' as const, icon: <Icon.check size={13} /> }
    : grant.revoked_at
      ? { label: 'Revoked', tone: '--text-muted' as const, icon: <Icon.minus size={13} /> }
      : { label: 'Expired', tone: '--text-muted' as const, icon: <Icon.minus size={13} /> }
  return (
    <span className="pill" style={{ fontSize: 11, color: `var(${state.tone})`, borderColor: `color-mix(in srgb, var(${state.tone}) 35%, transparent)`, background: `color-mix(in srgb, var(${state.tone}) 12%, transparent)` }}>
      {state.icon}{state.label}
    </span>
  )
}

function Tags({ items }: { items: string[] }) {
  if (items.length === 0) return <span className="text-xs" style={{ color: 'var(--text-muted)' }}>—</span>
  return (
    <div className="flex flex-wrap gap-1">
      {items.map((it) => (
        <span key={it} className="pill" style={{ fontSize: 11, color: 'var(--text-primary)', borderColor: 'var(--border)', background: 'var(--surface-2)' }}>{it}</span>
      ))}
    </div>
  )
}

const fmtDate = (iso: string) => { try { return new Date(iso).toLocaleDateString() } catch { return iso } }

export function Grants() {
  const { session } = useAuth()
  const { t } = useI18n()
  const toast = useToast()
  const { kbs } = useKb()

  const [lane, setLane] = useState<Lane>('issued')
  const [issued, setIssued] = useState<GrantView[] | null>(null)
  const [received, setReceived] = useState<GrantView[] | null>(null)
  const [disabled, setDisabled] = useState(false)  // federation flag off
  const [loadError, setLoadError] = useState<string | null>(null)

  // Create-grant form.
  const [subject, setSubject] = useState('')
  const [topics, setTopics] = useState('')
  const [kbSel, setKbSel] = useState<string[]>([])
  const [maxResults, setMaxResults] = useState('5')
  const [ttlDays, setTtlDays] = useState('180')
  const [busy, setBusy] = useState(false)
  const [formError, setFormError] = useState<string | null>(null)

  // Delegate (re-share a RECEIVED grant, narrowed within the parent).
  const [delegateFor, setDelegateFor] = useState<GrantView | null>(null)
  const [dSubject, setDSubject] = useState('')
  const [dTopics, setDTopics] = useState<string[]>([])
  const [dKbs, setDKbs] = useState<string[]>([])
  const [dMax, setDMax] = useState('5')
  const [dTtl, setDTtl] = useState('30')
  const [dBusy, setDBusy] = useState(false)
  const [dErr, setDErr] = useState<string | null>(null)

  // Backend gates issue/revoke on `admin`. Check both the coarse `role` label (owner/admin)
  // and the fine-grained `roles` capability array — the latter is what the API actually
  // enforces, so relying on `role` alone could show a mismatched UI state.
  const canManage = session?.role === 'owner' || session?.role === 'admin' || hasRole(session, 'admin')

  const load = useCallback(async () => {
    if (!session) return
    setIssued(null); setReceived(null); setLoadError(null); setDisabled(false)
    try {
      const [iss, rec] = await Promise.all([
        api.listGrants(session.token),
        api.listReceivedGrants(session.token),
      ])
      setIssued(iss); setReceived(rec)
    } catch (err) {
      if (isFederationDisabled(err)) { setDisabled(true); setIssued([]); setReceived([]); return }
      setIssued([]); setReceived([])
      setLoadError(err instanceof ApiError ? err.message : 'Could not load grants.')
    }
  }, [session])

  useEffect(() => { void load() }, [load])

  const toggleKb = (id: string) => setKbSel((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]))

  const resetForm = () => {
    setSubject(''); setTopics(''); setKbSel([]); setMaxResults('5'); setTtlDays('180'); setFormError(null)
  }

  async function create() {
    if (!session) return
    const subj = subject.trim()
    const topicList = topics.split(',').map((s) => s.trim()).filter(Boolean)
    if (!subj) return setFormError('Subject department id is required.')
    if (topicList.length === 0) return setFormError('Add at least one topic.')
    if (kbSel.length === 0) return setFormError('Select at least one knowledge base to share.')
    const body: CreateGrantBody = {
      subject_dept_id: subj,
      topics: topicList,
      kb_ids: kbSel,
      max_results_per_query: Number(maxResults) || 5,
      ttl_days: Number(ttlDays) || 180,
    }
    setBusy(true); setFormError(null)
    try {
      await api.createGrant(session.token, body)
      toast.push('success', `Granted ${subj} access to ${kbSel.length} KB${kbSel.length === 1 ? '' : 's'}.`)
      resetForm()
      void load()
    } catch (err) {
      if (isFederationDisabled(err)) { setDisabled(true); return }
      setFormError(err instanceof ApiError ? err.message : 'Could not issue the grant.')
    } finally {
      setBusy(false)
    }
  }

  async function revoke(g: GrantView) {
    if (!session || !window.confirm(`Revoke this grant to ${g.subject_dept_id}? They lose access immediately.`)) return
    try {
      await api.revokeGrant(session.token, g.grant_id)
      toast.push('info', `Revoked grant to ${g.subject_dept_id}.`)
      void load()
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Revoke failed.')
    }
  }

  function openDelegate(g: GrantView) {
    setDelegateFor(g); setDSubject(''); setDTopics(g.topics); setDKbs(g.kb_ids)
    setDMax(String(g.max_results_per_query)); setDTtl('30'); setDErr(null)
  }
  const dToggle = (arr: string[], set: (v: string[]) => void, id: string) =>
    set(arr.includes(id) ? arr.filter((x) => x !== id) : [...arr, id])

  async function delegate() {
    if (!session || !delegateFor) return
    const subj = dSubject.trim()
    if (!subj) return setDErr('Subject department id is required.')
    if (dTopics.length === 0) return setDErr('Pick at least one topic (a subset of the parent).')
    if (dKbs.length === 0) return setDErr('Pick at least one knowledge base (a subset of the parent).')
    const body: DelegateGrantBody = {
      subject_dept_id: subj, topics: dTopics, kb_ids: dKbs,
      max_results_per_query: Number(dMax) || 5, ttl_days: Number(dTtl) || 30,
    }
    setDBusy(true); setDErr(null)
    try {
      await api.delegateGrant(session.token, delegateFor.grant_id, body)
      toast.push('success', `Re-delegated to ${subj}.`)
      setDelegateFor(null); void load()
    } catch (err) {
      setDErr(err instanceof ApiError ? err.message : 'Delegation failed.')
    } finally { setDBusy(false) }
  }

  const all = lane === 'issued' ? issued : received
  // Revoking does NOT delete: a grant is a security artifact on the tamper-evident audit chain,
  // and "who ever had access to this KB?" is a question a regulator is entitled to ask. So the
  // row stays and the LIST hides it, rather than the record being destroyed to tidy a screen.
  const [showInactive, setShowInactive] = useState(false)
  const inactiveCount = (all ?? []).filter((g) => !isActive(g)).length
  const grants = all === null ? null : showInactive ? all : all.filter(isActive)

  return (
    <div>
      <PageHeader eyebrow={t('eyebrow.govern')} title={t('page.grants.title', 'Federation Grants')}
        lead={t('page.grants.lead', 'Share knowledge across departments — controlled, time-bound, and revocable.')} />

      {delegateFor && (
        <div className="mb-5">
          <Card>
            <div className="flex items-center justify-between mb-1">
              <h2 className="font-display font-semibold" style={{ fontSize: 16 }}>Re-delegate grant from <span className="mono">{delegateFor.issuer_dept_id}</span></h2>
              <button type="button" className="btn btn-subtle btn-icon" onClick={() => setDelegateFor(null)} aria-label="Close" style={{ fontSize: 18, lineHeight: 1, color: 'var(--text-muted)' }}>×</button>
            </div>
            <p className="field-help mb-3">Transitively share this received grant with another department. You can only <strong>narrow</strong> it — pick a subset of its topics and KBs; validity is capped by the parent, and revoking any ancestor revokes this too.</p>
            <div className="space-y-3">
              <Field label="Subject department id" help="The department you are re-delegating to (not yourself).">
                <input className="field-input" value={dSubject} onChange={(e) => setDSubject(e.target.value)} placeholder="dept-hr" />
              </Field>
              <div>
                <span className="field-label">Topics (subset of parent)</span>
                <div className="mt-1 flex flex-wrap gap-2">
                  {delegateFor.topics.map((tp) => (
                    <label key={tp} className="flex items-center gap-1.5 text-sm" style={{ cursor: 'pointer' }}>
                      <input type="checkbox" checked={dTopics.includes(tp)} onChange={() => dToggle(dTopics, setDTopics, tp)} />
                      <span style={{ color: 'var(--text-heading)' }}>{tp}</span>
                    </label>
                  ))}
                </div>
              </div>
              <div>
                <span className="field-label">Knowledge bases (subset of parent)</span>
                <div className="mt-1 flex flex-wrap gap-2">
                  {delegateFor.kb_ids.map((k) => (
                    <label key={k} className="flex items-center gap-1.5 text-sm" style={{ cursor: 'pointer' }}>
                      <input type="checkbox" checked={dKbs.includes(k)} onChange={() => dToggle(dKbs, setDKbs, k)} />
                      <span className="mono text-xs" style={{ color: 'var(--text-heading)' }}>{k}</span>
                    </label>
                  ))}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <Field label={`Max results / query (≤ ${delegateFor.max_results_per_query})`}>
                  <input className="field-input" type="number" min={1} max={delegateFor.max_results_per_query} value={dMax} onChange={(e) => setDMax(e.target.value)} />
                </Field>
                <Field label="Valid for (days)">
                  <input className="field-input" type="number" min={1} value={dTtl} onChange={(e) => setDTtl(e.target.value)} />
                </Field>
              </div>
            </div>
            <div className="flex gap-2 mt-4">
              <Button loading={dBusy} onClick={delegate} icon={!dBusy && <Icon.shield size={16} />}>Re-delegate</Button>
              <Button variant="subtle" onClick={() => setDelegateFor(null)}>Cancel</Button>
            </div>
            {dErr && <div className="mt-3"><Alert kind="error">{dErr}</Alert></div>}
          </Card>
        </div>
      )}

      {disabled && (
        <div className="mb-5">
          <Alert kind="info">
            <span className="font-semibold" style={{ color: 'var(--text-heading)' }}>Federation is disabled.</span>{' '}
            Cross-department grants are turned off. An operator must set <span className="mono">RAG_FEDERATION_ENABLED</span> to enable this feature.
          </Alert>
        </div>
      )}
      {loadError && !disabled && <div className="mb-5"><Alert kind="error">{loadError}</Alert></div>}

      <div className="grid lg:grid-cols-3 gap-5">
        <Card className="lg:col-span-2">
          <div className="flex items-center justify-between mb-4 gap-3 flex-wrap">
            <h2 className="font-display font-semibold" style={{ fontSize: 16 }}>{t('grants.card.title', 'Grants')}</h2>
            <div className="flex items-center gap-3 flex-wrap">
              {inactiveCount > 0 && (
                <button type="button" onClick={() => setShowInactive((v) => !v)} className="btn btn-subtle"
                  style={{ fontSize: 12, padding: '2px 10px', color: 'var(--text-muted)' }}
                  title="Revoked and expired grants are kept for the audit trail, never deleted">
                  {showInactive ? 'Hide' : 'Show'} {inactiveCount} revoked/expired
                </button>
              )}
              <SegmentedControl<Lane> value={lane} onChange={setLane}
                options={[
                  { value: 'issued', label: t('grants.tab.issued', 'Issued') },
                  { value: 'received', label: t('grants.tab.received', 'Received') },
                ]} />
            </div>
          </div>

          {grants === null && <Skeleton style={{ height: 140 }} />}

          {grants !== null && grants.length === 0 && !disabled && (
            <EmptyState icon={<Icon.shield size={30} />}
              title={lane === 'issued' ? t('grants.empty.issued.title', 'No grants issued yet') : t('grants.empty.received.title', 'No grants received')}
              hint={lane === 'issued'
                ? t('grants.empty.issued.hint', 'Issue a grant to let another department query your knowledge bases on specific topics.')
                : t('grants.empty.received.hint', 'When another department shares a knowledge base with you, it will appear here.')} />
          )}

          {grants !== null && grants.length > 0 && (
            <Table>
              <thead>
                <tr>
                  <th>{lane === 'issued' ? t('grants.col.subject', 'Subject dept') : t('grants.col.issuer', 'Issuer dept')}</th>
                  <th>{t('grants.col.topics', 'Topics')}</th>
                  <th>{t('grants.col.kbs', 'Knowledge bases')}</th>
                  {lane === 'issued' && <th className="num" style={{ textAlign: 'right' }}>{t('grants.col.max', 'Max/query')}</th>}
                  <th>{t('grants.col.expiry', 'Expires')}</th>
                  <th>{t('grants.col.status', 'Status')}</th>
                  {canManage && <th></th>}
                </tr>
              </thead>
              <tbody>
                {grants.map((g) => (
                  <tr key={g.grant_id}>
                    <td className="mono text-xs font-medium" style={{ color: 'var(--text-heading)' }}>
                      {lane === 'issued' ? g.subject_dept_id : g.issuer_dept_id}
                    </td>
                    <td><Tags items={g.topics} /></td>
                    <td><Tags items={g.kb_ids} /></td>
                    {lane === 'issued' && <td className="mono num text-xs" style={{ textAlign: 'right', color: 'var(--text-muted)' }}>{g.max_results_per_query}</td>}
                    <td className="mono num text-xs" style={{ color: 'var(--text-muted)' }}>{fmtDate(g.expires_at)}</td>
                    <td><GrantStatus grant={g} /></td>
                    {canManage && (
                      <td style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                        {g.active && lane === 'issued' && (
                          <button type="button" onClick={() => revoke(g)} aria-label="Revoke grant" title="Revoke" className="btn btn-subtle btn-icon" style={{ color: 'var(--danger)' }}>
                            <Icon.trash size={15} />
                          </button>
                        )}
                        {g.active && lane === 'received' && (
                          <button type="button" onClick={() => openDelegate(g)} title="Re-delegate to another department" className="btn btn-subtle" style={{ fontSize: 12 }}>
                            Delegate
                          </button>
                        )}
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
        </Card>

        <Card>
          <h2 className="font-display font-semibold mb-1" style={{ fontSize: 16 }}>{t('grants.card.create', 'Issue a grant')}</h2>
          {!canManage ? (
            <Alert kind="info">{t('grants.readonly', 'Only an owner or admin can issue or revoke grants.')}</Alert>
          ) : disabled ? (
            <Alert kind="info">{t('grants.disabled.form', 'Enable federation to issue grants.')}</Alert>
          ) : (
            <>
              <p className="field-help mb-3">{t('grants.create.help', 'Let another department query the selected knowledge bases on the topics you choose, until the grant expires or you revoke it.')}</p>
              <div className="space-y-3">
                <Field label={t('grants.field.subject', 'Subject department id')} help={t('grants.field.subject.help', 'The department that will receive access.')}>
                  <input className="field-input" value={subject} onChange={(e) => setSubject(e.target.value)} placeholder="dept-finance" />
                </Field>
                <Field label={t('grants.field.topics', 'Topics')} help={t('grants.field.topics.help', 'Comma-separated. Only queries on these topics are shared.')}>
                  <input className="field-input" value={topics} onChange={(e) => setTopics(e.target.value)} placeholder="budgets, procurement" />
                </Field>
                <div>
                  <span className="field-label">{t('grants.field.kbs', 'Knowledge bases')}</span>
                  {kbs.length === 0 ? (
                    <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>{t('grants.field.kbs.none', 'No knowledge bases yet.')}</p>
                  ) : (
                    <div className="mt-1 space-y-1 max-h-40 overflow-y-auto pr-1">
                      {kbs.map((k) => (
                        <label key={k.kb_id} className="flex items-center gap-2 text-sm" style={{ cursor: 'pointer' }}>
                          <input type="checkbox" checked={kbSel.includes(k.kb_id)} onChange={() => toggleKb(k.kb_id)} />
                          <span style={{ color: 'var(--text-heading)' }}>{k.name}</span>
                        </label>
                      ))}
                    </div>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <Field label={t('grants.field.max', 'Max results / query')}>
                    <input className="field-input" type="number" min={1} value={maxResults} onChange={(e) => setMaxResults(e.target.value)} />
                  </Field>
                  <Field label={t('grants.field.ttl', 'Valid for (days)')}>
                    <input className="field-input" type="number" min={1} value={ttlDays} onChange={(e) => setTtlDays(e.target.value)} />
                  </Field>
                </div>
              </div>
              <div className="flex gap-2 mt-4">
                <Button loading={busy} onClick={create} icon={!busy && <Icon.plus size={16} />}>{t('grants.action.issue', 'Issue grant')}</Button>
              </div>
              {formError && <div className="mt-3"><Alert kind="error">{formError}</Alert></div>}
            </>
          )}
        </Card>
      </div>
    </div>
  )
}
