import { useCallback, useEffect, useState } from 'react'
import { useI18n } from '../lib/i18n'
import { Icon } from '../components/icons'
import { useToast } from '../components/overlay'
import { Alert, Button, Card, Field, PageHeader, SegmentedControl } from '../components/ui'
import { Collapsible } from '../components/Collapsible'
import { ApiError, api, type DocumentSummary, type EgressRequest, type IngestResponse, type SyncResponse, type SyncScheduleView, type SyncSourceView } from '../lib/api'
import { useAuth } from '../lib/auth'
import { useKb } from '../lib/kb'
import { StructuredSources } from './Sources'

type Mode = 'web' | 'platform' | 'external' | 'sync'

const SOURCES: { icon: React.ReactNode; name: string; desc: string; note: string }[] = [
  { icon: <Icon.ask size={18} />, name: 'Web page', desc: 'Fetch a page by URL — visible text only (no scripts/menus). Host must be egress-allowlisted.', note: 'Sovereign-gated' },
  { icon: <Icon.cloud size={18} />, name: 'External S3 bucket', desc: 'Pull one object, or sync a whole folder, from your own S3-compatible bucket.', note: 'Sovereign-gated' },
  { icon: <Icon.database size={18} />, name: 'Platform object store', desc: 'Pull a document already staged in your own namespace by key.', note: 'Available' },
]

export function Connectors() {
  const { session } = useAuth()
  const { t } = useI18n()
  const { kbId, kbs } = useKb()
  const toast = useToast()
  const [mode, setMode] = useState<Mode>('web')
  const [docId, setDocId] = useState('')
  const [url, setUrl] = useState('')
  const [objectKey, setObjectKey] = useState('')
  const [prefix, setPrefix] = useState('')
  const [mime, setMime] = useState('application/pdf')
  const [endpoint, setEndpoint] = useState('')
  const [bucket, setBucket] = useState('')
  const [credRef, setCredRef] = useState('')
  const [secure, setSecure] = useState<'https' | 'http'>('https')
  const [provider, setProvider] = useState<'s3' | 'azure'>('s3')
  // The source this KB is already connected to. Shown so adding a bucket leaves visible
  // evidence, and so pointing a KB at the wrong one is noticeable rather than silent.
  const [connected, setConnected] = useState<SyncSourceView | null>(null)
  const [sched, setSched] = useState<SyncScheduleView | null>(null)
  const [docs, setDocs] = useState<DocumentSummary[] | null>(null)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<IngestResponse | null>(null)
  const [sync, setSync] = useState<SyncResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [blockedHost, setBlockedHost] = useState<string | null>(null)
  const [requesting, setRequesting] = useState(false)
  const kbName = kbs.find((k) => k.kb_id === kbId)?.name ?? kbId
  // A manager of THIS KB (or an owner/admin) can decide egress for it, so we let them allow the host
  // in one click instead of filing a request for someone else to approve.
  const [managedKbs, setManagedKbs] = useState<string[]>([])
  const isAdmin = session?.role === 'owner' || session?.role === 'admin'
  const canManageThisKb = isAdmin || managedKbs.includes(kbId)
  useEffect(() => {
    if (!session) return
    api.myManagedKbs(session.token).then(setManagedKbs).catch(() => setManagedKbs([]))
  }, [session])

  function hostOf(s: string): string {
    try { return new URL(s.includes('://') ? s : `https://${s}`).hostname } catch { return s }
  }

  async function requestApproval() {
    if (!session || !blockedHost) return
    setRequesting(true)
    try {
      await api.requestEgress(session.token, blockedHost, `Ingest into ${kbName}`, kbId)
      toast.push('success', `Approval requested for ${blockedHost}. An owner/admin will review it.`)
      setBlockedHost(null)
      // The approvals panel is a sibling component — tell it to reload so the new request
      // appears immediately instead of only after a page refresh.
      window.dispatchEvent(new CustomEvent('rag:egress-changed'))
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not submit request.')
    } finally {
      setRequesting(false)
    }
  }

  // Manager path: raise the request AND approve it (the backend authorises a KB manager to decide
  // egress for their own KB), then retry the ingest — no waiting on an owner/admin.
  async function allowAndIngest() {
    if (!session || !blockedHost) return
    setRequesting(true)
    try {
      const req = await api.requestEgress(session.token, blockedHost, `Ingest into ${kbName}`, kbId)
      await api.decideEgress(session.token, req.id, 'approve')
      toast.push('success', `${blockedHost} allowed for this KB — re-ingesting…`)
      window.dispatchEvent(new CustomEvent('rag:egress-changed'))
      setBlockedHost(null)
      await run()
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not allow the host.')
    } finally {
      setRequesting(false)
    }
  }

  const needsBucket = mode === 'external' || mode === 'sync'
  const ready =
    mode === 'web' ? Boolean(docId.trim() && /^https?:\/\//.test(url.trim()))
      : mode === 'sync' ? Boolean(endpoint.trim() && bucket.trim() && credRef.trim())
        : Boolean(docId.trim() && objectKey.trim() && (mode === 'platform' || (endpoint.trim() && bucket.trim() && credRef.trim())))

  const extSource = () => ({ endpoint: endpoint.trim(), bucket: bucket.trim(), secure: secure === 'https', credentials_ref: credRef.trim(), provider })

  const loadConnected = useCallback(async () => {
    if (!session || !kbId) return
    try {
      const v = await api.getSyncSchedule(session.token, kbId)
      setSched(v); setConnected(v.source ?? null)
    } catch { setSched(null); setConnected(null) }
    try { setDocs(await api.listDocuments(session.token, kbId)) } catch { setDocs(null) }
  }, [session, kbId])

  useEffect(() => { void loadConnected() }, [loadConnected])

  // Documents that came FROM the connected source: pulled by reference, and keyed by the object
  // key, so a synced document's id starts with the source's prefix. Counting every document in
  // the KB would credit the connector with files someone uploaded by hand.
  const fromSource = (docs ?? []).filter(
    (d) => d.source_kind === 'object_store' && (!connected?.prefix || d.doc_id.startsWith(connected.prefix)),
  )
  const newest = fromSource.reduce<string | null>(
    (acc, d) => (acc === null || d.ingested_at > acc ? d.ingested_at : acc), null,
  )

  async function run() {
    if (!session || !ready || busy) return
    setBusy(true); setError(null); setResult(null); setSync(null); setBlockedHost(null)
    try {
      if (mode === 'sync') {
        const r = await api.syncFolder(session.token, kbId, extSource(), prefix.trim())
        setSync(r)
        toast.push(r.failed ? 'info' : 'success', `Synced ${r.ingested}/${r.found} file(s)${r.failed ? `, ${r.failed} failed` : ''}`)
        void loadConnected() // the sync captures the source; show what is now connected
      } else if (mode === 'web') {
        const r = await api.ingest(session.token, kbId, { doc_id: docId.trim(), source_kind: 'web', url: url.trim() })
        setResult(r)
        toast.push('success', `Ingested ${r.doc_id} — ${r.chunk_count} chunk(s)`)
      } else {
        const r = await api.ingest(session.token, kbId, {
          doc_id: docId.trim(), mime, source_kind: 'object_store', object_key: objectKey.trim(),
          external_source: mode === 'external' ? extSource() : undefined,
        })
        setResult(r)
        toast.push('success', `Ingested ${r.doc_id} — ${r.chunk_count} chunk(s)`)
      }
    } catch (err) {
      if (err instanceof ApiError && err.code === 'sovereignty_violation') {
        setBlockedHost(hostOf(mode === 'web' ? url.trim() : endpoint.trim()))
        setError('That host is not on the sovereignty allowlist.')
      } else {
        const msg = err instanceof ApiError ? err.message : 'Ingestion failed'
        setError(msg); toast.push('error', msg)
      }
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <PageHeader eyebrow={t('eyebrow.ingest')} title={t('page.connectors.title')}
        lead={t('page.connectors.lead', { name: kbName })} />

      <div className="grid sm:grid-cols-3 gap-4 mb-5">
        {SOURCES.map((s) => (
          <Card key={s.name} className="!p-5">
            <div className="flex items-center justify-between mb-2">
              <span style={{ color: 'var(--action)' }}>{s.icon}</span>
              <span className="pill" style={{ color: 'var(--text-muted)', borderColor: 'var(--border)', background: 'transparent', fontSize: 11 }}>{s.note}</span>
            </div>
            <div className="font-display font-semibold mb-1" style={{ fontSize: 15 }}>{s.name}</div>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{s.desc}</p>
          </Card>
        ))}
      </div>

      <div className="grid lg:grid-cols-2 gap-5 items-stretch">
      <Card>
        <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
          <h2 className="font-display font-semibold" style={{ fontSize: 17 }}>Ingest from a source</h2>
          <SegmentedControl value={mode} onChange={setMode} options={[
            { value: 'web', label: 'Web page' },
            { value: 'external', label: 'S3 object' },
            { value: 'sync', label: 'S3 folder sync' },
            { value: 'platform', label: 'Platform store' },
          ]} />
        </div>

        {mode === 'web' && (
          <div className="grid sm:grid-cols-2 gap-4">
            <Field label="Document id" help="How this page appears in your KB."><input className="field-input" value={docId} onChange={(e) => setDocId(e.target.value)} placeholder="sla-page" /></Field>
            <Field label="Page URL" help="Host must be on the egress allowlist."><input className="field-input" value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://docs.example.gov.in/sla" /></Field>
          </div>
        )}

        {(mode === 'external' || mode === 'platform') && (
          <div className="grid sm:grid-cols-2 gap-4">
            <Field label="Document id"><input className="field-input" value={docId} onChange={(e) => setDocId(e.target.value)} placeholder="q1-report" /></Field>
            <Field label="MIME type"><input className="field-input" value={mime} onChange={(e) => setMime(e.target.value)} /></Field>
            <Field label="Object key" help={mode === 'external' ? 'Key within the external bucket.' : 'Key within your platform namespace.'}>
              <input className="field-input" value={objectKey} onChange={(e) => setObjectKey(e.target.value)} placeholder="reports/q1.pdf" />
            </Field>
          </div>
        )}

        {mode === 'sync' && (
          <Field label="Folder prefix" help="Every object under this prefix is ingested (up to 50 per sync). File types are detected by extension.">
            <input className="field-input" value={prefix} onChange={(e) => setPrefix(e.target.value)} placeholder="reports/" />
          </Field>
        )}

        {needsBucket && (
          <div className="mt-4 pt-4" style={{ borderTop: '1px solid var(--border)' }}>
            <div className="flex items-center gap-2 mb-3">
              <Icon.shield size={15} style={{ color: 'var(--success)' }} />
              <span className="text-sm" style={{ color: 'var(--text-muted)' }}>The endpoint must be on the egress allowlist; a foreign endpoint is refused.</span>
            </div>
            <div className="grid sm:grid-cols-2 gap-4">
              <Field label="Provider" help="S3-compatible (AWS, MinIO, Ceph) or Azure Blob Storage. For Azure: endpoint is <account>.blob.core.windows.net, bucket is the container, credential reference resolves to {ref}_account_key or {ref}_sas_token.">
                <select className="field-input" value={provider} onChange={(e) => setProvider(e.target.value as 's3' | 'azure')}>
                  <option value="s3">S3-compatible</option>
                  <option value="azure">Azure Blob Storage</option>
                </select>
              </Field>
              <Field label="Endpoint" help="host[:port] of the S3-compatible store."><input className="field-input" value={endpoint} onChange={(e) => setEndpoint(e.target.value)} placeholder="obj.sovereign.local" /></Field>
              <Field label="Bucket"><input className="field-input" value={bucket} onChange={(e) => setBucket(e.target.value)} placeholder="tenant-exports" /></Field>
              <Field label="Credentials reference" help="Secret NAME resolved server-side — never raw keys.">
                <input className="field-input" value={credRef} onChange={(e) => setCredRef(e.target.value)} placeholder="acmesrc" />
              </Field>
              <div>
                <span className="field-label">Transport</span>
                <SegmentedControl value={secure} onChange={setSecure} options={[{ value: 'https', label: 'HTTPS' }, { value: 'http', label: 'HTTP' }]} />
              </div>
            </div>
          </div>
        )}

        <Button className="mt-6" loading={busy} disabled={!ready} onClick={run} icon={!busy && <Icon.plus size={16} />}>
          {mode === 'sync' ? 'Sync folder' : 'Ingest document'}
        </Button>

        {result && (
          <div className="mt-5"><Alert kind="success">
            Ingested <strong>{result.doc_id}</strong> — {result.chunk_count} chunk(s), {result.pii_masked} PII span(s) masked.
            <span className="block mono text-xs mt-1" style={{ opacity: .8 }}>trace {result.trace_id}</span>
          </Alert></div>
        )}
        {sync && (
          <div className="mt-5">
            <Alert kind={sync.failed ? 'warning' : sync.more ? 'info' : 'success'}>
              Synced <strong>{sync.ingested}</strong> of {sync.found} file(s){sync.failed ? ` — ${sync.failed} failed` : ''}.
              {sync.more && (
                <div className="mt-1" style={{ fontSize: 12 }}>
                  <strong>More objects remain in this folder.</strong> Each run ingests a bounded batch and
                  resumes where it stopped — run the sync again (or schedule it) to continue. This is not a
                  complete sync yet.
                </div>
              )}
            </Alert>
            {sync.items.length > 0 && (
              <ul className="mt-3 space-y-1">
                {sync.items.map((it) => (
                  <li key={it.doc_id} className="flex items-center gap-2 text-sm">
                    <span style={{ color: it.ok ? 'var(--success)' : 'var(--danger)' }}>{it.ok ? <Icon.check size={14} /> : <Icon.x size={14} />}</span>
                    <span className="mono truncate" style={{ color: 'var(--text-heading)' }}>{it.doc_id}</span>
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{it.ok ? `${it.chunks} chunk(s)` : it.error}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
        {error && (
          <div className="mt-5">
            <Alert kind="error">{error}</Alert>
            {blockedHost && (
              <div className="mt-3 flex items-center gap-3 flex-wrap">
                {canManageThisKb ? (
                  <>
                    <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
                      Allow <span className="mono" style={{ color: 'var(--text-heading)' }}>{blockedHost}</span> for this KB — you manage it, so it takes effect immediately (no approval needed). Only this host is added to the sovereignty allowlist; the guard stays enforced for everything else.
                    </span>
                    <Button size="sm" loading={requesting} onClick={allowAndIngest} icon={<Icon.shield size={15} />}>Allow host &amp; ingest</Button>
                  </>
                ) : (
                  <>
                    <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
                      Request approval for <span className="mono" style={{ color: 'var(--text-heading)' }}>{blockedHost}</span> — an owner/admin or this KB's manager reviews it (the guard stays on).
                    </span>
                    <Button size="sm" variant="secondary" loading={requesting} onClick={requestApproval} icon={<Icon.shield size={15} />}>Request approval</Button>
                  </>
                )}
              </div>
            )}
          </div>
        )}
      </Card>

      {/* Right column: what this KB is actually connected to, and what came through it. The
          form alone said nothing about the RESULT of using it -- you could add a bucket and
          have no way to see it, or to notice a KB was pointed at the wrong one. */}
      <Card>
        <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
          <h2 className="font-display font-semibold" style={{ fontSize: 17 }}>Source overview</h2>
          {connected && (
            <span className="pill" style={{ fontSize: 11, color: 'var(--success)', borderColor: 'var(--success)', background: 'transparent' }}>
              <Icon.check size={12} /> Connected
            </span>
          )}
        </div>

        {!connected ? (
          <div className="text-sm" style={{ color: 'var(--text-muted)' }}>
            No external folder is connected to <strong>{kbName}</strong> yet. Run a folder sync and the
            source, its documents and its sync history appear here.
          </div>
        ) : (
          <>
            <div className="grid grid-cols-3 gap-3 mb-4">
              {[
                { label: 'Documents', value: String(fromSource.length) },
                { label: 'Per sync cap', value: String(connected.limit) },
                { label: 'Last sync', value: sync ? `${sync.ingested}/${sync.found}` : (sched?.last_run_at ? 'scheduled' : '—') },
              ].map((m) => (
                <div key={m.label} className="rounded-lg p-3" style={{ background: 'var(--surface-2, rgba(255,255,255,.03))' }}>
                  <div className="font-display font-semibold num" style={{ fontSize: 20, color: 'var(--text-heading)' }}>{m.value}</div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{m.label}</div>
                </div>
              ))}
            </div>

            <dl className="space-y-1.5 mb-4" style={{ fontSize: 13 }}>
              {[
                ['Provider', connected.provider === 'azure' ? 'Azure Blob Storage' : 'S3-compatible'],
                ['Endpoint', `${connected.secure ? 'https' : 'http'}://${connected.endpoint}`],
                [connected.provider === 'azure' ? 'Container' : 'Bucket', connected.bucket],
                ['Prefix', connected.prefix || '(whole bucket)'],
                ['Credentials', connected.credentials_ref],
              ].map(([k, v]) => (
                <div key={k} className="flex items-baseline justify-between gap-3">
                  <dt style={{ color: 'var(--text-muted)' }}>{k}</dt>
                  <dd className="mono truncate" style={{ color: 'var(--text-heading)' }} title={v}>{v}</dd>
                </div>
              ))}
              {newest && (
                <div className="flex items-baseline justify-between gap-3">
                  <dt style={{ color: 'var(--text-muted)' }}>Newest document</dt>
                  <dd className="mono" style={{ color: 'var(--text-heading)' }}>{new Date(newest).toLocaleString()}</dd>
                </div>
              )}
              {sched?.next_run_at && (
                <div className="flex items-baseline justify-between gap-3">
                  <dt style={{ color: 'var(--text-muted)' }}>Next re-sync</dt>
                  <dd className="mono" style={{ color: 'var(--text-heading)' }}>{new Date(sched.next_run_at).toLocaleString()}</dd>
                </div>
              )}
            </dl>

            {fromSource.length > 0 && (
              <>
                <div className="eyebrow mb-2">Documents from this source</div>
                <ul className="space-y-1 overflow-y-auto" style={{ maxHeight: 220 }}>
                  {fromSource.slice(0, 50).map((d) => (
                    <li key={d.doc_id} className="flex items-center gap-2" style={{ fontSize: 12 }}>
                      <span style={{ color: 'var(--success)' }}><Icon.check size={12} /></span>
                      <span className="mono truncate" style={{ color: 'var(--text-heading)' }} title={d.doc_id}>{d.doc_id}</span>
                      <span className="ml-auto whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>
                        {new Date(d.ingested_at).toLocaleDateString()}
                      </span>
                    </li>
                  ))}
                </ul>
              </>
            )}

            <p className="field-help mt-4">
              Credentials are resolved server-side from the reference above — the key itself never reaches
              this page. Running another sync replaces the connected source.
            </p>
          </>
        )}
      </Card>
      </div>

      <div className="mt-5 space-y-4">
        <Collapsible
          icon={<Icon.ask size={16} />}
          title="Crawl a website"
          subtitle="Crawl from a seed URL and ingest each in-scope page (bounded, sovereign)."
        >
          <CrawlCard />
        </Collapsible>

        <Collapsible
          icon={<Icon.activity size={16} />}
          title="Scheduled re-sync"
          subtitle="Re-pull the last folder-sync source on a schedule (delta only)."
        >
          <SyncScheduleCard />
        </Collapsible>

        <EgressApprovals />

        <Collapsible
          icon={<Icon.connectors size={16} />}
          title="Structured data sources"
          subtitle="Databases, REST APIs, SharePoint sites, and graph sync — beyond documents."
        >
          <StructuredSources />
        </Collapsible>
      </div>
    </div>
  )
}

function SyncScheduleCard() {
  const { session } = useAuth()
  const { kbId } = useKb()
  const toast = useToast()
  const [sched, setSched] = useState<SyncScheduleView | null>(null)
  const [hours, setHours] = useState(24)
  const [schedMode, setSchedMode] = useState<'interval' | 'cron'>('interval')
  const [cron, setCron] = useState('0 2 * * 1')
  const [tz, setTz] = useState('Asia/Kolkata')
  const [saving, setSaving] = useState(false)
  const [running, setRunning] = useState(false)

  useEffect(() => {
    if (!session || !kbId) return
    setSched(null)
    api.getSyncSchedule(session.token, kbId).then((s) => {
      setSched(s); setHours(s.interval_hours)
      if (s.cron) { setSchedMode('cron'); setCron(s.cron); setTz(s.timezone || 'Asia/Kolkata') }
    }).catch(() => setSched(null))
  }, [session, kbId])

  async function save(enabled: boolean) {
    if (!session) return
    setSaving(true)
    try {
      const body = schedMode === 'cron'
        ? { enabled, cron: cron.trim(), timezone: tz.trim() || 'UTC' }
        : { enabled, interval_hours: hours }
      const s = await api.setSyncSchedule(session.token, kbId, body)
      setSched(s)
      toast.push('success', !enabled ? 'Schedule paused.'
        : s.cron ? `Scheduled: ${s.cron} (${s.timezone}).` : `Scheduled re-sync every ${s.interval_hours}h.`)
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not save schedule.')
    } finally {
      setSaving(false)
    }
  }

  async function runNow() {
    if (!session) return
    setRunning(true)
    try {
      const r = await api.runSyncNow(session.token, kbId)
      toast.push('success', `Sync: ${r.created} new · ${r.updated} updated · ${r.unchanged} unchanged${r.failed ? ` · ${r.failed} failed` : ''}.`)
      api.getSyncSchedule(session.token, kbId).then(setSched).catch(() => undefined)
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Sync failed.')
    } finally {
      setRunning(false)
    }
  }

  return (
    <div>
      <p className="field-help mb-3">Re-pull the last folder-sync source on a schedule; only changed files are re-indexed (delta). Run a folder sync above first to capture the source.</p>
      {sched === null ? (
        <div className="skeleton" style={{ height: 70 }} />
      ) : !sched.source_configured ? (
        <Alert kind="info">No sync source captured yet — run an “S3 folder sync” above, then enable a schedule here.</Alert>
      ) : (
        <>
          <div className="mb-3" style={{ maxWidth: 260 }}>
            <SegmentedControl value={schedMode} onChange={(v) => setSchedMode(v as 'interval' | 'cron')}
              options={[{ value: 'interval', label: 'Every N hours' }, { value: 'cron', label: 'Calendar (cron)' }]} />
          </div>
          {schedMode === 'interval' ? (
            <label className="text-xs" style={{ color: 'var(--text-muted)' }}>Every (hours)
              <input className="field-input mt-1" style={{ width: 100 }} type="number" min={1} max={720} value={hours}
                onChange={(e) => setHours(Math.max(1, Math.min(720, Number(e.target.value) || 1)))} />
            </label>
          ) : (
            <div className="flex flex-col gap-2">
              <div className="flex flex-wrap gap-1.5">
                {[['Daily 00:00', '0 0 * * *'], ['Weekly Mon 02:00', '0 2 * * 1'], ['Monthly 1st 00:00', '0 0 1 * *'], ['Hourly', '0 * * * *']].map(([label, expr]) => (
                  <button key={expr} type="button" className="chip" style={{ cursor: 'pointer', color: cron === expr ? 'var(--accent)' : undefined }} onClick={() => setCron(expr)}>{label}</button>
                ))}
              </div>
              <div className="flex items-end gap-3 flex-wrap">
                <label className="text-xs" style={{ color: 'var(--text-muted)' }}>Cron (min hour dom mon dow)
                  <input className="field-input mt-1" style={{ width: 170, fontFamily: 'var(--font-mono, monospace)' }} value={cron} onChange={(e) => setCron(e.target.value)} placeholder="0 2 * * 1" />
                </label>
                <label className="text-xs" style={{ color: 'var(--text-muted)' }}>Timezone
                  <input className="field-input mt-1" style={{ width: 150 }} value={tz} onChange={(e) => setTz(e.target.value)} placeholder="Asia/Kolkata" />
                </label>
              </div>
            </div>
          )}
          <div className="flex items-center gap-3 flex-wrap mt-3">
            <Button size="sm" variant={sched.enabled ? 'secondary' : 'primary'} loading={saving} onClick={() => save(!sched.enabled)}>
              {sched.enabled ? 'Pause' : 'Enable'}
            </Button>
            <Button size="sm" variant="subtle" loading={running} onClick={runNow} icon={<Icon.play size={15} />}>Run now</Button>
          </div>
          <div className="flex flex-wrap gap-2 mt-3 text-xs">
            <span className="chip" style={{ color: sched.enabled ? 'var(--success)' : 'var(--text-muted)' }}>{sched.enabled ? (sched.cron ? `${sched.cron} · ${sched.timezone}` : `every ${sched.interval_hours}h`) : 'paused'}</span>
            {sched.enabled && sched.next_run_at && <span className="chip">next {new Date(sched.next_run_at).toLocaleString()}</span>}
            {sched.last_run_at && <span className="chip">last {new Date(sched.last_run_at).toLocaleString()}</span>}
            {!sched.scheduler_on && <span className="chip" style={{ color: 'var(--warning)' }}>scheduler off (operator)</span>}
          </div>
        </>
      )}
    </div>
  )
}


function EgressApprovals() {
  const { session } = useAuth()
  const toast = useToast()
  const isAdmin = session?.role === 'owner' || session?.role === 'admin'
  const [managedKbs, setManagedKbs] = useState<string[]>([])
  // An owner/admin decides any egress request; a KB manager decides requests raised for a KB they
  // manage (scoped by the request's kb_id).
  const canDecide = (r: EgressRequest) => isAdmin || (!!r.kb_id && managedKbs.includes(r.kb_id))
  const canApproveAny = isAdmin || managedKbs.length > 0
  const [reqs, setReqs] = useState<EgressRequest[] | null>(null)
  const [acting, setActing] = useState<string | null>(null)

  async function load() {
    if (!session) return
    try { setReqs(await api.listEgressRequests(session.token)) } catch { setReqs([]) }
  }
  useEffect(() => { void load() /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [session])
  useEffect(() => {
    if (!session) return
    api.myManagedKbs(session.token).then(setManagedKbs).catch(() => setManagedKbs([]))
  }, [session])
  useEffect(() => {
    const reload = () => { void load() }
    window.addEventListener('rag:egress-changed', reload)
    window.addEventListener('focus', reload)
    return () => {
      window.removeEventListener('rag:egress-changed', reload)
      window.removeEventListener('focus', reload)
    }
    /* eslint-disable-next-line react-hooks/exhaustive-deps */
  }, [session])

  async function decide(id: string, action: 'approve' | 'reject' | 'revoke') {
    if (!session) return
    setActing(id)
    try {
      await api.decideEgress(session.token, id, action)
      const msg = action === 'approve' ? 'Host approved — now allowlisted.'
        : action === 'revoke' ? 'Host revoked — removed from the allowlist.' : 'Request rejected.'
      toast.push('success', msg)
      await load()
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Action failed.')
    } finally {
      setActing(null)
    }
  }

  // Isolation: a KB manager only sees egress requests for KBs they manage (scoped by kb_id); an
  // owner/admin sees everything (incl. org-level requests with no kb_id). `canDecide` already
  // encodes exactly that, so filtering the whole list by it keeps a manager from even seeing (e.g.)
  // Finance's request while viewing Tech — not just from acting on it.
  const visible = (reqs ?? []).filter(canDecide)
  const pending = visible.filter((r) => r.status === 'pending')
  const approved = visible.filter((r) => r.status === 'approved')
  const history = visible.filter((r) => r.status === 'rejected' || r.status === 'revoked').slice(0, 6)
  if (reqs !== null && pending.length === 0 && approved.length === 0 && history.length === 0) return null

  return (
    <Collapsible
      icon={<Icon.shield size={16} />}
      title="Egress approval requests"
      subtitle={`${pending.length} pending · ${approved.length} allowed host${approved.length === 1 ? '' : 's'}`}
    >
      <p className="field-help mb-3">Requests to allow an external host. Approving adds ONLY that host to the sovereignty allowlist — the guard stays enforced for everything else. Revoke a host once the work is done to remove it again. Every decision is audited.{!canApproveAny && ' An owner/admin — or the manager of the KB the request was raised for — decides.'}</p>

      {pending.length > 0 && (
        <ul className="space-y-2 mb-4">
          {pending.map((r) => (
            <li key={r.id} className="flex items-center justify-between gap-3 flex-wrap rounded-lg p-2.5" style={{ border: '1px solid var(--border)' }}>
              <div className="min-w-0">
                <div className="mono font-semibold" style={{ color: 'var(--text-heading)' }}>{r.host}</div>
                <div className="text-xs" style={{ color: 'var(--text-muted)' }}>by {r.requested_by}{r.reason ? ` · ${r.reason}` : ''}</div>
              </div>
              {canDecide(r) && (
                <div className="flex gap-2">
                  <Button size="sm" loading={acting === r.id} onClick={() => decide(r.id, 'approve')} icon={<Icon.check size={14} />}>Approve</Button>
                  <Button size="sm" variant="subtle" onClick={() => decide(r.id, 'reject')} icon={<Icon.x size={14} />}>Reject</Button>
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
      {pending.length === 0 && <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>No pending requests.</p>}

      {approved.length > 0 && (
        <>
          <div className="eyebrow mb-2">Allowed hosts (active)</div>
          <ul className="space-y-1.5">
            {approved.map((r) => (
              <li key={r.id} className="flex items-center justify-between gap-3 flex-wrap rounded-md px-2.5 py-2" style={{ background: 'var(--surface-2)' }}>
                <span className="mono" style={{ fontSize: 12.5, color: 'var(--success)' }}>{r.host}</span>
                {canDecide(r) && (
                  <Button size="sm" variant="subtle" loading={acting === r.id} onClick={() => decide(r.id, 'revoke')} icon={<Icon.trash size={13} />} style={{ color: 'var(--danger)' }}>Revoke</Button>
                )}
              </li>
            ))}
          </ul>
        </>
      )}
      {history.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {history.map((r) => (
            <span key={r.id} className="chip mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>{r.host} · {r.status}</span>
          ))}
        </div>
      )}
    </Collapsible>
  )
}

function CrawlCard() {
  const { session } = useAuth()
  const { kbId } = useKb()
  const toast = useToast()
  const [seed, setSeed] = useState('')
  const [depth, setDepth] = useState(1)
  const [pages, setPages] = useState(20)
  const [sameHost, setSameHost] = useState(true)
  const [busy, setBusy] = useState(false)
  const [res, setRes] = useState<{ pages: number; created: number; unchanged: number; skipped_offsite: number; skipped_robots: number; failed: number; errors: string[] } | null>(null)

  async function crawl() {
    if (!session || busy || !/^https?:\/\//.test(seed.trim())) return
    setBusy(true); setRes(null)
    try {
      const r = await api.crawlWebsite(session.token, kbId, { seed_url: seed.trim(), max_depth: depth, max_pages: pages, same_host_only: sameHost })
      setRes(r)
      toast.push('success', `Crawled ${r.pages} page${r.pages === 1 ? '' : 's'} (${r.created} new, ${r.unchanged} unchanged).`)
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Crawl failed.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <p className="field-help mb-3">Crawl from a seed URL and ingest each in-scope page. Bounded by depth &amp; page limits, respects robots.txt, and every hop is egress-allowlisted (sovereign). Re-crawls only re-index changed pages.</p>
      <Field label="Seed URL"><input className="field-input" value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="https://docs.sovereign.local/" /></Field>
      <div className="grid grid-cols-2 gap-3 mt-3">
        <label className="text-xs" style={{ color: 'var(--text-muted)' }}>Max depth
          <input className="field-input mt-1" type="number" min={0} max={3} value={depth} onChange={(e) => setDepth(Math.max(0, Math.min(3, Number(e.target.value) || 0)))} /></label>
        <label className="text-xs" style={{ color: 'var(--text-muted)' }}>Max pages
          <input className="field-input mt-1" type="number" min={1} max={100} value={pages} onChange={(e) => setPages(Math.max(1, Math.min(100, Number(e.target.value) || 1)))} /></label>
      </div>
      <label className="flex items-center gap-2 mt-3 text-sm" style={{ color: 'var(--text-primary)' }}>
        <input type="checkbox" checked={sameHost} onChange={(e) => setSameHost(e.target.checked)} /> Same host only
      </label>
      <Button className="mt-4" loading={busy} disabled={!/^https?:\/\//.test(seed.trim())} onClick={crawl} icon={!busy && <Icon.play size={16} />}>
        {busy ? 'Crawling…' : 'Start crawl'}
      </Button>
      {res && (
        <div className="mt-4">
          <div className="flex flex-wrap gap-2 text-xs">
            <span className="chip">{res.pages} pages</span>
            <span className="chip" style={{ color: 'var(--success)' }}>{res.created} new</span>
            <span className="chip">{res.unchanged} unchanged</span>
            {res.skipped_offsite > 0 && <span className="chip" style={{ color: 'var(--warning)' }}>{res.skipped_offsite} off-site</span>}
            {res.skipped_robots > 0 && <span className="chip" style={{ color: 'var(--warning)' }}>{res.skipped_robots} robots</span>}
            {res.failed > 0 && <span className="chip" style={{ color: 'var(--danger)' }}>{res.failed} failed</span>}
          </div>
          {res.failed > 0 && res.errors.length > 0 && (
            <div className="mt-2 rounded-lg p-2.5 text-xs" style={{ background: 'color-mix(in srgb, var(--danger) 8%, transparent)', border: '1px solid color-mix(in srgb, var(--danger) 25%, transparent)' }}>
              <div className="eyebrow mb-1" style={{ color: 'var(--danger)' }}>Why pages failed</div>
              {res.errors.map((e, i) => <div key={i} style={{ color: 'var(--text-muted)' }}>• {e}</div>)}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
