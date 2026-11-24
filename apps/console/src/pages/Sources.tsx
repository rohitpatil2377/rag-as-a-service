// Structured source connections (#10) — NoSQL / REST / SharePoint, plus syncing them into
// the knowledge graph. Mirrors the SQL-connection card pattern in Data.tsx exactly (one
// connection per KB, PUT replaces it, DELETE disconnects with a confirm).
import { useEffect, useState } from 'react'
import { Icon } from '../components/icons'
import { useToast } from '../components/overlay'
import { Button, Field, Skeleton } from '../components/ui'
import { Collapsible } from '../components/Collapsible'
import {
  ApiError, api,
  type GraphSyncResult, type NoSqlConnectionInfo, type RestConnectionInfo, type SharePointConnectionInfo,
} from '../lib/api'
import { useAuth } from '../lib/auth'
import { useKb } from '../lib/kb'

function ConnectedBadge({ title, sub, onDisconnect }: { title: string; sub: string; onDisconnect: () => void }) {
  return (
    <div className="rounded-lg p-3 mb-3" style={{ background: 'color-mix(in srgb, var(--success) 7%, transparent)', border: '1px solid color-mix(in srgb, var(--success) 25%, transparent)' }}>
      <div className="flex items-center gap-2">
        <span aria-hidden style={{ width: 8, height: 8, borderRadius: 999, background: 'var(--success)', flexShrink: 0 }} />
        <div className="min-w-0">
          <div className="mono font-semibold text-sm truncate" style={{ color: 'var(--text-heading)' }}>{title}</div>
          <div className="text-xs truncate" style={{ color: 'var(--text-muted)' }}>{sub}</div>
        </div>
        <div className="flex-grow" />
        <Button size="sm" variant="subtle" onClick={onDisconnect}>Disconnect</Button>
      </div>
    </div>
  )
}

function NoSqlCard() {
  const { session } = useAuth()
  const { kbId } = useKb()
  const toast = useToast()
  const [conn, setConn] = useState<NoSqlConnectionInfo | null>(null)
  const [uri, setUri] = useState('')
  const [database, setDatabase] = useState('')
  const [collections, setCollections] = useState('')
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (!session || !kbId) return
    setConn(null)
    api.getNoSqlConnection(session.token, kbId)
      .then((c) => { setConn(c); if (c.configured) { setDatabase(c.database ?? ''); setCollections(c.collections.join(', ')) } })
      .catch(() => setConn({ configured: false, host: null, port: null, database: null, collections: [] }))
  }, [session, kbId])

  async function save() {
    if (!session) return
    const list = collections.split(/[\n,]/).map((s) => s.trim()).filter(Boolean)
    if (!uri.trim() || !database.trim() || list.length === 0) { toast.push('error', 'Give a URI, database name, and at least one collection.'); return }
    setSaving(true)
    try {
      const info = await api.setNoSqlConnection(session.token, kbId, { uri: uri.trim(), database: database.trim(), collections: list })
      setConn(info); setUri('')
      toast.push('success', `Connected to ${info.database ?? 'database'} · ${info.collections.length} collection(s).`)
      void api.syncSourcesToGraph(session.token, kbId).catch(() => {}) // auto-flow the new source into the graph
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not save connection.')
    } finally {
      setSaving(false)
    }
  }

  async function disconnect() {
    if (!session || !conn?.configured) return
    if (!window.confirm('Disconnect this NoSQL database? Graph sync will stop pulling from it until reconnected.')) return
    try {
      await api.deleteNoSqlConnection(session.token, kbId)
      setConn({ configured: false, host: null, port: null, database: null, collections: [] })
      setDatabase(''); setCollections('')
      toast.push('info', 'NoSQL database disconnected.')
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not disconnect.')
    }
  }

  return (
    <Collapsible
      icon={<Icon.database size={16} />}
      title="NoSQL connection"
      subtitle={conn?.configured ? `Connected · ${conn.database ?? 'database'}` : 'Not connected — MongoDB-compatible database.'}
    >
      {conn === null ? <Skeleton style={{ height: 90 }} /> : conn.configured ? (
        <>
          <ConnectedBadge title={conn.database ?? 'database'} sub={`${conn.host ?? ''}${conn.port ? `:${conn.port}` : ''}`} onDisconnect={disconnect} />
          <p className="field-help mb-2">Collections in scope:</p>
          <div className="flex flex-wrap gap-1.5 mb-4">
            {conn.collections.map((c) => <span key={c} className="chip mono" style={{ fontSize: 11 }}>{c}</span>)}
          </div>
        </>
      ) : (
        <p className="field-help mb-3">No NoSQL database connected yet.</p>
      )}
      <div className="space-y-3 pt-3" style={{ borderTop: '1px solid var(--border)' }}>
        {conn?.configured && <p className="field-help" style={{ marginBottom: 0 }}>One connection per knowledge base — saving below <b>replaces</b> the current one.</p>}
        <Field label="Connection URI" help="Host must be egress-allowlisted. e.g. mongodb://ro_user:***@db:27017">
          <input className="field-input mono" style={{ fontSize: 12 }} value={uri} onChange={(e) => setUri(e.target.value)} placeholder="mongodb://…" />
        </Field>
        <Field label="Database">
          <input className="field-input" value={database} onChange={(e) => setDatabase(e.target.value)} placeholder="hr" />
        </Field>
        <Field label="Collections" help="Comma/newline-separated — only these are read.">
          <textarea className="field-input mono" style={{ fontSize: 12 }} rows={2} value={collections} onChange={(e) => setCollections(e.target.value)} placeholder="employees, departments" />
        </Field>
        <Button size="sm" loading={saving} onClick={save} icon={<Icon.check size={15} />}>{conn?.configured ? 'Replace connection' : 'Save connection'}</Button>
      </div>
    </Collapsible>
  )
}

function RestCard() {
  const { session } = useAuth()
  const { kbId } = useKb()
  const toast = useToast()
  const [conn, setConn] = useState<RestConnectionInfo | null>(null)
  const [baseUrl, setBaseUrl] = useState('')
  const [path, setPath] = useState('')
  const [queryParam, setQueryParam] = useState('')
  const [token, setToken] = useState('')
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (!session || !kbId) return
    setConn(null)
    api.getRestConnection(session.token, kbId)
      .then((c) => { setConn(c); if (c.configured) { setPath(c.path ?? ''); setQueryParam(c.query_param ?? '') } })
      .catch(() => setConn({ configured: false, base_url: null, path: null, query_param: null, has_token: false }))
  }, [session, kbId])

  async function save() {
    if (!session) return
    if (!baseUrl.trim()) { toast.push('error', 'Give a base URL.'); return }
    setSaving(true)
    try {
      const info = await api.setRestConnection(session.token, kbId, {
        base_url: baseUrl.trim(),
        path: path.trim() || undefined,
        query_param: queryParam.trim() || undefined,
        token: token.trim() || undefined,
      })
      setConn(info); setBaseUrl(''); setToken('')
      toast.push('success', `Connected to ${info.base_url ?? 'endpoint'}.`)
      void api.syncSourcesToGraph(session.token, kbId).catch(() => {}) // auto-flow the new source into the graph
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not save connection.')
    } finally {
      setSaving(false)
    }
  }

  async function disconnect() {
    if (!session || !conn?.configured) return
    if (!window.confirm('Disconnect this REST endpoint? Graph sync will stop pulling from it until reconnected.')) return
    try {
      await api.deleteRestConnection(session.token, kbId)
      setConn({ configured: false, base_url: null, path: null, query_param: null, has_token: false })
      setPath(''); setQueryParam('')
      toast.push('info', 'REST endpoint disconnected.')
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not disconnect.')
    }
  }

  return (
    <Collapsible
      icon={<Icon.cloud size={16} />}
      title="REST connection"
      subtitle={conn?.configured ? `Connected · ${conn.base_url ?? 'endpoint'}` : 'Not connected — HTTP/JSON API endpoint.'}
    >
      {conn === null ? <Skeleton style={{ height: 90 }} /> : conn.configured ? (
        <ConnectedBadge title={conn.base_url ?? 'endpoint'}
          sub={`${conn.path ? `path: ${conn.path}` : 'no path set'}${conn.has_token ? ' · token set' : ''}`}
          onDisconnect={disconnect} />
      ) : (
        <p className="field-help mb-3">No REST endpoint connected yet.</p>
      )}
      <div className="space-y-3 pt-3" style={{ borderTop: '1px solid var(--border)' }}>
        {conn?.configured && <p className="field-help" style={{ marginBottom: 0 }}>Saving below <b>replaces</b> the current connection.</p>}
        <Field label="Base URL" help="Host must be egress-allowlisted.">
          <input className="field-input mono" style={{ fontSize: 12 }} value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="https://api.example.gov.in" />
        </Field>
        <Field label="Path" help="Optional — appended to the base URL.">
          <input className="field-input" value={path} onChange={(e) => setPath(e.target.value)} placeholder="/v1/records" />
        </Field>
        <Field label="Query param" help="Optional — name of a pagination/filter query parameter.">
          <input className="field-input" value={queryParam} onChange={(e) => setQueryParam(e.target.value)} placeholder="page" />
        </Field>
        <Field label="Token" help="Optional bearer token — never shown again once saved.">
          <input className="field-input mono" style={{ fontSize: 12 }} type="password" value={token} onChange={(e) => setToken(e.target.value)} placeholder={conn?.has_token ? '••••••••' : ''} />
        </Field>
        <Button size="sm" loading={saving} onClick={save} icon={<Icon.check size={15} />}>{conn?.configured ? 'Replace connection' : 'Save connection'}</Button>
      </div>
    </Collapsible>
  )
}

function SharePointCard() {
  const { session } = useAuth()
  const { kbId } = useKb()
  const toast = useToast()
  const [conn, setConn] = useState<SharePointConnectionInfo | null>(null)
  const [siteUrl, setSiteUrl] = useState('')
  const [token, setToken] = useState('')
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (!session || !kbId) return
    setConn(null)
    api.getSharePointConnection(session.token, kbId)
      .then(setConn)
      .catch(() => setConn({ configured: false, site_url: null, has_token: false }))
  }, [session, kbId])

  async function save() {
    if (!session) return
    if (!siteUrl.trim()) { toast.push('error', 'Give a SharePoint site URL.'); return }
    setSaving(true)
    try {
      const info = await api.setSharePointConnection(session.token, kbId, { site_url: siteUrl.trim(), token: token.trim() || undefined })
      setConn(info); setSiteUrl(''); setToken('')
      toast.push('success', `Connected to ${info.site_url ?? 'site'}.`)
      void api.syncSourcesToGraph(session.token, kbId).catch(() => {}) // auto-flow the new source into the graph
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not save connection.')
    } finally {
      setSaving(false)
    }
  }

  async function disconnect() {
    if (!session || !conn?.configured) return
    if (!window.confirm('Disconnect this SharePoint site? Graph sync will stop pulling from it until reconnected.')) return
    try {
      await api.deleteSharePointConnection(session.token, kbId)
      setConn({ configured: false, site_url: null, has_token: false })
      toast.push('info', 'SharePoint site disconnected.')
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not disconnect.')
    }
  }

  return (
    <Collapsible
      icon={<Icon.doc size={16} />}
      title="SharePoint connection"
      subtitle={conn?.configured ? `Connected · ${conn.site_url ?? 'site'}` : 'Not connected — SharePoint site.'}
    >
      {conn === null ? <Skeleton style={{ height: 90 }} /> : conn.configured ? (
        <ConnectedBadge title={conn.site_url ?? 'site'} sub={conn.has_token ? 'token set' : 'no token'} onDisconnect={disconnect} />
      ) : (
        <p className="field-help mb-3">No SharePoint site connected yet.</p>
      )}
      <div className="space-y-3 pt-3" style={{ borderTop: '1px solid var(--border)' }}>
        {conn?.configured && <p className="field-help" style={{ marginBottom: 0 }}>Saving below <b>replaces</b> the current connection.</p>}
        <Field label="Site URL" help="Host must be egress-allowlisted.">
          <input className="field-input mono" style={{ fontSize: 12 }} value={siteUrl} onChange={(e) => setSiteUrl(e.target.value)} placeholder="https://tenant.sharepoint.com/sites/hr" />
        </Field>
        <Field label="Token" help="Optional access token — never shown again once saved.">
          <input className="field-input mono" style={{ fontSize: 12 }} type="password" value={token} onChange={(e) => setToken(e.target.value)} placeholder={conn?.has_token ? '••••••••' : ''} />
        </Field>
        <Button size="sm" loading={saving} onClick={save} icon={<Icon.check size={15} />}>{conn?.configured ? 'Replace connection' : 'Save connection'}</Button>
      </div>
    </Collapsible>
  )
}

function GraphSyncCard() {
  const { session } = useAuth()
  const { kbId, kbs } = useKb()
  const toast = useToast()
  const kbName = kbs.find((k) => k.kb_id === kbId)?.name ?? kbId
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<GraphSyncResult | null>(null)

  async function run() {
    if (!session || !kbId) return
    setRunning(true); setResult(null)
    try {
      const r = await api.syncSourcesToGraph(session.token, kbId)
      setResult(r)
      const total = (r.sql ?? 0) + (r.nosql ?? 0) + (r.rest ?? 0) + (r.sharepoint ?? 0)
      toast.push('success', `Synced ${total} item(s) from connected sources into the graph.`)
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Sync failed.')
    } finally {
      setRunning(false)
    }
  }

  return (
    <Collapsible
      icon={<Icon.activity size={16} />}
      title="Sync sources to graph"
      subtitle="Pull connected sources into the knowledge graph."
    >
      <p className="field-help mb-3">Pull data from every connected source (SQL, NoSQL, REST, SharePoint) for {kbName} into the knowledge graph used by Architecture and graph-aware answers.</p>
      <Button size="sm" loading={running} onClick={run} icon={<Icon.refresh size={15} />}>Sync sources to graph</Button>
      {result && (
        <div className="flex flex-wrap gap-2 mt-4 text-xs">
          {result.sql != null && <span className="chip">SQL · {result.sql}</span>}
          {result.nosql != null && <span className="chip">NoSQL · {result.nosql}</span>}
          {result.rest != null && <span className="chip">REST · {result.rest}</span>}
          {result.sharepoint != null && <span className="chip">SharePoint · {result.sharepoint}</span>}
        </div>
      )}
    </Collapsible>
  )
}

// Embeddable structured-source section — rendered inside the Connectors page (databases, REST
// APIs, SharePoint sites, plus graph-sync), so document connectors and structured connectors live
// on one page instead of two.
export function StructuredSources() {
  return (
    <div className="grid lg:grid-cols-2 gap-5 items-stretch">
      <NoSqlCard />
      <RestCard />
      <SharePointCard />
      <GraphSyncCard />
    </div>
  )
}
