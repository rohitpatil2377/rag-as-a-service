import { useCallback, useEffect, useRef, useState } from 'react'
import { useI18n } from '../lib/i18n'
import { Icon } from '../components/icons'
import { useToast } from '../components/overlay'
import { Alert, Button, Card, Field, PageHeader, Skeleton, Table } from '../components/ui'
import { ApiError, api, isPendingModelRequest, type ModelConnectionItem, type ModelRequestInfo, type ModelRole, type ModelTestResult, type ModelHealthReport, type ModelsInfo, type OrgEmbedderStatus, type SystemInfo } from '../lib/api'
import { hasRole, useAuth } from '../lib/auth'
import { useKb } from '../lib/kb'

// MODELS ARE ORG-WIDE. The org admin manages the org registry directly and sets the org-wide
// default of each ROLE; a KB manager can REQUEST one (queued for org-admin approval); a member
// sees the generation registry read-only. A role's org-wide default is what the platform actually
// uses: generation answers with it, reranking reranks with it immediately, and an embedding entry
// is adopted only when it matches the vector space a KB was built in — otherwise it is STAGED and
// one apply re-embeds every KB (see the re-embed banner). There is no per-KB model form: a model
// added here is the organisation's, for the job it names.

const ORG_ROLES: { role: ModelRole; label: string; help: string }[] = [
  { role: 'generation', label: 'Generation', help: 'Writes the answers. The org-wide default answers every KB.' },
  { role: 'embedding', label: 'Embedding', help: 'Turns text into vectors at ingest and query time. Test reads its dimension.' },
  { role: 'reranking', label: 'Reranking', help: 'Re-orders retrieved passages before the answer. Test reads its wire shape. Takes effect immediately — its scores set the abstain confidence, so recalibrate the threshold after switching.' },
  { role: 'ocr', label: 'OCR', help: 'Reads scanned pages and images at ingest. Takes effect immediately — nothing stored depends on which engine read a page.' },
  { role: 'vision', label: 'Vision', help: 'Describes what an image shows, alongside its OCR text. Takes effect immediately.' },
]

// How often the console re-reads backend health. With the continuous server-side poller on, a read
// is served from a pre-computed snapshot (no probe on the request path), so ~20s keeps the dots and
// the "updated Ns ago" label live and cheap. When the poller is off the server still answers from
// its short probe cache, so a faster poll just returns the same cached answer.
const HEALTH_POLL_MS = 20_000

/** "just now" / "40s ago" / "3m ago" — how old the reading on screen is. */
function sinceLabel(checkedAt: string): string {
  const seconds = Math.max(0, Math.round((Date.now() - new Date(checkedAt).getTime()) / 1000))
  if (seconds < 10) return 'just now'
  if (seconds < 60) return `${seconds}s ago`
  return `${Math.round(seconds / 60)}m ago`
}

/** A failing probe's reason, which matters more than the configured detail when it is failing. */
function probeDetail(health: ModelHealthReport | null, role: string): string {
  const probe = health?.backends.find((b) => b.role === role)
  return probe && !probe.online ? probe.detail : ''
}

export function Models() {
  const { session } = useAuth()
  const { t } = useI18n()
  const { kbId } = useKb()
  const toast = useToast()
  // An org owner/admin manages the org-wide registry directly and approves manager requests.
  const isOrgAdmin = session?.role === 'owner' || hasRole(session, 'admin')
  // A KB manager (manages at least one KB but is not an org admin) can REQUEST an org-wide model.
  const [managedKbs, setManagedKbs] = useState<string[] | null>(null)
  const isManager = !isOrgAdmin && !!managedKbs && managedKbs.length > 0
  const [platform, setPlatform] = useState<ModelsInfo | null>(null)
  const [sysInfo, setSysInfo] = useState<SystemInfo | null>(null)
  // Live reachability per backend. The roster above reads CONFIGURATION; this says whether the
  // endpoint actually answers, and how fast. Probed server-side (cached ~30s), re-checkable.
  const [health, setHealth] = useState<ModelHealthReport | null>(null)
  const [checking, setChecking] = useState(false)
  const [acting, setActing] = useState<string | null>(null)
  // One poll at a time. Each probe is a real request to a real model endpoint, so a slow round
  // must not stack another on top of it — without this, a 30 s endpoint under a 30 s timer puts a
  // permanent queue of probes on the very backend that is already struggling.
  const polling = useRef(false)
  const loadHealth = useCallback(async (refresh = false, quiet = false) => {
    if (!session || polling.current) return
    polling.current = true
    if (!quiet) setChecking(true)  // background polls must not flicker the button to "Checking…"
    // Pass the KB: the generation row must be probed against the connection that answers IT,
    // not the platform default (they are different endpoints with different credentials).
    // finally, not a trailing statement: anything that escaped here would leave the guard stuck
    // true and silently stop every future poll — the table would then look live and be frozen.
    try {
      setHealth(await api.modelHealth(session.token, kbId || undefined, refresh))
    } catch {
      setHealth(null)
    } finally {
      if (!quiet) setChecking(false)
      polling.current = false
    }
  }, [session, kbId])
  useEffect(() => { void loadHealth() }, [loadHealth])

  // Re-render once a second so the "updated Ns ago" label counts up between polls instead of
  // sitting at whatever it said when the last response landed.
  const [, setNow] = useState(0)
  useEffect(() => {
    const timer = window.setInterval(() => setNow((n) => n + 1), 1000)
    return () => window.clearInterval(timer)
  }, [])

  // Keep status and latency live without anyone pressing anything. The cadence matches the
  // server's own 30 s probe cache, so a poll usually costs one cached response and the model
  // endpoints see at most one real probe every 30 s no matter how many consoles are open.
  // Paused while the tab is hidden — a background tab probing production models all night is
  // cost with nobody reading it — and refreshed immediately when it comes back, because that is
  // the moment the numbers on screen are known to be stale.
  useEffect(() => {
    if (!session) return undefined
    const tick = () => { if (!document.hidden) void loadHealth(false, true) }
    const timer = window.setInterval(tick, HEALTH_POLL_MS)
    const onVisible = () => { if (!document.hidden) void loadHealth(false, true) }
    document.addEventListener('visibilitychange', onVisible)
    return () => { window.clearInterval(timer); document.removeEventListener('visibilitychange', onVisible) }
  }, [session, loadHealth])
  // Org-wide generation models (org admin sees the full registry; everyone else the read-only list
  // via the KB model-connections listing filtered to scope="org").
  const [orgConns, setOrgConns] = useState<ModelConnectionItem[] | null>(null)
  // An org-wide embedder switch waiting to be applied. Adding an embedding model org-wide cannot
  // take effect on its own: every KB's stored vectors came from the previous model, so the switch
  // is staged here and one apply re-embeds them.
  const [orgEmb, setOrgEmb] = useState<OrgEmbedderStatus | null>(null)
  const [orgEmbBusy, setOrgEmbBusy] = useState(false)
  // Org-wide model requests: the org-admin approval queue OR a manager's own request status.
  const [requests, setRequests] = useState<ModelRequestInfo[] | null>(null)
  const [deciding, setDeciding] = useState<string | null>(null)

  // Org-wide add/request form (org admin adds directly; a manager submits a request).
  const [orgName, setOrgName] = useState('')
  const [orgRole, setOrgRole] = useState<ModelRole>('generation')
  const [orgEndpoint, setOrgEndpoint] = useState('')
  const [orgModel, setOrgModel] = useState('')
  const [orgHeader, setOrgHeader] = useState('apikey')
  const [orgApiKey, setOrgApiKey] = useState('')
  const [orgBusy, setOrgBusy] = useState<'test' | 'save' | null>(null)
  const [orgTest, setOrgTest] = useState<ModelTestResult | null>(null)
  const [orgError, setOrgError] = useState<string | null>(null)
  // A blocked endpoint is not a dead end: the sovereignty allowlist has an approval workflow
  // (POST /v1/egress/requests). An org admin can approve their own request, so for them this is
  // one click; anyone else files it for an owner/admin to decide. The guard is never disabled —
  // approval adds ONLY this host.
  const [blockedHost, setBlockedHost] = useState<string | null>(null)
  const [requesting, setRequesting] = useState(false)

  const load = useCallback(async () => {
    if (!session || !kbId) return
    const { token } = session
    api.models(token).then(setPlatform).catch(() => setPlatform(null))
    api.systemInfo(token).then(setSysInfo).catch(() => setSysInfo(null))
    // Fetch managed KBs and use the RESULT directly below — never depend on the managedKbs STATE in
    // this callback, or setting it would recreate `load`, re-fire the effect, and loop forever
    // (the page flickers and inputs lose focus). Depend only on the auth/route inputs.
    const kbs = await api.myManagedKbs(token).catch(() => [] as string[])
    setManagedKbs(kbs)
    // The org registry: org admin reads the authoritative list; everyone else sees org-wide models
    // read-only via the KB's model-connections list (scope="org").
    if (isOrgAdmin) {
      api.listOrgModelConnections(token).then(setOrgConns).catch(() => setOrgConns([]))
      api.orgEmbedderStatus(token).then(setOrgEmb).catch(() => setOrgEmb(null))
    } else {
      api.listModelConnections(token, kbId)
        .then((list) => setOrgConns(list.filter((c) => c.scope === 'org' && c.role === 'generation')))
        .catch(() => setOrgConns([]))
    }
    // Org-wide model requests: admin gets all pending; a manager gets their own. A plain member 403s
    // (harmless — swallow to []).
    if (isOrgAdmin || kbs.length > 0) {
      api.listMyModelRequests(token).then(setRequests).catch(() => setRequests([]))
    } else {
      setRequests([])
    }
  }, [session, kbId, isOrgAdmin])

  useEffect(() => { void load() }, [load])

  // --- Org-wide model handlers ---
  // An embedder without its dimension (or a reranker without its wire shape) cannot run
  // deterministically, so the server refuses it — the Test probe is what supplies both, and the
  // form must not offer a save that is going to 422.
  const probedDim = orgTest?.ok ? orgTest.dim ?? null : null
  const probedShape = orgTest?.ok ? orgTest.api_shape ?? null : null
  const roleProbeMissing =
    (orgRole === 'embedding' && probedDim === null) || (orgRole === 'reranking' && !probedShape)
  const orgReady = !!(orgName.trim() && orgEndpoint.trim() && orgModel.trim()) && !roleProbeMissing
  function resetOrgForm() {
    setOrgName(''); setOrgEndpoint(''); setOrgModel(''); setOrgHeader('apikey'); setOrgApiKey(''); setOrgTest(null); setOrgError(null); setOrgRole('generation')
  }
  async function runOrgTest() {
    if (!session || !orgEndpoint.trim() || !orgModel.trim()) return
    setOrgBusy('test'); setOrgError(null); setOrgTest(null)
    try {
      setOrgTest(await api.testOrgModelConnection(session.token, { endpoint: orgEndpoint.trim(), model: orgModel.trim(), header: orgHeader.trim(), api_key: orgApiKey || undefined, role: orgRole }))
    } catch (err) {
      setOrgError(err instanceof ApiError ? err.message : 'Test failed')
    } finally { setOrgBusy(null) }
  }
  function hostOf(s: string): string {
    try { return new URL(s.includes('://') ? s : `https://${s}`).hostname } catch { return s }
  }

  async function requestHostApproval() {
    if (!session || !blockedHost) return
    setRequesting(true)
    try {
      await api.requestEgress(session.token, blockedHost, `Org-wide model ${orgName.trim() || orgModel.trim()}`)
      toast.push('success', `Approval requested for ${blockedHost}. An owner/admin will review it.`)
      setBlockedHost(null); setOrgError(null)
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not submit request.')
    } finally { setRequesting(false) }
  }

  // Org-admin path: file the request and decide it, then retry the save. The request carries no
  // kb_id — an org-wide model is not scoped to a KB — so only an owner/admin can decide it.
  async function allowHostAndAdd() {
    if (!session || !blockedHost) return
    setRequesting(true)
    try {
      const req = await api.requestEgress(session.token, blockedHost, `Org-wide model ${orgName.trim() || orgModel.trim()}`)
      await api.decideEgress(session.token, req.id, 'approve')
      toast.push('success', `${blockedHost} allowlisted.`)
      setBlockedHost(null); setOrgError(null)
      await addOrg()
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not allow the host.')
    } finally { setRequesting(false) }
  }

  async function addOrg() {
    if (!session || !orgReady) return
    setOrgBusy('save'); setOrgError(null)
    try {
      const res = await api.addOrgModelConnection(session.token, {
        name: orgName.trim(), endpoint: orgEndpoint.trim(), model: orgModel.trim(), header: orgHeader.trim(),
        api_key: orgApiKey || undefined, role: orgRole,
        embed_dim: probedDim ?? undefined, api_shape: probedShape ?? undefined,
        // Adding a model org-wide PUTS IT IN FORCE for its role — that is what adding it means.
        // Defaults are per role, so an embedder never unseats the generation model. The previous
        // default stays in the list and is one click away. (An embedder is staged, not swapped:
        // see the re-embed banner.)
        make_default: true,
      })
      resetOrgForm()
      await load()
      if (isPendingModelRequest(res)) {
        toast.push('info', 'Requested — awaiting org-admin approval.')
      } else {
        toast.push('success', 'Org-wide model added — available in every KB.')
      }
    } catch (err) {
      if (err instanceof ApiError && err.code === 'sovereignty_violation') {
        setBlockedHost(hostOf(orgEndpoint.trim()))
        setOrgError('This endpoint is not on the sovereignty allowlist.')
      } else {
        const msg = err instanceof ApiError ? err.message : 'Save failed'
        setOrgError(msg); toast.push('error', msg)
      }
    } finally { setOrgBusy(null) }
  }
  async function removeOrg(conn: ModelConnectionItem) {
    if (!session || !window.confirm(`Remove org-wide model "${conn.name}"?`)) return
    setActing(conn.conn_id)
    try {
      await api.deleteOrgModelConnection(session.token, conn.conn_id)
      await load()
      toast.push('info', `Removed org-wide model "${conn.name}".`)
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Remove failed')
    } finally { setActing(null) }
  }
  async function makeOrgDefault(conn: ModelConnectionItem) {
    if (!session) return
    setActing(conn.conn_id)
    try {
      // Set-default via update: PUT the same fields with make_default=true (blank api_key keeps it).
      await api.updateOrgModelConnection(session.token, conn.conn_id, {
        name: conn.name, endpoint: conn.endpoint, model: conn.model, header: conn.header,
        role: conn.role, make_default: true,
      })
      await load()
      toast.push('success', 'Org-wide default updated — every KB now answers with it.')
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Failed')
    } finally { setActing(null) }
  }

  async function decideOrg(req: ModelRequestInfo, decision: 'approve' | 'reject') {
    if (!session) return
    const label = req.payload && typeof req.payload.name === 'string' ? req.payload.name : 'this model'
    if (decision === 'reject' && !window.confirm(`Reject the request to add "${label}" org-wide?`)) return
    setDeciding(req.id)
    try {
      await api.decideOrgModelRequest(session.token, req.id, decision)
      await load()
      toast.push(decision === 'approve' ? 'success' : 'info', decision === 'approve' ? 'Request approved — model added org-wide.' : 'Request rejected.')
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Decision failed')
    } finally { setDeciding(null) }
  }

  const orgDefault = orgConns?.find((c) => c.is_default && c.role === 'generation') ?? null
  const pendingRequests = (requests ?? []).filter((r) => r.status === 'pending')

  async function applyOrgEmbedder() {
    if (!session || !orgEmb?.pending) return
    const docs = orgEmb.documents
    if (!window.confirm(
      `Re-embed ${docs} document${docs === 1 ? '' : 's'} across ${orgEmb.kbs.length} knowledge base`
      + `${orgEmb.kbs.length === 1 ? '' : 's'} with ${orgEmb.pending.model}?\n\n`
      + 'Every vector is rebuilt. While a knowledge base rebuilds, its answers come from keyword '
      + 'search and reranking only. This cannot be undone except by switching back and re-embedding again.',
    )) return
    setOrgEmbBusy(true)
    try {
      const res = await api.orgEmbedderApply(session.token)
      const reindexed = res.applied.reduce((n, a) => n + a.reindexed, 0)
      if (res.failed.length) {
        toast.push('error', `Re-embedded ${reindexed} document(s); ${res.failed.length} knowledge base(s) failed — see the list.`)
      } else {
        toast.push('success', `Re-embedded ${reindexed} document(s) across ${res.applied.length} knowledge base(s).`)
      }
      await load()
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Apply failed')
    } finally { setOrgEmbBusy(false) }
  }

  /** The staged org-wide embedder: what it would move, and the one click that moves it. */
  function OrgEmbedderBanner() {
    if (!isOrgAdmin || !orgEmb?.pending) return null
    return (
      <Card className="mb-5">
        <div className="flex items-center gap-2 mb-1">
          <span style={{ color: 'var(--warning)' }}><Icon.alert size={16} /></span>
          <h2 className="font-display font-semibold" style={{ fontSize: 16 }}>Re-embed pending</h2>
        </div>
        <p className="field-help mb-3">
          <strong className="mono">{orgEmb.pending.model}</strong> is registered as this organisation&apos;s embedding
          model, but the documents already indexed were embedded with the previous one. Vectors from two different
          models cannot be compared, so the switch waits here: applying it re-embeds{' '}
          <strong>{orgEmb.documents}</strong> document{orgEmb.documents === 1 ? '' : 's'} across{' '}
          <strong>{orgEmb.kbs.length}</strong> knowledge base{orgEmb.kbs.length === 1 ? '' : 's'}. Until then every
          knowledge base keeps answering with the embedder that built it.
        </p>
        <ul className="mb-3 space-y-1">
          {orgEmb.kbs.map((k) => (
            <li key={k.kb_id} className="text-sm mono" style={{ color: 'var(--text-muted)' }}>
              {k.kb_id} · {k.documents} document{k.documents === 1 ? '' : 's'}
            </li>
          ))}
        </ul>
        {orgEmb.apply_enabled
          ? (
            <Button loading={orgEmbBusy} disabled={orgEmbBusy} onClick={() => void applyOrgEmbedder()} icon={<Icon.play size={16} />}>
              Re-embed every knowledge base
            </Button>
          )
          : <Alert kind="warning">The embedder switch is disabled on this deployment (RAG_EMBEDDER_SWITCH_ENABLED).</Alert>}
      </Card>
    )
  }

  // Shared read-only org-wide models list (used by manager + member views, and inside the admin card).
  function OrgModelsList({ canManage }: { canManage: boolean }) {
    if (orgConns === null) return <Skeleton style={{ height: 100 }} />
    if (orgConns.length === 0) {
      return (
        <div className="text-sm rounded-lg p-4" style={{ color: 'var(--text-muted)', background: 'var(--surface-2)', border: '1px dashed var(--border-strong)' }}>
          No org-wide models yet — answers use the platform default (<span className="mono">{platform?.active_model ?? '…'}</span>).
        </div>
      )
    }
    return (
      <ul className="space-y-2">
        {orgConns.map((c) => (
          <li key={c.conn_id} className="flex items-center gap-3 p-3 rounded-lg" style={{ border: `1px solid ${c.is_default ? 'color-mix(in srgb, var(--success) 40%, var(--border))' : 'var(--border)'}`, background: c.is_default ? 'color-mix(in srgb, var(--success) 6%, transparent)' : 'transparent' }}>
            <Icon.layers size={18} style={{ color: c.is_default ? 'var(--success)' : 'var(--action)' }} />
            <div className="flex-grow min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-semibold" style={{ color: 'var(--text-heading)' }}>{c.name}</span>
                <span className="pill" style={{ fontSize: 10, color: 'var(--action)', borderColor: 'color-mix(in srgb, var(--action) 35%, transparent)', background: 'color-mix(in srgb, var(--action) 12%, transparent)' }}>Org-wide</span>
                <span className="pill" style={{ fontSize: 10, color: 'var(--text-muted)' }}>{ORG_ROLES.find((r) => r.role === c.role)?.label ?? c.role}</span>
                {c.is_default && <span className="pill" style={{ fontSize: 10, color: 'var(--success)', borderColor: 'color-mix(in srgb, var(--success) 35%, transparent)', background: 'color-mix(in srgb, var(--success) 12%, transparent)' }}><Icon.check size={11} />default for {ORG_ROLES.find((r) => r.role === c.role)?.label.toLowerCase() ?? c.role}</span>}
              </div>
              <div className="mono truncate" style={{ fontSize: 11, color: 'var(--text-muted)' }}>{c.model} · {c.endpoint}</div>
            </div>
            {canManage && !c.is_default && <Button size="sm" variant="subtle" disabled={acting !== null} onClick={() => makeOrgDefault(c)}>Make default</Button>}
            {canManage && <button type="button" onClick={() => removeOrg(c)} disabled={acting !== null} aria-label={`Remove ${c.name}`} title="Remove" className="btn btn-subtle btn-icon" style={{ color: 'var(--danger)' }}><Icon.trash size={15} /></button>}
          </li>
        ))}
      </ul>
    )
  }


  // Shared "Models in use" table (reads live deployment config; reflects the org-wide generation
  // default as the effective generation model). Read-only. The SERVER decides how many rows an
  // account may see: generation only for members, the full roster for org owners/admins.
  function ModelsInUse() {
    if (!sysInfo?.models || sysInfo.models.length === 0) return null
    return (
      <Card className="mb-5">
        <div className="flex items-center gap-2 mb-1">
          <span style={{ color: 'var(--action)' }}><Icon.database size={16} /></span>
          <h2 className="font-display font-semibold" style={{ fontSize: 16 }}>Models in use</h2>
          <span className="flex-1" />
          {/* When the reading was taken. A status with no timestamp cannot be told from a stale
              one, and "online" is exactly the word you do not want to trust blindly. */}
          {health && (
            <span className="text-xs mono flex items-center gap-1.5" style={{ color: 'var(--text-muted)' }} title={new Date(health.checked_at).toLocaleString()}>
              {health.stale && (
                <span className="pill" title="The background snapshot is older than several poll intervals — the reading may not be current." style={{ fontSize: 10, color: 'var(--warning)', borderColor: 'color-mix(in srgb, var(--warning) 35%, transparent)', background: 'color-mix(in srgb, var(--warning) 12%, transparent)' }}>stale</span>
              )}
              {checking ? 'checking…' : `${health.source === 'poller' && !health.stale ? 'live · ' : ''}updated ${sinceLabel(health.checked_at)}`}
            </span>
          )}
          <Button variant="subtle" size="sm" onClick={() => void loadHealth(true)} disabled={checking}>
            {checking ? 'Checking…' : 'Re-check'}
          </Button>
        </div>
        <p className="field-help mb-4">What actually serves <strong>this knowledge base</strong>. Generation uses the org-wide default when one is set. Embedding, reranking and OCR are platform infrastructure used during ingestion and retrieval — identical for every KB, and listed here for org admins only. Read live from configuration; keys are never shown.</p>
        <Table>
          <thead><tr><th>Role</th><th>Model</th><th>Backend</th><th>Endpoint</th><th>Status</th><th>Latency</th><th>Detail</th></tr></thead>
          <tbody>
            {sysInfo.models.map((m, i) => {
              // The roster row reads CONFIGURATION. The probe says what this KB's request actually
              // reached — its own attach, or the org-wide default of that role — so where the two
              // disagree the probe wins, or the table would name one model and grade another.
              const probed = health?.backends.find((b) => b.role === m.role)
              const ov = m.role === 'Generation' ? orgDefault : undefined
              const resolved = !!probed?.model && probed.model !== m.model
              const mdl = probed?.model || (ov ? ov.model : m.model)
              const backend = ov || resolved ? 'openai_compat' : m.backend
              const endpoint = probed?.endpoint || (ov ? (ov.endpoint || '').replace(/^https?:\/\//, '').split('/')[0] : m.endpoint)
              const detail = ov ? (ov.name || m.detail) : m.detail
              return (
                <tr key={`${m.role}-${i}`}>
                  <td className="font-medium" style={{ color: 'var(--text-heading)' }}>
                    <span className="flex items-center gap-2">
                      {m.role}
                      {!ov && resolved && <span className="pill" style={{ fontSize: 10, color: 'var(--action)', borderColor: 'color-mix(in srgb, var(--action) 35%, transparent)', background: 'color-mix(in srgb, var(--action) 12%, transparent)' }}>This KB</span>}
                      {ov && <span className="pill" style={{ fontSize: 10, color: 'var(--action)', borderColor: 'color-mix(in srgb, var(--action) 35%, transparent)', background: 'color-mix(in srgb, var(--action) 12%, transparent)' }}>Org-wide</span>}
                    </span>
                  </td>
                  <td className="mono text-xs" style={{ color: 'var(--text-primary)' }}>{mdl}</td>
                  <td className="mono text-xs" style={{ color: 'var(--text-muted)' }}>{backend}</td>
                  <td className="mono text-xs" style={{ color: 'var(--text-muted)' }}>{endpoint || '—'}</td>
                  {(() => {
                    // A role with no probe is a local, in-cluster backend with nothing to reach
                    // over the network — say so, do not imply failure.
                    const probe = probed
                    const colour = !probe ? 'var(--text-muted)' : probe.online ? 'var(--success)' : 'var(--danger)'
                    const label = !probe ? 'not probed' : probe.online ? 'online' : 'offline'
                    return (
                      <>
                        <td className="text-xs" style={{ color: colour }} title={probe?.detail || undefined}>
                          <span className="flex items-center gap-1.5" style={{ fontWeight: probe && !probe.online ? 600 : 400 }}>
                            <span style={{ width: 7, height: 7, borderRadius: 999, background: colour, display: 'inline-block' }} />
                            {label}
                          </span>
                          {/* An offline backend was probably up before — show when, so an operator can
                              tell a just-now blip from a long outage. From the continuous poller. */}
                          {probe && !probe.online && probe.last_online && (
                            <span className="mono" style={{ display: 'block', fontSize: 10, color: 'var(--text-muted)' }}>
                              last up {sinceLabel(probe.last_online)}
                            </span>
                          )}
                        </td>
                        <td className="mono text-xs" style={{ color: 'var(--text-muted)' }}>
                          {probe?.latency_ms != null ? `${probe.latency_ms} ms` : '—'}
                        </td>
                      </>
                    )
                  })()}
                  <td className="text-xs" style={{ color: 'var(--text-muted)' }}>{probeDetail(health, m.role) || detail || '—'}</td>
                </tr>
              )
            })}
          </tbody>
        </Table>
      </Card>
    )
  }

  // The add/request form — shared shape for org admin (adds directly) and KB manager (requests).
  function OrgModelForm({ mode }: { mode: 'add' | 'request' }) {
    return (
      <>
        <div className="mt-2">
          <Field label="What this model does" help="The role decides how the platform uses it. A role's org-wide default is what every KB gets.">
            <div className="flex flex-wrap gap-2">
              {ORG_ROLES.map((r) => {
                const sel = r.role === orgRole
                return (
                  <button key={r.role} type="button" title={r.help}
                    onClick={() => { setOrgRole(r.role); setOrgTest(null); setOrgError(null) }}
                    className="chip interactive" style={{ cursor: 'pointer', fontSize: 12,
                      borderColor: sel ? 'var(--action)' : 'var(--border)',
                      background: sel ? 'color-mix(in srgb, var(--action) 12%, transparent)' : 'var(--surface-2)',
                      color: sel ? 'var(--action)' : 'var(--text-primary)' }}>
                    {sel && <Icon.check size={12} />}{r.label}
                  </button>
                )
              })}
            </div>
          </Field>
          <p className="field-help mt-2">{ORG_ROLES.find((r) => r.role === orgRole)?.help}</p>
        </div>
        <div className="mt-4">
          <Field label="Name" help="A label for this org-wide model.">
            <input className="field-input" value={orgName} onChange={(e) => setOrgName(e.target.value)} placeholder="AIOps Qwen3-32B (shared)" />
          </Field>
        </div>
        <div className="mt-4">
          <Field label="Endpoint URL" help="OpenAI-compatible base, e.g. https://models-dev.example.internal/openai/v1">
            <input className="field-input" value={orgEndpoint} onChange={(e) => setOrgEndpoint(e.target.value)} placeholder="https://models-dev.example.internal/openai/v1" />
          </Field>
        </div>
        <div className="grid sm:grid-cols-2 gap-4 mt-4">
          <Field label="Model / deployment" help="The deployment id from Test."><input className="field-input" value={orgModel} onChange={(e) => setOrgModel(e.target.value)} placeholder="qwendeploy" /></Field>
          <Field label="API key header" help="AIOps uses 'apikey'."><input className="field-input" value={orgHeader} onChange={(e) => setOrgHeader(e.target.value)} /></Field>
        </div>
        <div className="mt-4">
          <Field label="API key" help="Stored encrypted at rest — never shown again.">
            <input className="field-input" type="password" value={orgApiKey} onChange={(e) => setOrgApiKey(e.target.value)} placeholder="paste your key" />
          </Field>
        </div>
        <div className="flex gap-2 mt-6">
          <Button variant="ghost" loading={orgBusy === 'test'} disabled={!orgEndpoint.trim() || !orgModel.trim() || orgBusy !== null} onClick={runOrgTest} icon={<Icon.play size={16} />}>Test connection</Button>
          <Button loading={orgBusy === 'save'} disabled={!orgReady || orgBusy !== null} onClick={addOrg} icon={<Icon.check size={16} />}>
            {mode === 'add' ? 'Add org-wide model' : 'Request org-wide model'}
          </Button>
          {roleProbeMissing && (
            <span className="field-help self-center">
              {orgRole === 'embedding'
                ? 'Test first — the dimension is read from the endpoint, never typed.'
                : 'Test first — the rerank wire shape is read from the endpoint, never typed.'}
            </span>
          )}
        </div>
        {orgTest && (orgTest.ok
          ? (
            <div className="mt-4">
              <Alert kind="success">{
                orgRole === 'embedding' ? `Connected. Returns ${probedDim ?? '?'} dimensions.`
                  : orgRole === 'reranking' ? `Connected. Wire shape ${probedShape ?? '?'}.`
                    : `Connected.${orgTest.available.length ? ' Pick the model this endpoint serves:' : ' Endpoint reachable (no model catalog reported).'}`
              }</Alert>
              {orgTest.available.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-3">
                  {orgTest.available.map((mm) => {
                    const sel = mm === orgModel.trim()
                    return (
                      <button key={mm} type="button" onClick={() => setOrgModel(mm)}
                        className="chip mono interactive" style={{ cursor: 'pointer', fontSize: 12,
                          borderColor: sel ? 'var(--success)' : 'var(--border)',
                          background: sel ? 'color-mix(in srgb, var(--success) 12%, transparent)' : 'var(--surface-2)',
                          color: sel ? 'var(--success)' : 'var(--text-primary)' }}>
                        {sel && <Icon.check size={12} />}{mm}
                      </button>
                    )
                  })}
                </div>
              )}
            </div>
          )
          : <div className="mt-4"><Alert kind="error">Couldn&apos;t connect: <span className="mono">{orgTest.error}</span></Alert></div>)}
        {orgError && (
          <div className="mt-4">
            <Alert kind="error">{orgError}</Alert>
            {blockedHost && (
              <div className="mt-3 flex items-center gap-3 flex-wrap">
                {isOrgAdmin ? (
                  <>
                    <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
                      Allow <span className="mono" style={{ color: 'var(--text-heading)' }}>{blockedHost}</span> — you are an org admin, so this takes effect immediately. Only this host is added to the sovereignty allowlist; the guard stays enforced for everything else, and the decision is audited.
                    </span>
                    <Button size="sm" loading={requesting} onClick={allowHostAndAdd} icon={<Icon.shield size={15} />}>Allow host &amp; add model</Button>
                  </>
                ) : (
                  <>
                    <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
                      Request approval for <span className="mono" style={{ color: 'var(--text-heading)' }}>{blockedHost}</span> — an owner/admin reviews it (the guard stays on).
                    </span>
                    <Button size="sm" variant="secondary" loading={requesting} onClick={requestHostApproval} icon={<Icon.shield size={15} />}>Request approval</Button>
                  </>
                )}
              </div>
            )}
          </div>
        )}
      </>
    )
  }

  const header = (
    <PageHeader eyebrow={t('eyebrow.platform')} title={t('page.models.title')} lead={t('page.models.lead')} />
  )

  // The "Platform default" explanatory card was removed — Sarvam is the org-wide default anyway,
  // so surfacing a separate platform-default panel was redundant.
  void platform

  // --- ORG ADMIN VIEW: manage the registry + approve requests. ---
  if (isOrgAdmin) {
    return (
      <div>
        {header}

        {pendingRequests.length > 0 && (
          <Card className="mb-5">
            <div className="flex items-center gap-2 mb-1">
              <span style={{ color: 'var(--warning)' }}><Icon.alert size={16} /></span>
              <h2 className="font-display font-semibold" style={{ fontSize: 16 }}>Pending model requests</h2>
            </div>
            <p className="field-help mb-4">A KB manager requested these org-wide models. Approving <strong>adds</strong> the model to the org registry; rejecting discards it.</p>
            <ul className="space-y-2">
              {pendingRequests.map((r) => (
                <li key={r.id} className="flex items-center gap-3 p-3 rounded-lg" style={{ border: '1px solid var(--border)', background: 'var(--surface-2)' }}>
                  <Icon.layers size={18} style={{ color: 'var(--text-muted)' }} />
                  <div className="flex-grow min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-semibold" style={{ color: 'var(--text-heading)' }}>Add org-wide model</span>
                      {r.payload && typeof r.payload.name === 'string' ? <span className="pill" style={{ fontSize: 10 }}>{r.payload.name}</span> : null}
                    </div>
                    <div className="mono truncate" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      by {r.requested_by}
                      {r.payload && typeof r.payload.model === 'string' ? ` · ${r.payload.model}` : ''}
                      {r.payload && typeof r.payload.endpoint === 'string' ? ` · ${r.payload.endpoint}` : ''}
                    </div>
                  </div>
                  <Button size="sm" loading={deciding === r.id} disabled={deciding !== null} onClick={() => decideOrg(r, 'approve')} icon={<Icon.check size={14} />}>Approve</Button>
                  <Button size="sm" variant="subtle" disabled={deciding !== null} onClick={() => decideOrg(r, 'reject')} style={{ color: 'var(--danger)' }}>Reject</Button>
                </li>
              ))}
            </ul>
          </Card>
        )}

        {OrgEmbedderBanner()}

        <div className="grid lg:grid-cols-3 gap-5">
          <div className="lg:col-span-3 space-y-5">
            <Card>
              <div className="flex items-center gap-2 mb-1"><Icon.database size={16} style={{ color: 'var(--action)' }} /><h2 className="font-display font-semibold" style={{ fontSize: 16 }}>Org-wide models</h2></div>
              <p className="field-help mb-4">Models available to <strong>every KB</strong> in the organisation, one default per job. Generation answers questions; embedding, reranking, OCR and vision do the background work of ingesting and retrieving. The one marked <span className="pill" style={{ fontSize: 10 }}>default</span> for a job is what the whole organisation uses for it.</p>
              <div className="mb-5">{OrgModelsList({ canManage: true })}</div>
              <div className="pt-2" style={{ borderTop: '1px solid var(--border)' }}>
                <div className="flex items-center gap-2 mt-4 mb-1"><Icon.plus size={16} style={{ color: 'var(--text-muted)' }} /><h3 className="font-display font-semibold" style={{ fontSize: 15 }}>Add a model</h3></div>
                <p className="field-help mb-2">Connect an OpenAI-compatible endpoint for the whole organisation, and say which job it does. Adding it puts it in force for that job across every KB.</p>
                {OrgModelForm({ mode: 'add' })}
              </div>
            </Card>
          </div>
        </div>
      {ModelsInUse()}
      </div>
    )
  }

  // --- KB MANAGER VIEW: request an org-wide model + watch status + read-only registry. ---
  if (isManager) {
    return (
      <div>
        {header}
        <div className="mb-5">
          <Alert kind="info">Generation models are org-wide. As a KB manager you can <strong>request</strong> a new org-wide model — it becomes available once an org admin approves it.</Alert>
        </div>

        {requests && requests.length > 0 && (
          <Card className="mb-5">
            <div className="flex items-center gap-2 mb-1">
              <span style={{ color: 'var(--action)' }}><Icon.activity size={16} /></span>
              <h2 className="font-display font-semibold" style={{ fontSize: 16 }}>Your requests</h2>
            </div>
            <p className="field-help mb-4">Org-wide models you requested apply only after an org admin approves them.</p>
            <ul className="space-y-2">
              {requests.map((r) => {
                const color = r.status === 'approved' ? 'var(--success)' : r.status === 'rejected' ? 'var(--danger)' : 'var(--warning)'
                const label = r.status === 'pending' ? 'awaiting org-admin approval' : r.status
                return (
                  <li key={r.id} className="flex items-center gap-3 p-3 rounded-lg" style={{ border: '1px solid var(--border)' }}>
                    <div className="flex-grow min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-semibold" style={{ color: 'var(--text-heading)' }}>{r.payload && typeof r.payload.name === 'string' ? r.payload.name : 'Org-wide model'}</span>
                        <span className="pill" title={r.reason || undefined}
                          style={{ fontSize: 10, color, borderColor: `color-mix(in srgb, ${color} 35%, transparent)`, background: `color-mix(in srgb, ${color} 12%, transparent)` }}>
                          {label}
                        </span>
                      </div>
                      <div className="mono truncate" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                        {r.payload && typeof r.payload.model === 'string' ? r.payload.model : ''}
                        {r.decided_by ? ` · decided by ${r.decided_by}` : ''}
                      </div>
                    </div>
                  </li>
                )
              })}
            </ul>
          </Card>
        )}

        <div className="grid lg:grid-cols-3 gap-5">
          <div className="lg:col-span-3 space-y-5">
            <Card>
              <div className="flex items-center gap-2 mb-1"><Icon.database size={16} style={{ color: 'var(--action)' }} /><h2 className="font-display font-semibold" style={{ fontSize: 16 }}>Org-wide models</h2></div>
              <p className="field-help mb-4">Generation models available across the organisation. The <span className="pill" style={{ fontSize: 10 }}>default</span> is what your KB answers with.</p>
              {OrgModelsList({ canManage: false })}
            </Card>
            <Card>
              <div className="flex items-center gap-2 mb-1"><Icon.plus size={16} style={{ color: 'var(--text-muted)' }} /><h2 className="font-display font-semibold" style={{ fontSize: 16 }}>Request an org-wide model</h2></div>
              <p className="field-help mb-4">Submit a model for org-admin approval, saying which job it does. Once approved it is in force for that job across every KB.</p>
              {OrgModelForm({ mode: 'request' })}
            </Card>
          </div>
        </div>
      {ModelsInUse()}
      </div>
    )
  }

  // --- MEMBER VIEW: read-only org-wide models + default. ---
  return (
    <div>
      {header}
      <div className="mb-5">
        <Alert kind="info">You have read-only access to the organisation's models. To change them, ask an org admin.</Alert>
      </div>
      <Card>
        <div className="flex items-center gap-2 mb-1"><Icon.database size={16} style={{ color: 'var(--action)' }} /><h2 className="font-display font-semibold" style={{ fontSize: 16 }}>Org-wide models</h2></div>
        <p className="field-help mb-4">The generation models this organisation can answer with. The <span className="pill" style={{ fontSize: 10 }}>default</span> is what every KB uses.</p>
        {OrgModelsList({ canManage: false })}
      </Card>
      {ModelsInUse()}
    </div>
  )
}
