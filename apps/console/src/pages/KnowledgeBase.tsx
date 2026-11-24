import { useCallback, useEffect, useRef, useState } from 'react'
import { useI18n } from '../lib/i18n'
import { Icon } from '../components/icons'
import { useToast } from '../components/overlay'
import { Alert, Button, Card, EmptyState, Field, PageHeader, SegmentedControl, Skeleton, Table } from '../components/ui'
import { Uploader } from '../components/Uploader'
import { RolesCard } from '../components/RolesCard'
import { GraphCanvas } from '../components/GraphCanvas'
// Lazy: three.js (~600 KB) only downloads when a user actually switches a KB to 3D.
import { PipelineDag } from '../components/PipelineDag'
import { CitationViewer } from '../components/CitationViewer'
import { useIngestJobs } from '../lib/ingestJobs'
import { ApiError, SHARED_KB_ID, api, type DocumentSummary, type GraphCommunityInfo, type GraphStatsInfo, type GraphSubgraph, type IndexStatus, type KbSettingsInfo, type SystemInfo, type UsageReport } from '../lib/api'
import { hasRole, useAuth } from '../lib/auth'
import { useKb } from '../lib/kb'
import { useRoleLabels } from '../lib/roleLabels'

type Tab = 'documents' | 'settings' | 'integrations' | 'knowledge'

export function KnowledgeBase() {
  const { session } = useAuth()
  const { t } = useI18n()
  const { kbs, kbId, setKbId, create, refresh } = useKb()
  const toast = useToast()
  const ingest = useIngestJobs()
  const roleLabels = useRoleLabels()
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [deletingKb, setDeletingKb] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [docs, setDocs] = useState<DocumentSummary[] | null>(null)
  const [usage30, setUsage30] = useState<UsageReport | null>(null)
  const [docSearch, setDocSearch] = useState('')
  const [acting, setActing] = useState<string | null>(null)
  // The doc just uploaded in THIS session — the pipeline traces only this one, so it animates on a
  // real upload and stays idle (no replay) on a plain page visit.
  const [lastUploadedDocId, setLastUploadedDocId] = useState('')
  // Which document the reader opened. What is shown is the EXTRACTED text — the chunks that were
  // actually indexed — not a render of the original file: it is what retrieval can see, so it is
  // what tells you whether an answer could have come from this document. For a scan or an image
  // that is also the only honest view, since the file itself carries no text at all.
  const [viewDoc, setViewDoc] = useState('')
  // FR-C8: whether these documents were built by the indexing pipeline running NOW. A document
  // indexed by older code still answers -- from chunks this deployment would no longer produce --
  // and nothing else on this page would say so.
  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null)
  const [reindexing, setReindexing] = useState(false)
  const [settings, setSettings] = useState<KbSettingsInfo | null>(null)
  const [thr, setThr] = useState<{ abstain: string; faithful: string; flag: string }>({ abstain: '', faithful: '', flag: '' })
  const [savingThr, setSavingThr] = useState(false)
  const [persona, setPersona] = useState('')
  const [savingPersona, setSavingPersona] = useState(false)
  const [widgetCfg, setWidgetCfg] = useState<import('../lib/api').WidgetConfig | null>(null)
  const [savingWidget, setSavingWidget] = useState(false)
  const [botCfg, setBotCfg] = useState<import('../lib/api').BotConfig | null>(null)
  const [botToken, setBotToken] = useState<string | null>(null)
  const [taxonomyText, setTaxonomyText] = useState('')
  const [savingTax, setSavingTax] = useState(false)
  const [graph, setGraph] = useState<GraphStatsInfo | null>(null)
  const [subgraph, setSubgraph] = useState<GraphSubgraph | null>(null)
  const [communities, setCommunities] = useState<GraphCommunityInfo[] | null>(null)
  const [buildingComs, setBuildingComs] = useState(false)
  const [activeCommunity, setActiveCommunity] = useState<number | null>(null)
  const [hoverCommunity, setHoverCommunity] = useState<number | null>(null)
  const [tab, setTab] = useState<Tab>('documents')
  const [trash, setTrash] = useState<{ kb_id: string; name: string; trashed_at: string | null }[]>([])
  const [sys, setSys] = useState<SystemInfo | null>(null)

  // Backend gates ingest on `ingestor`, widget/bot config writes on `admin` — mirror both here
  // so a reader sees why the upload zone / integration cards are unavailable instead of
  // clicking through to a 403.
  const canIngest = hasRole(session, 'ingestor', 'admin')
  const canAdmin = hasRole(session, 'admin')

  // Platform-level UI toggles (non-secret) — decides whether optional tabs are surfaced.
  useEffect(() => {
    if (!session) return
    api.systemInfo(session.token).then(setSys).catch(() => setSys(null))
  }, [session])

  const loadDocs = useCallback(async (quiet = false) => {
    if (!session || !kbId) return
    if (!quiet) setDocs(null) // skeleton on first load only; live refreshes update in place
    try {
      setDocs(await api.listDocuments(session.token, kbId))
    } catch {
      if (!quiet) setDocs([])
    }
  }, [session, kbId])

  const loadTrash = useCallback(async () => {
    if (!session) return
    try { setTrash(await api.listTrashedKbs(session.token)) } catch { setTrash([]) }
  }, [session])

  const loadIndexStatus = useCallback(async () => {
    if (!session || !kbId) return
    try { setIndexStatus(await api.indexStatus(session.token, kbId)) } catch { setIndexStatus(null) }
  }, [session, kbId])

  useEffect(() => { void loadDocs() }, [loadDocs])
  useEffect(() => { void loadIndexStatus() }, [loadIndexStatus])
  useEffect(() => { void loadTrash() }, [loadTrash])

  // The ingest poller lives in a context above the router, so it knows the moment a job finishes —
  // but nothing told THIS page to re-read. The result was a completed pipeline (all four stages
  // green, chain verified) sitting next to "0 Documents" and an empty list, until a manual reload.
  // Re-fetch the documents and the KB stats on the transition into a terminal phase.
  const jobPhase = ingest.jobForKb(kbId)?.phase
  const prevPhase = useRef<string | undefined>(undefined)
  useEffect(() => {
    const was = prevPhase.current
    prevPhase.current = jobPhase
    if (was === 'processing' && (jobPhase === 'done' || jobPhase === 'failed')) {
      void loadDocs(true)   // quiet: update in place, no skeleton flash
      void refresh()        // KB list carries the document/query counters
      void loadIndexStatus() // a fresh ingest is stamped with the running pipeline
    }
  }, [jobPhase, loadDocs, refresh, loadIndexStatus])

  useEffect(() => {
    if (!session || !kbId) return
    setSettings(null)
    api.getKbSettings(session.token, kbId).then((info) => {
      setSettings(info)
      setThr({
        abstain: info.overrides.abstain_threshold?.toString() ?? '',
        faithful: info.overrides.faithful_threshold?.toString() ?? '',
        flag: info.overrides.flag_threshold?.toString() ?? '',
      })
    }).catch(() => setSettings(null))
    api.getPersona(session.token, kbId).then((p) => setPersona(p.persona)).catch(() => setPersona(''))
    setWidgetCfg(null)
    api.getWidgetConfig(session.token, kbId).then(setWidgetCfg).catch(() => setWidgetCfg(null))
    setBotCfg(null); setBotToken(null)
    api.getBotConfig(session.token, kbId).then(setBotCfg).catch(() => setBotCfg(null))
    api.getTaxonomy(session.token, kbId).then((tax) => {
      setTaxonomyText(Object.entries(tax.taxonomy).map(([label, kws]) => `${label}: ${kws.join(', ')}`).join('\n'))
    }).catch(() => setTaxonomyText(''))
    setUsage30(null)
    api.usage(session.token, kbId, '30d').then(setUsage30).catch(() => setUsage30(null))
    api.getGraph(session.token, kbId).then(setGraph).catch(() => setGraph(null))
    setSubgraph(null)
    api.getGraphSubgraph(session.token, kbId, 60).then(setSubgraph).catch(() => setSubgraph(null))
    setCommunities(null)
    setActiveCommunity(null); setHoverCommunity(null)
    api.getCommunities(session.token, kbId).then((r) => setCommunities(r.communities)).catch(() => setCommunities([]))
  }, [session, kbId])

  // Categorical ramps — mirror GraphCanvas so the Themes list swatches match the map.
  const RAMP_LIGHT = ['#2E4A7D', '#C25F12', '#0F7B3E', '#8A7028', '#1E8A8A', '#9B4B6E', '#5E7A32', '#A8562C']
  const RAMP_DARK = ['#7FA3DE', '#FFA940', '#4FBF7C', '#E8D094', '#4FC6C6', '#D890AE', '#A8CB6E', '#E79A72']
  const isDark = typeof document !== 'undefined' && document.documentElement.getAttribute('data-theme') === 'dark'
  const ramp = isDark ? RAMP_DARK : RAMP_LIGHT
  const themeColor = (orderIdx: number) => ramp[orderIdx % ramp.length]

  async function buildCommunities() {
    if (!session) return
    setBuildingComs(true)
    try {
      const r = await api.buildCommunities(session.token, kbId)
      const coms = await api.getCommunities(session.token, kbId)
      setCommunities(coms.communities)
      // Also refresh the subgraph/stats so the map's node→community colouring updates live
      // (previously only the themes list refreshed, so a page reload was needed).
      api.getGraph(session.token, kbId).then(setGraph).catch(() => {})
      api.getGraphSubgraph(session.token, kbId, 60).then(setSubgraph).catch(() => {})
      toast.push('success', `Found ${r.communities} theme${r.communities === 1 ? '' : 's'} across ${r.entities.toLocaleString()} entities.`)
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not build themes.')
    } finally {
      setBuildingComs(false)
    }
  }

  async function saveWidget(patch: Partial<import('../lib/api').WidgetConfig>) {
    if (!session || !widgetCfg) return
    setSavingWidget(true)
    try {
      const next = await api.setWidgetConfig(session.token, kbId, { ...widgetCfg, ...patch })
      setWidgetCfg(next)
      toast.push('success', next.enabled ? 'Widget saved — embed it on your site.' : 'Widget disabled.')
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not save widget.')
    } finally {
      setSavingWidget(false)
    }
  }

  async function saveTaxonomy() {
    if (!session) return
    setSavingTax(true)
    try {
      const tax: Record<string, string[]> = {}
      for (const line of taxonomyText.split('\n')) {
        const [label, rest] = line.split(':')
        if (label?.trim() && rest) tax[label.trim()] = rest.split(',').map((s) => s.trim()).filter(Boolean)
      }
      await api.setTaxonomy(session.token, kbId, tax)
      toast.push('success', 'Taxonomy saved — applied to new/re-indexed documents.')
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not save taxonomy.')
    } finally {
      setSavingTax(false)
    }
  }

  async function saveBot(patch: Partial<{ enabled: boolean; trigger: string }>) {
    if (!session || !botCfg) return
    try {
      const next = await api.setBotConfig(session.token, kbId, {
        enabled: patch.enabled ?? botCfg.enabled, platform: botCfg.platform, trigger: patch.trigger ?? botCfg.trigger,
      })
      setBotCfg(next)
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not save bot.')
    }
  }

  async function generateBotToken() {
    if (!session) return
    try {
      const r = await api.rotateBotToken(session.token, kbId)
      setBotToken(r.token)
      setBotCfg(await api.getBotConfig(session.token, kbId))
      toast.push('success', 'Token generated — copy it now; it is shown only once.')
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not generate token.')
    }
  }

  async function savePersona() {
    if (!session) return
    setSavingPersona(true)
    try {
      const p = await api.setPersona(session.token, kbId, persona.trim())
      setPersona(p.persona)
      toast.push('success', 'Answer style saved — applied to new answers.')
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Save failed')
    } finally {
      setSavingPersona(false)
    }
  }

  async function saveThresholds() {
    if (!session) return
    setSavingThr(true)
    try {
      const body: Record<string, number> = {}
      if (thr.abstain) body.abstain_threshold = Number(thr.abstain)
      if (thr.faithful) body.faithful_threshold = Number(thr.faithful)
      if (thr.flag) body.flag_threshold = Number(thr.flag)
      const info = await api.setKbSettings(session.token, kbId, body)
      setSettings(info)
      toast.push('success', 'Thresholds updated — effective immediately.')
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Save failed')
    } finally {
      setSavingThr(false)
    }
  }

  async function createKb() {
    if (!name.trim()) return
    setBusy(true); setError(null)
    try {
      const kb = await create(name.trim())
      setName('')
      toast.push('success', `Created knowledge base "${kb.name}"`)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Failed to create')
    } finally {
      setBusy(false)
    }
  }

  async function removeKb(kb: { kb_id: string; name: string }) {
    if (!session) return
    if (!window.confirm(`Move "${kb.name}" to Trash? Its data is kept and can be restored — it is not deleted yet.`)) return
    setDeletingKb(kb.kb_id)
    try {
      await api.deleteKb(session.token, kb.kb_id)
      await refresh()
      await loadTrash()
      toast.push('info', `Moved "${kb.name}" to Trash — restore it any time.`)
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Delete failed')
    } finally {
      setDeletingKb(null)
    }
  }

  async function restoreKb(kb: { kb_id: string; name: string }) {
    if (!session) return
    try {
      await api.restoreKb(session.token, kb.kb_id)
      await refresh(); await loadTrash()
      toast.push('success', `Restored "${kb.name}".`)
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Restore failed')
    }
  }

  async function purgeKb(kb: { kb_id: string; name: string }) {
    if (!session) return
    const typed = window.prompt(`Permanently delete "${kb.name}" and ALL its data? This CANNOT be undone.\n\nType the knowledge base name to confirm:`)
    if (typed === null) return
    if (typed.trim() !== kb.name) { toast.push('error', 'Name did not match — nothing was deleted.'); return }
    try {
      const r = await api.purgeKb(session.token, kb.kb_id, typed.trim())
      await loadTrash()
      toast.push('info', `Permanently deleted "${kb.name}" — ${r.documents} document(s) purged.`)
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Purge failed')
    }
  }

  async function removeDoc(docId: string) {
    if (!session || !window.confirm(`Delete "${docId}"? This removes it from this knowledge base and can't be undone.`)) return
    setActing(docId)
    try {
      await api.deleteDocument(session.token, kbId, docId)
      setDocs((d) => (d ?? []).filter((x) => x.doc_id !== docId))
      // the delete cascades into the knowledge graph server-side (DPDP erasure); refetch so
      // the graph card/canvas doesn't keep showing entities/edges from the deleted document.
      api.getGraph(session.token, kbId).then(setGraph).catch(() => setGraph(null))
      api.getGraphSubgraph(session.token, kbId, 60).then(setSubgraph).catch(() => setSubgraph(null))
      api.getCommunities(session.token, kbId).then((res) => setCommunities(res.communities)).catch(() => setCommunities([]))
      toast.push('info', `Deleted ${docId}`)
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Delete failed')
    } finally {
      setActing(null)
    }
  }

  async function reindexDoc(docId: string) {
    if (!session) return
    setActing(docId)
    try {
      const r = await api.reindexDocument(session.token, kbId, docId)
      toast.push('success', `Re-indexed ${docId} — ${r.chunk_count} chunk(s)`)
      void loadDocs()
      void loadIndexStatus()
      // Re-indexing re-extracts the doc's entities/edges server-side, so refresh the graph card/
      // canvas/themes — same as removeDoc; otherwise they keep showing pre-reindex entities.
      api.getGraph(session.token, kbId).then(setGraph).catch(() => setGraph(null))
      api.getGraphSubgraph(session.token, kbId, 60).then(setSubgraph).catch(() => setSubgraph(null))
      api.getCommunities(session.token, kbId).then((res) => setCommunities(res.communities)).catch(() => setCommunities([]))
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Re-index failed')
    } finally {
      setActing(null)
    }
  }

  async function reindexStale() {
    if (!session || reindexing) return
    setReindexing(true)
    try {
      const r = await api.reindexKb(session.token, kbId, true)
      const failed = r.failed.length
      toast.push(
        failed ? 'error' : 'success',
        failed
          ? `Rebuilt ${r.reindexed} of ${r.requested}; ${failed} failed — ${r.failed[0].error}`
          : `Rebuilt ${r.reindexed} document(s) with the current pipeline`,
      )
      void loadDocs(true)
      void loadIndexStatus()
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Re-index failed')
    } finally {
      setReindexing(false)
    }
  }

  const current = kbs.find((k) => k.kb_id === kbId)
  const graphOn = !!graph?.enabled
  const integrationsOn = !!sys?.features?.integrations
  const tabs: { value: Tab; label: string }[] = [
    { value: 'documents', label: t('kb.tab.documents') },
    { value: 'settings', label: t('kb.tab.settings') },
    ...(integrationsOn ? [{ value: 'integrations' as Tab, label: t('kb.tab.integrations') }] : []),
    ...(graphOn ? [{ value: 'knowledge' as Tab, label: t('kb.tab.knowledge') }] : []),
  ]
  const activeTab: Tab =
    (tab === 'knowledge' && !graphOn) || (tab === 'integrations' && !integrationsOn) ? 'documents' : tab

  // The graph is built in the backend during ingestion (and by a structured-source graph-sync);
  // refresh the view whenever the graph tab opens and shortly after an upload, so newly-ingested
  // entities appear without a manual page refresh.
  const loadGraph = useCallback(() => {
    if (!session || !kbId) return
    api.getGraph(session.token, kbId).then(setGraph).catch(() => setGraph(null))
    api.getGraphSubgraph(session.token, kbId, 60).then(setSubgraph).catch(() => setSubgraph(null))
  }, [session, kbId])
  useEffect(() => { if (activeTab === 'knowledge') loadGraph() }, [activeTab, loadGraph])

  return (
    <div>
      <PageHeader eyebrow={t('eyebrow.knowledge')} title={t('page.kb.title')} lead={t('page.kb.lead')} />

      <div className="grid lg:grid-cols-3 gap-5 items-start">
        {/* Left rail: the KB selector (context) — only this lives here now */}
        <div className="lg:col-span-1 lg:sticky self-start" style={{ top: 8 }}>
          <Card>
            <h2 className="font-display font-semibold mb-3" style={{ fontSize: 16 }}>{t('kb.card.your_kbs')}</h2>
            <ul className="space-y-1.5 overflow-y-auto" style={{ maxHeight: '60vh', paddingRight: 2 }}>
              {kbs.map((k) => (
                <li key={k.kb_id} className="flex items-center gap-1.5 rounded-md"
                  style={{ background: k.kb_id === kbId ? 'color-mix(in srgb, var(--action) 10%, transparent)' : 'transparent', border: `1px solid ${k.kb_id === kbId ? 'color-mix(in srgb, var(--action) 35%, transparent)' : 'var(--border)'}` }}>
                  <button type="button" onClick={() => setKbId(k.kb_id)}
                    className="flex-grow flex items-center gap-2.5 px-3 py-2.5 rounded-md text-left transition-colors min-w-0"
                    style={{ cursor: 'pointer', background: 'transparent', border: 'none' }}>
                    <Icon.database size={16} style={{ color: k.kb_id === kbId ? 'var(--action)' : 'var(--text-muted)' }} />
                    <span className="flex-grow min-w-0">
                      <span className="block font-semibold truncate" style={{ color: 'var(--text-heading)' }}>{k.name}</span>
                      <span className="block mono truncate" style={{ fontSize: 11, color: 'var(--text-muted)' }}>{k.kb_id}</span>
                    </span>
                    {k.is_default && <span className="pill" style={{ fontSize: 10, color: 'var(--text-muted)', borderColor: 'var(--border)', background: 'transparent' }}>default</span>}
                    {k.kb_id === SHARED_KB_ID && <span className="pill" title="Company-wide shared KB — every team can read it and add Public content; it can't be deleted." style={{ fontSize: 10, color: 'var(--action)', borderColor: 'color-mix(in srgb, var(--action) 35%, transparent)', background: 'color-mix(in srgb, var(--action) 8%, transparent)' }}>Shared · Public</span>}
                    {k.kb_id === kbId && <Icon.check size={15} style={{ color: 'var(--action)' }} />}
                  </button>
                  {!k.is_default && k.kb_id !== SHARED_KB_ID && (
                    <button type="button" onClick={() => removeKb(k)} disabled={deletingKb !== null}
                      aria-label={`Delete ${k.name}`} title="Delete this knowledge base"
                      className="btn btn-subtle btn-icon mr-1" style={{ color: 'var(--danger)' }}>
                      {deletingKb === k.kb_id
                        ? <span className="inline-block rounded-full animate-spin" style={{ width: 13, height: 13, border: '2px solid color-mix(in srgb, var(--danger) 30%, transparent)', borderTopColor: 'var(--danger)' }} />
                        : <Icon.trash size={14} />}
                    </button>
                  )}
                </li>
              ))}
              {kbs.length === 0 && <Skeleton style={{ height: 60 }} />}
            </ul>
            <div className="mt-4 pt-4" style={{ borderTop: '1px solid var(--border)' }}>
              <Field label="New knowledge base" help='Its own isolated stores. e.g. "HR Policies".'>
                <input className="field-input" value={name} onChange={(e) => setName(e.target.value)} placeholder="HR Policies" onKeyDown={(e) => e.key === 'Enter' && createKb()} />
              </Field>
              <Button className="mt-3" size="sm" loading={busy} disabled={!name.trim()} onClick={createKb} icon={!busy && <Icon.plus size={15} />}>Create</Button>
              {error && <div className="mt-3"><Alert kind="error">{error}</Alert></div>}
            </div>
          </Card>

          {trash.length > 0 && (
            <Card className="mt-5">
              <div className="flex items-center gap-2 mb-1"><Icon.trash size={15} style={{ color: 'var(--text-muted)' }} /><h2 className="font-display font-semibold" style={{ fontSize: 15 }}>Trash</h2></div>
              <p className="field-help mb-3">Deleted knowledge bases are kept here — restore them, or permanently delete (with confirmation).</p>
              <ul className="space-y-1.5">
                {trash.map((k) => (
                  <li key={k.kb_id} className="rounded-md px-2.5 py-2" style={{ border: '1px solid var(--border)' }}>
                    <div className="font-semibold truncate" style={{ color: 'var(--text-heading)', fontSize: 13 }}>{k.name}</div>
                    <div className="flex items-center gap-2 mt-1.5">
                      <Button size="sm" variant="secondary" onClick={() => restoreKb(k)} icon={<Icon.check size={13} />}>Restore</Button>
                      <Button size="sm" variant="subtle" onClick={() => purgeKb(k)} icon={<Icon.trash size={13} />} style={{ color: 'var(--danger)' }}>Delete permanently</Button>
                    </div>
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </div>

        {/* Right column: tabbed panels for the selected KB */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <div className="min-w-0">
              <h2 className="font-display font-semibold truncate" style={{ fontSize: 17, color: 'var(--text-heading)' }}>{current?.name ?? kbId}</h2>
              <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>{kbId}</span>
            </div>
            <SegmentedControl options={tabs} value={activeTab} onChange={setTab} />
          </div>

          {/* Live stat strip — one aligned card. Hidden on the graph tab so the graph gets the full height. */}
          {activeTab !== 'knowledge' && (
          <div className="surface rounded-xl grid grid-cols-2 sm:grid-cols-4" style={{ overflow: 'hidden' }}>
            {[
              { label: 'Documents', value: docs === null ? '…' : String(docs.length), icon: Icon.doc },
              { label: 'Queries · 30d', value: usage30 === null ? '…' : usage30.queries.toLocaleString(), icon: Icon.ask },
              { label: 'Tokens · 30d', value: usage30 === null ? '…' : usage30.total_tokens.toLocaleString(), icon: Icon.activity },
              { label: 'Last ingest', value: docs === null || docs.length === 0 ? '—' : new Date(Math.max(...docs.map((d) => new Date(d.ingested_at).getTime()))).toLocaleDateString(), icon: Icon.upload },
            ].map((s, i) => (
              <div key={s.label} className="flex items-center gap-3 px-4 py-3" style={{ borderLeft: i ? '1px solid var(--border)' : 'none' }}>
                <s.icon size={15} style={{ color: 'var(--action)', flexShrink: 0 }} />
                <div className="min-w-0">
                  <div className="font-display font-semibold num truncate" style={{ fontSize: 16, color: 'var(--text-heading)' }}>{s.value}</div>
                  <div className="truncate" style={{ fontSize: 11, color: 'var(--text-muted)' }}>{s.label}</div>
                </div>
              </div>
            ))}
          </div>
          )}

          {/* --- Documents --- */}
          {activeTab === 'documents' && (
            <>
              {/* Stale index (FR-C8). Shown only when it is actionable: a stale document answers
                  normally, so without this the only symptom is a subtly worse answer. */}
              {indexStatus && indexStatus.stale > 0 && (
                <Alert kind="warning">
                  <div className="flex items-start justify-between gap-3 flex-wrap">
                    <div>
                      <div className="font-semibold">
                        {indexStatus.stale} of {indexStatus.total} document{indexStatus.total === 1 ? '' : 's'} was indexed by an older pipeline
                      </div>
                      <div style={{ fontSize: 12 }}>
                        They still answer, but from chunks and vectors this deployment no longer produces — the
                        chunker, embedder or document readers changed since they were ingested. Rebuilding re-runs
                        extraction and embedding for each one; the current version keeps serving until its
                        replacement is indexed.
                        {indexStatus.auto_reindex && ' An operator has this knowledge base on automatic re-indexing.'}
                      </div>
                    </div>
                    {canIngest && (
                      <Button variant="secondary" size="sm" className="whitespace-nowrap" onClick={() => void reindexStale()} loading={reindexing}>
                        {reindexing ? 'Re-indexing…' : 'Re-index stale'}
                      </Button>
                    )}
                  </div>
                </Alert>
              )}
              <Card>
                <h2 className="font-display font-semibold mb-1" style={{ fontSize: 16 }}>{t('kb.card.add_documents', { name: current?.name ?? kbId })}</h2>
                <p className="field-help mb-3">Drag in one file or thousands — ingestion runs in the background (resumes if the server restarts) and you can watch progress below.</p>
                {canIngest ? (
                  <Uploader onIngested={(id, done) => {
                    void loadDocs(true)
                    if (done) { window.setTimeout(loadGraph, 1500); window.setTimeout(() => setLastUploadedDocId(''), 3000) } // completed: show green-complete briefly, then auto-drop the live pipeline
                    else if (id) setLastUploadedDocId(id) // ingesting: show the live pipeline
                  }} />
                ) : (
                  <Alert kind="info">Requires {roleLabels.ingestor} access to upload documents.</Alert>
                )}
              </Card>
              {/* Ingestion pipeline — right after the uploader so the process is visible the moment you
                  ingest; the resulting documents list follows below. Auto-traces the newest upload. */}
              <div>
                <PipelineDag docId={lastUploadedDocId} active={ingest.jobForKb(kbId)?.phase === 'processing'} />
              </div>
              <Card>
                <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
                  <div className="flex items-center gap-2">
                    <h2 className="font-display font-semibold" style={{ fontSize: 16 }}>{t('kb.card.documents')}</h2>
                    {docs !== null && docs.length > 0 && <span className="chip">{docs.length}</span>}
                  </div>
                  {docs !== null && docs.length > 3 && (
                    <input value={docSearch} onChange={(e) => setDocSearch(e.target.value)} placeholder="Search documents…"
                      aria-label="Search documents" className="field-input" style={{ minHeight: 34, maxWidth: 230, fontSize: 13 }} />
                  )}
                </div>
                {docs === null && <Skeleton style={{ height: 140 }} />}
                {docs !== null && docs.length === 0 && (
                  <EmptyState icon={<Icon.doc size={28} />} title="No documents in this knowledge base"
                    hint="Drop files above, or pull from the web / object storage via Connectors." />
                )}
                {docs !== null && docs.length > 0 && (
                  <div className="overflow-y-auto -mx-2 px-2" style={{ maxHeight: '52vh' }}>
                    <Table>
                      <thead><tr><th>Document</th><th>Type</th><th>Ingested</th><th></th></tr></thead>
                      <tbody>
                        {docs.filter((d) => {
                          const s = docSearch.trim().toLowerCase()
                          if (!s) return true
                          return d.doc_id.toLowerCase().includes(s)
                            || Object.values(d.metadata ?? {}).some((v) => String(v).toLowerCase().includes(s))
                        }).map((d) => (
                          <tr key={d.doc_id}>
                            <td className="font-medium" style={{ color: 'var(--text-heading)', maxWidth: 320 }} title={d.doc_id}>
                              <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{d.doc_id}</div>
                              {d.metadata && (d.metadata.doc_type || d.metadata.category || d.metadata.year) && (
                                <div className="flex flex-wrap gap-1 mt-1">
                                  {d.metadata.doc_type && <span className="chip" style={{ fontSize: 10 }}>{d.metadata.doc_type}</span>}
                                  {d.metadata.category && <span className="chip" style={{ fontSize: 10, color: 'var(--accent)' }}>{d.metadata.category}</span>}
                                  {d.metadata.year && <span className="chip mono" style={{ fontSize: 10 }}>{d.metadata.year}</span>}
                                </div>
                              )}
                            </td>
                            <td className="mono text-xs" style={{ color: 'var(--text-muted)' }}>{d.mime.split('/').pop()}</td>
                            <td className="mono num text-xs" style={{ color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>{new Date(d.ingested_at).toLocaleDateString()}</td>
                            <td style={{ width: 108, textAlign: 'right', whiteSpace: 'nowrap' }}>
                              <button type="button" onClick={() => setViewDoc(d.doc_id)}
                                aria-label={`View ${d.doc_id}`} title="View the text this document contributed to the index"
                                className="btn btn-subtle btn-icon"><Icon.eye size={15} /></button>
                              <button type="button" onClick={() => reindexDoc(d.doc_id)} disabled={acting !== null}
                                aria-label={`Re-index ${d.doc_id}`} title="Re-index from stored bytes"
                                className="btn btn-subtle btn-icon"><Icon.refresh size={15} /></button>
                              <button type="button" onClick={() => removeDoc(d.doc_id)} disabled={acting !== null}
                                aria-label={`Delete ${d.doc_id}`} title="Delete"
                                className="btn btn-subtle btn-icon" style={{ color: 'var(--danger)' }}><Icon.x size={15} /></button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  </div>
                )}
              </Card>
            </>
          )}

          {/* --- Settings --- */}
          {activeTab === 'settings' && (
            <div className="space-y-4">
            <div className="grid md:grid-cols-2 gap-4">
              <Card>
                <h2 className="font-display font-semibold mb-1" style={{ fontSize: 16 }}>{t('kb.card.thresholds')}</h2>
                <p className="field-help mb-3">Per-KB overrides for {current?.name ?? 'this KB'}; blank = platform default (shown in the hint).</p>
                <div className="space-y-3">
                  <Field label="Abstain below" help={`Default ${settings?.defaults.abstain_threshold ?? '…'} — higher = stricter, more honest abstains.`}>
                    <input className="field-input" type="number" min="0" max="1" step="0.05" value={thr.abstain}
                      onChange={(e) => setThr((s) => ({ ...s, abstain: e.target.value }))} placeholder={String(settings?.defaults.abstain_threshold ?? '')} />
                  </Field>
                  <Field label="Faithful at or above" help={`Default ${settings?.defaults.faithful_threshold ?? '…'} — answers below go to review or abstain.`}>
                    <input className="field-input" type="number" min="0" max="1" step="0.05" value={thr.faithful}
                      onChange={(e) => setThr((s) => ({ ...s, faithful: e.target.value }))} placeholder={String(settings?.defaults.faithful_threshold ?? '')} />
                  </Field>
                  <Field label="Flag for review at or above" help={`Default ${settings?.defaults.flag_threshold ?? '…'} — below this the platform abstains.`}>
                    <input className="field-input" type="number" min="0" max="1" step="0.05" value={thr.flag}
                      onChange={(e) => setThr((s) => ({ ...s, flag: e.target.value }))} placeholder={String(settings?.defaults.flag_threshold ?? '')} />
                  </Field>
                </div>
                <Button className="mt-4" size="sm" loading={savingThr} onClick={saveThresholds} icon={!savingThr && <Icon.check size={15} />}>Save thresholds</Button>
              </Card>

              <div className="space-y-4">
                <Card>
                  <h2 className="font-display font-semibold mb-1" style={{ fontSize: 16 }}>{t('kb.card.answer_style')}</h2>
                  <p className="field-help mb-3">How answers should read for {current?.name ?? 'this KB'} — tone, format, or output language. It shapes wording only; grounding &amp; citations always win. e.g. <span className="mono">"Answer in formal Hindi, in short bullet points."</span></p>
                  <textarea className="field-input" rows={3} value={persona} onChange={(e) => setPersona(e.target.value)}
                    placeholder="Default voice — concise, cited, neutral." style={{ fontSize: 14 }} />
                  <Button className="mt-3" size="sm" loading={savingPersona} onClick={savePersona} icon={!savingPersona && <Icon.check size={15} />}>Save style</Button>
                </Card>
                <Card>
                  <h2 className="font-display font-semibold mb-1" style={{ fontSize: 16 }}>{t('kb.card.taxonomy')}</h2>
                  <p className="field-help mb-3">Tag documents automatically at ingest. One label per line as <span className="mono">Label: keyword1, keyword2</span> — a document gets the label if any keyword appears. Tags become filters on retrieval. Entities, year &amp; doc-type are detected automatically.</p>
                  <textarea className="field-input" rows={3} value={taxonomyText} onChange={(e) => setTaxonomyText(e.target.value)}
                    placeholder={'HR: leave, salary, allowance\nLegal: agreement, clause, liability'} style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }} />
                  <Button className="mt-3" size="sm" loading={savingTax} onClick={saveTaxonomy} icon={!savingTax && <Icon.check size={15} />}>Save taxonomy</Button>
                </Card>
              </div>
            </div>
            {/* KB-admin/owner only — the card self-hides for everyone else. */}
            <RolesCard kbId={kbId} />
            {/* KB members & managers moved to the Team nav (org admin + KB managers manage them there). */}
            </div>
          )}

          {/* --- Integrations --- */}
          {activeTab === 'integrations' && integrationsOn && !canAdmin && (
            <Card><Alert kind="info">Requires admin role to view or change widget/bot integration settings.</Alert></Card>
          )}
          {activeTab === 'integrations' && integrationsOn && canAdmin && (
            <div className="grid md:grid-cols-2 gap-4">
              {widgetCfg ? (
                <Card>
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2"><Icon.cloud size={16} style={{ color: 'var(--text-muted)' }} /><h2 className="font-display font-semibold" style={{ fontSize: 16 }}>{t('kb.card.widget')}</h2></div>
                    <Button size="sm" variant={widgetCfg.enabled ? 'secondary' : 'primary'} loading={savingWidget} onClick={() => saveWidget({ enabled: !widgetCfg.enabled })}>
                      {widgetCfg.enabled ? 'Disable' : 'Enable'}
                    </Button>
                  </div>
                  <p className="field-help mb-3">A white-label chat bubble for your own site — answers only from this KB, with citations. The API key stays on your server; the browser gets a short-lived, read-only, single-KB token.</p>
                  {widgetCfg.enabled && (
                    <div className="space-y-2.5">
                      <div className="grid grid-cols-2 gap-2.5">
                        <label className="text-xs" style={{ color: 'var(--text-muted)' }}>Title
                          <input className="field-input mt-1" defaultValue={widgetCfg.title} onBlur={(e) => e.target.value !== widgetCfg.title && saveWidget({ title: e.target.value.slice(0, 80) })} />
                        </label>
                        <label className="text-xs" style={{ color: 'var(--text-muted)' }}>Accent colour
                          <input className="field-input mt-1" type="color" defaultValue={widgetCfg.accent} onBlur={(e) => e.target.value !== widgetCfg.accent && saveWidget({ accent: e.target.value })} style={{ height: 38, padding: 3 }} />
                        </label>
                      </div>
                      <label className="text-xs block" style={{ color: 'var(--text-muted)' }}>Welcome message
                        <input className="field-input mt-1" defaultValue={widgetCfg.welcome} onBlur={(e) => e.target.value !== widgetCfg.welcome && saveWidget({ welcome: e.target.value.slice(0, 280) })} />
                      </label>
                      <label className="text-xs block" style={{ color: 'var(--text-muted)' }}>Starter questions (one per line, up to 4)
                        <textarea className="field-input mt-1" rows={2} defaultValue={widgetCfg.suggestions.join('\n')} onBlur={(e) => saveWidget({ suggestions: e.target.value.split('\n').map((s) => s.trim()).filter(Boolean).slice(0, 4) })} style={{ fontSize: 13 }} />
                      </label>
                      <div>
                        <div className="eyebrow mb-1">Embed snippet</div>
                        <pre className="mono" style={{ fontSize: 11, background: 'var(--surface-2)', padding: 10, borderRadius: 8, overflowX: 'auto', color: 'var(--text-muted)' }}>{`<script src="https://cdn.you/ragaas-widget.js"\n  data-base="${window.location.origin}"\n  data-session-url="/api/ragaas-session"></script>`}</pre>
                        <p className="field-help mt-1">Add a server route that proxies <span className="mono">POST /v1/widget/session</span> with a read API key — see the widget README.</p>
                      </div>
                    </div>
                  )}
                </Card>
              ) : <Skeleton style={{ height: 120 }} />}

              {botCfg ? (
                <Card>
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2"><Icon.book size={16} style={{ color: 'var(--text-muted)' }} /><h2 className="font-display font-semibold" style={{ fontSize: 16 }}>{t('kb.card.bot')}</h2></div>
                    <Button size="sm" variant={botCfg.enabled ? 'secondary' : 'primary'} onClick={() => saveBot({ enabled: !botCfg.enabled })}>
                      {botCfg.enabled ? 'Disable' : 'Enable'}
                    </Button>
                  </div>
                  <p className="field-help mb-3">Answer questions from a Mattermost channel. The bot replies with grounded answers + citations, on-prem — no data leaves your network.</p>
                  {botCfg.enabled && (
                    <div className="space-y-2.5">
                      <div>
                        <div className="eyebrow mb-1">Outgoing-webhook URL (paste into Mattermost)</div>
                        <pre className="mono" style={{ fontSize: 11, background: 'var(--surface-2)', padding: 10, borderRadius: 8, overflowX: 'auto', color: 'var(--text-muted)' }}>{`${window.location.origin}${botCfg.webhook_path}`}</pre>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button size="sm" variant="ghost" onClick={generateBotToken} icon={<Icon.key size={14} />}>{botCfg.configured ? 'Regenerate token' : 'Generate token'}</Button>
                        <span className="text-xs" style={{ color: botCfg.configured ? 'var(--success)' : 'var(--warning)' }}>{botCfg.configured ? 'token set' : 'no token yet'}</span>
                      </div>
                      {botToken && (
                        <div className="rounded-lg p-2.5" style={{ background: 'color-mix(in srgb, var(--warning) 10%, transparent)', border: '1px solid color-mix(in srgb, var(--warning) 30%, transparent)' }}>
                          <div className="eyebrow mb-1">Token — copy now, shown once</div>
                          <pre className="mono" style={{ fontSize: 12, overflowX: 'auto' }}>{botToken}</pre>
                          <p className="field-help mt-1">In Mattermost: Integrations → Outgoing Webhooks → set this URL + token, and a trigger word (e.g. <span className="mono">@rag</span>).</p>
                        </div>
                      )}
                      <label className="text-xs block" style={{ color: 'var(--text-muted)' }}>Trigger word to strip (optional)
                        <input className="field-input mt-1" defaultValue={botCfg.trigger} placeholder="@rag" onBlur={(e) => e.target.value !== botCfg.trigger && saveBot({ trigger: e.target.value.slice(0, 40) })} />
                      </label>
                    </div>
                  )}
                </Card>
              ) : <Skeleton style={{ height: 120 }} />}
            </div>
          )}

          {/* --- Knowledge graph --- */}
          {activeTab === 'knowledge' && graphOn && (() => {
            const hasGraph = !!subgraph && subgraph.nodes.length > 0
            const maxSize = Math.max(1, ...(communities ?? []).map((c) => c.size))

            if (!hasGraph) {
              // Empty state — nothing extracted from this KB yet.
              return (
                <Card>
                  <div className="flex flex-col items-start gap-2" style={{ padding: '32px 8px' }}>
                    <span className="eyebrow">No graph yet</span>
                    <span className="font-display font-semibold" style={{ fontSize: 16, color: 'var(--text-heading)' }}>Nothing has been extracted from this knowledge base</span>
                    <p className="field-help" style={{ maxWidth: '58ch' }}>Entities and relationships are mined during indexing. Ingest documents first, then build themes — the graph clusters what it finds so broad questions can be answered across topics instead of a single passage.</p>
                    <div className="flex gap-2 mt-2">
                      <Button size="sm" onClick={() => setTab('documents')}>Add documents</Button>
                      {graph.nodes > 0 && (
                        <Button size="sm" variant="secondary" loading={buildingComs} onClick={buildCommunities}>Build themes</Button>
                      )}
                    </div>
                  </div>
                </Card>
              )
            }

            return (
              <div className="space-y-4">
                {/* header stats */}
                <div className="flex items-end justify-between flex-wrap gap-3">
                  <div className="flex items-center gap-2">
                    <Icon.database size={16} style={{ color: 'var(--text-muted)' }} />
                    <h2 className="font-display font-semibold" style={{ fontSize: 16 }}>{t('kb.card.graph')}</h2>
                  </div>
                  <div className="flex items-center gap-5" style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>
                    <span><span className="mono" style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-heading)' }}>{graph.nodes.toLocaleString()}</span> entities</span>
                    <span><span className="mono" style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-heading)' }}>{graph.edges.toLocaleString()}</span> relationships</span>
                    <span><span className="mono" style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-heading)' }}>{(communities ?? []).length}</span> themes</span>
                    <span className="mono" style={{ fontSize: 11 }}>showing top {subgraph!.nodes.length}</span>
                  </div>
                </div>

                {/* two-pane panel: Graph map + Themes */}
                <div style={{ border: '1px solid var(--border)', background: 'var(--surface)', borderRadius: 'var(--radius-lg)', display: 'flex', height: 'clamp(340px, calc(100vh - 330px), 680px)', overflow: 'hidden' }}>
                  {/* graph column — 2D canvas */}
                  <GraphCanvas
                    data={subgraph!}
                    communities={communities ?? []}
                    activeCommunity={activeCommunity}
                    hoverCommunity={hoverCommunity}
                    onActiveCommunity={setActiveCommunity}
                    height={560}
                  />

                  {/* themes column */}
                  <aside style={{ width: 296, flexShrink: 0, borderLeft: '1px solid var(--border)', display: 'flex', flexDirection: 'column' }}>
                    <div style={{ height: 44, flexShrink: 0, borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 16px' }}>
                      <span style={{ fontFamily: 'var(--font-heading)', fontSize: 13, fontWeight: 600, color: 'var(--text-heading)' }}>{t('kb.card.themes')}</span>
                      <div className="flex items-center gap-1.5">
                        <button type="button" onClick={() => setActiveCommunity(null)}
                          style={{ height: 24, padding: '0 9px', border: '1px solid var(--border)', borderRadius: 3, background: 'var(--surface)', color: 'var(--text-muted)', fontFamily: 'var(--font-body)', fontSize: 11, fontWeight: 600, cursor: 'pointer' }}>
                          {activeCommunity != null ? 'Clear filter' : 'All themes'}
                        </button>
                        <Button size="sm" variant="secondary" loading={buildingComs} onClick={buildCommunities} style={{ fontSize: 11, padding: '3px 9px', minHeight: 24 }}>
                          {communities && communities.length > 0 ? 'Rebuild' : 'Build'}
                        </Button>
                      </div>
                    </div>
                    <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
                      {communities === null && <div style={{ padding: 16 }}><Skeleton style={{ height: 200 }} /></div>}
                      {communities !== null && communities.length === 0 && (
                        <div style={{ padding: 16 }}><p className="text-xs" style={{ color: 'var(--text-muted)' }}>No themes yet — build them from the current graph.</p></div>
                      )}
                      {(communities ?? []).map((c, ci) => {
                        const col = themeColor(ci)
                        const on = activeCommunity === c.community_id
                        const pct = Math.round((c.size / maxSize) * 100)
                        return (
                          <button key={c.community_id} type="button"
                            onClick={() => setActiveCommunity(on ? null : c.community_id)}
                            onMouseEnter={() => setHoverCommunity(c.community_id)}
                            onMouseLeave={() => setHoverCommunity(null)}
                            style={{ display: 'flex', flexDirection: 'column', alignItems: 'stretch', gap: 5, width: '100%', textAlign: 'left', padding: '11px 16px 12px', background: on ? 'var(--sunken)' : 'transparent', border: 'none', borderBottom: '1px solid var(--hair)', borderLeft: `2px solid ${on ? col : 'transparent'}`, cursor: 'pointer', fontFamily: 'var(--font-body)', transition: 'background .12s' }}>
                            <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <span style={{ width: 8, height: 8, borderRadius: 2, background: col, flexShrink: 0 }} />
                              <span style={{ fontFamily: 'var(--font-heading)', fontSize: 12.5, fontWeight: 600, color: 'var(--text-heading)' }}>{c.label}</span>
                              <span className="mono" style={{ fontSize: 11.5, color: 'var(--text-muted)', marginLeft: 'auto' }}>{c.size}</span>
                            </span>
                            {c.summary && <span style={{ fontSize: 11.5, lineHeight: 1.5, color: 'var(--text-muted)' }}>{c.summary}</span>}
                            <span style={{ display: 'block', height: 2, background: 'var(--hair)', marginTop: 2 }}>
                              <span style={{ display: 'block', height: 2, width: `${pct}%`, background: col }} />
                            </span>
                          </button>
                        )
                      })}
                    </div>
                    <div style={{ flexShrink: 0, borderTop: '1px solid var(--border)', padding: '11px 16px', display: 'flex', flexDirection: 'column', gap: 4 }}>
                      <span style={{ fontSize: 11.5, lineHeight: 1.5, color: 'var(--text-muted)' }}>Broad questions are answered across themes rather than a single passage.</span>
                      <span className="mono" style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>leiden · res 1.0</span>
                    </div>
                  </aside>
                </div>
              </div>
            )
          })()}
        </div>
      </div>

      {/* Reuses the citation viewer: the same slide-over that proves an answer against its source
          reads a document on its own, with no chunk singled out. */}
      {viewDoc && session && kbId && (
        <CitationViewer
          token={session.token}
          kbId={kbId}
          target={{ docId: viewDoc, chunkId: '', quote: '', page: null }}
          onClose={() => setViewDoc('')}
        />
      )}
    </div>
  )
}
