// Dedicated full-page knowledge-graph explorer for the selected KB — a roomier home for the
// graph than the KB-page tab. Renders the 2D GraphCanvas with the themes sidebar.
import { useCallback, useEffect, useState } from 'react'
import { useToast } from '../components/overlay'
import { Button, Card, PageHeader, Skeleton } from '../components/ui'
import { GraphCanvas } from '../components/GraphCanvas'
import { api, type GraphCommunityInfo, type GraphStatsInfo, type GraphSubgraph } from '../lib/api'
import { ApiError } from '../lib/api'
import { useAuth } from '../lib/auth'
import { useKb } from '../lib/kb'

const RAMP_LIGHT = ['#2E4A7D', '#C25F12', '#0F7B3E', '#8A7028', '#1E8A8A', '#9B4B6E', '#5E7A32', '#A8562C']
const RAMP_DARK = ['#7FA3DE', '#FFA940', '#4FBF7C', '#E8D094', '#4FC6C6', '#D890AE', '#A8CB6E', '#E79A72']

export function Graph() {
  const { session } = useAuth()
  const { kbId, kbs } = useKb()
  const toast = useToast()
  const [stats, setStats] = useState<GraphStatsInfo | null>(null)
  const [subgraph, setSubgraph] = useState<GraphSubgraph | null>(null)
  const [communities, setCommunities] = useState<GraphCommunityInfo[] | null>(null)
  const [activeCommunity, setActiveCommunity] = useState<number | null>(null)
  const [hoverCommunity, setHoverCommunity] = useState<number | null>(null)
  const [building, setBuilding] = useState(false)

  const isDark = typeof document !== 'undefined' && document.documentElement.getAttribute('data-theme') === 'dark'
  const ramp = isDark ? RAMP_DARK : RAMP_LIGHT
  const themeColor = (i: number) => ramp[((i % ramp.length) + ramp.length) % ramp.length]
  const kbName = kbs.find((k) => k.kb_id === kbId)?.name ?? kbId

  const load = useCallback(async () => {
    if (!session || !kbId) return
    setSubgraph(null); setCommunities(null); setActiveCommunity(null); setHoverCommunity(null)
    api.getGraph(session.token, kbId).then(setStats).catch(() => setStats(null))
    api.getGraphSubgraph(session.token, kbId, 80).then(setSubgraph).catch(() => setSubgraph(null))
    api.getCommunities(session.token, kbId).then((r) => setCommunities(r.communities)).catch(() => setCommunities([]))
  }, [session, kbId])

  useEffect(() => { void load() }, [load])

  async function build() {
    if (!session) return
    setBuilding(true)
    try {
      const r = await api.buildCommunities(session.token, kbId)
      // Re-fetch the themes AND the subgraph/stats so the map's node→community colouring updates
      // live — previously only the themes list refreshed, so a full page reload was needed to see
      // the rebuilt graph.
      setCommunities((await api.getCommunities(session.token, kbId)).communities)
      api.getGraphSubgraph(session.token, kbId, 80).then(setSubgraph).catch(() => {})
      api.getGraph(session.token, kbId).then(setStats).catch(() => {})
      setActiveCommunity(null); setHoverCommunity(null)
      toast.push('success', `Found ${r.communities} theme${r.communities === 1 ? '' : 's'}.`)
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not build themes.')
    } finally { setBuilding(false) }
  }

  const hasGraph = !!subgraph && subgraph.nodes.length > 0

  return (
    <div>
      <PageHeader eyebrow="Knowledge" title="Knowledge Graph"
        lead={`Explore the entities and relationships mined from ${kbName} — switch to 3D for a spatial view.`} />

      {!hasGraph ? (
        <Card>
          <div className="flex flex-col items-start gap-2" style={{ padding: '32px 8px' }}>
            <span className="eyebrow">No graph yet</span>
            <span className="font-display font-semibold" style={{ fontSize: 16, color: 'var(--text-heading)' }}>
              Nothing has been extracted from {kbName}
            </span>
            <p className="field-help" style={{ maxWidth: '58ch' }}>
              Entities and relationships are mined during indexing. Ingest documents into this KB first, then build
              themes — the graph clusters what it finds so broad questions can be answered across topics.
            </p>
            {stats && stats.nodes > 0 && (
              <Button className="mt-2" size="sm" variant="secondary" loading={building} onClick={build}>Build themes</Button>
            )}
            {subgraph === null && <Skeleton style={{ height: 40, marginTop: 8 }} />}
          </div>
        </Card>
      ) : (
        <div className="space-y-4">
          <div className="flex items-end justify-between flex-wrap gap-3">
            <div className="flex items-center gap-5" style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>
              <span><span className="mono" style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-heading)' }}>{(stats?.nodes ?? 0).toLocaleString()}</span> entities</span>
              <span><span className="mono" style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-heading)' }}>{(stats?.edges ?? 0).toLocaleString()}</span> relationships</span>
              <span><span className="mono" style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-heading)' }}>{(communities ?? []).length}</span> themes</span>
              <span className="mono" style={{ fontSize: 11 }}>showing top {subgraph!.nodes.length}</span>
            </div>
          </div>

          {/* Roomy full-page panel: graph + themes */}
          <div style={{ border: '1px solid var(--border)', background: 'var(--surface)', borderRadius: 'var(--radius-lg)', display: 'flex', height: 'clamp(420px, calc(100vh - 240px), 900px)', overflow: 'hidden' }}>
            <GraphCanvas data={subgraph!} communities={communities ?? []} activeCommunity={activeCommunity} hoverCommunity={hoverCommunity} onActiveCommunity={setActiveCommunity} height={860} />

            {/* Themes sidebar */}
            <aside style={{ width: 300, flexShrink: 0, borderLeft: '1px solid var(--border)', display: 'flex', flexDirection: 'column' }}>
              <div style={{ height: 44, flexShrink: 0, borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 16px' }}>
                <span style={{ fontFamily: 'var(--font-heading)', fontSize: 13, fontWeight: 600, color: 'var(--text-heading)' }}>Themes</span>
                <div className="flex items-center gap-1.5">
                  {activeCommunity != null && (
                    <button type="button" onClick={() => setActiveCommunity(null)} style={{ height: 24, padding: '0 9px', border: '1px solid var(--border)', borderRadius: 3, background: 'var(--surface)', color: 'var(--text-muted)', fontSize: 11, fontWeight: 600, cursor: 'pointer' }}>Clear</button>
                  )}
                  <Button size="sm" variant="secondary" loading={building} onClick={build} style={{ fontSize: 11, padding: '3px 9px', minHeight: 24 }}>
                    {communities && communities.length > 0 ? 'Rebuild' : 'Build'}
                  </Button>
                </div>
              </div>
              <div style={{ overflowY: 'auto', flex: 1, padding: 8 }}>
                {(communities ?? []).length === 0 && <p className="field-help" style={{ padding: 8 }}>No themes yet — build them to cluster entities.</p>}
                {(communities ?? []).map((c, i) => (
                  <button key={c.community_id} type="button"
                    onClick={() => setActiveCommunity(activeCommunity === c.community_id ? null : c.community_id)}
                    onMouseEnter={() => setHoverCommunity(c.community_id)} onMouseLeave={() => setHoverCommunity(null)}
                    className="w-full text-left rounded-md" style={{
                      padding: '8px 10px', marginBottom: 4, cursor: 'pointer', border: '1px solid',
                      borderColor: activeCommunity === c.community_id ? 'color-mix(in srgb, var(--action) 40%, transparent)' : 'var(--border)',
                      background: activeCommunity === c.community_id ? 'color-mix(in srgb, var(--action) 8%, transparent)' : 'transparent',
                    }}>
                    <div className="flex items-center gap-2">
                      <span style={{ width: 9, height: 9, borderRadius: 999, background: themeColor(i), flexShrink: 0 }} />
                      <span className="font-semibold truncate" style={{ fontSize: 12.5, color: 'var(--text-heading)' }}>{c.label || `Theme ${c.community_id}`}</span>
                      <span className="mono" style={{ marginLeft: 'auto', fontSize: 10.5, color: 'var(--text-muted)' }}>{c.size}</span>
                    </div>
                    {c.top_entities?.length > 0 && (
                      <div className="truncate" style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>{c.top_entities.slice(0, 5).join(' · ')}</div>
                    )}
                  </button>
                ))}
              </div>
            </aside>
          </div>
        </div>
      )}
    </div>
  )
}
