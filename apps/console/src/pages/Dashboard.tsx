import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Icon } from '../components/icons'
import { Card, Skeleton, StatTile } from '../components/ui'
import { api, type HitlReview } from '../lib/api'
import { useAuth } from '../lib/auth'
import { useKb } from '../lib/kb'
import { Onboarding } from '../components/Onboarding'
import { useI18n } from '../lib/i18n'

function greeting(): string {
  const h = new Date().getHours()
  return h < 12 ? 'Good morning' : h < 17 ? 'Good afternoon' : 'Good evening'
}

export function Dashboard() {
  const { session } = useAuth()
  const { kbId, kbs, setKbId } = useKb()
  const navigate = useNavigate()
  const { t } = useI18n()
  const [q, setQ] = useState('')
  const [docs, setDocs] = useState<number | null>(null)
  const [reviews, setReviews] = useState<HitlReview[]>([])
  const [activeModel, setActiveModel] = useState<{ label: string; via: string } | null>(null)
  // Both KPI windows come from the SAME meter as the per-KB list below, so the numbers agree.
  const [day, setDay] = useState<number | null>(null)
  const [week, setWeek] = useState<number | null>(null)
  const [recentQs, setRecentQs] = useState<{ text: string; at: string }[] | null>(null)
  const [orgKbs, setOrgKbs] = useState<{ kb_id: string; name: string; queries: number }[] | null>(null)
  // Mirrors the /activity response. `detail` was omitted here, which is why the card silently
  // dropped the chunk counts the API computes for exactly this feed.
  const [activity, setActivity] = useState<{ action: string; label: string; detail: string; at: string }[] | null>(null)

  useEffect(() => {
    if (!session || !kbId) return
    const { token } = session
    setDocs(null); setActiveModel(null); setDay(null); setWeek(null); setActivity(null); setRecentQs(null)
    // Count LIVE documents (source of truth), not audit 'index' events — those persist in the
    // append-only audit log after a document is deleted, so they'd overstate the real count.
    api.listDocuments(token, kbId).then((d) => setDocs(d.length)).catch(() => setDocs(0))
    api.listReviews(token, kbId, 'pending').then(setReviews).catch(() => setReviews([]))
    api.usage(token, kbId, '24h').then((u) => setDay(u.queries)).catch(() => setDay(null))
    api.usage(token, kbId, '7d').then((u) => setWeek(u.queries)).catch(() => setWeek(null))
    api.systemInfo(token).then((s) => {
      api.getModelConnection(token, kbId).then((c) => {
        if (c.configured && c.active && c.model) setActiveModel({ label: c.model, via: 'your model' })
        else setActiveModel({ label: s.llm_model, via: `platform · ${s.llm_backend}` })
      }).catch(() => setActiveModel({ label: s.llm_model, via: `platform · ${s.llm_backend}` }))
    }).catch(() => undefined)
    // Real questions asked against this KB (audit trail) — never a hardcoded list. Greetings
    // and condenser artefacts left in older history are filtered out. The audit trail is a
    // management-tier read (server-gated), so only fetch it for admins/owners — an ordinary
    // member skips it (no doomed 403) and simply sees no recent-questions panel.
    const canReadAudit = (session.roles ?? []).some((r) => r === 'admin' || r === 'owner')
    if (canReadAudit) {
      api.listAudit(token, kbId, undefined, 80).then((events) => {
        const seen = new Set<string>()
        const qs: { text: string; at: string }[] = []
        for (const ev of events) {
          if (ev.action !== 'query') continue
          const text = String((ev.payload as Record<string, unknown>)?.query ?? '').trim()
          const key = text.toLowerCase()
          if (!text || text.length > 120 || seen.has(key)) continue
          if (/^(hi+|hey+|hello+|thanks?|thank you|ok(ay)?|bye|test(ing)?)\b/i.test(text) || text.includes('—')) continue
          seen.add(key); qs.push({ text, at: ev.at })
          if (qs.length === 6) break
        }
        setRecentQs(qs)
      }).catch(() => setRecentQs([]))
    } else {
      setRecentQs([])
    }
    api.activity(token, kbId).then((a) => setActivity(a.slice(0, 6))).catch(() => setActivity([]))
  }, [session, kbId])

  // Org-wide 7-day queries per KB (same meter as the KPI above).
  useEffect(() => {
    if (!session || kbs.length < 2) { setOrgKbs(null); return }
    let live = true
    Promise.all(kbs.slice(0, 6).map(async (k) => {
      try { return { kb_id: k.kb_id, name: k.name, queries: (await api.usage(session.token, k.kb_id, '7d')).queries } }
      catch { return { kb_id: k.kb_id, name: k.name, queries: 0 } }
    })).then((rows) => { if (live) setOrgKbs(rows.sort((a, b) => b.queries - a.queries)) })
    return () => { live = false }
  }, [session, kbs])

  const loading = docs === null
  const kbName = kbs.find((k) => k.kb_id === kbId)?.name ?? kbId

  // Quick actions, filtered to what this user's roles can actually open (same rule as the nav)
  // so nobody is offered a link that 403s.
  const roles = session?.roles ?? []
  const can = (...need: string[]) => need.some((r) => roles.includes(r))
  const actions: QuickAction[] = [
    ...(can('ingestor', 'admin') ? [
      { to: '/knowledge-base', icon: <Icon.upload size={16} />, t: 'Add documents', d: 'Upload files to this KB' },
      { to: '/connectors', icon: <Icon.connectors size={16} />, t: 'Connect a source', d: 'Web, S3 or object store' },
    ] : []),
    ...(can('reviewer', 'admin') ? [
      { to: '/reviews', icon: <Icon.review size={16} />, t: 'Review answers', d: reviews.length ? `${reviews.length} need review` : 'Nothing waiting', badge: reviews.length || undefined },
    ] : []),
    ...(can('admin') ? [
      { to: '/audit', icon: <Icon.audit size={16} />, t: 'Audit trail', d: 'Tamper-evident log' },
    ] : []),
  ]

  function ask(question: string) {
    if (question.trim()) navigate(`/ask?q=${encodeURIComponent(question.trim())}`)
  }

  return (
    <div>
      {/* First-run onboarding (FR-H10) — self-hides once setup is complete. */}
      <Onboarding />

      <div className="mb-4">
        <div className="flex items-center gap-3 mb-1"><span className="w-9 rule-gold" /><span className="eyebrow">{t('dashboard.eyebrow', 'RAG as a Service')}</span></div>
        <h1 className="font-display font-semibold tracking-tight" style={{ fontSize: 24, letterSpacing: '-0.01em' }}>
          {greeting()}, {session?.name?.split(' ')[0] ?? session?.subject ?? ''}
        </h1>
        <div className="flex items-center gap-2 flex-wrap text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
          <span className="font-semibold" style={{ color: 'var(--action)' }}>{kbName}</span>
          {docs !== null && <>· {docs} document{docs === 1 ? '' : 's'}</>}
          {activeModel && <>· <span className="mono">{activeModel.label.split('/').pop()}</span> ({activeModel.via})</>}
        </div>
      </div>

      <Card className="!p-3 mb-3">
        <form onSubmit={(e) => { e.preventDefault(); ask(q) }} className="flex gap-3 items-center">
          <span style={{ color: 'var(--text-muted)' }}><Icon.ask size={18} /></span>
          <input className="field-input" style={{ border: 'none', boxShadow: 'none', fontSize: 15, minHeight: 34 }}
            value={q} onChange={(e) => setQ(e.target.value)} placeholder={`Ask ${kbName} anything…`} autoFocus />
          <button type="submit" className="btn btn-primary" disabled={!q.trim()}>Ask</button>
        </form>
      </Card>
      {loading && <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">{[0, 1, 2, 3].map((i) => <Skeleton key={i} style={{ height: 92 }} />)}</div>}

      {/* Render once loaded even with zero documents: the tiles show zeros, the activity card
          has its own "Nothing yet" empty state, and the quick actions (incl. Add documents) stay
          visible — so a fresh KB reads as empty-but-ready, not broken/blank. */}
      {!loading && (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
            <Link to="/knowledge-base"><StatTile label="Documents" value={docs} sub="in this knowledge base" tone="--text-heading" icon={<Icon.database size={18} />} /></Link>
            {/* Usage & Showback is retired; the query trail itself is the better destination
                and is strictly more informative than the page these used to link to. */}
            <Link to="/audit"><StatTile label="Queries · 24h" value={day ?? '—'} sub={kbName} tone="--text-heading" icon={<Icon.ask size={18} />} /></Link>
            <Link to="/audit"><StatTile label="Queries · 7d" value={week ?? '—'} sub={kbName} tone="--text-heading" icon={<Icon.activity size={18} />} /></Link>
            <Link to="/reviews"><StatTile label="Review queue" value={reviews.length} sub={reviews.length ? 'need a decision' : 'all clear'} tone={reviews.length ? '--warning' : '--text-muted'} icon={<Icon.review size={18} />} /></Link>
          </div>

          <div className={`grid gap-4 mb-4 items-stretch ${orgKbs && orgKbs.length > 1 ? 'lg:grid-cols-2' : ''}`}>
            {/* Recent questions — real traffic against this KB; click to ask again. Falls back
                to ingest/index activity when the KB has no query history yet. */}
            <Card>
              <div className="flex items-center justify-between mb-2.5">
                <h2 className="font-display font-semibold" style={{ fontSize: 15 }}>
                  {recentQs && recentQs.length > 0 ? 'Recent questions' : 'Recent activity'}
                </h2>
                <Link to="/audit" className="text-xs underline" style={{ color: 'var(--action)' }}>audit trail →</Link>
              </div>
              {recentQs === null && activity === null && <Skeleton style={{ height: 120 }} />}
              {recentQs !== null && recentQs.length > 0 && (
                <ul className="space-y-1">
                  {recentQs.map((r) => (
                    <li key={r.text}>
                      <button type="button" onClick={() => ask(r.text)}
                        className="w-full text-left rounded-md px-2 py-1.5 interactive flex items-baseline justify-between gap-3"
                        style={{ background: 'none', border: 'none', cursor: 'pointer' }} title="Ask this again">
                        {/* min-w-0: without it `truncate` cannot engage and a long question
                            pushes the date out of the card instead of ellipsising. */}
                        <span className="truncate min-w-0 text-sm" title={r.text} style={{ color: 'var(--text-primary)' }}>{r.text}</span>
                        <span className="mono text-xs shrink-0" style={{ color: 'var(--text-muted)' }}>{new Date(r.at).toLocaleDateString(undefined, { day: 'numeric', month: 'short' })}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
              {recentQs !== null && recentQs.length === 0 && activity !== null && activity.length > 0 && (
                <ul className="space-y-1.5">
                  {activity.map((a, i) => (
                    <li key={i} className="flex items-baseline gap-2.5 text-sm">
                      {/* Fixed-width action column so document names start on a common left edge:
                          inline, "ingest" and "index" differ in width and every row began at a
                          different x, which is what makes the list unreadable at a glance. */}
                      <span className="mono text-xs shrink-0" style={{ width: 52, color: 'var(--text-muted)' }}>{a.action}</span>
                      {/* min-w-0 is what makes `truncate` work at all: a flex item defaults to
                          min-width:auto and so refuses to shrink below its content, meaning a
                          long name (an OCR upload's UUID) pushed the timestamp instead of
                          ellipsising. title= keeps the full name reachable. */}
                      <span className="truncate min-w-0 flex-1" title={a.label} style={{ color: 'var(--text-primary)' }}>{a.label}</span>
                      {/* The API computes a detail for exactly this feed ("12 chunks",
                          "erasure certificate"); it was being dropped on the floor here. */}
                      {a.detail && <span className="text-xs shrink-0" style={{ color: 'var(--text-muted)' }}>{a.detail}</span>}
                      <span className="mono text-xs shrink-0" style={{ color: 'var(--text-muted)' }}>{new Date(a.at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                    </li>
                  ))}
                </ul>
              )}
              {recentQs !== null && recentQs.length === 0 && activity !== null && activity.length === 0 && (
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Nothing yet — ask your first question above.</p>
              )}
            </Card>

            {orgKbs && orgKbs.length > 1 ? (
              <Card>
                <h2 className="font-display font-semibold mb-2.5" style={{ fontSize: 15 }}>All knowledge bases · queries 7d</h2>
                <ul className="space-y-2">
                  {orgKbs.map((k) => {
                    const max = Math.max(1, ...orgKbs.map((x) => x.queries))
                    const isCurrent = k.kb_id === kbId
                    return (
                      <li key={k.kb_id}>
                        <button type="button" onClick={() => setKbId(k.kb_id)} className="w-full text-left" style={{ background: 'none', border: 'none', padding: 0, cursor: 'pointer' }}>
                          <div className="flex items-center justify-between text-sm mb-1">
                            <span className="font-medium truncate min-w-0" title={k.name} style={{ color: isCurrent ? 'var(--action)' : 'var(--text-heading)' }}>
                              {k.name}{isCurrent && <span className="text-xs font-normal" style={{ color: 'var(--text-muted)' }}> · selected</span>}
                            </span>
                            <span className="mono num text-xs" style={{ color: 'var(--text-muted)' }}>{k.queries}</span>
                          </div>
                          <div style={{ height: 5, borderRadius: 999, background: 'var(--surface-2)', overflow: 'hidden' }}>
                            <div style={{ width: `${(k.queries / max) * 100}%`, height: '100%', borderRadius: 999, background: isCurrent ? 'var(--action)' : 'color-mix(in srgb, var(--action) 45%, transparent)' }} />
                          </div>
                        </button>
                      </li>
                    )
                  })}
                </ul>
              </Card>
            ) : null}
          </div>

          {actions.length > 0 && (
            <Card>
              <h2 className="font-display font-semibold mb-3" style={{ fontSize: 15 }}>Quick actions</h2>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                {actions.map((a) => <Action key={a.to} {...a} />)}
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  )
}

interface QuickAction { to: string; icon: React.ReactNode; t: string; d: string; badge?: number }

function Action({ to, icon, t, d, badge }: QuickAction) {
  return (
    <Link to={to} className="surface interactive px-3 py-3 flex gap-2.5 items-start">
      <span className="shrink-0 rounded-lg flex items-center justify-center"
        style={{ width: 30, height: 30, color: 'var(--action)', background: 'color-mix(in srgb, var(--action) 12%, transparent)' }}>
        {icon}
      </span>
      <span className="min-w-0">
        <span className="flex items-center gap-1.5">
          <span className="font-semibold text-sm truncate" style={{ color: 'var(--text-heading)' }}>{t}</span>
          {badge ? (
            <span className="mono shrink-0" style={{ fontSize: 10, fontWeight: 700, lineHeight: '15px', minWidth: 15, height: 15, padding: '0 4px', textAlign: 'center', borderRadius: 8, color: 'var(--warning)', background: 'color-mix(in srgb, var(--warning) 18%, transparent)' }}>{badge}</span>
          ) : null}
        </span>
        <span className="block text-xs truncate" style={{ color: 'var(--text-muted)' }}>{d}</span>
      </span>
    </Link>
  )
}
