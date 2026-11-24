import { useCallback, useEffect, useState } from 'react'
import { useI18n } from '../lib/i18n'
import { useToast } from '../components/overlay'
import { Icon } from '../components/icons'
import { AnswerPill, Alert, Button, Card, PageHeader, SegmentedControl, Skeleton, StatTile } from '../components/ui'
import { ApiError, api, type AnswerStatus, type EvalReport, type EvalRunsResponse, type QualityReport } from '../lib/api'
import { useAuth } from '../lib/auth'
import { useKb } from '../lib/kb'

type Window = '24h' | '7d' | '30d' | 'all'
const WINDOWS: { value: Window; label: string }[] = [
  { value: '24h', label: '24h' }, { value: '7d', label: '7d' }, { value: '30d', label: '30d' }, { value: 'all', label: 'All' },
]

const BUCKETS: { key: string; label: string; tone: string }[] = [
  { key: 'high', label: 'Faithful (≥0.7)', tone: '--success' },
  { key: 'mid', label: 'Borderline', tone: '--warning' },
  { key: 'low', label: 'Unsupported (<0.4)', tone: '--danger' },
]

const REASON_LABEL: Record<string, string> = {
  retrieval_confidence_below_threshold: 'Nothing relevant found',
  faithfulness_below_threshold: 'Answer not supported by sources',
  blocked_prompt_injection: 'Prompt injection blocked',
  blocked_toxic_output: 'Toxic output blocked',
}

export function Quality() {
  const { session } = useAuth()
  const { t } = useI18n()
  const { kbId, kbs } = useKb()
  const [window, setWindow] = useState<Window>('7d')
  const [report, setReport] = useState<QualityReport | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [goldenText, setGoldenText] = useState('What is the platform uptime target? :: 99.5')
  const [running, setRunning] = useState(false)
  const [evalR, setEvalR] = useState<EvalReport | null>(null)
  const [cont, setCont] = useState<EvalRunsResponse | null>(null)
  const [contBusy, setContBusy] = useState(false)
  const toast = useToast()
  const kbName = kbs.find((k) => k.kb_id === kbId)?.name ?? kbId

  // Load this KB's saved golden set into the runner.
  useEffect(() => {
    if (!session || !kbId) return
    api.getGoldenSet(session.token, kbId).then((g) => {
      if (g.questions.length) {
        setGoldenText(g.questions.map((q) => (q.expected ? `${q.question} :: ${q.expected}` : q.question)).join('\n'))
      }
    }).catch(() => undefined)
  }, [session, kbId])

  function parseGolden(): { question: string; expected?: string }[] {
    return goldenText.split('\n').map((ln) => ln.trim()).filter(Boolean).slice(0, 50).map((ln) => {
      const [q, expected] = ln.split('::').map((x) => x.trim())
      return expected ? { question: q, expected } : { question: q }
    })
  }

  async function saveGolden() {
    if (!session) return
    try {
      await api.setGoldenSet(session.token, kbId, parseGolden())
      toast.push('success', 'Golden set saved with this knowledge base.')
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Save failed')
    }
  }

  const [generating, setGenerating] = useState(false)
  async function generateGolden() {
    if (!session || generating) return
    setGenerating(true)
    try {
      const g = await api.generateGoldenSet(session.token, kbId, 6)
      if (!g.questions.length) {
        toast.push('error', 'No drafts — add documents to this KB first.')
        return
      }
      const drafted = g.questions.map((q) => (q.expected ? `${q.question} :: ${q.expected}` : q.question)).join('\n')
      // Append to whatever is already in the box so a review-in-progress isn't lost.
      setGoldenText((cur) => (cur.trim() ? `${cur.trim()}\n${drafted}` : drafted))
      toast.push('success', `Drafted ${g.questions.length} question${g.questions.length === 1 ? '' : 's'} from your documents — review, then Save.`)
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Generation failed')
    } finally {
      setGenerating(false)
    }
  }

  const load = useCallback(async () => {
    if (!session || !kbId) return
    setError(null)
    try {
      setReport(await api.quality(session.token, kbId, window))
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Failed to load quality report')
    }
  }, [session, kbId, window])

  useEffect(() => { setReport(null); void load() }, [load])

  const loadCont = useCallback(async () => {
    if (!session || !kbId) return
    try { setCont(await api.getEvalSchedule(session.token, kbId)) } catch { setCont(null) }
  }, [session, kbId])

  useEffect(() => { setCont(null); void loadCont() }, [loadCont])

  async function saveSchedule(patch: Partial<{ enabled: boolean; interval_hours: number; min_grounded_rate: number; min_faithfulness: number }>) {
    if (!session || !cont) return
    const s = cont.schedule
    const body = {
      enabled: patch.enabled ?? s?.enabled ?? false,
      interval_hours: patch.interval_hours ?? s?.interval_hours ?? 24,
      min_grounded_rate: patch.min_grounded_rate ?? s?.min_grounded_rate ?? 0.6,
      min_faithfulness: patch.min_faithfulness ?? s?.min_faithfulness ?? 0.5,
    }
    try {
      await api.setEvalSchedule(session.token, kbId, body)
      await loadCont()
      toast.push('success', body.enabled ? 'Continuous evaluation enabled.' : 'Continuous evaluation paused.')
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Could not update schedule.')
    }
  }

  async function runContNow() {
    if (!session || contBusy) return
    setContBusy(true)
    try {
      const r = await api.runEvalNow(session.token, kbId)
      await loadCont()
      toast.push(r.breached ? 'error' : 'success',
        r.breached ? `Breach: ${r.breach_reasons.join('; ')}` : `Healthy — ${r.total} scored, ${r.grounded_rate === null ? '—' : Math.round(r.grounded_rate * 100) + '%'} grounded.`)
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Run failed.')
    } finally {
      setContBusy(false)
    }
  }

  async function runGolden() {
    if (!session || running) return
    const questions = parseGolden().slice(0, 20)
    if (!questions.length) return
    setRunning(true); setEvalR(null); setError(null)
    try {
      setEvalR(await api.runEval(session.token, kbId, questions))
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Evaluation failed')
    } finally {
      setRunning(false)
    }
  }

  const bucketTotal = report ? Object.values(report.faithfulness_buckets).reduce((a, b) => a + b, 0) : 0

  return (
    <div>
      <PageHeader eyebrow={t('eyebrow.govern')} title={t('page.quality.title')}
        lead={t('page.quality.lead', { name: kbName })}
        actions={<SegmentedControl value={window} onChange={setWindow} options={WINDOWS} />} />

      {error && <div className="mb-4"><Alert kind="error">{error}</Alert></div>}
      {!report && !error && <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">{[0, 1, 2, 3].map((i) => <Skeleton key={i} style={{ height: 104 }} />)}</div>}

      {report && (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-5 mb-6">
            <StatTile label="Grounded rate" value={report.grounded_rate === null ? '—' : `${Math.round(report.grounded_rate * 100)}%`}
              sub={`${report.grounded} of ${report.answers} answers`} tone={report.grounded_rate === null ? '--text-muted' : report.grounded_rate >= 0.8 ? '--success' : '--warning'} icon={<Icon.check size={18} />} />
            <StatTile label="Avg faithfulness" value={report.faithfulness_avg === null ? '—' : report.faithfulness_avg.toFixed(2)}
              sub="verifier score (0–1)" tone={report.faithfulness_avg === null ? '--text-muted' : report.faithfulness_avg >= 0.7 ? '--success' : '--warning'} icon={<Icon.shield size={18} />} />
            <StatTile label="Held for review" value={report.flagged} sub="borderline answers → humans" tone={report.flagged ? '--warning' : '--text-muted'} icon={<Icon.review size={18} />} />
            <StatTile label="Human agreement" value={report.human_agreement_rate === null ? '—' : `${Math.round(report.human_agreement_rate * 100)}%`}
              sub={`${report.reviews_accepted} accepted · ${report.reviews_modified} modified · ${report.reviews_overridden} overridden`}
              tone={report.human_agreement_rate === null ? '--text-muted' : '--info'} icon={<Icon.shield size={18} />} />
            <StatTile label="End-user satisfaction" value={report.satisfaction_rate === null ? '—' : `${Math.round(report.satisfaction_rate * 100)}%`}
              sub={`${report.feedback_up} up · ${report.feedback_down} down`}
              tone={report.satisfaction_rate === null ? '--text-muted' : report.satisfaction_rate >= 0.7 ? '--success' : '--warning'} icon={<Icon.thumbUp size={18} />} />
          </div>

          <div className="grid lg:grid-cols-2 gap-5 mb-6">
            <Card>
              <h2 className="font-display font-semibold mb-4" style={{ fontSize: 16 }}>{t('quality.card.faithfulness_dist')}</h2>
              {bucketTotal === 0 ? (
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No verified answers in this window yet — ask something.</p>
              ) : (
                <div className="space-y-3">
                  {BUCKETS.map((b) => {
                    const n = report.faithfulness_buckets[b.key] ?? 0
                    const pct = Math.round((n / bucketTotal) * 100)
                    return (
                      <div key={b.key}>
                        <div className="flex items-center justify-between text-sm mb-1">
                          <span style={{ color: 'var(--text-primary)' }}>{b.label}</span>
                          <span className="mono num text-xs" style={{ color: 'var(--text-muted)' }}>{n} · {pct}%</span>
                        </div>
                        <div className="rounded-full overflow-hidden" style={{ height: 8, background: 'color-mix(in srgb, var(--text-muted) 14%, transparent)' }}>
                          <div className="h-full rounded-full" style={{ width: `${Math.max(pct, n ? 3 : 0)}%`, background: `color-mix(in srgb, var(${b.tone}) 80%, transparent)` }} />
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </Card>

            <Card>
              <h2 className="font-display font-semibold mb-4" style={{ fontSize: 16 }}>{t('quality.card.withheld')}</h2>
              {Object.keys(report.abstain_reasons).length === 0 && Object.keys(report.guardrail_blocks).length === 0 ? (
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No abstains or blocks in this window.</p>
              ) : (
                <ul className="space-y-2.5">
                  {Object.entries(report.abstain_reasons).map(([reason, n]) => (
                    <li key={reason} className="flex items-center justify-between text-sm">
                      <span className="flex items-center gap-2" style={{ color: 'var(--text-primary)' }}><Icon.minus size={14} style={{ color: 'var(--text-muted)' }} />{REASON_LABEL[reason] ?? reason}</span>
                      <span className="mono num" style={{ color: 'var(--text-muted)' }}>{n}</span>
                    </li>
                  ))}
                  {Object.entries(report.guardrail_blocks).map(([stage, n]) => (
                    <li key={stage} className="flex items-center justify-between text-sm">
                      <span className="flex items-center gap-2" style={{ color: 'var(--text-primary)' }}><Icon.shield size={14} style={{ color: 'var(--danger)' }} />Guardrail block ({stage})</span>
                      <span className="mono num" style={{ color: 'var(--text-muted)' }}>{n}</span>
                    </li>
                  ))}
                </ul>
              )}
            </Card>
          </div>
        </>
      )}

      {/* Continuous evaluation (FR-G9) */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-1 flex-wrap gap-2">
          <div className="flex items-center gap-2"><Icon.activity size={16} style={{ color: 'var(--text-muted)' }} /><h2 className="font-display font-semibold" style={{ fontSize: 16 }}>{t('quality.card.continuous_eval')}</h2></div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" loading={contBusy} onClick={runContNow} icon={!contBusy && <Icon.play size={15} />} style={{ fontSize: 13 }}>
              {contBusy ? 'Running…' : 'Run now'}
            </Button>
            <Button variant={cont?.schedule?.enabled ? 'secondary' : 'primary'} onClick={() => saveSchedule({ enabled: !cont?.schedule?.enabled })} style={{ fontSize: 13 }}>
              {cont?.schedule?.enabled ? 'Pause' : 'Enable'}
            </Button>
          </div>
        </div>
        <p className="field-help mb-3">
          Periodically scores this KB's golden set plus a sample of real questions users asked, and alerts if answer quality drifts below your floor.
          {cont && !cont.enabled && <span style={{ color: 'var(--warning)' }}> Scheduler is off platform-wide — “Run now” still works; scheduled runs need <span className="mono">RAG_CONTINUOUS_EVAL_ENABLED</span>.</span>}
        </p>

        {cont?.schedule?.enabled && (
          <div className="grid grid-cols-2 lg:grid-cols-3 gap-3 mb-4">
            <label className="text-xs" style={{ color: 'var(--text-muted)' }}>Interval (hours)
              <input className="field-input mt-1" type="number" min={1} max={720} defaultValue={cont.schedule.interval_hours}
                onBlur={(e) => saveSchedule({ interval_hours: Math.max(1, Math.min(720, Number(e.target.value) || 24)) })} />
            </label>
            <label className="text-xs" style={{ color: 'var(--text-muted)' }}>Min grounded rate
              <input className="field-input mt-1" type="number" min={0} max={1} step={0.05} defaultValue={cont.schedule.min_grounded_rate}
                onBlur={(e) => saveSchedule({ min_grounded_rate: Math.max(0, Math.min(1, Number(e.target.value) || 0)) })} />
            </label>
            <label className="text-xs" style={{ color: 'var(--text-muted)' }}>Min faithfulness
              <input className="field-input mt-1" type="number" min={0} max={1} step={0.05} defaultValue={cont.schedule.min_faithfulness}
                onBlur={(e) => saveSchedule({ min_faithfulness: Math.max(0, Math.min(1, Number(e.target.value) || 0)) })} />
            </label>
          </div>
        )}

        {cont && cont.runs.length > 0 && (
          <>
            <div className="eyebrow mb-2">Recent runs</div>
            <ul className="space-y-1.5">
              {cont.runs.slice(0, 8).map((r, i) => (
                <li key={i} className="flex items-center justify-between text-sm surface-2 px-3 py-2 rounded-lg">
                  <span className="flex items-center gap-2">
                    {r.action === 'eval_alert'
                      ? <span className="pill" style={{ color: 'var(--danger)', borderColor: 'color-mix(in srgb, var(--danger) 35%, transparent)', background: 'color-mix(in srgb, var(--danger) 12%, transparent)' }}>alert</span>
                      : r.breached
                        ? <span className="pill" style={{ color: 'var(--warning)' }}>breach</span>
                        : <span className="pill" style={{ color: 'var(--success)', borderColor: 'color-mix(in srgb, var(--success) 35%, transparent)', background: 'color-mix(in srgb, var(--success) 12%, transparent)' }}>ok</span>}
                    <span style={{ color: 'var(--text-muted)' }} className="mono text-xs">{new Date(r.at).toLocaleString()}</span>
                  </span>
                  <span className="mono text-xs" style={{ color: 'var(--text-muted)' }}>
                    {r.action === 'eval_alert'
                      ? (r.reasons as string[] | undefined)?.join('; ')
                      : `${r.total ?? 0} scored · ${r.grounded_rate === null || r.grounded_rate === undefined ? '—' : Math.round((r.grounded_rate as number) * 100) + '%'} grounded · faith ${r.faithfulness_avg ?? '—'}`}
                  </span>
                </li>
              ))}
            </ul>
          </>
        )}
      </Card>

      {/* Golden-set evaluation */}
      <Card>
        <div className="flex items-center justify-between mb-1 flex-wrap gap-2">
          <div className="flex items-center gap-2"><Icon.activity size={16} style={{ color: 'var(--text-muted)' }} /><h2 className="font-display font-semibold" style={{ fontSize: 16 }}>{t('quality.card.run_eval')}</h2></div>
          <Button variant="ghost" loading={generating} onClick={generateGolden} icon={!generating && <Icon.doc size={15} />} style={{ fontSize: 13 }}>
            {generating ? 'Drafting…' : 'Generate from documents'}
          </Button>
        </div>
        <p className="field-help mb-4">One question per line. Add <span className="mono">:: expected fact</span> to score whether the answer contains it. Up to 20 questions run through the real pipeline, then scored on the full sovereign RAGAS suite — answer relevancy, context precision, and (with an expected fact) context recall. <span style={{ color: 'var(--text-muted)' }}>“Generate from documents” drafts a baseline from your own content (FR-G8) — review and edit before saving.</span></p>
        <textarea className="field-input" rows={4} value={goldenText} onChange={(e) => setGoldenText(e.target.value)}
          placeholder={'What is the uptime target? :: 99.5\nWhat is the notice period? :: 90 days'} style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }} />
        <div className="flex gap-2 mt-4">
          <Button loading={running} disabled={!goldenText.trim()} onClick={runGolden} icon={!running && <Icon.play size={16} />}>
            {running ? 'Evaluating…' : 'Run evaluation'}
          </Button>
          <Button variant="ghost" disabled={!goldenText.trim() || running} onClick={saveGolden} icon={<Icon.check size={16} />}>
            Save as golden set
          </Button>
        </div>

        {evalR && (
          <div className="mt-6">
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
              <StatTile label="Fact hit rate" value={evalR.hit_rate === null ? '—' : `${Math.round(evalR.hit_rate * 100)}%`} sub={`${evalR.hits}/${evalR.with_expected} expected facts found`} tone={evalR.hit_rate !== null && evalR.hit_rate >= 0.8 ? '--success' : '--warning'} icon={<Icon.check size={18} />} />
              <StatTile label="Grounded" value={`${evalR.grounded}/${evalR.total}`} tone="--success" icon={<Icon.check size={18} />} />
              <StatTile label="Abstained" value={evalR.abstained} sub="honest declines" tone="--text-muted" icon={<Icon.minus size={18} />} />
              <StatTile label="Avg faithfulness" value={evalR.faithfulness_avg === null ? '—' : evalR.faithfulness_avg.toFixed(2)} tone="--info" icon={<Icon.shield size={18} />} />
            </div>

            {evalR.ragas_scored > 0 && (
              <div className="mb-4">
                <div className="flex items-center gap-2 mb-2">
                  <span className="eyebrow" style={{ color: 'var(--action)' }}>Sovereign RAGAS</span>
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    retrieval quality · judged on your own Sarvam + BGE-M3 · {evalR.ragas_scored}/{evalR.total} scored
                  </span>
                </div>
                <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
                  <StatTile label="Answer relevancy" value={evalR.answer_relevancy_avg === null ? '—' : evalR.answer_relevancy_avg.toFixed(2)}
                    sub="answer addresses the question" tone={evalR.answer_relevancy_avg !== null && evalR.answer_relevancy_avg >= 0.7 ? '--success' : '--warning'} icon={<Icon.activity size={18} />} />
                  <StatTile label="Context precision" value={evalR.context_precision_avg === null ? '—' : evalR.context_precision_avg.toFixed(2)}
                    sub="useful evidence ranked on top" tone={evalR.context_precision_avg !== null && evalR.context_precision_avg >= 0.7 ? '--success' : '--warning'} icon={<Icon.activity size={18} />} />
                  <StatTile label="Context recall" value={evalR.context_recall_avg === null ? '—' : evalR.context_recall_avg.toFixed(2)}
                    sub="expected evidence retrieved" tone={evalR.context_recall_avg !== null && evalR.context_recall_avg >= 0.7 ? '--success' : '--warning'} icon={<Icon.activity size={18} />} />
                </div>
              </div>
            )}

            <ul className="space-y-2">
              {evalR.rows.map((r, i) => (
                <li key={i} className="surface-2 p-3 rounded-lg">
                  <div className="flex items-center justify-between gap-3 mb-1 flex-wrap">
                    <span className="text-sm font-semibold" style={{ color: 'var(--text-heading)' }}>{r.question}</span>
                    <span className="flex items-center gap-2">
                      {r.hit !== null && (
                        <span className="pill" style={r.hit ? { color: 'var(--success)', borderColor: 'color-mix(in srgb, var(--success) 35%, transparent)', background: 'color-mix(in srgb, var(--success) 12%, transparent)' } : { color: 'var(--danger)', borderColor: 'color-mix(in srgb, var(--danger) 35%, transparent)', background: 'color-mix(in srgb, var(--danger) 12%, transparent)' }}>
                          {r.hit ? 'fact found' : 'fact missing'}
                        </span>
                      )}
                      <AnswerPill status={r.status as AnswerStatus} />
                    </span>
                  </div>
                  <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{r.answer}</p>
                  {(r.answer_relevancy !== null || r.context_precision !== null || r.context_recall !== null) && (
                    <div className="flex gap-3 mt-2 text-xs mono" style={{ color: 'var(--text-muted)' }}>
                      {r.faithfulness !== null && <span title="Faithfulness">faith {r.faithfulness.toFixed(2)}</span>}
                      {r.answer_relevancy !== null && <span title="Answer relevancy">rel {r.answer_relevancy.toFixed(2)}</span>}
                      {r.context_precision !== null && <span title="Context precision">prec {r.context_precision.toFixed(2)}</span>}
                      {r.context_recall !== null && <span title="Context recall">rec {r.context_recall.toFixed(2)}</span>}
                    </div>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}
      </Card>
    </div>
  )
}
