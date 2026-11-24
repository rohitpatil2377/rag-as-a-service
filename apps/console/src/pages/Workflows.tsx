import { useEffect, useState } from 'react'
import { Icon } from '../components/icons'
import { useToast } from '../components/overlay'
import { Alert, Button, Card, EmptyState, Field, PageHeader, Skeleton } from '../components/ui'
import { ApiError, api, type WorkflowDef, type WorkflowResult } from '../lib/api'
import { hasRole, useAuth } from '../lib/auth'
import { useKb } from '../lib/kb'

const _EXAMPLE = {
  id: 'triage',
  name: 'Ticket triage',
  start: 'answer',
  steps: [
    { id: 'answer', type: 'answer', config: { question_from: 'query', output: 'a' }, next: 'route' },
    { id: 'route', type: 'decision', rules: [{ all: [{ field: 'a.grounded', op: 'eq', value: true }], goto: 'done' }], next: 'escalate' },
    { id: 'done', type: 'output' },
    { id: 'escalate', type: 'output' },
  ],
}

export function Workflows() {
  const { session } = useAuth()
  const { kbId, kbs } = useKb()
  const toast = useToast()
  const kbName = kbs.find((k) => k.kb_id === kbId)?.name ?? kbId

  const [defs, setDefs] = useState<WorkflowDef[] | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [inputs, setInputs] = useState('{\n  "query": "What is the leave policy?"\n}')
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<WorkflowResult | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [defText, setDefText] = useState(JSON.stringify(_EXAMPLE, null, 2))
  const [saving, setSaving] = useState(false)
  const [showEditor, setShowEditor] = useState(false)
  // Backend gates save/delete/run on `admin` — mirror that here so a non-admin sees disabled
  // controls (with why) instead of clicking through to a 403.
  const canManage = hasRole(session, 'admin')

  async function load() {
    if (!session || !kbId) return
    setDefs(null)
    try {
      const list = await api.listWorkflows(session.token, kbId)
      setDefs(list)
      setSelected((s) => s ?? list[0]?.id ?? null)
    } catch {
      setDefs([])
    }
  }

  useEffect(() => { setSelected(null); setResult(null); void load() /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [session, kbId])

  async function saveDef() {
    if (!session) return
    let parsed: WorkflowDef
    try { parsed = JSON.parse(defText) } catch { toast.push('error', 'Definition is not valid JSON.'); return }
    if (!parsed.id || !parsed.start || !Array.isArray(parsed.steps)) { toast.push('error', 'Definition needs id, start, and steps[].'); return }
    setSaving(true)
    try {
      await api.saveWorkflow(session.token, kbId, parsed)
      toast.push('success', `Saved workflow "${parsed.name || parsed.id}".`)
      setShowEditor(false)
      setSelected(parsed.id)
      await load()
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Save failed')
    } finally {
      setSaving(false)
    }
  }

  async function removeDef(id: string) {
    if (!session || !window.confirm(`Delete workflow "${id}"?`)) return
    try {
      await api.deleteWorkflow(session.token, kbId, id)
      if (selected === id) setSelected(null)
      await load()
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Delete failed')
    }
  }

  async function run() {
    if (!session || !selected) return
    let parsedInputs: Record<string, unknown>
    try { parsedInputs = JSON.parse(inputs || '{}') } catch { setErr('Inputs must be valid JSON.'); return }
    setRunning(true); setResult(null); setErr(null)
    try {
      setResult(await api.runWorkflow(session.token, kbId, selected, parsedInputs))
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : 'Run failed')
    } finally {
      setRunning(false)
    }
  }

  const current = defs?.find((d) => d.id === selected) ?? null

  return (
    <div>
      <PageHeader eyebrow="Automate" title="Workflows"
        lead={`Deterministic, auditable multi-step agents for ${kbName} — retrieve, answer, query SQL, and branch on rules, with a full decision trace.`}
        actions={<Button size="sm" onClick={() => setShowEditor((v) => !v)} icon={<Icon.plus size={15} />}>{showEditor ? 'Close editor' : 'New / edit'}</Button>} />

      {showEditor && (
        <Card className="mb-5">
          <h2 className="font-display font-semibold mb-1" style={{ fontSize: 16 }}>Workflow definition</h2>
          <p className="field-help mb-3">A declarative JSON workflow. Step types: <span className="mono">answer</span>, <span className="mono">sql</span>, <span className="mono">retrieve</span>, <span className="mono">set</span>, <span className="mono">decision</span>, <span className="mono">output</span>. Branches use a rules engine over the run context.</p>
          <textarea className="field-input mono" rows={14} value={defText} onChange={(e) => setDefText(e.target.value)} style={{ fontSize: 12 }} />
          <Button className="mt-3" size="sm" loading={saving} disabled={!canManage} title={canManage ? undefined : 'Requires admin role'}
            onClick={saveDef} icon={<Icon.check size={15} />}>Save workflow</Button>
        </Card>
      )}

      <div className="grid lg:grid-cols-3 gap-5 items-start">
        {/* List */}
        <div className="lg:col-span-1">
          <Card>
            <h2 className="font-display font-semibold mb-3" style={{ fontSize: 16 }}>Your workflows</h2>
            {defs === null ? <Skeleton style={{ height: 80 }} /> : defs.length === 0 ? (
              <EmptyState icon={<Icon.connectors size={24} />} title="No workflows yet" hint="Create one with “New / edit”." />
            ) : (
              <ul className="space-y-1.5">
                {defs.map((d) => (
                  <li key={d.id} className="flex items-center gap-1.5 rounded-md"
                    style={{ background: d.id === selected ? 'color-mix(in srgb, var(--action) 10%, transparent)' : 'transparent', border: `1px solid ${d.id === selected ? 'color-mix(in srgb, var(--action) 35%, transparent)' : 'var(--border)'}` }}>
                    <button type="button" onClick={() => { setSelected(d.id); setResult(null) }}
                      className="flex-grow flex items-center gap-2.5 px-3 py-2.5 text-left min-w-0"
                      style={{ cursor: 'pointer', background: 'transparent', border: 'none' }}>
                      <Icon.connectors size={15} style={{ color: d.id === selected ? 'var(--action)' : 'var(--text-muted)' }} />
                      <span className="flex-grow min-w-0">
                        <span className="block font-semibold truncate" style={{ color: 'var(--text-heading)' }}>{d.name || d.id}</span>
                        <span className="block mono truncate" style={{ fontSize: 11, color: 'var(--text-muted)' }}>{d.steps.length} step(s)</span>
                      </span>
                    </button>
                    <button type="button" onClick={() => removeDef(d.id)} disabled={!canManage}
                      aria-label={`Delete ${d.id}`} title={canManage ? `Delete ${d.id}` : 'Requires admin role'}
                      className="btn btn-subtle btn-icon mr-1" style={{ color: 'var(--danger)' }}><Icon.trash size={14} /></button>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </div>

        {/* Run + trace */}
        <div className="lg:col-span-2 space-y-4">
          {!current ? (
            <Card><EmptyState icon={<Icon.connectors size={28} />} title="Select or create a workflow" hint="Pick a workflow to run it and see its decision trace." /></Card>
          ) : (
            <>
              <Card>
                <h2 className="font-display font-semibold mb-3" style={{ fontSize: 16 }}>Run “{current.name || current.id}”</h2>
                <Field label="Inputs (JSON)" help="Initial context, e.g. the user’s question.">
                  <textarea className="field-input mono" rows={4} value={inputs} onChange={(e) => setInputs(e.target.value)} style={{ fontSize: 12.5 }} />
                </Field>
                <Button className="mt-3" size="sm" loading={running} disabled={!canManage} title={canManage ? undefined : 'Requires admin role'}
                  onClick={run} icon={<Icon.play size={15} />}>Run workflow</Button>
              </Card>

              {err && <Alert kind="error">{err}</Alert>}

              {result && (
                <Card>
                  <div className="flex items-center justify-between mb-3">
                    <h2 className="font-display font-semibold" style={{ fontSize: 16 }}>Decision trace</h2>
                    <span className="pill" style={{ fontSize: 11, color: result.status === 'completed' ? 'var(--success)' : 'var(--warning)', borderColor: 'currentColor', background: 'transparent', textTransform: 'capitalize' }}>{result.status.replace('_', ' ')}</span>
                  </div>
                  <ol className="space-y-2">
                    {result.trace.map((s, i) => (
                      <li key={i} className="flex items-start gap-3">
                        <span className="mono shrink-0" style={{ fontSize: 11, color: 'var(--text-muted)', width: 18, textAlign: 'right' }}>{i + 1}</span>
                        <div className="flex-grow min-w-0 rounded-md p-2.5" style={{ border: '1px solid var(--border)' }}>
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="font-semibold" style={{ fontSize: 13, color: 'var(--text-heading)' }}>{s.step_id}</span>
                            <span className="chip mono" style={{ fontSize: 10 }}>{s.type}</span>
                            {s.goto && <span className="text-xs" style={{ color: 'var(--text-muted)' }}>→ {s.goto}{s.matched_rule ? ' (rule)' : ''}</span>}
                          </div>
                          {s.summary && <div className="mono mt-1" style={{ fontSize: 11.5, color: 'var(--text-muted)', wordBreak: 'break-word' }}>{s.summary}</div>}
                        </div>
                      </li>
                    ))}
                  </ol>
                  {result.error && <div className="mt-3"><Alert kind="error">{result.error}</Alert></div>}
                </Card>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
