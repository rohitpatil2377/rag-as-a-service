import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Icon } from '../components/icons'
import { useToast } from '../components/overlay'
import { Button, Card, Field, PageHeader, Skeleton, Table } from '../components/ui'
import { ApiError, api, type AgentRunResult, type CreatedApiKey, type McpInfo } from '../lib/api'
import { useAuth } from '../lib/auth'
import { useI18n } from '../lib/i18n'

function CopyBlock({ text }: { text: string }) {
  const toast = useToast()
  return (
    <div className="relative">
      <pre className="mono" style={{ fontSize: 12, background: 'var(--surface-2)', padding: '12px 14px', borderRadius: 10, overflowX: 'auto', color: 'var(--text-primary)', border: '1px solid var(--border)' }}>{text}</pre>
      <button onClick={() => { void navigator.clipboard.writeText(text); toast.push('success', 'Copied.') }}
        className="absolute" style={{ top: 8, right: 8, background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 6, padding: '3px 8px', fontSize: 11, cursor: 'pointer', color: 'var(--text-muted)' }}>
        Copy
      </button>
    </div>
  )
}

export function Agents() {
  const { session } = useAuth()
  const { t } = useI18n()
  const toast = useToast()
  const [info, setInfo] = useState<McpInfo | null>(null)
  const [managesKb, setManagesKb] = useState(false)
  const [genName, setGenName] = useState('MCP access token')
  const [genKey, setGenKey] = useState<CreatedApiKey | null>(null)
  const [generating, setGenerating] = useState(false)
  const [goal, setGoal] = useState('')
  const [running, setRunning] = useState(false)
  const [run, setRun] = useState<AgentRunResult | null>(null)
  const origin = window.location.origin
  // Owner/admin, OR a KB manager (manages ≥1 KB) — managers get read-only agent tokens + agent runs.
  const canGenerate = !!session && (session.roles.includes('admin') || session.roles.includes('owner') || managesKb)

  useEffect(() => {
    if (!session) return
    api.mcpInfo(session.token).then(setInfo).catch(() => setInfo(null))
    api.myManagedKbs(session.token).then((ks) => setManagesKb(ks.length > 0)).catch(() => setManagesKb(false))
  }, [session])

  async function generateToken() {
    if (!session) return
    setGenerating(true)
    try {
      // Read scope = reader role only: least privilege for an agent (retrieve + query, no ingest/admin).
      const created = await api.createApiKey(session.token, genName.trim() || 'MCP access token', 'read')
      setGenKey(created)
      toast.push('success', 'Access token generated — copy it now, it is shown only once.')
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not generate a token.')
    } finally {
      setGenerating(false)
    }
  }

  async function runAgent() {
    if (!session || !goal.trim()) return
    setRunning(true); setRun(null)
    try {
      setRun(await api.runAgent(session.token, goal.trim()))
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Agent run failed.')
    } finally {
      setRunning(false)
    }
  }

  const url = info ? `${origin}${info.endpoint_path}` : `${origin}/mcp`
  // Once a token is generated, the config + curl snippets carry the REAL key so they're copy-paste ready.
  const bearer = genKey?.key ?? '<YOUR_API_KEY>'
  const desktopConfig = JSON.stringify(
    { mcpServers: { 'ragaas': { url, headers: { Authorization: `Bearer ${bearer}` } } } },
    null, 2,
  )

  return (
    <div>
      <PageHeader eyebrow={t('eyebrow.platform')} title={t('page.agents.title')}
        lead={t('page.agents.lead')} />

      {!info ? <Skeleton style={{ height: 160 }} /> : (
        <div className="space-y-5">
          <Card>
            <div className="flex items-center gap-2 mb-3"><Icon.layers size={16} style={{ color: 'var(--text-muted)' }} /><h2 className="font-display font-semibold" style={{ fontSize: 16 }}>Connection</h2></div>
            <dl className="grid sm:grid-cols-2 gap-x-8 gap-y-3 text-sm">
              <div><dt className="eyebrow mb-0.5">Endpoint URL</dt><dd className="mono" style={{ color: 'var(--text-heading)', wordBreak: 'break-all' }}>{url}</dd></div>
              <div><dt className="eyebrow mb-0.5">Transport</dt><dd style={{ color: 'var(--text-heading)' }}>{info.transport}</dd></div>
              <div className="sm:col-span-2"><dt className="eyebrow mb-0.5">Authentication</dt><dd style={{ color: 'var(--text-muted)' }}>{info.auth} — generate an access token below, or manage keys in <Link to="/team" className="underline font-semibold" style={{ color: 'var(--action)' }}>Team → API keys</Link>.</dd></div>
              <div className="sm:col-span-2"><dt className="eyebrow mb-0.5">Tenancy</dt><dd style={{ color: 'var(--text-muted)' }}>{info.tenancy}</dd></div>
            </dl>
          </Card>

          <Card>
            <div className="flex items-center gap-2 mb-1"><Icon.layers size={16} style={{ color: 'var(--text-muted)' }} /><h2 className="font-display font-semibold" style={{ fontSize: 16 }}>Run an agent</h2></div>
            <p className="field-help mb-4">Give the agent a goal. It <strong>plans</strong> over the tools below, an independent <strong>critic</strong> reviews the plan, and <strong>write steps pause for human approval</strong> — every step is tenant-scoped and audited.</p>
            {canGenerate ? (
              <div className="flex items-end gap-3 flex-wrap">
                <div style={{ flex: '1 1 320px', minWidth: 0 }}>
                  <Field label="Goal" help="e.g. List the knowledge bases, then summarise what data they contain.">
                    <input value={goal} onChange={(e) => setGoal(e.target.value)} placeholder="What should the agent accomplish?"
                      style={{ width: '100%', height: 36, padding: '0 12px', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--surface-2)', color: 'var(--text-primary)', fontSize: 13 }} />
                  </Field>
                </div>
                <Button onClick={runAgent} loading={running} disabled={running || !goal.trim()} icon={<Icon.plus size={15} />}>Run agent</Button>
              </div>
            ) : (
              <p className="text-sm rounded-lg p-3" style={{ color: 'var(--text-muted)', background: 'var(--surface-2)', border: '1px dashed var(--border-strong)' }}>Only an owner, admin, or KB manager can run agents.</p>
            )}
            {run && (
              <div className="mt-4 space-y-3 text-sm">
                <div>
                  <div className="eyebrow mb-1">Plan</div>
                  <ol className="space-y-1" style={{ color: 'var(--text-muted)' }}>
                    {run.plan.steps.map((s, i) => (
                      <li key={i}><span className="mono" style={{ color: 'var(--text-heading)' }}>{s.tool}</span>{s.rationale ? ` — ${s.rationale}` : ''}</li>
                    ))}
                  </ol>
                </div>
                {run.review && (
                  <div className="rounded-lg p-3" style={{ background: 'var(--surface-2)', border: '1px solid var(--border)' }}>
                    <div className="eyebrow mb-1">Critic review — {run.review.approved ? 'approved' : 'blocked'}</div>
                    {run.review.concerns?.length > 0 && <ul style={{ color: 'var(--text-muted)' }}>{run.review.concerns.map((c, i) => <li key={i}>• {c}</li>)}</ul>}
                    {run.review.suggestion && <p className="mt-1" style={{ color: 'var(--text-muted)' }}>Suggestion: {run.review.suggestion}</p>}
                  </div>
                )}
                {run.pending_approval_id && <p style={{ color: 'var(--action)', fontWeight: 600 }}>Paused for human approval (id {run.pending_approval_id}) — a critical step needs sign-off.</p>}
                {run.halted_reason && !run.answer && <p style={{ color: 'var(--text-muted)' }}>Halted: <span className="mono">{run.halted_reason}</span></p>}
                {run.answer && (
                  <div><div className="eyebrow mb-1">Answer</div><p style={{ color: 'var(--text-heading)', whiteSpace: 'pre-wrap' }}>{run.answer}</p></div>
                )}
              </div>
            )}
          </Card>

          <Card>
            <div className="flex items-center gap-2 mb-1"><Icon.key size={16} style={{ color: 'var(--text-muted)' }} /><h2 className="font-display font-semibold" style={{ fontSize: 16 }}>Access token</h2></div>
            <p className="field-help mb-4">Agents authenticate with a scoped access token — a <strong>read-only</strong> key bound to this workspace (retrieve + query only, no ingest or admin). Generate one and paste it into the client config below.</p>
            {canGenerate ? (
              <div className="flex items-end gap-3 flex-wrap">
                <div style={{ flex: '1 1 260px', minWidth: 0 }}>
                  <Field label="Token name" help="A label so you can recognise and revoke it later.">
                    <input value={genName} onChange={(e) => setGenName(e.target.value)} placeholder="MCP access token"
                      style={{ width: '100%', height: 36, padding: '0 12px', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--surface-2)', color: 'var(--text-primary)', fontSize: 13 }} />
                  </Field>
                </div>
                <Button onClick={generateToken} loading={generating} disabled={generating} icon={<Icon.plus size={15} />}>Generate access token</Button>
              </div>
            ) : (
              <p className="text-sm rounded-lg p-3" style={{ color: 'var(--text-muted)', background: 'var(--surface-2)', border: '1px dashed var(--border-strong)' }}>
                Only an owner, admin, or KB manager can generate access tokens. Ask an admin, or manage keys in <Link to="/team" className="underline font-semibold" style={{ color: 'var(--action)' }}>Team → API keys</Link>.
              </p>
            )}
            {genKey && (
              <div className="mt-4">
                <p className="field-help mb-2" style={{ color: 'var(--action)', fontWeight: 600 }}>Copy this now — it is shown once and cannot be retrieved again.</p>
                <CopyBlock text={genKey.key} />
                <p className="field-help mt-2">Token <span className="mono">{genKey.name}</span> · scope <span className="mono">read</span>. The config and curl below are now filled in with it.</p>
              </div>
            )}
          </Card>

          <Card>
            <div className="flex items-center gap-2 mb-3"><Icon.layers size={16} style={{ color: 'var(--text-muted)' }} /><h2 className="font-display font-semibold" style={{ fontSize: 16 }}>Available tools</h2></div>
            <Table>
              <thead><tr><th>Tool</th><th>What it does</th></tr></thead>
              <tbody>
                {info.tools.map((tool) => (
                  <tr key={tool.name}>
                    <td className="mono" style={{ color: 'var(--text-heading)', whiteSpace: 'nowrap' }}>{tool.name}</td>
                    <td style={{ color: 'var(--text-muted)' }}>{tool.description}</td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </Card>

          <Card>
            <div className="flex items-center gap-2 mb-1"><Icon.connectors size={16} style={{ color: 'var(--text-muted)' }} /><h2 className="font-display font-semibold" style={{ fontSize: 16 }}>Client setup</h2></div>
            <p className="field-help mb-3">Point any MCP-capable agent host at the endpoint. Example remote-MCP client config{genKey ? ' (token filled in from above)' : ' (replace the key with a generated access token)'}:</p>
            <CopyBlock text={desktopConfig} />
            <p className="field-help mt-4 mb-2">Or drive it programmatically — the same auth works with our SDKs and the raw <span className="mono">/v1</span> API. Quick check with curl:</p>
            <CopyBlock text={`curl -sN ${url} \\\n  -H "Authorization: Bearer ${bearer}" \\\n  -H "Accept: text/event-stream"`} />
          </Card>
        </div>
      )}
    </div>
  )
}
