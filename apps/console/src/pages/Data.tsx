import { useCallback, useEffect, useState } from 'react'
import { Icon } from '../components/icons'
import { useToast } from '../components/overlay'
import { Alert, Button, Card, EmptyState, Field, PageHeader, SegmentedControl, Skeleton } from '../components/ui'
import { ApiError, api, type SqlAnswer, type SqlConnectionItem } from '../lib/api'
import { useAuth } from '../lib/auth'
import { useKb } from '../lib/kb'

type Mode = 'ask' | 'sql'

export function Data() {
  const { session } = useAuth()
  const { kbId, kbs } = useKb()
  const toast = useToast()
  const kbName = kbs.find((k) => k.kb_id === kbId)?.name ?? kbId

  const [conns, setConns] = useState<SqlConnectionItem[] | null>(null)
  const [acting, setActing] = useState<string | null>(null)

  // add form
  const [name, setName] = useState('')
  const [dsn, setDsn] = useState('')
  const [tables, setTables] = useState('')
  const [saving, setSaving] = useState(false)
  const [addErr, setAddErr] = useState<string | null>(null)

  // query
  const [connId, setConnId] = useState<string>('')
  const [mode, setMode] = useState<Mode>('ask')
  const [input, setInput] = useState('')
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<SqlAnswer | null>(null)
  const [err, setErr] = useState<string | null>(null)

  const load = useCallback(async () => {
    if (!session || !kbId) return
    const list = await api.listSqlConnections(session.token, kbId).catch(() => [])
    setConns(list)
    // Keep the query picker pointed at a connection that still exists; default to
    // the KB's default connection the first time the list loads.
    setConnId((prev) => (prev && list.some((c) => c.conn_id === prev)) ? prev : (list.find((c) => c.is_default) ?? list[0])?.conn_id ?? '')
  }, [session, kbId])

  useEffect(() => {
    setConns(null); setResult(null); setErr(null); setConnId('')
    void load()
  }, [load])

  async function addConnection() {
    if (!session) return
    const tableList = tables.split(/[\n,]/).map((s) => s.trim()).filter(Boolean)
    if (!name.trim() || !dsn.trim() || tableList.length === 0) {
      toast.push('error', 'Give a name, a DSN, and at least one allowlisted table.')
      return
    }
    setSaving(true); setAddErr(null)
    try {
      const conn = await api.addSqlConnection(session.token, kbId, {
        name: name.trim(), dsn: dsn.trim(), tables: tableList, make_default: !conns?.length,
      })
      setName(''); setDsn(''); setTables('')
      await load()
      toast.push('success', `Connected "${conn.name}" · ${conn.tables.length} table(s) allowlisted.`)
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : 'Could not save connection.'
      setAddErr(msg); toast.push('error', msg)
    } finally {
      setSaving(false)
    }
  }

  async function makeDefault(conn: SqlConnectionItem) {
    if (!session) return
    setActing(conn.conn_id)
    try {
      await api.setDefaultSqlConnection(session.token, kbId, conn.conn_id)
      await load()
      toast.push('success', `"${conn.name}" is now the default connection.`)
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Failed')
    } finally {
      setActing(null)
    }
  }

  async function remove(conn: SqlConnectionItem) {
    if (!session) return
    if (!window.confirm(`Disconnect "${conn.name}"? Queries against it will stop working until it's reconnected.`)) return
    setActing(conn.conn_id)
    try {
      await api.deleteSqlConnection(session.token, kbId, conn.conn_id)
      await load()
      toast.push('info', `Disconnected "${conn.name}".`)
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not disconnect.')
    } finally {
      setActing(null)
    }
  }

  async function run() {
    if (!session || !input.trim() || !connId) return
    setRunning(true); setResult(null); setErr(null)
    try {
      const r = mode === 'ask'
        ? await api.askSql(session.token, kbId, input.trim(), connId)
        : await api.executeSql(session.token, kbId, input.trim(), connId)
      setResult(r)
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : 'Query failed')
    } finally {
      setRunning(false)
    }
  }

  const selected = conns?.find((c) => c.conn_id === connId) ?? null

  return (
    <div>
      <PageHeader eyebrow="Knowledge" title="Data & SQL"
        lead={`Ask ${kbName}’s connected databases in plain language. Every query is governed — read-only, allowlisted tables, bounded, and audited.`} />

      <div className="grid lg:grid-cols-3 gap-5 items-start">
        {/* Connections */}
        <div className="lg:col-span-1 space-y-5">
          <Card>
            <div className="flex items-center gap-2 mb-1"><Icon.database size={16} style={{ color: 'var(--text-muted)' }} /><h2 className="font-display font-semibold" style={{ fontSize: 16 }}>Database connections</h2></div>
            <p className="field-help mb-3">A knowledge base can connect several databases — pick which one a question targets below the query box.</p>
            {conns === null ? <Skeleton style={{ height: 90 }} /> : conns.length === 0 ? (
              <p className="field-help mb-3">No database connected yet. Register a read-only connection below (admin).</p>
            ) : (
              <ul className="space-y-2 mb-1">
                {conns.map((c) => (
                  <li key={c.conn_id} className="rounded-lg p-3" style={{ border: `1px solid ${c.is_default ? 'color-mix(in srgb, var(--success) 40%, var(--border))' : 'var(--border)'}`, background: c.is_default ? 'color-mix(in srgb, var(--success) 6%, transparent)' : 'transparent' }}>
                    <div className="flex items-center gap-2">
                      <span aria-hidden style={{ width: 8, height: 8, borderRadius: 999, background: 'var(--success)', flexShrink: 0 }} />
                      <div className="min-w-0 flex-grow">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-sm truncate" style={{ color: 'var(--text-heading)' }}>{c.name}</span>
                          {c.is_default && <span className="pill" style={{ fontSize: 10, color: 'var(--success)', borderColor: 'color-mix(in srgb, var(--success) 35%, transparent)', background: 'color-mix(in srgb, var(--success) 12%, transparent)' }}><Icon.check size={11} />default</span>}
                        </div>
                        <div className="mono text-xs truncate" style={{ color: 'var(--text-muted)' }}>{c.database ?? '?'} · {c.host}{c.port ? `:${c.port}` : ''} · {c.dialect}</div>
                      </div>
                      {!c.is_default && <Button size="sm" variant="subtle" disabled={acting !== null} onClick={() => makeDefault(c)}>Make default</Button>}
                      <button type="button" onClick={() => remove(c)} disabled={acting !== null} aria-label={`Disconnect ${c.name}`} title="Disconnect" className="btn btn-subtle btn-icon" style={{ color: 'var(--danger)' }}><Icon.trash size={15} /></button>
                    </div>
                    <div className="flex flex-wrap gap-1.5 mt-2">
                      {c.tables.map((tb) => <span key={tb} className="chip mono" style={{ fontSize: 11 }}>{tb}</span>)}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </Card>

          <Card>
            <div className="flex items-center gap-2 mb-4"><Icon.plus size={16} style={{ color: 'var(--text-muted)' }} /><h2 className="font-display font-semibold" style={{ fontSize: 16 }}>Add a connection</h2></div>
            <div className="space-y-3">
              <Field label="Name" help="A label for this connection, e.g. 'Sales DB'.">
                <input className="field-input" value={name} onChange={(e) => setName(e.target.value)} placeholder="Sales DB" />
              </Field>
              <Field label="Connection string (DSN)" help="Host must be egress-allowlisted. Dialect is detected from the scheme. e.g. postgresql://ro_user:***@db:5432/hr">
                <input className="field-input mono" style={{ fontSize: 12 }} value={dsn} onChange={(e) => setDsn(e.target.value)} placeholder="postgresql://…" />
              </Field>
              <Field label="Allowlisted tables" help="Comma/newline-separated — only these can be read.">
                <textarea className="field-input mono" style={{ fontSize: 12 }} rows={2} value={tables} onChange={(e) => setTables(e.target.value)} placeholder="employees, departments" />
              </Field>
              <Button size="sm" loading={saving} onClick={addConnection} icon={<Icon.check size={15} />}>Add connection</Button>
              {addErr && <Alert kind="error">{addErr}</Alert>}
            </div>
          </Card>
        </div>

        {/* Query */}
        <div className="lg:col-span-2 space-y-4">
          <Card>
            <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
              <h2 className="font-display font-semibold" style={{ fontSize: 16 }}>Query</h2>
              <SegmentedControl options={[{ value: 'ask', label: 'Ask in English' }, { value: 'sql', label: 'Write SQL' }]} value={mode} onChange={(m) => { setMode(m); setInput(''); setResult(null); setErr(null) }} />
            </div>
            {conns && conns.length > 1 && (
              <Field label="Connection" help="Which database this question is asked against.">
                <select className="field-input" value={connId} onChange={(e) => setConnId(e.target.value)}>
                  {conns.map((c) => <option key={c.conn_id} value={c.conn_id}>{c.name}{c.is_default ? ' (default)' : ''}</option>)}
                </select>
              </Field>
            )}
            <textarea className={mode === 'sql' ? 'field-input mono' : 'field-input'} rows={3}
              style={mode === 'sql' ? { fontSize: 13 } : undefined}
              value={input} onChange={(e) => setInput(e.target.value)}
              placeholder={mode === 'ask' ? 'e.g. How many employees are in each department?' : 'SELECT dept, COUNT(*) FROM employees GROUP BY dept'} />
            <div className="flex items-center gap-2 mt-3">
              <Button size="sm" loading={running} disabled={!connId || !input.trim()} onClick={run} icon={<Icon.ask size={15} />}>
                {mode === 'ask' ? 'Ask' : 'Run'}
              </Button>
              {!connId && <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Connect a database first.</span>}
              {selected && <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Asking <span className="mono">{selected.name}</span></span>}
            </div>
          </Card>

          {err && <Alert kind="error">{err}</Alert>}

          {result && (
            <Card>
              <div className="flex items-center justify-between mb-2">
                <h2 className="font-display font-semibold" style={{ fontSize: 16 }}>Result</h2>
                <div className="flex items-center gap-2">
                  {result.connection_name && <span className="chip mono" style={{ fontSize: 11 }}>{result.connection_name}</span>}
                  <span className="chip mono" style={{ fontSize: 11 }}>{result.row_count} row(s)</span>
                </div>
              </div>
              {/* Lead with the ANSWER — the result table — since the user asked in English. */}
              {result.row_count === 0 ? (
                <EmptyState icon={<Icon.doc size={24} />} title="No rows" />
              ) : (
                <div className="surface" style={{ overflowX: 'auto', overflowY: 'auto', maxHeight: '46vh', borderRadius: 8 }}>
                  <table className="tbl" style={{ minWidth: 'max-content' }}>
                    <thead><tr>{result.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
                    <tbody>
                      {result.rows.map((row, i) => (
                        <tr key={i}>{row.map((cell, j) => {
                          const v = cell === null ? '—' : String(cell)
                          return (
                            <td key={j} className="mono" title={v.length > 60 ? v : undefined}
                              style={{ fontSize: 12.5, maxWidth: 280, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                              {v}
                            </td>
                          )
                        })}</tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {/* The SQL is secondary to the answer — collapsed by default; expand to inspect the
                  query generated from the question and the governed form actually run. */}
              {(result.generated_sql || result.sql) && (
                <details className="mt-3">
                  <summary className="eyebrow" style={{ cursor: 'pointer', userSelect: 'none' }}>
                    {result.generated_sql ? 'View generated SQL' : 'View SQL'}
                  </summary>
                  <div className="mt-2">
                    {result.generated_sql && (
                      <div className="mb-3">
                        <div className="eyebrow mb-1">Generated SQL</div>
                        <pre className="mono" style={{ fontSize: 12, background: 'var(--surface-2)', padding: 10, borderRadius: 8, overflowX: 'auto', color: 'var(--text-muted)' }}>{result.generated_sql}</pre>
                      </div>
                    )}
                    <div>
                      <div className="eyebrow mb-1">Executed (governed) SQL</div>
                      <pre className="mono" style={{ fontSize: 12, background: 'var(--surface-2)', padding: 10, borderRadius: 8, overflowX: 'auto', color: 'var(--text-primary)' }}>{result.sql}</pre>
                    </div>
                  </div>
                </details>
              )}
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
