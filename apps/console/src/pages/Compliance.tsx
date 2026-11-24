import { useCallback, useEffect, useState } from 'react'
import { useI18n } from '../lib/i18n'
import { Icon } from '../components/icons'
import { useToast } from '../components/overlay'
import { Alert, Button, Card, Field, PageHeader, Skeleton, StatTile } from '../components/ui'
import { ApiError, api, type Attestation, type ComplianceReport, type CryptoStatus, type ErasureCertificate, type RetentionResult } from '../lib/api'
import { useAuth } from '../lib/auth'
import { useKb } from '../lib/kb'

export function Compliance() {
  const { session } = useAuth()
  const { t } = useI18n()
  const { kbId, kbs } = useKb()
  const toast = useToast()
  const kbName = kbs.find((k) => k.kb_id === kbId)?.name ?? kbId

  const [att, setAtt] = useState<Attestation | null>(null)
  const [attErr, setAttErr] = useState<string | null>(null)
  const [days, setDays] = useState('')
  const [savingRet, setSavingRet] = useState(false)
  const [applying, setApplying] = useState(false)
  const [retResult, setRetResult] = useState<RetentionResult | null>(null)
  const [subject, setSubject] = useState('')
  const [docIds, setDocIds] = useState('')
  const [reason, setReason] = useState('data_principal_request')
  const [erasing, setErasing] = useState(false)
  const [cert, setCert] = useState<ErasureCertificate | null>(null)
  const [report, setReport] = useState<ComplianceReport | null>(null)
  const [crypto, setCrypto] = useState<CryptoStatus | null>(null)
  const [rotating, setRotating] = useState(false)

  const loadAtt = useCallback(async () => {
    if (!session || !kbId) return
    setAtt(null); setAttErr(null)
    try {
      setAtt(await api.attestation(session.token, kbId))
    } catch (err) {
      setAttErr(err instanceof ApiError ? err.message : 'Could not load attestation')
    }
  }, [session, kbId])

  useEffect(() => { void loadAtt() }, [loadAtt])

  useEffect(() => {
    if (!session || !kbId) return
    setReport(null); setCrypto(null)
    api.complianceFrameworks(session.token, kbId).then(setReport).catch(() => setReport(null))
    api.cryptoStatus(session.token, kbId).then(setCrypto).catch(() => setCrypto(null))
  }, [session, kbId])

  async function rotateKey() {
    if (!session || !window.confirm('Rotate this KB’s encryption key? Stored model keys are re-encrypted with the new key.')) return
    setRotating(true)
    try {
      const r = await api.rotateCryptoKey(session.token, kbId)
      setCrypto(await api.cryptoStatus(session.token, kbId))
      toast.push('success', `Key rotated · ${r.reencrypted_connections} secret(s) re-encrypted.`)
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Rotation failed')
    } finally {
      setRotating(false)
    }
  }

  useEffect(() => {
    if (!session || !kbId) return
    api.getRetention(session.token, kbId).then((r) => setDays(r.retention_days ? String(r.retention_days) : '')).catch(() => undefined)
    setRetResult(null); setCert(null)
  }, [session, kbId])

  async function saveRetention() {
    if (!session) return
    setSavingRet(true)
    try {
      const r = await api.setRetention(session.token, kbId, Number(days) || 0)
      setDays(r.retention_days ? String(r.retention_days) : '')
      toast.push('success', r.retention_days ? `Documents older than ${r.retention_days} days will be purged on apply.` : 'Retention disabled — documents kept indefinitely.')
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Save failed')
    } finally {
      setSavingRet(false)
    }
  }

  async function applyRetention() {
    if (!session || !window.confirm('Erase every document older than the retention window now? This cannot be undone.')) return
    setApplying(true); setRetResult(null)
    try {
      const r = await api.applyRetention(session.token, kbId)
      setRetResult(r)
      toast.push(r.documents_erased.length ? 'info' : 'success', r.documents_erased.length ? `Purged ${r.documents_erased.length} expired document(s).` : 'Nothing past the retention window.')
      void loadAtt()
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Apply failed')
    } finally {
      setApplying(false)
    }
  }

  async function runErasure() {
    if (!session) return
    const ids = docIds.split(/[\n,]/).map((s) => s.trim()).filter(Boolean)
    if (!subject.trim() && ids.length === 0) { toast.push('error', 'Give a data-principal subject or document ids.'); return }
    const what = subject.trim() ? `all data mentioning "${subject.trim()}"` : `${ids.length} document(s)`
    if (!window.confirm(`Erase ${what} from every store and chat history? This is irreversible and issues a DPDP erasure certificate.`)) return
    setErasing(true); setCert(null)
    try {
      const c = await api.erase(session.token, kbId, { subject: subject.trim() || undefined, doc_ids: ids, reason })
      setCert(c)
      setSubject(''); setDocIds('')
      toast.push('success', `Erased ${c.documents_erased.length} document(s) · certificate issued.`)
      void loadAtt()
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Erasure failed')
    } finally {
      setErasing(false)
    }
  }

  return (
    <div>
      <PageHeader eyebrow={t('eyebrow.govern')} title={t('page.compliance.title')}
        lead={t('page.compliance.lead', { name: kbName })}
        actions={<Button variant="subtle" size="sm" onClick={loadAtt} icon={<Icon.scale size={15} />}>Refresh</Button>} />

      {/* Residency attestation */}
      {attErr && <div className="mb-4"><Alert kind="error">{attErr}</Alert></div>}
      {!att && !attErr && <div className="grid grid-cols-2 lg:grid-cols-4 gap-5 mb-6">{[0, 1, 2, 3].map((i) => <Skeleton key={i} style={{ height: 104 }} />)}</div>}
      {att && (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-5 mb-6">
            <StatTile label="Data residency" value="India" sub="self-hosted · sovereign" tone="--success" icon={<Icon.shield size={18} />} />
            <StatTile label="Sovereignty guard" value={att.sovereignty_enforced ? 'Enforced' : 'Off'} sub="runtime egress allowlist" tone={att.sovereignty_enforced ? '--success' : '--danger'} icon={<Icon.shield size={18} />} />
            <StatTile label="Audit chain" value={att.audit_chain_verified ? 'Verified' : 'Broken'} sub={`${att.audit_events} events`} tone={att.audit_chain_verified ? '--success' : '--danger'} icon={<Icon.check size={18} />} />
            <StatTile label="Retention" value={att.retention_days ? `${att.retention_days}d` : 'None'} sub={`${att.documents} documents held`} tone={att.retention_days ? '--info' : '--text-muted'} icon={<Icon.scale size={18} />} />
          </div>

          <Card className="mb-6">
            <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
              <h2 className="font-display font-semibold" style={{ fontSize: 16 }}>{t('compliance.card.attestation')}</h2>
              <Button variant="ghost" size="sm" onClick={() => downloadJson(`attestation-${kbId}.json`, att)} icon={<Icon.doc size={15} />}>Export JSON</Button>
            </div>
            <dl className="grid sm:grid-cols-2 gap-x-8 gap-y-2.5 text-sm">
              <Row k="Data location">{att.data_residency}</Row>
              <Row k="Encryption at rest">{att.encryption_at_rest ? 'AES-256' : '—'}</Row>
              <Row k="Encryption in transit">{att.encryption_in_transit ? 'TLS 1.2+' : '—'}</Row>
              <Row k="Egress allowlist">{att.egress_allowlist.length} host(s)</Row>
              <Row k="Generated">{new Date(att.generated_at).toLocaleString()}</Row>
              <Row k="Audit events">{att.audit_events}</Row>
            </dl>
            <div className="flex flex-wrap gap-1.5 mt-3">
              {att.egress_allowlist.map((h) => <span key={h} className="chip mono" style={{ fontSize: 11 }}>{h}</span>)}
            </div>
            {att.notes.length > 0 && (
              <ul className="mt-4 space-y-1.5">
                {att.notes.map((n, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm" style={{ color: 'var(--warning)' }}><Icon.alert size={14} className="mt-0.5 shrink-0" />{n}</li>
                ))}
              </ul>
            )}
          </Card>
        </>
      )}

      {/* Responsible-AI framework attestation (#7) */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-1 flex-wrap gap-2">
          <div className="flex items-center gap-2"><Icon.scale size={16} style={{ color: 'var(--text-muted)' }} /><h2 className="font-display font-semibold" style={{ fontSize: 16 }}>Responsible-AI frameworks</h2></div>
          {report && <span className="chip mono" style={{ fontSize: 11 }}>{report.summary.satisfied}/{report.summary.controls} controls satisfied</span>}
        </div>
        <p className="field-help mb-4">This platform’s live controls mapped to the frameworks Indian buyers cite. Status is computed from your actual configuration — not a static PDF.</p>
        {!report ? <Skeleton style={{ height: 120 }} /> : (
          <div className="grid md:grid-cols-2 gap-4">
            {report.frameworks.map((fw) => (
              <div key={fw.id} className="rounded-lg p-3" style={{ border: '1px solid var(--border)' }}>
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <div className="font-display font-semibold" style={{ fontSize: 14, color: 'var(--text-heading)' }}>{fw.name}</div>
                    <div className="mono" style={{ fontSize: 10, color: 'var(--text-muted)' }}>{fw.version}</div>
                  </div>
                  <span className="chip mono" style={{ fontSize: 11, color: fw.satisfied === fw.total ? 'var(--success)' : 'var(--info)' }}>{fw.satisfied}/{fw.total}</span>
                </div>
                <ul className="space-y-1.5">
                  {fw.controls.map((c) => (
                    <li key={c.id} className="flex items-start gap-2" style={{ fontSize: 12.5 }} title={c.evidence.join(' · ')}>
                      <span className="mt-1 shrink-0 rounded-full" style={{ width: 7, height: 7, background: statusColor(c.status) }} />
                      <span style={{ color: 'var(--text-primary)' }}>{c.title}</span>
                      <span className="ml-auto pill" style={{ fontSize: 9.5, color: statusColor(c.status), borderColor: 'currentColor', background: 'transparent', textTransform: 'capitalize', whiteSpace: 'nowrap' }}>{c.status.replace('_', ' ')}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        )}
        {report && report.notes.length > 0 && (
          <ul className="mt-4 space-y-1.5">
            {report.notes.map((n, i) => <li key={i} className="text-xs" style={{ color: 'var(--text-muted)' }}>• {n}</li>)}
          </ul>
        )}
      </Card>

      <div className="grid lg:grid-cols-2 gap-5">
        {/* Encryption / BYOK (#6) */}
        <Card>
          <div className="flex items-center gap-2 mb-1"><Icon.shield size={16} style={{ color: 'var(--text-muted)' }} /><h2 className="font-display font-semibold" style={{ fontSize: 16 }}>Encryption &amp; keys</h2></div>
          <p className="field-help mb-4">Each KB’s secrets are envelope-encrypted with its own data-encryption key. Rotate to re-key without downtime; an external KMS enables BYOK / crypto-shred.</p>
          {!crypto ? <Skeleton style={{ height: 90 }} /> : (
            <>
              <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm mb-3">
                <Row k="Envelope encryption">{crypto.envelope_encryption ? 'On' : 'Off'}</Row>
                <Row k="KMS backend">{crypto.kms_backend}</Row>
                <Row k="BYOK">{crypto.byok ? 'Yes' : 'Platform key'}</Row>
                <Row k="Key versions">{crypto.dek_versions}</Row>
              </dl>
              <Button size="sm" variant="secondary" loading={rotating} onClick={rotateKey} icon={<Icon.refresh size={15} />}>Rotate key</Button>
            </>
          )}
        </Card>

        {/* Retention */}
        <Card>
          <div className="flex items-center gap-2 mb-1"><Icon.scale size={16} style={{ color: 'var(--text-muted)' }} /><h2 className="font-display font-semibold" style={{ fontSize: 16 }}>{t('compliance.card.retention')}</h2></div>
          <p className="field-help mb-4">Documents older than this are purged when you apply the policy. Blank or 0 keeps them indefinitely. (Scheduled enforcement is coming with triggers.)</p>
          <Field label="Keep documents for (days)" help="0 or blank = keep forever.">
            <input className="field-input" type="number" min="0" value={days} onChange={(e) => setDays(e.target.value)} placeholder="e.g. 365" />
          </Field>
          <div className="flex gap-2 mt-4">
            <Button size="sm" loading={savingRet} onClick={saveRetention} icon={<Icon.check size={15} />}>Save policy</Button>
            <Button size="sm" variant="subtle" loading={applying} onClick={applyRetention} icon={<Icon.trash size={15} />}>Apply now</Button>
          </div>
          {retResult && (
            <div className="mt-4 text-sm surface-2 rounded-lg p-3">
              {retResult.documents_erased.length
                ? <span>Purged: {retResult.documents_erased.map((d) => <span key={d} className="chip mono mr-1" style={{ fontSize: 11 }}>{d}</span>)}</span>
                : <span style={{ color: 'var(--text-muted)' }}>No documents past the {retResult.retention_days}-day window.</span>}
            </div>
          )}
        </Card>

        {/* Right to erasure */}
        <Card>
          <div className="flex items-center gap-2 mb-1"><Icon.trash size={16} style={{ color: 'var(--danger)' }} /><h2 className="font-display font-semibold" style={{ fontSize: 16 }}>{t('compliance.card.erasure')}</h2></div>
          <p className="field-help mb-4">DPDP data-principal request. Erases matching documents from every store and scrubs the chat history that quoted them, then issues a certificate. Irreversible.</p>
          <Field label="Data principal / subject" help="Erases every document mentioning this (id, metadata, or content).">
            <input className="field-input" value={subject} onChange={(e) => setSubject(e.target.value)} placeholder="e.g. Anita Desai" />
          </Field>
          <Field label="…or specific document ids" help="Comma- or newline-separated.">
            <textarea className="field-input" rows={2} value={docIds} onChange={(e) => setDocIds(e.target.value)} placeholder="record-123, record-456" style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }} />
          </Field>
          <Field label="Reason (for the certificate)">
            <input className="field-input" value={reason} onChange={(e) => setReason(e.target.value)} />
          </Field>
          <Button className="mt-2" size="sm" variant="danger" loading={erasing} onClick={runErasure} icon={<Icon.trash size={15} />}>Erase &amp; certify</Button>

          {cert && (
            <div className="mt-4 rounded-lg p-3 text-sm" style={{ background: 'color-mix(in srgb, var(--success) 8%, transparent)', border: '1px solid color-mix(in srgb, var(--success) 30%, transparent)' }}>
              <div className="flex items-center gap-2 font-semibold mb-2" style={{ color: 'var(--success)' }}><Icon.check size={15} />Erasure certificate</div>
              <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
                <dt style={{ color: 'var(--text-muted)' }}>Documents</dt><dd>{cert.documents_erased.length ? cert.documents_erased.join(', ') : '—'}</dd>
                <dt style={{ color: 'var(--text-muted)' }}>Chunks purged</dt><dd className="mono">{cert.chunks_purged}</dd>
                <dt style={{ color: 'var(--text-muted)' }}>Chats scrubbed</dt><dd className="mono">{cert.conversations_purged}</dd>
                <dt style={{ color: 'var(--text-muted)' }}>Certificate</dt><dd className="mono" style={{ fontSize: 11, wordBreak: 'break-all' }}>{cert.certificate_hash}</dd>
              </dl>
              <Button className="mt-3" variant="ghost" size="sm" onClick={() => downloadJson(`erasure-${cert.certificate_hash.slice(0, 12)}.json`, cert)} icon={<Icon.doc size={14} />}>Export certificate</Button>
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}

function Row({ k, children }: { k: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 py-0.5" style={{ borderBottom: '1px solid var(--border)' }}>
      <dt style={{ color: 'var(--text-muted)' }}>{k}</dt>
      <dd className="text-right" style={{ color: 'var(--text-primary)' }}>{children}</dd>
    </div>
  )
}

function statusColor(status: string): string {
  if (status === 'satisfied') return 'var(--success)'
  if (status === 'partial') return 'var(--warning)'
  if (status === 'manual') return 'var(--info)'
  return 'var(--text-muted)'
}

function downloadJson(name: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = name; a.click()
  URL.revokeObjectURL(url)
}
