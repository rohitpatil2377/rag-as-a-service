// Platform → Air-gap Preflight — would this deployment survive fully offline? Reads
// GET /v1/platform/airgap/preflight and classifies each violation. Sarvam (self-hostable) and
// AIOps (internal Acme network) are treated as ACCEPTED EXCEPTIONS, not blockers — so "ready"
// reflects the real, fixable blockers (foreign hosted services, missing update key).
import { useCallback, useEffect, useState } from 'react'
import { Card, PageHeader, Skeleton } from '../components/ui'
import { api, ApiError, type AirgapReport, type AirgapViolation } from '../lib/api'
import { useAuth } from '../lib/auth'

// An accepted exception returns WHY it's acceptable; a real blocker returns null.
// All of these are self-hostable / internal — in a true air-gap install they're pointed at
// on-box or intranet equivalents. Only genuinely un-hostable config (e.g. a missing local
// update-verification key) stays a blocker.
function exemptReason(v: AirgapViolation): string | null {
  const val = (v.value || '').toLowerCase()
  if (val.includes('sarvam')) return 'Sarvam — self-hostable; run the model on-box / intranet vLLM'
  if (val.includes('aiops')) return 'AIOps — internal Acme network (sovereign), not foreign egress'
  if (val.includes('jina')) return 'Reranker — self-hostable; run bge-reranker on-box (TEI)'
  if (val.includes('datalab')) return 'OCR / vision (Chandra) — self-hostable; run Chandra on-box (vLLM)'
  return null
}

export function Airgap() {
  const { session } = useAuth()
  const [rep, setRep] = useState<AirgapReport | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    if (!session) return
    setError(null)
    try { setRep(await api.airgapPreflight(session.token)) }
    catch (e) { setError(e instanceof ApiError ? e.message : 'Could not run the air-gap preflight.') }
  }, [session])
  useEffect(() => { void load() }, [load])

  const blocking = (rep?.violations ?? []).filter((v) => !exemptReason(v))
  const exempt = (rep?.violations ?? []).filter((v) => exemptReason(v))
  const ready = rep != null && blocking.length === 0

  return (
    <div className="space-y-4">
      <PageHeader eyebrow="Platform" title="Air-gap Preflight"
        lead="Would this deployment pass a fully-offline (air-gapped) install? Sarvam and AIOps are treated as accepted exceptions (self-hostable / internal); everything else must resolve on-box." />

      {error && <Card><p className="field-help" style={{ padding: 12, color: 'var(--danger, #E8556B)' }}>{error}</p></Card>}
      {rep === null && !error && <Card><Skeleton style={{ height: 160 }} /></Card>}

      {rep && (
        <>
          {/* Verdict banner */}
          <Card>
            <div style={{ display: 'flex', alignItems: 'center', gap: 14, padding: 4 }}>
              <span style={{ fontSize: 26 }}>{ready ? '✅' : '⛔'}</span>
              <div>
                <div style={{ fontFamily: 'var(--font-heading)', fontSize: 17, fontWeight: 700, color: ready ? '#1a9a76' : 'var(--text-heading)' }}>
                  {ready ? 'Air-gap ready' : `Not air-gap ready — ${blocking.length} blocker${blocking.length === 1 ? '' : 's'}`}
                </div>
                <div style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>
                  Enforcement is currently <strong style={{ color: rep.air_gap_enabled ? '#1a9a76' : '#a97a1e' }}>{rep.air_gap_enabled ? 'ON' : 'OFF'}</strong>
                  {' · '}{blocking.length} blocking · {exempt.length} accepted exception{exempt.length === 1 ? '' : 's'}
                  {ready && !rep.air_gap_enabled && ' · once the exceptions point at their self-hosted/internal endpoints, this node can go offline.'}
                </div>
              </div>
            </div>
          </Card>

          {/* Blockers */}
          <Card>
            <div style={{ fontFamily: 'var(--font-heading)', fontSize: 14, fontWeight: 600, color: 'var(--text-heading)', marginBottom: 8 }}>
              Blockers {blocking.length > 0 && <span className="mono" style={{ fontSize: 12, color: 'var(--danger, #E8556B)' }}>({blocking.length})</span>}
            </div>
            {blocking.length === 0 ? (
              <p className="field-help">No blocking violations — every remote dependency is either on-box or an accepted exception. 🎉</p>
            ) : (
              blocking.map((v, i) => (
                <div key={i} style={{ padding: '10px 0', borderTop: '1px solid var(--border)' }}>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'baseline' }}>
                    <span className="mono" style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text-heading)' }}>{v.setting}</span>
                    {v.value && <span className="mono" style={{ fontSize: 11.5, color: 'var(--danger, #E8556B)' }}>{v.value}</span>}
                  </div>
                  <div style={{ fontSize: 12.5, color: 'var(--text-muted)', marginTop: 2 }}>{v.reason}</div>
                </div>
              ))
            )}
          </Card>

          {/* Accepted exceptions */}
          {exempt.length > 0 && (
            <Card>
              <div style={{ fontFamily: 'var(--font-heading)', fontSize: 14, fontWeight: 600, color: 'var(--text-heading)', marginBottom: 4 }}>
                Accepted exceptions <span className="mono" style={{ fontSize: 12, color: 'var(--text-muted)' }}>({exempt.length})</span>
              </div>
              <p className="field-help" style={{ marginBottom: 8 }}>Not counted as blockers — self-hostable or on the internal network. In a real air-gap install, point these at their on-box / intranet endpoints.</p>
              {exempt.map((v, i) => (
                <div key={i} style={{ padding: '9px 0', borderTop: '1px solid var(--border)', opacity: 0.85 }}>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'baseline' }}>
                    <span className="mono" style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text-heading)' }}>{v.setting}</span>
                    <span className="mono" style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>{v.value}</span>
                    <span style={{ marginLeft: 'auto', fontSize: 10.5, fontWeight: 600, padding: '1px 7px', borderRadius: 999, background: 'color-mix(in srgb, #3FD1A8 16%, transparent)', color: '#1a9a76' }}>exempt</span>
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>{exemptReason(v)}</div>
                </div>
              ))}
            </Card>
          )}
        </>
      )}
    </div>
  )
}
