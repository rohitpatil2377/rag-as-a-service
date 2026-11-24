// Platform → Residency — the signed, per-node data-residency attestation an operator hands a
// regulator. Read-only (GET /v1/platform/residency/node); offers the signed token as a download.
import { useCallback, useEffect, useState } from 'react'
import { useToast } from '../components/overlay'
import { Button, Card, PageHeader, Skeleton } from '../components/ui'
import { api, ApiError, type ResidencyAttestation } from '../lib/api'
import { useAuth } from '../lib/auth'

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', gap: 16, padding: '10px 0', borderTop: '1px solid var(--border)' }}>
      <div style={{ width: 190, flexShrink: 0, fontSize: 12.5, color: 'var(--text-muted)' }}>{k}</div>
      <div style={{ fontSize: 13, color: 'var(--text-heading)', wordBreak: 'break-word' }}>{v}</div>
    </div>
  )
}

export function Residency() {
  const { session } = useAuth()
  const toast = useToast()
  const [att, setAtt] = useState<ResidencyAttestation | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    if (!session) return
    setError(null)
    try { setAtt(await api.residency(session.token)) }
    catch (e) { setError(e instanceof ApiError ? e.message : 'Could not load the residency attestation.') }
  }, [session])
  useEffect(() => { void load() }, [load])

  function download() {
    if (!att) return
    const blob = new Blob([JSON.stringify(att, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = `residency-${att.node_id || 'node'}-${att.attestation_id || 'attestation'}.json`
    a.click(); URL.revokeObjectURL(url)
    toast.push('success', 'Signed attestation downloaded.')
  }

  return (
    <div className="space-y-4">
      <PageHeader eyebrow="Platform" title="Data Residency"
        lead="A cryptographically signed, per-node attestation that this node's data stays in-region — hand it to a regulator; they can verify it offline with the signer's public key." />

      {error && <Card><p className="field-help" style={{ padding: 12, color: 'var(--danger, #E8556B)' }}>{error}</p></Card>}

      {att === null && !error && <Card><Skeleton style={{ height: 200 }} /></Card>}

      {att && (
        <>
          <Card>
            {/* Headline residency status */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '4px 4px 12px' }}>
              <span style={{ fontSize: 22 }}>{att.no_foreign_egress ? '🛡️' : '⚠️'}</span>
              <div>
                <div style={{ fontFamily: 'var(--font-heading)', fontSize: 16, fontWeight: 700, color: 'var(--text-heading)' }}>
                  {att.region || att.data_residency || 'Region unset'}
                </div>
                <div style={{ fontSize: 12.5, color: att.no_foreign_egress ? '#1a9a76' : '#a97a1e', fontWeight: 600 }}>
                  {att.no_foreign_egress ? 'No foreign egress — data stays in region' : 'Foreign egress possible — review configuration'}
                </div>
              </div>
              <Button className="ml-auto" onClick={download}>Download signed attestation</Button>
            </div>

            <Row k="Node ID" v={<span className="mono">{att.node_id || '—'}</span>} />
            <Row k="Region" v={att.region || '—'} />
            <Row k="Data residency" v={att.data_residency || '—'} />
            <Row k="No foreign egress" v={att.no_foreign_egress ? 'Yes' : 'No'} />
            <Row k="Resident KBs" v={att.kb_ids?.length ? <span className="mono" style={{ fontSize: 12 }}>{att.kb_ids.join(', ')}</span> : <span style={{ color: 'var(--text-muted)' }}>none</span>} />
            <Row k="Generated at" v={att.generated_at ? new Date(att.generated_at).toLocaleString() : '—'} />
          </Card>

          <Card>
            <div style={{ fontFamily: 'var(--font-heading)', fontSize: 14, fontWeight: 600, color: 'var(--text-heading)', marginBottom: 4 }}>Signature</div>
            <p className="field-help" style={{ marginBottom: 10 }}>Ed25519 detached signature over the attestation claims — proves it wasn't forged or altered in transit.</p>
            <Row k="Signing key id" v={<span className="mono" style={{ fontSize: 12 }}>{att.signer_kid || '—'}</span>} />
            <Row k="Signature" v={att.signed_token
              ? <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)', wordBreak: 'break-all' }}>{att.signed_token}</span>
              : <span style={{ color: '#a97a1e' }}>unsigned (no signing key configured on this node)</span>} />
          </Card>
        </>
      )}
    </div>
  )
}
