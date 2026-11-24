import { useEffect, useState } from 'react'
import { useI18n } from '../lib/i18n'
import { Icon } from '../components/icons'
import { useToast } from '../components/overlay'
import { Alert, Button, Card, Field, PageHeader, Skeleton } from '../components/ui'
import { ApiError, api, type CustomRole, type FeatureFlag, type SystemInfo } from '../lib/api'
import { hasRole, useAuth } from '../lib/auth'
import { ROLE_LABEL_DEFAULTS, invalidateRoleLabelsCache } from '../lib/roleLabels'
import { invalidateCustomRolesCache } from '../lib/customRoles'

const CAPS = ['reader', 'ingestor', 'reviewer', 'admin'] as const
const slug = (s: string) => s.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'role'

// The 4 capability roles an org can rename (matches the backend's CAP_ROLES — owner is a fixed
// structural concept, not a capability tier, so it's never customizable here).
const CUSTOMIZABLE_ROLES: { value: string; hint: string }[] = [
  { value: 'reader', hint: 'Ask & read answers' },
  { value: 'ingestor', hint: 'Add documents & connectors' },
  { value: 'reviewer', hint: 'Review queue & quality' },
  { value: 'admin', hint: 'Manage everything' },
]

export function Settings() {
  const { session } = useAuth()
  const { t } = useI18n()
  const toast = useToast()
  const [info, setInfo] = useState<SystemInfo | null>(null)
  const [flags, setFlags] = useState<FeatureFlag[] | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const canEdit = session?.role === 'owner' || session?.role === 'admin'
  // Role labels (org-wide, cosmetic-only renaming of the fixed capability roles). Gated the same
  // way member management is gated on Team — owner/admin, matching the backend's PUT check.
  const canEditRoleLabels = session?.role === 'owner' || session?.role === 'admin' || hasRole(session, 'admin')
  const [roleLabels, setRoleLabels] = useState<Record<string, string> | null>(null)
  const [roleLabelsBusy, setRoleLabelsBusy] = useState(false)
  // Custom named roles (label + capability bundle), org-wide.
  const [customRoles, setCustomRoles] = useState<CustomRole[] | null>(null)
  const [customRolesBusy, setCustomRolesBusy] = useState(false)

  useEffect(() => {
    if (!session) return
    api.systemInfo(session.token).then(setInfo).catch((e) => setError(e instanceof ApiError ? e.message : 'Failed to load'))
    api.getFeatures(session.token).then((v) => setFlags(v.flags)).catch(() => setFlags([]))
    api.getRoleLabels(session.token).then((v) => setRoleLabels(v.labels)).catch(() => setRoleLabels({ ...ROLE_LABEL_DEFAULTS }))
    api.getCustomRoles(session.token).then((v) => setCustomRoles(v.roles)).catch(() => setCustomRoles([]))
  }, [session])

  function addCustomRole() {
    setCustomRoles((cur) => [...(cur ?? []), { id: `role-${(cur?.length ?? 0) + 1}`, label: '', caps: ['reader'] }])
  }
  function updateCustomRole(i: number, patch: Partial<CustomRole>) {
    setCustomRoles((cur) => (cur ?? []).map((r, idx) => (idx === i ? { ...r, ...patch } : r)))
  }
  function toggleCap(i: number, cap: string) {
    setCustomRoles((cur) => (cur ?? []).map((r, idx) => idx === i ? { ...r, caps: r.caps.includes(cap) ? r.caps.filter((c) => c !== cap) : [...r.caps, cap] } : r))
  }
  function removeCustomRole(i: number) {
    setCustomRoles((cur) => (cur ?? []).filter((_, idx) => idx !== i))
  }
  async function saveCustomRoles() {
    if (!session || !customRoles) return
    if (customRoles.some((r) => !r.label.trim())) { toast.push('error', 'Every custom role needs a name.'); return }
    if (customRoles.some((r) => r.caps.length === 0)) { toast.push('error', 'Every custom role needs at least one capability.'); return }
    setCustomRolesBusy(true)
    try {
      const payload = customRoles.map((r) => ({ id: slug(r.id || r.label), label: r.label.trim(), caps: r.caps }))
      const v = await api.setCustomRoles(session.token, payload)
      setCustomRoles(v.roles)
      invalidateCustomRolesCache(session.tenantId, v.roles)
      toast.push('success', 'Custom roles saved.')
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not save custom roles.')
    } finally {
      setCustomRolesBusy(false)
    }
  }

  function setRoleLabel(role: string, value: string) {
    setRoleLabels((cur) => ({ ...(cur ?? ROLE_LABEL_DEFAULTS), [role]: value }))
  }

  async function saveRoleLabels() {
    if (!session || !roleLabels) return
    setRoleLabelsBusy(true)
    try {
      const body = Object.fromEntries(CUSTOMIZABLE_ROLES.map((r) => [r.value, (roleLabels[r.value] ?? '').trim()]))
      const v = await api.setRoleLabels(session.token, body)
      setRoleLabels(v.labels)
      invalidateRoleLabelsCache(session.tenantId, v.labels)
      toast.push('success', 'Role labels saved.')
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not save role labels.')
    } finally {
      setRoleLabelsBusy(false)
    }
  }

  async function toggle(f: FeatureFlag) {
    if (!session || f.locked || busy) return
    setBusy(f.name)
    try {
      const v = await api.setFeature(session.token, f.name, !f.enabled)
      setFlags(v.flags)
      toast.push('success', `${f.label} ${!f.enabled ? 'enabled' : 'disabled'}.`)
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Could not change this setting.')
    } finally {
      setBusy(null)
    }
  }

  return (
    <div>
      <PageHeader eyebrow={t('eyebrow.platform')} title={t('page.settings.title')}
        lead={t('page.settings.lead')} />

      {error && <Alert kind="error">{error}</Alert>}
      {!info && !error && <div className="grid lg:grid-cols-2 gap-5">{[0, 1, 2, 3].map((i) => <Skeleton key={i} style={{ height: 160 }} />)}</div>}

      {info && (
        <>
        <div className="grid lg:grid-cols-2 gap-5">
          <Card>
            <SectionTitle icon={<Icon.shield size={16} />} title="Sovereignty & environment" />
            <Row k="Egress guard" v={info.sovereign_enforce ? 'Enforced' : 'Disabled'} tone={info.sovereign_enforce ? '--success' : '--danger'} />
            <Row k="Environment" v={info.env} />
          </Card>

          <Card>
            <SectionTitle icon={<Icon.layers size={16} />} title="Models & stores" />
            <Row k="LLM backend" v={info.llm_backend} mono />
            <Row k="LLM model" v={info.llm_model} mono />
            <Row k="Embeddings" v={info.embed_model} mono />
            <Row k="Vector DB" v={info.vector_backend} mono />
            <Row k="Sparse index" v={info.sparse_backend} mono />
          </Card>

          <Card>
            <SectionTitle icon={<Icon.layers size={16} />} title="Capabilities" />
            {flags === null ? <Skeleton style={{ height: 120 }} /> : (
              <div className="space-y-1.5">
                {[...flags].sort((a, b) => Number(a.locked) - Number(b.locked)).map((f) => (
                  <div key={f.name} className="flex items-center justify-between gap-3 py-1">
                    <span className="flex items-center gap-2 min-w-0" style={{ fontSize: 13.5 }}>
                      {f.locked && <Icon.shield size={13} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />}
                      <span className="truncate" style={{ color: 'var(--text-primary)' }}>{f.label}</span>
                    </span>
                    <Toggle on={f.enabled} locked={f.locked || !canEdit} busy={busy === f.name} onClick={() => toggle(f)} />
                  </div>
                ))}
              </div>
            )}
            <p className="field-help mt-4">
              Toggle behavioural features on/off live — no restart. <span style={{ color: 'var(--text-muted)' }}>
              <Icon.shield size={11} className="inline" /> Security &amp; infrastructure flags are operator-managed
              (environment only) and shown read-only.</span>{!canEdit && ' Sign in as an owner/admin to change these.'}
            </p>
          </Card>

          <Card>
            <SectionTitle icon={<Icon.scale size={16} />} title="Grounding thresholds" />
            <Row k="Abstain below" v={info.thresholds.abstain?.toFixed(2)} mono />
            <Row k="Faithful ≥" v={info.thresholds.faithful?.toFixed(2)} mono />
            <Row k="Flag-for-review ≥" v={info.thresholds.flag?.toFixed(2)} mono />
            <Row k="Faithfulness backend" v={info.faithfulness_backend} mono />
          </Card>

          <Card>
            <SectionTitle icon={<Icon.shield size={16} />} title="Role labels" />
            {roleLabels === null ? <Skeleton style={{ height: 120 }} /> : canEditRoleLabels ? (
              <>
                <p className="field-help mb-3">Rename how the fixed capability roles appear across the console (e.g. "Contributor" → "Editor"). Permissions themselves don't change — this is display-only, per organisation.</p>
                <div className="space-y-3">
                  {CUSTOMIZABLE_ROLES.map((r) => (
                    <Field key={r.value} label={`${r.value[0].toUpperCase()}${r.value.slice(1)} role label`} help={r.hint}>
                      <input
                        className="field-input"
                        maxLength={40}
                        value={roleLabels[r.value] ?? ''}
                        onChange={(e) => setRoleLabel(r.value, e.target.value)}
                      />
                    </Field>
                  ))}
                </div>
                <Button className="mt-4" loading={roleLabelsBusy}
                  disabled={CUSTOMIZABLE_ROLES.some((r) => !(roleLabels[r.value] ?? '').trim())}
                  onClick={saveRoleLabels}>Save labels</Button>
              </>
            ) : (
              <>
                <p className="field-help mb-3">How the fixed capability roles appear across the console.</p>
                {CUSTOMIZABLE_ROLES.map((r) => (
                  <Row key={r.value} k={`${r.value[0].toUpperCase()}${r.value.slice(1)}`} v={roleLabels[r.value] ?? ROLE_LABEL_DEFAULTS[r.value]} />
                ))}
                <p className="field-help mt-3">Sign in as an owner/admin to rename these.</p>
              </>
            )}
          </Card>

          <Card>
            <SectionTitle icon={<Icon.shield size={16} />} title="Custom roles" />
            {customRoles === null ? <Skeleton style={{ height: 120 }} /> : canEditRoleLabels ? (
              <>
                <p className="field-help mb-3">Create named roles as a bundle of the 4 capabilities. Assign them to members on the Team page — a member's role shows as this name when their capabilities match. Permissions are still just the underlying capabilities; this is a convenient preset.</p>
                <div className="space-y-3">
                  {customRoles.length === 0 && <p className="field-help">No custom roles yet. Add one below (e.g. "Analyst" = Ask + Review).</p>}
                  {customRoles.map((r, i) => (
                    <div key={i} style={{ border: '1px solid var(--border)', borderRadius: 8, padding: 12 }}>
                      <div className="flex items-center gap-2 mb-2">
                        <input className="field-input" placeholder="Role name (e.g. Analyst)" maxLength={40} value={r.label}
                          onChange={(e) => updateCustomRole(i, { label: e.target.value })} style={{ flex: 1 }} />
                        <button type="button" onClick={() => removeCustomRole(i)} title="Remove role" className="btn btn-subtle" style={{ fontSize: 12, color: 'var(--danger, #E8556B)', padding: '4px 10px' }}>Remove</button>
                      </div>
                      <div className="flex flex-wrap gap-3">
                        {CAPS.map((c) => (
                          <label key={c} className="flex items-center gap-1.5 text-sm" style={{ cursor: 'pointer' }}>
                            <input type="checkbox" checked={r.caps.includes(c)} onChange={() => toggleCap(i, c)} />
                            <span style={{ color: 'var(--text-heading)' }}>{roleLabels?.[c] ?? c}</span>
                          </label>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
                <div className="flex items-center gap-2 mt-4">
                  <Button variant="secondary" onClick={addCustomRole}>+ Add role</Button>
                  <Button loading={customRolesBusy} onClick={saveCustomRoles}>Save custom roles</Button>
                </div>
              </>
            ) : (
              <>
                <p className="field-help mb-3">Named capability presets defined by your organisation's admins.</p>
                {customRoles.length === 0
                  ? <p className="field-help">No custom roles defined.</p>
                  : customRoles.map((r) => <Row key={r.id} k={r.label} v={r.caps.map((c) => roleLabels?.[c] ?? c).join(' · ')} />)}
              </>
            )}
          </Card>
        </div>

        {/* "Models in use" lives on the Models page (Platform → Models) — not duplicated here. */}
        </>
      )}
    </div>
  )
}

function SectionTitle({ icon, title }: { icon: React.ReactNode; title: string }) {
  return <div className="flex items-center gap-2 mb-4"><span style={{ color: 'var(--action)' }}>{icon}</span><h2 className="font-display font-semibold" style={{ fontSize: 16 }}>{title}</h2></div>
}

function Row({ k, v, mono, tone }: { k: string; v: string; mono?: boolean; tone?: string }) {
  return (
    <div className="flex items-center justify-between text-sm py-1.5" style={{ borderBottom: '1px solid var(--border)' }}>
      <span style={{ color: 'var(--text-muted)' }}>{k}</span>
      <span className={mono ? 'mono' : 'font-semibold'} style={{ color: tone ? `var(${tone})` : 'var(--text-heading)' }}>{v}</span>
    </div>
  )
}

function Toggle({ on, locked, busy, onClick }: { on: boolean; locked?: boolean; busy?: boolean; onClick: () => void }) {
  if (locked) {
    return <span className="pill" style={{ fontSize: 11, color: on ? 'var(--success)' : 'var(--text-muted)', borderColor: 'var(--border)', background: 'transparent' }}>{on ? 'On' : 'Off'}</span>
  }
  return (
    <button type="button" onClick={onClick} disabled={busy} aria-pressed={on} aria-label={on ? 'Enabled' : 'Disabled'}
      style={{
        width: 38, height: 22, borderRadius: 999, border: 'none', cursor: busy ? 'wait' : 'pointer', flexShrink: 0,
        padding: 2, transition: 'background .15s', opacity: busy ? 0.6 : 1,
        background: on ? 'var(--action)' : 'color-mix(in srgb, var(--text-muted) 35%, transparent)',
      }}>
      <span style={{
        display: 'block', width: 18, height: 18, borderRadius: '50%', background: '#fff',
        transform: on ? 'translateX(16px)' : 'translateX(0)', transition: 'transform .15s',
        boxShadow: '0 1px 2px rgba(0,0,0,.2)',
      }} />
    </button>
  )
}
