import { useCallback, useEffect, useState } from 'react'
import { useI18n } from '../lib/i18n'
import { Icon } from '../components/icons'
import { useToast } from '../components/overlay'
import { Alert, Button, Card, EmptyState, HitlPill, PageHeader, SegmentedControl, Skeleton } from '../components/ui'
import { Markdown } from '../components/Markdown'
import { ApiError, api, type HitlAction, type HitlReview, type HitlStatus } from '../lib/api'
import { hasRole, useAuth } from '../lib/auth'
import { useKb } from '../lib/kb'
import { useRoleLabels } from '../lib/roleLabels'

const ACTIONS: { action: HitlAction; label: string; variant: 'primary' | 'ghost' | 'secondary' }[] = [
  { action: 'accept', label: 'Accept', variant: 'primary' },
  { action: 'flag', label: 'Flag', variant: 'ghost' },
  { action: 'override', label: 'Override', variant: 'secondary' },
]

export function Reviews() {
  const { session } = useAuth()
  const { t } = useI18n()
  const { kbId } = useKb()
  const toast = useToast()
  const roleLabels = useRoleLabels()
  const [filter, setFilter] = useState<'pending' | 'all'>('pending')
  const [reviews, setReviews] = useState<HitlReview[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [acting, setActing] = useState<string | null>(null)
  // Backend gates HITL act() on `reviewer` — same check here so a reader sees why the
  // buttons are disabled instead of clicking through to a 403.
  const canAct = hasRole(session, 'reviewer', 'admin')

  const load = useCallback(async () => {
    if (!session) return
    setLoading(true); setError(null)
    try {
      const status = filter === 'pending' ? ('pending' as HitlStatus) : undefined
      setReviews(await api.listReviews(session.token, kbId, status))
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Failed to load reviews')
    } finally {
      setLoading(false)
    }
  }, [session, kbId, filter])

  useEffect(() => { void load() }, [load])

  async function act(r: HitlReview, action: HitlAction) {
    if (!session) return
    setActing(r.review_id)
    try {
      await api.actReview(session.token, kbId, r.review_id, action, session.subject)
      toast.push('success', `Review ${action}ed`)
      await load()
    } catch (err) {
      toast.push('error', err instanceof ApiError ? err.message : 'Action failed')
    } finally {
      setActing(null)
    }
  }

  return (
    <div>
      <PageHeader eyebrow={t('eyebrow.govern')} title={t('page.reviews.title')}
        lead={t('page.reviews.lead')}
        actions={<SegmentedControl value={filter} onChange={setFilter} options={[{ value: 'pending', label: 'Pending' }, { value: 'all', label: 'All' }]} />} />

      {loading && <div className="space-y-4">{[0, 1].map((i) => <Skeleton key={i} style={{ height: 140 }} />)}</div>}
      {error && <Alert kind="error">{error}</Alert>}
      {!loading && reviews.length === 0 && !error && (
        <EmptyState icon={<Icon.review size={28} />} title="Nothing to review" hint="Flagged answers from Ask will appear here for a human decision." />
      )}

      <div className="space-y-4">
        {!loading && reviews.map((r) => (
          <Card key={r.review_id}>
            <div className="flex items-center justify-between mb-3">
              <HitlPill status={r.status} />
              <span className="mono text-xs" style={{ color: 'var(--text-muted)' }}>{new Date(r.updated_at).toLocaleString()}</span>
            </div>
            <div className="mono text-xs mb-1" style={{ color: 'var(--text-muted)' }}>QUESTION</div>
            <p className="font-medium mb-3" style={{ color: 'var(--text-heading)' }}>{r.query}</p>
            <div className="mono text-xs mb-1" style={{ color: 'var(--text-muted)' }}>ANSWER</div>
            <div className="text-sm mb-3" style={{ color: 'var(--text-primary)' }}><Markdown text={r.answer} /></div>
            {r.warnings.length > 0 && <Alert kind="warning"><strong>Unsupported claims:</strong> {r.warnings.join(' · ')}</Alert>}
            {r.status === 'pending' ? (
              <div className="flex gap-2 mt-4">
                {ACTIONS.map((a) => (
                  <Button key={a.action} size="sm" variant={a.variant} loading={acting === r.review_id}
                    disabled={!canAct} title={canAct ? undefined : `Requires ${roleLabels.reviewer} access`}
                    onClick={() => act(r, a.action)}>{a.label}</Button>
                ))}
              </div>
            ) : r.reviewer && (
              <div className="mt-3 mono text-xs" style={{ color: 'var(--text-muted)' }}>reviewed by {r.reviewer}</div>
            )}
          </Card>
        ))}
      </div>
    </div>
  )
}
