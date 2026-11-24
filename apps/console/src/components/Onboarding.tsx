import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Icon } from './icons'
import { Button, Card } from './ui'
import { useToast } from './overlay'
import { api } from '../lib/api'
import { useAuth } from '../lib/auth'
import { useKb } from '../lib/kb'

const DISMISS_KEY = 'rag.onboarding.dismissed'

// A tiny, clearly-fictional demo corpus so a new tenant can reach a grounded answer in one
// click. Deliberately domain-neutral (a made-up product) — never real/sensitive content.
const SAMPLE_DOCS: { doc_id: string; title: string; text: string }[] = [
  { doc_id: 'sample-service-levels', title: 'Acme Cloud — Service Levels',
    text: 'Acme Cloud guarantees 99.9% monthly uptime. Support responds to Critical incidents within 4 hours and to Standard requests within 2 business days. Scheduled maintenance is announced at least 72 hours in advance.' },
  { doc_id: 'sample-security', title: 'Acme Cloud — Security',
    text: 'All customer data on Acme Cloud is encrypted in transit and at rest using AES-256. Access requires multi-factor authentication. Acme Cloud is audited annually against SOC 2 Type II. Data is stored only in the customer-selected region.' },
  { doc_id: 'sample-support', title: 'Acme Cloud — Support & Escalation',
    text: 'Acme Cloud support operates 24x7 for Critical issues. Customers raise tickets through the portal. Unresolved Critical tickets are escalated to a duty manager after 4 hours and to engineering leadership after 8 hours.' },
]

function b64(s: string): string {
  // btoa needs latin1; sample text is ASCII so this is safe.
  return btoa(unescape(encodeURIComponent(s)))
}

interface Step {
  done: boolean
  title: string
  body: string
  action: React.ReactNode
}

export function Onboarding() {
  const { session } = useAuth()
  const { kbs, kbId, create, setKbId, refresh } = useKb()
  const toast = useToast()
  const [docs, setDocs] = useState<number | null>(null)
  const [asked, setAsked] = useState<boolean | null>(null)
  const [busy, setBusy] = useState(false)
  const [dismissed, setDismissed] = useState(() => localStorage.getItem(DISMISS_KEY) === '1')

  const probe = useCallback(async () => {
    if (!session || !kbId || !kbs.length) { setDocs(0); setAsked(false); return }
    const [d, a] = await Promise.allSettled([
      api.listDocuments(session.token, kbId),
      api.listAudit(session.token, kbId),
    ])
    setDocs(d.status === 'fulfilled' ? d.value.length : 0)
    setAsked(a.status === 'fulfilled' ? a.value.some((e) => e.action === 'query') : false)
  }, [session, kbId, kbs.length])

  useEffect(() => { void probe() }, [probe])

  async function loadSample() {
    if (!session || busy) return
    setBusy(true)
    try {
      let target = kbId
      if (!kbs.length) {
        const kb = await create('Getting started')
        target = kb.kb_id
        setKbId(target)
      }
      for (const d of SAMPLE_DOCS) {
        await api.ingest(session.token, target, { doc_id: d.doc_id, mime: 'text/plain', content_b64: b64(d.text) })
      }
      await refresh()
      await probe()
      toast.push('success', 'Sample corpus loaded — now ask it a question, e.g. “What is the uptime guarantee?”')
    } catch {
      toast.push('error', 'Could not load the sample corpus.')
    } finally {
      setBusy(false)
    }
  }

  if (dismissed || docs === null || asked === null) return null

  const hasKb = kbs.length > 0
  const steps: Step[] = [
    {
      done: hasKb,
      title: 'Create a knowledge base',
      body: 'A knowledge base holds one collection of documents you can ask questions about.',
      action: <Link to="/knowledge-base"><Button variant="primary" style={{ fontSize: 13 }}>Create a KB</Button></Link>,
    },
    {
      done: docs > 0,
      title: 'Add documents',
      body: 'Upload your own files, or load a small sample corpus to try it instantly.',
      action: (
        <div className="flex gap-2">
          <Link to="/knowledge-base"><Button variant={hasKb ? 'secondary' : 'ghost'} disabled={!hasKb} icon={<Icon.upload size={15} />} style={{ fontSize: 13 }}>Upload</Button></Link>
          <Button variant="ghost" loading={busy} onClick={loadSample} icon={!busy && <Icon.plus size={15} />} style={{ fontSize: 13 }}>Load sample corpus</Button>
        </div>
      ),
    },
    {
      done: !!asked,
      title: 'Ask your first question',
      body: 'Get an answer grounded in your documents — with citations, or an honest abstain.',
      action: <Link to="/ask"><Button variant="secondary" disabled={docs === 0} icon={<Icon.ask size={15} />} style={{ fontSize: 13 }}>Go to Ask</Button></Link>,
    },
  ]

  const doneCount = steps.filter((s) => s.done).length
  if (doneCount === steps.length) return null // fully set up — don't nag

  return (
    <Card className="mb-8">
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2"><span className="w-6 rule-gold" /><span className="eyebrow">Get started</span></div>
        <button onClick={() => { localStorage.setItem(DISMISS_KEY, '1'); setDismissed(true) }}
          className="text-xs flex items-center gap-1" style={{ color: 'var(--text-muted)' }} aria-label="Dismiss">
          <Icon.x size={13} /> Dismiss
        </button>
      </div>
      <h2 className="font-display font-semibold mb-1" style={{ fontSize: 18 }}>Three steps to your first grounded answer</h2>
      <p className="field-help mb-4">{doneCount} of {steps.length} done</p>
      <ol className="space-y-3">
        {steps.map((s, i) => (
          <li key={i} className="flex items-start gap-3">
            <span className="flex-shrink-0 flex items-center justify-center rounded-full" style={{
              width: 24, height: 24, marginTop: 1,
              background: s.done ? 'color-mix(in srgb, var(--success) 16%, transparent)' : 'color-mix(in srgb, var(--text-muted) 12%, transparent)',
              color: s.done ? 'var(--success)' : 'var(--text-muted)',
            }}>
              {s.done ? <Icon.check size={14} /> : <span className="mono" style={{ fontSize: 12 }}>{i + 1}</span>}
            </span>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <span className="font-semibold" style={{ fontSize: 14, color: 'var(--text-heading)', textDecoration: s.done ? 'line-through' : 'none', opacity: s.done ? 0.6 : 1 }}>{s.title}</span>
                {!s.done && s.action}
              </div>
              {!s.done && <p className="text-sm mt-0.5" style={{ color: 'var(--text-muted)' }}>{s.body}</p>}
            </div>
          </li>
        ))}
      </ol>
    </Card>
  )
}
