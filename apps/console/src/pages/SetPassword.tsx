// Public page reached from an activation / password-reset link (?token=…). The visitor sets a
// password, which activates the account, then signs in. No auth required — the token is the credential.
import { useState } from 'react'
import { BRAND_NAME, TAGLINE } from '../lib/brand'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { Icon } from '../components/icons'
import { Alert, Button, Field, PasswordInput, ThemeToggle } from '../components/ui'
import { ApiError, api } from '../lib/api'

export function SetPassword() {
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const token = params.get('token') ?? ''
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [done, setDone] = useState(false)

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    if (password.length < 8) return setError('Password must be at least 8 characters.')
    if (password !== confirm) return setError('The passwords do not match.')
    setBusy(true)
    try {
      await api.setPassword(token, password)
      setDone(true)
      setTimeout(() => navigate('/login'), 1800)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not set your password. The link may have expired.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-6" style={{ background: 'var(--canvas)' }}>
      <div className="absolute top-4 right-4"><ThemeToggle /></div>
      <div className="surface w-full" style={{ maxWidth: 420, padding: 28 }}>
        <div className="eyebrow" style={{ letterSpacing: '.22em' }}>{BRAND_NAME}</div>
        {/* NOT `uppercase` — text-transform would render "RAGaaS" as "RAGAAS". */}
        <div className="mono mb-3" style={{ fontSize: 9.5, letterSpacing: '.16em', color: 'var(--text-muted)' }}>{TAGLINE}</div>
        <h1 className="font-display font-semibold mb-1" style={{ fontSize: 24 }}>Set your password</h1>
        <p className="field-help mb-5">Choose a password to activate your account and sign in.</p>

        {!token && <Alert kind="error">This link is missing its token. Ask your administrator to send a new activation link.</Alert>}

        {done ? (
          <Alert kind="success"><Icon.check size={15} /> Password set. Taking you to sign-in…</Alert>
        ) : token ? (
          <form onSubmit={submit} noValidate className="space-y-4">
            <Field label="New password" help="At least 8 characters.">
              <PasswordInput autoComplete="new-password" value={password} onChange={(e) => setPassword(e.target.value)} autoFocus />
            </Field>
            <Field label="Confirm password">
              <PasswordInput autoComplete="new-password" value={confirm} onChange={(e) => setConfirm(e.target.value)} />
            </Field>
            {error && <Alert kind="error">{error}</Alert>}
            <Button type="submit" className="w-full" loading={busy}>Set password &amp; continue</Button>
          </form>
        ) : null}
      </div>
    </div>
  )
}
