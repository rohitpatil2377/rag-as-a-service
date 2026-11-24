// Sign-in / create-organisation — animated brand panel left, sign-in card right.
//
// The RAGAAS intro splash plays full-screen once and hands over to this scene.
//
// FITS THE VIEWPORT, ALWAYS. The shell is `h-screen overflow-hidden`, not `min-h-screen`:
// the previous version grew past the fold and clipped the last proof point mid-word behind a
// page scrollbar. Every band is `shrink-0` except the two that are allowed to absorb slack —
// the loop (`flex-1 min-h-0`, and the SVG letterboxes itself via preserveAspectRatio) and the
// card's scroll area, which only scrolls on a viewport genuinely too short for the form.
//
// The panel is NOT the intro's end-frame. Embedding the intro iframe here left a frozen lockup
// the moment its 6s timeline finished — dead exactly when the user arrived. `BrandScene` runs
// the product's own story continuously on a 9s cycle, in CSS/SVG: no library, no bundle cost.
import { useEffect, useState } from 'react'
import { Alert, Button, Field, PasswordInput, ThemeToggle } from '../components/ui'
import { BrandMark } from '../components/BrandMark'
import { ApiError, api } from '../lib/api'
import { useAuth } from '../lib/auth'
import { TAGLINE } from '../lib/brand'
import { useTheme } from '../lib/theme'

// Per-character reveal, echoing the intro's glyph-by-glyph tagline. The real string rides on a
// visually-hidden copy so a screen reader hears the phrase, not 25 separate letters.
function StreamedTagline({ delay = 0 }: { delay?: number }) {
  return (
    <span className="login-tag">
      <span className="sr-only">{TAGLINE}</span>
      {TAGLINE.split('').map((c, i) => (
        <span key={i} aria-hidden="true" style={{ animationDelay: `${delay + i * 26}ms` }}>{c}</span>
      ))}
    </span>
  )
}

// Claims worded to match what the code enforces: the grounded-or-abstain gate (FR-E2), the
// `assert_sovereign` egress allowlist, and the append-only audit chain. Deliberately NOT
// "data never leaves India" — deploy routes rerank/generation through hosted APIs.
const PROOF = [
  { k: 'Grounded, or it abstains', v: 'Every answer carries citations to its source.' },
  { k: 'Sovereign by configuration', v: 'Egress is refused unless the host is on the allowlist.' },
  { k: 'Tamper-evident', v: 'Every query and access decision lands on an append-only audit chain.' },
]

// --- The brand panel scene: the RAG pipeline ------------------------------------------------
// The product is Retrieval-Augmented Generation as a service, so the panel shows exactly that,
// left to right across the band: a private corpus -> the passages retrieval pulls -> the
// context window they assemble into -> an answer generated line by line, each line granted its
// citation only as it lands.
//
// Deliberately NOT the logo. The lockup already sits ~50px above this, so rendering the same
// mark at 8x made the hero a duplicate of something already on screen.
//
// Built with MASS. The scenes before this were hairlines (1-4.5px) in the faintest tokens, which
// is why they read as wireframes: filled slabs, gradient beams with a soft wide pass under a
// crisp one, and a real feGaussianBlur bloom on the generation core.

// Corpus shelf: [y, width, lit] — three of them are what this query retrieves.
const DOCS: [number, number, boolean][] = [
  [86, 150, false], [134, 178, false], [182, 162, true], [230, 186, false],
  [278, 158, true], [326, 176, false], [374, 168, true], [422, 152, false],
  [470, 180, false], [518, 164, false],
]
// Answer lines: [y, width] — ragged right, the way real prose sets.
const LINES: [number, number][] = [[214, 300], [258, 340], [302, 268], [346, 322]]

function BrandScene() {
  return (
    <svg viewBox="0 0 1200 640" className="w-full h-full" role="img"
      aria-label="The retrieval-augmented generation pipeline: passages are retrieved from a private corpus, assembled into a context window, and generated into an answer whose every line carries a citation.">
      <defs>
        <linearGradient id="ragPull" x1="0" y1="0" x2="1" y2="0">
          <stop className="rg-pull-0" offset="0" />
          <stop className="rg-pull-1" offset="1" />
        </linearGradient>
        <linearGradient id="ragPush" x1="0" y1="0" x2="1" y2="0">
          <stop className="rg-push-0" offset="0" />
          <stop className="rg-push-1" offset="1" />
        </linearGradient>
        <radialGradient id="ragGlow">
          <stop className="rg-glow-0" offset="0" />
          <stop className="rg-glow-1" offset="1" />
        </radialGradient>
        <filter id="ragBloom" x="-80%" y="-80%" width="260%" height="260%" colorInterpolationFilters="sRGB">
          <feGaussianBlur stdDeviation="22" result="b" />
          <feColorMatrix in="b" type="matrix"
            values="1.45 0 0 0 0  0 1.05 0 0 0  0 0 .32 0 0  0 0 0 .92 0" />
        </filter>
      </defs>

      {/* ---------- 1. the corpus ---------- */}
      <text className="rg-cap" x="64" y="58">CORPUS</text>
      {DOCS.map(([y, w, lit], i) => (
        <rect key={i} className={lit ? 'rg-doc rg-doc-lit' : 'rg-doc'} x="64" y={y} width={w} height="26" rx="4"
          style={lit ? { animationDelay: `${0.16 * DOCS.slice(0, i).filter((d) => d[2]).length}s` } : undefined} />
      ))}

      {/* ---------- 2. retrieval ---------- */}
      {[182, 278, 374].map((y, i) => (
        <g key={`p${i}`}>
          <path className="rg-pull" d={`M250 ${y + 13} C 380 ${y + 13}, 430 320, 548 320`} pathLength={1}
            strokeWidth="13" opacity=".18" style={{ animationDelay: `${i * 0.16}s` }} />
          <path className="rg-pull" d={`M250 ${y + 13} C 380 ${y + 13}, 430 320, 548 320`} pathLength={1}
            strokeWidth="3" style={{ animationDelay: `${i * 0.16}s` }} />
        </g>
      ))}

      {/* ---------- 3. the context window, where generation happens ---------- */}
      <circle className="rg-halo" cx="626" cy="320" r="188" fill="url(#ragGlow)" />
      <g className="rg-core">
        <rect className="rg-core-bloom" x="552" y="228" width="148" height="184" rx="12" filter="url(#ragBloom)" />
        <rect className="rg-core-box" x="552" y="228" width="148" height="184" rx="12" />
        <rect className="rg-ctx rg-ctx-1" x="576" y="258" width="100" height="16" rx="4" />
        <rect className="rg-ctx rg-ctx-2" x="576" y="288" width="100" height="16" rx="4" />
        <rect className="rg-ctx rg-ctx-3" x="576" y="318" width="100" height="16" rx="4" />
        <rect className="rg-ctx-q" x="576" y="360" width="100" height="24" rx="6" />
      </g>
      <text className="rg-cap rg-cap-mid" x="626" y="452">CONTEXT</text>

      {/* ---------- 4. the generated answer ---------- */}
      <text className="rg-cap" x="768" y="58">GROUNDED ANSWER</text>
      {LINES.map(([y, w], i) => (
        <g key={`a${i}`}>
          <path className="rg-push" d={`M704 320 C 740 320, 748 ${y + 9}, 768 ${y + 9}`} pathLength={1}
            strokeWidth="2.5" style={{ animationDelay: `${0.55 + i * 0.2}s` }} />
          <rect className="rg-line" x="768" y={y} width={w} height="18" rx="4"
            style={{ animationDelay: `${0.55 + i * 0.2}s` }} />
          <rect className="rg-cite" x={784 + w} y={y + 2} width="14" height="14"
            transform={`rotate(45 ${791 + w} ${y + 9})`}
            style={{ animationDelay: `${0.7 + i * 0.2}s` }} />
        </g>
      ))}
    </svg>
  )
}

export function Login() {
  const { login, signup } = useAuth()
  const { theme } = useTheme()
  const [mode, setMode] = useState<'login' | 'signup' | 'forgot'>('login')
  const [sent, setSent] = useState(false)
  const [org, setOrg] = useState('')
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Brand intro — a full-screen splash that plays every time the sign-in page loads, then
  // reveals the sign-in scene behind it. Skippable; auto-dismisses after the animation.
  const [showIntro, setShowIntro] = useState(true)
  const [introFading, setIntroFading] = useState(false)
  const skipIntro = () => setIntroFading(true) // triggers the fade-out; unmounts after the transition
  useEffect(() => {
    if (!showIntro) return
    // Let the logo animation run to its LIGHT "dawn" finish (~3.5s, shortened) so it doesn't cut off
    // mid-dark, then start the fade into the sign-in scene.
    const done = setTimeout(() => setIntroFading(true), 3600)
    return () => clearTimeout(done)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showIntro])
  useEffect(() => {
    if (!introFading) return
    const t = setTimeout(() => setShowIntro(false), 750) // matches the overlay opacity transition
    return () => clearTimeout(t)
  }, [introFading])

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    // Validate explicitly so a malformed email shows a message instead of the browser
    // silently blocking submit (the button appeared to do nothing — QA ISS-009).
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim())) {
      setError('Enter a valid email address (e.g. you@org.in).')
      return
    }
    if (mode === 'forgot') {
      setBusy(true)
      try { await api.forgotPassword(email.trim()) } catch { /* uniform response regardless */ }
      setBusy(false); setSent(true)
      return
    }
    if (mode === 'signup') {
      if (!org.trim() || !name.trim()) { setError('Organisation name and your name are required.'); return }
      if (password.length < 8) { setError('Password must be at least 8 characters.'); return }
    }
    setBusy(true)
    try {
      if (mode === 'signup') await signup(org.trim(), name.trim(), email.trim(), password)
      else await login(email.trim(), password)
    } catch (err) {
      const fallback = mode === 'signup' ? 'Could not create the organisation.' : 'Sign-in failed. Check your email and password.'
      setError(err instanceof ApiError ? err.message : fallback)
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      {showIntro && (
        <div style={{ position: 'fixed', inset: 0, zIndex: 100,
          // The plate behind the iframe, and what the cross-fade resolves through. It must be the
          // surface the intro ENDS on, not a fixed navy: in light theme the animation dawns to
          // ivory, and fading that through a near-black plate flashed dark mid-transition.
          background: theme === 'dark' ? '#070C1A' : '#EBEFF7',
          opacity: introFading ? 0 : 1, transition: 'opacity 0.75s ease', pointerEvents: introFading ? 'none' : 'auto' }}>
          {/* `?end=dark` in dark theme: the splash holds its dark field instead of dawning to
              ivory, so it lands on the same surface the sign-in screen is about to show. */}
          <iframe src={`/ragaas-intro.html${theme === 'dark' ? '?end=dark' : ''}`} title="RAG as a Service"
            style={{ width: '100%', height: '100%', border: 0, display: 'block' }} />
          <button type="button" onClick={skipIntro} aria-label="Skip intro"
            style={{ position: 'fixed', top: 18, right: 20, zIndex: 101, padding: '7px 16px', borderRadius: 8,
              border: '1px solid rgba(233,238,247,.28)', background: 'rgba(13,27,58,.55)', color: '#E9EEF7',
              fontSize: 13, cursor: 'pointer', fontFamily: 'ui-monospace, monospace', backdropFilter: 'blur(4px)' }}>
            Skip intro →
          </button>
        </div>
      )}

      {/* h-screen + overflow-hidden: the scene fits the fold, never the page scrollbar. */}
      <div className="h-screen overflow-hidden flex" style={{ background: 'var(--canvas)' }}>
        {/* ---------- LEFT: animated brand panel ---------- */}
        {/* Hidden below lg: on a phone the form is the only thing worth the viewport. */}
        {/* Seam colour lives in CSS as --rl-seam so it flips with the theme: the panel's own
            edge navy in dark, the standard --border hairline in light. Never a contrasting
            line — that drew the eye to the join and made the panels look stitched together. */}
        <aside className="rl-panel relative hidden lg:flex flex-col shrink-0 overflow-hidden"
          style={{ width: '60%' }}>
          {/* Saffron sweep — ambient, on an 11s clock deliberately out of phase with the 9s
              retrieval cycle so the two never lock into a visible beat. */}
          <div aria-hidden className="rl-sweep absolute inset-y-0 pointer-events-none"
            style={{ left: '-40%', width: '40%', filter: 'blur(8px)',
              background: 'linear-gradient(90deg, transparent, rgba(245,165,36,.10) 45%, rgba(245,165,36,.17) 55%, transparent)' }} />

          <div className="relative flex flex-col h-full min-h-0" style={{ zIndex: 2, padding: 'clamp(22px, 3.4vh, 40px) clamp(28px, 3.4vw, 52px)' }}>
            <div className="login-rise flex items-center gap-4 shrink-0" style={{ animationDelay: '80ms' }}>
              <span className="logo-plate inline-flex items-center"><BrandMark size={44} /></span>
              <div className="leading-tight" style={{ borderLeft: '2px solid var(--action)', paddingLeft: 14 }}>
                <div className="font-display font-bold" style={{ fontSize: 21, letterSpacing: '-0.01em', color: 'var(--rl-ink)' }}>
                  RagaaS <span style={{ color: 'var(--action)' }}>RagaaS</span>
                </div>
                {/* NOT `uppercase`: text-transform would render "RAGaaS" as "RAGAAS". */}
                <div className="mono" style={{ fontSize: 10, letterSpacing: '.16em', color: 'var(--rl-muted)' }}>
                  <StreamedTagline delay={300} />
                </div>
              </div>
            </div>

            {/* The only band that absorbs slack. The SVG letterboxes itself, so it shrinks to
                whatever height is left rather than pushing the proof points off-screen. */}
            <div className="login-rise flex-1 min-h-0 flex items-center justify-center" style={{ animationDelay: '260ms', padding: '10px 0' }}>
              <BrandScene />
            </div>

            <div className="shrink-0">
              <div className="mono" style={{ fontSize: 11, letterSpacing: '.26em', textTransform: 'uppercase', color: 'var(--action)', marginBottom: 18 }}>
                Sovereign RAG-as-a-Service
              </div>
              <ul style={{ listStyle: 'none', margin: 0, padding: 0, maxWidth: 580, display: 'grid', gap: 15 }}>
                {PROOF.map((p, i) => (
                  <li key={p.k} className="login-rise flex gap-3" style={{ animationDelay: `${460 + i * 120}ms` }}>
                    <span aria-hidden style={{ width: 3, alignSelf: 'stretch', background: 'var(--action)', opacity: .9, flexShrink: 0, borderRadius: 3 }} />
                    <div style={{ minWidth: 0 }}>
                      <div style={{ fontSize: 15.5, fontWeight: 600, color: 'var(--rl-ink)', letterSpacing: '-0.005em' }}>{p.k}</div>
                      <div style={{ fontSize: 13.5, lineHeight: 1.5, color: 'var(--rl-muted)' }}>{p.v}</div>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </aside>

        {/* ---------- RIGHT: form panel ---------- */}
        <div className="login-scene relative flex flex-col flex-grow min-w-0 overflow-hidden">
          {/* Faint panning grid + ONE warm corner wash. The navy aura that used to sit here was
              what muddied the seam into grey — with a navy panel alongside, a second navy bloom
              on the ivory side just dirtied it. Saffron only. */}
          <div aria-hidden className="login-grid absolute inset-0" style={{ backgroundImage: 'linear-gradient(var(--border) 1px, transparent 1px), linear-gradient(90deg, var(--border) 1px, transparent 1px)', backgroundSize: '44px 44px' }} />
          <div aria-hidden className="login-aura login-aura-saffron" />
          <svg aria-hidden className="absolute bottom-0 left-0 w-full" viewBox="0 0 1200 160" preserveAspectRatio="none" style={{ height: 104, opacity: .34 }}>
            <polyline className="login-chartline" fill="none" stroke="var(--accent)" strokeWidth="1.5"
              points="0,120 90,112 180,124 270,96 360,108 450,80 540,98 630,64 720,84 810,52 900,70 990,40 1080,56 1200,30" />
            <polyline fill="none" stroke="var(--border-strong)" strokeWidth="1"
              points="0,140 120,134 240,142 360,122 480,132 600,112 720,124 840,102 960,114 1080,92 1200,84" />
          </svg>

          {/* Brand lockup only below lg — above it the left panel carries the brand. */}
          <header className="relative z-10 flex items-center justify-between px-6 sm:px-10 pt-5 shrink-0">
            <div className="login-rise flex items-center gap-4 lg:hidden" style={{ animationDelay: '0ms' }}>
              <span className="logo-plate inline-flex items-center"><BrandMark size={42} /></span>
              <div className="leading-tight" style={{ borderLeft: '2px solid var(--accent)', paddingLeft: 14 }}>
                <div className="font-display font-bold" style={{ fontSize: 19, letterSpacing: '-0.01em', color: 'var(--text-heading)' }}>
                  RagaaS <span style={{ color: 'var(--action)' }}>RagaaS</span>
                </div>
                <div className="mono" style={{ fontSize: 10, letterSpacing: '.16em', color: 'var(--text-muted)' }}>
                  <StreamedTagline delay={300} />
                </div>
              </div>
            </div>
            <div className="login-rise ml-auto" style={{ animationDelay: '120ms' }}><ThemeToggle /></div>
          </header>

          {/* min-h-0 + overflow-y-auto: scrolls ONLY if the viewport is genuinely too short
              (e.g. the signup form on a laptop at 150% zoom) — never the whole page. */}
          {/* scrollbarGutter: reserve the gutter so the bar appearing (an error alert, or signup's
              two extra fields) never shoves the card sideways — that jump reads as the page moving. */}
          <main className="relative z-10 flex-1 min-h-0 overflow-y-auto flex items-center justify-center px-6 sm:px-10 py-6"
            style={{ scrollbarGutter: 'stable' }}>
            <div className="login-rise surface p-7 w-full" style={{ borderRadius: 16, maxWidth: 420, animationDelay: '200ms' }}>
              <div className="login-rise eyebrow mb-2" style={{ letterSpacing: '.22em', animationDelay: '320ms' }}>{mode === 'signup' ? 'Get started' : mode === 'forgot' ? 'Password reset' : 'Welcome back'}</div>
              <h2 className="login-rise font-display font-semibold mb-1" style={{ fontSize: 22, animationDelay: '380ms' }}>
                {mode === 'signup' ? 'Create your organisation' : mode === 'forgot' ? 'Reset your password' : 'Sign in'}
              </h2>
              <p className="login-rise text-sm mb-6" style={{ color: 'var(--text-muted)', animationDelay: '440ms' }}>
                {mode === 'signup' ? 'Your own isolated, sovereign workspace — ready in seconds.' : mode === 'forgot' ? "Enter your email and we'll send a reset link." : 'Access your knowledge workspace.'}
              </p>
              {mode === 'forgot' && sent ? (
                <>
                  <Alert kind="info">If an account exists for that email, a reset link is on its way. Check your inbox.</Alert>
                  <Button className="w-full mt-4" variant="subtle" onClick={() => { setMode('login'); setSent(false); setError(null) }}>Back to sign in</Button>
                </>
              ) : (
                <form onSubmit={submit} noValidate className="space-y-4">
                  {mode === 'signup' && (
                    <>
                      <div className="login-rise" style={{ animationDelay: '500ms' }}>
                        <Field label="Organisation name"><input className="field-input" value={org} onChange={(e) => setOrg(e.target.value)} placeholder="Acme Bank" required /></Field>
                      </div>
                      <div className="login-rise" style={{ animationDelay: '545ms' }}>
                        <Field label="Your name"><input className="field-input" value={name} onChange={(e) => setName(e.target.value)} placeholder="Alex Mercer" required /></Field>
                      </div>
                    </>
                  )}
                  <div className="login-rise" style={{ animationDelay: '500ms' }}>
                    <Field label="Work email"><input className="field-input" type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@org.in" required /></Field>
                  </div>
                  {mode !== 'forgot' && (
                    <div className="login-rise" style={{ animationDelay: '560ms' }}>
                      <Field label="Password" help={mode === 'signup' ? 'At least 8 characters.' : undefined}>
                        <PasswordInput autoComplete={mode === 'signup' ? 'new-password' : 'current-password'} value={password} onChange={(e) => setPassword(e.target.value)} required minLength={mode === 'signup' ? 8 : undefined} />
                      </Field>
                    </div>
                  )}
                  {mode === 'login' && (
                    <div className="login-rise text-right" style={{ marginTop: -8, animationDelay: '620ms' }}>
                      <button type="button" onClick={() => { setMode('forgot'); setError(null) }} className="text-xs" style={{ color: 'var(--action)', background: 'none', border: 'none', cursor: 'pointer' }}>Forgot password?</button>
                    </div>
                  )}
                  {error && <Alert kind="error">{error}</Alert>}
                  <div className="login-rise" style={{ animationDelay: '680ms' }}>
                    <Button type="submit" className="w-full" loading={busy}>{mode === 'signup' ? 'Create organisation' : mode === 'forgot' ? 'Send reset link' : 'Sign in'}</Button>
                  </div>
                </form>
              )}
              <div className="login-rise flex items-center justify-center gap-2 mt-4 text-xs" style={{ color: 'var(--text-muted)', animationDelay: '740ms' }}>
                <span className="inline-block rounded-full" style={{ width: 7, height: 7, background: 'var(--success)' }} />
                Secured session · Acme Sovereign Cloud
              </div>
              {mode === 'forgot' ? (
                <p className="login-rise text-sm mt-4 text-center" style={{ color: 'var(--text-muted)', animationDelay: '790ms' }}>
                  Remembered it?{' '}
                  <button type="button" onClick={() => { setMode('login'); setSent(false); setError(null) }}
                    className="font-semibold" style={{ color: 'var(--action)', cursor: 'pointer', background: 'none', border: 0, padding: 0 }}>Sign in</button>
                </p>
              ) : (
                <p className="login-rise text-sm mt-4 text-center" style={{ color: 'var(--text-muted)', animationDelay: '790ms' }}>
                  {mode === 'signup' ? 'Already have an account? ' : 'New to RAG as a Service? '}
                  <button type="button" onClick={() => { setMode(mode === 'signup' ? 'login' : 'signup'); setError(null) }}
                    className="font-semibold" style={{ color: 'var(--action)', cursor: 'pointer', background: 'none', border: 0, padding: 0 }}>
                    {mode === 'signup' ? 'Sign in' : 'Create an organisation'}
                  </button>
                </p>
              )}
            </div>
          </main>

          <footer className="login-rise relative z-10 px-6 sm:px-10 pb-4 shrink-0" style={{ animationDelay: '860ms' }}>
            <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>© 2026 Acme Software Solution Ltd. · Nashik, India</span>
          </footer>
        </div>
      </div>
    </>
  )
}
