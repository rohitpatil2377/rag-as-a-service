/**
 * RAG as a Service — embeddable chat widget (FR-I6).
 *
 * A dependency-free, self-contained script. Drop it on any page:
 *
 *   <script src="https://cdn.you/ragaas-widget.js"
 *           data-base="https://rag.internal"
 *           data-session-url="/api/ragaas-session"></script>
 *
 * `data-session-url` is an endpoint on YOUR server that holds the read-scoped API key and
 * proxies POST /v1/widget/session — so the key never reaches the browser. It must return the
 * JSON that endpoint returns: { token, kb_id, expires_in, config }.
 *
 * The widget then talks to RAG as a Service directly with the short-lived, single-KB, reader-only token.
 */
(function () {
  interface WidgetConfig {
    enabled: boolean
    title: string
    welcome: string
    accent: string
    placeholder: string
    suggestions: string[]
  }
  interface Session {
    token: string
    kb_id: string
    expires_in: number
    config: WidgetConfig
  }
  interface Citation { doc_id: string; quote: string }
  interface Answer {
    status: string
    answer: string
    citations: Citation[]
    suggested_followups: string[]
  }

  const script = document.currentScript as HTMLScriptElement | null
  const base = (script?.getAttribute('data-base') || '').replace(/\/$/, '')
  const sessionUrl = script?.getAttribute('data-session-url') || ''
  if (!base || !sessionUrl) {
    // eslint-disable-next-line no-console
    console.error('[ragaas-widget] missing data-base or data-session-url')
    return
  }

  let session: Session | null = null
  let open = false
  let busy = false

  const esc = (s: string) => s.replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c] as string))

  // --- styles (scoped by the .rag-w prefix) ---
  function injectStyles(accent: string) {
    const css = `
    .rag-w-btn{position:fixed;bottom:20px;right:20px;width:56px;height:56px;border-radius:50%;
      background:${accent};color:#fff;border:none;cursor:pointer;box-shadow:0 6px 20px rgba(0,0,0,.25);
      font-size:24px;z-index:2147483000;display:flex;align-items:center;justify-content:center}
    .rag-w-panel{position:fixed;bottom:88px;right:20px;width:370px;max-width:calc(100vw - 32px);
      height:520px;max-height:calc(100vh - 120px);background:#fff;border-radius:14px;z-index:2147483000;
      box-shadow:0 12px 40px rgba(0,0,0,.28);display:none;flex-direction:column;overflow:hidden;
      font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#1a1a2e}
    .rag-w-panel.open{display:flex}
    .rag-w-hd{background:${accent};color:#fff;padding:14px 16px;font-weight:600;font-size:15px;
      display:flex;justify-content:space-between;align-items:center}
    .rag-w-x{background:none;border:none;color:#fff;cursor:pointer;font-size:20px;line-height:1}
    .rag-w-body{flex:1;overflow-y:auto;padding:14px;background:#f7f8fa}
    .rag-w-msg{margin-bottom:12px;font-size:14px;line-height:1.5}
    .rag-w-q{text-align:right}
    .rag-w-q span{display:inline-block;background:${accent};color:#fff;padding:8px 12px;border-radius:12px 12px 2px 12px;max-width:85%}
    .rag-w-a span{display:inline-block;background:#fff;border:1px solid #e6e8ee;padding:8px 12px;border-radius:12px 12px 12px 2px;max-width:90%}
    .rag-w-cite{font-size:11px;color:#6b7280;margin-top:6px}
    .rag-w-sugg{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
    .rag-w-sugg button{background:#fff;border:1px solid ${accent};color:${accent};border-radius:14px;
      padding:4px 10px;font-size:12px;cursor:pointer}
    .rag-w-ft{border-top:1px solid #e6e8ee;padding:10px;display:flex;gap:8px;background:#fff}
    .rag-w-ft input{flex:1;border:1px solid #d7dae2;border-radius:8px;padding:9px 11px;font-size:14px;outline:none}
    .rag-w-ft button{background:${accent};color:#fff;border:none;border-radius:8px;padding:0 14px;cursor:pointer}
    .rag-w-powered{font-size:10px;color:#9aa0ab;text-align:center;padding:4px}`
    const el = document.createElement('style')
    el.textContent = css
    document.head.appendChild(el)
  }

  let bodyEl: HTMLElement
  let inputEl: HTMLInputElement

  function addMsg(kind: 'q' | 'a', html: string) {
    const d = document.createElement('div')
    d.className = `rag-w-msg rag-w-${kind}`
    d.innerHTML = `<span>${html}</span>`
    bodyEl.appendChild(d)
    bodyEl.scrollTop = bodyEl.scrollHeight
    return d
  }

  async function ask(question: string) {
    if (!session || busy || !question.trim()) return
    busy = true
    inputEl.value = ''
    addMsg('q', esc(question))
    const pending = addMsg('a', '…')
    try {
      const resp = await fetch(`${base}/v1/kb/${encodeURIComponent(session.kb_id)}/query`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${session.token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: question }),
      })
      if (!resp.ok) throw new Error(String(resp.status))
      const a: Answer = await resp.json()
      let html = esc(a.answer)
      if (a.status !== 'grounded') html = `<em>${esc(a.answer)}</em>`
      if (a.citations?.length) {
        html += `<div class="rag-w-cite">Sources: ${a.citations.map((c) => esc(c.doc_id)).join(', ')}</div>`
      }
      if (a.suggested_followups?.length) {
        html += '<div class="rag-w-sugg">' +
          a.suggested_followups.map((q) => `<button data-q="${esc(q)}">${esc(q)}</button>`).join('') + '</div>'
      }
      pending.innerHTML = `<span>${html}</span>`
      pending.querySelectorAll('button[data-q]').forEach((b) =>
        b.addEventListener('click', () => ask((b as HTMLElement).getAttribute('data-q') || '')))
    } catch {
      pending.innerHTML = '<span><em>Sorry — something went wrong. Please try again.</em></span>'
    } finally {
      busy = false
      bodyEl.scrollTop = bodyEl.scrollHeight
    }
  }

  function render(cfg: WidgetConfig) {
    injectStyles(cfg.accent)
    const btn = document.createElement('button')
    btn.className = 'rag-w-btn'
    btn.setAttribute('aria-label', cfg.title)
    btn.innerHTML = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H8l-4 4V5a2 2 0 0 1 2-2h13a2 2 0 0 1 2 2z"/></svg>'

    const panel = document.createElement('div')
    panel.className = 'rag-w-panel'
    panel.innerHTML =
      `<div class="rag-w-hd">${esc(cfg.title)}<button class="rag-w-x" aria-label="Close">×</button></div>` +
      '<div class="rag-w-body"></div>' +
      `<div class="rag-w-ft"><input type="text" placeholder="${esc(cfg.placeholder)}"/><button aria-label="Send"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2 11 13M22 2l-7 20-4-9-9-4z"/></svg></button></div>` +
      '<div class="rag-w-powered">Powered by RAG as a Service · grounded answers with citations</div>'
    document.body.appendChild(btn)
    document.body.appendChild(panel)

    bodyEl = panel.querySelector('.rag-w-body') as HTMLElement
    inputEl = panel.querySelector('input') as HTMLInputElement
    const sendBtn = panel.querySelector('.rag-w-ft button') as HTMLButtonElement

    addMsg('a', esc(cfg.welcome))
    if (cfg.suggestions?.length) {
      const wrap = document.createElement('div')
      wrap.className = 'rag-w-sugg'
      wrap.innerHTML = cfg.suggestions.map((q) => `<button data-q="${esc(q)}">${esc(q)}</button>`).join('')
      bodyEl.appendChild(wrap)
      wrap.querySelectorAll('button[data-q]').forEach((b) =>
        b.addEventListener('click', () => ask((b as HTMLElement).getAttribute('data-q') || '')))
    }

    const toggle = () => { open = !open; panel.classList.toggle('open', open); if (open) inputEl.focus() }
    btn.addEventListener('click', toggle)
    ;(panel.querySelector('.rag-w-x') as HTMLElement).addEventListener('click', toggle)
    sendBtn.addEventListener('click', () => ask(inputEl.value))
    inputEl.addEventListener('keydown', (e) => { if ((e as KeyboardEvent).key === 'Enter') ask(inputEl.value) })
  }

  async function boot() {
    try {
      const resp = await fetch(sessionUrl, { method: 'POST' })
      if (!resp.ok) throw new Error(String(resp.status))
      session = (await resp.json()) as Session
      if (!session.config?.enabled) return
      render(session.config)
    } catch {
      // eslint-disable-next-line no-console
      console.error('[ragaas-widget] could not start a session')
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot)
  else boot()
})()
