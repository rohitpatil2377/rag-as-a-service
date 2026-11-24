import { Component, useCallback, useEffect, useRef, useState } from 'react'
import type { ErrorInfo, MouseEvent as RMouseEvent, ReactNode, WheelEvent as RWheelEvent } from 'react'

// Lazy-loaded Mermaid diagram renderer for ```mermaid fenced code blocks in model answers.
// The `mermaid` package is sizeable, so it's dynamically imported on first use rather than
// bundled into the main chunk. Rendering happens into a plain DOM ref (mermaid needs a real
// element to render into) — never through dangerouslySetInnerHTML with raw model output;
// mermaid's own `securityLevel: 'strict'` sanitizes any HTML that sneaks into node labels.

type MermaidModule = typeof import('mermaid')

let mermaidModulePromise: Promise<MermaidModule> | null = null
function loadMermaid(): Promise<MermaidModule> {
  if (!mermaidModulePromise) mermaidModulePromise = import('mermaid')
  return mermaidModulePromise
}

function isDarkTheme(): boolean {
  return (document.documentElement.getAttribute('data-theme') || 'light') === 'dark'
}

// Resolve the platform's actual design tokens (design/tokens/tokens.css) into a Mermaid
// themeVariables map, so diagrams look native instead of Mermaid's generic default/dark theme.
// Same pattern GraphCanvas.tsx uses for its <canvas> palette: resolve CSS custom properties at
// render time (Mermaid, like <canvas>, can't inherit CSS) and re-resolve on theme flip.
function readThemeVariables() {
  const cs = getComputedStyle(document.documentElement)
  const v = (name: string, fallback: string) => cs.getPropertyValue(name).trim() || fallback
  const dark = isDarkTheme()
  return {
    darkMode: dark,
    background: v('--surface', dark ? '#1A2B4A' : '#FFFFFF'),
    primaryColor: v('--surface-2', dark ? '#22375C' : '#FAF7F2'),
    primaryTextColor: v('--text-primary', dark ? '#F5F7FA' : '#0D1B3A'),
    primaryBorderColor: v('--border-strong', dark ? '#3A5378' : '#CBD5E1'),
    secondaryColor: v('--surface', dark ? '#1A2B4A' : '#FFFFFF'),
    secondaryBorderColor: v('--border', dark ? '#2E456B' : '#E2E8F0'),
    tertiaryColor: v('--canvas', dark ? '#0A1327' : '#FAF7F2'),
    tertiaryBorderColor: v('--border', dark ? '#2E456B' : '#E2E8F0'),
    lineColor: v('--graph-edge', dark ? '#47598C' : '#9A9484'),
    textColor: v('--text-primary', dark ? '#F5F7FA' : '#0D1B3A'),
    mainBkg: v('--surface-2', dark ? '#22375C' : '#FAF7F2'),
    nodeBorder: v('--border-strong', dark ? '#3A5378' : '#CBD5E1'),
    nodeTextColor: v('--text-primary', dark ? '#F5F7FA' : '#0D1B3A'),
    clusterBkg: v('--canvas', dark ? '#0A1327' : '#FAF7F2'),
    clusterBorder: v('--border', dark ? '#2E456B' : '#E2E8F0'),
    titleColor: v('--text-heading', dark ? '#FFFFFF' : '#1A2B4A'),
    edgeLabelBackground: v('--surface', dark ? '#1A2B4A' : '#FFFFFF'),
    // sequence / state diagram actors, notes
    actorBkg: v('--surface-2', dark ? '#22375C' : '#FAF7F2'),
    actorBorder: v('--border-strong', dark ? '#3A5378' : '#CBD5E1'),
    actorTextColor: v('--text-primary', dark ? '#F5F7FA' : '#0D1B3A'),
    signalColor: v('--text-primary', dark ? '#F5F7FA' : '#0D1B3A'),
    signalTextColor: v('--text-primary', dark ? '#F5F7FA' : '#0D1B3A'),
    labelBoxBkgColor: v('--surface-2', dark ? '#22375C' : '#FAF7F2'),
    labelTextColor: v('--text-primary', dark ? '#F5F7FA' : '#0D1B3A'),
    noteBkgColor: v('--accent', dark ? '#E8D094' : '#C9A961'),
    noteTextColor: v('--text-heading', dark ? '#0D1B3A' : '#1A2B4A'),
    noteBorderColor: v('--border-strong', dark ? '#3A5378' : '#CBD5E1'),
    // accent used for the platform's brand colour (saffron) — active/decision nodes
    activationBkgColor: v('--action', dark ? '#FFA940' : '#E87722'),
    activationBorderColor: v('--action-hover', dark ? '#E87722' : '#C95F0E'),
    fontFamily: "'Inter', sans-serif",
    fontSize: '14px',
    errorBkgColor: v('--danger-bg', dark ? 'rgba(248,113,113,.12)' : '#FEF2F2'),
    errorTextColor: v('--danger', dark ? '#F87171' : '#DC2626'),
  }
}

let renderCounter = 0

// Mermaid keeps GLOBAL config + parser state, so two overlapping render() calls corrupt each other
// and throw — e.g. a mount render still in-flight when a data-theme flip triggers another. That's
// why a diagram which renders perfectly in isolation intermittently falls back to raw source inside
// the app. Guard it: SERIALIZE every render through one chain, use a FRESH id per attempt (reusing
// an id collides with a leftover node → throws), and retry once with the stock theme if the themed
// render throws (rules out any themeVariables/theme-state issue) before giving up.
let renderChain: Promise<unknown> = Promise.resolve()

async function renderMermaidSvg(code: string): Promise<string> {
  const attempt = async (): Promise<string> => {
    const { default: mermaid } = await loadMermaid()
    // htmlLabels:false renders labels as SVG <text> (not <foreignObject>), so the SVG rasterises to
    // PNG reliably across engines (foreignObject text is dropped by some canvas implementations).
    const base = {
      startOnLoad: false,
      securityLevel: 'strict' as const,
      fontFamily: "'Inter', sans-serif",
      htmlLabels: false,
      flowchart: { htmlLabels: false },
    }
    const src = code.trim()
    const freshId = () => `mermaid-${++renderCounter}-${Math.floor(performance.now())}`
    try {
      mermaid.initialize({ ...base, theme: 'base', themeVariables: readThemeVariables() })
      const { svg } = await mermaid.render(freshId(), src)
      return svg
    } catch {
      mermaid.initialize({ ...base, theme: 'default' })
      const { svg } = await mermaid.render(freshId(), src)
      return svg
    }
  }
  const result = renderChain.then(attempt, attempt)
  renderChain = result.catch(() => undefined)
  return result
}

// `naturalSize`: render the SVG at its own intrinsic size (not shrunk to the container width) so a
// dense diagram stays readable — the container then scrolls. Default keeps the fit-to-width behavior
// model answers rely on.
// Raw-source fallback shared by the async render failure AND the error boundary — a bad diagram
// shows its source (still useful/copyable) instead of blanking the answer.
function RawSourceFallback({ code, note }: { code: string; note?: string }) {
  return (
    <div style={{ margin: '8px 0' }}>
      <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4, display: 'flex', gap: 8, alignItems: 'center' }}>
        <span>{note || "Couldn't render diagram — showing raw source"}</span>
        <CopySourceButton code={code} />
      </div>
      <pre
        className="mono"
        style={{
          margin: 0, padding: '10px 14px', background: 'var(--surface-2)',
          border: '1px solid var(--border)', borderRadius: 'var(--radius-md)',
          overflowX: 'auto', fontSize: 13.5, color: 'var(--text-primary)', lineHeight: 1.55,
        }}
      >
        <code>{code}</code>
      </pre>
    </div>
  )
}

// A crash during rendering (mermaid internals, an unexpected SVG shape, a theme-repaint race that
// escapes the async guard) must NOT blank or crash the whole answer. This boundary catches any such
// synchronous error and degrades to the raw source, matching the async failure fallback.
class MermaidErrorBoundary extends Component<{ code: string; children: ReactNode }, { hasError: boolean }> {
  state = { hasError: false }
  static getDerivedStateFromError(): { hasError: boolean } {
    return { hasError: true }
  }
  componentDidCatch(_error: Error, _info: ErrorInfo): void {
    // swallowed on purpose — the raw-source fallback below is the user-facing recovery
  }
  render(): ReactNode {
    if (this.state.hasError) return <RawSourceFallback code={this.props.code} />
    return this.props.children
  }
}

export function MermaidBlock(props: { code: string; naturalSize?: boolean }) {
  return (
    <MermaidErrorBoundary code={props.code}>
      <MermaidDiagram {...props} />
    </MermaidErrorBoundary>
  )
}

function MermaidDiagram({ code, naturalSize = false }: { code: string; naturalSize?: boolean }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [failed, setFailed] = useState(false)
  const [svgMarkup, setSvgMarkup] = useState('')   // captured for the fullscreen viewer + download
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    let cancelled = false

    const run = async () => {
      try {
        const svg = await renderMermaidSvg(code)
        if (cancelled || !containerRef.current) return
        containerRef.current.innerHTML = svg
        if (naturalSize) {
          // Render at the diagram's TRUE pixel size (from its viewBox), not shrunk to the
          // container — otherwise a wide flowchart collapses to an unreadable thumbnail.
          const el = containerRef.current.querySelector('svg')
          if (el) {
            el.style.maxWidth = 'none'
            el.style.height = 'auto'
            const vb = el.getAttribute('viewBox')
            const w = vb ? parseFloat(vb.split(/\s+/)[2] || '0') : 0
            el.style.width = w > 0 ? `${Math.round(w)}px` : 'auto'
          }
        }
        setFailed(false)
        setSvgMarkup(svg)
      } catch {
        // Both the themed and the default-theme attempt failed — fall back to raw source rather
        // than letting the whole answer render crash.
        if (!cancelled) setFailed(true)
      }
    }

    void run()

    // Re-render on light/dark flip (baked-in colours) — debounced so a burst of attribute changes
    // during load doesn't fire overlapping renders (the serialization above is the real guard).
    let debounce: ReturnType<typeof setTimeout> | undefined
    const themeWatch = new MutationObserver(() => {
      if (debounce) clearTimeout(debounce)
      debounce = setTimeout(() => { void run() }, 80)
    })
    themeWatch.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })

    return () => { cancelled = true; if (debounce) clearTimeout(debounce); themeWatch.disconnect() }
  }, [code, naturalSize])

  if (failed) return <RawSourceFallback code={code} />

  // In-chat: fit-to-width but CAP the height so a large diagram scrolls inside its own box instead
  // of blowing out the page; a hover toolbar offers fullscreen (pan/zoom) + download for big ones.
  return (
    <>
      <div className="mermaid-wrap" style={{ position: 'relative', margin: '10px 0' }}>
        <div
          ref={containerRef}
          className="mermaid-diagram"
          style={{
            padding: '16px', background: 'var(--surface-2)',
            border: '1px solid var(--border)', borderRadius: 'var(--radius-md)',
            overflow: 'auto', maxHeight: '68vh',
            display: 'flex', justifyContent: naturalSize ? 'flex-start' : 'center',
          }}
        />
        {svgMarkup && (
          <div style={{ position: 'absolute', top: 10, right: 10, display: 'flex', gap: 6 }}>
            <DiagramTool title="Expand — fullscreen, pan &amp; zoom" onClick={() => setExpanded(true)}>⛶ Expand</DiagramTool>
            <DiagramTool title="Download as PNG" onClick={() => downloadPng(svgMarkup)}>↓ PNG</DiagramTool>
            <DiagramTool title="Download as SVG" onClick={() => downloadSvg(svgMarkup)}>↓ SVG</DiagramTool>
            <CopySourceButton code={code} />
          </div>
        )}
      </div>
      {expanded && <FullscreenDiagram svg={svgMarkup} code={code} onClose={() => setExpanded(false)} />}
    </>
  )
}

function DiagramTool({ children, onClick, title }: { children: ReactNode; onClick: () => void; title: string }) {
  return (
    <button type="button" onClick={onClick} title={title}
      style={{
        fontSize: 12, padding: '4px 9px', borderRadius: 7, cursor: 'pointer',
        border: '1px solid var(--border-strong)', background: 'var(--surface)',
        color: 'var(--text-primary)', backdropFilter: 'blur(4px)', display: 'inline-flex', alignItems: 'center', gap: 4,
      }}>
      {children}
    </button>
  )
}

function downloadSvg(svg: string) {
  const blob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'architecture-diagram.svg'
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

// Rasterise the rendered SVG to a PNG entirely in the browser (no deps): draw the SVG onto a 2x
// canvas over the surface colour (so it isn't transparent on dark), then download the canvas blob.
function downloadPng(svg: string) {
  const img = new Image()
  const src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg)
  img.onload = () => {
    let w = img.naturalWidth
    let h = img.naturalHeight
    if (!w || !h) {
      const m = svg.match(/viewBox="[\d.-]+ [\d.-]+ ([\d.]+) ([\d.]+)"/)
      if (m) { w = parseFloat(m[1]); h = parseFloat(m[2]) }
    }
    if (!w || !h) { w = 1200; h = 800 }
    const scale = 2
    const canvas = document.createElement('canvas')
    canvas.width = Math.max(1, Math.round(w * scale))
    canvas.height = Math.max(1, Math.round(h * scale))
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--surface').trim() || '#ffffff'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.setTransform(scale, 0, 0, scale, 0, 0)
    ctx.drawImage(img, 0, 0, w, h)
    canvas.toBlob((blob) => {
      if (!blob) return
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'architecture-diagram.png'
      a.click()
      setTimeout(() => URL.revokeObjectURL(url), 1000)
    }, 'image/png')
  }
  img.src = src
}

// Copy the raw mermaid source to the clipboard with brief inline confirmation.
function CopySourceButton({ code }: { code: string }) {
  const [copied, setCopied] = useState(false)
  const copy = () => {
    const done = () => { setCopied(true); setTimeout(() => setCopied(false), 1400) }
    try {
      navigator.clipboard?.writeText(code).then(done, done)
    } catch {
      done()
    }
  }
  return (
    <DiagramTool title="Copy Mermaid source" onClick={copy}>
      {copied ? '✓ Copied' : '⧉ Source'}
    </DiagramTool>
  )
}

// Fullscreen viewer for diagrams larger than the chat: drag to pan, wheel / +/- to zoom. No deps —
// a single transformed container holding the same rendered SVG.
function FullscreenDiagram({ svg, code, onClose }: { svg: string; code: string; onClose: () => void }) {
  const [t, setT] = useState({ x: 0, y: 0, z: 1 })
  const drag = useRef<{ px: number; py: number; ox: number; oy: number } | null>(null)
  const [grabbing, setGrabbing] = useState(false)
  const stageRef = useRef<HTMLDivElement>(null)
  // Open already fitted: scale the diagram to fill ~90% of the screen (measured from its intrinsic
  // viewBox, so it's transform-independent and re-fits correctly). Reset re-runs this.
  const fitToScreen = useCallback(() => {
    const el = stageRef.current?.querySelector('svg')
    if (!el) return
    let w = 0, h = 0
    const vb = el.getAttribute('viewBox')
    if (vb) { const p = vb.split(/\s+/); w = parseFloat(p[2] || '0'); h = parseFloat(p[3] || '0') }
    if (!w || !h) { const r = el.getBoundingClientRect(); w = r.width; h = r.height }
    if (!w || !h) return
    const fit = Math.min((window.innerWidth * 0.92) / w, (window.innerHeight * 0.84) / h)
    setT({ x: 0, y: 0, z: Math.min(6, Math.max(0.15, fit)) })
  }, [])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  useEffect(() => { const id = setTimeout(fitToScreen, 30); return () => clearTimeout(id) }, [fitToScreen])
  const zoom = (factor: number) => setT((s) => ({ ...s, z: Math.min(6, Math.max(0.15, s.z * factor)) }))
  const onWheel = (e: RWheelEvent) => { zoom(e.deltaY < 0 ? 1.12 : 1 / 1.12) }
  const onDown = (e: RMouseEvent) => { drag.current = { px: e.clientX, py: e.clientY, ox: t.x, oy: t.y }; setGrabbing(true) }
  const onMove = (e: RMouseEvent) => {
    const d = drag.current
    if (!d) return
    setT((s) => ({ ...s, x: d.ox + (e.clientX - d.px), y: d.oy + (e.clientY - d.py) }))
  }
  const stop = () => { drag.current = null; setGrabbing(false) }
  return (
    <div onClick={onClose}
      style={{ position: 'fixed', inset: 0, zIndex: 200, background: 'color-mix(in srgb, #05070f 84%, transparent)', backdropFilter: 'blur(2px)' }}>
      <div onClick={(e) => e.stopPropagation()} onWheel={onWheel} onMouseDown={onDown} onMouseMove={onMove} onMouseUp={stop} onMouseLeave={stop}
        style={{ position: 'absolute', inset: 0, overflow: 'hidden', cursor: grabbing ? 'grabbing' : 'grab' }}>
        <div
          ref={stageRef}
          style={{ position: 'absolute', top: '50%', left: '50%', transform: `translate(-50%, -50%) translate(${t.x}px, ${t.y}px) scale(${t.z})`, transformOrigin: 'center center' }}
          dangerouslySetInnerHTML={{ __html: svg }}
        />
      </div>
      <div onClick={(e) => e.stopPropagation()} style={{ position: 'absolute', top: 16, right: 16, display: 'flex', gap: 8 }}>
        <FsBtn onClick={() => zoom(1.2)} title="Zoom in">＋</FsBtn>
        <FsBtn onClick={() => zoom(1 / 1.2)} title="Zoom out">－</FsBtn>
        <FsBtn onClick={fitToScreen} title="Fit to screen">Fit</FsBtn>
        <FsBtn onClick={() => downloadPng(svg)} title="Download PNG">↓ PNG</FsBtn>
        <FsBtn onClick={() => downloadSvg(svg)} title="Download SVG">↓ SVG</FsBtn>
        <FsBtn onClick={() => { try { navigator.clipboard?.writeText(code) } catch { /* ignore */ } }} title="Copy Mermaid source">⧉ Source</FsBtn>
        <FsBtn onClick={onClose} title="Close (Esc)">✕ Close</FsBtn>
      </div>
      <div style={{ position: 'absolute', bottom: 16, left: '50%', transform: 'translateX(-50%)', fontSize: 12, color: '#cbd5e1', background: 'rgba(0,0,0,.4)', padding: '6px 12px', borderRadius: 8 }}>
        Drag to pan · scroll to zoom · Esc to close
      </div>
    </div>
  )
}

function FsBtn({ children, onClick, title }: { children: ReactNode; onClick: () => void; title: string }) {
  return (
    <button type="button" onClick={onClick} title={title}
      style={{
        fontSize: 13, padding: '6px 12px', borderRadius: 8, cursor: 'pointer',
        border: '1px solid rgba(255,255,255,.22)', background: 'rgba(255,255,255,.1)', color: '#E9EEF7',
        fontFamily: 'ui-monospace, monospace', backdropFilter: 'blur(4px)',
      }}>
      {children}
    </button>
  )
}
