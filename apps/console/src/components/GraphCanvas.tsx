// Hand-rolled force-directed knowledge graph on <canvas> — no external library
// (sovereign, bundled). Ported from the RagaaS "Knowledge graph" design deliverable:
// physics settle, quadratic edges, degree-sized/community-coloured discs, pan/zoom/drag,
// hover + selection cross-highlighting with the Themes list, a detail popover with
// neighbours, and a HUD. Wired to LIVE subgraph + communities API data.
import { useEffect, useMemo, useRef, useState } from 'react'
import type { GraphCommunityInfo, GraphSubgraph } from '../lib/api'

// Categorical ramp derived from the Acme primitives — one entry per community, fixed
// order, medium chroma at a common lightness so no theme reads as louder than another.
// Teal #0EA5E9 is deliberately absent: the brand reserves it for AI-predicted series.
const RAMP_LIGHT = ['#2E4A7D', '#C25F12', '#0F7B3E', '#8A7028', '#1E8A8A', '#9B4B6E', '#5E7A32', '#A8562C']
const RAMP_DARK = ['#7FA3DE', '#FFA940', '#4FBF7C', '#E8D094', '#4FC6C6', '#D890AE', '#A8CB6E', '#E79A72']
const NEUTRAL_LIGHT = '#9A9484'
const NEUTRAL_DARK = '#6B7A94'

interface Pt { x: number; y: number; vx: number; vy: number }
interface View { s: number; tx: number; ty: number }
interface Ease extends View { k: number }

interface Palette {
  edge: string; muted: string; label: string; heading: string; ring: string; plate: string
}
interface NeighborView { i: number; label: string; degree: number; color: string }

export interface GraphCanvasProps {
  data: GraphSubgraph
  communities: GraphCommunityInfo[]
  activeCommunity: number | null
  onActiveCommunity: (id: number | null) => void
  // Optional hover signal from the Themes list, so hovering a theme lights its nodes.
  hoverCommunity?: number | null
  height?: number
}

function readTheme(): boolean {
  return (document.documentElement.getAttribute('data-theme') || 'light') === 'dark'
}

export function GraphCanvas({
  data, communities, activeCommunity, onActiveCommunity, hoverCommunity = null, height = 560,
}: GraphCanvasProps) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Deterministic community ordering → colour index. `communities` is size-desc; a node
  // whose community isn't listed (e.g. id -1) gets the neutral fallback.
  const orderIndex = useMemo(() => {
    const m = new Map<number, number>()
    communities.forEach((c, i) => m.set(c.community_id, i))
    return m
  }, [communities])
  const comById = useMemo(() => {
    const m = new Map<number, GraphCommunityInfo>()
    communities.forEach((c) => m.set(c.community_id, c))
    return m
  }, [communities])

  // React-facing state — drives the HUD, popover, zoom label. Everything the render loop
  // needs lives on refs so the animation loop never re-subscribes.
  const [hover, setHover] = useState<number | null>(null)
  const [sel, setSel] = useState<number | null>(null)
  const [zoom, setZoom] = useState(1)
  const [pop, setPop] = useState<{ left: number; top: number }>({ left: 0, top: 0 })

  // Mutable engine state shared between the effect closure and event handlers.
  const engine = useRef({
    pts: [] as Pt[],
    adj: [] as number[][],
    view: { s: 1, tx: 0, ty: 0 } as View,
    ease: null as Ease | null,
    W: 0, H: 0, dpr: 1,
    alpha: 1, entry: 1,
    drag: null as number | null,
    pan: null as { x: number; y: number; tx: number; ty: number } | null,
    hoverIdx: null as number | null,
    moved: false,
    reduce: false,
    maxDeg: 1,
    dark: false,
    ramp: RAMP_LIGHT as string[],
    neutral: NEUTRAL_LIGHT,
    C: {} as Palette,
    activeCom: null as number | null,
    hoverCom: null as number | null,
    sel: null as number | null,
    ctx: null as CanvasRenderingContext2D | null,
    raf: 0,
    // methods installed by the mount effect (undefined until then)
    draw: undefined as ((c: CanvasRenderingContext2D) => void) | undefined,
    frameCommunity: undefined as ((ci: number | null) => void) | undefined,
    select: undefined as ((i: number) => void) | undefined,
    pick: undefined as ((x: number, y: number) => number | null) | undefined,
    zoomBy: undefined as ((k: number) => void) | undefined,
    fitView: undefined as ((animate: boolean) => void) | undefined,
    reheat: undefined as (() => void) | undefined,
  })

  // Resolve the community colour for a node via its community id (deterministic order).
  const colorForNode = useRef<(i: number) => string>(() => '#888')

  // Keep engine mirrors of interaction state fresh so the draw loop reads the latest.
  useEffect(() => { engine.current.activeCom = activeCommunity; poke() }, [activeCommunity])
  useEffect(() => { engine.current.hoverCom = hoverCommunity ?? null; poke() }, [hoverCommunity])
  useEffect(() => { engine.current.sel = sel; poke() }, [sel])
  useEffect(() => { engine.current.hoverIdx = hover; poke() }, [hover])

  function poke() {
    const e = engine.current
    if (e.ctx && e.draw) e.draw(e.ctx)
  }
  // `draw` is stored on the engine so `poke()` (outside the effect) can trigger a repaint.

  // Frame a community (pin) — recompute popover on active change too.
  useEffect(() => {
    const e = engine.current
    if (!e.pts.length || !e.frameCommunity) return
    e.frameCommunity(activeCommunity)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeCommunity])

  useEffect(() => {
    const canvas = canvasRef.current
    const wrap = wrapRef.current
    if (!canvas || !wrap) return
    const ctx0 = canvas.getContext('2d')
    if (!ctx0) return
    const ctx: CanvasRenderingContext2D = ctx0
    const e = engine.current
    e.ctx = ctx

    const nodes = data.nodes
    const edges = data.edges
    const n = nodes.length

    const readPalette = () => {
      const cs = getComputedStyle(document.documentElement)
      const v = (name: string, fb: string) => cs.getPropertyValue(name).trim() || fb
      e.dark = readTheme()
      e.ramp = e.dark ? RAMP_DARK : RAMP_LIGHT
      e.neutral = e.dark ? NEUTRAL_DARK : NEUTRAL_LIGHT
      e.C = {
        edge: v('--graph-edge', e.dark ? '#47598C' : '#9A9484'),
        muted: v('--text-muted', e.dark ? '#A6B2C6' : '#5A6478'),
        label: v('--graph-label', e.dark ? '#C7D2E2' : '#3D4658'),
        heading: v('--text-heading', e.dark ? '#FFFFFF' : '#0D1B3A'),
        ring: v('--graph-ring', e.dark ? '#0B1730' : '#FBFAF7'),
        plate: v('--graph-plate', e.dark ? 'rgba(11,23,48,.86)' : 'rgba(251,250,247,.86)'),
      }
    }
    readPalette()

    // Colour resolver bound to the current ramp — falls back to neutral for unlisted comms.
    colorForNode.current = (i: number) => {
      const cid = nodes[i].community
      const oi = orderIndex.get(cid)
      if (oi == null) return e.neutral
      return e.ramp[oi % e.ramp.length]
    }
    const nodeColor = (i: number) => colorForNode.current(i)

    e.reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    e.maxDeg = Math.max(1, ...nodes.map((d) => d.degree))
    const radius = (i: number) => 2 + 4.5 * Math.sqrt(nodes[i].degree / e.maxDeg)  // even smaller discs

    // adjacency for neighbour lookup + edge highlighting
    e.adj = nodes.map(() => [])
    for (const [a, b] of edges) { e.adj[a].push(b); e.adj[b].push(a) }

    // Deterministic PRNG — the same corpus always lays out the same way.
    const rng = (seed: number) => { let s = seed >>> 0; return () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296 }
    const r = rng(77)
    const comCount = Math.max(1, communities.length)
    // Deterministic PHYLLOTAXIS (sunflower) disc: places EVERY node in one compact round globe —
    // even disconnected components, which a force layout would always leave as separate clusters.
    // Nodes are ordered by community (then degree) so each theme occupies a contiguous arc/region.
    const order = nodes.map((_, i) => i).sort((a, b) => {
      const ca = orderIndex.get(nodes[a].community) ?? comCount
      const cb = orderIndex.get(nodes[b].community) ?? comCount
      return ca - cb || nodes[b].degree - nodes[a].degree
    })
    const rank = new Array<number>(nodes.length)
    order.forEach((idx, k) => { rank[idx] = k })
    const GOLDEN = Math.PI * (3 - Math.sqrt(5))
    const spacing = 27
    e.pts = nodes.map((_, i) => {
      const k = rank[i]
      const ang = k * GOLDEN
      const rad = spacing * Math.sqrt(k + 0.5)
      return { x: Math.cos(ang) * rad + (r() - 0.5) * 6, y: Math.sin(ang) * rad + (r() - 0.5) * 6, vx: 0, vy: 0 }
    })
    e.view = { s: 1, tx: 0, ty: 0 }
    e.alpha = 1; e.entry = 1; e.drag = null; e.pan = null; e.hoverIdx = null; e.ease = null
    e.dpr = Math.min(window.devicePixelRatio || 1, 2)

    // Entry fade — a CSS transition on the element, so it can't stall if rAF throttles.
    canvas.style.opacity = '0'
    canvas.style.transition = e.reduce ? 'none' : 'opacity .55s cubic-bezier(.2,.6,.2,1)'
    setTimeout(() => { canvas.style.opacity = '1' }, 40)

    // ── physics ────────────────────────────────────────────────
    const link = 50
    // Forces are DELIBERATELY weak: the phyllotaxis seed already IS the globe; the sim only nudges
    // (collision + light gravity) to keep it round and non-overlapping. Strong springs/cohesion
    // would pull disconnected components back into separate clusters, breaking the globe.
    const charge = 650
    const step = () => {
      const pts = e.pts, a = e.alpha
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          let dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y
          let d2 = dx * dx + dy * dy || 0.01
          if (d2 > 160000) continue
          const d = Math.sqrt(d2)
          const f = (charge * a) / d2
          dx /= d; dy /= d
          pts[i].vx += dx * f; pts[i].vy += dy * f
          pts[j].vx -= dx * f; pts[j].vy -= dy * f
          const min = radius(i) + radius(j) + 7
          if (d < min) { const push = (min - d) * 0.22; pts[i].vx += dx * push; pts[i].vy += dy * push; pts[j].vx -= dx * push; pts[j].vy -= dy * push }
        }
      }
      for (const [x, y] of edges) {
        let dx = pts[y].x - pts[x].x, dy = pts[y].y - pts[x].y
        const d = Math.sqrt(dx * dx + dy * dy) || 0.01
        const f = ((d - link) / d) * 0.012 * a
        dx *= f; dy *= f
        pts[x].vx += dx; pts[x].vy += dy
        pts[y].vx -= dx; pts[y].vy -= dy
      }
      // Community cohesion — pull each entity toward its theme centroid (keyed by colour
      // order so unlisted nodes cluster together in one "other" bucket).
      const cx: Record<number, number> = {}, cy: Record<number, number> = {}, cn: Record<number, number> = {}
      for (let i = 0; i < n; i++) {
        const c = orderIndex.get(nodes[i].community) ?? -1
        cx[c] = (cx[c] || 0) + pts[i].x; cy[c] = (cy[c] || 0) + pts[i].y; cn[c] = (cn[c] || 0) + 1
      }
      for (let i = 0; i < n; i++) {
        const c = orderIndex.get(nodes[i].community) ?? -1
        pts[i].vx += (cx[c] / cn[c] - pts[i].x) * 0.004 * a
        pts[i].vy += (cy[c] / cn[c] - pts[i].y) * 0.004 * a
      }
      for (let i = 0; i < n; i++) {
        if (i === e.drag) { pts[i].vx = 0; pts[i].vy = 0; continue }
        // Strong central gravity dominates repulsion so ALL nodes — even DISCONNECTED components
        // (which edges can't pull together) — collapse toward the centre into ONE round globe,
        // rather than separate clusters drifting apart. Collision below keeps them from piling up.
        pts[i].vx += -pts[i].x * 0.018 * a
        pts[i].vy += -pts[i].y * 0.018 * a
        pts[i].vx *= 0.80; pts[i].vy *= 0.80
        pts[i].x += pts[i].vx; pts[i].y += pts[i].vy
      }
      e.alpha *= 0.988
    }

    // ── view helpers ───────────────────────────────────────────
    const fitView = (animate: boolean) => {
      if (!e.pts.length || !e.W) return
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity
      e.pts.forEach((p, i) => { const rr = radius(i) + 14; x0 = Math.min(x0, p.x - rr); y0 = Math.min(y0, p.y - rr); x1 = Math.max(x1, p.x + rr); y1 = Math.max(y1, p.y + rr) })
      const s = Math.max(0.35, Math.min(2.2, Math.min(e.W / (x1 - x0), e.H / (y1 - y0))))
      const target: Ease = { s, tx: e.W / 2 - ((x0 + x1) / 2) * s, ty: e.H / 2 - ((y0 + y1) / 2) * s, k: 0.12 }
      if (animate && !e.reduce) e.ease = target
      else { e.view = { s: target.s, tx: target.tx, ty: target.ty }; e.ease = null }
      setZoom(target.s)
    }

    const frameCommunity = (ci: number | null) => {
      if (!e.pts.length) return
      if (ci == null) { fitView(true); return }
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity
      e.pts.forEach((p, i) => {
        if (nodes[i].community !== ci) return
        const rr = radius(i) + 54
        x0 = Math.min(x0, p.x - rr); y0 = Math.min(y0, p.y - rr); x1 = Math.max(x1, p.x + rr); y1 = Math.max(y1, p.y + rr)
      })
      if (!isFinite(x0)) return
      const s = Math.max(0.4, Math.min(2.4, Math.min(e.W / (x1 - x0), e.H / (y1 - y0))))
      e.ease = { s, tx: e.W / 2 - ((x0 + x1) / 2) * s, ty: e.H / 2 - ((y0 + y1) / 2) * s, k: 0.13 }
      setZoom(s)
    }
    e.frameCommunity = frameCommunity

    const applyEase = (dt: number) => {
      if (!e.ease) return
      const ez = e.ease, v = e.view
      const k = Math.min(1, ez.k * Math.max(1, (dt || 16.7) / 16.7))
      v.s += (ez.s - v.s) * k; v.tx += (ez.tx - v.tx) * k; v.ty += (ez.ty - v.ty) * k
      if (Math.abs(ez.s - v.s) < 0.002 && Math.abs(ez.tx - v.tx) < 0.6 && Math.abs(ez.ty - v.ty) < 0.6) {
        e.view = { s: ez.s, tx: ez.tx, ty: ez.ty }; e.ease = null; setZoom(ez.s)
      }
    }

    // ── paint ──────────────────────────────────────────────────
    const draw = (c2d: CanvasRenderingContext2D) => {
      const { W, H, dpr, view } = e
      const sel_ = e.sel, hov = e.hoverIdx, com = e.activeCom ?? e.hoverCom
      const focus = hov != null ? hov : sel_
      const near = focus != null ? new Set(e.adj[focus]) : null
      const C = e.C
      const labelFloor = 0.6  // main hubs labelled (a few more than 0.75); rest show on hover/select

      c2d.setTransform(dpr, 0, 0, dpr, 0, 0)
      c2d.clearRect(0, 0, W, H)
      c2d.setTransform(dpr * view.s, 0, 0, dpr * view.s, dpr * view.tx, dpr * view.ty)

      const lit = (i: number) => {
        if (com != null) return nodes[i].community === com
        if (focus == null) return true
        return i === focus || (near != null && near.has(i))
      }
      const dimming = focus != null || com != null

      // edges — quadratic curve, brighten when they touch the focus or lie inside a theme
      for (const [a, b] of edges) {
        const on = com != null ? (nodes[a].community === com && nodes[b].community === com)
          : focus != null ? (a === focus || b === focus) : false
        const pa = e.pts[a], pb = e.pts[b]
        const mx = (pa.x + pb.x) / 2, my = (pa.y + pb.y) / 2
        const dx = pb.x - pa.x, dy = pb.y - pa.y
        c2d.beginPath()
        c2d.moveTo(pa.x, pa.y)
        c2d.quadraticCurveTo(mx - dy * 0.07, my + dx * 0.07, pb.x, pb.y)
        c2d.strokeStyle = on
          ? (nodes[a].community === nodes[b].community ? nodeColor(a) : C.muted)
          : C.edge
        c2d.globalAlpha = e.entry * (on ? 0.62 : dimming ? 0.06 : 0.24)
        c2d.lineWidth = (on ? 1.4 : 0.9) / view.s
        c2d.stroke()
      }
      c2d.globalAlpha = 1

      // nodes
      for (let i = 0; i < n; i++) {
        const p = e.pts[i], rr = radius(i), col = nodeColor(i)
        const on = lit(i)
        const isFocus = i === focus
        c2d.globalAlpha = e.entry * (on ? 1 : 0.16)
        if (isFocus || (com != null && on)) { c2d.shadowColor = col; c2d.shadowBlur = 16 / view.s } else { c2d.shadowBlur = 0 }
        c2d.beginPath(); c2d.arc(p.x, p.y, rr, 0, Math.PI * 2)
        c2d.fillStyle = col; c2d.fill()
        c2d.shadowBlur = 0
        c2d.lineWidth = (isFocus ? 2.2 : 1.6) / view.s
        c2d.strokeStyle = isFocus ? C.heading : C.ring
        c2d.stroke()
        if (i === sel_) {
          c2d.beginPath(); c2d.arc(p.x, p.y, rr + 5 / view.s, 0, Math.PI * 2)
          c2d.strokeStyle = C.heading; c2d.lineWidth = 1 / view.s; c2d.globalAlpha = e.entry * 0.55; c2d.stroke()
        }
      }
      c2d.globalAlpha = 1

      // labels — declutter by degree, zoom and interaction
      c2d.textAlign = 'center'; c2d.textBaseline = 'bottom'
      const fs = 11 / view.s
      // Indic-capable fallbacks so Devanagari/Tamil/… entity labels render real glyphs on the
      // canvas instead of tofu boxes (Inter has no Indic coverage; the generic fallback misses some
      // conjuncts). Covers Linux/mac (Noto) and Windows (Nirmala UI).
      c2d.font = '600 ' + fs + 'px Inter, "Noto Sans Tamil", "Noto Sans Devanagari", "Noto Sans", "Nirmala UI", system-ui, sans-serif'
      for (let i = 0; i < n; i++) {
        const norm = nodes[i].degree / e.maxDeg
        const forced = i === focus || (near != null && near.has(i)) || i === sel_ || (com != null && nodes[i].community === com)
        // Sparse by default: only the strongest hubs are labelled at rest; the rest reveal on
        // hover/select (forced above), when a community is active, or as you zoom right in.
        // Only the main hubs at rest — no zoom-flood. Other names appear ONLY on hover/select (forced).
        const byZoom = norm >= labelFloor
        if (!forced && (!byZoom || (dimming && !lit(i)))) continue
        const p = e.pts[i], rr = radius(i)
        const t = nodes[i].label.length > 26 ? nodes[i].label.slice(0, 25) + '…' : nodes[i].label
        const w = c2d.measureText(t).width
        c2d.globalAlpha = e.entry * (lit(i) ? 1 : 0.2)
        c2d.fillStyle = C.plate
        c2d.fillRect(p.x - w / 2 - 3 / view.s, p.y - rr - 4 / view.s - fs, w + 6 / view.s, fs + 2 / view.s)
        c2d.fillStyle = i === focus || i === sel_ ? C.heading : C.label
        c2d.fillText(t, p.x, p.y - rr - 3 / view.s)
      }
      c2d.globalAlpha = 1
      c2d.setTransform(dpr, 0, 0, dpr, 0, 0)

      // reposition popover BESIDE the selected node (never on top of it): prefer the right of
      // the node, flip to the left if it would overflow, and clamp to the viewport.
      if (sel_ != null && e.pts[sel_] && e.W) {
        const p = e.pts[sel_]
        const sx = p.x * view.s + view.tx, sy = p.y * view.s + view.ty
        const nr = radius(sel_) * view.s
        const popW = 244, gap = 14
        let left = sx + nr + gap                       // to the right of the node
        if (left + popW > e.W - 12) left = sx - nr - gap - popW  // flip to the left if no room
        left = Math.max(12, Math.min(e.W - popW - 12, left))
        const top = Math.max(12, Math.min(Math.max(12, e.H - 230), sy - 40))
        setPop((prev) => (prev.left !== left || prev.top !== top ? { left, top } : prev))
      }
    }
    e.draw = draw

    // ── pick / select ──────────────────────────────────────────
    const pick = (gx: number, gy: number): number | null => {
      let best: number | null = null, bd = Infinity
      for (let i = 0; i < e.pts.length; i++) {
        const dx = e.pts[i].x - gx, dy = e.pts[i].y - gy, d = dx * dx + dy * dy
        const rr = radius(i) + 7 / e.view.s
        if (d < rr * rr && d < bd) { bd = d; best = i }
      }
      return best
    }
    const select = (i: number) => {
      // Select + highlight IN PLACE — do NOT move the camera. Auto-zooming to the node on every
      // click threw the rest of the graph off-screen ("goes away to empty"). Double-click still
      // deliberately frames the group; the popover + neighbour highlight are enough for a click.
      setSel(i); e.sel = i
    }
    e.select = select
    e.pick = pick

    // ── sizing ─────────────────────────────────────────────────
    const resize = () => {
      e.W = wrap.clientWidth || 800; e.H = wrap.clientHeight || height
      canvas.width = e.W * e.dpr; canvas.height = e.H * e.dpr
      canvas.style.width = e.W + 'px'; canvas.style.height = e.H + 'px'
      fitView(false)
      draw(ctx)
    }
    resize()
    const ro = new ResizeObserver(resize); ro.observe(wrap)

    // Settle headlessly before first paint so the graph fades in already readable.
    for (let k = 0; k < 420; k++) step()
    e.alpha = e.reduce ? 0 : 0.06
    fitView(false)
    draw(ctx)

    // Time-based loop — a throttled/backgrounded tab still settles and catches up.
    let last = performance.now()
    const loop = (now?: number) => {
      const t = now || performance.now()
      const dt = Math.min(400, t - last)
      last = t
      if (!e.reduce) {
        const iters = Math.max(1, Math.min(24, Math.round(dt / 16.7)))
        for (let k = 0; k < iters && e.alpha > 0.004; k++) step()
      }
      if (e.ease) applyEase(dt)
      draw(ctx)
      e.raf = requestAnimationFrame(loop)
    }
    loop(performance.now())

    // ── theme flip → re-resolve palette + repaint ──────────────
    const themeWatch = new MutationObserver(() => { readPalette(); draw(ctx) })
    themeWatch.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })

    // ── pointer ────────────────────────────────────────────────
    const local = (ev: MouseEvent) => { const b = canvas.getBoundingClientRect(); return { x: ev.clientX - b.left, y: ev.clientY - b.top } }
    const toGraph = (p: { x: number; y: number }) => ({ x: (p.x - e.view.tx) / e.view.s, y: (p.y - e.view.ty) / e.view.s })

    const onMove = (ev: MouseEvent) => {
      const p = local(ev)
      if (e.drag != null) {
        const g = toGraph(p)
        e.pts[e.drag].x = g.x; e.pts[e.drag].y = g.y; e.pts[e.drag].vx = 0; e.pts[e.drag].vy = 0
        e.alpha = Math.max(e.alpha, 0.28); e.moved = true; return
      }
      if (e.pan) { e.view.tx = e.pan.tx + (p.x - e.pan.x); e.view.ty = e.pan.ty + (p.y - e.pan.y); e.moved = true; return }
      const gm = toGraph(p)
      const hit = pick(gm.x, gm.y)
      canvas.style.cursor = hit != null ? 'pointer' : 'grab'
      if (hit !== e.hoverIdx) { e.hoverIdx = hit; setHover(hit) }
    }
    const onDown = (ev: PointerEvent) => {
      const p = local(ev); const g = toGraph(p); const hit = pick(g.x, g.y)
      e.moved = false
      // Pointer capture keeps move/up events flowing to THIS canvas even when the cursor leaves
      // it mid-drag — without it, a plain mousemove stops firing at the edge and the pan/drag dies.
      try { canvas.setPointerCapture(ev.pointerId) } catch { /* older browsers: window fallback */ }
      if (hit != null) { e.drag = hit; canvas.style.cursor = 'grabbing' }
      else { e.pan = { x: p.x, y: p.y, tx: e.view.tx, ty: e.view.ty }; canvas.style.cursor = 'grabbing' }
    }
    const onUp = (ev?: PointerEvent) => {
      e.drag = null; e.pan = null; canvas.style.cursor = 'grab'
      if (ev) { try { canvas.releasePointerCapture(ev.pointerId) } catch { /* no-op */ } }
    }
    const onClick = (ev: MouseEvent) => {
      if (e.moved) { e.moved = false; return }
      const g = toGraph(local(ev)); const hit = pick(g.x, g.y)
      if (hit != null) select(hit)
      else { setSel(null); e.sel = null }
    }
    // Only clear HOVER on leave — NOT an in-progress drag/pan (pointer capture keeps that alive
    // outside the canvas; clearing it here is what made panning past the edge stop dead).
    const onLeave = () => { if (e.drag == null && e.pan == null) { e.hoverIdx = null; setHover(null) } }
    const onWheel = (ev: WheelEvent) => {
      ev.preventDefault()
      const p = local(ev)
      const k = Math.exp(-ev.deltaY * 0.0016)
      const s2 = Math.max(0.35, Math.min(4, e.view.s * k))
      const f = s2 / e.view.s
      e.view.tx = p.x - (p.x - e.view.tx) * f
      e.view.ty = p.y - (p.y - e.view.ty) * f
      e.view.s = s2; setZoom(s2)
    }
    const onDbl = (ev: MouseEvent) => {
      // The two clicks of a double-click already selected the node (and may have eased the view),
      // so re-picking at the cursor can miss and fall through to "empty". Use the SELECTED node
      // index (view-independent); only fall back to a fresh pick if nothing is selected.
      const g = toGraph(local(ev))
      const hit = e.sel != null ? e.sel : pick(g.x, g.y)
      setSel(null); e.sel = null
      if (hit != null) {
        // double-click a node → focus its GROUP: pin + frame + highlight that community.
        const ci = nodes[hit].community
        onActiveCommunity(ci)
        frameCommunity(ci)
      } else {
        // double-click empty space → reset the whole view
        onActiveCommunity(null); fitView(true)
      }
    }

    canvas.addEventListener('pointermove', onMove)
    canvas.addEventListener('pointerdown', onDown)
    canvas.addEventListener('pointerup', onUp)
    canvas.addEventListener('pointercancel', onUp)
    canvas.addEventListener('click', onClick)
    canvas.addEventListener('mouseleave', onLeave)
    canvas.addEventListener('dblclick', onDbl)
    canvas.addEventListener('wheel', onWheel, { passive: false })

    // Expose toolbar hooks on the engine so the JSX buttons can drive the view.
    const zoomBy = (k: number) => {
      const s2 = Math.max(0.35, Math.min(4, e.view.s * k))
      const f = s2 / e.view.s
      e.view.tx = e.W / 2 - (e.W / 2 - e.view.tx) * f
      e.view.ty = e.H / 2 - (e.H / 2 - e.view.ty) * f
      e.view.s = s2; setZoom(s2)
    }
    e.zoomBy = zoomBy
    e.fitView = fitView
    e.reheat = () => {
      setSel(null); e.sel = null; onActiveCommunity(null)
      e.alpha = 1
      setTimeout(() => fitView(true), 1400)
    }

    return () => {
      cancelAnimationFrame(e.raf)
      ro.disconnect()
      themeWatch.disconnect()
      canvas.removeEventListener('pointermove', onMove)
      canvas.removeEventListener('pointerdown', onDown)
      canvas.removeEventListener('pointerup', onUp)
      canvas.removeEventListener('pointercancel', onUp)
      canvas.removeEventListener('click', onClick)
      canvas.removeEventListener('mouseleave', onLeave)
      canvas.removeEventListener('dblclick', onDbl)
      canvas.removeEventListener('wheel', onWheel)
      e.ctx = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, communities, orderIndex, height])

  // ── derived HUD / popover values (from live data) ────────────
  const nodes = data.nodes
  const focus = hover != null ? hover : sel
  const activeCom = activeCommunity ?? hoverCommunity ?? null
  const hudTitle = focus != null
    ? nodes[focus].label
    : activeCom != null
      ? (comById.get(activeCom)?.label ?? `Theme ${activeCom}`)
      : `Top ${nodes.length} entities by degree`
  const hudMeta = focus != null
    ? `${nodes[focus].degree} links · ${comById.get(nodes[focus].community)?.label ?? 'unclustered'}`
    : activeCom != null
      ? `${nodes.filter((nd) => nd.community === activeCom).length} entities in this theme`
      : 'hover an entity to trace its links'

  const neighbors: NeighborView[] = useMemo(() => {
    if (sel == null) return []
    const adj = engine.current.adj
    const list = (adj[sel] ?? []).map((i) => ({ i, ...nodes[i] }))
    list.sort((a, b) => b.degree - a.degree)
    return list.slice(0, 6).map((nd) => ({ i: nd.i, label: nd.label, degree: nd.degree, color: colorForNode.current(nd.i) }))
  }, [sel, nodes])

  const selColor = sel != null ? colorForNode.current(sel) : 'var(--border)'
  const selTheme = sel != null ? (comById.get(nodes[sel].community)?.label ?? 'unclustered') : ''

  const btn = (extra?: React.CSSProperties): React.CSSProperties => ({
    height: 26, display: 'flex', alignItems: 'center', justifyContent: 'center',
    border: '1px solid var(--border)', borderRadius: 3, background: 'var(--surface)',
    color: 'var(--text-muted)', fontFamily: 'var(--font-body)', fontSize: 11.5, fontWeight: 600,
    cursor: 'pointer', lineHeight: 1, ...extra,
  })

  const clickSelect = (i: number) => engine.current.select?.(i)
  const zoomBy = (k: number) => engine.current.zoomBy?.(k)
  const doFit = () => { setSel(null); onActiveCommunity(null); engine.current.fitView?.(true) }
  const doReheat = () => engine.current.reheat?.()

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0 }}>
      {/* toolbar */}
      <div style={{ height: 44, flexShrink: 0, borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 16px', gap: 12 }}>
        <span style={{ fontFamily: 'var(--font-heading)', fontSize: 13, fontWeight: 600, color: 'var(--text-heading)' }}>Graph map</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span className="mono" style={{ fontSize: 11.5, color: 'var(--text-muted)', marginRight: 6 }}>{Math.round(zoom * 100)}%</span>
          <button type="button" onClick={() => zoomBy(0.8)} title="Zoom out" style={btn({ width: 26, fontSize: 14 })}>−</button>
          <button type="button" onClick={() => zoomBy(1.25)} title="Zoom in" style={btn({ width: 26, fontSize: 14 })}>+</button>
          <button type="button" onClick={doFit} style={btn({ padding: '0 10px' })}>Fit</button>
          <button type="button" onClick={doReheat} style={btn({ padding: '0 10px' })}>Re-layout</button>
        </div>
      </div>

      {/* canvas field */}
      <div ref={wrapRef} style={{ position: 'relative', flex: 1, minHeight: 0, background: 'var(--sunken)', overflow: 'hidden' }}>
        <canvas ref={canvasRef} style={{ display: 'block', touchAction: 'none', cursor: 'grab' }} />

        <div style={{ position: 'absolute', top: 10, left: 12, display: 'flex', flexDirection: 'column', gap: 2, pointerEvents: 'none', background: 'color-mix(in srgb, var(--sunken) 78%, transparent)', backdropFilter: 'blur(3px)', padding: '5px 9px', borderRadius: 8 }}>
          <span style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text-heading)' }}>{hudTitle}</span>
          <span className="mono" style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>{hudMeta}</span>
        </div>

        <div className="mono" style={{ position: 'absolute', bottom: 10, left: 12, fontSize: 11.5, color: 'var(--text-muted)', pointerEvents: 'none', letterSpacing: '.04em', background: 'color-mix(in srgb, var(--sunken) 78%, transparent)', backdropFilter: 'blur(3px)', padding: '4px 9px', borderRadius: 8 }}>
          scroll to zoom · drag a node · drag the field to pan · double-click to reset
        </div>

        {sel != null && (
          <div style={{ position: 'absolute', left: pop.left, top: pop.top, width: 244, background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 10, boxShadow: '0 18px 36px -20px var(--shadow-2)', overflow: 'hidden', zIndex: 5 }}>
            <div style={{ padding: '11px 14px 10px', borderBottom: '1px solid var(--hair)', display: 'flex', flexDirection: 'column', gap: 5 }}>
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8 }}>
                <span style={{ fontFamily: 'var(--font-heading)', fontSize: 13.5, fontWeight: 600, color: 'var(--text-heading)', lineHeight: 1.3 }}>{nodes[sel].label}</span>
                <button type="button" onClick={() => { setSel(null); engine.current.sel = null }} style={{ flexShrink: 0, width: 18, height: 18, border: 'none', background: 'none', color: 'var(--text-muted)', fontSize: 13, lineHeight: 1, cursor: 'pointer', padding: 0 }}>×</button>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                <span style={{ width: 8, height: 8, borderRadius: 2, background: selColor, flexShrink: 0 }} />
                <span style={{ fontSize: 11.5, color: 'var(--text-muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{selTheme}</span>
                <span className="mono" style={{ fontSize: 11.5, color: 'var(--text-muted)', marginLeft: 'auto', whiteSpace: 'nowrap', flexShrink: 0 }}>{nodes[sel].degree} links</span>
              </div>
            </div>
            <div style={{ padding: '10px 14px 12px', display: 'flex', flexDirection: 'column', gap: 6 }}>
              <span className="mono" style={{ fontSize: 11.5, fontWeight: 600, letterSpacing: '.18em', textTransform: 'uppercase', color: 'var(--text-muted)' }}>Connected to</span>
              {neighbors.length === 0 && <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>No connections in this view.</span>}
              {neighbors.map((nb) => (
                <button key={nb.i} type="button" onClick={() => clickSelect(nb.i)}
                  style={{ display: 'flex', alignItems: 'center', gap: 8, width: '100%', textAlign: 'left', background: 'none', border: 'none', padding: '2px 0', cursor: 'pointer', fontFamily: 'var(--font-body)' }}>
                  <span style={{ width: 5, height: 5, borderRadius: 999, background: nb.color, flexShrink: 0 }} />
                  <span style={{ fontSize: 12, color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{nb.label}</span>
                  <span className="mono" style={{ fontSize: 11.5, color: 'var(--text-muted)', marginLeft: 'auto', flexShrink: 0 }}>{nb.degree}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
