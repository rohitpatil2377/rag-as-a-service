import { Fragment, type ReactNode } from 'react'
import { MermaidBlock } from './Mermaid'

// Minimal, safe markdown renderer for model answers (no dependencies, no dangerouslySetInnerHTML).
// Handles the subset models actually emit: headings, bold, italic, inline code, ordered +
// bulleted lists, blockquotes, paragraphs, fenced code blocks (```lang ... ```) — plus citation
// markers like [1] rendered as badges. ```mermaid fences render as diagrams (see Mermaid.tsx);
// any other language renders as a plain monospace code block.

type Block =
  | { kind: 'h'; level: number; text: string }
  | { kind: 'ul'; items: { text: string; depth: number }[] }
  | { kind: 'ol'; items: { n: string; text: string; depth: number }[] }
  | { kind: 'quote'; text: string }
  | { kind: 'p'; text: string }
  | { kind: 'code'; lang: string; code: string }
  | { kind: 'table'; headers: string[]; rows: string[][] }

const H = /^(#{1,4})\s+(.*)$/
// A sub-point is indented under its parent. Two spaces is one level (what the answer prompt
// asks for); a tab or four spaces reads as one level too, since that is what people type.
// Capped: an answer nested deeper than two levels is a formatting accident, not structure.
// One marker per level. A sub-point that repeated the parent's filled disc read as a sibling;
// the shape of the marker is what says "this belongs to the line above".
const MARKERS = ['•', '◦', '▪']
const INDENT_PER_LEVEL = 2
const MAX_DEPTH = 2
function depthOf(lead: string): number {
  const spaces = lead.replace(/\t/g, '  ').length
  return Math.min(MAX_DEPTH, Math.floor(spaces / INDENT_PER_LEVEL))
}

const UL = /^(\s*)[-*•]\s+(.*)$/
const OL = /^(\s*)(\d+)[.)]\s+(.*)$/
const QUOTE = /^\s*>\s+(.*)$/
const FENCE = /^```\s*([a-zA-Z0-9_+-]*)\s*$/

function toBlocks(src: string): Block[] {
  const lines = src.replace(/\r\n/g, '\n').split('\n')
  const blocks: Block[] = []
  let para: string[] = []
  const flush = () => { if (para.length) { blocks.push({ kind: 'p', text: para.join(' ') }); para = [] } }

  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i]
    const line = raw.trimEnd()

    const fence = line.match(FENCE)
    if (fence) {
      flush()
      const lang = fence[1].toLowerCase()
      const codeLines: string[] = []
      i++
      while (i < lines.length && !FENCE.test(lines[i])) { codeLines.push(lines[i]); i++ }
      // i now sits on the closing fence (or ran past the end, if the model never closed it) —
      // the outer for-loop's i++ advances past it.
      blocks.push({ kind: 'code', lang, code: codeLines.join('\n') })
      continue
    }

    if (!line.trim()) { flush(); continue }
    let m: RegExpMatchArray | null
    if ((m = line.match(H))) { flush(); blocks.push({ kind: 'h', level: m[1].length, text: m[2] }); continue }
    if ((m = line.match(OL))) {
      flush()
      const last = blocks[blocks.length - 1]
      const item = { n: m[2], text: m[3], depth: depthOf(m[1]) }
      if (last?.kind === 'ol') last.items.push(item)
      else blocks.push({ kind: 'ol', items: [item] })
      continue
    }
    if ((m = line.match(UL))) {
      flush()
      const last = blocks[blocks.length - 1]
      const item = { text: m[2], depth: depthOf(m[1]) }
      if (last?.kind === 'ul') last.items.push(item)
      else blocks.push({ kind: 'ul', items: [item] })
      continue
    }
    if ((m = line.match(QUOTE))) { flush(); blocks.push({ kind: 'quote', text: m[1] }); continue }
    // GFM table: a header row of `| a | b |` followed by a `|---|---|` separator, then body rows.
    const sep = i + 1 < lines.length ? lines[i + 1].trim() : ''
    if (line.includes('|') && /-/.test(sep) && /^\|?[\s:|-]+\|?$/.test(sep)) {
      flush()
      const cells = (s: string) => s.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim())
      const headers = cells(line)
      const rows: string[][] = []
      let j = i + 2
      while (j < lines.length && lines[j].includes('|') && lines[j].trim()) { rows.push(cells(lines[j])); j++ }
      blocks.push({ kind: 'table', headers, rows })
      i = j - 1
      continue
    }
    para.push(line.trim())
  }
  flush()
  return blocks
}

// Inline: **bold**, *italic* / _italic_, `code`, and [1] citation markers.
function renderInline(text: string): ReactNode[] {
  const out: ReactNode[] = []
  const re = /(\*\*[^*]+\*\*|__[^_]+__|\*[^*]+\*|_[^_]+_|`[^`]+`|\[\d+(?:,\s*\d+)*\])/g
  let last = 0
  let key = 0
  let m: RegExpExecArray | null
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(<Fragment key={key++}>{text.slice(last, m.index)}</Fragment>)
    const tok = m[0]
    if (tok.startsWith('**') || tok.startsWith('__')) {
      out.push(<strong key={key++} style={{ color: 'var(--text-heading)', fontWeight: 700 }}>{tok.slice(2, -2)}</strong>)
    } else if (tok.startsWith('`')) {
      out.push(<code key={key++} className="mono" style={{ fontSize: '0.88em', background: 'color-mix(in srgb, var(--text-muted) 14%, transparent)', borderRadius: 4, padding: '1px 5px' }}>{tok.slice(1, -1)}</code>)
    } else if (tok.startsWith('[')) {
      out.push(<sup key={key++} className="mono" style={{ color: 'var(--success)', fontWeight: 700, fontSize: '0.72em', padding: '0 1px' }}>{tok}</sup>)
    } else {
      out.push(<em key={key++} style={{ color: 'var(--text-primary)' }}>{tok.slice(1, -1)}</em>)
    }
    last = m.index + tok.length
  }
  if (last < text.length) out.push(<Fragment key={key++}>{text.slice(last)}</Fragment>)
  return out
}

export function Markdown({ text }: { text: string }) {
  const blocks = toBlocks(text)
  return (
    <div className="md" style={{ fontSize: 15.5, lineHeight: 1.68, color: 'var(--text-primary)' }}>
      {blocks.map((b, i) => {
        if (b.kind === 'h') {
          const size = b.level <= 1 ? 20 : b.level === 2 ? 17 : 15
          return <div key={i} className="font-display font-semibold" style={{ fontSize: size, color: 'var(--text-heading)', margin: i ? '14px 0 6px' : '0 0 6px' }}>{renderInline(b.text)}</div>
        }
        if (b.kind === 'ul') {
          // Indentation and a different marker per level, so a sub-point reads as belonging to
          // the line above it rather than as another item of equal standing.
          return (
            <ul key={i} style={{ margin: '6px 0', paddingLeft: 22, listStyle: 'none' }}>
              {b.items.map((it, j) => (
                <li key={j} style={{ margin: '4px 0', paddingLeft: it.depth * 18, display: 'flex', gap: 8 }}>
                  <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{MARKERS[it.depth] ?? MARKERS[MARKERS.length - 1]}</span>
                  <span style={{ flex: 1 }}>{renderInline(it.text)}</span>
                </li>
              ))}
            </ul>
          )
        }
        if (b.kind === 'ol') {
          return (
            <ol key={i} style={{ margin: '6px 0', paddingLeft: 4, listStyle: 'none' }}>
              {b.items.map((it, j) => (
                <li key={j} style={{ margin: '6px 0', display: 'flex', gap: 10, paddingLeft: it.depth * 18 }}>
                  <span className="mono" style={{ color: it.depth ? 'var(--text-muted)' : 'var(--action)', fontWeight: it.depth ? 500 : 700, minWidth: 18, textAlign: 'right' }}>{it.n}.</span>
                  <span style={{ flex: 1 }}>{renderInline(it.text)}</span>
                </li>
              ))}
            </ol>
          )
        }
        if (b.kind === 'quote') {
          return <blockquote key={i} style={{ margin: '8px 0', paddingLeft: 12, borderLeft: '3px solid var(--accent)', color: 'var(--text-muted)', fontStyle: 'italic' }}>{renderInline(b.text)}</blockquote>
        }
        if (b.kind === 'code') {
          if (b.lang === 'mermaid') return <MermaidBlock key={i} code={b.code} />
          return (
            <pre
              key={i}
              className="mono"
              style={{
                margin: '8px 0', padding: '10px 14px', background: 'var(--surface-2)',
                border: '1px solid var(--border)', borderRadius: 'var(--radius-md)',
                overflowX: 'auto', fontSize: 13.5, color: 'var(--text-primary)', lineHeight: 1.55,
              }}
            >
              <code>{b.code}</code>
            </pre>
          )
        }
        if (b.kind === 'table') {
          return (
            <div key={i} className="surface" style={{ overflowX: 'auto', margin: '10px 0', borderRadius: 8 }}>
              <table className="tbl" style={{ minWidth: 'max-content' }}>
                <thead><tr>{b.headers.map((h, j) => <th key={j}>{renderInline(h)}</th>)}</tr></thead>
                <tbody>
                  {b.rows.map((r, j) => (
                    <tr key={j}>{r.map((c, k) => <td key={k} style={{ fontSize: 13.5 }}>{renderInline(c)}</td>)}</tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        }
        return <p key={i} style={{ margin: i ? '8px 0' : '0 0 8px' }}>{renderInline(b.text)}</p>
      })}
    </div>
  )
}
