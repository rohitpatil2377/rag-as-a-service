import { useEffect, useRef, useState } from 'react'
import { Icon } from './icons'
import { Skeleton } from './ui'
import { ApiError, api, type DocumentContent } from '../lib/api'
import type { ViewerTarget } from '../pages/Ask'

// In-document citation viewer (FR-I3). A slide-over that reconstructs the source document
// from its chunks (in reading order) and highlights the cited chunk — and, within it, the
// exact supporting quote — so a reader can verify an answer against its source in context.
//
// The same component also just READS a document, opened from the documents list with no citation
// (empty chunkId). That mode is not a highlight with nothing highlighted: with no cited span to
// draw the eye, every chunk is the content, so the page is rendered at full contrast rather than
// as muted context, and the footer drops the sentence about a highlight that is not there.
export function CitationViewer({ token, kbId, target, onClose }: {
  token: string; kbId: string; target: ViewerTarget; onClose: () => void
}) {
  const [doc, setDoc] = useState<DocumentContent | null>(null)
  const [error, setError] = useState<string | null>(null)
  const cited = useRef<HTMLDivElement>(null)
  // No citation to anchor on: opened to READ the document, not to verify an answer against it.
  const reading = !target.chunkId

  useEffect(() => {
    let live = true
    setDoc(null); setError(null)
    api.documentContent(token, kbId, target.docId)
      .then((d) => { if (live) setDoc(d) })
      .catch((e) => { if (live) setError(e instanceof ApiError ? e.message : 'Could not load the source document') })
    return () => { live = false }
  }, [token, kbId, target.docId])

  // Scroll the cited span into view once the document is rendered.
  useEffect(() => { if (doc) cited.current?.scrollIntoView({ block: 'center', behavior: 'smooth' }) }, [doc])

  // Close on Escape.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div className="fixed inset-0 z-50 flex justify-end" role="dialog" aria-modal="true" aria-label={`Source: ${target.docId}`}>
      <div onClick={onClose} className="absolute inset-0" style={{ background: 'color-mix(in srgb, var(--ink, #0D1B3A) 55%, transparent)', backdropFilter: 'blur(2px)' }} />
      <aside className="relative h-full flex flex-col shadow-2xl" style={{ width: 'min(560px, 94vw)', background: 'var(--surface)', borderLeft: '1px solid var(--border)' }}>
        <header className="flex items-start justify-between gap-3 p-5" style={{ borderBottom: '1px solid var(--border)' }}>
          <div className="min-w-0">
            <div className="flex items-center gap-2 mb-1"><Icon.book size={16} style={{ color: 'var(--accent)' }} /><span className="eyebrow">{reading ? 'Document' : 'Source document'}</span></div>
            <h2 className="font-display font-semibold truncate" style={{ fontSize: 17, color: 'var(--text-heading)' }}>{target.docId}</h2>
            {target.page != null && <span className="mono text-xs" style={{ color: 'var(--text-muted)' }}>cited page {target.page}</span>}
          </div>
          <button type="button" onClick={onClose} aria-label="Close" className="btn btn-subtle btn-icon"><Icon.x size={18} /></button>
        </header>

        <div className="flex-grow overflow-y-auto p-5">
          {!doc && !error && <div className="space-y-3">{[0, 1, 2, 3, 4].map((i) => <Skeleton key={i} style={{ height: i === 2 ? 60 : 22 }} />)}</div>}
          {error && <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--danger)' }}><Icon.alert size={16} />{error}</div>}
          {doc && (
            <div className="space-y-3" style={{ fontSize: 14.5, lineHeight: 1.7 }}>
              {doc.chunks.map((ch) => {
                const isCited = ch.chunk_id === target.chunkId
                return (
                  <div key={ch.chunk_id} ref={isCited ? cited : undefined}
                    className="rounded-lg" style={isCited
                      ? { background: 'color-mix(in srgb, var(--accent) 14%, transparent)', border: '1px solid color-mix(in srgb, var(--accent) 45%, transparent)', padding: '10px 12px' }
                      : { padding: '2px 12px', color: reading ? undefined : 'var(--text-muted)' }}>
                    {ch.page != null && <div className="mono text-xs mb-1" style={{ color: 'var(--text-muted)', opacity: 0.7 }}>p.{ch.page}</div>}
                    <p style={{ color: isCited || reading ? 'var(--text-primary)' : undefined, whiteSpace: 'pre-wrap' }}>
                      {isCited ? highlightQuote(ch.text, target.quote) : ch.text}
                    </p>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        <footer className="p-4 text-xs" style={{ borderTop: '1px solid var(--border)', color: 'var(--text-muted)' }}>
          {reading
            ? `Reconstructed from ${doc ? doc.chunks.length : 0} indexed chunk${doc && doc.chunks.length === 1 ? '' : 's'} · this is the text retrieval can see, not a render of the original file.`
            : 'Reconstructed from indexed chunks · the highlighted span is what grounded the answer.'}
        </footer>
      </aside>
    </div>
  )
}

// Highlight the supporting quote inside the cited chunk. Falls back to the whole chunk
// when the quote can't be located verbatim (e.g. after de-hyphenation/normalisation).
function highlightQuote(text: string, quote: string) {
  const q = quote.trim()
  const at = q ? text.toLowerCase().indexOf(q.toLowerCase()) : -1
  if (at < 0) return text
  return (
    <>
      {text.slice(0, at)}
      <mark style={{ background: 'color-mix(in srgb, var(--action) 32%, transparent)', color: 'var(--text-heading)', borderRadius: 3, padding: '0 2px' }}>
        {text.slice(at, at + q.length)}
      </mark>
      {text.slice(at + q.length)}
    </>
  )
}
