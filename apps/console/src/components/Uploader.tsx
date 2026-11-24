// Unified document uploader — drag-and-drop OR browse, one file or thousands.
// The upload lifecycle (staging + background ingest) is owned by the global IngestJobsProvider,
// so switching tabs or refreshing never loses progress — this component just collects files and
// renders the shared job state for the current KB (which rehydrates when you return here).
import { useEffect, useRef, useState } from 'react'
import { useKb } from '../lib/kb'
import { useIngestJobs } from '../lib/ingestJobs'
import { Icon } from './icons'
import { Button, SegmentedControl } from './ui'

export function Uploader({ onIngested }: { onIngested?: (docId?: string, done?: boolean) => void }) {
  const { kbId } = useKb()
  const ingest = useIngestJobs()
  const [files, setFiles] = useState<File[]>([])
  const [strategy, setStrategy] = useState<'clause' | 'fixed'>('clause')
  // Per-upload data classification (0=Public..3=Restricted). Applied to every doc in the batch and
  // capped server-side at the ingestor's clearance (you can't publish above what you may read).
  const [classification, setClassification] = useState<'0' | '1' | '2' | '3'>('0')
  // Auto-raise a document's classification when detected PII (Aadhaar/PAN/card…) implies a higher
  // level — so a KYC doc can't land as Public. On by default; the uploader can opt out per batch.
  const [piiAuto, setPiiAuto] = useState(true)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const job = ingest.jobForKb(kbId)
  const staging = job?.phase === 'staging'
  const processing = job?.phase === 'processing'
  const finished = job && (job.phase === 'done' || job.phase === 'failed') ? job : null
  const showDropzone = !staging && !processing

  // Refresh the documents list live — each time a doc finishes (done increments) and on
  // completion — so the table fills in alongside the progress bar instead of only at the end.
  const prevDone = useRef(0)
  const prevPhase = useRef<string>('')
  // The newest doc id of the batch we just started — handed to the parent so the pipeline traces
  // the document ACTUALLY being ingested (not the newest already-listed one), and only while it is
  // ingesting: `done` tells the parent to hide the pipeline once the run finishes.
  const lastDocId = useRef<string>('')
  useEffect(() => {
    if (!job) return
    // Fire only on a real transition (start → processing, each doc done, terminal) — not on every
    // render (the parent's onIngested is a fresh closure each time, which is in the dep array).
    if (job.done === prevDone.current && job.phase === prevPhase.current) return
    prevDone.current = job.done
    prevPhase.current = job.phase
    const terminal = job.phase === 'done' || job.phase === 'failed'
    onIngested?.(lastDocId.current || undefined, terminal)
  }, [job?.done, job?.phase, onIngested])

  function addFiles(list: FileList | null) {
    if (!list) return
    const incoming = Array.from(list)
    setFiles((prev) => {
      const seen = new Set(prev.map((f) => f.name + f.size))
      return [...prev, ...incoming.filter((f) => !seen.has(f.name + f.size))]
    })
  }

  async function run() {
    if (files.length === 0) return
    const toIngest = files
    lastDocId.current = toIngest[toIngest.length - 1]?.name ?? ''
    setFiles([])
    await ingest.start(kbId, toIngest, strategy, piiAuto, Number(classification))
  }

  const pct = job && job.total ? Math.round(((job.done + job.failed) / job.total) * 100) : 0

  return (
    <div>
      {showDropzone && (
        <>
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); addFiles(e.dataTransfer.files) }}
            onClick={() => inputRef.current?.click()}
            role="button" tabIndex={0} onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
            className="flex flex-col items-center justify-center text-center rounded-xl transition-colors"
            style={{ padding: '28px 18px', cursor: 'pointer', border: `2px dashed ${dragOver ? 'var(--action)' : 'var(--border-strong)'}`, background: dragOver ? 'color-mix(in srgb, var(--action) 6%, transparent)' : 'transparent' }}>
            <div className="mb-2" style={{ color: 'var(--text-muted)' }}><Icon.upload size={22} /></div>
            <div className="font-semibold text-sm" style={{ color: 'var(--text-heading)' }}>Drag &amp; drop files here</div>
            <div className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
              or <span style={{ color: 'var(--action)' }}>browse</span> — PDF, DOCX, TXT, MD, HTML, PNG/JPG (OCR) · one file or thousands
            </div>
            <input ref={inputRef} type="file" multiple hidden
              accept=".pdf,.txt,.md,.docx,.html,.png,.jpg,.jpeg,.tiff,.webp,text/plain,text/markdown,text/html,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,image/png,image/jpeg,image/tiff,image/webp"
              onChange={(e) => { addFiles(e.target.files); if (inputRef.current) inputRef.current.value = '' }} />
          </div>

          {files.length > 0 && (
            <>
              <ul className="mt-3 space-y-1.5 max-h-52 overflow-y-auto pr-1">
                {files.map((f, i) => (
                  <li key={f.name + f.size} className="surface-2 flex items-center gap-3 px-3 py-2 rounded-lg">
                    <Icon.doc size={15} style={{ color: 'var(--text-muted)' }} />
                    <div className="flex-grow min-w-0">
                      <div className="text-sm font-medium truncate" style={{ color: 'var(--text-heading)' }}>{f.name}</div>
                      <div className="text-xs" style={{ color: 'var(--text-muted)' }}>{(f.size / 1024).toFixed(0)} KB</div>
                    </div>
                    <button type="button" aria-label={`Remove ${f.name}`} className="btn btn-subtle btn-icon"
                      onClick={() => setFiles((prev) => prev.filter((_, idx) => idx !== i))}><Icon.x size={14} /></button>
                  </li>
                ))}
              </ul>
              <div className="flex items-center gap-3 mt-3 flex-wrap">
                <span className="field-label">Chunking</span>
                <SegmentedControl value={strategy} onChange={setStrategy} options={[{ value: 'clause', label: 'Clause' }, { value: 'fixed', label: 'Fixed' }]} />
                <span className="field-label">Classification</span>
                <SegmentedControl value={classification} onChange={setClassification}
                  options={[{ value: '0', label: 'Public' }, { value: '1', label: 'Internal' }, { value: '2', label: 'Confidential' }, { value: '3', label: 'Restricted' }]} />
                <label className="flex items-center gap-1.5 text-xs select-none" style={{ color: 'var(--text-muted)', cursor: 'pointer' }}
                  title="If a document's detected PII (Aadhaar/PAN/card…) implies a higher data classification than requested, raise it automatically — so sensitive docs can't be published as Public.">
                  <input type="checkbox" checked={piiAuto} onChange={(e) => setPiiAuto(e.target.checked)} style={{ cursor: 'pointer' }} />
                  Auto-classify PII
                </label>
                <div className="flex-grow" />
                <Button size="sm" variant="subtle" onClick={() => setFiles([])}>Clear</Button>
                <Button size="sm" onClick={run} icon={<Icon.upload size={15} />}>Ingest {files.length} file{files.length === 1 ? '' : 's'}</Button>
              </div>
              <div className="text-xs mt-1.5" style={{ color: 'var(--text-muted)' }}>
                Classification is capped at your clearance — you can't ingest documents more sensitive than you're cleared to read.
              </div>
            </>
          )}
          {finished && (
            <div className="rounded-lg p-3 mt-3" style={{ background: 'var(--surface-2)' }}>
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold" style={{ color: 'var(--text-heading)' }}>Job {finished.status} — {finished.done} ingested{finished.failed ? `, ${finished.failed} failed` : ''}</span>
                {finished.failed > 0 && <Button size="sm" variant="subtle" onClick={() => ingest.retry(finished.jobId)} icon={<Icon.refresh size={14} />}>Retry failed</Button>}
              </div>
              {finished.failures.length > 0 && (
                <ul className="mt-2 space-y-0.5">
                  {finished.failures.map((f, i) => <li key={i} className="text-xs" style={{ color: 'var(--text-muted)' }}>• <span className="mono">{f.doc_id}</span>: {f.error}</li>)}
                </ul>
              )}
            </div>
          )}
        </>
      )}

      {staging && job && (
        <div className="rounded-lg p-3" style={{ background: 'var(--surface-2)' }}>
          <div className="text-sm mb-2" style={{ color: 'var(--text-heading)' }}>Uploading files… {job.staged}/{job.total}</div>
          <Bar pct={job.total ? Math.round((job.staged / job.total) * 100) : 0} />
          <div className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>You can switch tabs — this keeps running.</div>
        </div>
      )}

      {processing && job && (
        <div className="rounded-lg p-3" style={{ background: 'var(--surface-2)' }}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-semibold" style={{ color: 'var(--text-heading)' }}>Ingesting… {job.done + job.failed}/{job.total}</span>
            <span className="pill" style={{ fontSize: 11 }}>{job.status}</span>
          </div>
          <Bar pct={pct} />
          <div className="flex gap-2 mt-2 text-xs">
            <span className="chip" style={{ color: 'var(--success)' }}>{job.done} done</span>
            {job.failed > 0 && <span className="chip" style={{ color: 'var(--danger)' }}>{job.failed} failed</span>}
          </div>
          <div className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>Runs in the background — track it from the topbar on any page.</div>
        </div>
      )}
    </div>
  )
}

function Bar({ pct }: { pct: number }) {
  return (
    <div style={{ height: 8, borderRadius: 999, background: 'var(--border)', overflow: 'hidden' }}>
      <div style={{ width: `${pct}%`, height: '100%', background: 'var(--action)', transition: 'width .3s' }} />
    </div>
  )
}
