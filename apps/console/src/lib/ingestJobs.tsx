// Global ingest-job tracker. Ingestion is a resilient SERVER-SIDE background job; the only
// thing lost on navigation was the *view* of it, because the Uploader held job state locally.
// This context lives ABOVE the router, so staging + processing + polling survive navigating
// between pages, and (via localStorage) a full browser refresh. The topbar IngestIndicator and
// the KB Uploader both read from here, so progress is visible everywhere and rehydrates on return.
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { ApiError, api } from './api'
import { useAuth } from './auth'
import { useKb } from './kb'
import { useToast } from '../components/overlay'

export type IngestPhase = 'staging' | 'processing' | 'done' | 'failed'

export interface TrackedIngest {
  jobId: string
  kbId: string
  kbName: string
  phase: IngestPhase
  staged: number // files staged so far (staging phase)
  total: number
  done: number
  failed: number
  status: string
  error: string | null
  failures: { doc_id: string; error: string }[]
}

interface IngestJobsCtx {
  jobs: TrackedIngest[]
  active: TrackedIngest[] // staging | processing
  start: (kbId: string, files: File[], strategy: 'clause' | 'fixed', piiAutoclassify?: boolean, classification?: number) => Promise<void>
  retry: (jobId: string) => Promise<void>
  cancel: (jobId: string) => Promise<void>
  dismiss: (jobId: string) => void
  jobForKb: (kbId: string) => TrackedIngest | undefined
}

const Ctx = createContext<IngestJobsCtx | null>(null)
const KEY = 'rag.ingest.active'
// Files larger than this upload as raw binary (streamed server-side to the object store) instead
// of base64-in-JSON — no ~1.37x bloat, no whole-file-in-memory. 100 MB keeps the common small-file
// path on the simple base64 route while routing big uploads (incl. >1 GB) to the streaming one.
const STREAM_THRESHOLD = 100 * 1024 * 1024

type Persisted = { jobId: string; kbId: string; kbName: string }
const loadPersisted = (): Persisted[] => {
  try { return JSON.parse(localStorage.getItem(KEY) ?? '[]') as Persisted[] } catch { return [] }
}
const savePersisted = (list: Persisted[]) => localStorage.setItem(KEY, JSON.stringify(list))
const addPersisted = (p: Persisted) => savePersisted([...loadPersisted().filter((x) => x.jobId !== p.jobId), p])
const removePersisted = (jobId: string) => savePersisted(loadPersisted().filter((x) => x.jobId !== jobId))

function toBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader()
    r.onload = () => resolve((r.result as string).split(',', 2)[1] ?? '')
    r.onerror = reject
    r.readAsDataURL(file)
  })
}
const MIME: Record<string, string> = {
  pdf: 'application/pdf', txt: 'text/plain', md: 'text/markdown', html: 'text/html',
  docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  // Images route to the OCR/vision path (Chandra) when multimodal is enabled.
  png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', tiff: 'image/tiff', webp: 'image/webp',
}
const mimeOf = (name: string) => MIME[name.split('.').pop()?.toLowerCase() ?? ''] ?? 'application/octet-stream'

export function IngestJobsProvider({ children }: { children: ReactNode }) {
  const { session } = useAuth()
  const { kbs } = useKb()
  const toast = useToast()
  const [jobs, setJobs] = useState<TrackedIngest[]>([])
  const jobsRef = useRef(jobs)
  jobsRef.current = jobs

  const nameOf = useCallback((id: string) => kbs.find((k) => k.kb_id === id)?.name ?? id, [kbs])

  // Rehydrate any jobs that were running when the tab was closed/refreshed; polling resumes below.
  useEffect(() => {
    const persisted = loadPersisted()
    if (!persisted.length) return
    setJobs((prev) => {
      const have = new Set(prev.map((j) => j.jobId))
      const extra = persisted.filter((p) => !have.has(p.jobId)).map<TrackedIngest>((p) => ({
        jobId: p.jobId, kbId: p.kbId, kbName: p.kbName, phase: 'processing',
        staged: 0, total: 0, done: 0, failed: 0, status: 'running', error: null, failures: [],
      }))
      return extra.length ? [...prev, ...extra] : prev
    })
  }, [])

  // Single poller for every processing job, independent of which page is mounted.
  useEffect(() => {
    if (!session) return
    const t = setInterval(() => {
      const processing = jobsRef.current.filter((j) => j.phase === 'processing')
      processing.forEach(async (j) => {
        try {
          const v = await api.getIngestJob(session.token, j.kbId, j.jobId)
          const terminal = ['completed', 'partial', 'failed', 'cancelled'].includes(v.job.status)
          const phase: IngestPhase = terminal ? (v.job.done === 0 && v.job.failed > 0 ? 'failed' : 'done') : 'processing'
          setJobs((prev) => prev.map((x) => x.jobId === j.jobId
            ? { ...x, total: v.job.total, done: v.job.done, failed: v.job.failed, status: v.job.status, error: v.job.error, failures: v.failures, phase }
            : x))
          if (terminal) {
            removePersisted(j.jobId)
            toast.push(v.job.failed ? 'info' : 'success',
              `${j.kbName}: ingest ${v.job.status} — ${v.job.done} done${v.job.failed ? `, ${v.job.failed} failed` : ''}.`)
            // Keep the finished tile briefly, then self-clear (failed jobs linger for the retry action).
            if (!v.job.failed) window.setTimeout(() => setJobs((prev) => prev.filter((x) => x.jobId !== j.jobId)), 8000)
          }
        } catch (e) {
          if (e instanceof ApiError && e.status === 404) { // job gone server-side — stop tracking it
            removePersisted(j.jobId)
            setJobs((prev) => prev.filter((x) => x.jobId !== j.jobId))
          }
        }
      })
    }, 1500)
    return () => clearInterval(t)
  }, [session, toast])

  const start = useCallback(async (kbId: string, files: File[], strategy: 'clause' | 'fixed', piiAutoclassify?: boolean, classification?: number) => {
    if (!session || files.length === 0) return
    const kbName = nameOf(kbId)
    const tmpId = `staging-${kbId}-${files.length}-${new Date().getTime()}`
    setJobs((prev) => [...prev, {
      jobId: tmpId, kbId, kbName, phase: 'staging', staged: 0, total: files.length,
      done: 0, failed: 0, status: 'building', error: null, failures: [],
    }])
    try {
      const created = await api.createIngestJob(session.token, kbId, piiAutoclassify, classification)
      for (let i = 0; i < files.length; i++) {
        const f = files[i]
        // Large files go up as RAW BINARY (streamed server-side straight to the object store) to
        // avoid the ~1.37x base64 bloat and holding the whole file in memory; small files keep the
        // simple base64 path. Either way they stage into the SAME job and ingest by reference.
        if (f.size > STREAM_THRESHOLD) {
          await api.addIngestJobItemFile(session.token, kbId, created.job_id, f, { mime: mimeOf(f.name), strategy })
        } else {
          const content_b64 = await toBase64(f)
          await api.addIngestJobItem(session.token, kbId, created.job_id, { doc_id: f.name, content_b64, mime: mimeOf(f.name), strategy })
        }
        setJobs((prev) => prev.map((x) => (x.jobId === tmpId ? { ...x, staged: i + 1 } : x)))
      }
      const started = await api.startIngestJob(session.token, kbId, created.job_id)
      setJobs((prev) => prev.map((x) => (x.jobId === tmpId
        ? { ...x, jobId: started.job_id, phase: 'processing', total: started.total, status: started.status } : x)))
      addPersisted({ jobId: started.job_id, kbId, kbName })
      toast.push('info', `${kbName}: queued ${started.total} document(s) — ingesting in the background.`)
    } catch (e) {
      setJobs((prev) => prev.filter((x) => x.jobId !== tmpId))
      toast.push('error', e instanceof ApiError ? e.message : 'Could not start ingest.')
    }
  }, [session, nameOf, toast])

  const retry = useCallback(async (jobId: string) => {
    if (!session) return
    const tj = jobsRef.current.find((x) => x.jobId === jobId)
    if (!tj) return
    try {
      const j = await api.retryIngestJob(session.token, tj.kbId, jobId)
      setJobs((prev) => prev.map((x) => (x.jobId === jobId
        ? { ...x, phase: 'processing', status: j.status, done: j.done, failed: j.failed, total: j.total } : x)))
      addPersisted({ jobId, kbId: tj.kbId, kbName: tj.kbName })
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Retry failed.')
    }
  }, [session, toast])

  const cancel = useCallback(async (jobId: string) => {
    if (!session) return
    const tj = jobsRef.current.find((x) => x.jobId === jobId)
    if (!tj) return
    try {
      await api.cancelIngestJob(session.token, tj.kbId, jobId)
      // In-flight items finish, the rest are skipped; the poller reflects the terminal state.
      toast.push('info', `${tj.kbName}: cancelling ingestion — in-flight documents finish, the rest are skipped.`)
    } catch (e) {
      toast.push('error', e instanceof ApiError ? e.message : 'Cancel failed.')
    }
  }, [session, toast])

  const dismiss = useCallback((jobId: string) => {
    removePersisted(jobId)
    setJobs((prev) => prev.filter((x) => x.jobId !== jobId))
  }, [])

  const value = useMemo<IngestJobsCtx>(() => ({
    jobs,
    active: jobs.filter((j) => j.phase === 'staging' || j.phase === 'processing'),
    start, retry, cancel, dismiss,
    jobForKb: (kbId: string) => { const m = jobs.filter((x) => x.kbId === kbId); return m[m.length - 1] },
  }), [jobs, start, retry, cancel, dismiss])

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useIngestJobs(): IngestJobsCtx {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useIngestJobs must be used within IngestJobsProvider')
  return ctx
}
