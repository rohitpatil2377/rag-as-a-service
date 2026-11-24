// Knowledge-base context. An org (tenant) owns many KBs (HR, Finance, …); every
// KB-scoped screen operates on the *selected* KB, not the raw tenant id.
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { api, type KbInfo } from './api'
import { useAuth } from './auth'

interface KbCtx {
  kbs: KbInfo[]
  kbId: string           // the selected KB (namespace) — use this for /v1/kb/{kbId}/… calls
  setKbId: (id: string) => void
  refresh: () => Promise<void>
  create: (name: string) => Promise<KbInfo>
  activeModel: { label: string } | null   // the per-question model picked on Ask; topbar mirrors it
  setActiveModel: (m: { label: string } | null) => void
}

const Ctx = createContext<KbCtx | null>(null)
const KEY = 'rag.kb'

export function KbProvider({ children }: { children: ReactNode }) {
  const { session } = useAuth()
  const [kbs, setKbs] = useState<KbInfo[]>([])
  const [kbId, setKbIdState] = useState<string>(() => localStorage.getItem(KEY) ?? '')
  const [activeModel, setActiveModel] = useState<{ label: string } | null>(null)

  const setKbId = useCallback((id: string) => {
    localStorage.setItem(KEY, id)
    setKbIdState(id)
    setActiveModel(null)   // a different KB has its own default model
  }, [])

  const refresh = useCallback(async () => {
    if (!session) return
    const list = await api.listKbs(session.token, session.tenantId).catch(() => [] as KbInfo[])
    setKbs(list)
    // Keep the stored selection if it still exists, else the first KB, else none (empty org).
    setKbIdState((cur) => {
      if (cur && list.some((k) => k.kb_id === cur)) return cur
      const next = list[0]?.kb_id ?? ''
      localStorage.setItem(KEY, next)
      return next
    })
  }, [session])

  useEffect(() => { void refresh() }, [refresh])

  const create = useCallback(async (name: string) => {
    if (!session) throw new Error('not signed in')
    const kb = await api.createKb(session.token, name)
    await refresh()
    setKbId(kb.kb_id)
    return kb
  }, [session, refresh, setKbId])

  // The effective selection: the stored one if still valid, else the first KB, else empty.
  const effectiveKbId = (kbs.some((k) => k.kb_id === kbId) ? kbId : kbs[0]?.kb_id) ?? ''
  const value = useMemo(() => ({ kbs, kbId: effectiveKbId, setKbId, refresh, create, activeModel, setActiveModel }),
    [kbs, effectiveKbId, setKbId, refresh, create, activeModel])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useKb(): KbCtx {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useKb must be used within KbProvider')
  return ctx
}
