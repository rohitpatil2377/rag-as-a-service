// Minimal, dependency-free i18n (same shape as react-i18next: a `t(key)` lookup over
// per-locale dictionaries with English fallback). Migrating to react-i18next later is
// mechanical — the keys and call sites don't change.
import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'
import { en } from '../locales/en'
import { hi } from '../locales/hi'

export type Lang = 'en' | 'hi'

export const LANGS: { value: Lang; label: string }[] = [
  { value: 'en', label: 'English' },
  { value: 'hi', label: 'हिन्दी' },
]

const DICTS: Record<Lang, Record<string, string>> = { en, hi }
const KEY = 'rag.lang'

type Vars = Record<string, string | number>
interface I18nCtx {
  lang: Lang
  setLang: (l: Lang) => void
  t: (key: string, fallbackOrVars?: string | Vars, vars?: Vars) => string
}

const Ctx = createContext<I18nCtx | null>(null)

function interpolate(s: string, vars?: Vars): string {
  if (!vars) return s
  return s.replace(/\{(\w+)\}/g, (_, k) => (k in vars ? String(vars[k]) : `{${k}}`))
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(() => (localStorage.getItem(KEY) as Lang) || 'en')

  const setLang = useCallback((l: Lang) => {
    setLangState(l)
    localStorage.setItem(KEY, l)
    document.documentElement.lang = l
  }, [])

  const t = useCallback(
    (key: string, fallbackOrVars?: string | Vars, vars?: Vars): string => {
      const fallback = typeof fallbackOrVars === 'string' ? fallbackOrVars : undefined
      const v = typeof fallbackOrVars === 'object' ? fallbackOrVars : vars
      const raw = DICTS[lang][key] ?? en[key] ?? fallback ?? key
      return interpolate(raw, v)
    },
    [lang],
  )

  const value = useMemo(() => ({ lang, setLang, t }), [lang, setLang, t])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useI18n(): I18nCtx {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useI18n must be used within I18nProvider')
  return ctx
}
