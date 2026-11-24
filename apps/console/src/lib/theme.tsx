// Theme system — light default, dark first-class. Sets data-theme on <html>, persists
// the choice, and honors the OS preference only when the user hasn't chosen yet.
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'

export type Theme = 'light' | 'dark'
const KEY = 'rag.theme'

interface ThemeCtx {
  theme: Theme
  toggle: () => void
  setTheme: (t: Theme) => void
}
const Ctx = createContext<ThemeCtx | null>(null)

function initial(): Theme {
  const stored = localStorage.getItem(KEY)
  if (stored === 'light' || stored === 'dark') return stored
  // Default is light (institutional daytime); adopt OS dark only if explicitly set.
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(initial)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  const setTheme = useCallback((t: Theme) => {
    localStorage.setItem(KEY, t)
    setThemeState(t)
  }, [])
  const toggle = useCallback(() => setTheme(theme === 'light' ? 'dark' : 'light'), [theme, setTheme])

  const value = useMemo(() => ({ theme, toggle, setTheme }), [theme, toggle, setTheme])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useTheme(): ThemeCtx {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useTheme must be used within ThemeProvider')
  return ctx
}
