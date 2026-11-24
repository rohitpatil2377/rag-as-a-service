import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { App } from './App'
import { ToastProvider } from './components/overlay'
import { AuthProvider } from './lib/auth'
import { I18nProvider } from './lib/i18n'
import { IngestJobsProvider } from './lib/ingestJobs'
import { KbProvider } from './lib/kb'
import { ThemeProvider } from './lib/theme'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <I18nProvider>
      <ThemeProvider>
        <BrowserRouter>
          <AuthProvider>
            <ToastProvider>
              <KbProvider>
                <IngestJobsProvider>
                  <App />
                </IngestJobsProvider>
              </KbProvider>
            </ToastProvider>
          </AuthProvider>
        </BrowserRouter>
      </ThemeProvider>
    </I18nProvider>
  </StrictMode>,
)
