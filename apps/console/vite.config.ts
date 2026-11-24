import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev server proxies the API so the SPA calls it same-origin (also covered by CORS).
// In production the built static files are served behind the ingress, same-origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/v1': { target: 'http://localhost:8000', changeOrigin: true },
      '/health': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
  build: { outDir: 'dist', sourcemap: true },
})
