# ragaas-console — Acme RAG as a Service console (Phase-1)

Sovereign, self-hostable **React + Vite + TS + Tailwind** SPA for the `/v1` API
(FR-H3). Consumes the shared **Acme design tokens** (`design/tokens`) via the Tailwind
preset, so the brand system (palette 50-30-12-6-2, Sora/Inter, spacing/radii, saffron
focus rings, light/dark) is inherited from one source of truth.

## Screens
- **Login** — dev JWT via `/v1/dev/token` (dev_jwt mode; OIDC SSO in GA).
- **Dashboard** — health + orientation.
- **Knowledge Base** — provision the tenant namespace (FR-H1).
- **Upload** — ingest → parse → PII-mask → chunk → embed → index (FR-A/B/C).
- **Ask** — grounded / abstained / **flagged-for-review** answer with citations +
  faithfulness score (FR-D/E, E4).
- **Review Queue** — HITL gate: accept / flag / override; unsupported claims shown (FR-F6/E4).
- **Audit Trail** — tamper-evident hash-chained events (FR-G4).

## Run (dev)
```bash
# from repo root, backend up (make up) or `uv run rag-api` on :8000
cd apps/console
npm install
npm run dev        # http://localhost:5173 (proxies /v1 + /health to :8000)
```

## Build (static, self-hosted)
```bash
npm run build      # tsc --noEmit && vite build  ->  dist/
```
`dist/` is static; in production it is served same-origin behind the ingress (no Node
server to operate — sovereign-friendly). Nothing calls a foreign service.

## Notes
- Auth token carries the tenant; the API enforces isolation server-side — the SPA never
  decides authorization.
- CORS for `:5173` is enabled on the API in dev only (`RAG_CORS_ALLOW_ORIGINS`).
