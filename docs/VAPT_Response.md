# RAG as a Service — VAPT Response & Security Hardening

**Scan:** Acme WebScanner (automated) · target `http://192.0.2.14/` · 2026-07-13
**Result triaged:** 63 findings — 0 CRITICAL, 26 HIGH, 8 MEDIUM, 21 LOW, 8 INFO.

This document triages every finding class, states the disposition, and records the fix.
The overwhelming majority collapse into **three** actions: add HTTP security headers,
hide version banners, and harden the deployment (close infra ports, force `prod` mode,
terminate TLS). Two HIGH categories (20 of the 26 HIGH) are scanner **false positives**
explained below.

---

## 1. Disposition summary

| Finding class | Count | Severity | Disposition | Where fixed |
|---|---|---|---|---|
| Missing security headers (HSTS, CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, Permissions-Policy) | 42 | HIGH | **Fixed in code** | API middleware + console nginx |
| `Server:` version disclosure (`uvicorn`, `nginx/1.31.2`) | — | LOW/INFO | **Fixed in code** | uvicorn `server_header=False`, generic banner, `server_tokens off` |
| Exposed internal ports (PostgreSQL 5432, OpenSearch/"Elasticsearch" 9200, MinIO 9000/9001, Prometheus 9090, Grafana/"Dev Server" 3000, TEI/"HTTP-Alt" 8080, SSH 22, HTTP 80) | 8 | HIGH/MED/LOW | **Fixed by deployment** | `RAG_INFRA_BIND=127.0.0.1` |
| No TLS / HTTP only / HSTS absent | 1 | INFO | **Deployment** | TLS at ingress; app now emits HSTS |
| "Excessive Data Exposure `['bearer']`" | 10 | HIGH | **False positive** | see §3 |
| "Open Redirect `/v1?q=//evil.com`" | 1 | HIGH | **False positive** | see §3 |
| `/v1/dev/token` reachable | — | — | **Already gated in code**; deployment must set `RAG_ENV=prod` | see §4 |
| SSL/TLS, Technology & CVE (informational) | 2 | INFO | Informational | — |

---

## 2. What changed in the code

### 2.1 HTTP security headers on every API response
`libs/common/src/rag_common/middleware.py` now installs a `_security_headers_middleware`
that stamps, on **every** response (success *and* error):

- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `Content-Security-Policy: default-src 'none'; frame-ancestors 'none'; base-uri 'none'`
  — the API serves **JSON only**, so the strictest possible CSP is correct.
- `X-Frame-Options: DENY`
- `X-Content-Type-Options: nosniff`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy: geolocation=(), microphone=(), camera=(), payment=()`
- `Server: RagaaS` (generic — replaces the version-disclosing default)

All values are configurable via `Settings` (`security_headers_enabled`, `hsts_max_age`,
`csp_api`, `permissions_policy`, `server_banner`). uvicorn is launched with
`server_header=False` / `date_header=False` so no version banner leaves the process.

### 2.2 Console (SPA) security headers
`apps/console/nginx.conf` adds `server_tokens off;` and the same header set on the static
SPA, with a CSP appropriate for an HTML app: `default-src 'self'`, `script-src 'self'`
(the Vite build emits no inline scripts), `style-src 'self' 'unsafe-inline'` (Tailwind +
React style props), Google Fonts origins allowlisted, `frame-ancestors 'none'`,
`object-src 'none'`. The proxied `/v1`, `/health`, `/mcp` responses inherit the API
middleware's headers, so there is no duplication.

### 2.3 Deployment: close the internal ports (`RAG_INFRA_BIND`)
Every internal service port in `infra/compose/docker-compose.dev.yml` is now published
through `${RAG_INFRA_BIND:-0.0.0.0}`. Dev is unchanged (default `0.0.0.0`). Production
sets `RAG_INFRA_BIND=127.0.0.1` (see `infra/compose/.env.prod.example`) so PostgreSQL,
OpenSearch, MinIO, Prometheus, Grafana, TEI, llama, OTel and the API bind to **localhost
only** — reachable via SSH tunnel for admin, invisible to a network port scan. Only the
console (`:80`) stays public. This closes the 8 port-scanner findings — the most serious
*real* exposure in the report (an unauthenticated PostgreSQL/OpenSearch on the network).

---

## 3. False positives (explained, not "fixed")

**"Excessive Data Exposure — sensitive field `['bearer']`" (10 × HIGH).**
The scanner flagged the literal word *bearer* in the body
`{"code":"unauthorized","message":"missing bearer token","details":{}}`. That is the
API **correctly rejecting an unauthenticated request** (HTTP 401) on `/v1/knowledge-bases/trash`,
`/v1/agent/mcp`, etc. No data is exposed — it is an auth challenge. The word "bearer"
in an error string is not a leaked field. **No vulnerability.** (Endpoints that require
auth return 401; this is the intended, tested behaviour.)

**"Open Redirect — `/v1?q=//evil.com`" (1 × HIGH).**
The evidence header is `Location: http://192.0.2.14/v1/?q=//evil.com` — the redirect
target host is the **server's own host**, not `evil.com`. This is Starlette's automatic
trailing-slash redirect (`/v1` → `/v1/`) echoing the original query string. The
application never redirects to a user-supplied external host. **Not exploitable as an
open redirect.** (Verified locally: `/v1?q=…` produces only same-origin responses.)

---

## 4. `/v1/dev/token` — already gated

The report reached `/v1/dev/token` because the scanned box was running in **dev** mode.
The code already disables it in production:

```python
# services/control-plane/.../api.py
if settings.env == "prod" or settings.auth_mode != "dev_jwt":
    raise NotFoundError("endpoint not available")   # 404
```

**Action:** deploy with `RAG_ENV=prod` (and, for real SSO/MFA, `RAG_AUTH_MODE=oidc`).
Covered by `.env.prod.example`.

---

## 5. Production go-live checklist (security)

1. `cp infra/compose/.env.prod.example infra/compose/.env` and set `RAG_INFRA_BIND=127.0.0.1`, `RAG_ENV=prod`.
2. Rotate every dev default credential (Postgres, MinIO, Grafana, `RAG_MASTER_KEY`) via the SecretsProvider — never the compose defaults.
3. Terminate **TLS 1.2/1.3** at the ingress in front of the console; redirect 80 → 443. HSTS is already emitted.
4. Enable OpenSearch security (`DISABLE_SECURITY_PLUGIN=true` is **dev-only**) and MinIO/Grafana auth.
5. Re-run the scan → the 42 header findings and 8 port findings should clear; the "bearer"/redirect items are documented false positives.

---

## 6. Follow-ups (tracked, not blocking)

- **Self-host the web fonts** (Inter/Sora/JetBrains Mono) instead of Google Fonts, to
  drop the browser-side external origins from the console CSP and fully honour the
  sovereignty posture. Then tighten `style-src`/`font-src` to `'self'`.
- Optionally reword the 401 body so naive scanners stop matching "bearer" (cosmetic).
- Wire the security-header values into the runtime feature-flag admin surface.
