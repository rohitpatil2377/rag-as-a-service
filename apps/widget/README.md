# RAG as a Service — Embeddable Chat Widget (FR-I6)

A dependency-free, white-label chat bubble any tenant can drop into their own portal. It
answers **only** from a chosen knowledge base, with citations — inheriting every platform
control (grounding, guardrails, faithfulness, ACLs).

## Security model — the key never touches the browser

```
 host page  ──①──▶  YOUR server (holds a read-scoped API key ek_…)
    ▲                   │
    │                   ②  POST {RAGAAS}/v1/widget/session  (Bearer ek_…)  { kb_id }
    │                   ▼
    │              RAG as a Service ──▶ { token (15-min, reader, single-KB, public-ACL), config }
    └────③ returns { token, kb_id, config } ────┘
 widget ──④──▶  POST {RAGAAS}/v1/kb/{kb}/query  (Bearer token)
```

The **short-lived token** is bound to exactly one KB (`tenant_id == kb_id`), `reader`-only,
and public-ACL — so an anonymous embed can never ingest, reach another KB, or see
ACL-restricted content. The API key stays on your server.

## 1. Enable + theme the widget (once, as a KB admin)

```bash
curl -X PUT "$RAGAAS/v1/kb/$KB/widget/config" -H "Authorization: Bearer $ADMIN_JWT" \
  -H 'Content-Type: application/json' \
  -d '{"enabled":true,"title":"Ask Acme","welcome":"Hi! Ask about our docs.",
       "accent":"#1a3d7c","suggestions":["What is the SLA?","How is my data secured?"]}'
```

## 2. Add a session proxy on your server (holds the API key)

```js
// Express example — the ONLY place your API key lives.
app.post('/api/ragaas-session', async (_req, res) => {
  const r = await fetch(`${process.env.RAGAAS}/v1/widget/session`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${process.env.RAGAAS_WIDGET_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ kb_id: process.env.RAGAAS_KB }),
  })
  res.status(r.status).json(await r.json())
})
```

## 3. Drop the embed on any page

```html
<script src="https://cdn.you/ragaas-widget.js"
        data-base="https://rag.internal"
        data-session-url="/api/ragaas-session"></script>
```

That's it — a themed chat bubble appears, grounded in your KB.

## Build

```bash
npm run build   # tsc → dist/widget.js  (single, dependency-free, ~8 KB)
```

`index.html` is a local demo harness (point `data-session-url` at a mock that returns a
`/v1/widget/session` response).
