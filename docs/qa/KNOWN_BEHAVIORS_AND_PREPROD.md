# Known behaviors & pre-production checklist

The "miscellaneous but necessary" sweep — the non-feature things that bite in QA/prod. Each
item is tagged with what was actually checked (✅ verified good · 🔧 fixed this pass · 📋 by-design
nuance to understand · ⏳ deferred to the prod cutover). Everything here is evidence-based, not
assumed.

## ✅ Verified good — don't re-chase these
- **No cross-clearance cache leak.** The semantic-cache key includes the caller's `clearance`
  *and* ACL (`{clearance}|{acl}`), so a higher-clearance answer is never served to a
  lower-clearance caller. (`semantic_cache.py`.)
- **Long-query crash is fixed.** OpenSearch BM25 queries are capped at 512 terms, so a
  ~2000-word question no longer trips `maxClauseCount=1024` into an uncaught 500.
  (`opensearch_index.py`.)
- **Stores auto-restart.** postgres / opensearch / qdrant / minio / nli-checker all run
  `restart: unless-stopped` (opensearch had an Exit-137 OOM history).
- **Deep readiness probe exists.** `/readyz` returns **503** when any store a query needs
  (qdrant/opensearch/tei/postgres) is unreachable — distinct from the shallow `/health`
  liveness. (Now wired to a Docker healthcheck — see 🔧 below.)
- **Erasure reaches the knowledge graph.** The delete path calls `graph_store.remove_chunks`,
  so a DPDP erasure removes the doc's graph nodes/edges, not just its chunks.
- **DR restore is proven.** A real backup restored into a scratch DB with completeness matching
  live exactly (50 schemas, all federation grants, all registries). Live data untouched.
- **MCP RBAC parity.** No MCP tool is more permissive than its REST twin; the past zero-role
  bypass is not regressed (MCP uses exact role membership, stricter than REST).

## 🔧 Fixed this pass
- **RBAC — 10 under-gated endpoints closed.** A bare `reader` could previously: change a KB's
  grounding thresholds, persona, retention window; **run retention purge and DPDP erasure
  (data-destructive)**; read the full audit trail; overwrite the golden set; run evals. All now
  require the KB's management tier (`_can_manage_kb`) or the right capability role, live-verified
  (reader → 403, admin → 200). The Dashboard's recent-questions widget was guarded to match the
  audit gate. (See `SECURITY_RBAC.md` if present, and the console nav fixes for Knowledge Graph +
  Guardrails visibility.)
- **Provisioning is now idempotent/resumable.** A partial provision (org+owner created, then a
  store timeout before the fleet record) used to 422 into an unrecoverable stuck state; it now
  **resumes to completion** on retry without re-creating the org/owner. Verified 6/6.
- **Container healthcheck wired to `/readyz`.** All four api instances now report **unhealthy**
  when they can't serve (a store is down), instead of showing "Up" regardless. Applies on the
  next recreate.
- **Ingest partial-failure status** (`partial`) and **citation re-alignment** — see the main
  commit / `FEDERATION_QA.md`.

## 📋 By-design nuances QA should understand (not bugs)
- **Multi-KB `/ask` audits at org scope.** A single-KB query audits under that KB's schema
  (correct), but a *cross-KB* `/ask` answer audits at the **org** level by design (it spans
  KBs). So a per-KB Quality/Compliance dashboard will not show cross-KB answers — that is
  expected, not missing data.
- **Activity feed omits actor identity.** `GET /kb/{id}/activity` is a deliberate reduced
  "job-centre" slice for members (action/label/time only) — unlike the full `/audit` (now
  management-gated).
- **Conversations are KB-shared.** Any member can see/delete a KB's chat threads (no per-user
  owner). If chat history must be tamper-resistant, gate `DELETE …/conversations/{id}` —
  flagged as an optional hardening.
- **`/agent/run` binds the caller's actor but not roles/clearance** (unlike `workflows/run`,
  which binds them). Orchestrator steps are HITL/dual-control gated, but confirm as a design
  point that an agent run cannot exceed the caller's own role.

## 📋 Quality ceilings — manage expectations
- **Output-language reliability is Hindi-strong, weaker elsewhere.** Measured: English-question-
  stays-English ~**94% Hindi**, but ~**66%** for Tamil/Bengali/Telugu; **Marathi** collapses to
  Hindi in script detection and **Urdu** isn't in the script table at all. Native-language
  questions are reliable (~100%). For a pan-India product this is a real, documented ceiling; the
  fix is model-side (instruction-following), not a prompt bug.
- **RAGAS certification needs ≥8 golden questions/KB.** A thin golden set correctly returns
  INCONCLUSIVE, never a false PASS. Expanding each KB's golden set is a QA data task.

## 📋 DPDP nuance
- **Erasure vs backups.** Erasure cascades correctly through the live stores (incl. the graph
  store), but erased data survives in **backups** until they age out. Define backup retention to
  match the erasure SLA and document it in the DPDP posture.

## ⏳ Deferred to the prod cutover (dev-only today, by owner's call)
These are acceptable while the platform is behind the dev VPN for QA; they MUST be closed before
production:
1. **`/dev/token` is open** (`RAG_ENV=dev`) — anyone who can reach the API can mint an
   owner/admin/clearance-3 token for ANY tenant. Disable outside true-local dev / move to
   OIDC+password auth at cutover.
2. **No shared Redis** (`RAG_REDIS_URL` unset) on the 4-api mesh — per-org rate limits are
   effectively ~4×, semantic-cache invalidation isn't cross-replica, and the eval/sync leader
   lease isn't coordinated. Point all replicas at one Redis.
3. **Sovereignty egress** — the allowlist still contains foreign hosts (`api.jina.ai`,
   `www.datalab.to`) and test model endpoints. The documented config-only cutover (internal-only
   model URLs + tightened allowlist) closes the runtime-sovereignty claim.

## Still worth a look (not yet done)
- **Full secrets-hygiene scan** (this pass was RBAC-focused): confirm no secret is logged or
  echoed in error/`/system/info` payloads; master Fernet/KMS key handling at prod.
- **Sovereignty-guard coverage**: confirm every outbound path (models, connectors, crawl, bots)
  passes `assert_sovereign` — 22 raw httpx call sites to reconcile against 35 guard usages.
