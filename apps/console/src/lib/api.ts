// Typed client for the RAG as a Service /v1 API. Mirrors the pydantic contracts in
// libs/schemas. All calls are tenant-scoped; the bearer token carries the tenant.

export type AnswerStatus = 'grounded' | 'abstained' | 'flagged_for_review'

export type GuardrailTier = 'platform' | 'org' | 'kb'
export interface GuardrailCatalogItem {
  id: string; label: string; category: string; kind: 'toggle' | 'threshold'
  constitutional: boolean; default_owner: GuardrailTier; help: string
}
export interface GuardrailEntry { value: boolean | number; locked: boolean }
export interface GuardrailEffective {
  id: string; label: string; category: string; kind: 'toggle' | 'threshold'
  value: boolean | number; owner: GuardrailTier | null; locked_by: GuardrailTier | null
  constitutional: boolean; help: string
}

export interface ModelHealth {
  name: string
  /** The roster role this backend serves, stamped server-side so the table can match on it. */
  role: string
  endpoint: string
  online: boolean
  latency_ms: number | null
  detail: string
  /** The model the probe actually reached — may differ from the configured one. */
  model?: string
  /** When this backend was last observed online (from the continuous poller). Null if never up. */
  last_online?: string | null
  /** Consecutive failed probes; the dot flips offline only past the debounce threshold. */
  consecutive_failures?: number
}

export interface ModelHealthReport {
  checked_at: string
  cached: boolean
  /** 'poller' = served from the continuous background snapshot (no probe this request); 'live' = probed now. */
  source?: 'poller' | 'live'
  /** True when a served poller snapshot is older than several poll intervals — the reading may not be current. */
  stale?: boolean
  backends: ModelHealth[]
}

export interface ModelStatus {
  source: 'connected' | 'default'
  name: string
  model: string
  endpoint: string | null
  online: boolean | null
  latency_ms: number | null
  detail: string
}

export interface Citation {
  chunk_id: string
  doc_id: string
  page: number | null
  source_uri: string | null
  quote: string
  score: number
  modality?: string  // 'text' | 'visual' — 'visual' marks an image source (multimodal)
}

export interface AgentStep {
  step: string
  detail: string
}

// --- verification provenance ---------------------------------------------------------------------
//
// `GROUNDED` says the answer's claims are tied to cited evidence. It says nothing about WHICH
// checker verified them, or whether the intended one was reachable — so an answer scored by a
// lexical fallback after an NLI read error used to look identical to an NLI-verified one at every
// surface a user has. These types carry that difference to the UI.
export type VerificationMethod = 'nli' | 'lexical' | 'lexical_fallback' | 'unavailable'

export interface MethodCount { method: VerificationMethod; count: number }
export interface VerificationFinding { method: VerificationMethod; reason_code: string | null; detail: string | null }
export interface VerificationBatch {
  method: VerificationMethod
  degraded: boolean
  score: number | null
  reason_code: string | null
}

interface VerificationCommon {
  // Whether the INTENDED mechanism was available. Orthogonal to `outcome`: a run where every
  // batch fell back to the lexical checker is `scored` (one method, so its scores ARE comparable)
  // and degraded (NLI never ran). Both facts matter and neither implies the other.
  degraded: boolean
  // ORDERED distribution. A UI must render "NLI: 12 · Lexical fallback: 1" and must not be able
  // to render "NLI" by picking the largest count — which is why this is a list, not a map.
  methods: MethodCount[]
  batches: VerificationBatch[]
  findings?: VerificationFinding[]
}

// A DISCRIMINATED UNION ON `outcome`, so `score` is unreachable without narrowing. Omitting the
// key would not have been enough on its own: an absent property is `undefined`, and
// `v.score ?? 0` yields 0 for `undefined` exactly as for `null`, so a careless read turns "not
// comparable" into a failing grade. Here the compiler refuses the read instead.
export type Verification =
  // One method produced every score, so one number is meaningful.
  | (VerificationCommon & { outcome: 'scored'; score: number })
  // Several methods did. No mean is reported: an NLI score and a lexical score are not the same
  // measurement, and averaging them yields a number that means nothing and looks authoritative.
  | (VerificationCommon & { outcome: 'mixed' })
  // At least one batch was never checked.
  | (VerificationCommon & { outcome: 'unavailable' })

export interface GroundedAnswer {
  tenant_id: string
  query: string
  status: AnswerStatus
  answer: string
  citations: Citation[]
  confidence: number
  abstain_reason: string | null
  faithfulness_score: number | null
  // Sentences a check did not find supported. Reachable on a GROUNDED answer too, not only the
  // flagged path: the numeric guard can report a finding on an answer that routes GROUNDED on its
  // score. A reader seeing an empty list must be able to trust that nothing was flagged.
  unsupported_claims: string[]
  review_id: string | null
  trace_id: string | null
  model: string | null
  suggested_followups?: string[]
  agent_trace?: AgentStep[]
  // FR-E6 structured output — the architecture feature rides its editable diagram here
  // (rendered by the console) rather than as a fenced block in the answer text.
  structured?: { mermaid?: string; plantuml?: string; architecture?: unknown; diagram_note?: string } | null
  // Auto+confirm architecture diagram: true when the question showed POTENTIAL (but not strong)
  // architecture intent, so the console offers an on-demand "generate diagram" chip.
  diagram_available?: boolean
  // Non-authoritative completeness signal (Slice 1). Present when the completeness guard ran;
  // when `complete` is false the answer may be partial (truncated or missing a relevant point).
  completeness?: Completeness | null
  // How this answer was verified. `null`/absent when no verification ran at all — which is
  // DIFFERENT from `outcome: 'unavailable'`, where verification was attempted and could not
  // reach a judgement.
  verification?: Verification | null
}

export interface Completeness {
  complete: boolean
  reason: string | null
  covered: number
  expected: number
  truncated: boolean
}

export interface KbInfo {
  kb_id: string
  name: string
  is_default: boolean
  published?: boolean
  personal?: boolean   // a user-owned KB (creator is its manager), not a governed dept KB
  owner?: string       // email of the personal-KB owner (empty for governed dept KBs)
}

// The one platform-wide shared KB (Federated RAG Tier C). It's surfaced in every tenant's KB
// list by the backend but owned by no tenant — so the UI must not offer to delete it, and badges
// it as shared. Must match knowledge_base.SHARED_KB_ID on the backend.
export const SHARED_KB_ID = 'public__company-wide'

export interface McpInfo {
  enabled: boolean
  endpoint_path: string
  transport: string
  auth: string
  tenancy: string
  tools: { name: string; description: string }[]
}

// The plan-execute-critique agent's run result (POST /v1/agent/run).
export interface AgentPlanStep { tool: string; args: Record<string, unknown>; rationale?: string }
export interface AgentObservation { tool: string; ok: boolean; result?: unknown; error?: string; approval_id?: string | null }
export interface AgentRunResult {
  goal: string
  plan: { goal: string; steps: AgentPlanStep[] }
  review?: { approved: boolean; concerns: string[]; suggestion?: string } | null
  observations: AgentObservation[]
  answer: string
  pending_approval_id?: string | null
  halted_reason: string
}

export interface DocumentSummary {
  doc_id: string
  mime: string
  source_kind: string
  uri: string | null
  ingested_at: string
  metadata?: Record<string, string>
}

export interface IngestResponse {
  doc_id: string
  chunk_count: number
  pii_masked: number
  trace_id: string
  outcome?: string // created | updated | unchanged | quarantined | blocked_pii
  duplicate_of?: string | null
  classification?: number | null // final data-classification (raised by the PII floor when applicable)
  reason?: string | null // detail when outcome === 'blocked_pii'
}

export type HitlStatus = 'pending' | 'accepted' | 'modified' | 'flagged' | 'overridden'
export type HitlAction = 'view' | 'accept' | 'modify' | 'flag' | 'override'

export interface HitlReview {
  review_id: string
  tenant_id: string
  trace_id: string
  status: HitlStatus
  query: string
  answer: string
  citations: string[]
  reason: string | null
  warnings: string[]
  reviewer: string | null
  note: string | null
  updated_at: string
}

export interface AuditEvent {
  event_id: string
  tenant_id: string
  trace_id: string
  action: string
  actor: string
  at: string
  payload: Record<string, unknown>
  prev_hash: string | null
  event_hash: string | null
}

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message)
  }
}

// Parse a response body as JSON, tolerating empty and non-JSON bodies (e.g. nginx's HTML
// 413/502 pages) so an error never surfaces as an opaque "SyntaxError".
function safeJson(text: string): Record<string, unknown> | null {
  if (!text) return null
  try {
    return JSON.parse(text) as Record<string, unknown>
  } catch {
    return null
  }
}

// Human, actionable messages for transport-level failures that don't carry a JSON body.
function friendlyStatus(status: number, statusText: string): string {
  if (status === 413) return 'That file is too large to upload. Try a smaller document (up to 1 GB).'
  if (status === 502 || status === 504) return 'The server is starting up or unreachable. Give it a moment and retry.'
  if (status === 401) return 'Your session has expired. Sign in again.'
  return statusText || `Request failed (${status})`
}

async function request<T>(path: string, token: string | null, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json', ...(init.headers as Record<string, string>) }
  if (token) headers.Authorization = `Bearer ${token}`
  let res: Response
  try {
    res = await fetch(path, { ...init, headers })
  } catch {
    throw new ApiError(0, 'network', 'Could not reach the server. Check your connection and that the API is running.')
  }
  const text = await res.text()
  const body = safeJson(text)
  if (!res.ok) {
    const code = (body && (body.code as string)) || 'error'
    const msg = (body && (body.message as string)) || friendlyStatus(res.status, res.statusText)
    throw new ApiError(res.status, code, msg)
  }
  return body as T
}

export interface DevTokenResponse {
  access_token: string
  token_type: string
  tenant_id: string
}

export interface AuthResponse {
  access_token: string
  token_type: string
  email: string
  name: string
  org_id: string
  org_name: string
  role: string
  roles: string[]
  kb_access: string[]
}

export interface Member {
  email: string
  name: string
  role: string
  status: 'invited' | 'active' | 'suspended'
  roles: string[]
  kb_access: string[]
  // Not currently returned by list/add/update responses (backend tracks it, but MemberInfo
  // doesn't serialize it yet) — kept optional so the type stays honest and forward-compatible.
  clearance?: number
}

export interface ActivationResult {
  email_sent: boolean
  link: string
}

// A custom named role = a label + a chosen subset of the 4 capability primitives
// (reader/ingestor/reviewer/admin). Pure sugar over caps: assigning it sets those caps; the
// label is reverse-matched from a member's cap-set for display. See GET/PUT /v1/org/roles.
export interface CustomRole {
  id: string
  label: string
  caps: string[]
}

// Cross-KB access request / grant (see the access-request endpoints).
export interface AccessRequest {
  id: string
  kb_id: string
  kb_name: string
  requester: string
  requested_clearance: number
  granted_clearance: number | null
  reason: string
  status: 'pending' | 'approved' | 'rejected' | 'revoked'
  decided_by: string
  created_at: string
  decided_at: string
}
export interface KbDirectoryEntry {
  kb_id: string
  name: string
  has_access: boolean
  request_status: '' | 'pending' | 'approved' | 'rejected' | 'revoked'
  granted_clearance: number | null
}

// Org-wide display labels for the fixed capability roles (reader/ingestor/reviewer/admin) plus
// the read-only 'owner' label. Cosmetic only — the role VALUES never change, only what they're
// called in the UI. See lib/roleLabels.ts for the shared fetch hook + fallback defaults.
export interface RoleLabels {
  labels: Record<string, string>
}

export interface ApiKeyInfo {
  key_id: string
  name: string
  prefix: string
  scope: string
  created_at: string
}

export interface CreatedApiKey extends ApiKeyInfo {
  key: string
}

export interface SystemModelRole {
  role: string
  backend: string
  model: string
  endpoint: string
  detail: string
}

export interface SystemInfo {
  env: string
  sovereign_enforce: boolean
  llm_backend: string
  llm_model: string
  embed_model: string
  vector_backend: string
  sparse_backend: string
  guardrails_backend: string
  faithfulness_backend: string
  features: Record<string, boolean>
  thresholds: Record<string, number>
  models?: SystemModelRole[]
}

export interface ModelsInfo {
  endpoint: string
  backend: string
  active_model: string
  available: string[]
  error: string | null
}

export interface ModelConnectionInfo {
  configured: boolean
  endpoint: string | null
  model: string | null
  header: string | null
  active: boolean
  has_key: boolean
}

// Each role is a different job, not a variant of one: generation writes answers, embedding makes
// vectors, reranking orders passages, ocr reads scans, vision describes images. The faithfulness
// (NLI) judge is deliberately absent — it is the platform's grounding gate, not a tenant choice.
export type ModelRole = 'generation' | 'embedding' | 'reranking' | 'ocr' | 'vision'

/** An org-wide embedder switch waiting to be applied (it re-embeds every KB listed). */
export interface OrgEmbedderStatus {
  pending: { model?: string; endpoint?: string; dim?: number } | null
  kbs: { kb_id: string; documents: number }[]
  documents: number
  apply_enabled: boolean
}

export interface OrgEmbedderApplyResult {
  applied: { kb_id: string; reindexed: number; failed: string[] }[]
  failed: { kb_id: string; error: string }[]
}

export interface ModelConnectionBody {
  endpoint: string
  model: string
  header?: string
  api_key?: string
  active?: boolean
  role?: ModelRole
  embed_dim?: number
  api_shape?: string
}

export interface ModelConnectionItem {
  conn_id: string
  name: string
  endpoint: string
  model: string
  header: string
  has_key: boolean
  is_default: boolean
  role: ModelRole
  embed_dim: number | null
  api_shape: string
  // "kb" = registered on this KB; "org" = an org-wide model (available, READ-ONLY in the KB view).
  scope: 'kb' | 'org'
}

export interface AddModelConnectionBody {
  name: string
  endpoint: string
  model: string
  header?: string
  api_key?: string
  make_default?: boolean
  role?: ModelRole
  embed_dim?: number
  api_shape?: string
}

// A model-change request awaiting an org-admin decision. A KB manager's write to a model
// endpoint returns this (HTTP 202) instead of applying; an org admin sees these on the queue.
export type ModelRequestAction = 'add' | 'update' | 'default' | 'delete' | 'embedder_apply' | 'org_add'
export interface ModelRequestInfo {
  id: string
  kb_id: string  // "" for an org-wide (org_add) request
  kb_name: string
  action: ModelRequestAction
  conn_id: string | null
  payload: Record<string, unknown> | null
  requested_by: string
  requested_at: string
  status: 'pending' | 'approved' | 'rejected'
  decided_by: string
  decided_at: string
  reason: string
}

// A model-connection write may APPLY (org admin -> ModelConnectionItem) or be QUEUED as a
// pending request (KB manager -> ModelRequestInfo). Discriminate on the `status` field, which
// only the pending-request shape carries.
export type ModelWriteResult = ModelConnectionItem | ModelRequestInfo
export function isPendingModelRequest(r: unknown): r is ModelRequestInfo {
  return !!r && typeof r === 'object' && typeof (r as ModelRequestInfo).status === 'string' && 'requested_by' in r
}

export interface ModelTestResult {
  ok: boolean
  available: string[]
  error: string | null
  dim?: number | null
  api_shape?: string | null
  score_sample?: number | null
}

export interface EmbedderRecord {
  conn_id: string
  endpoint: string
  model: string
  dim: number | null
  sig: string
}

export interface EmbedderStatus {
  active: EmbedderRecord | null
  pending: EmbedderRecord | null
  documents: number
  apply_enabled: boolean
}

export interface EmbedderApplyResult {
  active: EmbedderRecord
  reindexed: number
  failed: string[]
}

export interface ConnectionStatus {
  conn_id: string
  online: boolean
  latency_ms: number | null
  detail: string
}

export interface UsageLine {
  unit: string
  count: number
  rate: number
  cost: number
}

export interface BudgetInfo {
  kb_id: string
  budget_tokens: number
  alert_pct: number
  used_tokens: number
  remaining_tokens: number
  pct_used: number | null
  over_alert: boolean
  over_budget: boolean
  period_start: string
}

export interface UsageReport {
  tenant_id: string
  window: string
  since: string | null
  counts: Record<string, number>
  queries: number
  documents: number
  abstains: number
  guardrail_blocks: number
  generations: number
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  currency: string
  rate_card_indicative: boolean
  lines: UsageLine[]
  total_cost: number
}

export interface GraphStatsInfo {
  enabled: boolean
  nodes: number
  edges: number
  top_entities: { name: string; mentions: number }[]
}

export interface GraphCommunityInfo {
  community_id: number
  label: string
  size: number
  top_entities: string[]
  summary: string
}

export interface GraphCommunitiesResponse {
  enabled: boolean
  communities: GraphCommunityInfo[]
}

export interface GraphVizNode {
  id: string
  label: string
  degree: number
  community: number
}

export interface GraphSubgraph {
  enabled: boolean
  nodes: GraphVizNode[]
  edges: [number, number][]
  total_nodes: number
  total_edges: number
}

export interface WidgetConfig {
  enabled: boolean
  title: string
  welcome: string
  accent: string
  placeholder: string
  suggestions: string[]
}

export interface BotConfig {
  enabled: boolean
  platform: string
  trigger: string
  configured: boolean
  webhook_path: string
}

export interface KbSettings {
  abstain_threshold: number | null
  faithful_threshold: number | null
  flag_threshold: number | null
}

export interface KbSettingsInfo {
  overrides: KbSettings
  defaults: Record<string, number>
}

export interface GoldenQuestion {
  question: string
  expected?: string | null
}

export interface QualityReport {
  tenant_id: string
  window: string
  answers: number
  grounded: number
  flagged: number
  abstained_faithfulness: number
  grounded_rate: number | null
  faithfulness_avg: number | null
  faithfulness_buckets: Record<string, number>
  abstain_reasons: Record<string, number>
  guardrail_blocks: Record<string, number>
  reviews_accepted: number
  reviews_modified: number
  reviews_overridden: number
  human_agreement_rate: number | null
  feedback_up: number
  feedback_down: number
  satisfaction_rate: number | null
}

export interface EvalRow {
  question: string
  status: string
  hit: boolean | null
  faithfulness: number | null
  answer: string
  answer_relevancy: number | null
  context_precision: number | null
  context_recall: number | null
}

export interface EvalReport {
  total: number
  grounded: number
  abstained: number
  flagged: number
  hits: number
  with_expected: number
  hit_rate: number | null
  faithfulness_avg: number | null
  answer_relevancy_avg: number | null
  context_precision_avg: number | null
  context_recall_avg: number | null
  ragas_scored: number
  rows: EvalRow[]
}

export interface EvalSchedule {
  kb_id: string
  enabled: boolean
  interval_hours: number
  min_grounded_rate: number
  min_faithfulness: number
  last_run_at: string | null
}

export interface EvalRunResult {
  kb_id: string
  at: string
  golden_n: number
  sampled_n: number
  total: number
  grounded: number
  abstained: number
  flagged: number
  grounded_rate: number | null
  faithfulness_avg: number | null
  hit_rate: number | null
  breached: boolean
  breach_reasons: string[]
}

export interface EvalRunsResponse {
  enabled: boolean
  schedule: EvalSchedule | null
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  runs: Record<string, any>[]
}

export interface ExternalObjectSource {
  endpoint: string
  bucket: string
  secure: boolean
  credentials_ref: string
  /** Object-storage API. "s3" covers AWS/MinIO/Ceph; "azure" is Azure Blob Storage, which has no
   *  S3 API — there the endpoint is <account>.blob.core.windows.net and the bucket is a container. */
  provider?: 's3' | 'azure'
}

/** FR-C8 stale-index detection: a document is indexed by whatever pipeline ran at ingest. */
export interface IndexStatus {
  kb_id: string
  /** Fingerprint of the running indexing pipeline. */
  current_version: string
  total: number
  /** Documents indexed by a different pipeline, or carrying no stamp at all. */
  stale: number
  stale_doc_ids: string[]
  /** True when an operator opted this KB into automatic re-indexing. */
  auto_reindex: boolean
}

export interface ReindexKbResult {
  kb_id: string
  requested: number
  reindexed: number
  skipped_current: number
  failed: { doc_id: string; error: string }[]
}

// Ingest either inline (content_b64) or by-reference (object_key), keyed by source_kind.
// external_source (with source_kind=object_store) pulls from an external tenant bucket (A2b).
export interface IngestPayload {
  doc_id: string
  mime?: string
  strategy?: string
  source_kind?: 'upload' | 'object_store' | 'web'
  content_b64?: string
  object_key?: string
  external_source?: ExternalObjectSource
  url?: string
  // Enable the PII-driven classification floor for this ingest (raise/block a doc whose detected
  // PII implies a higher classification). Omitted = platform default when the feature is enabled.
  pii_autoclassify?: boolean
}

export interface SyncItem {
  doc_id: string
  ok: boolean
  chunks: number
  error: string | null
}

export interface SyncResponse {
  found: number
  ingested: number
  failed: number
  /** True when the bucket holds more objects than this run took — run it again to continue. */
  more?: boolean
  items: SyncItem[]
}

// Scheduled / delta sync (A8)
/** The captured folder source — non-secret fields only; credentials_ref is a NAME, not a key. */
export interface SyncSourceView {
  provider: string
  endpoint: string
  bucket: string
  prefix: string
  secure: boolean
  credentials_ref: string
  limit: number
}
export interface SyncScheduleView {
  enabled: boolean
  interval_hours: number
  cron: string
  timezone: string
  last_run_at: string | null
  next_run_at: string | null
  source_configured: boolean
  source: SyncSourceView | null
  scheduler_on: boolean
}
export interface SyncRunResult {
  found: number
  ingested: number
  created: number
  updated: number
  unchanged: number
  failed: number
}

// Runtime feature flags (admin-toggleable capabilities)
export interface FeatureFlag {
  name: string
  label: string
  enabled: boolean
  locked: boolean
}
export interface FeatureFlagsView {
  flags: FeatureFlag[]
}

// Async bulk ingestion
export interface IngestJob {
  job_id: string
  kb_id: string
  actor: string
  status: 'building' | 'queued' | 'running' | 'completed' | 'partial' | 'failed' | 'cancelled'
  total: number
  done: number
  failed: number
  created_at: string
  updated_at: string
  error: string | null
}
export interface IngestJobView {
  job: IngestJob
  failures: { doc_id: string; error: string }[]
}

// Egress-host approval requests
export interface EgressRequest {
  id: string
  host: string
  tenant_id: string
  kb_id: string | null
  requested_by: string
  reason: string
  status: 'pending' | 'approved' | 'rejected' | 'revoked'
  created_at: string
  decided_by: string | null
  decided_at: string | null
}

// --- Federation grants (Federated RAG) ---
// A grant lets a subject department query a set of this department's KBs on given topics,
// until it expires or is revoked. Federation is flag-gated: the endpoints 403 with
// code=forbidden + details.flag=federation_enabled when RAG_FEDERATION_ENABLED is off.
export interface GrantView {
  grant_id: string
  issuer_dept_id: string
  subject_dept_id: string
  topics: string[]
  kb_ids: string[]
  max_results_per_query: number
  not_before: string
  expires_at: string
  revoked_at: string | null
  active: boolean
}

export interface CreateGrantBody {
  subject_dept_id: string
  topics: string[]
  kb_ids: string[]
  max_results_per_query?: number
  ttl_days?: number
}

// Re-delegation of a received grant: same shape, but each field may only NARROW the parent.
export interface DelegateGrantBody {
  subject_dept_id: string
  topics: string[]
  kb_ids: string[]
  max_results_per_query?: number
  ttl_days?: number
}

// A signed trust receipt for one answer (opaque to the UI — we only display validity).
export type TrustReceipt = Record<string, unknown>

// --- KB roles & member assignments (named RBAC per KB) ---
// A role bundles a clearance level (0=Public … 3=Restricted), ingest/review capability, and a
// human description. Assignments map a member's email to one named role within this KB.
export interface KbRole {
  clearance: number   // 0..3 → Public / Internal / Confidential / Restricted
  can_ingest: boolean
  can_review: boolean
  description: string
}
export type KbRoles = Record<string, KbRole>
export type KbRoleAssignments = Record<string, string>  // email → role name

export type StreamEvent =
  | { type: 'phase'; phase: string }
  | { type: 'condensed'; text: string }
  | { type: 'token'; text: string }
  | { type: 'done'; answer: GroundedAnswer }
  | { type: 'followups'; items: string[] }
  | { type: 'error'; message: string }

export interface ConversationInfo {
  conversation_id: string
  title: string
  created_at: string
  updated_at: string
}

export interface ConversationMessage {
  message_id: string
  role: 'user' | 'assistant'
  content: string
  trace_id: string | null
  status: AnswerStatus | null
  citations: Citation[]
  faithfulness: number | null
  created_at: string
}

export interface ConversationDetail {
  conversation: ConversationInfo
  messages: ConversationMessage[]
}

export interface ErasureCertificate {
  kb_id: string
  requested_by: string
  reason: string
  subject: string | null
  documents_erased: string[]
  chunks_purged: number
  conversations_purged: number
  certificate_hash: string
  erased_at: string
}

export interface RetentionInfo { kb_id: string; retention_days: number }

export interface RetentionResult {
  kb_id: string
  retention_days: number
  documents_erased: string[]
  certificate: ErasureCertificate | null
}

export interface Attestation {
  kb_id: string
  generated_at: string
  data_residency: string
  sovereignty_enforced: boolean
  egress_allowlist: string[]
  encryption_at_rest: boolean
  encryption_in_transit: boolean
  audit_chain_verified: boolean
  audit_events: number
  retention_days: number
  documents: number
  notes: string[]
}

// --- Responsible-AI compliance pack (#7) ---
export interface ComplianceControl {
  id: string
  title: string
  requirement: string
  status: 'satisfied' | 'partial' | 'manual' | 'not_applicable'
  evidence: string[]
}
export interface ComplianceFramework {
  id: string
  name: string
  version: string
  controls: ComplianceControl[]
  satisfied: number
  total: number
}
export interface ComplianceReport {
  kb_id: string
  generated_at: string
  frameworks: ComplianceFramework[]
  summary: { frameworks: number; controls: number; satisfied: number }
  notes: string[]
}

// --- Envelope encryption / BYOK (#6) ---
export interface CryptoStatus {
  envelope_encryption: boolean
  kms_backend: string
  byok: boolean
  kms_key_id_configured: boolean
  dek_versions: number
  provisioned: boolean
}
export interface CryptoRotateResult {
  rotated: boolean
  dek_versions: number
  reencrypted_connections: number
}

// --- Governed NL→SQL (#8) — a KB may register MANY DB connections ---
export interface SqlConnectionInfo {
  configured: boolean
  conn_id: string | null
  name: string | null
  dialect: string
  tables: string[]
  host: string | null
  port: number | null
  database: string | null
}
export interface SqlConnectionItem {
  conn_id: string
  name: string
  dialect: string
  tables: string[]
  is_default: boolean
  host: string | null
  port: number | null
  database: string | null
}
export interface AddSqlConnectionBody {
  name: string
  dsn: string
  tables: string[]
  dialect?: string
  make_default?: boolean
}
export interface UpdateSqlConnectionBody {
  name: string
  dsn?: string
  tables: string[]
  dialect?: string
}
export interface SqlAnswer {
  question: string | null
  connection_id: string | null
  connection_name: string | null
  sql: string
  generated_sql: string | null
  columns: string[]
  rows: (string | number | boolean | null)[][]
  row_count: number
}

// --- Structured source connections (#10) — NoSQL / REST / SharePoint, mirroring the SQL
// connection shape above (one connection per KB; PUT replaces it). Sync pulls them into the
// knowledge graph used by Architecture and graph-aware answers.
export interface NoSqlConnectionInfo {
  configured: boolean
  host: string | null
  port: number | null
  database: string | null
  collections: string[]
}
export interface NoSqlConnectionBody {
  uri: string
  database: string
  collections: string[]
}

export interface RestConnectionInfo {
  configured: boolean
  base_url: string | null
  path: string | null
  query_param: string | null
  has_token: boolean
}
export interface RestConnectionBody {
  base_url: string
  path?: string
  query_param?: string
  token?: string
}

export interface SharePointConnectionInfo {
  configured: boolean
  site_url: string | null
  has_token: boolean
}
export interface SharePointConnectionBody {
  site_url: string
  token?: string
}

export interface GraphSyncResult {
  sql?: number
  nosql?: number
  rest?: number
  sharepoint?: number
}

// --- Architecture diagrams (RAG-generated system diagrams over a KB's documents + sources) ---
export type ArchAudience = 'non-technical' | 'technical'

export interface ArchComponent {
  id: string
  name: string
  kind: string
  description: string
}

export interface ArchEdge {
  source: string
  target: string
  relation: string
  label: string
}

export interface Architecture {
  title: string
  components: ArchComponent[]
  edges: ArchEdge[]
  summary: string
}

export interface ArchitectureAnswer {
  architecture: Architecture
  mermaid: string
  plantuml: string
  explanation: string
  citations: Citation[]
}

// --- Agentic workflows (#9) ---
export interface WorkflowStepTrace {
  step_id: string
  type: string
  summary: string
  goto: string | null
  matched_rule: boolean
}
export interface WorkflowResult {
  workflow_id: string
  status: string
  trace: WorkflowStepTrace[]
  output: Record<string, unknown>
  error: string | null
}
export interface WorkflowDef {
  id: string
  name: string
  start: string
  steps: {
    id: string
    type: string
    config?: Record<string, unknown>
    rules?: { all?: unknown[]; any?: unknown[]; goto: string }[]
    next?: string | null
  }[]
  max_steps?: number
}

export interface SourceChunk {
  chunk_id: string
  text: string
  page: number | null
}

export interface DocumentContent {
  doc_id: string
  mime: string
  source_uri: string | null
  chunks: SourceChunk[]
}

// Stream one conversational turn (FR-I1). Emits a `condensed` event first when the
// follow-up was rewritten against the thread, then phase/token/done StreamEvents.
export async function* streamConversationTurn(
  token: string, kbId: string, conversationId: string, query: string, model?: string, recencyDays?: number,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const res = await fetch(
    `/v1/kb/${encodeURIComponent(kbId)}/conversations/${encodeURIComponent(conversationId)}/ask/stream`,
    {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, model: model || undefined, recency_days: recencyDays || undefined }),
      signal,
    },
  )
  if (!res.ok || !res.body) throw new ApiError(res.status, 'stream_error', friendlyStatus(res.status, 'The answer stream could not start.'))
  const reader = res.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    let idx: number
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const chunk = buf.slice(0, idx).trim()
      buf = buf.slice(idx + 2)
      if (chunk.startsWith('data:')) yield JSON.parse(chunk.slice(5).trim()) as StreamEvent
    }
  }
}

export interface IngestDagNode {
  id: string
  label: string
  stage: string
  status: string // 'defined' | 'ran' | 'skipped' | 'failed'
  metrics?: Record<string, string | number>
  duration_ms?: number | null
  at_ms?: number | null
  event_hash?: string
}

export interface IngestDagGraph {
  nodes: IngestDagNode[]
  edges: { source: string; target: string }[]
  doc_id?: string
  title?: string
}

export interface IngestDagTrust {
  stages_total: number
  stages_ran: number
  verified: boolean
  note: string
}

export interface IngestDagResponse {
  dag: IngestDagGraph
  mermaid: string
  trust: IngestDagTrust
}

// --- Platform / fleet (SUPER_ADMIN only) ---------------------------------------------
export interface TenantRecord {
  org_id: string
  name: string
  admin_email: string
  status: 'active' | 'suspended'
  plan: string
  max_documents_per_kb: number
  monthly_budget: number
  provisioned_at: string | null
  updated_at: string | null
  managed?: boolean
}
export interface TenantSummary {
  tenant: TenantRecord
  knowledge_bases: number
  users: number
}
export interface ResidencyAttestation {
  attestation_id: string
  node_id: string
  region: string
  kb_ids: string[]
  data_residency: string
  no_foreign_egress: boolean
  generated_at: string
  signer_kid: string
  signed_token: string
}
export interface AirgapViolation { setting: string; value: string; reason: string }
export interface AirgapReport { air_gap_enabled: boolean; clean: boolean; violations: AirgapViolation[] }

export const api = {
  health: () => request<{ status: string; env: string }>('/health', null),

  // --- Platform (SUPER_ADMIN only) — fleet/org lifecycle ------------------------------
  listTenants: (token: string) => request<TenantRecord[]>('/v1/platform/tenants', token),
  getTenant: (token: string, orgId: string) =>
    request<TenantSummary>(`/v1/platform/tenants/${encodeURIComponent(orgId)}`, token),
  provisionTenant: (token: string, body: { org_id: string; name: string; admin_email: string; password: string; plan: string }) =>
    request<TenantRecord>('/v1/platform/tenants', token, { method: 'POST', body: JSON.stringify(body) }),
  suspendTenant: (token: string, orgId: string) =>
    request<TenantRecord>(`/v1/platform/tenants/${encodeURIComponent(orgId)}/suspend`, token, { method: 'POST' }),
  resumeTenant: (token: string, orgId: string) =>
    request<TenantRecord>(`/v1/platform/tenants/${encodeURIComponent(orgId)}/resume`, token, { method: 'POST' }),
  setTenantPlan: (token: string, orgId: string, plan: string) =>
    request<TenantRecord>(`/v1/platform/tenants/${encodeURIComponent(orgId)}/plan`, token, { method: 'POST', body: JSON.stringify({ plan }) }),
  deprovisionTenant: (token: string, orgId: string) =>
    request<Record<string, number>>(`/v1/platform/tenants/${encodeURIComponent(orgId)}`, token, { method: 'DELETE' }),
  residency: (token: string) => request<ResidencyAttestation>('/v1/platform/residency/node', token),
  airgapPreflight: (token: string) => request<AirgapReport>('/v1/platform/airgap/preflight', token),

  createConversation: (token: string, kbId: string, firstQuery: string) =>
    request<ConversationInfo>(`/v1/kb/${encodeURIComponent(kbId)}/conversations`, token, {
      method: 'POST',
      body: JSON.stringify({ query: firstQuery }),
    }),

  listConversations: (token: string, kbId: string) =>
    request<ConversationInfo[]>(`/v1/kb/${encodeURIComponent(kbId)}/conversations`, token),

  getConversation: (token: string, kbId: string, conversationId: string) =>
    request<ConversationDetail>(
      `/v1/kb/${encodeURIComponent(kbId)}/conversations/${encodeURIComponent(conversationId)}`, token),

  deleteConversation: (token: string, kbId: string, conversationId: string) =>
    request<unknown>(
      `/v1/kb/${encodeURIComponent(kbId)}/conversations/${encodeURIComponent(conversationId)}`, token,
      { method: 'DELETE' }),

  submitFeedback: (token: string, kbId: string, traceId: string, rating: 'up' | 'down', comment?: string) =>
    request<unknown>(`/v1/kb/${encodeURIComponent(kbId)}/feedback`, token, {
      method: 'POST',
      body: JSON.stringify({ trace_id: traceId, rating, comment: comment || undefined }),
    }),

  documentContent: (token: string, kbId: string, docId: string) =>
    request<DocumentContent>(
      `/v1/kb/${encodeURIComponent(kbId)}/documents/${encodeURIComponent(docId)}/content`, token),

  erase: (token: string, kbId: string, body: { doc_ids?: string[]; subject?: string; reason?: string }) =>
    request<ErasureCertificate>(`/v1/kb/${encodeURIComponent(kbId)}/erasure`, token, {
      method: 'POST', body: JSON.stringify(body),
    }),

  getRetention: (token: string, kbId: string) =>
    request<RetentionInfo>(`/v1/kb/${encodeURIComponent(kbId)}/retention`, token),

  setRetention: (token: string, kbId: string, days: number) =>
    request<RetentionInfo>(`/v1/kb/${encodeURIComponent(kbId)}/retention`, token, {
      method: 'PUT', body: JSON.stringify({ retention_days: days }),
    }),

  applyRetention: (token: string, kbId: string) =>
    request<RetentionResult>(`/v1/kb/${encodeURIComponent(kbId)}/retention/apply`, token, { method: 'POST' }),

  attestation: (token: string, kbId: string) =>
    request<Attestation>(`/v1/kb/${encodeURIComponent(kbId)}/attestation`, token),

  devToken: (subject: string, tenantId: string, groups: string[] = []) =>
    request<DevTokenResponse>('/v1/dev/token', null, {
      method: 'POST',
      body: JSON.stringify({ subject, tenant_id: tenantId, groups }),
    }),

  signup: (orgName: string, name: string, email: string, password: string) =>
    request<AuthResponse>('/v1/auth/signup', null, {
      method: 'POST',
      body: JSON.stringify({ org_name: orgName, name, email, password }),
    }),

  authLogin: (email: string, password: string) =>
    request<AuthResponse>('/v1/auth/login', null, { method: 'POST', body: JSON.stringify({ email, password }) }),

  listMembers: (token: string) => request<Member[]>('/v1/org/members', token),

  listApiKeys: (token: string) => request<ApiKeyInfo[]>('/v1/api-keys', token),

  createApiKey: (token: string, name: string, scope: 'read' | 'ingest' | 'admin') =>
    request<CreatedApiKey>('/v1/api-keys', token, { method: 'POST', body: JSON.stringify({ name, scope }) }),

  revokeApiKey: (token: string, keyId: string) =>
    request<unknown>(`/v1/api-keys/${encodeURIComponent(keyId)}`, token, { method: 'DELETE' }),

  addMember: (token: string, m: { name: string; email: string; password?: string; roles: string[]; kb_access: string[] }) =>
    request<Member>('/v1/org/members', token, { method: 'POST', body: JSON.stringify(m) }),

  // clearance (0=Public..3=Restricted) is enforced no-escalation server-side: the backend 403s
  // if the caller's own clearance is below the value being granted (can_grant_clearance).
  updateMember: (token: string, email: string, body: { roles: string[]; kb_access: string[]; clearance?: number }) =>
    request<Member>(`/v1/org/members/${encodeURIComponent(email)}`, token, { method: 'PATCH', body: JSON.stringify(body) }),

  activateMember: (token: string, email: string) =>
    request<ActivationResult>(`/v1/org/members/${encodeURIComponent(email)}/activate`, token, { method: 'POST' }),

  setMemberStatus: (token: string, email: string, status: 'active' | 'suspended') =>
    request<Member>(`/v1/org/members/${encodeURIComponent(email)}/status`, token, { method: 'POST', body: JSON.stringify({ status }) }),
  revokeMember: (token: string, email: string) =>
    request<void>(`/v1/org/members/${encodeURIComponent(email)}`, token, { method: 'DELETE' }),
  getCustomRoles: (token: string) => request<{ roles: CustomRole[] }>('/v1/org/roles', token),
  setCustomRoles: (token: string, roles: CustomRole[]) =>
    request<{ roles: CustomRole[] }>('/v1/org/roles', token, { method: 'PUT', body: JSON.stringify({ roles }) }),
  // KB managers (delegated KB-scoped member management)
  getKbManagers: (token: string, kbId: string) =>
    request<{ managers: string[] }>(`/v1/kb/${encodeURIComponent(kbId)}/managers`, token),
  setKbManagers: (token: string, kbId: string, managers: string[]) =>
    request<{ managers: string[] }>(`/v1/kb/${encodeURIComponent(kbId)}/managers`, token, { method: 'PUT', body: JSON.stringify({ managers }) }),
  listKbMembers: (token: string, kbId: string) =>
    request<Member[]>(`/v1/kb/${encodeURIComponent(kbId)}/members`, token),
  addKbMember: (token: string, kbId: string, m: { name: string; email: string; password?: string; roles: string[] }) =>
    request<Member>(`/v1/kb/${encodeURIComponent(kbId)}/members`, token, { method: 'POST', body: JSON.stringify(m) }),
  removeKbMember: (token: string, kbId: string, email: string) =>
    request<void>(`/v1/kb/${encodeURIComponent(kbId)}/members/${encodeURIComponent(email)}`, token, { method: 'DELETE' }),
  myManagedKbs: (token: string) => request<string[]>('/v1/me/managed-kbs', token),
  // Cross-KB access requests
  kbDirectory: (token: string) => request<KbDirectoryEntry[]>('/v1/org/kb-directory', token),
  requestAccess: (token: string, kbId: string, clearance: number, reason: string) =>
    request<AccessRequest>(`/v1/kb/${encodeURIComponent(kbId)}/access-requests`, token, { method: 'POST', body: JSON.stringify({ clearance, reason }) }),
  listAccessRequests: (token: string, kbId: string) =>
    request<AccessRequest[]>(`/v1/kb/${encodeURIComponent(kbId)}/access-requests`, token),
  decideAccessRequest: (token: string, kbId: string, reqId: string, approve: boolean, clearance: number) =>
    request<AccessRequest>(`/v1/kb/${encodeURIComponent(kbId)}/access-requests/${encodeURIComponent(reqId)}/decide`, token, { method: 'POST', body: JSON.stringify({ approve, clearance }) }),
  revokeAccess: (token: string, kbId: string, email: string) =>
    request<void>(`/v1/kb/${encodeURIComponent(kbId)}/access/${encodeURIComponent(email)}/revoke`, token, { method: 'POST' }),
  myAccess: (token: string) => request<AccessRequest[]>('/v1/me/access', token),
  accessOverview: (token: string) => request<AccessRequest[]>('/v1/org/access-overview', token),

  getRoleLabels: (token: string) => request<RoleLabels>('/v1/org/role-labels', token),

  setRoleLabels: (token: string, labels: Record<string, string>) =>
    request<RoleLabels>('/v1/org/role-labels', token, { method: 'PUT', body: JSON.stringify({ labels }) }),

  // Public (token in the link is the credential) — no bearer.
  setPassword: (resetToken: string, password: string) =>
    request<unknown>('/v1/auth/set-password', null, { method: 'POST', body: JSON.stringify({ token: resetToken, password }) }),

  forgotPassword: (email: string) =>
    request<{ ok: boolean }>('/v1/auth/forgot-password', null, { method: 'POST', body: JSON.stringify({ email }) }),

  listDocuments: (token: string, kbId: string) =>
    request<DocumentSummary[]>(`/v1/kb/${encodeURIComponent(kbId)}/documents`, token),

  activity: (token: string, kbId: string) =>
    request<{ action: string; label: string; detail: string; at: string }[]>(
      `/v1/kb/${encodeURIComponent(kbId)}/activity`, token),

  /** Live probe of every configured model endpoint (admins) or just the answering model
   *  (members). Each call is a real request to a real endpoint, so the server caches briefly —
   *  `refresh` is the Re-check button and forces a fresh round. */
  modelHealth: (token: string, kbId?: string, refresh = false) => {
    const q = new URLSearchParams()
    if (kbId) q.set('kb_id', kbId)
    if (refresh) q.set('refresh', 'true')
    const qs = q.toString()
    return request<ModelHealthReport>(`/v1/system/model-health${qs ? `?${qs}` : ''}`, token)
  },

  modelStatus: (token: string, kbId: string, refresh = false) =>
    request<ModelStatus>(
      `/v1/kb/${encodeURIComponent(kbId)}/model-status${refresh ? '?refresh=true' : ''}`, token),

  mcpInfo: (token: string) =>
    request<McpInfo>('/v1/agent/mcp', token),

  // Run the plan-execute-critique agent: LLM plans over the in-process tool catalog, a critic
  // reviews, critical (write) steps pause for dual-control approval, safe steps run + are audited.
  runAgent: (token: string, goal: string) =>
    request<AgentRunResult>('/v1/agent/run', token, { method: 'POST', body: JSON.stringify({ goal }) }),

  getTaxonomy: (token: string, kbId: string) =>
    request<{ taxonomy: Record<string, string[]> }>(`/v1/kb/${encodeURIComponent(kbId)}/taxonomy`, token),

  setTaxonomy: (token: string, kbId: string, taxonomy: Record<string, string[]>) =>
    request<{ taxonomy: Record<string, string[]> }>(`/v1/kb/${encodeURIComponent(kbId)}/taxonomy`, token, {
      method: 'PUT', body: JSON.stringify({ taxonomy }),
    }),

  deleteDocument: (token: string, kbId: string, docId: string) =>
    request<unknown>(`/v1/kb/${encodeURIComponent(kbId)}/documents/${encodeURIComponent(docId)}`, token, { method: 'DELETE' }),

  reindexDocument: (token: string, kbId: string, docId: string, strategy = 'clause') =>
    request<IngestResponse>(`/v1/kb/${encodeURIComponent(kbId)}/documents/${encodeURIComponent(docId)}/reindex`, token, {
      method: 'POST',
      body: JSON.stringify({ strategy }),
    }),

  /** Whether this KB's documents were built by the indexing pipeline that is running now. */
  indexStatus: (token: string, kbId: string) =>
    request<IndexStatus>(`/v1/kb/${encodeURIComponent(kbId)}/index-status`, token),

  /** Rebuild the KB's stale documents. Sequential and capped server-side. */
  reindexKb: (token: string, kbId: string, staleOnly = true) =>
    request<ReindexKbResult>(`/v1/kb/${encodeURIComponent(kbId)}/reindex`, token, {
      method: 'POST',
      body: JSON.stringify({ stale_only: staleOnly }),
    }),

  listKbs: (token: string, _tenantId: string) =>
    request<KbInfo[]>('/v1/knowledge-bases', token),

  createKb: (token: string, displayName: string) =>
    request<KbInfo>('/v1/knowledge-bases', token, {
      method: 'POST',
      body: JSON.stringify({ display_name: displayName }),
    }),

  // Personal (user-owned) KBs: any member creates their own KB and shares it with org members.
  createPersonalKb: (token: string, displayName: string) =>
    request<KbInfo>('/v1/me/kbs', token, {
      method: 'POST',
      body: JSON.stringify({ display_name: displayName }),
    }),
  shareKb: (token: string, kbId: string, email: string, clearance: number) =>
    request<AccessRequest>(`/v1/kb/${encodeURIComponent(kbId)}/share`, token, {
      method: 'POST',
      body: JSON.stringify({ email, clearance }),
    }),

  // Soft-delete: moves the KB to Trash (recoverable).
  deleteKb: (token: string, kbId: string) =>
    request<{ kb_id: string; status: string }>(
      `/v1/knowledge-bases/${encodeURIComponent(kbId)}`, token, { method: 'DELETE' }),
  listTrashedKbs: (token: string) =>
    request<{ kb_id: string; name: string; trashed_at: string | null }[]>('/v1/knowledge-bases/trash', token),
  restoreKb: (token: string, kbId: string) =>
    request<KbInfo>(`/v1/knowledge-bases/${encodeURIComponent(kbId)}/restore`, token, { method: 'POST' }),
  purgeKb: (token: string, kbId: string, confirmName: string) =>
    request<{ kb_id: string; documents: number; chunks: number }>(
      `/v1/knowledge-bases/${encodeURIComponent(kbId)}/purge`, token, {
        method: 'POST', body: JSON.stringify({ confirm_name: confirmName }),
      }),

  systemInfo: (token: string) => request<SystemInfo>('/v1/system/info', token),

  askAcross: (token: string, query: string, kbIds: string[] = [], model?: string) =>
    request<GroundedAnswer>('/v1/ask', token, {
      method: 'POST',
      body: JSON.stringify({ query, kb_ids: kbIds, model: model || undefined }),
    }),

  // Federated RAG: answer over own KBs PLUS partner departments that granted access
  // (federate_from). Grant-gated server-side; un-granted departments are skipped silently.
  askFederated: (token: string, query: string, federateFrom: string[], model?: string) =>
    request<GroundedAnswer>('/v1/ask', token, {
      method: 'POST',
      body: JSON.stringify({ query, federate_from: federateFrom, model: model || undefined }),
    }),

  models: (token: string) => request<ModelsInfo>('/v1/system/models', token),

  // The KB's DEFAULT connection (used by Ask/Dashboard's "answering with").
  getModelConnection: (token: string, kbId: string) =>
    request<ModelConnectionInfo>(`/v1/kb/${encodeURIComponent(kbId)}/model-connection`, token),

  // Full registry: every model connected to this KB.
  listModelConnections: (token: string, kbId: string) =>
    request<ModelConnectionItem[]>(`/v1/kb/${encodeURIComponent(kbId)}/model-connections`, token),

  // Model-connection WRITES: an org owner/admin's call APPLIES immediately (ModelConnectionItem);
  // a KB manager's call is QUEUED as a pending request (ModelRequestInfo, HTTP 202). Callers
  // discriminate with isPendingModelRequest().
  addModelConnection: (token: string, kbId: string, body: AddModelConnectionBody) =>
    request<ModelWriteResult>(`/v1/kb/${encodeURIComponent(kbId)}/model-connections`, token, {
      method: 'POST', body: JSON.stringify(body),
    }),

  updateModelConnection: (token: string, kbId: string, connId: string, body: AddModelConnectionBody) =>
    request<ModelWriteResult>(`/v1/kb/${encodeURIComponent(kbId)}/model-connections/${encodeURIComponent(connId)}`, token, {
      method: 'PUT', body: JSON.stringify(body),
    }),

  setDefaultConnection: (token: string, kbId: string, connId: string) =>
    request<ModelWriteResult>(`/v1/kb/${encodeURIComponent(kbId)}/model-connections/${encodeURIComponent(connId)}/default`, token, {
      method: 'POST',
    }),

  deleteModelConnection: (token: string, kbId: string, connId: string) =>
    request<ModelRequestInfo | null>(`/v1/kb/${encodeURIComponent(kbId)}/model-connections/${encodeURIComponent(connId)}`, token, { method: 'DELETE' }),

  testModelConnection: (token: string, kbId: string, body: ModelConnectionBody) =>
    request<ModelTestResult>(`/v1/kb/${encodeURIComponent(kbId)}/model-connections/test`, token, {
      method: 'POST', body: JSON.stringify(body),
    }),

  embedderStatus: (token: string, kbId: string) =>
    request<EmbedderStatus>(`/v1/kb/${encodeURIComponent(kbId)}/embedder/status`, token),

  connectionStatus: (token: string, kbId: string, connId: string, refresh = false) =>
    request<ConnectionStatus>(
      `/v1/kb/${encodeURIComponent(kbId)}/model-connections/${encodeURIComponent(connId)}/status${refresh ? '?refresh=true' : ''}`,
      token,
    ),

  // Org admin: applies the pending embedder switch (EmbedderApplyResult). KB manager: queues a
  // pending request (ModelRequestInfo, 202).
  embedderApply: (token: string, kbId: string) =>
    request<EmbedderApplyResult | ModelRequestInfo>(`/v1/kb/${encodeURIComponent(kbId)}/embedder/apply`, token, { method: 'POST' }),

  // --- Model-change approval (governance for per-KB model connections) ---
  // Org admin: the full approval queue across the org.
  listOrgModelRequests: (token: string) =>
    request<ModelRequestInfo[]>('/v1/org/model-requests', token),
  // Org admin OR this KB's manager: this KB's requests (a manager watches their own status).
  listKbModelRequests: (token: string, kbId: string) =>
    request<ModelRequestInfo[]>(`/v1/kb/${encodeURIComponent(kbId)}/model-requests`, token),
  // Org admin only: approve (re-applies the stored action) or reject a pending request.
  decideModelRequest: (token: string, kbId: string, requestId: string, decision: 'approve' | 'reject', reason?: string) =>
    request<ModelRequestInfo>(
      `/v1/kb/${encodeURIComponent(kbId)}/model-requests/${encodeURIComponent(requestId)}/decide`,
      token,
      { method: 'POST', body: JSON.stringify({ decision, reason: reason ?? '' }) },
    ),

  // Org admin OR a KB manager: ORG-WIDE (org_add) model requests. An org admin sees ALL pending
  // org requests (the approval queue); a manager sees only their OWN (status watch).
  listMyModelRequests: (token: string) =>
    request<ModelRequestInfo[]>('/v1/org/model-requests', token),
  // Org admin only: approve (adds the model to the org registry) or reject a pending org request.
  decideOrgModelRequest: (token: string, requestId: string, decision: 'approve' | 'reject', reason?: string) =>
    request<ModelRequestInfo>(
      `/v1/org/model-requests/${encodeURIComponent(requestId)}/decide`,
      token,
      { method: 'POST', body: JSON.stringify({ decision, reason: reason ?? '' }) },
    ),

  // --- Org-wide model connections (generation models are org-wide only) ---
  // An org admin manages the registry directly (add/edit/delete/set-default). A KB manager's
  // addOrgModelConnection is QUEUED as a pending org request (ModelRequestInfo, 202) — discriminate
  // with isPendingModelRequest(). listOrgModelConnections is org-admin only.
  listOrgModelConnections: (token: string) =>
    request<ModelConnectionItem[]>('/v1/org/model-connections', token),
  addOrgModelConnection: (token: string, body: AddModelConnectionBody) =>
    request<ModelWriteResult>('/v1/org/model-connections', token, {
      method: 'POST', body: JSON.stringify(body),
    }),
  updateOrgModelConnection: (token: string, connId: string, body: AddModelConnectionBody) =>
    request<ModelConnectionItem>(`/v1/org/model-connections/${encodeURIComponent(connId)}`, token, {
      method: 'PUT', body: JSON.stringify(body),
    }),
  deleteOrgModelConnection: (token: string, connId: string) =>
    request<null>(`/v1/org/model-connections/${encodeURIComponent(connId)}`, token, { method: 'DELETE' }),
  orgEmbedderStatus: (token: string) =>
    request<OrgEmbedderStatus>('/v1/org/embedder/status', token),
  orgEmbedderApply: (token: string) =>
    request<OrgEmbedderApplyResult>('/v1/org/embedder/apply', token, { method: 'POST' }),

  testOrgModelConnection: (token: string, body: ModelConnectionBody) =>
    request<ModelTestResult>('/v1/org/model-connections/test', token, {
      method: 'POST', body: JSON.stringify(body),
    }),

  getGraph: (token: string, kbId: string) =>
    request<GraphStatsInfo>(`/v1/kb/${encodeURIComponent(kbId)}/graph`, token),

  getGraphSubgraph: (token: string, kbId: string, limit = 60) =>
    request<GraphSubgraph>(`/v1/kb/${encodeURIComponent(kbId)}/graph/subgraph?limit=${limit}`, token),

  getCommunities: (token: string, kbId: string) =>
    request<GraphCommunitiesResponse>(`/v1/kb/${encodeURIComponent(kbId)}/graph/communities`, token),

  buildCommunities: (token: string, kbId: string) =>
    request<{ communities: number; entities: number }>(
      `/v1/kb/${encodeURIComponent(kbId)}/graph/communities`, token, { method: 'POST' },
    ),

  // Named per-KB roles (clearance + ingest/review capabilities). The endpoint returns and
  // accepts the roles map directly (contract-fixed), mirroring the taxonomy getter/setter.
  // The backend wraps these as { roles: {...} } / { assignments: {...} }; unwrap to the bare
  // map the UI expects (rendering the envelope object as a child crashes React — err #31).
  getKbRoles: (token: string, kbId: string) =>
    request<{ roles: KbRoles }>(`/v1/kb/${encodeURIComponent(kbId)}/roles`, token).then((r) => r.roles ?? ({} as KbRoles)),

  setKbRoles: (token: string, kbId: string, roles: KbRoles) =>
    request<{ roles: KbRoles }>(`/v1/kb/${encodeURIComponent(kbId)}/roles`, token, {
      method: 'PUT', body: JSON.stringify({ roles }),
    }).then((r) => r.roles ?? ({} as KbRoles)),

  // Member → role assignments within this KB ({ email: role_name }).
  getKbRoleAssignments: (token: string, kbId: string) =>
    request<{ assignments: KbRoleAssignments }>(`/v1/kb/${encodeURIComponent(kbId)}/roles/assignments`, token).then((r) => r.assignments ?? ({} as KbRoleAssignments)),

  setKbRoleAssignments: (token: string, kbId: string, assignments: KbRoleAssignments) =>
    request<{ assignments: KbRoleAssignments }>(`/v1/kb/${encodeURIComponent(kbId)}/roles/assignments`, token, {
      method: 'PUT', body: JSON.stringify({ assignments }),
    }).then((r) => r.assignments ?? ({} as KbRoleAssignments)),

  getPersona: (token: string, kbId: string) =>
    request<{ persona: string }>(`/v1/kb/${encodeURIComponent(kbId)}/persona`, token),

  setPersona: (token: string, kbId: string, persona: string) =>
    request<{ persona: string }>(`/v1/kb/${encodeURIComponent(kbId)}/persona`, token, {
      method: 'PUT', body: JSON.stringify({ persona }),
    }),

  getWidgetConfig: (token: string, kbId: string) =>
    request<WidgetConfig>(`/v1/kb/${encodeURIComponent(kbId)}/widget/config`, token),

  setWidgetConfig: (token: string, kbId: string, config: WidgetConfig) =>
    request<WidgetConfig>(`/v1/kb/${encodeURIComponent(kbId)}/widget/config`, token, {
      method: 'PUT', body: JSON.stringify(config),
    }),

  getBotConfig: (token: string, kbId: string) =>
    request<BotConfig>(`/v1/kb/${encodeURIComponent(kbId)}/bot/config`, token),

  setBotConfig: (token: string, kbId: string, body: { enabled: boolean; platform: string; trigger: string }) =>
    request<BotConfig>(`/v1/kb/${encodeURIComponent(kbId)}/bot/config`, token, {
      method: 'PUT', body: JSON.stringify(body),
    }),

  rotateBotToken: (token: string, kbId: string) =>
    request<{ token: string; webhook_path: string }>(`/v1/kb/${encodeURIComponent(kbId)}/bot/token`, token, { method: 'POST' }),

  getKbSettings: (token: string, kbId: string) =>
    request<KbSettingsInfo>(`/v1/kb/${encodeURIComponent(kbId)}/settings`, token),

  setKbSettings: (token: string, kbId: string, body: Partial<KbSettings>) =>
    request<KbSettingsInfo>(`/v1/kb/${encodeURIComponent(kbId)}/settings`, token, {
      method: 'PUT',
      body: JSON.stringify(body),
    }),

  getGoldenSet: (token: string, kbId: string) =>
    request<{ questions: GoldenQuestion[] }>(`/v1/kb/${encodeURIComponent(kbId)}/golden-set`, token),

  setGoldenSet: (token: string, kbId: string, questions: GoldenQuestion[]) =>
    request<{ questions: GoldenQuestion[] }>(`/v1/kb/${encodeURIComponent(kbId)}/golden-set`, token, {
      method: 'PUT',
      body: JSON.stringify({ questions }),
    }),

  generateGoldenSet: (token: string, kbId: string, count = 5) =>
    request<{ questions: GoldenQuestion[] }>(
      `/v1/kb/${encodeURIComponent(kbId)}/golden-set/generate?count=${count}`, token, { method: 'POST' },
    ),

  quality: (token: string, kbId: string, window: string) =>
    request<QualityReport>(`/v1/kb/${encodeURIComponent(kbId)}/quality?window=${window}`, token),

  runEval: (token: string, kbId: string, questions: { question: string; expected?: string }[]) =>
    request<EvalReport>(`/v1/kb/${encodeURIComponent(kbId)}/eval`, token, {
      method: 'POST',
      body: JSON.stringify({ questions }),
    }),

  getEvalSchedule: (token: string, kbId: string) =>
    request<EvalRunsResponse>(`/v1/kb/${encodeURIComponent(kbId)}/eval/schedule`, token),

  setEvalSchedule: (token: string, kbId: string, body: Omit<EvalSchedule, 'kb_id' | 'last_run_at'>) =>
    request<EvalSchedule>(`/v1/kb/${encodeURIComponent(kbId)}/eval/schedule`, token, {
      method: 'PUT', body: JSON.stringify(body),
    }),

  runEvalNow: (token: string, kbId: string) =>
    request<EvalRunResult>(`/v1/kb/${encodeURIComponent(kbId)}/eval/run-now`, token, { method: 'POST' }),

  usage: (token: string, kbId: string, window: string) =>
    request<UsageReport>(`/v1/kb/${encodeURIComponent(kbId)}/usage?window=${window}`, token),

  getBudget: (token: string, kbId: string) =>
    request<BudgetInfo>(`/v1/kb/${encodeURIComponent(kbId)}/budget`, token),

  setBudget: (token: string, kbId: string, budgetTokens: number, alertPct = 0.8) =>
    request<BudgetInfo>(`/v1/kb/${encodeURIComponent(kbId)}/budget`, token, {
      method: 'PUT', body: JSON.stringify({ budget_tokens: budgetTokens, alert_pct: alertPct }),
    }),

  ingest: (token: string, kbId: string, payload: IngestPayload) =>
    request<IngestResponse>(`/v1/kb/${encodeURIComponent(kbId)}/ingest`, token, {
      method: 'POST',
      body: JSON.stringify({ strategy: 'clause', ...payload }),
    }),

  syncFolder: (token: string, kbId: string, externalSource: ExternalObjectSource, prefix: string) =>
    request<SyncResponse>(`/v1/kb/${encodeURIComponent(kbId)}/ingest/sync`, token, {
      method: 'POST',
      body: JSON.stringify({ external_source: externalSource, prefix }),
    }),

  crawlWebsite: (token: string, kbId: string, body: { seed_url: string; max_depth: number; max_pages: number; same_host_only: boolean }) =>
    request<{ pages: number; created: number; updated: number; unchanged: number; failed: number; skipped_offsite: number; skipped_robots: number; errors: string[] }>(
      `/v1/kb/${encodeURIComponent(kbId)}/ingest/crawl`, token, { method: 'POST', body: JSON.stringify(body) },
    ),

  query: (token: string, kbId: string, query: string, topK = 5) =>
    request<GroundedAnswer>(`/v1/kb/${encodeURIComponent(kbId)}/query`, token, {
      method: 'POST',
      body: JSON.stringify({ query, top_k: topK }),
    }),

  listReviews: (token: string, kbId: string, status?: HitlStatus) =>
    request<HitlReview[]>(
      `/v1/kb/${encodeURIComponent(kbId)}/hitl${status ? `?status=${status}` : ''}`,
      token,
    ),

  getReview: (token: string, kbId: string, reviewId: string) =>
    request<HitlReview>(`/v1/kb/${encodeURIComponent(kbId)}/hitl/${reviewId}`, token),

  actReview: (
    token: string,
    kbId: string,
    reviewId: string,
    action: HitlAction,
    reviewer: string,
    note?: string,
    modifiedAnswer?: string,
  ) =>
    request<HitlReview>(`/v1/kb/${encodeURIComponent(kbId)}/hitl/${reviewId}/act`, token, {
      method: 'POST',
      body: JSON.stringify({ action, reviewer, note, modified_answer: modifiedAnswer }),
    }),

  listAudit: (token: string, kbId: string, traceId?: string, limit?: number) =>
    request<AuditEvent[]>(
      `/v1/kb/${encodeURIComponent(kbId)}/audit?${traceId ? `trace_id=${traceId}&` : ''}${limit ? `limit=${limit}` : ''}`,
      token,
    ),

  // Responsible-AI compliance pack (#7)
  complianceFrameworks: (token: string, kbId: string, framework?: string) =>
    request<ComplianceReport>(
      `/v1/kb/${encodeURIComponent(kbId)}/compliance/frameworks${framework ? `?framework=${framework}` : ''}`, token),

  // Envelope encryption / BYOK (#6)
  cryptoStatus: (token: string, kbId: string) =>
    request<CryptoStatus>(`/v1/kb/${encodeURIComponent(kbId)}/crypto/status`, token),
  rotateCryptoKey: (token: string, kbId: string) =>
    request<CryptoRotateResult>(`/v1/kb/${encodeURIComponent(kbId)}/crypto/rotate`, token, { method: 'POST' }),

  // Governed NL→SQL (#8) — a KB may register MANY DB connections
  getSqlConnection: (token: string, kbId: string) =>
    request<SqlConnectionInfo>(`/v1/kb/${encodeURIComponent(kbId)}/sql/connection`, token),

  listSqlConnections: (token: string, kbId: string) =>
    request<SqlConnectionItem[]>(`/v1/kb/${encodeURIComponent(kbId)}/sql/connections`, token),

  addSqlConnection: (token: string, kbId: string, body: AddSqlConnectionBody) =>
    request<SqlConnectionItem>(`/v1/kb/${encodeURIComponent(kbId)}/sql/connections`, token, {
      method: 'POST', body: JSON.stringify(body),
    }),

  updateSqlConnection: (token: string, kbId: string, connId: string, body: UpdateSqlConnectionBody) =>
    request<SqlConnectionItem>(`/v1/kb/${encodeURIComponent(kbId)}/sql/connections/${encodeURIComponent(connId)}`, token, {
      method: 'PUT', body: JSON.stringify(body),
    }),

  setDefaultSqlConnection: (token: string, kbId: string, connId: string) =>
    request<SqlConnectionItem>(`/v1/kb/${encodeURIComponent(kbId)}/sql/connections/${encodeURIComponent(connId)}/default`, token, {
      method: 'POST',
    }),

  deleteSqlConnection: (token: string, kbId: string, connId: string) =>
    request<unknown>(`/v1/kb/${encodeURIComponent(kbId)}/sql/connections/${encodeURIComponent(connId)}`, token, { method: 'DELETE' }),

  askSql: (token: string, kbId: string, question: string, connectionId?: string) =>
    request<SqlAnswer>(`/v1/kb/${encodeURIComponent(kbId)}/sql/query`, token, {
      method: 'POST', body: JSON.stringify({ question, connection_id: connectionId ?? null }),
    }),

  executeSql: (token: string, kbId: string, sql: string, connectionId?: string) =>
    request<SqlAnswer>(`/v1/kb/${encodeURIComponent(kbId)}/sql/execute`, token, {
      method: 'POST', body: JSON.stringify({ sql, connection_id: connectionId ?? null }),
    }),

  // Structured source connections (#10) — NoSQL
  getNoSqlConnection: (token: string, kbId: string) =>
    request<NoSqlConnectionInfo>(`/v1/kb/${encodeURIComponent(kbId)}/nosql/connection`, token),
  setNoSqlConnection: (token: string, kbId: string, body: NoSqlConnectionBody) =>
    request<NoSqlConnectionInfo>(`/v1/kb/${encodeURIComponent(kbId)}/nosql/connection`, token, {
      method: 'PUT', body: JSON.stringify(body),
    }),
  deleteNoSqlConnection: (token: string, kbId: string) =>
    request<unknown>(`/v1/kb/${encodeURIComponent(kbId)}/nosql/connection`, token, { method: 'DELETE' }),

  // Structured source connections (#10) — REST
  getRestConnection: (token: string, kbId: string) =>
    request<RestConnectionInfo>(`/v1/kb/${encodeURIComponent(kbId)}/rest/connection`, token),
  setRestConnection: (token: string, kbId: string, body: RestConnectionBody) =>
    request<RestConnectionInfo>(`/v1/kb/${encodeURIComponent(kbId)}/rest/connection`, token, {
      method: 'PUT', body: JSON.stringify(body),
    }),
  deleteRestConnection: (token: string, kbId: string) =>
    request<unknown>(`/v1/kb/${encodeURIComponent(kbId)}/rest/connection`, token, { method: 'DELETE' }),

  // Structured source connections (#10) — SharePoint
  getSharePointConnection: (token: string, kbId: string) =>
    request<SharePointConnectionInfo>(`/v1/kb/${encodeURIComponent(kbId)}/sharepoint/connection`, token),
  setSharePointConnection: (token: string, kbId: string, body: SharePointConnectionBody) =>
    request<SharePointConnectionInfo>(`/v1/kb/${encodeURIComponent(kbId)}/sharepoint/connection`, token, {
      method: 'PUT', body: JSON.stringify(body),
    }),
  deleteSharePointConnection: (token: string, kbId: string) =>
    request<unknown>(`/v1/kb/${encodeURIComponent(kbId)}/sharepoint/connection`, token, { method: 'DELETE' }),

  // Pull every connected source (SQL/NoSQL/REST/SharePoint) for a KB into the knowledge graph.
  syncSourcesToGraph: (token: string, kbId: string) =>
    request<GraphSyncResult>(`/v1/kb/${encodeURIComponent(kbId)}/sources/graph-sync`, token, { method: 'POST' }),

  // Architecture: generate a system diagram (Mermaid + PlantUML) grounded in the KB's
  // documents and connected sources, for a chosen topic and audience.
  generateArchitecture: (token: string, kbId: string, topic: string, audience: ArchAudience) =>
    request<ArchitectureAnswer>(`/v1/kb/${encodeURIComponent(kbId)}/architecture`, token, {
      method: 'POST', body: JSON.stringify({ topic, audience }),
    }),

  // Async bulk ingestion
  createIngestJob: (token: string, kbId: string, piiAutoclassify?: boolean, classification?: number) =>
    request<IngestJob>(`/v1/kb/${encodeURIComponent(kbId)}/ingest/jobs`, token, {
      method: 'POST', body: JSON.stringify({ pii_autoclassify: piiAutoclassify ?? null, classification: classification ?? 0 }),
    }),
  addIngestJobItem: (token: string, kbId: string, jobId: string, doc: { doc_id: string; content_b64: string; mime: string; strategy: string }) =>
    request<unknown>(`/v1/kb/${encodeURIComponent(kbId)}/ingest/jobs/${jobId}/items`, token, {
      method: 'POST', body: JSON.stringify(doc),
    }),
  // Large-file path: upload RAW BINARY (multipart, no base64) — the server streams it straight to
  // the object store and the existing job worker ingests it by reference. No Content-Type header:
  // the browser sets the multipart boundary. Avoids the ~1.37x base64 bloat + in-memory buffering.
  addIngestJobItemFile: async (token: string, kbId: string, jobId: string, file: File, opts: { mime: string; strategy: string }) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('doc_id', file.name)
    fd.append('mime', opts.mime)
    fd.append('strategy', opts.strategy)
    let res: Response
    try {
      res = await fetch(`/v1/kb/${encodeURIComponent(kbId)}/ingest/jobs/${jobId}/items/file`, {
        method: 'POST', headers: { Authorization: `Bearer ${token}` }, body: fd,
      })
    } catch {
      throw new ApiError(0, 'network', 'Could not reach the server. Check your connection and that the API is running.')
    }
    if (!res.ok) {
      const body = safeJson(await res.text())
      throw new ApiError(res.status, (body?.code as string) || 'error', (body?.message as string) || friendlyStatus(res.status, res.statusText))
    }
  },
  startIngestJob: (token: string, kbId: string, jobId: string) =>
    request<IngestJob>(`/v1/kb/${encodeURIComponent(kbId)}/ingest/jobs/${jobId}/start`, token, { method: 'POST' }),
  getIngestJob: (token: string, kbId: string, jobId: string) =>
    request<IngestJobView>(`/v1/kb/${encodeURIComponent(kbId)}/ingest/jobs/${jobId}`, token),
  listIngestJobs: (token: string, kbId: string) =>
    request<IngestJob[]>(`/v1/kb/${encodeURIComponent(kbId)}/ingest/jobs`, token),
  retryIngestJob: (token: string, kbId: string, jobId: string) =>
    request<IngestJob>(`/v1/kb/${encodeURIComponent(kbId)}/ingest/jobs/${jobId}/retry`, token, { method: 'POST' }),
  cancelIngestJob: (token: string, kbId: string, jobId: string) =>
    request<void>(`/v1/kb/${encodeURIComponent(kbId)}/ingest/jobs/${jobId}/cancel`, token, { method: 'POST' }),

  // Egress-host approval requests
  requestEgress: (token: string, host: string, reason?: string, kbId?: string) =>
    request<EgressRequest>('/v1/egress/requests', token, {
      method: 'POST', body: JSON.stringify({ host, reason: reason || '', kb_id: kbId }),
    }),
  listEgressRequests: (token: string, status?: string) =>
    request<EgressRequest[]>(`/v1/egress/requests${status ? `?status=${status}` : ''}`, token),
  decideEgress: (token: string, id: string, action: 'approve' | 'reject' | 'revoke') =>
    request<EgressRequest>(`/v1/egress/requests/${encodeURIComponent(id)}/${action}`, token, { method: 'POST' }),

  // Runtime feature flags (admin toggles)
  getFeatures: (token: string) => request<FeatureFlagsView>('/v1/admin/features', token),
  setFeature: (token: string, name: string, enabled: boolean) =>
    request<FeatureFlagsView>(`/v1/admin/features/${encodeURIComponent(name)}`, token, {
      method: 'PUT', body: JSON.stringify({ enabled }),
    }),

  // Scheduled / delta connector sync (A8)
  getSyncSchedule: (token: string, kbId: string) =>
    request<SyncScheduleView>(`/v1/kb/${encodeURIComponent(kbId)}/sync/schedule`, token),
  setSyncSchedule: (token: string, kbId: string, body: { enabled: boolean; interval_hours?: number; cron?: string; timezone?: string }) =>
    request<SyncScheduleView>(`/v1/kb/${encodeURIComponent(kbId)}/sync/schedule`, token, {
      method: 'PUT', body: JSON.stringify(body),
    }),
  runSyncNow: (token: string, kbId: string) =>
    request<SyncRunResult>(`/v1/kb/${encodeURIComponent(kbId)}/sync/run-now`, token, { method: 'POST' }),

  // Agentic workflows (#9)
  listWorkflows: (token: string, kbId: string) =>
    request<WorkflowDef[]>(`/v1/kb/${encodeURIComponent(kbId)}/workflows`, token),
  saveWorkflow: (token: string, kbId: string, def: WorkflowDef) =>
    request<WorkflowDef>(`/v1/kb/${encodeURIComponent(kbId)}/workflows/${encodeURIComponent(def.id)}`, token, {
      method: 'PUT', body: JSON.stringify(def),
    }),
  deleteWorkflow: (token: string, kbId: string, workflowId: string) =>
    request<unknown>(`/v1/kb/${encodeURIComponent(kbId)}/workflows/${encodeURIComponent(workflowId)}`, token, {
      method: 'DELETE',
    }),
  runWorkflow: (token: string, kbId: string, workflowId: string, inputs: Record<string, unknown>) =>
    request<WorkflowResult>(`/v1/kb/${encodeURIComponent(kbId)}/workflows/${encodeURIComponent(workflowId)}/run`, token, {
      method: 'POST', body: JSON.stringify({ inputs }),
    }),

  // Federation grants (Federated RAG)
  listGrants: (token: string) => request<GrantView[]>('/v1/grants', token),
  listReceivedGrants: (token: string) => request<GrantView[]>('/v1/grants/received', token),
  createGrant: (token: string, body: CreateGrantBody) =>
    request<GrantView>('/v1/grants', token, { method: 'POST', body: JSON.stringify(body) }),
  revokeGrant: (token: string, grantId: string) =>
    request<unknown>(`/v1/grants/${encodeURIComponent(grantId)}`, token, { method: 'DELETE' }),
  // Transitively re-delegate a RECEIVED grant to another department (bounded: the child may
  // only narrow the parent; revoking any ancestor kills it). Owner/admin, federation on.
  delegateGrant: (token: string, grantId: string, body: DelegateGrantBody) =>
    request<GrantView>(`/v1/grants/${encodeURIComponent(grantId)}/delegate`, token, { method: 'POST', body: JSON.stringify(body) }),
  // Trust receipts: pull an answer's signed receipt (by trace id) and verify its signature.
  getReceipt: (token: string, answerId: string) =>
    request<TrustReceipt>(`/v1/receipts/${encodeURIComponent(answerId)}`, token),
  verifyReceipt: (token: string, receipt: TrustReceipt) =>
    request<{ valid: boolean }>('/v1/receipts/verify', token, { method: 'POST', body: JSON.stringify(receipt) }),
  // Ingestion pipeline DAG (backend traceability, audit-verified).
  getIngestPipelineDag: (token: string) => request<IngestDagResponse>('/v1/ingest/dag', token),
  getDocumentDag: (token: string, kbId: string, docId: string) =>
    request<IngestDagResponse>(
      `/v1/kb/${encodeURIComponent(kbId)}/documents/${encodeURIComponent(docId)}/dag`,
      token,
    ),
  // --- Hierarchical guardrail policy (top-down RBAC lock model) ---
  getGuardrailCatalog: (token: string) =>
    request<GuardrailCatalogItem[]>('/v1/guardrails/catalog', token),
  getGuardrailsEffective: (token: string, kbId: string) =>
    request<GuardrailEffective[]>(`/v1/kb/${encodeURIComponent(kbId)}/guardrails/effective`, token),
  setKbGuardrails: (token: string, kbId: string, updates: Record<string, GuardrailEntry>) =>
    request<unknown>(`/v1/kb/${encodeURIComponent(kbId)}/guardrails`, token, {
      method: 'PUT',
      body: JSON.stringify({ updates }),
    }),
  setOrgGuardrails: (token: string, updates: Record<string, GuardrailEntry>) =>
    request<unknown>('/v1/org/guardrails', token, { method: 'PUT', body: JSON.stringify({ updates }) }),
  setPlatformGuardrails: (token: string, updates: Record<string, GuardrailEntry>) =>
    request<unknown>('/v1/platform/guardrails', token, { method: 'PUT', body: JSON.stringify({ updates }) }),
}
