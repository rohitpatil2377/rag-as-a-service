"""12-factor configuration for RAG as a Service.

All configuration comes from the environment (or the process's injected secrets),
never from code. Settings are grouped by concern and validated at startup so a
misconfigured deployment fails fast rather than at first request.

Sovereignty note: `sovereign_enforce` + `allowed_egress_hosts` back the runtime
guard that blocks any egress to non-allowlisted (e.g. foreign/managed) endpoints.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_PREFIX = "RAG_"


class AppEnv(StrEnum):
    dev = "dev"
    staging = "staging"
    prod = "prod"


class _Base(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=_ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )


# The bounds on `layout_chunk_max_chars`, named so its Field and the per-doc_type map that
# overrides it cannot drift apart — a map entry must not be able to set a size the field itself
# would reject.
_LAYOUT_CHUNK_MIN_CHARS = 200
_LAYOUT_CHUNK_MAX_CHARS = 8192


class Settings(_Base):
    """Root application settings. Access via :func:`get_settings`."""

    # --- app ---
    env: AppEnv = AppEnv.dev
    log_level: str = "INFO"
    log_json: bool = True
    api_host: str = "0.0.0.0"  # noqa: S104 - bind inside container; ingress fronts it
    api_port: int = 8000
    # Startup preflight (svc_control_plane.preflight) always PROBES every configured remote model
    # backend and logs one warning per problem + an attestation line — non-blocking by default.
    # When true, boot FAILS if a CRITICAL backend (generation LLM / embedder / reranker) is
    # unreachable or rejects its credential, so a broken deploy fails loud instead of degrading
    # silently. Default false preserves the never-block behaviour.
    preflight_strict: bool = False

    # --- sovereignty guard ---
    sovereign_enforce: bool = True
    allowed_egress_hosts: str = "localhost"

    # --- auth ---
    auth_mode: str = "dev_jwt"  # dev_jwt | oidc
    dev_jwt_secret: str = "dev-only-insecure-secret-change-me-32b+"  # noqa: S105 - dev default; prod uses OIDC
    # OIDC / SSO (real IdP: Keycloak / Azure AD / any OIDC provider). JWTs are verified against
    # the provider's JWKS (RS256/ES256); MFA is enforced by the IdP and asserted via `amr`/`acr`.
    oidc_issuer: str = ""
    oidc_jwks_url: str = ""  # allowlisted (assert_sovereign) before fetch
    oidc_audience: str = ""  # expected `aud`; empty = don't verify audience
    oidc_subject_claim: str = "sub"
    oidc_tenant_claim: str = "org"  # which claim carries the org/tenant id
    oidc_roles_claim: str = "roles"
    oidc_groups_claim: str = "groups"  # ACL groups (FR-A6)
    oidc_require_mfa: bool = False  # reject tokens without MFA evidence in `amr`/`acr`
    oidc_acr_mfa_values: str = "mfa,otp,hwk,http://schemas.microsoft.com/claims/multipleauthn"
    # On-prem Active Directory / LDAP bind login (banking pack). The login endpoint binds the
    # user's credentials against the directory; on success the platform issues its own JWT.
    ldap_enabled: bool = False
    ldap_server: str = ""  # ldap://host:389 or ldaps://host:636 (host must be egress-allowlisted)
    ldap_user_dn_template: str = ""  # e.g. "{username}@bank.local" (AD UPN) or a full bind DN
    ldap_base_dn: str = ""  # search base for reading mail/displayName/memberOf (optional)
    ldap_default_org: str = "default"  # org id AD users are provisioned into (single-org deployments)
    ldap_role_map: str = ""  # AD group → role, e.g. "Domain Admins:admin,RAG-Reviewers:reviewer"

    # --- email (member activation / password reset) ---
    # Sovereign: mail goes ONLY through this operator SMTP relay, whose host must be on the
    # egress allowlist (assert_sovereign). Leave smtp_host empty on air-gapped sites — the
    # activation/reset flow then falls back to a link the admin copies and shares out-of-band.
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = "no-reply@ragaas.local"
    smtp_starttls: bool = True
    # Public base URL of the console, used to build absolute activation/reset links in emails
    # (e.g. https://rag.acme.internal). Empty -> the console builds the link from its own origin.
    console_base_url: str = ""

    # --- secrets ---
    secrets_provider: str = "env"  # env | vault
    # Master key for encrypting client-supplied secrets at rest (per-tenant model API keys).
    # Dev default; prod MUST inject a strong key via HSM/KMS. Any string — a 32-byte
    # Fernet key is derived from it, so rotating it invalidates existing ciphertexts.
    secrets_master_key: str = "dev-insecure-master-key-change-in-prod"
    # Envelope encryption / BYOK (FR-H9): each tenant gets its own Data Encryption Key (DEK),
    # stored wrapped by a Key Encryption Key (KEK). "local" derives the KEK from
    # secrets_master_key (dev default, same trust as before); "external" wraps/unwraps the DEK
    # in a sovereign KMS/HSM (kms_key_id names the KEK there) so revoking that key crypto-shreds
    # the tenant's data. The KMS endpoint is an external dependency → egress-allowlisted.
    kms_backend: str = Field(default="local", pattern="^(local|external)$")
    kms_key_id: str = ""
    # External KMS endpoint (Vault Transit base URL) — egress-allowlisted (sovereign). Empty
    # ⇒ the external wrapper fails closed. Token via RAG_KMS_TOKEN (dev) or a SecretsProvider.
    kms_endpoint: str = ""
    kms_token: str = ""

    # --- stores ---
    # When true, the factory wires in-memory adapters + a deterministic local embedder
    # (no external stores/models). Enables the walking skeleton and CI to run with zero
    # infra; the SAME pipeline runs against the sovereign stack when false. Dev/test only.
    inmemory: bool = False
    vector_backend: str = "qdrant"
    qdrant_url: str = "http://qdrant:6333"
    # --- Qdrant at-scale collection config (applied at collection CREATION; existing
    #     collections keep their config until re-indexed). Defaults = today's behaviour. ---
    # Quantization shrinks the in-RAM vector footprint (mandatory past a few tens of millions
    # of vectors): scalar=int8 (~4x), binary (~32x, needs rescore). See docs/production-sizing.md.
    qdrant_quantization: str = Field(default="none", pattern="^(none|scalar|binary)$")
    # Keep original (full-precision) vectors on disk, quantized vectors in RAM.
    qdrant_on_disk: bool = False
    # Horizontal sharding + HA replication across a Qdrant cluster (1 = single node, today).
    qdrant_shards: int = Field(default=1, ge=1, le=1024)
    qdrant_replication: int = Field(default=1, ge=1, le=16)
    sparse_backend: str = "opensearch"
    opensearch_url: str = "http://opensearch:9200"
    object_backend: str = "minio"
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"  # noqa: S105 - dev default; prod via SecretsProvider
    minio_secure: bool = False
    postgres_dsn: str = "postgresql+psycopg://rag:rag@postgres:5432/rag"

    # --- embeddings ---
    # tei = self-hosted TEI (/embed); openai_compat = any OpenAI-compatible /embeddings endpoint
    # (e.g. the AIOps platform bge-m3 on GPU) with apikey-header auth — set tei_url to that base + a key.
    # bge_m3_multi = the bge-m3 /embed endpoint returning dense+learned-sparse+ColBERT in one call
    # (Slice 2b, gated by multivector_enabled + Bearer auth); it is a drop-in DENSE embedder plus
    # the additive embed_multi() capability.
    embed_backend: str = "tei"  # tei | openai_compat | bge_m3_multi
    tei_url: str = "http://tei:80"
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024
    # openai_compat embedder auth (AIOps/hosted). Host must be egress-allowlisted.
    embed_api_key: str = ""
    embed_api_key_header: str = "apikey"

    # --- multi-vector retrieval (Slice 2b) — dense + learned-sparse + ColBERT via bge-m3 /embed ---
    # Master flag (default OFF, additive). A KB whose ACTIVE embedder is the `bge_m3_multi` adapter
    # produces three signals per child from ONE /embed call — dense (unchanged 1024-d),
    # learned-sparse (term→weight) and a ColBERT multivector — stored in a NAMED Qdrant collection;
    # retrieval fuses the extra legs per-KB. OFF ⇒ every path is byte-identical to today (the
    # multivector legs no-op, collections stay single-unnamed-vector, the abstain gate is untouched).
    #
    # Turning it on for a POPULATED KB is an ON-DISK schema change: named vectors + a named sparse
    # vector + a ColBERT multivector cannot be added to an existing UNNAMED collection in place, so
    # the collection must be RECREATED (unnamed→named) and the corpus reindexed. That rides the
    # existing delete_tenant + write-then-swap per-doc reindex (embedder_switch.apply). A KB that is
    # never reindexed keeps its unnamed dense-only collection and the new legs no-op. Leave OFF in
    # dev compose until the corpora have been reindexed.
    multivector_enabled: bool = False
    # `bge_m3_multi` /embed endpoint (dense+sparse+colbert in one call). Empty ⇒ falls back to
    # `tei_url`. Host must be egress-allowlisted (sovereign; assert_sovereign runs on it).
    embed_multi_url: str = ""
    # Embedding endpoint auth mode. "raw" sends the key verbatim in `embed_api_key_header` (AIOps
    # /embeddings, unchanged — the openai_compat adapter is always raw). "bearer" sends
    # `Authorization: Bearer <embed_api_key>` — the mode the bge-m3 /embed endpoint expects; ONLY
    # the bge_m3_multi adapter honours it. The key VALUE lives in the gitignored override.yml (via
    # RAG_EMBED_API_KEY), never in code.
    embed_api_auth: str = Field(default="raw", pattern="^(raw|bearer)$")
    # How many dense candidates ColBERT MaxSim re-scores (the Qdrant Prefetch pool). Bounds MaxSim
    # to the top-M dense hits — the multivector is stored rescore-only (HNSW m=0), no independent
    # index. Per-KB overridable via the `colbert_prefetch_pool` tenant_config key (_THRESHOLD_KEYS).
    colbert_prefetch_pool: int = Field(default=100, ge=1, le=1000)

    @model_validator(mode="after")
    def _bge_m3_multi_needs_bearer(self) -> Settings:
        """The bge-m3 /embed endpoint authenticates with Bearer, not the raw apikey header. Catch a
        misconfigured global backend at BOOT (a failed Settings construction) rather than at the
        first embed call. Per-KB/org attaches carry their own auth (the adapter forces Bearer for
        the bge_m3_multi backend), so this guards only the env-level default."""
        if self.embed_backend == "bge_m3_multi" and self.embed_api_auth != "bearer":
            raise ValueError(
                "RAG_EMBED_BACKEND=bge_m3_multi requires RAG_EMBED_API_AUTH=bearer "
                "(the bge-m3 /embed endpoint uses Bearer auth)"
            )
        return self

    # --- reranking (FR-D2) — a cross-encoder over the fused candidate pool ---
    # The reranker sets the FINAL order and the confidence the abstain gate reads, so it runs in
    # every environment. "lexical" (term-overlap, no model) exists only for an operator who
    # deliberately accepts a weaker scorer; it is never selected by the environment name, and its
    # score is an overlap fraction, NOT comparable to a cross-encoder abstain threshold.
    # There is deliberately no "none": nothing would then write `score`, every span would reach
    # the gate unscored, and the KB would abstain on every query. An operator who sets it gets a
    # startup ValidationError naming the two real values rather than a silent cross-encoder.
    # cross_encoder = self-hosted TEI /rerank; openai_compat = an OpenAI-compatible /rerank
    # endpoint (e.g. AIOps bge-reranker-v2-m3 on GPU) with apikey-header auth.
    rerank_backend: str = Field(default="cross_encoder", pattern="^(cross_encoder|lexical|openai_compat)$")
    rerank_url: str = "http://reranker:80"  # TEI /rerank server (host must be egress-allowlisted)
    # openai_compat reranker: auth + wire shape (tei | cohere [+sigmoid]). AIOps bge-reranker = cohere.
    rerank_api_key: str = ""
    rerank_api_key_header: str = "apikey"
    rerank_api_shape: str = "cohere"
    # The cross-encoder the operator expects to be serving, and the one `abstain_threshold` was
    # calibrated against. The adapter VERIFIES it against the endpoint's own /info and refuses a
    # mismatch: a different model has a different score distribution, so silently swapping the
    # container (e.g. switching to the GPU tier) would move the gate without moving the number.
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # How many fused candidates the cross-encoder scores. Bigger = better recall into the final
    # ranking and linearly more reranker latency (measured on CPU: ~100 ms/candidate for a 22M
    # model, ~500 ms/candidate for a 278M one). Raise it only with a GPU reranker.
    rerank_candidates: int = Field(default=50, ge=1, le=500)
    # --- version conflict (which document is current) --------------------------------------
    # A real corpus holds more than one version of the same policy. Measured live, two Acme
    # handbooks listed DIFFERENT members of the same committee; the platform reported both and
    # said the sources differ -- honest, but a reader cannot tell which list is in force, and at a
    # thousand documents that becomes the normal answer.
    #
    # When on, each span carries its document's effective date and the ones a newer document
    # replaces are marked, so the answer leads with what is current AND records what it replaced.
    # Superseded sources are kept, never hidden: "what did the policy say before?" is a real
    # question and dropping the old version answers it wrongly by omission.
    version_conflict_enabled: bool = False
    # How alike two document titles must be, as a share of their combined vocabulary, before the
    # platform will call one a version of the other. Supersession is a strong claim and it is only
    # true WITHIN a document: measured live, an unbounded comparison told a reader the Night Shift
    # Allowance Policy had been superseded by the HR Handbook, two unrelated policies that were
    # merely retrieved together. Lower to catch heavier renamings, raise if unrelated documents in
    # your corpus share most of their title words.
    version_family_similarity: float = Field(default=0.6, ge=0.0, le=1.0)

    # --- neighbour expansion (contiguity recall) -------------------------------------------
    # Retrieval ranks each passage on its own merits, which fails any question whose answer is a
    # list longer than one passage: the middle of a list shares almost no vocabulary with the
    # question, so it ranks below passages that merely mention the topic. Measured on a real HR
    # handbook, one of six core values appeared in NO retrieved passage at top_k=5 OR top_k=10 --
    # raising top_k does not help, because the problem is ranking, not depth.
    #
    # Pulling the passages either side of a hit fixes it structurally: a list is contiguous, so
    # retrieving any part should bring the rest. Default OFF because it changes what reaches the
    # prompt for every query; turn it on after measuring on the golden set.
    neighbour_expansion_enabled: bool = False
    # How many passages either side of a retrieved one to include. 1 recovers a list spanning two
    # passages, 2 spans three (the measured core-values case). Above ~3 the prompt fills with
    # context nobody asked for and precision falls.
    neighbour_window: int = Field(default=2, ge=0, le=5)
    # How many of the TOP spans get expanded. Expanding all 20 spans of a result would fetch 20
    # documents' passage lists and could double the prompt; expanding the few the reranker ranked
    # highest gets the contiguity benefit at a fraction of the cost.
    neighbour_expand_top: int = Field(default=3, ge=0, le=20)

    # Operator ceiling on how many spans survive reranking, regardless of a caller's top_k.
    # Bounds the generation prompt; set above any top_k you expect so it does not silently bite.
    rerank_top_k: int = Field(default=20, ge=1, le=100)
    # TEI rejects a request above its max_client_batch_size (32 by default); we batch to fit.
    rerank_batch_size: int = Field(default=32, ge=1, le=256)
    rerank_timeout_seconds: float = Field(default=30.0, gt=0)
    # Optional SECONDARY reranker. When rerank_fallback_backend is set, the PRIMARY reranker
    # (above) is wrapped so a primary failure (timeout / 5xx / unreachable) transparently reranks
    # with this one instead — retrieval keeps producing a comparable confidence. Use case: jina
    # (primary, better cross-lingual) + AIOps bge-reranker (secondary) for resilience. Empty = off.
    # NOTE: the two rerankers score on their own scales; the abstain gate is calibrated to the
    # PRIMARY, so a fallback window is a degraded mode, not a like-for-like swap.
    rerank_fallback_backend: str = Field(default="", pattern="^(|openai_compat|cross_encoder|lexical)$")
    rerank_fallback_url: str = ""
    rerank_fallback_model: str = ""
    rerank_fallback_api_key: str = ""
    rerank_fallback_api_key_header: str = "apikey"
    rerank_fallback_api_shape: str = "cohere"
    # Embedding a whole document's chunks is one call per batch, and on the CPU tier a large
    # batch is slow. Measured 2026-07-31: re-indexing the three real corpora, the three LARGEST
    # documents (HR handbook, Abstract, CAG RFP) all failed with httpx.ReadTimeout at the old
    # hard-coded 30 s while every smaller document succeeded — so the limit scaled with document
    # size, which is exactly the case a re-index exercises and a single-document ingest does not.
    embed_timeout_seconds: float = Field(default=180.0, gt=0)

    # --- llm gateway ---
    # openai_compat covers ANY OpenAI-compatible server: self-hosted vLLM/llama.cpp OR a
    # managed sovereign endpoint like the AIOps platform Model-as-a-Service (models-dev.example.internal).
    llm_backend: str = "extractive_dev"  # openai_compat | vllm | llamacpp | extractive_dev
    vllm_url: str = "http://vllm:8000/v1"  # OpenAI-compatible base URL (host must be egress-allowlisted)
    llm_model: str = "Qwen/Qwen2.5-3B-Instruct"
    # Optional API key for a managed endpoint (Acme MaaS uses header 'apikey'). Empty for
    # unauthenticated self-hosted servers. Secret — inject via env/SecretsProvider, never commit.
    llm_api_key: str = ""
    llm_api_key_header: str = "apikey"
    # Optional CA bundle (PEM path inside the container) for model endpoints behind a
    # private CA — mounted by the operator, applied to the LLM/embedder/reranker clients
    # ONLY (never the internal stores). Empty = system trust store. There is deliberately
    # no way to disable verification: a sovereign platform does not ship verify=False.
    model_ca_bundle: str = ""
    # Max generation tokens. Reasoning models (Qwen-thinking/QwQ/R1) spend a large budget
    # on <think>…</think> before answering, so keep this generous; raise it if answers come
    # back empty/truncated for a reasoning model.
    llm_max_tokens: int = Field(default=1024, ge=64)
    # Per-request read timeout (seconds) for the chat/completions call. This is a HARD ceiling on
    # how long a single generation may take, and it interacts with llm_max_tokens: at a measured
    # ~38 output tokens/s on the UAT endpoint at concurrency 12, a 4000-token cap needs ~105 s and
    # a 6000-token cap ~158 s. Raise this BEFORE raising a token cap — a request that outlives the
    # timeout is lost entirely, and a map/reduce stage silently drops that section from the answer.
    # Was 120 s, which left ~15 s of margin at the shipped 4000-token map cap: not enough. One of
    # two runs at that cap lost 8 of 54 map calls to the timeout, and a 6000-token comparison lost
    # 102 calls across two runs — every one read as "endpoint unreachable" while the endpoint was
    # healthy. At 240 s all four arms completed 47 of 47 calls with zero failures. A healthy call
    # is unaffected (it returns long before either ceiling); what grows is how long a genuinely
    # stalled request takes to give up, so watch elapsed_s and the timeout count after a deploy.
    llm_timeout_seconds: float = Field(default=240.0, gt=0)
    # Reasoning ("thinking") budget for a reasoning chat model (e.g. sarvam-105b): "low"/"medium"/
    # "high" set the depth; "off" DISABLES thinking entirely (sends reasoning_effort=null). Thinking
    # is billed as completion tokens and, under a low max_tokens cap, can consume the whole budget
    # before the answer — leaving an empty completion. Turn OFF for deterministic paths (NL→SQL).
    llm_reasoning_effort: str = "medium"

    # --- adaptive answer depth (FR-D5/E2) — the router's "heavy" path answers comprehensively ---
    # A complex/comparative/"list everything" question needs MORE evidence and a LONGER answer:
    # the router's heavy path retrieves heavy_top_k passages (vs the query's top_k) and lets the
    # model write up to heavy_max_tokens with a "be thorough & complete" instruction. Simple
    # questions stay fast + concise. Defaults are tuned to COMPLETE on a slow shared endpoint
    # (a long output hits the endpoint's own ~60s timeout → 504); the thorough prompt favours a
    # COMPLETE concise list over long prose. Raise both on a fast GPU model (e.g. 20 / 3000).
    heavy_top_k: int = Field(default=10, ge=1, le=100)
    # Passages a normal question retrieves, and therefore what the prompt can hold.
    #
    # MEASURED, not chosen: on the SOP golden set (24 questions x 3 runs, semantic cache off)
    # 10 scored 91.7%, 16 scored 95.8%, and 24 scored 95.8% -- so 16 is where the gain stops.
    # Refusals stayed at 100% throughout, which is the number that had to hold: more context is
    # only safe if it does not start manufacturing answers to unanswerable questions.
    query_top_k: int = Field(default=16, ge=1, le=50)
    heavy_max_tokens: int = Field(default=2048, ge=256)

    # --- chunking (FR-C1) — how a document is cut before embedding ---
    # ONE declared default. These used to be four literals across four API entry points
    # (/ingest said fixed, reindex/sync/bulk said clause) plus three more in the pipeline,
    # crawler and MCP server, so how a document was chunked depended on which door it came
    # through — and a re-index silently converted it to whatever reindex happened to say.
    # The strategy actually used is recorded on the document, and a re-index reproduces it
    # unless the caller explicitly asks for a different one.
    chunk_strategy: str = Field(default="clause", pattern="^(fixed|clause|parent_child)$")
    # FIXED only: character window and its overlap. Overlap must stay below size or the
    # window never advances (the chunker refuses it).
    chunk_size: int = Field(default=900, ge=64, le=8192)
    chunk_overlap: int = Field(default=150, ge=0, le=4096)
    # CLAUSE only: the size the accumulator aims for. NOTE this is currently a FLOOR, not a
    # cap — the buffer is tested after a clause is appended, so a chunk can overshoot by a
    # whole clause. Owned by the oversized-chunk task.
    chunk_max_chars: int = Field(default=700, ge=64, le=8192)

    # --- parent/child chunking (FR-C2, Option B) — small children + a per-section context bank ---
    # Additive + flag-gated + per-KB opt-in: a KB adopts it by recording chunk_strategy=parent_child
    # and reindexing through the existing embedder-switch loop. When OFF (default), nothing changes:
    # get_chunker is only asked for PARENT_CHILD when a KB explicitly selects it, and _parent_expand
    # (query time) no-ops. The three sizing/scope knobs below are per-KB overridable via
    # kb_threshold_overrides (_THRESHOLD_KEYS) so a pharma table KB and an HR prose KB can differ.
    parent_child_enabled: bool = False
    # Children are TIGHT for retrieval precision (below the CLAUSE 700 so a narrow lookup matches a
    # specific span and the abstain gate scores something concrete).
    parent_child_child_max_chars: int = Field(default=350, ge=64, le=8192)
    # Parents align with sections._MAX_SECTION_SPAN (2000): a larger section becomes several bounded
    # parents, a smaller one becomes a single parent = the section.
    parent_child_parent_max_chars: int = Field(default=1800, ge=128, le=16384)
    # Query-time cap on distinct parents attached to a result (ordered by best child score) so the
    # prompt's CONTEXT bank cannot blow the budget.
    parent_child_max_parents: int = Field(default=5, ge=1, le=50)
    # Aggregate CHARACTER budget for the whole parent CONTEXT block (rag-core is pure — no tokenizer
    # at query time — so a deterministic char budget, not a token count). Parents are kept best-child-
    # score-first until the total would exceed this; the rest are dropped. Bounds max_parents times
    # parent_max_chars (~9000) down to something that does not crowd the child SOURCES.
    parent_child_context_max_chars: int = Field(default=5000, ge=256, le=32768)

    # Similarity at which two chunks are the same text for practical purposes. Measured on a
    # real corpus, genuine reprints sit at 0.88-1.0 and unrelated policy prose below 0.3, so
    # anything in between is a wide, empty gap. Raise it to report only verbatim reprints.
    near_duplicate_threshold: float = Field(default=0.85, ge=0.5, le=1.0)

    # --- grounding (grounded-or-abstain hard gate) ---
    # Calibrated 2026-07-30 against `rerank_model` above — see docs/calibration.md. It is a
    # claim about ONE model's score distribution, which is why the reranker refuses to serve a
    # different model than the one configured: changing that without re-running
    # scripts/calibrate_abstain.py silently moves the gate.
    #
    # Measured on acme__hr_policies (10 answerable / 10 unanswerable): answerable min 0.4059,
    # unanswerable max 0.0575 — a clean gap, with any floor in 0.10-0.40 scoring 10/10 and
    # 10/10. 0.15 is chosen for margin, not for the score: ~7x above the best unanswerable and
    # ~2.7x below the weakest answerable. The previous 0.35 also scored 10/10 but sat 0.05
    # under the weakest real question, so one slightly harder question would have fallen off.
    abstain_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    # DEFAULT OFF (0 = disabled) -- the hypothesis behind it did not survive measurement.
    #
    # The idea: a model answering "I don't know." while retrieval scored 0.92 is refusing the
    # PHRASING, not the evidence, so re-framing a bare noun phrase as an instruction and trying
    # once more should rescue it. One run supported that. Five runs per phrasing did not: the
    # bare phrase and the imperative one both abstained 5/5, and the single earlier success was
    # noise in a non-deterministic reasoning model.
    #
    # The real cause of that case was elsewhere -- "all the 10 SOPs" names no identifiers, so the
    # enumeration coverage quota never fires and the spans genuinely do not hold ten items; the
    # model is right to refuse. Naming them ("explain PBSOP001 ... PBSOP010") answers with nine
    # citations. Discovering unnamed items is a real feature, not a retry.
    #
    # The mechanism is kept, tested and one setting away, because it may still hold for other
    # models or corpora -- but it ships off until something measures it. Above this score the
    # generator re-frames once and retries (see as_explicit_request); below it a refusal is
    # agreeing with the gate and must always stand.
    refusal_retry_min_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # The SAME gate, for the scale a `lexical` reranker produces. It is a separate number
    # because it is a claim about a different scorer, and one value across two scales is the
    # P50 bug class (see grounding.gate_for).
    #
    # Measured 2026-07-31, 20 questions against a real 40-page RFE (NICSI), term-overlap
    # scoring: answerable 0.5714-0.8750, unanswerable 0.3333-0.6250. The populations OVERLAP —
    # unlike the cross-encoder there is NO threshold that separates them. At the cross-encoder's
    # calibrated 0.15 it blocked 0/10 unanswerable ("what is the of on" overlaps every chunk);
    # 0.60 is the best available operating point at 9/10 blocked for 8/10 answerable kept, and
    # abstaining is the safe direction for a grounded-or-abstain platform.
    #
    # Read that as: the lexical reranker is a WIRING TWIN, not a ranker a regulated deployment
    # should ground answers on. `build_reranker` already warns when it is selected.
    abstain_threshold_lexical: float = Field(default=0.60, ge=0.0, le=1.0)

    # --- faithfulness verification (FR-E4) — three-way routing thresholds ---
    # score >= faithful_threshold -> GROUNDED; flag_threshold..faithful -> FLAGGED_FOR_REVIEW;
    # < flag_threshold -> ABSTAINED. Per-tenant overrides via the catalog (TODO: calibrate on
    # the golden set). Conservative defaults.
    faithfulness_enabled: bool = True
    # lexical (token overlap, CI default) | llm (LLM-as-judge) | nli (entailment cross-encoder).
    # NLI is the accuracy upgrade: a dedicated Natural-Language-Inference model checks whether
    # each answer sentence is ENTAILED by the cited spans — not token overlap, not the generator
    # grading itself. Crucially it is MODEL-AGNOSTIC: the faithful/flag thresholds calibrate
    # against the NLI model, so swapping the generative LLM needs no threshold recalibration.
    faithfulness_backend: str = "lexical"  # lexical | llm | nli — the PLATFORM default checker
    # IDF-weighted share of a sentence's words that ONE retrieved span must contain for it to count
    # as supported (rag_core.anchoring). NLI alone only detects CONTRADICTION, so a
    # true-but-unrelated sentence passes (live: a parametric answer scored 1.0 citing a source that
    # never mentioned it). 0 disables.
    # 0.25, measured 2026-09-11 against the unweighted anchor at its old 0.30 on two sets:
    # 138 sentences of REAL grounded answers (wrongly failed: 14 -> 13) and 39 labelled sentences
    # incl. deliberately synonym-heavy paraphrases (invented sentences caught: 4 -> 6 of 15).
    # Better on both at once; 0.30 would have caught more fabrications but failed 7 real
    # sentences the old anchor passed, and 0.40 more still. A lexical check cannot tell a heavy
    # paraphrase from an invention -- the threshold sets that trade, and real answers decide it.
    faithfulness_anchor_threshold: float = 0.25
    # BM25-style length normalization of the anchor, penalise-only (a short span is never boosted --
    # a symmetric version picks dense table-of-contents chunks). OFF by default, by measurement: it
    # changed no outcome on a corpus chunked at <= 700 characters, and it demonstrably discounts a
    # VERBATIM quote whose passage is merely longer than its neighbours. Available for corpora with
    # long evidence spans; set 0.5 and re-measure before relying on it.
    faithfulness_anchor_length_norm: float = Field(default=0.0, ge=0.0, le=1.0)
    # One bounded re-attempt when an answer fails its own faithfulness check: the model is told
    # which sentences the sources did not support and rewrites using only what they do. Kept only
    # if it verifies better, so it can never degrade an answer. Costs one extra generation on the
    # answers that were going to be flagged or refused anyway.
    self_correction_enabled: bool = True
    # When a KB has connected its own model (FR-E1), judge faithfulness with THAT model
    # instead of the platform lexical fallback — a real generative answer deserves a real
    # judge, not a keyword overlap (fixes over-flagging). Adds one model call per answer.
    faithfulness_tenant_model_judge: bool = True
    faithful_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    flag_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    # Notation verbatim-grounding (FR-E4, Slice 4). EXTENDS the always-on numeric check: alongside
    # invented figures, flag a unit-bearing quantity (5 g vs a source's 5 mg), a chemical formula
    # (H₂SO₃ vs H₂SO₄, sub/superscript↔ASCII normalised) or a SMILES string that appears in no
    # cited span. Default OFF ⇒ the faithfulness checker is byte-identical to the numeric-only one.
    # Per-KB adaptive: overridable via the `notation_grounding_enabled` key in tenant_config
    # (`_THRESHOLD_KEYS`, stored 1.0/0.0) so a pharma KB turns it on hard while an HR KB leaves it
    # off. Enabled in docker-compose.dev.yml. See rag_core.notation for the (conservative,
    # never-false-flag) detection.
    notation_grounding_enabled: bool = False
    # NLI checker (faithfulness_backend == "nli"): self-hosted TEI /predict cross-encoder.
    # Host must be egress-allowlisted (sovereign). support_threshold = per-sentence P(entailment)
    # above which a sentence counts as supported.
    nli_url: str = "http://nli-checker:80"
    # How much source text is put in front of ONE answer sentence when judging entailment.
    # Bounded in characters because the NLI cross-encoder has a fixed token window (mDeBERTa-v3:
    # 512) and degrades to a confident CONTRADICTION once the premise runs past it — measured on
    # the HR corpus with a sentence quoted verbatim from its source: 400ch -> 0.969, 700ch ->
    # 0.950, 1400ch -> 0.981, 1983ch -> 0.279. ~1000 sits above one 700-char chunk (so a whole
    # chunk always fits) and well inside the window once the hypothesis is added.
    nli_premise_max_chars: int = Field(default=1000, ge=200)
    # Ceiling on the passages a DECOMPOSED query may carry into generation. The merge used to cap
    # at top_k however many sub-questions ran, so a question naming ten things could hold at most
    # ten passages and the strongest one or two took most of them: measured live, a ten-SOP
    # question was answered for two of the ten and declared the rest absent, though each answered
    # in full when asked alone. The merge is round-robin, so a larger budget is spread across the
    # sub-questions rather than handed to the loudest. Bounded because every passage is context
    # the generation model pays for.
    transform_span_budget: int = Field(default=40, ge=1)
    # ENUMERATION COVERAGE. A question naming several identifiers ("PBSOP001 ... PBSOP010") gets
    # one retrieval per item so each is actually looked for, instead of the loudest two taking the
    # whole passage budget and the answer reporting the rest as absent. Bounded twice: how many
    # items are honoured, and how many passages each keeps.
    coverage_max_items: int = Field(default=12, ge=1)
    coverage_per_item: int = Field(default=3, ge=1)
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    nli_support_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    # Per-/predict timeout. mDeBERTa on CPU is slow per sentence (long premise = union of spans),
    # so this is generous; a timeout falls back to the lexical checker rather than failing the answer.
    nli_timeout_seconds: float = Field(default=60.0, gt=0)
    # How many /predict calls may be in flight at once. The checker scores every answer sentence
    # concurrently, which is right for the 5-20 sentences a single-pass answer has and wrong for
    # the hundreds a RAG v2 map-reduce answer has: measured on the AEGCL corrigendum, a 377-claim
    # aggregate answer opened more requests than httpx's default 100-connection pool, every extra
    # one waited for a slot, the wait counted against `nli_timeout_seconds`, and the PoolTimeout
    # took the WHOLE answer down the fail-soft path to the lexical checker. Silent, because that
    # fallback is a warning log and a plausible score. Kept under httpx's default pool so a
    # sentence never waits on a connection, and the CPU cross-encoder is the bottleneck anyway —
    # past its core count more concurrency buys nothing.
    nli_max_concurrency: int = Field(default=16, ge=1)
    # How many points one Qdrant upsert request may carry. Qdrant caps a request at
    # `max_request_size_mb` (32 by default) and answers an oversized one with a 400 while the
    # client is still uploading, which surfaces as a bare httpx ReadError and a 500 from ingest —
    # no partial write, nothing indexed, and no mention of size anywhere in the error. One
    # document can easily exceed it: measured on an 80-page tender, RAG v2 ingest turned 68 text
    # chunks into 1801 points (1425 of them table ROWS) at ~20 KB each = 36.9 MB in a single
    # call. 256 keeps a request near 5 MB at 1024 dims, well inside the cap and inside the
    # headroom a larger embedding model or a wider payload would eat.
    qdrant_upsert_batch: int = Field(default=256, ge=1)

    # --- Coverage-aware generation (the direct completeness fix) — additive, default OFF ---
    # After a GROUNDED answer, detect whether the model OMITTED a distinct, clearly-relevant
    # retrieved point (retrieval had all N, the answer listed fewer) and, if so, do ONE targeted
    # re-prompt to add it back — using ONLY the same retrieved spans + the same citation rules,
    # accepted only if it reduces the uncovered-span count WITHOUT regressing the faithfulness gate
    # (grounded-or-abstain is never loosened). Default OFF ⇒ answer() is byte-identical: no extra
    # LLM call, no audit key, no trace step. Costs one extra generation ONLY when a drop is
    # detected — opt-in and bounded to a single re-prompt.
    coverage_check_enabled: bool = False
    # A span counts as "should be covered" only when its rerank score clears this floor — a
    # low-score span the model correctly omitted must not be forced back in (that injects noise).
    coverage_min_score: float = Field(default=0.5, ge=0.0, le=1.0)
    # A high-score span is "uncovered" when fewer than this fraction of its salient content terms
    # appear in the answer — i.e. its point is largely absent, not merely paraphrased in part.
    coverage_ratio: float = Field(default=0.4, ge=0.0, le=1.0)
    # Hard cap on how many omitted points one re-prompt is asked to add back (highest-score first).
    coverage_max_spans: int = Field(default=6, ge=1)

    # --- Strict, adaptive completeness guard (Slice 1) — additive, default OFF ---
    # Supersedes the single coverage re-prompt above when ON: a GROUNDED answer that is TRUNCATED
    # (finish_reason=length or a structural mid-sentence/dangling-enumeration cutoff) is CONTINUED in
    # bounded steps, and a GROUNDED-but-incomplete answer runs a bounded coverage LOOP; if neither
    # can be resolved within the bounds the answer is MARKED (GroundedAnswer.completeness.complete=
    # false + reason) rather than shipped as whole. Every bound/budget below is a CEILING — the value
    # actually used is derived per query/KB (demand, production, spans above the relevance bar), never
    # a bare literal in the hot path. Default OFF ⇒ answer() is byte-identical (no continuation, no
    # coverage loop, no completeness field). All three ceilings are per-KB overridable via
    # kb_threshold_overrides. Enable in deploy compose (RAG_COMPLETENESS_GUARD_ENABLED=true).
    completeness_guard_enabled: bool = False
    # Ceiling on bounded continuation attempts for a truncated completion. The ACTUAL bound is derived
    # per query from its demand (an exhaustive "list every" question earns the full ceiling, an
    # ordinary one half) and every attempt early-exits the moment truncation clears — so this caps a
    # genuinely still-cut-off answer, it is not a fixed N. Per-KB overridable.
    completeness_max_continuations: int = Field(default=3, ge=1, le=10)
    # Each continuation escalates the token budget by this factor, compounded per attempt and capped
    # at heavy_max_tokens — cheaper than regenerating the whole answer at a huge budget up front, and
    # relative to what was actually produced rather than an absolute number. Per-KB overridable.
    completeness_continuation_budget_factor: float = Field(default=1.5, ge=1.0, le=4.0)
    # Ceiling on the bounded coverage re-prompt LOOP: each pass targets the STILL-missing relevant
    # spans and is accepted only if it reduces the uncovered count WITHOUT regressing faithfulness
    # (the better answer is always kept, never a worse one). Per-KB overridable.
    completeness_coverage_max_loops: int = Field(default=3, ge=1, le=10)
    # Hard cap on the TOTAL extra guard-driven LLM calls across a SINGLE request (continuation +
    # coverage passes, summed across docs on the doc-summary fan-out). Backstops even a pathological
    # per-KB override so the guard can never fan out to hundreds of calls; when exhausted the guard
    # stops looping and marks the answer complete=false (reason "guard_budget_exhausted"). Sized for
    # one query's continuation ceiling + coverage ceiling with headroom; per-KB overridable.
    completeness_max_guard_calls: int = Field(default=12, ge=1, le=100)
    # Faithfulness TOLERANCE band for the never-regress check: a genuinely more-complete
    # continuation/coverage rewrite is accepted even if the (over-flagging) small NLI judge dips the
    # score by up to this much, so a 0.01 wobble does not reject a real improvement. Tiny by design;
    # 0 restores the strict "score must not drop at all" rule. Per-KB overridable.
    completeness_faithfulness_tolerance: float = Field(default=0.02, ge=0.0, le=0.2)

    # --- Tier 3: agentic coverage-verification for the WHOLE-DOCUMENT read (additive, default OFF) ---
    # Master flag gating ONLY (a) the substantive-passage-aware whole-document completeness measure
    # (boilerplate — page markers, letterhead footers, SOP master-listing rows, revision-history
    # scaffolding, duplicate headers — no longer counts toward expected/covered) and (b) the bounded
    # coverage-refine loop that re-reads any genuinely-uncovered SUBSTANTIVE passages through the SAME
    # grounded whole-document seam and folds their omitted points in as new cited sentences. OFF ⇒ the
    # whole-document path is BYTE-IDENTICAL to today (boilerplate still counts, no refine loop runs);
    # it does NOT touch the normal RAG path's coverage behaviour. Reuses completeness_coverage_max_loops
    # for the loop bound. Enable in deploy compose (RAG_COVERAGE_VERIFY_ENABLED=true).
    coverage_verify_enabled: bool = False

    # --- Embeddable widget (FR-I6) — short-lived session token minted from a read API key ---
    widget_session_ttl_seconds: int = Field(default=900, ge=60, le=86400)

    # --- OCR & document intelligence (FR-B1/B2) — pluggable engine for scanned docs ---
    # "noop" refuses scanned docs clearly (dev default); "tesseract" OCRs them when the
    # tesseract binary + pytesseract/pdf2image are present (prod image / GPUaaS pool).
    # "paddleocr"/"trocr" are optional stronger engines (better on low-quality scans and
    # handwriting respectively) — import-guarded, plugged in via their extras.
    ocr_backend: str = Field(default="noop", pattern="^(noop|tesseract|paddleocr|trocr|chandra|chandra_api)$")
    ocr_lang: str = "eng"  # tesseract lang codes, e.g. "eng+hin" for Hindi (FR-B1 Indic)
    # Image preprocessing before OCR (deskew/grayscale/upscale/denoise/binarize) — the single
    # biggest accuracy lever on faint receipts, halftone print and phone photos. CPU-only.
    ocr_preprocess: bool = True
    # Hard Otsu binarization — OFF by default: Tesseract 5's LSTM reads grayscale better, and
    # binarizing destroys thin strokes on clean-ish scans. Turn ON only for noisy bitonal faxes.
    ocr_binarize: bool = False
    ocr_psm: int = Field(default=3, ge=0, le=13)  # tesseract page-segmentation mode (3=auto)
    ocr_oem: int = Field(default=1, ge=0, le=3)  # OCR engine mode (1=LSTM)
    # PaddleOCR only: folder of pre-mirrored det/rec/cls models for an air-gapped/sovereign
    # deployment (empty ⇒ PaddleOCR resolves its own models on a one-time bootstrap).
    ocr_model_dir: str = ""
    # A PDF page with fewer real characters than this has no usable text layer and is routed to
    # OCR — decided PER PAGE, so a diagram-heavy but born-digital PDF keeps the text layer on the
    # pages that have one. Raise it for corpora whose pages carry only a caption or a stamp;
    # lower it if sparse-but-real pages are being sent to OCR needlessly. Must stay in step with
    # svc_doc_intelligence.classify.DEFAULT_SCANNED_CHARS_PER_PAGE (pinned by a test).
    ocr_scanned_chars_per_page: int = Field(default=20, ge=0)
    # How PDF pages that carry embedded images are handled. "text_only" (default) reads a
    # page's text layer directly — embedded images on otherwise-texty pages are NOT read, so
    # figures/tables/charts rendered as images are missed. "ocr" ALSO sends any page with
    # image content through the OCR engine (Chandra) so those embedded figures/tables/charts
    # are captured, then merged back per page. Env var: RAG_PDF_IMAGE_PAGES.
    pdf_image_pages: str = Field(default="text_only", pattern="^(text_only|ocr)$")
    # --- the ingestion OCR check (svc_doc_intelligence.ocr_check) ----------------------------
    # Every document is checked for text that exists only as pixels. Image files and scanned
    # pages always go to OCR; these settings govern EMBEDDED images — a scan pasted onto a texty
    # PDF page, a screenshot on a slide, a signed letter inside a .docx. Measured live before the
    # check existed: a JPEG e-ticket was indexed as a 32-character caption while OCR, which read
    # 4,544 characters from it, was never called. Turn OFF only to trade that recall for ingest
    # speed on a corpus known to hold no text-bearing images.
    ocr_embedded_images_enabled: bool = True
    # An embedded image whose SHORTER side is below this many of its own pixels is decoration
    # (logo, icon, bullet) and is not sent to OCR.
    ocr_embedded_min_edge_px: int = Field(default=200, ge=1)
    # On a PDF page, an image must ALSO cover at least this share of the page to be read, so a
    # high-resolution logo printed thumbnail-small does not cost a model call on every page.
    ocr_embedded_min_page_share: float = Field(default=0.10, ge=0.0, le=1.0)
    # Hard bound on embedded images OCR'd per document. Chandra takes ~30 s an image; past the
    # bound the remaining images are NOT read and the document is stamped
    # `ocr_image_cap_reached`, so the gap is visible rather than silent.
    ocr_embedded_max_images: int = Field(default=20, ge=0)
    # How many OCR calls one document may run at once (embedded images).
    ocr_concurrency: int = Field(default=4, ge=1, le=32)
    # Output ceiling for ONE OCR call. Not a cost knob — a liveness one. A vision model that
    # cannot read an image (measured live on a handwritten chemistry worksheet) does not fail: it
    # transcribes a few lines and then repeats one token until its budget runs out. At the
    # platform's general 4000-token budget that took over 60 s and the AIOps gateway returned 504,
    # so the whole ingest failed on a page nothing could have read anyway. Measured on that image:
    # 1200 tokens = 28 s, 2000 = 46 s, 2500 = 57 s, 4000 = gateway 504. 2000 leaves real margin
    # under the 60 s ceiling and is well above a dense A4 page of transcription (~1200 tokens).
    ocr_max_tokens: int = Field(default=2000, ge=64)

    # --- SIEM audit forwarding — banking pack. Ship the audit chain to the bank's SOC. OFF by
    # default; the endpoint host must be egress-allowlisted (sovereign). ---
    siem_enabled: bool = False
    siem_transport: str = Field(default="webhook", pattern="^(webhook|syslog)$")
    siem_endpoint: str = ""  # webhook URL (e.g. Splunk HEC) or host:port for syslog
    siem_format: str = Field(default="json", pattern="^(json|cef)$")
    siem_token: str = ""  # optional auth token (e.g. Splunk HEC token)

    # --- Maker-checker (dual control) — banking pack. Default OFF: nothing is gated. ---
    dual_control_enabled: bool = False
    # Which actions require a second-admin approval when enabled (comma list; empty = all):
    # publish_kb, ingest_classified, purge, erasure, security_change.
    dual_control_actions: str = ""
    # Pages whose mean OCR confidence is below this are flagged low-confidence for review (FR-B1).
    ocr_min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # Structured field extraction (dates, amounts, GSTIN/PAN/IFSC) into filterable metadata.
    field_extraction_enabled: bool = False

    # --- PII detection & masking / redaction (FR-B6/G2) — always on; de-identify at ingest ---
    # Only de-identified text is ever chunked/embedded/indexed. Backend "regex" is
    # dependency-free + checksum-aware (Aadhaar Verhoeff, card Luhn); "presidio" adds NER
    # (PERSON/LOCATION) — opt-in heavy dep. Strategy: redact ([REDACTED:TYPE]) / mask (keep
    # last 4) / hash (deterministic salted token — same value → same token). Entities: empty
    # = the full default set; a comma list (e.g. "EMAIL,PHONE,AADHAAR") narrows it.
    pii_masking_backend: str = Field(default="regex", pattern="^(regex|presidio)$")
    pii_strategy: str = Field(default="redact", pattern="^(redact|mask|hash)$")
    pii_entities: str = ""
    pii_hash_salt: str = ""  # salt for the 'hash' strategy; set for stable tokens across restarts
    # PII-driven classification floor (fail-closed net for "everyone can ingest"): when a document's
    # detected structured PII (Aadhaar/PAN/card/… — checksum-validated by the masker) implies a
    # HIGHER data classification than requested, raise it to that floor (or block). Master flag; the
    # per-INGEST toggle (IngestRequest.pii_autoclassify) turns it on/off for a given upload.
    pii_classification_enabled: bool = False
    # Mode when auto-classify runs: raise = silently bump to the floor and index; block = reject so
    # the user reclassifies. A floor ABOVE the ingestor's own clearance is ALWAYS blocked regardless
    # (can't store content more secret than the ingestor is cleared to read).
    pii_classification_policy: str = Field(default="raise", pattern="^(raise|block)$")

    # --- Indic multilingual (content) — detection + translation seam ---
    # Detect a document's language (script-based) and store it as filterable metadata.
    # Translation "noop" = off; "llm" translates via the KB's own model. Multilingual
    # embeddings/generation are a model swap via embed_model / the model registry.
    language_detection_enabled: bool = False
    translation_backend: str = Field(default="noop", pattern="^(noop|llm)$")

    # --- Governed NL→SQL over a tenant RDBMS (FR-D10) — additive, default OFF ---
    # A tenant registers a read-only DB connection (DSN + table allowlist). NL questions are
    # translated to SQL by the tenant's model, then every statement is sqlglot-validated
    # (SELECT-only, allowlisted tables, forbidden-function denylist) and bounded before it runs.
    # The DB host is an external dependency → egress-allowlisted. Read-only by construction.
    nl2sql_enabled: bool = False
    # Generation budget for the NL→SQL model call. Reasoning models think before writing SQL;
    # a small cap truncates mid-<think> and the parse fails ("could not parse SQL").
    nl2sql_max_tokens: int = Field(default=1500, ge=64)
    # Unified KB (documents + database): when a KB has a registered DB connection, a query runs
    # NL→SQL alongside document retrieval and fuses the structured result into the SAME grounded
    # answer (cited as a database source). Requires nl2sql_enabled. Off by default; fail-safe to a
    # documents-only answer if the DB query errors or returns nothing.
    unified_kb_enabled: bool = False
    # Graph bridge: upsert the fused structured facts into the KB knowledge graph so a DB / NoSQL /
    # REST / SharePoint entity shares a node with the documents that mention it (cross-source entity
    # resolution) — making the connection visible + traversable in GraphRAG. Needs graphrag_enabled.
    graph_bridge_enabled: bool = False
    # Auto-route "explain/generate the architecture of X" questions through the normal /query path to
    # the architecture flow (typed extraction + Mermaid/PlantUML + plain-language explanation). Off by
    # default; when off, /query is unchanged and the /kb/{id}/architecture endpoint still works.
    architecture_intent_enabled: bool = False
    # Token budget for the architecture extractor (typed JSON) + explanation. Generous so a LARGE
    # architecture's JSON isn't truncated AND a reasoning model (spends the first chunk of the budget
    # thinking) still has room to emit it — 1200 left sarvam-105b returning empty. NOTE the true
    # ceiling is the MODEL's own max output (sarvam-105b rejects >~4096 with a 400) — this can't be
    # "uncapped"; extraction also RETRIES on an empty completion (reasoning models are stochastic).
    architecture_max_tokens: int = Field(default=4000, ge=512)
    # Cap on the number of components a single architecture diagram renders. A big system's
    # extraction can overrun this; when it does the extractor KEEPS the most-connected components
    # (by edge degree) and notes "N of M shown" rather than truncating arbitrarily. Raised from the
    # old hard-coded 12 so larger systems keep their structure; bounded so a runaway extraction can't
    # emit an unreadable hairball.
    arch_max_components: int = Field(default=20, ge=1, le=100)
    # How many EXTRA extraction attempts on an empty completion (each retry is one more LLM call,
    # with early-exit on the first non-empty result). A reasoning model (sarvam-105b) intermittently
    # spends its whole budget in <think> and returns nothing — one attempt succeeded only ~1/3 of the
    # time in practice, so the auto-render path needs several tries to be reliable. Bounded to keep
    # the worst-case latency finite; the console still degrades to an on-demand chip if all fail.
    architecture_extract_retries: int = Field(default=5, ge=0, le=10)
    # Narrate the architecture in LLM prose (a SECOND slow reasoning-model call) vs the instant
    # deterministic summary. Default OFF for latency — the diagram is the deliverable; turn on for
    # richer non-technical explanation where the extra ~25s (sarvam) is acceptable.
    architecture_llm_explanation: bool = False
    # Agent orchestrator (Agentic RAG + MCP): a dynamic plan-and-execute agent that plans over the
    # in-process tool catalog, has a critic review the plan, and gates CRITICAL tool calls behind a
    # dual-control human approval. Off by default; POST /v1/agent/run 404s when disabled.
    agent_orchestrator_enabled: bool = False
    # Auto-route "explain/optimize the process/workflow of X" questions through /query to the
    # process-flow understanding + gap-analysis flow (typed process + Mermaid/PlantUML + gaps +
    # recommendations). Off by default; the POST /kb/{id}/process endpoint works regardless.
    process_intent_enabled: bool = False
    nl2sql_max_rows: int = Field(default=1000, ge=1, le=10000)
    nl2sql_statement_timeout_ms: int = Field(default=5000, ge=100)

    # --- Agentic-workflow runtime (FR-F) — additive, default OFF ---
    # Declarative multi-step workflows (RAG answer / governed SQL / retrieve / rules-based
    # branching) with a recorded decision trace. Deterministic — same inputs, same path.
    agentic_workflows_enabled: bool = False

    # --- Personal (user-owned) knowledge bases — additive, default OFF ---
    # When ON, any org member can create their OWN knowledge base (namespace-isolated like every
    # KB) and share it with other members of the same org — the creator is auto-appointed its KB
    # manager, so sharing reuses the existing access-grant flow. The governed dept-KB path
    # (owner/admin-only create) is untouched; this only adds a self-service lane.
    personal_kb_enabled: bool = False

    # --- Multimodal (vision) RAG — additive, default OFF ---
    # When ON, an image document (image/* MIME) is auto-captioned by a sovereign vision LLM
    # and the caption is indexed like text (so a text query retrieves the image via its
    # caption), while the raw image bytes stay in the object store for the vision LLM to READ
    # at answer time. OFF ⇒ images take the existing OCR path, unchanged.
    multimodal_enabled: bool = False
    # What ACCOMPANIES an image's OCR text when multimodal is on. Every image is read by the OCR
    # engine either way (svc_doc_intelligence.ocr_check) — skipping that is what indexed a scanned
    # e-ticket as 32 characters of caption while OCR read 4,544 from the same file:
    #   caption = ALSO add the vision LLM's description of what the picture shows (photos, scenes,
    #             diagrams whose meaning is not in their words)
    #   ocr     = the OCR reading alone (screenshots, forms, scanned pages — a caption of a page
    #             of text adds noise, and on this deployment the vision model sometimes returns
    #             an echo of its own prompt)
    image_extract_backend: str = Field(default="caption", pattern="^(caption|ocr)$")
    # Vision LLM (reads image bytes at generation). chandra = the Datalab VLM that also serves
    # OCR (one model, two roles — recommended); openai_compat = any self-hosted VLM via vLLM;
    # echo = deterministic zero-model twin for CI/demo.
    # echo | chandra (self-hosted vLLM) | chandra_api (hosted Chandra via Datalab) | openai_compat
    vision_llm_backend: str = "echo"
    vision_llm_url: str = "http://vllm-vision:8000/v1"
    vision_llm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    # Optional API key for a HOSTED vision endpoint (e.g. Sarvam's API) — lets us test against
    # a managed model before self-hosting on a GPU. Empty for self-hosted vLLM. The host must
    # still be egress-allowlisted (sovereignty is not bypassed by having a key).
    vision_llm_api_key: str = ""
    vision_llm_api_key_header: str = "Authorization"
    # Diagram-structure extraction (additive, default OFF): for an IMAGE document, ALSO ask the
    # vision LLM to read the diagram's boxes/arrows and feed the parsed nodes/edges into the SAME
    # GraphRAG store _index_graph writes to (graph_extract_backend's co-occurrence pass still runs
    # unchanged on the image's OCR/caption text) — so a diagram's STRUCTURE, not just its text
    # labels, becomes graph-walkable. No-op unless BOTH this and graphrag_enabled are true; never
    # replaces OCR/caption text (hybrid BM25 keeps indexing that exactly as before).
    diagram_graph_enabled: bool = False

    # --- Chandra (Datalab) — ONE vision model serving both OCR and the vision gateway ---
    # A VLM-based document model: strong OCR (tables/forms/handwriting/reading-order, Indic) AND
    # open image Q&A. Set RAG_OCR_BACKEND=chandra to use it for OCR and vision_llm_backend=chandra
    # for the vision gateway — both read this ONE config block, so Chandra is configured once.
    # Speaks the OpenAI-compatible vision chat API (self-hosted via vLLM, or a hosted endpoint).
    chandra_url: str = "http://chandra:8000/v1"
    chandra_model: str = "datalab-to/chandra"
    chandra_api_key: str = ""  # for a hosted Chandra API; empty for self-hosted. Host still allowlisted.
    chandra_api_key_header: str = "Authorization"

    # --- Chandra via the HOSTED Datalab API (backend `chandra_api`) — async /convert ---
    # `RAG_OCR_BACKEND=chandra_api` / `vision_llm_backend=chandra_api` run Chandra on Datalab's
    # cloud (submit→poll) — for testing before self-hosting. Same model as the `chandra` backend,
    # different serving shape. The connection vars keep the `datalab_` prefix because it IS
    # Datalab's endpoint + key (datalab.to, X-API-Key). Host egress-allowlisted.
    # EMPTY by default. A sovereign deployment must not carry a foreign endpoint it never
    # chose: with a default here, anything that reported "the OCR endpoint" named
    # www.datalab.to even while OCR ran on self-hosted Chandra — which is exactly what the
    # Models page showed an operator who had removed Datalab weeks earlier. An operator who
    # genuinely wants the hosted API sets this explicitly, and the host must still be on the
    # egress allowlist.
    datalab_url: str = ""
    datalab_api_key: str = ""
    datalab_api_key_header: str = "X-API-Key"
    # CLIP-style cross-modal embedder (Phase-2 visual retrieval; caption path needs none).
    # Cross-modal (image<->text) recall: when ON, an image's CLIP vector is indexed at ingest
    # into a SEPARATE vision collection and a text query ALSO searches it, so an image is
    # retrieved by VISUAL similarity, not only its caption text. Needs multimodal_enabled too.
    # Default OFF — the caption path (multimodal_enabled) is unaffected. Phase-2 seam.
    vision_recall_enabled: bool = False
    vision_embed_backend: str = "local"  # local (twin) | tei
    vision_tei_url: str = "http://tei-vision:80"
    vision_embed_model: str = "nomic-ai/nomic-embed-vision-v1.5"
    vision_embed_dim: int = Field(default=768, ge=1)

    # --- Federated RAG federation (cross-tenant P2P RAG) — additive, default OFF ---
    # Opt-in cross-tenant/cross-department query federation, governed by publisher-approved
    # Ed25519-signed grants. OFF ⇒ the hard tenant-isolation 403 is the only cross-dept path
    # (today's behaviour). ON ⇒ a department with a valid grant can query the granting
    # department's KBs for the granted topics only. Isolation is never loosened — a grant is
    # the sole, revocable, auditable mechanism that opens a scoped door.
    federation_enabled: bool = False
    # This node's department identity in the federation mesh (defaults to the tenant at runtime).
    federation_dept_id: str = ""
    # Federation transport (ADR-0017): "local" retrieves co-located peer KBs in-process (default,
    # monolith); "mesh" routes to the local sovereign-node, which dials remote peers over libp2p.
    federation_transport: str = Field(default="local", pattern="^(local|mesh)$")
    # Base URL of THIS deployment's local sovereign-node outbound control API (mesh transport).
    # Local/intranet infrastructure, like the store URLs — NOT egress-guarded.
    federation_node_url: str = ""
    # Optional per-peer transport override as JSON: {"<dept>": {"transport": "mesh", "node_url": "..."}}
    federation_peers: str = ""
    # ADR-0017 L1: when ON, federated retrieval is filtered fail-closed to spans whose primary
    # taxonomy category is one of the grant's topics. OFF = today's behaviour (topics validated
    # by the mesh node but not content-filtered). Assumes grant.topics name the KB's categories.
    federation_topic_enforcement: bool = False
    # Per-peer timeout (seconds) for a mesh federated query. A slow/down granted peer is
    # reported unreachable after this, never allowed to hang the whole federated answer (T1).
    federation_peer_timeout_seconds: float = 8.0
    # A caller denied by >= this many peers in ONE federated query trips a FEDERATION_ALERT
    # (grant-probing signal for the SOC). 0 disables the anomaly alert.
    federation_denial_alert_threshold: int = 3
    # Max delegation chain depth for transitive grants (A->B->C...). 1 = direct grants only (no
    # re-delegation), 2 = one hop of re-sharing, 3 = A->B->C. Bounds how far a KB owner's data
    # can travel by delegation — the "bounded trust" ceiling. Each hop can only NARROW access.
    federation_max_grant_depth: int = 3
    # Issuer Ed25519 signing key (base64 raw 32 bytes). Empty ⇒ a deterministic dev key derived
    # from dev_jwt_secret (in-memory/CI only). Production injects a real key via secrets.
    # Platform-operator allowlist (SUPER_ADMIN): comma-separated emails that operate the fleet
    # (tenant control plane). Separation-of-duties: a super admin manages tenants, never reads
    # tenant DATA. Empty = no platform admins (self-serve only).
    platform_admin_emails: str = ""
    # Offline-update: the Acme RELEASE public key (base64 Ed25519), pre-installed at an
    # air-gapped site to VERIFY an update bundle came from Acme before loading it. Empty =
    # verification disabled (never in a real air-gapped deployment).
    update_release_public_key: str = ""
    grant_signing_key: str = ""
    # Key id (kid) for the ACTIVE signing key — recorded on every grant-issue audit event so
    # an operator can tell which key signed a grant (and when a retired key is safe to drop).
    grant_signing_key_id: str = "dev"
    # Retired-but-still-TRUSTED verification keys for a rotation OVERLAP: a JSON list of base64
    # raw Ed25519 PUBLIC keys. On rotation the OLD public key moves here so grants it already
    # signed keep verifying until they expire, while new grants are signed by grant_signing_key.
    # Drop a key once no unexpired grant was signed with it. Empty in dev/CI.
    grant_retired_verify_keys: str = ""

    # --- Provenance signing (trust receipts, model SBOM, residency attestation) ---
    # ONE provider-side Ed25519 key backs every signed provenance artifact the platform emits.
    # Empty -> a deterministic dev key derived from dev_jwt_secret (in-memory/CI only; never prod).
    provenance_signing_key: str = ""
    provenance_signing_key_id: str = "dev"
    # Emit a signed trust receipt per grounded answer, recorded on the audit chain. Off by default.
    trust_receipts_enabled: bool = False
    # Model SBOM: JSON list of {name, role, source_origin, license, weight_sha256, revision}
    # declaring the models this deployment runs. Empty -> /v1/sbom returns 404 (not configured).
    model_sbom_models: str = ""
    # Per-node residency attestation identity (also the SBOM deployment id).
    residency_node_id: str = ""
    residency_region: str = "India"

    # --- Air-gap runtime mode (fully offline deployment) ---
    # When on, the app refuses to START if any config would breach air-gap — remote egress
    # (smtp/siem/jwks), off-box model/OCR backends, or a missing update-verify key. The
    # preflight report is at GET /v1/platform/airgap/preflight. See rag_common.airgap.
    air_gap_enabled: bool = False

    # --- Indic cross-lingual retrieval ---
    # Ask in Hindi/Marathi, retrieve English/other-script content: the query is translated
    # (via the KB's own model) and retrieval runs on original + translation. Off by default.
    cross_lingual_enabled: bool = False
    cross_lingual_target_lang: str = "English"

    # --- Automated DPDP erasure (retention-driven) ---
    # A scheduled worker erases documents past their retention window (cascade + signed
    # certificate on the audit chain). Off by default; retention_days is the window when on.
    erasure_schedule_enabled: bool = False
    erasure_retention_days: int = Field(default=365, ge=1)

    # --- Async bulk ingestion — background job queue for volume (1000+ docs) ---
    # Uploaded files are staged to the object store and ingested by an in-process worker so the
    # request never blocks; progress is tracked and jobs resume after a restart. Concurrency
    # bounds how many documents embed at once (raise it on a GPU embedding tier).
    bulk_ingest_worker_enabled: bool = True
    bulk_ingest_concurrency: int = Field(default=2, ge=1, le=32)
    bulk_ingest_poll_seconds: float = Field(default=2.0, ge=0.5, le=60)

    # --- Auto-metadata & taxonomy tagging (FR-B7) — additive, default OFF (new-ingest-only) ---
    # When on, ingest derives filterable metadata (entities/year/doc_type/taxonomy category)
    # from the extracted text and stamps it on the document + its chunks (D8-filterable).
    auto_metadata_enabled: bool = False

    # --- Recency-aware retrieval (FR-D9) — per-query, default-off ---
    # Over-fetch multiplier so a time-window filter doesn't starve recall; half-life (days)
    # for the freshness-boost decay.
    recency_overfetch: int = Field(default=4, ge=1, le=10)
    # Answer from the document a question NAMES. Measured on a two-document corpus:
    # "summarise SOP Writing Guidance Pharma" retrieved 5 of 6 passages from the OTHER SOP
    # document (both are about SOPs, so on content alone they are near neighbours) and answered
    # confidently at faithfulness 1.0 — because that gate asks whether the answer is supported by
    # the passages, never whether they are the passages that were asked for.
    #
    # Matching is deliberately conservative and refuses ambiguity: "summarise the SOP" scopes to
    # nothing when two documents are both SOPs. See rag_core.docscope.
    doc_scope_enabled: bool = True
    recency_halflife_days: float = Field(default=180.0, gt=0.0, le=3650.0)

    # --- Suggested follow-up questions (FR-I8) — additive, default OFF ---
    # After a grounded answer, offer follow-ups generated ONLY from the retrieved context
    # (non-authoritative). "heuristic" is entity-anchored/CPU-free; "llm" uses the KB's model.
    followups_enabled: bool = False
    followups_backend: str = Field(default="heuristic", pattern="^(heuristic|llm)$")
    followups_count: int = Field(default=3, ge=1, le=5)
    # Generation budget for LLM follow-ups. Must be generous for REASONING models (Qwen-thinking/
    # QwQ/R1): they spend a large budget inside <think>…</think> first, so a small cap yields no
    # actual questions and silently falls back to the heuristic template. 1024 leaves room to think
    # AND emit the follow-ups.
    followups_max_tokens: int = Field(default=1024, ge=64)

    # --- Grounded query disambiguation (Layer-2 item 4) — additive, default OFF ---
    # When a query is AMBIGUOUS (matches several distinct KB topics shallowly, or would abstain
    # while the KB does hold candidate topics), offer 2-4 grounded "Did you mean …?" options built
    # ONLY from the retrieved spans (non-authoritative — never changes the answer/citations/status).
    # BIASED TOWARD UNDER-TRIGGERING: options are offered only when the retrieval forms
    # >= clarify_min_clusters distinct doc clusters AND the answer is not a confident single-topic
    # hit (max retrieval score < clarify_confidence_ceiling, or the answer abstained).
    clarify_enabled: bool = False
    clarify_backend: str = Field(default="heuristic", pattern="^(heuristic|llm)$")
    clarify_min_clusters: int = Field(default=2, ge=2)
    clarify_confidence_ceiling: float = Field(default=0.75, ge=0.0, le=1.0)
    clarify_count: int = Field(default=3, ge=1, le=5)

    # --- Continuous evaluation (FR-G9) — scheduled quality runs + drift alerting ---
    # Master switch for the in-process scheduler loop (default OFF; ON in dev compose).
    # A run scores a KB's golden set AND a sample of recent live QUERY traffic (re-run
    # through the real pipeline), records an EVAL_RUN audit event, and emits an EVAL_ALERT
    # when grounded-rate / avg-faithfulness fall below the per-KB floor. Alerting is
    # in-platform only (audit + logs + console) — no external egress by design.
    continuous_eval_enabled: bool = False
    continuous_eval_tick_minutes: int = Field(default=15, ge=1, le=1440)  # how often the loop checks for due KBs
    continuous_eval_default_interval_hours: int = Field(default=24, ge=1, le=720)
    continuous_eval_sample_rate: float = Field(default=0.2, ge=0.0, le=1.0)  # fraction of recent queries to re-score
    continuous_eval_sample_max: int = Field(default=10, ge=1, le=20)  # hard cap on sampled queries per run
    continuous_eval_lookback_events: int = Field(default=300, ge=10, le=5000)  # audit events scanned for sampling
    continuous_eval_min_grounded_rate: float = Field(default=0.6, ge=0.0, le=1.0)  # default breach floor
    continuous_eval_min_faithfulness: float = Field(default=0.5, ge=0.0, le=1.0)  # default breach floor

    # --- Scheduled connector sync (FR-A8) — delta re-sync of enrolled KBs on their interval ---
    # Rides the same in-process scheduler loop as continuous eval; content-hash delta means a
    # re-sync only re-indexes changed documents.
    sync_scheduler_enabled: bool = False

    # --- Continuous model-health poll — 24/7 self-check, replaces click-to-recheck ---
    # An in-process asyncio loop probes the PLATFORM model backends every health_poll_seconds and
    # publishes a snapshot (Redis when scaled, so all replicas show identical status; in-process
    # otherwise) that /system/model-health serves WITHOUT probing on the request path. Only the
    # Redis leader replica probes on a timer. A backend flips online->offline only after
    # health_poll_fail_threshold CONSECUTIVE failed probes (a single transient failure never flips
    # the dot). Default OFF (byte-identical to on-demand probing); enabled in deploy compose.
    health_poll_enabled: bool = False
    health_poll_seconds: int = Field(default=60, ge=5, le=3600)  # how often the leader probes
    health_poll_fail_threshold: int = Field(default=2, ge=1, le=10)  # consecutive fails before offline

    # --- contextual retrieval (FR-C2) — additive, new-ingest-only (ADR-0007) ---
    # When on, indexing prepends document context to each chunk BEFORE embedding/BM25
    # (original chunk still stored for citation). OFF by default; existing tenants
    # adopt it by re-ingesting.
    contextual_retrieval_enabled: bool = False
    contextual_retrieval_backend: str = "metadata"  # metadata (deterministic) | llm

    # --- adaptive query router (FR-D5) — additive, default OFF (ADR-0008) ---
    adaptive_router_enabled: bool = False
    router_backend: str = "heuristic"  # heuristic (deterministic) | llm

    # --- exact technical-token matching (Slice 2a) — additive, flag-gated, per-KB adaptive ---
    # The `standard` analyzer shreds chemical formulas (H₂SO₄), SMILES (CC(=O)Oc1ccccc1C(=O)O)
    # and identifiers (SOP-BULK-001) into fragments, so a query naming one under-retrieves the
    # chunk that contains it. When ON, BM25 also matches a whitespace-tokenised `text.exact`
    # sub-field (whole-token, case-preserving) with a boost. OFF ⇒ query body byte-identical.
    # Per-KB adaptive via the `exact_boost` threshold override (a pharma KB can weight exact
    # matches higher than an HR KB); a per-KB value falls back to this global boost when unset.
    exact_term_match_enabled: bool = False
    exact_term_boost: float = Field(default=2.0, ge=0)

    # --- RDKit chemical normalization (Slice 6, Phase C) — additive, flag-gated, chemistry-only ---
    # The Slice-2a exact path matches a SMILES verbatim, but two SMILES that are the SAME molecule
    # written differently (ethanol OCC vs CCO; aromatic benzene c1ccccc1 vs Kekulé C1=CC=CC=C1) do
    # NOT match. RDKit canonicalizes any valid SMILES to one canonical string + an InChIKey, so
    # equivalent molecules collapse to the same token. When ON (and RDKit is installed AND the KB's
    # domain profile is `chemistry` AND its per-KB `chem_normalization_enabled` gate is set), ingest
    # stores those canonical terms as ADDITIONAL searchable exact tokens (the displayed chunk text is
    # untouched) and the query is augmented with the same canonical terms so it matches them.
    # Master, default OFF, FAIL-SAFE: off, RDKit-absent, or a non-chemistry KB ⇒ no canonical tokens
    # written, no query rewrite ⇒ byte-identical to today. This CHANGES what is indexed, so like the
    # other index-shaping knobs it is a REINDEX knob — existing chemistry KBs adopt it by reindexing.
    chem_normalization_enabled: bool = False

    # --- glossary / synonym query expansion — additive, default OFF (byte-identical when off) ---
    # Bridges plain natural-language wording to the canonical SOP/GMP terminology BEFORE retrieval,
    # so a casual question ("who signs off when a test fails?") retrieves the right SOP chunk. When
    # ON, matched glossary triggers append their canonical terms to `query.text` — the field BOTH
    # the dense embedder and the cross-encoder reranker read (validated to flip a confident-wrong
    # retrieval to the correct document). Deterministic + idempotent + fail-safe: no match ⇒ query
    # unchanged, so OFF (or on with nothing matching) is byte-identical. `glossary_path` empty ⇒
    # the bundled default glossary; else a JSON overlay of the same shape merged over the default.
    glossary_expansion_enabled: bool = False
    glossary_path: str = ""
    # How a matched casual term is rewritten. `append` (default, safe) KEEPS the user's words and
    # adds the canonical terms after them — best recall, but a strong lexical lookalike can still
    # out-rank the right doc. `replace` SWAPS each matched casual trigger for its canonical terms —
    # disambiguates adversarial lookalikes (validated: flips the wrong top chunk to the right one),
    # at some recall risk when the casual word was itself the right signal. Trial `replace` on a
    # golden set before preferring it. Acronyms stay additive (bidirectional) in both modes.
    glossary_mode: str = Field(default="append", pattern="^(append|replace)$")

    # --- semantic intent routing — additive, default OFF (byte-identical when off) ---
    # When on, the query path classifies a question's semantic INTENT before retrieval:
    # "catalog" ("what documents/files/artifacts are in this KB?") answers from the KB
    # inventory instead of an out-of-scope abstain; "summarize" ("recap this conversation")
    # summarizes the thread on the conversation path; everything else is normal RAG. The
    # heuristic backend is deterministic keyword FALLBACK; "llm" is the real semantic path
    # (one tiny classification call to the KB's model, fail-safe to "rag" on any error).
    intent_routing_enabled: bool = False
    intent_backend: str = Field(default="heuristic", pattern="^(heuristic|llm)$")
    # doc_summary intent: summarize EACH document in the KB. Bounded because it is N generation
    # calls (one per doc) — summarize at most this many docs (noting "showing N of M" when more),
    # each grounded on this many representative chunks retrieved via a per-doc `doc_ids` scope.
    doc_summary_max_docs: int = Field(default=8, ge=1, le=50)
    doc_summary_chunks_per_doc: int = Field(default=6, ge=1, le=50)

    # --- agentic query planner — additive, MASTER, default OFF, fail-safe, sovereign ---
    # When on, an LLM reads the COMPLETE query + light KB context and returns a structured execution
    # plan (strategy/coverage/target/shape); the pipeline selects the answer mechanism FROM the plan
    # instead of from a single hard-coded intent label — genuine query understanding that generalises
    # to unseen phrasings and can elect to READ THE WHOLE DOCUMENT when comprehensive coverage is
    # wanted. Additive: OFF ⇒ the existing intent-routing block is untouched (byte-identical).
    # Fail-safe: the planner falls back to the deterministic heuristic plan on ANY error and a
    # planner failure NEVER 500s a query (worst case is today's behaviour). Sovereign: the plan call
    # runs on the KB's own gateway (the same sovereign model that answers), no foreign/managed calls.
    query_planner_enabled: bool = False
    # Whole-single-document map-reduce summary (the mechanism the planner's whole_document strategy
    # needs): MAP prompts the model over batches of THIS many chunks; REDUCE assembles the batch
    # extracts into one grounded, fully-cited answer over ALL of the document's chunks.
    whole_doc_summary_batch_chunks: int = Field(default=12, ge=1, le=100)
    # Prefer a SINGLE full-context read (the whole document in one generation call, like an uploaded
    # doc to a frontier chat model) when the document's chunks fit under this token budget (chars/4
    # heuristic) — no cross-batch synthesis loss. Only ABOVE this does the mechanism fall back to the
    # batched map-reduce. A headroom-safe fraction of a ~128k context window (room for prompt+output).
    whole_doc_full_read_token_budget: int = Field(default=96000, ge=1000, le=1000000)
    # The generation model's real context window (tokens). The EFFECTIVE full-read budget is
    # min(whole_doc_full_read_token_budget, llm_context_window - reserve_for_prompt_and_output), so a
    # smaller-window model never gets a single oversized prompt (which would error or silently
    # truncate) — the mechanism forces map-reduce instead. Set it to the served model's window.
    llm_context_window: int = Field(default=128000, ge=2000, le=2000000)
    # When a whole_document plan's target matches MULTIPLE visible documents (e.g. "Vendor
    # Agreement" → the 2023 AND 2024 files), summarise EACH matched doc and combine — up to this
    # many. Beyond the cap the pipeline does NOT guess or dump: it answers from normal RAG (passages)
    # and surfaces the matched titles as non-authoritative clarification options ("did you mean…?").
    whole_doc_multi_target_cap: int = Field(default=3, ge=1, le=20)
    # Safety cap on how many chunks the whole-doc summary reads; a document beyond this is summarised
    # up to the cap and its completeness payload is marked truncated=true (honest partial coverage).
    whole_doc_summary_max_chunks: int = Field(default=400, ge=1, le=5000)
    # Max concurrent MAP / coverage-refine batches on the whole-document read. The batches are
    # INDEPENDENT (each is one grounded gen.answer over its own chunk slice), so they run concurrently
    # under this bounded semaphore — cutting whole-doc latency from N sequential model calls to about
    # ceil(N/cap) waves. Kept small by default to stay gentle on the shared generation endpoint.
    whole_doc_map_concurrency: int = Field(default=4, ge=1, le=32)

    # --- query transformation (FR-D6) — additive, default OFF (ADR-0009) ---
    # Applies on the router's heavy path (or all queries if the router is off): transform
    # into sub-queries, retrieve each, merge/dedupe by chunk keeping best score.
    query_transform_enabled: bool = False
    # "auto" composes the deterministic decomposition with model reformulation: they fix
    # DIFFERENT failures (splitting a compound question vs rephrasing one that is worded
    # unlike the documents), and the previous one-of choice meant forfeiting whichever was
    # not selected. Measured: a scenario question scored 0.0084 and abstained while the
    # answering sentence sat in the corpus and scored 0.9886 when asked as a rule question.
    query_transform_backend: str = Field(default="auto", pattern="^(auto|decompose|hyde|rewrite)$")

    # --- guardrails (FR-G1 prompt-injection · FR-G3 toxicity) — default ON (ADR-0010) ---
    # Safety-first for a govt platform; heuristic backend is deterministic (won't break CI).
    # A block surfaces as ABSTAINED (blocked_*) + a distinct AuditAction.GUARDRAIL event.
    guardrails_enabled: bool = True
    guardrails_backend: str = "heuristic"  # heuristic (deterministic) | llm

    # --- MCP server (FR-H2) — the stdio server is bound to ONE tenant at launch ---
    mcp_tenant: str = ""  # tenant this MCP server instance acts for (isolation)
    mcp_token: str = ""  # optional dev JWT to validate against mcp_tenant
    mcp_plugins: str = ""  # comma-separated import paths of MCP tool-plugin modules (each exposes register(registry))

    # --- Quotas & rate limits (FR-H1) — noisy-neighbour isolation, operator-configurable ---
    # Rate limits are per ORG over a sliding 60s window; the document cap is per KB.
    # In-process limiter (fits the modular monolith); a shared Redis limiter is the
    # multi-replica follow-up. 0 disables an individual limit.
    quota_enabled: bool = True
    quota_queries_per_minute: int = Field(default=60, ge=0)
    quota_ingests_per_minute: int = Field(default=30, ge=0)
    quota_max_documents_per_kb: int = Field(default=1000, ge=0)

    # --- Scale / HA coordination (multi-replica hardening) ---
    # Empty RAG_REDIS_URL = single-process default (in-memory primitives, current
    # behaviour, zero deps). Set it (e.g. redis://redis:6379/0) to run N horizontally
    # scaled API replicas: the per-org rate limiter becomes a shared sliding window (one
    # busy tenant can't out-run the cap by spraying replicas), the semantic cache gains
    # cross-replica invalidation coherence, and the background scheduler runs on exactly
    # ONE replica via a Redis leader lock. Redis is a self-hosted internal store (like
    # Postgres/Qdrant), NOT an AI/cloud egress — so it is not routed through the
    # sovereignty guard. Pluggable seam: production plugs in Redis with no call-site change.
    redis_url: str = ""
    leader_lock_ttl_seconds: int = Field(default=90, ge=10)  # scheduler leadership lease

    # --- Semantic cache (FR-E5, P2) — additive, default OFF ---
    # Caches GROUNDED answers per KB by query-embedding similarity; a repeat question
    # returns instantly with the same cited answer. Invalidated on any ingest/delete/
    # re-index in that KB. In-process (monolith); Redis is the multi-replica follow-up.
    semantic_cache_enabled: bool = False
    semantic_cache_threshold: float = Field(default=0.97, ge=0.5, le=1.0)
    semantic_cache_max_entries: int = Field(default=256, ge=8)

    # --- Per-KB embedder switch (FR-E1 roles) ---
    # Attaching an embedding model to a KB with documents stages a PENDING switch; the
    # explicit apply endpoint recreates the vector collection at the new dimension and
    # re-embeds every document. ON by default because nothing changes without that
    # operator POST; the flag exists to disable the apply path wholesale.
    embedder_switch_enabled: bool = True

    # --- Stale-index detection + re-index (FR-C8) ---
    # A document is indexed by whatever pipeline was running when it was ingested. Change the
    # chunker, embedder, contextual retrieval or the readers and every existing document still
    # answers -- from chunks a different pipeline produced. The stamp (index_pipeline_version)
    # makes that visible; these settings decide what, if anything, acts on it automatically.
    #
    # Comma-separated kb_ids the automatic re-indexer keeps current. Empty (default) = OFF, and
    # staleness is reported but never acted on. Opt-in BY NAME rather than "every KB", because
    # this is a write over tenant data on a schedule: the operator chooses which corpora are
    # kept current and which are frozen.
    auto_reindex_kbs: str = ""
    auto_reindex_interval_minutes: int = Field(default=60, ge=1, le=10_080)
    # Documents re-indexed per pass. A re-index is one extraction + one embedding pass per
    # document against the SHARED model endpoints, so an unbounded corpus-wide rebuild is how a
    # re-index makes every live question time out.
    reindex_batch_limit: int = Field(default=50, ge=1, le=10_000)

    # --- Agentic retrieval loop (FR-F2, P2 — CRAG-lite) — additive, default OFF ---
    # When the first retrieval is too weak to answer, transform the query (decompose/
    # rewrite), retrieve again, and keep the better result — one bounded retry, audited.
    agentic_retry_enabled: bool = False

    # --- GraphRAG (FR-C6/D4, P2) — knowledge graph + local graph retrieval ---
    # Default OFF: enabling it changes what's indexed (entities/edges extracted per chunk).
    # Extraction backend "heuristic" is deterministic/CPU-free; "llm" uses the KB's model.
    # At query time, entities in the question link to their graph neighbours and those
    # chunks are merged into hybrid retrieval — lifting multi-hop recall.
    graphrag_enabled: bool = False
    # heuristic (English capitals) | llm (model emits entity JSON) | translate (translate a
    # non-English chunk to English, extract there where NER is reliable, map entities back —
    # the robust path for Indic text, since direct Hindi NER/LLM extraction is unreliable).
    graph_extract_backend: str = Field(default="heuristic", pattern="^(heuristic|llm|translate)$")
    # Token budget for the LLM graph extractor. High by default so a REASONING model (which
    # spends the first few hundred tokens thinking) still has room to emit the entity JSON —
    # 512 starved it and produced an empty/sparse graph, especially on non-Latin scripts.
    graph_extract_max_tokens: int = Field(default=3000, ge=256)
    # bounded fan-out for per-chunk graph extraction + per-community summaries; caps concurrent
    # model/Sarvam calls during ingest so we don't serialize (slow) or flood (rate-limit) them.
    graph_index_concurrency: int = Field(default=8, ge=1, le=32)
    # Token budget for the translate-then-extract translation calls (chunk→English, entities→back).
    graph_translate_max_tokens: int = Field(default=2000, ge=256)
    # Which translator backs translate-then-extract (and cross-lingual helpers). "llm" reuses the
    # KB chat model — but a REASONING chat model (Sarvam-105b) is unreliable at translation: it
    # echoes the source back or returns content=null, so the graph gets no entities. "sarvam" uses
    # Sarvam's DEDICATED translation endpoint (Sarvam-Translate / Mayura — a purpose-built,
    # deterministic Indic↔English model). Hosted at api.sarvam.ai today; the open GPL-3.0 weights
    # (sarvamai/sarvam-translate) are self-hostable on GPU for a fully sovereign deployment.
    translate_backend: str = Field(default="llm", pattern="^(llm|sarvam)$")
    # Hosted Sarvam Translate API. Key/host default to the LLM's (same api.sarvam.ai subscription),
    # so setting translate_backend=sarvam is enough. Input is chunked to translate_max_chars.
    # SOVEREIGN OPTION: point translate_sarvam_url at a self-hosted sarvam-translate (open GPL-3.0
    # weights, sarvamai/sarvam-translate, served on your own GPU) to avoid the hosted endpoint.
    translate_sarvam_url: str = "https://api.sarvam.ai"
    translate_sarvam_model: str = "sarvam-translate:v1"
    translate_sarvam_api_key: str = ""  # falls back to llm_api_key when empty
    translate_sarvam_api_key_header: str = "api-subscription-key"
    translate_max_chars: int = Field(default=1800, ge=200, le=2000)  # per-request cap (API limit ~2000)
    graph_retrieval_hops: int = Field(default=1, ge=1, le=3)
    # How many graph-linked chunks are FETCHED for scoring, before the reranker picks the best
    # `graph_retrieval_max_chunks` of them. The graph walk has no relevance signal and returns
    # its ids lexicographically, so truncating before scoring picks an arbitrary sample; this
    # is the widening that lets the cross-encoder actually select. It is the dominant cost of
    # a graph-augmented query (~105 ms/candidate on a CPU MiniLM-L-6, linear) — lower it if
    # graph queries are too slow, raise it with a GPU reranker.
    graph_retrieval_candidates: int = Field(default=30, ge=1, le=200)
    graph_retrieval_max_chunks: int = Field(default=10, ge=1, le=50)
    # Slice 2 — communities + global/thematic search (GraphRAG map-reduce).
    # Communities are detected on demand (POST .../graph/communities), never per-ingest.
    # "heuristic" labels a theme by its entities (CPU-free); "llm" writes prose via the KB model.
    # A broad/aggregate question ("main themes?") then answers from the most relevant
    # communities' chunks (KB-wide breadth) instead of a narrow top-k; citations stay real.
    graph_summary_backend: str = Field(default="heuristic", pattern="^(heuristic|llm)$")
    graph_global_search: bool = True
    graph_community_min_size: int = Field(default=2, ge=1, le=20)
    graph_global_communities: int = Field(default=4, ge=1, le=20)
    graph_global_chunks_per_community: int = Field(default=2, ge=1, le=10)

    # --- Domain-adaptive profile (Slice 6) — detect a KB's domain, SUGGEST its knobs ---
    # Master, default OFF. When on, the profile-detection endpoints run and the domain-relation
    # (sub-slice B) extractor may fire at ingest; when off, everything is byte-identical to today
    # (no detector, no domain-relation extractor, no new tenant_config keys written).
    # Detect-and-SUGGEST only: detection never moves a knob — an operator confirms via /profile/apply.
    domain_profile_enabled: bool = False
    # "heuristic" classifies the corpus with the deterministic reused signals (chemistry = notation
    # density, legal/financial/code = cue tables). "llm" refines that verdict via the KB's own model
    # with the heuristic verdict as prior + fallback (mirrors graph_extract_backend).
    domain_profile_backend: str = Field(default="heuristic", pattern="^(heuristic|llm)$")
    # Backend for the typed domain-relation extractor (sub-slice B): "heuristic" verb-pattern
    # (default, CI-safe) | "llm" constrained to the profile's relation vocabulary (heuristic
    # fallback on bad JSON). Only fires when the master flag is on AND the KB's profile declares a
    # relation vocabulary AND the per-KB `domain_relations_enabled` gate is set.
    domain_relation_backend: str = Field(default="heuristic", pattern="^(heuristic|llm)$")
    domain_relation_max_tokens: int = Field(default=512, ge=256)

    # --- RAG v2 ingest (upgrade plan "Hard constraints") — master flag, default OFF -------------
    # Gates the v2 ingest path: capture + persistence of the extraction engine's layout overlay
    # (Phase 2, its FIRST reader — `svc_control_plane.pipeline._persist_layout_overlay`), then
    # section-tree parsing, the document profile card, and the writes that populate the v2 catalog
    # tables (`sections`/`blocks`/`facets`/`doc_relations`). OFF ⇒ the ingest path is
    # byte-identical to today: no layout object is written and `documents.minio_layout_key` stays
    # NULL. The per-tenant DDL that creates those tables is deliberately UNCONDITIONAL (they are
    # empty and additive; gating the DDL on an env var would
    # make a tenant's schema shape depend on when it was last touched — see the reasoning in
    # `rag_clients.postgres_catalog._v2_ddl`).
    # `RAG_V2_QUERY` and `RAG_V2_VERIFY` from the plan are NOT here yet: `RAG_V2_QUERY` needs a
    # tri-state (off/shadow/on) that the global BOOLEAN-only platform flag store cannot hold
    # (discovery §8.6), so it is a later phase's decision rather than a boolean added early.
    v2_ingest: bool = False
    # Phase 5 target chunk size for the layout-aware chunker, in CHARACTERS. The plan asks for
    # 300-500 TOKENS, configurable per doc_type; 1600 characters is ~310 tokens of English prose
    # and ~370 of serialized table rows at this corpus's measured chars/token ratios, so it sits
    # inside that band for the scripts the band was written for. The mapping and its imprecision
    # (a dense script or a run of ToC dot leaders will exceed 500 tokens) are set out at
    # `rag_core.layout_chunking._DEFAULT_LAYOUT_MAX_CHARS`, which holds the same default so
    # rag-core stays usable on its own. Read only on the layout-aware path. This is the
    # deployment-wide value; `layout_chunk_max_chars_by_doc_type` below overrides it per document
    # class now that the Phase-3 profile card supplies a `doc_type` to key on.
    layout_chunk_max_chars: int = Field(default=1600, ge=_LAYOUT_CHUNK_MIN_CHARS, le=_LAYOUT_CHUNK_MAX_CHARS)
    # Above this page count no layout overlay is captured at all and the document chunks from the
    # flat text stream. A GUARD against a pathological PDF, not a size policy: the layout capture
    # is a second pdfplumber pass per page plus one JSON object holding every line's bbox and font
    # metrics, both linear in pages with no bound. Measured at 29 blocks/page on a tender-shaped
    # fixture: +0-20% extraction time and 6.7 KiB/page of overlay, so 2000 pages is ~13 MiB and
    # ~58k blocks serialized, stored, re-fetched and re-parsed inside the API event loop. The
    # default sits far above any real tender ON PURPOSE — a 2000-page corrigendum is the document
    # this feature exists for, and silently excluding it would be the wrong failure.
    layout_capture_max_pages: int = Field(default=3000, ge=1)
    # Hard ceiling on the serialized overlay object. The page cap bounds the normal case; this
    # catches the pathological page (thousands of tiny text runs) that the page count cannot see.
    # Over it, the object is NOT written and the reason is logged — see
    # `svc_control_plane.pipeline._persist_layout_overlay`.
    layout_overlay_max_bytes: int = Field(default=32 * 1024 * 1024, ge=64 * 1024)
    # --- Phase 3: the document profile card ----------------------------------------------------
    # One model call per document on the v2 ingest path, prompted by
    # `rag_core.prompts/doc_profile.md`. Everything here is read only when `v2_ingest` is
    # on; with it off no call is made and no card is stored.
    #
    # How much of the document's OPENING the prompt is shown. The plan says "first ~2 pages of
    # text", which this takes literally via `page_map` (per-page CHARACTER counts) and then bounds:
    # a page is not a fixed number of characters, and a dense two-page annexure can be far more
    # text than a deployment's context window has room for beside the breadcrumb list. 12000
    # characters is ~2400 tokens of English prose, which leaves a 128k-window model untroubled and
    # still fits the 8k-window deployments in `infra/helm`.
    #
    # IT IS ALSO THE EVIDENCE WINDOW. A declared relation's quote is validated against exactly the
    # text the prompt was shown (`doc_profile.parse_doc_profile`), so raising this widens what the
    # model may legitimately cite and lowering it narrows it.
    doc_profile_head_chars: int = Field(default=12_000, ge=500)
    # How many section breadcrumbs the prompt lists. The plan asks for the FULL list; a 200-page
    # tender has thousands, which would crowd out the document text above. Truncated from the end
    # with a count of what was dropped, so the model is never silently shown a partial outline.
    doc_profile_max_breadcrumbs: int = Field(default=400, ge=10)
    # Completion budget for the profile call. A 15-facet taxonomy with descriptions plus a handful
    # of quoted relations runs ~900 tokens; the headroom is for the quotes, which are verbatim and
    # can be long.
    doc_profile_max_tokens: int = Field(default=1400, ge=256)
    # Plan §3's "config map: doc_type -> chunk profile", as `doc_type=chars` pairs:
    # `"corrigendum=1200,contract=2000"`. A dict-typed setting would be the first one in this file
    # and brings pydantic-settings' complex-type env parsing with it; a comma list matches the
    # style of every other multi-valued setting here and is parsed by
    # `rag_common.config.parse_doc_type_chunk_chars`.
    #
    # EMPTY BY DEFAULT, AND SHIPPED EMPTY ON PURPOSE. The plan says routing "may" use `doc_type` to
    # pick chunking parameters, and nothing in this repository has measured what a corrigendum or a
    # contract should be chunked at — `layout_chunk_max_chars`'s own 1600 was calibrated against a
    # token budget, not a document class. So the mechanism is here and carries no invented numbers:
    # with it empty, every document chunks at `layout_chunk_max_chars` exactly as it did before.
    layout_chunk_max_chars_by_doc_type: str = ""
    # --- Phase 4: section enrichment -----------------------------------------------------------
    # ONE MODEL CALL PER SECTION, which is the whole reason this has its own switch rather than
    # riding on `v2_ingest` alone. Phase 3 costs a document one call; this costs it one per
    # enrichable section, so a 200-section tender is a different order of spend and latency and an
    # operator must be able to decline it without giving up v2 chunking, the profile card and the
    # structure tree. Default ON because with `v2_ingest` off — which is every deployment —
    # nothing here is read at all, and a deployment that turns v2 ingest on should get the phase
    # the plan describes rather than a silently incomplete one.
    section_enrich_enabled: bool = True
    # How many of a document's sections may be enriched. A HARD CEILING against a pathological
    # outline, not a quality knob: the enrichable set is already filtered by
    # `section_enrich_min_chars` below, and 400 sections of real prose is a very large tender.
    # Sections past the cap are skipped in document order and the count is logged, so a truncated
    # document says so rather than looking merely unenriched.
    section_enrich_max_sections: int = Field(default=400, ge=1)
    # Below this many characters of its OWN text a section is not enriched. A container heading
    # with no body, a one-line cross-reference and a page title carry nothing for a summary to
    # summarise or a proposition to come from, and they are the majority of nodes in a deep
    # outline — enriching them would spend most of the document's calls on the sections with the
    # least in them. 200 characters is the same floor `_LAYOUT_CHUNK_MIN_CHARS` sets for a chunk.
    section_enrich_min_chars: int = Field(default=200, ge=1)
    # TOTAL calls per window, counting the first — NOT a retry count, so 1 means one call and no
    # retry. Default 2 is exactly the behaviour this setting replaced, so it changes nothing until
    # an operator raises it.
    #
    # RAISING THIS DOES NOT FIX "no JSON object in the reply", which is measured rather than
    # assumed. On `corrigendum-5.pdf` (2026-11-02), 2 attempts lost three substantive sections and
    # 5 attempts lost two — but not the same two: Enterprise Cabinet/Rack and Firewalls recovered
    # while `21. Switches`, which had succeeded at 2 attempts, failed all five. The membership
    # churned; the reliability did not improve.
    #
    # That failure is OUTPUT TRUNCATION at `section_enrich_max_tokens`, not instruction-following.
    # Every failing reply is a valid JSON object cut off mid-structure, at 965-1,227 tokens against
    # a 1,200 cap. Unlike `architecture_extract_retries` (5) and `doc_profile_retries` (4), whose
    # failure really is an empty or non-JSON reply, more attempts here only re-roll reply length.
    #
    # Kept because it is the right shape for a genuinely flaky reply and costs nothing at 2.
    section_enrich_max_attempts: int = Field(default=2, ge=1, le=10)
    # A section longer than this is split into windows, each enriched separately and the results
    # merged (plan §4, "very long sections: split into windows and merge"). The same 12000 as
    # `doc_profile_head_chars` and for the same reason: it is ~2400 tokens of English prose, which
    # leaves room for the profile JSON and the schema above it inside an 8k-window deployment.
    #
    # IT IS ALSO THE EVIDENCE WINDOW. Every quote is validated against the window the model was
    # shown, never the whole section (`section_enrich.parse_section_enrichment`).
    section_enrich_window_chars: int = Field(default=12_000, ge=500)
    # How many windows one section may be split into. Past it the rest of the section is NOT read,
    # and the drop is logged. An appendix that is one 400-page "section" is a real extraction
    # outcome, and reading it in forty calls would spend a document's whole budget on one node.
    section_enrich_max_windows: int = Field(default=4, ge=1)
    # Concurrent enrichment calls per document. The plan says "parallel, batched"; unbounded
    # parallelism against a tenant's own endpoint is a self-inflicted denial of service, and on
    # the CPU llama.cpp tier it is a queue with extra steps.
    section_enrich_concurrency: int = Field(default=4, ge=1, le=32)
    # Completion budget for one enrichment call. A 4-sentence summary, a handful of key-values and
    # a half-dozen quoted propositions run ~700 tokens; the headroom is for the quotes, which are
    # verbatim and can be long.
    #
    # 3000 IS A MEASURED OPERATING POINT, not headroom. It was 1200, and 1200 is what lost three
    # substantive sections of `corrigendum-5.pdf`. On the largest of them (`20. Video Wall`, 6,720
    # characters, one window) a paired experiment on BYTE-IDENTICAL input (2026-11-02) moved only
    # this number:
    #
    #     1200   parse 0/3   -- valid JSON cut off mid-structure every run
    #     3000   parse 3/3   -- 5-6 facets, 10-11 key values, 14-17 propositions,
    #                           quote validation 27/27, 24/24, 25/25 (1.000 every run)
    #     4000   parse 3/3   -- but quote validation 24/24, 20/21, 19/22 (0.864 on the last),
    #                           BELOW the QUOTE_RATE_FLOOR of 0.95
    #
    # So MORE IS WORSE above this: the extra room buys claims the document does not support. The
    # rejected alternative was a smaller `section_enrich_window_chars` (4000), which truncated the
    # final window in all three runs and merged the surviving windows into a result that LOOKED
    # enriched -- see `merge_windows` for the invariant that now refuses that.
    #
    # THIS IS A CEILING, NOT A SPEND. A section that finishes in 800 tokens still costs 800; the
    # enricher records what each section actually consumed (`EnrichReport.completion_tokens`,
    # `calls_at_cap`) so this number is monitored against real consumption rather than assumed.
    # Rollback is this setting at 1200.
    section_enrich_max_tokens: int = Field(default=3000, ge=256)
    # --- Phase 12: the query log ---------------------------------------------------------------
    # The plan's `query_log` table (§1.1): one row per answered question, carrying the query, the
    # answer, the documents it cited and the PER-STAGE latency. Phase 1 created the table and
    # Phase 10's erasure scrub already reads it; this is the switch that starts writing it.
    #
    # DEFAULT OFF, AND NOT BECAUSE IT IS UNFINISHED. The row stores the user's question and the
    # answer text, which quotes the corpus — the same class of derived record as the chat history
    # (`ConversationStore`), which is why `delete_query_log_citing` and the DPDP erasure path were
    # built for it before anything wrote it. Starting to store that in every deployment on an
    # upgrade is not a decision this file should make silently. Turn it on where the latency
    # numbers are wanted; erasure, document delete and retention all already reach the rows.
    #
    # NOT gated on `v2_ingest`: it measures the query path, which v2 ingest does not change,
    # and §8.2's whole argument is that the baseline must be measurable BEFORE the phases it is
    # meant to measure land.
    query_log_enabled: bool = False
    # Bounds on the two free-text columns. A row is an operations record, not a transcript: an
    # answer that ran to 40 KB would be stored in full on every query, and the part that matters
    # for latency work is that the row exists and says what was asked. Truncation is marked in
    # the stored text, never silent.
    query_log_max_query_chars: int = Field(default=2_000, ge=100)
    query_log_max_answer_chars: int = Field(default=8_000, ge=100)
    # --- Phase 10: answer verification ---------------------------------------------------------
    # The plan's own flag name (`RAG_V2_VERIFY`). When on, every claim in a grounded answer is
    # checked against THE SPAN IT CITES — not against the union of everything retrieved, which is
    # what the pre-Phase-10 numeric guard does and why it cannot see a mis-attribution. The checks
    # are deterministic and pure (`rag_core.verification`); the NLI half of the plan's Phase
    # 10 already exists and is unchanged (discovery §8.21).
    #
    # DEFAULT OFF because it EDITS THE ANSWER: a failing claim is footnoted or dropped, which is a
    # user-visible consequence and the plan's own remedy. Flag-off is byte-identical.
    v2_verify: bool = False
    # What to do with a claim that fails. The plan: "regenerate the failing bullet once with the
    # quote shown; if still failing, drop it and add it to a 'could not be verified' footnote
    # (configurable: drop vs footnote)".
    #
    #   report   — record the verdict and change nothing. For measuring the check against a live
    #              corpus before letting it touch an answer, which is how a guard like this should
    #              be adopted.
    #   footnote — append "could not be verified" naming the failing claims (DEFAULT). The
    #              conservative action: the reader keeps what the model said AND learns which part
    #              does not check out.
    #   drop     — remove the failing claims and footnote them. Stronger, and riskier: a verifier
    #              false positive deletes real content.
    #
    # REGENERATE IS NOT IMPLEMENTED and is not a silent omission — see discovery §8.22.
    v2_verify_action: str = Field(default="footnote", pattern="^(report|footnote|drop)$")
    # --- Phase 7: the query router -------------------------------------------------------------
    # One decision per question, read off the PROFILE CARDS of the in-scope documents: which of
    # the plan's six intents this is, which documents and which of their own facets it touches,
    # the facet-specific sub-queries an aggregate needs, and whether the amendment view applies.
    # Mapped onto the QueryPlan the pipeline already routes on (discovery §8.4), so it selects an
    # EXISTING answer mechanism rather than introducing a second routing vocabulary.
    #
    # DEFAULT OFF because it changes what retrieval does: a decision's `doc_ids` scopes retrieval
    # and its `sub_queries` drive decomposition. Flag-off is byte-identical.
    #
    # NOT the plan's `RAG_V2_QUERY`, which is a TRI-STATE (off/shadow/on) governing the whole v2
    # query path and which the boolean-only platform flag store cannot hold (§8.6). This is the
    # router alone, which is what exists.
    v2_router: bool = False
    # heuristic (default) = deterministic cue tables, no model call, and the only path whose intent
    # accuracy can be asserted against a labelled gold set — which matters because the plan's
    # acceptance IS an accuracy figure. llm = one short gateway call over the cards, falling back
    # to the heuristic decision on any trouble. Same shape as every other backend seam here.
    v2_router_backend: str = Field(default="heuristic", pattern="^(heuristic|llm)$")
    # How many documents' cards the router is shown when the caller named none. A SAMPLE, not a
    # ranking: the plan says "top docs by doc-level retrieval" and there is no doc-level retrieval
    # to ask, because the `doc` chunk level has no producer (§8.17). Bounded because the cards go
    # into a prompt on the latency path.
    v2_router_max_cards: int = Field(default=20, ge=1, le=200)
    # Completion budget for the router call. A decision is a small object; the budget is small
    # because this runs before EVERY answer and a slow router is a worse router than a less
    # accurate one.
    v2_router_max_tokens: int = Field(default=400, ge=128)

    # --- RAG v2 Phase 8: retrieval strategies ---
    # Each router intent gets the plan's §8 recipe: which retrieval GRAIN the hybrid primitive is
    # asked for (`RetrievalQuery.levels`), whether hits expand to their parent section, and —
    # for the aggregate class — a SECTION PASS that reads every section of the in-scope documents
    # out of the catalog and reranks them instead of searching for passages.
    #
    # DEFAULT OFF because it changes what retrieval returns. Flag-off is byte-identical, and this
    # flag is ALSO gated on the Phase-7 router: a strategy is chosen by an intent, and with no
    # decision there is no intent to choose by.
    v2_retrieval: bool = False
    # The plan's `section_threshold`, on the cross-encoder's absolute [0,1] scale. Deliberately
    # LOW — "recall over precision" in the plan's own words, and measured on this deployment's
    # BGE reranker: a facet sub-query scores its own section ~0.65 and an unrelated one ~1e-5, so
    # anything in that gap separates them. An over-selected section costs one map call in Phase 9
    # (which returns [] for a section with nothing relevant); an under-selected one costs a
    # missing requirement, which is the failure this whole phase exists to fix.
    v2_section_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    # The SCALE-FREE companion to the absolute threshold above, applied by OR: a section is also
    # selected when it scores within this fraction of the best-scoring section. It exists because
    # the absolute threshold, measured against the real AEGCL corrigendum, selected 0 of 78
    # sections at 0.15 AND at every value down to 0.001 — a cross-encoder score is absolute for a
    # fixed CANDIDATE SHAPE, and 0.15 was calibrated against a retrieved PASSAGE, not against a
    # breadcrumb plus a one-line summary. The ratios still separated cleanly (1.0 / 0.657 / 0.24
    # for the relevant sections, 0.000 for the rest), so the ratio is what the floor reads.
    # 0.0 turns it off and leaves the plan's absolute threshold alone.
    v2_section_relative: float = Field(default=0.1, ge=0.0, le=1.0)
    # The plan's `max_sections`. A cap on the Phase-9 map fan-out, not a quality knob: every
    # section past it is recorded and stated in the coverage line rather than silently dropped.
    v2_max_sections: int = Field(default=40, ge=1, le=500)
    # How many section cards may reach the CROSS-ENCODER, which is a different bound from
    # `v2_max_sections` and guards a different cost: that one caps the Phase-9 map fan-out,
    # this one caps the latency of the selection itself. Every probe scores every candidate, so
    # a 500-section tender with 7 probes is 3,500 cross-encoder scorings before a word is
    # generated. The cards past the cap are scored by nothing and so can only be selected by the
    # facet or amendment unions; the shortfall is logged, never silent. 400 matches
    # `section_enrich_max_sections`, so a document whose sections were all enrichable is all
    # scoreable.
    v2_max_section_candidates: int = Field(default=400, ge=1, le=2000)

    # --- RAG v2 Phase 9: generation by map then reduce ---
    # One extraction call per SELECTED section (the map), then one call that writes the answer
    # from the extracted items and nothing else (the reduce). It exists because 38 selected
    # sections of an 80-page tender are far past any context window worth trusting, and a model
    # given all of them at once does not fail loudly — it answers from the first few and the
    # last few, and the requirement in the middle is simply absent from an answer that reads
    # complete. Map-reduce turns "the model did not read it" into a COUNT, which is what the
    # coverage statement reports.
    #
    # DEFAULT OFF because it REPLACES the answer for an aggregate-class question. It is also
    # gated on `v2_retrieval`: without a section selection there is nothing to map over.
    v2_generate: bool = False
    # Concurrency for the map fan-out, against the KB's OWN endpoint. Unbounded parallelism
    # against a tenant's endpoint is a self-inflicted denial of service, and on the CPU tier it is
    # a queue with extra steps — so this is a cap, not a target. But 6 was chosen for that reason
    # ALONE, before anything had been timed, and it is the single largest contributor to the one
    # §0 criterion that misses by an order of magnitude.
    #
    # MEASURED end to end on this deployment (gemma4-31b, the real AEGCL corrigendum, 39 sections
    # selected, 2026-10-23), with ZERO failures at every level tried:
    #
    #     conc  6   total 666 s                                   <- the old default
    #     conc 12   total 485 s   = map 302 s + reduce 99 s + ~83 s selection   <- this default
    #
    # COUNT THE CALLS, NOT THE SECTIONS. A section longer than `v2_map_section_chars` is
    # split into up to `v2_map_max_windows` windows and each window is its own call, so 39
    # selected sections are about 60 map calls. That is why the map stage is 302 s and not the
    # ~196 s a per-section reading of the throughput below would predict.
    #
    # Per-call throughput, measured separately against this endpoint on real map prompts:
    #
    #     conc  4    51 s/call    0.08 calls/s
    #     conc 12    60 s/call    0.20 calls/s
    #     conc 20    74 s/call    0.26 calls/s
    #     conc 39   101 s/call    0.39 calls/s
    #
    # The endpoint batches well: tripling from 4 to 12 costs 18% on each individual call and
    # returns 2.5x the throughput. Past 12 it keeps paying but less cleanly — per-call latency
    # rises roughly linearly while throughput flattens — and a tenant endpoint smaller than this
    # one feels it first, which is what this cap exists to prevent. 12 is where the curve is
    # still nearly free. An operator who knows their endpoint can read off what 20 or 39 buys:
    # at ~60 calls, conc 20 is a ~230 s map stage and conc 39 about 155 s.
    #
    # WHAT NO SETTING HERE CAN DO is reach the plan's 60 s aggregate target. The cost is DECODE,
    # not prefill — a 5,346-character prompt and a 15,815-character one both take 48 s because
    # both emit ~2,000 tokens. The map stage alone emits ~115,000 tokens, so 60 s would need
    # ~1,900 tok/s sustained and this endpoint peaks at ~771. That gap is serving capacity, and
    # closing it is a hardware or a scope decision, not a tuning one.
    # How many EXTRA doc-profile attempts on an unusable reply (early-exit on the first usable
    # card; each is one more call). The profiler allowed exactly one retry, on the stated grounds
    # that "retrying the identical prompt would be pointless at temperature=0.0: a greedy decode
    # re-reads the same prompt and writes the same bytes". That premise does not hold in any
    # deployment that sets `RAG_LLM_REASONING_EFFORT=off`, because
    # `OpenAICompatGateway._eff_temp` then raises 0.0 to 0.6 on purpose, to stop a
    # no-reasoning model looping. So the attempts are independent samples and more of them help.
    #
    # It matters because the failure is SILENT: `profile` is fail-safe and returns None, the
    # document ingests with no Phase-3 card, and Phase 4's prompt, Phase 7's routing and Phase 8's
    # facets all quietly get less, with nothing reporting a fault.
    #
    # MEASURED on the real AEGCL corrigendum against gemma4-31b, and the shape matters more than
    # the rate: eight consecutive calls on the identical prompt all returned a usable card, so the
    # steady-state rate is high. But failures arrive in CLUSTERS, and a cluster is what defeats a
    # one-retry budget — a dump run the same afternoon hit two consecutive `no JSON object in the
    # reply` replies and gave up, losing the card for that document entirely. Independent samples
    # during a bad patch are correlated, so the budget has to be deep enough to ride one out
    # rather than sized from the average.
    #
    # It is nearly free: the loop early-exits on the first usable card, so in the ordinary case
    # this costs exactly one call, same as before. 4 is sized to survive the longest cluster seen
    # (3) with headroom. Same shape and the same reason as `architecture_extract_retries`.
    doc_profile_retries: int = Field(default=4, ge=0, le=10)
    v2_map_concurrency: int = Field(default=12, ge=1, le=32)
    # Completion budget for one section's extraction. MEASURED, not guessed: at 1200 — the value
    # Phase 4's enricher uses — 18 of 38 map calls over the real AEGCL corrigendum were cut off
    # mid-object, because a dense DRS section yields a dozen items and every one carries a
    # verbatim quote. A truncated reply is salvaged to its last complete item
    # (`json_reply.extract_json_array`), so the budget is not a cliff.
    #
    # RAISED 2400 -> 4000 ON THE MEASUREMENT. 2400 was set where a TYPICAL section finishes, and
    # a tender's dense sections are not typical. The same 12,000-character window of the real
    # corrigendum, same prompt, three budgets:
    #
    #     max_tokens 2400    47.6 s    21 items
    #     max_tokens 4000    79.2 s    35 items
    #     max_tokens 6000   120.0 s    40 items
    #
    # So at 2400 that window gave up barely half of what it holds. 4000 is the knee: it recovers
    # two thirds of the gap, where 6000 buys five more items for another 50% of the clock.
    #
    # THE REASON THIS WENT UNNOTICED FOR SO LONG IS THAT IT LEAVES NO TRACE. A budget-limited map
    # reply is not truncated JSON — the model closes the array cleanly and returns a complete,
    # well-formed, SHORT list. `dropped_unquoted` is 0 and `sections_truncated` says nothing,
    # because the section WAS fully shown; the model simply stopped enumerating it. Raising this
    # costs nothing on a sparse section, which still stops at EOS well inside the budget.
    v2_map_max_tokens: int = Field(default=4000, ge=256)
    # How much of a section's text ONE map call is shown. A longer section is split into
    # windows, up to `v2_map_max_windows`.
    v2_map_section_chars: int = Field(default=12_000, ge=1000)
    # How many windows one section may cost. MEASURED: truncating a long section instead of
    # windowing it was the single largest cause of a missed requirement on the real AEGCL
    # corrigendum — twelve of thirty-eight selected sections overran 12,000 characters, and a
    # tender section puts its specification table first and its certification and
    # bill-of-material clauses LAST, so every "UL 62368", "HBOM" and "SBOM" in the document fell
    # off the end of a truncated prompt and appeared nowhere in the answer.
    #
    # BOUNDED ANYWAY, because this is the latency path. A section past the bound is still
    # truncated and `MapReport.sections_truncated` counts it — that is the number to watch.
    #
    # WIDENED 3 -> 6 ON THE MEASUREMENT, which is what the line above asked anyone to do. At 3
    # the cap is 36,000 characters, and on the real AEGCL corrigendum two gold requirements sat
    # just past it: the PeopleCert/ITIL clause at offset 36,178 — 178 characters outside — and
    # the ISO 27001 clause at 65,487. Both were simply never shown to the model.
    #
    # The cost is small because long sections are rare even in a tender. Measured on that
    # document, total map calls by bound:
    #
    #     windows 3   cap 36,000   3 sections truncated   74 calls
    #     windows 4   cap 48,000   2 sections truncated   77 calls   (+4%)
    #     windows 6   cap 72,000   1 section  truncated   81 calls   (+9.5%)
    #
    # CORPUS-DEPENDENT, and that is the honest caveat: +9.5% holds where a handful of sections
    # are long. A corpus of uniformly huge sections would pay far more, which is why this stays
    # a bound and did not become unlimited.
    #
    # THE VALIDATION CEILING IS 32; THE DEFAULT IS STILL 6. Raising the ceiling changes nothing
    # about what ships — it changes what can be MEASURED, and the previous ceiling of 10 made
    # the one experiment this bound most needs impossible to run.
    #
    # WHAT THE CEILING IS DERIVED FROM. The quantity that actually matters to an answer is how
    # much of a section the map stage can read: `v2_map_section_chars * this`. At the
    # shipped 12,000 that is 72,000 characters, which covers the longest section in the AEGCL
    # corrigendum (58,484) with room to spare. A SMALLER window at the same coverage needs
    # proportionally more of them: MEASURED on that corpus, the longest section took 5 windows
    # at 12,000 characters, 11 at 6,000 and 22 at 3,000. A ceiling of 10 could not express
    # either of the smaller sizes, so neither was a configurable option however well it
    # measured. 32 clears the measured 22 with headroom and is the same ceiling
    # `v2_map_concurrency` carries, so one section can never ask for more simultaneous calls
    # than the stage is ever allowed to have in flight.
    #
    # It did NOT block the window-size SWEEP, which constructed `LLMSectionMapper` directly and
    # never went through this field. The ceiling is a shipping constraint, not a measurement
    # one, and it is being raised so a measured result can become a deployable setting.
    #
    # THE RESOURCE LIMIT IT RESPECTS. Since the scheduler change, windows are bounded as CALLS
    # by `v2_map_concurrency`, not as sections — so a high window count costs tokens and
    # money, not connections or memory. The worst case is `v2_max_sections * this` calls of
    # `v2_map_max_tokens` each; at this ceiling and that default of 40 sections, 40 x 32 x
    # 4,000 = 5.1M output tokens for ONE question, which is why 32 is a ceiling to experiment
    # under and 6 is what ships. (The real cost is far lower — only sections past
    # `v2_map_section_chars` window at all, and 46 windows covered 35 sections on the
    # corrigendum — but a bound is sized from its worst case, not its typical one.) Watch
    # `MapReport.windows` against `sections` before raising the default, and `sections_truncated`
    # before deciding it is too low.
    v2_map_max_windows: int = Field(default=6, ge=1, le=32)
    # Completion budget for the reduce call. Generous: this is the whole answer, grouped by
    # facet, over as many as `v2_max_sections` sections' worth of items.
    v2_reduce_max_tokens: int = Field(default=4000, ge=512)
    # Read and write `map_cache`. The plan's §9.1 cache, keyed by (section, query signature,
    # model). Off makes every question re-extract, which is correct but costs one model call per
    # selected section every time.
    v2_map_cache_enabled: bool = True

    # --- RAG v2 Phase 12: the two upstream caches ---
    # NEITHER IS AN ANSWER CACHE. `semantic_cache` caches answers and has to re-check the stored
    # gate every time it serves one, because an answer is a judgement a threshold could have
    # changed. These cache things that are deterministic for fixed inputs — a router decision is
    # a function of the question's words and the cards in scope; a section's reranker score is a
    # function of (probe, candidate text) — so a hit cannot serve what a config change should
    # have stopped. Both are invalidated by the per-KB version counter `semantic_cache` already
    # bumps on any ingest, delete or reindex.
    v2_router_cache_enabled: bool = True
    v2_router_cache_size: int = Field(default=512, ge=1, le=100_000)
    # Worth more than the router cache: on the real AEGCL corrigendum the section pass is nine
    # probes over seventy-two candidates, and EIGHT of those nine probes are the document's own
    # facet names — identical for every question asked of that document. Two differently-worded
    # compliance questions over the same corrigendum share eight of their nine rerank calls.
    v2_score_cache_enabled: bool = True
    # Entries are (probe, candidate text) -> float, so the cap is in section-scorings, not
    # questions: 72 sections x 9 probes is 648 for one document.
    v2_score_cache_size: int = Field(default=20_000, ge=1, le=1_000_000)

    # --- Conversational RAG (FR-I1, v1.1) — multi-turn chat over the same pipeline ---
    # A follow-up turn is condensed into a standalone query (so retrieval sees the full
    # intent) before entering the UNCHANGED answer pipeline — grounding, guardrails,
    # thresholds, cache and agentic retry all still apply per turn. The heuristic backend
    # is deterministic and adds no latency; "llm" rewrites via the tenant's own gateway.
    # "off" disables conversational memory entirely — every turn is answered standalone, no prior
    # chat is fused into the query (each question is independent; the thread is display-only).
    conversation_condense_backend: str = Field(default="heuristic", pattern="^(off|heuristic|llm)$")
    conversation_history_turns: int = Field(default=6, ge=1, le=40)

    # --- FinOps / usage showback (FR-H4) — indicative rate card, operator-configurable ---
    # Nominal defaults so showback works out of the box; each deployment sets its real
    # rates via env. Metering aggregates the audit trail (no token accounting yet).
    finops_currency: str = "INR"
    finops_cost_per_query: float = Field(default=0.5, ge=0.0)
    finops_cost_per_document: float = Field(default=2.0, ge=0.0)
    finops_cost_per_1k_tokens: float = Field(default=0.1, ge=0.0)

    # --- console / CORS (dev SPA at :5173 calls the API cross-origin) ---
    cors_allow_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- console UI toggles (surfaced non-secret via /v1/system/info `features`) ---
    # Show the KB "Integrations" tab (embeddable widget + chat-platform bot). The backends
    # stay available regardless; this only controls whether the console surfaces the tab.
    # Default OFF — a deployment that doesn't distribute via widget/chat keeps a leaner UI;
    # set true to re-surface it (no rebuild — the console reads it at runtime).
    ui_integrations_enabled: bool = False

    # --- HTTP security headers (VAPT hardening; applied to every API response) ---
    security_headers_enabled: bool = True
    # HSTS is only honoured by browsers over HTTPS; harmless over HTTP, so we always send it
    # once TLS is terminated at the ingress. Set to 0 to omit the header entirely.
    hsts_max_age: int = Field(default=31536000, ge=0)
    # The API returns JSON only (never HTML), so the strictest CSP is correct here — it
    # forbids the response from sourcing or framing anything. The SPA gets its own CSP in nginx.
    csp_api: str = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"
    permissions_policy: str = "geolocation=(), microphone=(), camera=(), payment=()"
    # Generic Server banner — replaces uvicorn's version-disclosing default (info-leak hardening).
    server_banner: str = "RagaaS"

    # --- observability ---
    otel_enabled: bool = True
    otel_exporter_otlp_endpoint: str = "http://otel-collector:4317"
    otel_service_name: str = "ragaas-api"

    @property
    def egress_allowlist(self) -> frozenset[str]:
        """Hosts permitted for outbound calls when the sovereignty guard is on."""
        return frozenset(h.strip() for h in self.allowed_egress_hosts.split(",") if h.strip())

    @property
    def doc_type_chunk_chars(self) -> dict[str, int]:
        """``layout_chunk_max_chars_by_doc_type`` parsed: ``{"corrigendum": 1200, ...}``.

        Same shape as :attr:`egress_allowlist` — a comma list on the field, a parsed value here.
        Keys are lowercased to match `DocProfile.doc_type`, which is lowercased for exactly this
        reason. A malformed or out-of-range pair is SKIPPED rather than raising: this is an
        operator tuning knob read on the ingest path, and a typo in it must not take ingestion down
        for every document in the deployment. The bounds mirror the `layout_chunk_max_chars` Field
        above, so the map cannot set a size the field itself would reject."""
        out: dict[str, int] = {}
        for pair in self.layout_chunk_max_chars_by_doc_type.split(","):
            name, _, raw = pair.partition("=")
            if not name.strip() or not raw.strip().isdigit():
                continue
            if _LAYOUT_CHUNK_MIN_CHARS <= (chars := int(raw.strip())) <= _LAYOUT_CHUNK_MAX_CHARS:
                out[name.strip().lower()] = chars
        return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide, cached settings instance."""
    return Settings()


def model_tls_verify() -> bool | str:
    """httpx ``verify=`` for MODEL-endpoint clients (LLM gateway, embedder, reranker):
    the operator's CA-bundle path when configured, else normal system verification.
    Never False — a private-CA endpoint gets its CA mounted, not verification disabled."""
    return get_settings().model_ca_bundle or True
