# RAG as a Service

A sovereign, multi-tenant, agentic **RAG-as-a-Service** platform: the intelligence layer
between your document store and the applications and agents that consume it. Point it at
your data and get cited, trustworthy answers — or an explicit abstention — with every
component self-hosted, so nothing leaves your own infrastructure.

Every answer carries citations to its sources, or the platform says it does not know.
That invariant is enforced at the type level, not by convention: a `GroundedAnswer` with
`status=GROUNDED` and zero citations fails validation.

- **Architecture map:** [`CLAUDE.md`](CLAUDE.md) — the layer-by-layer tour of the repo
- **Design decisions:** [`docs/adr/`](docs/adr/) — 26 ADRs covering why each seam looks the way it does
- **Operator guide:** [`docs/PLATFORM_GUIDE.md`](docs/PLATFORM_GUIDE.md)
- **Visual language:** [`design/tokens/`](design/tokens/)

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/) and Docker + Compose v2.

```bash
make install    # uv sync — all libs/services/apps + dev tools
make ci         # the quality gate: ruff (whole repo), mypy strict, 2,947 unit tests, ~2 min
make demo       # ingest a sample doc -> grounded cited answer -> abstain demo
```

To run against the real stack instead of in-memory adapters:

```bash
make up         # docker compose: Qdrant, OpenSearch, S3, Postgres, TEI, API, console, observability
make migrate    # Alembic baseline schema against the live Postgres
make demo ARGS=--stores   # or: uv run python -m demo_cli.run --stores --ask "..."
```

`make up` is **self-contained**: no secrets, no model endpoint, no GPU, nothing to sign up for.
It starts 13 services and generation defaults to `extractive_dev`, which composes the answer out
of the retrieved spans — so retrieval, citations and the grounded-or-abstain gate are all
exercised, deterministically, with no model to download. For real local generation:

```bash
docker compose -f infra/compose/docker-compose.dev.yml --profile llm up -d llama
# then set on the api: RAG_LLM_BACKEND=openai_compat, RAG_VLLM_URL=http://llama:8080/v1
```

That pulls a ~2 GB Qwen2.5-3B GGUF once and then runs offline. The NLI faithfulness checker is
likewise opt-in (`--profile nli`); the default lexical checker needs no model.

**After a code change, use `make redeploy`** (`SVC=api` for one service) — never a hand-written
`docker compose -f docker-compose.dev.yml ...` line. The dev file is the base; a configured box
copies the `*.example.yml` overlays next to it to add model endpoints (`deploy`), secrets
(`override`), federation nodes (`mesh`) and resource limits (`local`), and the Makefile layers
whichever exist. Running `dev.yml` alone on such a box silently drops the model configuration:
embeddings fall back to the local 384-dim default and every query then fails against a 1024-dim
collection with `expected dim: 1024, got 384`. `make compose-files` prints the set the Makefile
uses; a running container records its own, so the two should match:

```bash
make redeploy                 # rebuild + restart api and console
make redeploy SVC=api         # just the API
make compose-files            # what the Makefile will use
docker inspect ragaas-dev-api-1 \
  --format '{{index .Config.Labels "com.docker.compose.project.config_files"}}'
```

Console: http://localhost:8082 · API docs: http://localhost:8000/docs · Grafana:
http://localhost:3000 (admin/admin) · Qdrant dashboard: http://localhost:6333/dashboard

> **On the object store:** the dev stack serves S3 from **SeaweedFS**, not MinIO. MinIO withdrew
> their public container images (the Docker Hub repository is gone; quay.io answers 401 to an
> anonymous pull), which made a pinned tag unpullable and broke `make up` on a fresh clone. The
> wire protocol, credentials and bucket layout are unchanged, and the `MinioObjectStore` adapter
> is untouched — it speaks S3 through the `minio` Python *client*, which is unaffected. The
> compose service is still named `minio` because that name is the hostname in several env vars.

## What a complete validation run is

THREE COMMANDS, and a green `make ci` is the first of them, not all of them:

```bash
make ci                              # ruff over the WHOLE repo, mypy strict over
                                     # libs services apps sdks/python, 2,947 unit tests
uv run pytest -q -m "e2e"            # 394 tests, in-memory profile, no infrastructure
make up && uv run pytest -q -m "integration"   # 114 tests, needs the live stores
```

`make test` is `pytest -m "not integration and not e2e and not eval"`, so **`make ci` excludes
the 394 e2e tests**. The GitHub `quality` job runs them in a step of their own, but the local
gate does not — so run them yourself before claiming work is done, and never describe a partial
run as a full repository pass.

**Do not substitute a path-scoped lint or typecheck for `make ci`.** The scopes differ by whole
directories and the difference is invisible until something ships: `mypy libs services apps`
omits `sdks/python`, and `ruff check libs services apps` omits the repo-root `tests/` directory
entirely. Both scopes have let a defect through before, which is why the gate is defined once.

`make hooks` installs `.githooks/pre-push`, which runs `make ci` automatically. It is a local
safeguard, not enforcement: it needs installing in each clone and `--no-verify` bypasses it.
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) is the real gate, and it runs only on
`main` and on pull requests — a feature branch pushed straight up is covered by the hook alone.
The workflow calls `make ci` rather than restating its commands, because restating them is how
its mypy scope drifted from the Makefile's once before; it then runs the e2e suite as a separate
step, since `make ci` excludes it and it needs no infrastructure.

## Real stores vs in-memory

Every store/model Protocol (`VectorStore`, `SparseIndex`, `ObjectStore`,
`CatalogAdapter`, `AuditSink`, `Embedder`, `LLMGateway`, `Reranker`) has **two
implementations**: a real adapter (Qdrant/OpenSearch/MinIO/Postgres/TEI/vLLM) and an
in-memory twin (`libs/clients/src/rag_clients/memory.py`) with the same per-tenant
namespacing. One env var switches between them:

| | Default (`make test`, CI) | Real stores |
|---|---|---|
| Trigger | nothing set | `RAG_INMEMORY=true` set by the code path, or `RAG_TEST_REAL_STORES=true` for tests |
| Speed | ~4s, zero infra | requires `make up` first |
| LLM/embeddings | `LocalHashEmbedder` (deterministic hash, not semantic) + `ExtractiveDevGateway` (no model) | TEI (`bge-small-en-v1.5`) + `extractive_dev` or real vLLM (`--profile gpu`) |
| Forbidden in prod | — | in-memory is forbidden when `RAG_ENV=prod` (factory raises) |

**Run the exact same test suite against real stores:**
```bash
make up
export RAG_TEST_REAL_STORES=true RAG_ENV=dev RAG_OTEL_ENABLED=false RAG_LOG_JSON=false
export RAG_QDRANT_URL=http://localhost:6333 RAG_OPENSEARCH_URL=http://localhost:9200
export RAG_MINIO_ENDPOINT=localhost:9000 RAG_TEI_URL=http://localhost:8080
export RAG_RERANK_URL=http://localhost:8083
export RAG_POSTGRES_DSN="postgresql+psycopg://rag:rag@localhost:5432/rag"
# REQUIRED, and the easiest thing to get wrong. These are container env in the compose
# file, but a host-side pytest run builds its own Settings, whose defaults are bge-m3/1024
# while the dev `tei` service serves bge-small at 384. Omit them and every collection is
# created at 1024, then the first upsert dies with "expected dim: 1024, got 384".
export RAG_EMBED_BACKEND=tei RAG_EMBED_MODEL=BAAI/bge-small-en-v1.5 RAG_EMBED_DIM=384
export RAG_LLM_BACKEND=extractive_dev RAG_ALLOWED_EGRESS_HOSTS=localhost
uv run alembic -c libs/clients/alembic.ini upgrade head
uv run pytest -q -m integration
```
(If you remapped a published port because of a clash — see
`infra/compose/docker-compose.local.example.yml` — use the remapped port in the DSN/URLs.)
Both profiles are green. The integration suite was last run against live Qdrant +
OpenSearch + SeaweedFS + Postgres at **113 passed, 1 skipped** — twice consecutively, which
is the check that matters, because two of its tests used to pass only on a virgin database.
The skip is the multivector isolation probe, which has nothing to assert under a
single-vector embedder. Real-adapter bugs found and fixed during the first pass (an
unprovisioned tenant's search must return "no results," not an error; exception
normalisation on `ObjectStore.get`; a missing `ensure_schema` call; connection-pool leaks
with no `aclose` on shutdown) are documented in
[`docs/adr/0004-inmemory-adapters-for-fast-ci.md`](docs/adr/0004-inmemory-adapters-for-fast-ci.md).

## Cross-department sharing: the shared KB (Federated RAG Tier C)

Three distinct tiers govern who can read what, and they compose rather than overlap:

- **Tier A — within a department.** Per-document `classification` (0=Public..3=Restricted)
  + `acl` groups + org-unit hierarchy (`rag_common.rbac.can_read_kb`). Governs who *inside*
  one tenant sees what. Unchanged by this feature.
- **Tier B — targeted, grant-gated cross-department.** The signed-Grant federation mesh
  (`POST /v1/grants`, `federate_from` on `/v1/ask`) — one department explicitly authorises
  *one* other department for specific KBs/topics. See the P2P federation docs/ADRs.
  Unchanged by this feature.
- **Tier C — company-wide, no grant.** `SHARED_KB_ID` (`public__company-wide`,
  `svc_control_plane.knowledge_base`) is the one platform-wide KB, owned by no tenant.
  Any `ingestor`-role principal, in any tenant, may choose it as an ingest destination
  instead of their own KB — a one-time decision at ingest time, not a later toggle. Every
  tenant's default `/v1/ask` (no `kb_ids`, no `federate_from`) searches it automatically
  alongside their own KBs, so a Tech-department question about HR's leave policy just
  works if HR chose to put that document here.

Rules worth knowing before using it:
- **Public-only, enforced.** Ingesting anything above `classification=0` into the shared
  KB is a 403 — it's readable by literally everyone, so it can never hold Confidential/
  Restricted content, even by mistake.
- **No duplication.** A document lives in exactly one place (its own department's KB, or
  the shared KB) — never both, so there's no drift between two copies to keep in sync.
  Changing your mind later means re-ingesting into the other KB, same as any document move.
  Existing content is unaffected — nothing is retroactively shared.
- **Retrieval across KBs is concurrent, not sequential** (`pipeline.answer_across`) — every
  KB in play (own + shared, or an explicit multi-KB `kb_ids`) is queried at once, so the
  shared KB riding along on every default ask doesn't multiply query latency by KB count.

## Memory sizing

The default `make up` is laptop-sized: **13 containers, ~2.5–3 GB resident** (measured with
`docker stats` on a 16 GB machine). Nothing large is downloaded, because generation runs on
`extractive_dev` and both model servers are opt-in. Opting in costs roughly:

| Profile | Adds | Extra RAM | First-start download |
|---|---|---|---|
| default | 13 services | ~2.5–3 GB | TEI pulls `bge-small-en-v1.5` once (~130 MB) |
| `--profile llm` | llama.cpp (Qwen2.5-3B) | ~3 GB | ~2 GB GGUF, once |
| `--profile nli` | NLI faithfulness checker | ~1.5 GB | ~1 GB, once |

Every heavy container declares an explicit `mem_limit`, and OpenSearch's JVM heap is capped
(`-Xms512m -Xmx1g`). That matters on a box **without swap**, where an OOM spike kills a
container outright rather than slowing it down: the limits stop one service starving the
others. If the host is tight, copy `infra/compose/docker-compose.local.example.yml` to
`docker-compose.local.yml` and shrink the heap there. Note that `docker stats` showing low
`free` is normal — Linux holds reclaimable page cache, and `available` is the OOM-relevant
figure. Dev profile only; production sizing lives in `infra/helm`.

**LLM tier.** The default needs no model at all. Real local generation is `llama.cpp` on CPU
serving **Qwen2.5-3B-Instruct** (GGUF Q4_K_M, Apache-2.0), a few seconds per short answer —
drop to the 1.5B GGUF if latency matters more than answer quality. vLLM plus a larger model is
the GPU tier (`--profile gpu`). See
[`docs/adr/0006-llm-serving-and-model-choice.md`](docs/adr/0006-llm-serving-and-model-choice.md).

**Why keep in-memory as the default:** it doubles as an adapter-conformance check (the
same test bodies run against both, so a real adapter that silently drifts from its
Protocol contract fails a test) and keeps the inner dev loop fast. It is never a
substitute for the real-store pass before trusting a change — see the ADR.

**Sovereignty note on TEI:** on first `make up`, the `tei` service downloads
`BAAI/bge-small-en-v1.5` from `huggingface.co` once, then caches the weights in the
`tei-cache` volume — every later start is fully offline. This is a **documented,
one-time local dev bootstrap**, not a runtime call the platform makes — the runtime
egress guard (`rag_common.sovereignty.assert_sovereign`) is unaffected and still
blocks the running API from calling any non-allowlisted host. Production must point
`HF_ENDPOINT` at an internal model mirror instead, which keeps the sovereignty guarantee intact.

## Repository layout

```
apps/            api      — rag_api, the modular monolith that mounts every service router
                 console  — React + Vite + TS admin SPA (nginx-served, proxies /v1)
                 mcp      — MCP server (stdio + streamable HTTP) for agent clients
                 widget   — dependency-free embeddable chat widget
                 demo-cli — walking-skeleton driver (runs over ASGI, no network)
services/        7 service modules: ingestion, doc-intelligence, indexing, retrieval,
                 generation, guardrails-eval, control-plane (the /v1 API, HITL, workflows)
libs/            common (config/logging/tenancy/auth/sovereignty), schemas (pydantic
                 contracts), rag-core (Protocols + pure logic, no framework lock-in),
                 clients (real adapters + their in-memory twins)
sdks/            python + typescript clients over /v1
sovereign-node/  Go libp2p node for the opt-in federation mesh (vendored deps)
design/tokens/   design tokens (CSS vars, JSON, Tailwind preset)
infra/           compose (local dev stack), helm (k8s chart), k8s, airgap bundle,
                 otel/prometheus/grafana (observability)
docs/adr/        26 Architecture Decision Records — read before reversing a decision
scripts/         operational tooling: eval gate, RAGAS gate, calibration, backup, air-gap
eval/            typed gold suites for the evaluation harness
tests/           integration (tenant isolation, real stores), e2e (vertical slice)
sample-data/     demo corpus + golden/ (eval ground truth)
```

## Development

```bash
make lint         # ruff check + format check
make fmt           # ruff format + autofix
make typecheck     # mypy --strict across libs/services/apps
make ci             # lint + typecheck + test (what CI runs on every PR)
```

No secrets in code or the repo — copy `.env.example` to `.env` for local config;
production secrets come from the `SecretsProvider` (Vault/HSM adapter, Phase-1).
