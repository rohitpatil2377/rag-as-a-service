# Production deployment — image-based, no git on the target

Written for the engineer installing RAG as a Service on a customer VM.

The target VM needs Docker and network reach to the Acme model hosts. It does **not** need git,
internet, a GPU, or a checkout of this repository.

---

## 0. What this profile assumes

`docker-compose.prod.yml` is a single file — deliberately not the five-file dev merge, whose
order has caused real incidents (`mesh.yml` merges *after* `override.yml`, silently replacing
values set there). It also drops the p2p federation services and the test fixtures, which a
single-tenant customer POC does not use.

Models are **remote**: generation, embeddings, reranking and OCR all run on Acme hosts. The only
model on the VM is the NLI faithfulness checker (CPU). So the image bundle carries no weights.

| Runs on the VM | Runs remotely |
|---|---|
| api, console, postgres, qdrant, opensearch, minio, nli-checker | gemma4-31b, bge-m3, bge-reranker, chandra-ocr |

**Confirm before you start:** the VM can reach `models.example.internal` and
`models-uat.example.internal`. If it cannot, this profile will not work and the models must be
hosted locally — which needs a GPU and a different conversation.

---

## 1. Build the images (on a machine with the repo)

```bash
docker compose -f infra/compose/docker-compose.prod.yml build api console   # or tag your own
docker tag ragaas-api:latest     ragaas-api:v1
docker tag ragaas-console:latest ragaas-console:v1
docker save ragaas-api:v1 ragaas-console:v1 | gzip > ragaas-images.tgz
```

Carry `ragaas-images.tgz`, `docker-compose.prod.yml` and `.env.prod-vm.example` to the VM.
Pull the four store images (`postgres:16-alpine`, `qdrant/qdrant:v1.18.0`,
`opensearchproject/opensearch:2.17.1`, `minio/minio:...`) and the TEI image the same way if the
VM has no registry access.

## 2. Configure (on the VM)

```bash
cp .env.prod-vm.example .env.prod
$EDITOR .env.prod          # fill in EVERY blank — see the checklist below
docker load < ragaas-images.tgz
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d
```

### Secrets checklist

Nothing here has a safe default, and **a missing model key does not fail at boot** — it fails the
first time somebody asks a question, as a 401 from that model's host.

- [ ] `POSTGRES_PASSWORD`, `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD` — fresh, not the dev values
- [ ] `RAG_SECRETS_MASTER_KEY` — `openssl rand -base64 48`
- [ ] `RAG_LLM_API_KEY` + `RAG_VISION_LLM_API_KEY` (uat-bodhi)
- [ ] `RAG_EMBED_API_KEY`, `RAG_RERANK_API_KEY`, `RAG_CHANDRA_API_KEY` (poc-aiops)

## 3. Verify — do not skip this

This profile has been run end to end on a throwaway stack: signup, login, create KB, ingest,
query (grounded, cited, faithfulness 1.00 on gemma4-31b), and a correct abstain. Three bugs were
found by doing that rather than by reading it -- see the notes below.


```bash
scripts/smoke_test.sh http://localhost:8000
```

`/health` returns ok while the embedder 401s, the reranker is unreachable or the master key is
missing. The smoke test ingests a document and asks about it, which is the only check that
exercises the whole chain — and it asserts the platform **abstains** on an unanswerable question,
because a deployment that answers everything is the failure that matters most here.

Expect `7 passed, 0 failed`.

---

## 4. Things that will bite you

**`/v1/dev/token` must be closed.** At `RAG_ENV=dev` it mints an admin JWT for any tenant to
anyone who can reach the API. This profile sets `RAG_ENV=prod`, which disables it. The console
signs in through `/v1/auth/login`, so real users are unaffected. The smoke test tells you which
mode you are in.

**There is no TLS here.** The console serves plain HTTP. Terminate TLS in front of it before a
customer signs in — credentials and document content both cross that connection.

**Do not restore a Postgres dump onto a VM with a different master key.** Stored credentials are
encrypted with a key derived from `RAG_SECRETS_MASTER_KEY`. Restore without it and every stored
credential becomes undecryptable: the registry looks populated and nothing works. Start from an
empty database, or carry the key with the dump.

**You do not need to re-add the models through the UI.** Resolution falls back KB → org registry
→ platform default, and `.env.prod` sets the platform defaults to the same four models this
platform was tested against. The Models page will show only the read-only platform default; add
org entries through the UI only if per-KB model switching is wanted.

**First boot downloads the NLI model.** `nli-checker` pulls its weights once from huggingface.co
and caches them in a volume; every later start is offline. On a VM with no internet, pre-seed
that volume or point `HF_ENDPOINT` at an internal mirror — otherwise first boot hangs there.

**Adding a host to `RAG_ALLOWED_EGRESS_HOSTS` is a sovereignty decision**, not a config edit.
The allowlist is fail-closed and is the control that makes the sovereignty claim true. Real Azure
Blob (`<account>.blob.core.windows.net`) is a foreign endpoint and belongs in that conversation.

**The stores must be healthy before the api starts.** The compose gates on all four; without
that, an operator who brings the stack up and signs in immediately gets a 500 from a half-ready
OpenSearch and reasonably concludes the deployment is broken.

**Watch OpenSearch memory on first start.** Exit 137 (OOM-kill) during a concurrent build and
startup is how a previous deploy failed. `restart: unless-stopped` covers it; if it loops, give
the VM more RAM or lower `OPENSEARCH_JAVA_OPTS`.

---

## 5. Operating

| Task | Command |
|---|---|
| Logs | `docker compose --env-file .env.prod -f docker-compose.prod.yml logs -f api` |
| Backup | `scripts/backup.sh` — **restore has never been exercised against this stack; test it before relying on it** |
| Quality check | `scripts/eval_gate.py --kb <kb> --repeat 3` (needs a golden set on that KB) |
| Stale index | `GET /v1/kb/{id}/index-status` — non-zero `stale` means the pipeline changed since those documents were indexed |
