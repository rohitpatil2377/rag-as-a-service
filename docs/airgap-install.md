# Air-gapped installation

Install the RAG as a Service / RAG as a Service stack in an environment with **no internet** —
no `huggingface.co`, no container registry, no PyPI. The platform is already designed to be
sovereign (every component self-hostable, runtime egress guard on); this makes the *install*
sovereign too: one tarball in, everything runs offline.

## How it works

The single friction point for an offline install is that the inference servers (TEI,
llama.cpp) normally download their OSS model weights from Hugging Face on first run. The
bundle removes that: on a **connected build machine** we warm those servers once, then
capture their populated cache volumes. On the target, we restore those volumes and start
with `HF_HUB_OFFLINE=1`, so any accidental fetch fails loudly instead of silently phoning
home.

```
 CONNECTED build box                        AIR-GAPPED target
 ───────────────────                        ─────────────────
 airgap-bundle.sh                           airgap-install.sh
   • docker compose build/pull   ─┐         • sha256sum -c  (integrity)
   • warm model caches            │  USB    • docker load images.tar
   • docker save  -> images.tar   ├───────► • restore model-cache volumes
   • export model volumes .tgz    │  scp    • docker compose up -d  (offline)
   • SHA256SUMS + tarball        ─┘         • health check
```

What is bundled: every image from `docker compose config --images`, the model-weight cache
volumes (`tei-cache`, `reranker-cache`, `nli-cache`, `llama-cache`), the compose file, an
env template, the installer, and `SHA256SUMS`.

What is **not** bundled: data volumes (`qdrant-data`, `opensearch-data`, `minio-data`,
`pg-data`) — those start empty on the target and fill as you ingest.

## Prerequisites

Both machines: Docker Engine + Docker Compose v2. The build machine also needs internet and
enough disk for the images + weights (budget ~15–25 GB for the bundle; the CPU model set is
a few GB, images the rest).

## Build the bundle (connected machine)

```bash
# From the repo root:
scripts/airgap-bundle.sh --list          # sanity: show images + volumes, do nothing
scripts/airgap-bundle.sh                 # full bundle -> ./airgap-out + ragaas-airgap-<ver>.tar
# options:
#   -o DIR            output dir (default ./airgap-out)
#   --with-optional   also bundle the vision model (nomic-embed-vision)
#   --skip-warm       reuse caches already populated by a prior `make up`
```

Copy `ragaas-airgap-<ver>.tar` to your transfer medium.

## Install (air-gapped target)

```bash
tar xf ragaas-airgap-<ver>.tar && cd airgap-out
./scripts/airgap-install.sh --verify-only     # check integrity first (optional)
./scripts/airgap-install.sh                    # load + restore + start, fully offline
```

The installer creates `infra/compose/.env` from the prod template on first run.

## After install — before production

- **Edit `infra/compose/.env`**: set `RAG_ENV=prod`, a strong `RAG_JWT_SECRET`, real
  Postgres/MinIO credentials, and confirm `RAG_AUTH_MODE` (OIDC for prod; the dev-token
  endpoint is disabled when `RAG_ENV=prod`).
- **Egress allowlist**: air-gapped means the runtime sovereignty guard should have an empty
  or intranet-only allowlist. Do not add public hosts.
- **Model mirror (optional)**: if you later want to update models without this warm-and-
  export dance, point `HF_ENDPOINT` at an internal Hugging Face mirror and drop
  `HF_HUB_OFFLINE`. Until then, offline caches are authoritative.
- **MCP over the network**: the MCP endpoint at `/mcp` works the same offline — see
  `apps/mcp/plugin/README.md` to connect agents on the intranet.

## Updating an existing air-gapped install

Re-run the bundle on the connected box (it re-warms to pick up model/image changes), carry
the new tarball over, and run the installer again. It skips volumes that are already
populated; delete a cache volume first if you intend to replace its weights.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `checksum mismatch` | Bundle corrupted in transit — re-copy the tarball. |
| API not healthy | `docker compose -p ragaas-airgap logs api` — usually a store not up yet or a bad `.env`. |
| A model server restarts / 500s | Its cache volume didn't restore. Confirm `volumes/<name>.tgz` exists and re-run install; check `HF_HUB_OFFLINE=1` isn't masking a genuinely missing model. |
| `timed out warming caches` (build) | Slow download; raise `WARM_TIMEOUT=1800 scripts/airgap-bundle.sh`. |
