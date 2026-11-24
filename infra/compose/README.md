# Compose stacks

## The layering

`docker-compose.dev.yml` is the **base and is self-contained** — `make up` on a fresh clone needs
no secrets, no model endpoint and no GPU. Everything else is an optional overlay, included by the
Makefile only if the file exists:

| File | Adds | Tracked? |
|---|---|---|
| `docker-compose.dev.yml` | The 13-service dev stack | yes |
| `docker-compose.deploy.example.yml` | Hosted model endpoints, thresholds, feature flags | example only |
| `docker-compose.override.example.yml` | API-key **values** — the only secrets | example only |
| `docker-compose.mesh.example.yml` | Federation: per-department API instances + libp2p nodes | example only |
| `docker-compose.local.example.yml` | Box-local resource limits and port remaps | example only |

Each `*.example.yml` is **copied** to the name without `.example` to activate it; the activated
copies are gitignored because they are machine-specific. `make compose-files` prints the set in use.

**This ordering is not decorative.** Running `docker compose -f docker-compose.dev.yml` alone on a
box that has a deploy overlay silently drops the model configuration, and every query then fails
with `expected dim: 1024, got 384`. Recover the set a running container was built from:

```bash
docker inspect ragaas-dev-api-1 \
  --format '{{index .Config.Labels "com.docker.compose.project.config_files"}}'
```

## Two gotchas

- **Compose appends list fields.** An overlay that sets `ports:` adds to the base mapping instead
  of replacing it, so a port remap needs `ports: !override [...]` or the original binding survives
  and still collides.
- **The `minio` service does not run MinIO.** MinIO withdrew their public images, so the dev and CI
  stacks serve S3 from SeaweedFS. The service keeps the name because it is the hostname in
  `RAG_MINIO_ENDPOINT`, the egress allowlist, `scripts/wait_for_stores.py` and the test env.

## Opt-in profiles

```bash
docker compose -f docker-compose.dev.yml --profile llm up -d llama        # real local generation
docker compose -f docker-compose.dev.yml --profile nli up -d nli-checker  # NLI faithfulness
docker compose -f docker-compose.dev.yml --profile gpu up -d              # vLLM + GPU reranker
```

`docker-compose.ci.yml` + `ci.env` is a separate, disposable stack on remapped ports, so a
real-store test run never touches a shared dev stack. `docker-compose.prod.yml` is a production
topology and is not what `make up` uses.
