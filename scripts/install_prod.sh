#!/usr/bin/env bash
# install.sh — install RAG as a Service from the bundle, on a VM with no git and no registry.
#
# Run from inside the unpacked bundle directory. Idempotent: safe to re-run after fixing a value
# in .env.prod.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

ok()   { printf '\033[32m  ✓\033[0m %s\n' "$1"; }
warn() { printf '\033[33m  !\033[0m %s\n' "$1"; }
die()  { printf '\033[31m  ✗ %s\033[0m\n' "$1"; exit 1; }

printf '\nRAG as a Service — install\n\n'

command -v docker >/dev/null || die "docker is not installed"
docker compose version >/dev/null 2>&1 || die "docker compose v2 is required"
ok "docker present"

# 1. Integrity. A truncated scp otherwise surfaces later as a corrupt image and gets blamed on
#    the platform rather than on the transfer.
if [ -f SHA256SUMS ]; then
  sha256sum -c SHA256SUMS --quiet --ignore-missing 2>/dev/null \
    && ok "bundle checksums verified" \
    || die "checksum mismatch — the transfer is corrupt, copy the tarball again"
fi

# 2. Config must exist and be complete BEFORE anything starts. A missing model key does not fail
#    at boot; it fails at the first question, which is a far more expensive place to find it.
if [ ! -f .env.prod ]; then
  cp .env.prod.example .env.prod
  chmod 600 .env.prod
  warn "created .env.prod from the template — FILL IN EVERY BLANK, then re-run this script"
  printf '\n  Blanks to fill:\n'
  grep -nE '^[A-Z_]+=$' .env.prod | sed 's/^/    /'
  printf '\n  Generate the master key with:  openssl rand -base64 48\n\n'
  exit 1
fi
MISSING=$(grep -cE '^[A-Z_]+=$' .env.prod || true)
[ "$MISSING" -eq 0 ] || { grep -nE '^[A-Z_]+=$' .env.prod | sed 's/^/    /'; die "$MISSING setting(s) still blank in .env.prod"; }
chmod 600 .env.prod
ok ".env.prod complete"

# 3. Images.
if [ -f images.tar.gz ]; then
  printf '  … loading images (a few minutes)\n'
  gunzip -c images.tar.gz | docker load >/dev/null
  ok "images loaded"
fi

# 4. NLI weights, so the first boot does not reach for huggingface.co.
if [ -f nli-cache.tar.gz ]; then
  docker volume create ragaas_nli-cache >/dev/null
  docker run --rm -v ragaas_nli-cache:/to -v "$PWD":/from alpine \
    sh -c 'tar xzf /from/nli-cache.tar.gz -C /to' >/dev/null
  ok "NLI model weights restored (no download needed on first boot)"
else
  warn "no NLI weights in the bundle — first boot will pull ~1GB from huggingface.co"
fi

# 5. Start. The compose file gates the api on all four stores being healthy, so this waits.
printf '  … starting (OpenSearch takes ~40s to report healthy)\n'
docker compose -p ragaas --env-file .env.prod -f docker-compose.prod.yml up -d
ok "stack started"

# 6. Prove it works rather than assume it. /health goes green while the embedder 401s.
API_PORT=$(grep -E '^RAG_API_HOST_PORT=' .env.prod | cut -d= -f2); API_PORT=${API_PORT:-8000}
printf '  … waiting for the API\n'
for _ in $(seq 1 60); do
  curl -fsS --max-time 5 "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1 && break
  sleep 5
done
curl -fsS --max-time 5 "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1 \
  && ok "API healthy" \
  || die "API did not come up — docker compose -p ragaas logs api"

printf '\n  Now run the end-to-end check:\n    ./smoke_test.sh http://127.0.0.1:%s\n\n' "$API_PORT"
printf '  Then open the console and sign up — the first account creates the organisation.\n\n'
