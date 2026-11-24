#!/usr/bin/env bash
# airgap-install.sh — install the RAG as a Service stack on an AIR-GAPPED host.
#
# Run this from inside an extracted bundle (produced by airgap-bundle.sh). It verifies
# integrity, loads every docker image, restores the model-weight caches, then brings the
# stack up entirely offline — no network, no huggingface.co.
#
# Usage:
#   ./scripts/airgap-install.sh                 # verify -> load -> restore -> up
#   ./scripts/airgap-install.sh --verify-only   # just check SHA256SUMS
#   ./scripts/airgap-install.sh --no-up         # load + restore, do not start the stack
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BUNDLE_ROOT"
# shellcheck source=/dev/null
source infra/airgap/manifest.env

VERIFY_ONLY=0
NO_UP=0
log() { printf '\033[1;36m[install]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[install] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --verify-only) VERIFY_ONLY=1; shift ;;
    --no-up) NO_UP=1; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

command -v docker >/dev/null || die "docker not found on target"
docker compose version >/dev/null 2>&1 || die "docker compose v2 required on target"

# 1. Integrity.
log "verifying SHA256SUMS ..."
[ -f SHA256SUMS ] || die "SHA256SUMS missing — is this a complete bundle?"
sha256sum -c SHA256SUMS --quiet || die "checksum mismatch — bundle is corrupt or tampered"
log "integrity OK."
[ "$VERIFY_ONLY" = 1 ] && { log "--verify-only: done."; exit 0; }

# 2. Load images.
log "loading docker images (large) ..."
docker load -i images.tar

# 3. Restore model-weight caches into named volumes (skip if already present + populated).
export COMPOSE_PROJECT_NAME
for tgz in volumes/*.tgz; do
  [ -e "$tgz" ] || continue
  v="$(basename "$tgz" .tgz)"
  full="${COMPOSE_PROJECT_NAME}_${v}"
  if docker volume inspect "$full" >/dev/null 2>&1 \
     && [ "$(docker run --rm -v "${full}:/v" busybox du -s /v 2>/dev/null | awk '{print $1}')" -gt 51200 ]; then
    log "volume $full already populated — skipping."
    continue
  fi
  log "restoring $full from $tgz ..."
  docker volume create "$full" >/dev/null
  docker run --rm -v "${full}:/v" -v "$BUNDLE_ROOT/volumes:/in:ro" busybox \
    tar xzf "/in/$(basename "$tgz")" -C /v
done

# 4. Env file.
COMPOSE_FILE_LOCAL="infra/compose/$(basename "$COMPOSE_FILE")"
if [ ! -f infra/compose/.env ] && [ -f infra/compose/.env.prod.example ]; then
  cp infra/compose/.env.prod.example infra/compose/.env
  log "created infra/compose/.env from the prod template — REVIEW IT before production use"
  log "  (set real secrets: RAG_JWT_SECRET, DB/MinIO creds, RAG_ENV=prod, egress allowlist)"
fi

if [ "$NO_UP" = 1 ]; then
  log "--no-up: images loaded + caches restored. Start later with:"
  log "  COMPOSE_PROJECT_NAME=$COMPOSE_PROJECT_NAME docker compose -f $COMPOSE_FILE_LOCAL up -d"
  exit 0
fi

# 5. Bring the stack up — offline. HF_HUB_OFFLINE hard-fails any accidental model fetch.
log "starting the stack (offline) ..."
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  docker compose -p "$COMPOSE_PROJECT_NAME" -f "$COMPOSE_FILE_LOCAL" up -d

log "waiting for the API to report healthy ..."
for _ in $(seq 1 60); do
  if curl -fsS -m3 http://localhost:8000/health >/dev/null 2>&1; then
    log "API healthy. Console: http://<host>:8082/  API: http://<host>:8000/"
    exit 0
  fi
  sleep 5
done
die "API did not become healthy in time — check: docker compose -p $COMPOSE_PROJECT_NAME logs api"
