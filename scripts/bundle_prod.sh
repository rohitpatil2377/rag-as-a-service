#!/usr/bin/env bash
# bundle_prod.sh — build ONE transferable bundle for a production VM.
#
# The target VM has no git and no registry access. Everything it needs travels in a single
# tarball you scp across: every docker image, the NLI model weights, the compose file, the
# environment template, the installer and the smoke test.
#
# Model weights: only the NLI checker's. Generation, embeddings, reranking and OCR are REMOTE
# (poc-aiops / uat-bodhi), so their weights never enter the bundle -- that is the whole reason
# this is ~2GB rather than ~40GB. The NLI weights ARE included because that model runs on the VM
# and would otherwise be pulled from huggingface.co on first boot, which is the one step that can
# hang an install in front of a customer.
#
# Secrets are NEVER bundled. The template ships blank; it is filled in on the target.
#
#   scripts/bundle_prod.sh              # -> ./dist/ragaas-prod-<date>.tar.gz
#   scripts/bundle_prod.sh -o /tmp/out
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
OUT="./dist"
[ "${1:-}" = "-o" ] && OUT="$2"

COMPOSE="infra/compose/docker-compose.prod.yml"
STAMP="$(date +%Y%m%d)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

log() { printf '\033[1;36m[bundle]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[bundle] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# 1. Build the two images we own. Tagged exactly as the compose file expects.
log "building api + console"
docker build -q -t ragaas-api:latest     -f apps/api/Dockerfile      . >/dev/null
docker build -q -t ragaas-console:latest -f apps/console/Dockerfile  . >/dev/null

# 2. Every image the compose file names. Derived from the file itself so the bundle can never
#    drift from what will actually run -- a hand-maintained list is how a store image gets
#    forgotten and the install fails on a VM with no way to pull it.
log "resolving images from $COMPOSE"
IMAGES=$(docker compose --env-file infra/compose/.env.prod-vm.example -f "$COMPOSE" config --images 2>/dev/null | sort -u)
[ -n "$IMAGES" ] || die "no images resolved from $COMPOSE"
printf '%s\n' "$IMAGES" | sed 's/^/  /' >&2

log "pulling any image not present locally"
for img in $IMAGES; do
  docker image inspect "$img" >/dev/null 2>&1 || { log "  pull $img"; docker pull -q "$img" >/dev/null; }
done

log "saving images (this is the slow part)"
# shellcheck disable=SC2086
docker save $IMAGES | gzip -1 > "$STAGE/images.tar.gz"

# 3. The NLI weights, taken from a volume that already holds them.
NLI_VOL="$(docker volume ls --format '{{.Name}}' | grep -E '_nli-cache$' | head -1)"
if [ -n "$NLI_VOL" ]; then
  log "exporting NLI model weights from $NLI_VOL"
  docker run --rm -v "$NLI_VOL":/from -v "$STAGE":/to alpine \
    tar czf /to/nli-cache.tar.gz -C /from . 2>/dev/null
else
  log "WARNING: no nli-cache volume found — the target will download the NLI model on first boot"
fi

# 4. Everything a human needs on the other side.
cp "$COMPOSE"                          "$STAGE/docker-compose.prod.yml"
cp infra/compose/.env.prod-vm.example  "$STAGE/.env.prod.example"
cp docs/DEPLOY_PROD.md                 "$STAGE/DEPLOY.md"
cp scripts/smoke_test.sh               "$STAGE/smoke_test.sh"
cp scripts/install_prod.sh             "$STAGE/install.sh"
chmod +x "$STAGE/install.sh" "$STAGE/smoke_test.sh"

# 5. Checksums, so a truncated scp is caught before it is blamed on the platform.
( cd "$STAGE" && sha256sum ./* > SHA256SUMS )

mkdir -p "$OUT"
TARBALL="$OUT/ragaas-prod-$STAMP.tar.gz"
tar czf "$TARBALL" -C "$STAGE" .
log "done: $TARBALL ($(du -h "$TARBALL" | cut -f1))"
cat >&2 <<EOF

  Next:
    scp $TARBALL <user>@<prod-vm>:/opt/
    ssh <user>@<prod-vm>
    mkdir -p /opt/ragaas && tar xzf /opt/$(basename "$TARBALL") -C /opt/ragaas
    cd /opt/ragaas && ./install.sh

EOF
