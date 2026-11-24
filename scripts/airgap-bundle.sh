#!/usr/bin/env bash
# airgap-bundle.sh — build a fully self-contained install bundle on a CONNECTED machine.
#
# The bundle is a single tarball you carry (USB / one-way transfer) into the air-gapped
# environment. It contains every docker image, the OSS model weights (warmed into their
# caches so there is NO runtime huggingface.co pull), the compose file, an env template,
# the matching installer, and SHA256SUMS.
#
# Design: images come from `docker compose config --images` so they never drift from the
# compose file. Model weights are captured by warming the inference servers ONCE here, then
# exporting their named cache volumes — this captures each server's exact on-disk layout
# (TEI, llama.cpp) with no guessing. Data volumes (qdrant/opensearch/minio/pg) are NOT
# bundled; they start empty on the target.
#
# Usage:
#   scripts/airgap-bundle.sh                      # full bundle into ./airgap-out
#   scripts/airgap-bundle.sh -o /tmp/out          # custom output dir
#   scripts/airgap-bundle.sh --list               # print images + volumes, do nothing
#   scripts/airgap-bundle.sh --skip-warm          # reuse already-populated model caches
#   scripts/airgap-bundle.sh --with-optional      # also bundle the vision model cache
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source infra/airgap/manifest.env

OUT="./airgap-out"
LIST_ONLY=0
SKIP_WARM=0
WITH_OPTIONAL=0
WARM_TIMEOUT="${WARM_TIMEOUT:-1200}"   # seconds to wait for model downloads

log() { printf '\033[1;36m[bundle]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[bundle] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    -o) OUT="$2"; shift 2 ;;
    --list) LIST_ONLY=1; shift ;;
    --skip-warm) SKIP_WARM=1; shift ;;
    --with-optional) WITH_OPTIONAL=1; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

command -v docker >/dev/null || die "docker not found"
docker compose version >/dev/null 2>&1 || die "docker compose v2 required"

export COMPOSE_PROJECT_NAME
DC=(docker compose -p "$COMPOSE_PROJECT_NAME" -f "$COMPOSE_FILE")

VOLS="$MODEL_VOLUMES"
[ "$WITH_OPTIONAL" = 1 ] && VOLS="$VOLS $OPTIONAL_MODEL_VOLUMES"

log "images from compose:"
IMAGES="$("${DC[@]}" config --images | sort -u)"
printf '  %s\n' $IMAGES >&2
log "model-cache volumes to bundle: $VOLS"

if [ "$LIST_ONLY" = 1 ]; then
  log "--list: nothing written."
  exit 0
fi

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
log "output dir: $OUT"

# 1. Build the images we own; pull the rest.
log "building local images (api, console) ..."
"${DC[@]}" build
log "pulling external images ..."
"${DC[@]}" pull --ignore-buildable || "${DC[@]}" pull || true

# 2. Warm the model caches (download weights once) unless reused.
if [ "$SKIP_WARM" = 0 ]; then
  log "warming model servers so caches populate (up to ${WARM_TIMEOUT}s) ..."
  "${DC[@]}" up -d
  deadline=$(( $(date +%s) + WARM_TIMEOUT ))
  # Consider the caches warm once every model volume is non-trivially populated.
  while :; do
    ready=1
    for v in $VOLS; do
      full="${COMPOSE_PROJECT_NAME}_${v}"
      bytes="$(docker run --rm -v "${full}:/v" busybox du -s /v 2>/dev/null | awk '{print $1}')"
      # >50 MiB (in 1K blocks) means real weights landed, not just an empty dir.
      [ "${bytes:-0}" -gt 51200 ] || ready=0
    done
    [ "$ready" = 1 ] && { log "all model caches populated."; break; }
    [ "$(date +%s)" -lt "$deadline" ] || die "timed out warming caches; inspect: ${DC[*]} logs"
    sleep 10
  done
  "${DC[@]}" stop
fi

# 3. Save images.
log "saving images -> images.tar (this is large) ..."
# shellcheck disable=SC2086
docker save $IMAGES -o "$OUT/images.tar"

# 4. Export model-cache volumes.
mkdir -p "$OUT/volumes"
for v in $VOLS; do
  full="${COMPOSE_PROJECT_NAME}_${v}"
  docker volume inspect "$full" >/dev/null 2>&1 || die "volume $full missing (warm first, or --skip-warm was wrong)"
  log "exporting volume $full ..."
  docker run --rm -v "${full}:/v:ro" -v "$OUT/volumes:/out" busybox \
    tar czf "/out/${v}.tgz" -C /v .
done

# 5. Stage compose + env + installer + manifest.
log "staging compose, env template, installer ..."
mkdir -p "$OUT/infra/compose" "$OUT/infra/airgap" "$OUT/scripts"
cp "$COMPOSE_FILE" "$OUT/infra/compose/"
cp infra/compose/.env.prod.example "$OUT/infra/compose/" 2>/dev/null || true
cp infra/airgap/manifest.env "$OUT/infra/airgap/"
cp scripts/airgap-install.sh "$OUT/scripts/"
chmod +x "$OUT/scripts/airgap-install.sh"
printf '%s\n' "$IMAGES" > "$OUT/IMAGES.txt"
printf '%s\n' "$VOLS" | tr ' ' '\n' > "$OUT/MODEL_VOLUMES.txt"

# 6. Checksums + final tarball.
log "writing SHA256SUMS ..."
( cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )

STAMP_VER="${BUNDLE_VERSION:-$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo dev)}"
TARBALL="ragaas-airgap-${STAMP_VER}.tar"
log "creating $TARBALL ..."
tar cf "$TARBALL" -C "$(dirname "$OUT")" "$(basename "$OUT")"

log "DONE."
log "bundle dir : $OUT"
log "tarball    : $REPO_ROOT/$TARBALL  ($(du -h "$TARBALL" | awk '{print $1}'))"
log "next: copy the tarball to the air-gapped host and run scripts/airgap-install.sh"
