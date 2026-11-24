#!/usr/bin/env bash
# Bootstrap the decentralized N-department mesh (currently dept-a, dept-b, dept-c, dept-d).
# Onboarding a new department is: add its name/service to DEPTS/SVCS below, add matching
# node-<x>/api-<x> blocks to docker-compose.mesh.example.yml (same pattern as the existing ones), then
# re-run this script — steps 1-6 below are all indexed off DEPTS/SVCS, not hardcoded to 3.
#
# Runs ON the deployment host (the VM), from infra/compose/. It:
#   1. seeds an empty registry so the bind-mount exists,
#   2. builds + starts the three sovereign-nodes (each generates + persists a libp2p identity),
#   3. reads each node's printed PeerID,
#   4. mints a per-department bridge token (dev JWT for that dept's tenant),
#   5. writes the shared operator registry (dept -> shared dev grant-secret + /dns4 peer addr),
#   6. recreates the nodes with their bridge token + the populated registry so they dial peers.
#
# Idempotent: re-running regenerates the registry from the current identities.
#
# NOTE (dev/POC): all departments derive their grant-signing key from one shared dev secret
# (the Python signer's default), so a grant signed by the api verifies at every node — this is
# the documented dev-interop model. In production each dept gets a DISTINCT KMS-backed key and
# the registry carries `pubkey_b64` instead of `dev_secret`.
set -euo pipefail

CD="$(cd "$(dirname "$0")/.." && pwd)"          # infra/compose
COMPOSE="docker compose \
  -f $CD/docker-compose.dev.yml \
  -f $CD/docker-compose.deploy.example.yml \
  -f $CD/docker-compose.override.yml \
  -f $CD/docker-compose.mesh.example.yml"
PROJECT="ragaas-dev"
API_CTR="${API_CTR:-ragaas-dev-api-1}"
PY="/app/.venv/bin/python"
DEV_SECRET="${MESH_DEV_SECRET:-dev-only-insecure-secret-change-me-32b+}"
REG="$CD/mesh/registry.json"

DEPTS=(dept-a dept-b dept-c dept-d)
SVCS=(node-a node-b node-c node-d)
N=${#DEPTS[@]}

echo "==> [1/6] seed empty registry"
mkdir -p "$CD/mesh"; echo '{"departments":[]}' > "$REG"

echo "==> [2/6] build + start nodes (generate identities)"
$COMPOSE up -d --build "${SVCS[@]}"

echo "==> [3/6] read each node's PeerID"
declare -A PID
for i in $(seq 0 $((N-1))); do
  ctr="${PROJECT}-${SVCS[$i]}-1"
  line=""
  for _ in $(seq 1 30); do
    line=$(docker logs "$ctr" 2>&1 | grep -m1 'peer id:' || true)
    [ -n "$line" ] && break; sleep 1
  done
  [ -z "$line" ] && { echo "ERROR: ${SVCS[$i]} never printed a peer id"; docker logs --tail 20 "$ctr"; exit 1; }
  PID[${SVCS[$i]}]=$(echo "$line" | awk '{print $NF}')
  echo "    ${DEPTS[$i]} -> ${SVCS[$i]} peer id ${PID[${SVCS[$i]}]}"
done

# Bringing the nodes up reconciles the api (depends_on) and can recreate it, so wait for the
# api to be READY before minting tokens against it — otherwise step 4 races a restarting api.
echo "==> waiting for api readiness"
for _ in $(seq 1 24); do
  code=$(docker exec "$API_CTR" $PY -c "import httpx;print(httpx.get('http://localhost:8000/health',timeout=5).status_code)" 2>/dev/null || true)
  [ "$code" = "200" ] && { echo "    api ready"; break; }
  sleep 5
done

echo "==> [4/6] mint per-department bridge tokens"
declare -A TOK
for i in $(seq 0 $((N-1))); do
  d="${DEPTS[$i]}"
  TOK[$d]=$(docker exec -i "$API_CTR" $PY - "$d" <<'PY'
import sys, httpx
dept = sys.argv[1]
r = httpx.post("http://localhost:8000/v1/dev/token",
               json={"subject": f"mesh-bridge@{dept}", "tenant_id": dept,
                     "roles": ["owner", "admin", "reader"]}, timeout=15)
r.raise_for_status(); print(r.json()["access_token"])
PY
)
  echo "    ${d}: token minted"
done

echo "==> [5/6] write shared operator registry"
{
  echo '{'
  echo '  "departments": ['
  for i in $(seq 0 $((N-1))); do
    d="${DEPTS[$i]}"; s="${SVCS[$i]}"
    sep=","; [ "$i" -eq $((N-1)) ] && sep=""
    echo "    { \"dept\": \"${d}\", \"dev_secret\": \"${DEV_SECRET}\", \"peer_addr\": \"/dns4/${PROJECT}-${s}-1/tcp/4001/p2p/${PID[$s]}\" }${sep}"
  done
  echo '  ]'
  echo '}'
} > "$REG"
echo "    wrote $REG"

echo "==> [6/6] recreate nodes with bridge tokens + populated registry"
# Each dept's bridge token is passed as DEPT_<X>_TOKEN (matching the ${DEPT_<X>_TOKEN:-} refs
# in docker-compose.mesh.example.yml's node-<x> commands) — built generically off DEPTS, not hardcoded.
env_assign=()
for i in $(seq 0 $((N-1))); do
  d="${DEPTS[$i]}"
  suffix="$(echo "${d#dept-}" | tr '[:lower:]' '[:upper:]')"   # dept-d -> D
  env_assign+=("DEPT_${suffix}_TOKEN=${TOK[$d]}")
done
env "${env_assign[@]}" $COMPOSE up -d --force-recreate "${SVCS[@]}"

echo
echo "mesh up. verify:"
for s in "${SVCS[@]}"; do
  ctr="${PROJECT}-${s}-1"
  echo "  $s: $(docker exec "$ctr" wget -qO- http://127.0.0.1:7070/health 2>/dev/null || echo unreachable)"
done
echo "next: ./mesh/demo.sh"
