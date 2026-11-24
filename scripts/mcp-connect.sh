#!/usr/bin/env bash
# mcp-connect.sh — mint a platform token and print ready-to-paste MCP client wiring.
#
# The RAG as a Service MCP server is mounted inside the platform at /mcp and gated by a
# platform JWT. This helper mints a dev token for a tenant and emits the exact
# `claude mcp add` command plus a .mcp.json snippet for Claude Code / Cursor.
#
# DEV token: the /v1/dev/token endpoint is enabled only when RAG_AUTH_MODE=dev_jwt and
# RAG_ENV != prod. In production, supply a real service-account API key (ek_…) instead of
# minting one here.
#
# Usage:
#   scripts/mcp-connect.sh                       # tenant=default, base=http://localhost:8000
#   scripts/mcp-connect.sh -t my-org -u https://rag.example.gov.in
#   RAG_BASE_URL=https://rag.example.gov.in scripts/mcp-connect.sh -t my-org -r reader,ingestor
set -euo pipefail

TENANT="default"
BASE_URL="${RAG_BASE_URL:-http://localhost:8000}"
ROLES="reader"
GROUPS=""
SUBJECT="mcp-client"

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while getopts ":t:u:r:g:s:h" opt; do
  case "$opt" in
    t) TENANT="$OPTARG" ;;
    u) BASE_URL="$OPTARG" ;;
    r) ROLES="$OPTARG" ;;
    g) GROUPS="$OPTARG" ;;
    s) SUBJECT="$OPTARG" ;;
    h) usage 0 ;;
    *) echo "unknown option: -$OPTARG" >&2; usage 1 ;;
  esac
done

BASE_URL="${BASE_URL%/}"
MCP_URL="${BASE_URL}/mcp/"

# Build JSON arrays from comma-separated inputs without assuming jq is installed.
to_json_array() {
  local IFS=','; local out=""; local item
  for item in $1; do
    [ -z "$item" ] && continue
    out="${out:+$out,}\"${item}\""
  done
  printf '[%s]' "$out"
}
ROLES_JSON="$(to_json_array "$ROLES")"
GROUPS_JSON="$(to_json_array "$GROUPS")"

payload=$(printf '{"subject":"%s","tenant_id":"%s","roles":%s,"groups":%s,"clearance":"public"}' \
  "$SUBJECT" "$TENANT" "$ROLES_JSON" "$GROUPS_JSON")

echo "Minting dev token for tenant '${TENANT}' at ${BASE_URL} ..." >&2
resp="$(curl -fsS -m 10 -X POST "${BASE_URL}/v1/dev/token" \
  -H 'Content-Type: application/json' -d "$payload")" || {
    echo "ERROR: token request failed. Is the platform up and RAG_AUTH_MODE=dev_jwt?" >&2
    exit 1
  }

TOKEN="$(printf '%s' "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')"
[ -n "$TOKEN" ] || { echo "ERROR: no access_token in response: $resp" >&2; exit 1; }

cat <<EOF

# ── Claude Code (CLI) ────────────────────────────────────────────────────────
claude mcp add --transport http ragaas ${MCP_URL} \\
  --header "Authorization: Bearer ${TOKEN}"

# ── .mcp.json (Claude Code project scope / Cursor .cursor/mcp.json) ───────────
{
  "mcpServers": {
    "ragaas": {
      "type": "http",
      "url": "${MCP_URL}",
      "headers": { "Authorization": "Bearer ${TOKEN}" }
    }
  }
}

# Token is scoped to tenant '${TENANT}', roles [${ROLES}]. It expires per the platform's
# JWT TTL — re-run this script to refresh. Never commit the token to a repo.
EOF
