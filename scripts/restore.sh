#!/usr/bin/env bash
# RAG as a Service — restore the stateful stores from a scripts/backup.sh directory (DR).
#
# Reverses backup.sh:
#   AUTHORITATIVE (must restore):
#     - PostgreSQL  ← pg_restore (catalog, tamper-evident audit chain, every global registry
#                     INCLUDING the federation grant table)
#     - MinIO       ← raw document bytes
#   DERIVED (restore for speed; otherwise rebuildable via re-index):
#     - Qdrant, OpenSearch
#
# After PostgreSQL is back it VERIFIES the federation grant table (grant-table DR): the table
# exists and its row count is reported, so a restore that silently lost grants is caught.
#
# Usage:  scripts/restore.sh <backup-dir> [--yes]
#   --yes         skip the destructive-restore confirmation
# Env overrides: same container names as backup.sh; RAG_RESTORE_DERIVED=0 to skip Qdrant/OpenSearch.
set -euo pipefail

PG=${RAG_PG_CONTAINER:-ragaas-dev-postgres-1}
MINIO=${RAG_MINIO_CONTAINER:-ragaas-dev-minio-1}
QDRANT=${RAG_QDRANT_CONTAINER:-ragaas-dev-qdrant-1}
OPENSEARCH=${RAG_OPENSEARCH_CONTAINER:-ragaas-dev-opensearch-1}
DERIVED=${RAG_RESTORE_DERIVED:-1}

SRC="${1:-}"
CONFIRM="${2:-}"
[ -n "$SRC" ] || { echo "usage: scripts/restore.sh <backup-dir> [--yes]" >&2; exit 2; }
[ -d "$SRC" ] || { echo "error: backup dir not found: $SRC" >&2; exit 2; }
[ -f "$SRC/postgres.dump" ] || { echo "error: $SRC/postgres.dump missing — not a valid backup" >&2; exit 2; }

if [ "$CONFIRM" != "--yes" ]; then
  echo "This OVERWRITES the live stores from $SRC (Postgres + MinIO$([ "$DERIVED" = 1 ] && echo ' + Qdrant + OpenSearch'))."
  printf "Type 'restore' to proceed: "
  read -r reply
  [ "$reply" = "restore" ] || { echo "aborted."; exit 1; }
fi

echo "→ RAG as a Service restore ← $SRC"

echo "  [1] PostgreSQL (authoritative)…"
# --clean --if-exists drops existing objects first so the restore is idempotent; grants,
# audit chain and catalog all come back with it.
docker exec -i "$PG" pg_restore -U rag -d rag --clean --if-exists --no-owner < "$SRC/postgres.dump"

echo "  [2] MinIO (authoritative: raw bytes)…"
if [ -d "$SRC/minio" ]; then
  docker cp "$SRC/minio/." "$MINIO:/data" >/dev/null
else
  echo "    (no minio/ in backup — skipped)"
fi

if [ "$DERIVED" = "1" ]; then
  echo "  [3] Qdrant (derived)…"
  if [ -d "$SRC/qdrant" ]; then
    docker cp "$SRC/qdrant/." "$QDRANT:/qdrant/storage" >/dev/null 2>&1 || echo "    (skipped)"
    docker restart "$QDRANT" >/dev/null 2>&1 || true
  else
    echo "    (not in backup — rebuild via re-index)"
  fi
  echo "  [4] OpenSearch (derived)…"
  if [ -d "$SRC/opensearch" ]; then
    docker cp "$SRC/opensearch/." "$OPENSEARCH:/usr/share/opensearch/data" >/dev/null 2>&1 || echo "    (skipped)"
    docker restart "$OPENSEARCH" >/dev/null 2>&1 || true
  else
    echo "    (not in backup — rebuild via re-index)"
  fi
else
  echo "  [3/4] skipping derived stores (RAG_RESTORE_DERIVED=0) — rebuild via re-index."
fi

echo "  [5] Grant-table DR verification…"
# The federation grant table lives in the rag_identity schema. Confirm it survived the restore
# and report the count of LIVE (non-revoked) grants — a silent loss here breaks federation.
grants=$(docker exec "$PG" psql -U rag -d rag -tAc \
  "SELECT count(*) FROM rag_identity.federation_grants WHERE revoked_at IS NULL" 2>/dev/null || echo "ERR")
if [ "$grants" = "ERR" ]; then
  echo "    ⚠ could not read rag_identity.federation_grants — verify the identity schema restored."
else
  echo "    ✓ federation grant table present — $grants live grant(s)."
fi

echo "DONE. Re-index derived stores if you skipped them (see docs/Backup_DR.md),"
echo "and confirm audit-chain integrity via GET /v1/kb/{id}/attestation."
