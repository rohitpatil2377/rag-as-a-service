#!/usr/bin/env bash
# RAG as a Service — back up the stateful stores to a timestamped directory.
#
# Backs up (see docs/Backup_DR.md for the model):
#   AUTHORITATIVE (must have):
#     - PostgreSQL  → pg_dump (catalog, tamper-evident audit chain, every global registry)
#     - MinIO       → raw document bytes
#   DERIVED (rebuildable from the above via re-index — backed up for fast recovery):
#     - Qdrant      → vector collections
#     - OpenSearch  → BM25 indexes
#
# Usage:  scripts/backup.sh [output-dir]     (default: backups/<timestamp>)
# Env overrides: RAG_PG_CONTAINER, RAG_MINIO_CONTAINER, RAG_QDRANT_CONTAINER,
#                RAG_OPENSEARCH_CONTAINER, RAG_BACKUP_DERIVED=0 to skip Qdrant/OpenSearch.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PG=${RAG_PG_CONTAINER:-ragaas-dev-postgres-1}
MINIO=${RAG_MINIO_CONTAINER:-ragaas-dev-minio-1}
QDRANT=${RAG_QDRANT_CONTAINER:-ragaas-dev-qdrant-1}
OPENSEARCH=${RAG_OPENSEARCH_CONTAINER:-ragaas-dev-opensearch-1}
DERIVED=${RAG_BACKUP_DERIVED:-1}
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="${1:-$ROOT/backups/$STAMP}"
mkdir -p "$OUT"
echo "→ RAG as a Service backup → $OUT"

echo "  [1] PostgreSQL (authoritative: metadata + audit)…"
docker exec "$PG" pg_dump -U rag -d rag -Fc > "$OUT/postgres.dump"

echo "  [2] MinIO (authoritative: raw document bytes)…"
docker cp -q "$MINIO:/data" "$OUT/minio" 2>/dev/null || docker cp "$MINIO:/data" "$OUT/minio" >/dev/null

if [ "$DERIVED" = "1" ]; then
  echo "  [3] Qdrant (derived: vectors)…"
  docker cp "$QDRANT:/qdrant/storage" "$OUT/qdrant" >/dev/null 2>&1 || echo "    (skipped — rebuildable)"
  echo "  [4] OpenSearch (derived: BM25)…"
  docker cp "$OPENSEARCH:/usr/share/opensearch/data" "$OUT/opensearch" >/dev/null 2>&1 || echo "    (skipped — rebuildable)"
else
  echo "  [3/4] skipping derived stores (RAG_BACKUP_DERIVED=0) — rebuild via re-index on restore."
fi

cat > "$OUT/MANIFEST.txt" <<EOF
RAG as a Service backup
created:        $STAMP
authoritative:  postgres.dump (metadata + tamper-evident audit), minio/ (raw bytes)
derived:        $([ "$DERIVED" = 1 ] && echo "qdrant/, opensearch/ (rebuildable via re-index)" || echo "NOT INCLUDED — rebuild via re-index")
restore:        scripts/restore.sh "$OUT"
EOF
echo "→ done — $(du -sh "$OUT" | cut -f1)  ($OUT)"
