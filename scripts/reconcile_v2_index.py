"""Per-level index reconciliation — the RAG v2 Phase 6 acceptance check (read-only).

The plan's Phase 6 acceptance reads "counts per level reconcile across Qdrant, OpenSearch,
Postgres" into a parallel ``_v2`` namespace. This build has no such namespace: v2 chunks are
indexed into the tenant's OWN collection/index/schema, which is a recorded deviation with its
own decision record (ADR-0024). The criterion itself is unchanged in substance — it just has to
be read WITHIN the tenant namespace, grouped by ``level``, which is what this script answers.

Nothing is written. It asks each store to count its own units per level (one aggregation per
store — no vectors, documents or rows are transferred) and compares the three.

    docker exec -it <api-container> python scripts/reconcile_v2_index.py --kb acme__hr
    uv run python scripts/reconcile_v2_index.py --kb acme__hr --kb acme__tech --json

Run it INSIDE the api container, or from the host with the store URLs overridden — the defaults
name the in-compose hostnames:

    RAG_POSTGRES_DSN="postgresql+psycopg://rag:rag@localhost:5432/rag" \
    RAG_QDRANT_URL="http://localhost:6333" RAG_OPENSEARCH_URL="http://localhost:9200" \
    uv run python scripts/reconcile_v2_index.py --kb acme__hr

Exit code is 1 if any KB disagrees, so this is usable as a gate. KB ids must be given
explicitly: the platform deliberately offers no cross-tenant enumeration (see CLAUDE.md
"Tenancy"). To get the full list on the box, from the platform Postgres:

    SELECT nspname FROM pg_namespace WHERE nspname LIKE 't\\_%';   -- schema per tenant

Reading the output: PARENT context records (FR-C2) are excluded from the catalog side, because
they are never embedded or BM25-indexed by design. Everything else should agree exactly. A
disagreement is reported, not diagnosed — the usual causes are an interrupted ingest (one store
written, another not) and a reindex whose swap failed partway, both of which a reindex repairs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

os.environ.setdefault("RAG_OTEL_ENABLED", "false")

# How a level with no name is shown. The stores key it as None (a v1 chunk, which has no level);
# `(v1)` is clearer than `None` in a table an operator reads.
_V1 = "(v1)"


async def _counts(kb_id: str) -> dict[str, dict[str | None, int]]:
    from rag_clients.factory import get_catalog, get_sparse_index, get_vector_store

    return {
        "catalog": await get_catalog().count_by_level(kb_id),
        "vector": await get_vector_store().count_by_level(kb_id),
        "sparse": await get_sparse_index().count_by_level(kb_id),
    }


def reconcile(counts: dict[str, dict[str | None, int]]) -> dict[str, Any]:
    """Compare the three stores' per-level counts. Pure, so it is testable without any store."""
    stores = list(counts)
    levels = sorted({lvl for c in counts.values() for lvl in c}, key=lambda lvl: (lvl is None, lvl or ""))
    rows = [
        {
            "level": _V1 if lvl is None else lvl,
            **{store: counts[store].get(lvl, 0) for store in stores},
            "agrees": len({counts[store].get(lvl, 0) for store in stores}) == 1,
        }
        for lvl in levels
    ]
    totals = {store: sum(counts[store].values()) for store in stores}
    return {
        "levels": rows,
        "totals": totals,
        "reconciled": all(r["agrees"] for r in rows) and len(set(totals.values())) <= 1,
    }


def _print(kb_id: str, report: dict[str, Any]) -> None:
    stores = list(report["totals"])
    print(f"\n{kb_id}")
    print("  {:<12}{}".format("level", "".join(f"{s:>10}" for s in stores)))
    for row in report["levels"]:
        cells = "".join(f"{row[s]:>10}" for s in stores)
        print(f"  {row['level']:<12}{cells}   {'ok' if row['agrees'] else 'MISMATCH'}")
    cells = "".join(f"{report['totals'][s]:>10}" for s in stores)
    print(f"  {'TOTAL':<12}{cells}")
    print(f"  verdict: {'reconciled' if report['reconciled'] else 'DISAGREES'}")


async def _main(kb_ids: list[str], as_json: bool) -> int:
    reports = {kb: reconcile(await _counts(kb)) for kb in kb_ids}
    if as_json:
        print(json.dumps(reports, indent=2))
    else:
        for kb, report in reports.items():
            _print(kb, report)
    return 0 if all(r["reconciled"] for r in reports.values()) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kb", action="append", required=True, help="KB id (repeatable)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()
    return asyncio.run(_main(args.kb, args.json))


if __name__ == "__main__":
    raise SystemExit(main())
