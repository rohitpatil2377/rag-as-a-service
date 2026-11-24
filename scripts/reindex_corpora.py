"""Re-index documents in the named KBs from their stored raw bytes (FR-C8).

The single re-index the remediation plan holds everything for: it carries the chunking fixes,
the section-attribution fixes and the content-derived chunk ids (P44) into the stores in ONE
pass. Strategy is passed EXPLICITLY (P50) — reindex used to default to a different chunker than
ingest, so a before/after measured across it would compare two different chunkers.

    uv run python scripts/reindex_corpora.py kb1 kb2 ...
    uv run python scripts/reindex_corpora.py acme__hr --dry-run
    uv run python scripts/reindex_corpora.py acme__hr --target v2 --docs corrigendum-v.pdf

``--dry-run`` writes NOTHING: it reports the running pipeline fingerprint, which of the KB's
documents that fingerprint no longer matches, and exactly what a real run would touch. Run it
first — a corpus re-index competes with live questions for the same model endpoints.

``--target`` (RAG v2 Phase 6):
  * ``asis`` (default) — today's behaviour: re-chunk with ``--strategy``.
  * ``v2`` — send NO strategy so a document is chunked from its structure tree, then VERIFY each
    document came back recorded as ``layout_aware``. The flag that enables the v2 path
    (``RAG_V2_INGEST``) is server-side and this script cannot see it, so the check is on the
    OUTCOME: a document that silently came back v1 is reported as a failure, not as success.
    Note there is no separate ``_v2`` index to target — v2 chunks live in the KB's own namespace
    and are distinguished by ``level`` (ADR-0024), so "target" means the chunking generation.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.parse

import httpx

BASE = "http://localhost:8000"
_OK = 200
# Stamped on a document by the indexing service; `layout_aware` is what the v2 chunking path
# records. Kept as a literal rather than imported so the script stays a thin HTTP client.
_STRATEGY_KEY = "index_chunk_strategy"
_V2_STRATEGY = "layout_aware"


def _token(kb: str) -> str:
    r = httpx.post(
        f"{BASE}/v1/dev/token",
        timeout=30,
        json={"subject": "reindex", "org_id": "acme", "tenant_id": kb, "roles": ["reader", "admin", "ingestor"]},
    )
    r.raise_for_status()
    return str(r.json()["access_token"])


def _documents(kb: str, headers: dict[str, str]) -> list[dict[str, object]]:
    docs = httpx.get(f"{BASE}/v1/kb/{kb}/documents", headers=headers, timeout=120).json()
    listed = docs if isinstance(docs, list) else docs.get("documents", [])
    return [d for d in listed if isinstance(d, dict)]


def _selected(docs: list[dict[str, object]], wanted: str) -> list[dict[str, object]]:
    """``--docs`` applied. ``all`` (the default) keeps every document; otherwise the ids are taken
    literally, and one that the KB does not hold is reported rather than silently dropped — a typo
    in a doc id would otherwise look like a successful pass over nothing."""
    if wanted == "all":
        return docs
    ids = [d.strip() for d in wanted.split(",") if d.strip()]
    by_id = {str(d.get("doc_id")): d for d in docs}
    missing = [i for i in ids if i not in by_id]
    for i in missing:
        print(f"  !!   not in this KB: {i}")
    return [by_id[i] for i in ids if i in by_id]


def _dry_run(kb: str, headers: dict[str, str], docs: list[dict[str, object]], target: str) -> int:
    """Report what a real run would do. No writes, and no reindex call is made at all."""
    status = httpx.get(f"{BASE}/v1/kb/{kb}/index-status", headers=headers, timeout=120).json()
    stale = set(status.get("stale_doc_ids", []))
    print(f"  running pipeline: {status.get('current_version')}")
    print(f"  {status.get('total')} documents, {status.get('stale')} not built by it")
    for d in docs:
        doc_id = str(d.get("doc_id"))
        metadata = d.get("metadata") if isinstance(d.get("metadata"), dict) else {}
        recorded = str(metadata.get(_STRATEGY_KEY, "-")) if isinstance(metadata, dict) else "-"
        marks = "stale" if doc_id in stale else "current"
        becomes = _V2_STRATEGY if target == "v2" else "(as asked)"
        print(f"  plan {doc_id[:56]:58s} {marks:8s} {recorded:14s} -> {becomes}")
    print(f"  would re-index {len(docs)} documents; nothing was written")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kbs", nargs="+")
    ap.add_argument("--strategy", default="clause", help="chunker to re-chunk with when --target is asis")
    ap.add_argument("--docs", default="all", help="comma-separated doc ids, or 'all' (default)")
    ap.add_argument("--target", default="asis", choices=("asis", "v2"), help="chunking generation to index into")
    ap.add_argument("--dry-run", action="store_true", help="report what would happen; write nothing")
    args = ap.parse_args()

    total_docs = total_chunks = failures = 0
    for kb in args.kbs:
        headers = {"Authorization": f"Bearer {_token(kb)}"}
        docs = _selected(_documents(kb, headers), args.docs)
        print(f"\n=== {kb} — {len(docs)} documents ===")
        if args.dry_run:
            _dry_run(kb, headers, docs, args.target)
            continue
        # `--target v2` sends NO strategy: the server's layout path overrides whatever a document
        # was chunked as before, which is exactly how an existing corpus adopts v2.
        body: dict[str, str] = {} if args.target == "v2" else {"strategy": args.strategy}
        for d in docs:
            doc_id = str(d.get("doc_id"))
            started = time.time()
            r = httpx.post(
                f"{BASE}/v1/kb/{kb}/documents/{urllib.parse.quote(doc_id, safe='')}/reindex",
                headers=headers,
                json=body,
                timeout=1800,
            )
            if r.status_code != _OK:
                failures += 1
                print(f"  FAIL {doc_id[:60]:62s} {r.status_code} {r.text[:120]}")
                continue
            n = r.json().get("chunk_count", 0)
            total_docs += 1
            total_chunks += n
            print(f"  ok   {doc_id[:60]:62s} {n:5d} chunks  {time.time() - started:6.1f}s")

        if args.target == "v2" and docs:
            # VERIFY, do not assume. `RAG_V2_INGEST` is server-side and invisible from here,
            # and with it off a reindex succeeds and quietly produces v1 chunks — a pass that
            # reported success while changing nothing is the outcome worth failing on. One read
            # per KB, after the pass, rather than one per document.
            recorded = {
                str(d.get("doc_id")): str((d.get("metadata") or {}).get(_STRATEGY_KEY, "-"))  # type: ignore[union-attr]
                for d in _documents(kb, headers)
            }
            not_v2 = [str(d.get("doc_id")) for d in docs if recorded.get(str(d.get("doc_id"))) != _V2_STRATEGY]
            if not_v2:
                failures += len(not_v2)
                print(
                    f"  !!   {len(not_v2)} documents are NOT recorded as {_V2_STRATEGY} after this pass "
                    "— RAG_V2_INGEST is off on the server, or these documents produced no markup"
                )
                for doc_id in not_v2[:10]:
                    print(f"       {doc_id[:60]:62s} {recorded.get(doc_id, '-')}")

    if args.dry_run:
        return 0
    print(f"\nre-indexed {total_docs} documents, {total_chunks} chunks, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
