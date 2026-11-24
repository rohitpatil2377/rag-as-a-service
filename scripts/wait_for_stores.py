"""Block until the dev stack's stores are reachable (used by `make up`).

Runs INSIDE the compose network (``docker compose exec api``), not on the host: the stores
publish no host ports, so from the host the service names do not resolve and every check
failed against a perfectly healthy stack — making `make up` exit non-zero every time.

Polls each store's health endpoint with a bounded timeout so `make demo` doesn't race
a not-yet-ready dependency. Sovereign: only local/compose hosts are contacted.
"""

from __future__ import annotations

import socket
import sys
import time
from collections.abc import Callable
from urllib.parse import urlparse

import httpx

from rag_common import get_settings

_TIMEOUT_S = 120
_INTERVAL_S = 2
_HTTP_SERVER_ERROR = 500


def _http_ok(url: str) -> bool:
    try:
        return httpx.get(url, timeout=3).status_code < _HTTP_SERVER_ERROR
    except Exception:
        return False


def _tcp_ok(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


def main() -> int:
    s = get_settings()
    pg = urlparse(s.postgres_dsn.replace("+psycopg", ""))
    checks: dict[str, Callable[[], bool]] = {
        "qdrant": lambda: _http_ok(f"{s.qdrant_url}/healthz"),
        "opensearch": lambda: _http_ok(f"{s.opensearch_url}/_cluster/health"),
        "minio": lambda: _http_ok(f"http://{s.minio_endpoint}/minio/health/live"),
        "tei": lambda: _http_ok(f"{s.tei_url}/health"),
        "postgres": lambda: _tcp_ok(pg.hostname or "localhost", pg.port or 5432),
    }
    deadline = time.monotonic() + _TIMEOUT_S
    pending = set(checks)
    while pending and time.monotonic() < deadline:
        for name in sorted(pending):
            if checks[name]():
                print(f"  ✓ {name} ready")
                pending.discard(name)
        if pending:
            time.sleep(_INTERVAL_S)
    if pending:
        print(f"  ✗ not ready: {', '.join(sorted(pending))}", file=sys.stderr)
        return 1
    print("all stores ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
