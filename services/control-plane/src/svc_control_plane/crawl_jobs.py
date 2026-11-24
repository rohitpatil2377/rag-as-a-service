"""Web crawler (FR-A9) — bounded BFS over a seed URL, sovereign per hop.

Extends the single-URL web source (FR-A3). From a seed, follow in-scope links up to depth /
page limits, respecting robots.txt, with EVERY hop gated by the egress allowlist (enforced
inside the web connector, so an off-allowlist host is refused before any bytes are read).
Each page is ingested through the normal pipeline (so A8 delta makes a re-crawl cheap, B7
auto-tags it, etc.). Link extraction reads the raw HTML the ingest already stored — one
network fetch per page.
"""

from __future__ import annotations

from datetime import UTC, datetime
from urllib import robotparser

import httpx
from pydantic import BaseModel, Field
from svc_ingestion.connectors import WEB_USER_AGENT

from rag_clients.factory import get_object_store
from rag_common import SovereigntyViolationError, assert_sovereign, get_logger, raw_object_key
from rag_core.crawl import canonical, extract_links, in_scope
from rag_schemas import DocumentEnvelope
from rag_schemas.common import SourceKind
from svc_control_plane import pipeline

_log = get_logger(__name__)


class CrawlResult(BaseModel):
    pages: int = 0
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    failed: int = 0
    skipped_offsite: int = 0
    skipped_robots: int = 0
    errors: list[str] = Field(default_factory=list, description="Sample of distinct failure reasons.")


_ROBOTS_TIMEOUT = 15.0


async def _load_robots(seed: str) -> robotparser.RobotFileParser | None:
    """Best-effort robots.txt for the seed's origin (sovereign-checked); None on any failure
    (permissive — a missing/blocked robots.txt never blocks the crawl)."""
    origin = seed.split("/", 3)[:3]
    robots_url = "/".join(origin) + "/robots.txt"
    try:
        assert_sovereign(robots_url)
        async with httpx.AsyncClient(
            timeout=_ROBOTS_TIMEOUT, follow_redirects=True, headers={"User-Agent": WEB_USER_AGENT}
        ) as client:
            resp = await client.get(robots_url)
        if resp.status_code >= 400:  # noqa: PLR2004
            return None
        rp = robotparser.RobotFileParser()
        rp.parse(resp.text.splitlines())
        return rp
    except Exception:
        return None


async def crawl_site(
    kb_id: str,
    seed: str,
    *,
    actor: str,
    max_depth: int = 1,
    max_pages: int = 20,
    same_host_only: bool = True,
    path_prefix: str | None = None,
    respect_robots: bool = True,
) -> CrawlResult:
    """Crawl from ``seed`` and ingest each in-scope page (bounded BFS)."""
    seed = canonical(seed)
    assert_sovereign(seed)  # fail fast if the seed host isn't allowlisted
    robots = await _load_robots(seed) if respect_robots else None
    obj = get_object_store()

    result = CrawlResult()
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(seed, 0)]

    while queue and result.pages < max_pages:
        url, depth = queue.pop(0)
        url = canonical(url)
        if url in visited:
            continue
        visited.add(url)
        if robots is not None and not robots.can_fetch("*", url):
            result.skipped_robots += 1
            continue

        doc = DocumentEnvelope(
            tenant_id=kb_id,
            doc_id=url,
            source_id="web-crawl",
            source_kind=SourceKind.WEB,
            uri=url,
            mime="text/html",
            ingested_at=datetime.now(UTC),
        )
        try:
            outcome = await pipeline.ingest_document(doc, actor=actor)  # platform default
        except SovereigntyViolationError:
            result.skipped_offsite += 1
            continue
        except Exception as exc:
            result.failed += 1
            reason = str(exc)[:200] or type(exc).__name__  # ConnectTimeout has no message string
            if reason not in result.errors and len(result.errors) < 3:  # noqa: PLR2004 — small sample
                result.errors.append(reason)
            _log.warning("crawl_page_failed", kb_id=kb_id, url=url, error=reason)
            continue
        result.pages += 1
        setattr(result, outcome.outcome, getattr(result, outcome.outcome) + 1)

        if depth >= max_depth or result.pages >= max_pages:
            continue
        try:
            raw = await obj.get(kb_id, raw_object_key(url))
        except KeyError:
            continue
        for link in extract_links(raw.decode("utf-8", "ignore"), url):
            cl = canonical(link)
            if cl not in visited and in_scope(cl, seed, same_host_only=same_host_only, path_prefix=path_prefix):
                queue.append((cl, depth + 1))

    _log.info("crawl_done", kb_id=kb_id, seed=seed, pages=result.pages, visited=len(visited))
    return result
