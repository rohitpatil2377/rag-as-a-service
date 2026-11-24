"""Web-crawl pure logic (FR-A9) — link extraction + scope rules, no I/O.

Extends the single-URL web source (FR-A3) to a bounded crawl: from a seed URL, follow
in-scope links up to depth/page limits. Every hop is still gated by the sovereign egress
allowlist and (optionally) robots.txt at the I/O layer; this module only decides *which*
links are in scope, so the traversal is deterministic and unit-testable.
"""

from __future__ import annotations

import re
from urllib.parse import urldefrag, urljoin, urlparse

_HREF_RE = re.compile(r"""<a\b[^>]*?\bhref\s*=\s*["']([^"'#][^"']*)["']""", re.IGNORECASE)
_SKIP_SCHEMES = ("mailto:", "javascript:", "tel:", "data:", "ftp:")
_NON_PAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".css", ".js", ".ico", ".zip", ".mp4", ".woff")


def canonical(url: str) -> str:
    """Drop the fragment and trailing slash so the same page isn't visited twice."""
    clean, _ = urldefrag(url.strip())
    if clean.endswith("/") and len(urlparse(clean).path) > 1:
        clean = clean.rstrip("/")
    return clean


def extract_links(html: str, base_url: str) -> list[str]:
    """Absolute, de-duplicated http(s) links from an HTML page (first-seen order)."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in _HREF_RE.findall(html):
        if raw.lower().startswith(_SKIP_SCHEMES):
            continue
        absu = canonical(urljoin(base_url, raw))
        scheme = urlparse(absu).scheme
        if scheme not in ("http", "https") or absu.lower().endswith(_NON_PAGE_SUFFIXES):
            continue
        if absu not in seen:
            seen.add(absu)
            out.append(absu)
    return out


def in_scope(url: str, seed: str, *, same_host_only: bool, path_prefix: str | None = None) -> bool:
    """Whether a link is within the crawl's scope: same host (if required) and, optionally,
    under the seed's path prefix (so a crawl of ``/docs`` doesn't wander the whole site)."""
    u, s = urlparse(url), urlparse(seed)
    if same_host_only and u.netloc != s.netloc:
        return False
    return not (path_prefix and not u.path.startswith(path_prefix))
